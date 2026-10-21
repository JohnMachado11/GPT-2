import json
import os
import re

import numpy as np
import pytest
import torch

import gpt2.checkpoint as checkpoint_module
import gpt2.train.loop as loop_module
from gpt2.checkpoint import (
    best_val_loss_so_far, load_checkpoint, load_last, new_run_dir, run_name, save_atomic, save_best, save_last,
)
from gpt2.model.config import GPTConfig
from gpt2.model.gpt import GPT
from gpt2.train.data import RandomWindows, ShardLoader
from gpt2.train.loop import run_settings, train
from gpt2.train.optimizer import AdamW
from gpt2.train.shards import write_shard


DATA = torch.arange(400) % 50                               # an easy repeating pattern


def make_run(seed=0, dropout=0.1, weight_decay=0.1):
    torch.manual_seed(seed)
    config = GPTConfig(vocab_size=50, context_length=8, embed_dim=16, num_layers=2, num_heads=2, dropout=dropout)
    model = GPT(config)
    return model, AdamW(model.parameters(), weight_decay=weight_decay)


TRAIN_DEFAULTS = dict(max_learning_rate=1e-2, warmup_steps=3, min_learning_rate=1e-3, max_gradient_norm=1.0)


def run_train(model, optimizer, batches, run_dir, num_steps=20, **overrides):
    arguments = dict(TRAIN_DEFAULTS, num_steps=num_steps, val_batches=RandomWindows(DATA, 4, 8, seed=0),
                     eval_interval=5, run_dir=run_dir, checkpoint_interval=5)
    arguments.update(overrides)
    return train(model, optimizer, batches, **arguments)


def settings_for(batches, num_steps=20, accumulation_steps=1):
    # the settings run_train() trains with -- built by the same function train() uses
    return run_settings(batches, num_steps, TRAIN_DEFAULTS["max_learning_rate"], TRAIN_DEFAULTS["warmup_steps"],
                        TRAIN_DEFAULTS["min_learning_rate"], TRAIN_DEFAULTS["max_gradient_norm"], accumulation_steps)


EMPTY_HISTORY = {"steps": [], "train_loss": [], "val_steps": [], "val_loss": []}


def same_weights(model_a, model_b):
    return all(torch.equal(a, b) for a, b in zip(model_a.parameters(), model_b.parameters()))


class DisconnectAfter(RandomWindows):
    # a batch source that "loses the connection" partway through, like Colab dropping
    def __init__(self, *args, batches_before_disconnect, **kwargs):
        super().__init__(*args, **kwargs)
        self.remaining = batches_before_disconnect

    def next_batch(self):
        if self.remaining == 0:
            raise KeyboardInterrupt("simulated disconnect")
        self.remaining -= 1
        return super().next_batch()


class ShardsDisconnectAfter(ShardLoader):
    # the same simulated disconnect, for a run reading shard files from disk
    def __init__(self, *args, batches_before_disconnect, **kwargs):
        super().__init__(*args, **kwargs)
        self.remaining = batches_before_disconnect

    def next_batch(self):
        if self.remaining == 0:
            raise KeyboardInterrupt("simulated disconnect")
        self.remaining -= 1
        return super().next_batch()


def poison_gradient_on_call(model, call_number):
    # make ONE element of one gradient NaN on the Nth backward pass -- the loss stays perfectly normal
    calls = {"count": 0}

    def hook(gradient):
        calls["count"] += 1
        if calls["count"] == call_number:
            gradient = gradient.clone()
            gradient.view(-1)[0] = float("nan")
        return gradient

    return model.token_embedding.weight.register_hook(hook)


# ---------- run folders, names, atomic saves ----------

def test_run_name_encodes_shape_size_and_a_sortable_timestamp():
    model, _ = make_run()
    assert re.fullmatch(r"gpt2_L2_D16_0\.\d+M_20\d{6}-\d{6}", run_name(model, model.config))


def test_new_run_dir_writes_config_and_never_reuses_a_folder(tmp_path, monkeypatch):
    model, _ = make_run()
    monkeypatch.setattr(checkpoint_module, "run_name", lambda model, config: "same_name")    # force a collision
    first = new_run_dir(model, model.config, tmp_path)
    second = new_run_dir(model, model.config, tmp_path)
    assert (first.name, second.name) == ("same_name", "same_name-2")
    assert json.loads((first / "config.json").read_text())["embed_dim"] == 16


def test_save_atomic_replaces_the_file_and_leaves_no_temp_file(tmp_path):
    save_atomic({"x": torch.tensor([1])}, tmp_path / "last.pt")
    save_atomic({"x": torch.tensor([2])}, tmp_path / "last.pt")
    assert torch.load(tmp_path / "last.pt")["x"].tolist() == [2]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["last.pt"]


def test_a_crash_in_the_middle_of_saving_leaves_the_previous_checkpoint_intact(tmp_path, monkeypatch):
    # the disconnect happens WHILE last.pt is being written: the old last.pt must survive untouched
    save_atomic({"step": 499}, tmp_path / "last.pt")

    def save_that_dies_halfway(state, file):
        file.write(b"half a file")                        # some bytes land on disk...
        raise KeyboardInterrupt("disconnected mid-save")  # ...then the connection drops

    monkeypatch.setattr(torch, "save", save_that_dies_halfway)
    with pytest.raises(KeyboardInterrupt):
        save_atomic({"step": 999}, tmp_path / "last.pt")
    monkeypatch.undo()
    assert torch.load(tmp_path / "last.pt")["step"] == 499
    assert sorted(p.name for p in tmp_path.iterdir()) == ["last.pt"]    # the half-written temp file is gone


def test_a_failed_cleanup_doesnt_hide_why_the_save_stopped(tmp_path, monkeypatch):
    def save_that_dies(state, file):
        raise KeyboardInterrupt("disconnected mid-save")
    def unlink_that_fails(self, missing_ok=False):
        raise OSError("the disk went away")
    monkeypatch.setattr(torch, "save", save_that_dies)
    monkeypatch.setattr(type(tmp_path), "unlink", unlink_that_fails)
    with pytest.raises(KeyboardInterrupt):                 # the real reason, not the cleanup's OSError
        save_atomic({"step": 1}, tmp_path / "last.pt")


# ---------- loading a trained model ----------

def test_load_checkpoint_round_trips_weights_and_config(tmp_path):
    model, _ = make_run()
    save_best(tmp_path, model, step=3, val_loss=2.5)
    reloaded = load_checkpoint(tmp_path)
    assert same_weights(model, reloaded)
    assert reloaded.config == model.config


def test_load_checkpoint_prefers_best_pt_over_last_pt(tmp_path):
    # last.pt is the END of the run, which may have overfit -- querying must get best.pt's weights
    model, optimizer = make_run()
    save_best(tmp_path, model, step=0, val_loss=1.0)
    best_weights = [p.clone() for p in model.parameters()]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)                                     # last.pt now holds DIFFERENT weights
    save_last(tmp_path, model, optimizer, 5, RandomWindows(DATA, 4, 8), EMPTY_HISTORY, {})
    reloaded = load_checkpoint(tmp_path)
    assert all(torch.equal(a, b) for a, b in zip(best_weights, reloaded.parameters()))


def test_load_checkpoint_falls_back_to_last_pt_when_no_validation_ran(tmp_path):
    model, optimizer = make_run()
    run_train(model, optimizer, RandomWindows(DATA, 4, 8), tmp_path / "run", num_steps=6, val_batches=None)
    assert not (tmp_path / "run" / "best.pt").exists()
    assert same_weights(model, load_checkpoint(tmp_path / "run"))


def test_loading_a_run_with_nothing_saved_yet_says_so(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(FileNotFoundError, match="nothing has been saved"):
        load_checkpoint(tmp_path / "run")


# ---------- never saving broken weights ----------

def test_saves_refuse_weights_with_nan_and_keep_the_previous_checkpoint(tmp_path):
    model, optimizer = make_run()
    batches = RandomWindows(DATA, 4, 8)
    save_last(tmp_path, model, optimizer, 4, batches, EMPTY_HISTORY, {})
    save_best(tmp_path, model, step=4, val_loss=2.0)
    with torch.no_grad():
        model.token_embedding.weight[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="refusing to save"):
        save_last(tmp_path, model, optimizer, 9, batches, EMPTY_HISTORY, {})
    with pytest.raises(RuntimeError, match="refusing to save"):
        save_best(tmp_path, model, step=9, val_loss=1.0)
    assert torch.load(tmp_path / "last.pt")["step"] == 4
    assert torch.load(tmp_path / "best.pt")["step"] == 4


# ---------- divergence: a bad gradient must never reach the weights or a checkpoint ----------

def test_a_nan_gradient_with_a_normal_loss_stops_training_before_the_update(tmp_path):
    # the dangerous case: the loss looks fine, but one gradient element is NaN. Without the gradient
    # check, the update below would turn every weight into NaN and the final save would write it out.
    model, optimizer = make_run(dropout=0.0)
    poison_gradient_on_call(model, call_number=6)           # the 6th backward = the final step of 6
    with pytest.raises(RuntimeError, match="gradients became nan"):
        run_train(model, optimizer, RandomWindows(DATA, 4, 8), tmp_path / "run", num_steps=6, checkpoint_interval=5)
    assert all(torch.isfinite(p).all() for p in model.parameters())          # the update never happened
    saved = torch.load(tmp_path / "run" / "last.pt")                         # the step-4 checkpoint...
    assert saved["step"] == 4
    assert all(torch.isfinite(v).all() for v in saved["model_state"].values())   # ...is intact and finite


def test_after_a_nan_gradient_the_run_resumes_from_the_last_good_checkpoint(tmp_path):
    model, optimizer = make_run(dropout=0.0)
    hook = poison_gradient_on_call(model, call_number=10)   # step 9, which is also a checkpoint step
    with pytest.raises(RuntimeError, match="gradients became nan"):
        run_train(model, optimizer, RandomWindows(DATA, 4, 8), tmp_path / "run", num_steps=15)
    hook.remove()
    assert torch.load(tmp_path / "run" / "last.pt")["step"] == 4            # step 9's save never happened

    model, optimizer = make_run(dropout=0.0, seed=7)
    history = run_train(model, optimizer, RandomWindows(DATA, 4, 8), tmp_path / "run", num_steps=15)
    assert history["steps"] == list(range(15))
    assert all(torch.isfinite(p).all() for p in model.parameters())


# ---------- resuming: exact continuation ----------

@pytest.mark.parametrize("accumulation_steps", [1, 2])
def test_interrupted_and_resumed_run_matches_an_uninterrupted_one(tmp_path, accumulation_steps):
    # 20 steps straight, vs: disconnect partway (last.pt is from step 9; with accumulation the disconnect
    # lands in the MIDDLE of a step's micro-batches) -> brand-new model, optimizer and batches, as after
    # a restart -> resume -> finish. Dropout is on, so this also proves the random state is restored.
    straight, straight_opt = make_run()
    straight_history = run_train(straight, straight_opt, RandomWindows(DATA, 4, 8), tmp_path / "straight",
                                 accumulation_steps=accumulation_steps)

    interrupted, interrupted_opt = make_run()
    batches_until_disconnect = 10 * accumulation_steps + 1
    with pytest.raises(KeyboardInterrupt):
        run_train(interrupted, interrupted_opt,
                  DisconnectAfter(DATA, 4, 8, batches_before_disconnect=batches_until_disconnect),
                  tmp_path / "resumed", accumulation_steps=accumulation_steps)
    assert torch.load(tmp_path / "resumed" / "last.pt")["step"] == 9

    resumed, resumed_opt = make_run(seed=123)               # different seed: everything must come from last.pt
    resumed_history = run_train(resumed, resumed_opt, RandomWindows(DATA, 4, 8), tmp_path / "resumed",
                                accumulation_steps=accumulation_steps)

    assert same_weights(straight, resumed)
    assert resumed_history == straight_history              # the loss curve covers the whole run, unchanged


def test_interrupted_run_reading_shards_resumes_at_the_right_token(tmp_path):
    write_shard(tmp_path / "train_000000.bin", (torch.arange(300) % 50).tolist())
    write_shard(tmp_path / "train_000001.bin", ((torch.arange(300) * 7) % 50).tolist())
    shards = sorted(tmp_path.glob("train_*.bin"))

    straight, straight_opt = make_run()
    run_train(straight, straight_opt, ShardLoader(shards, 4, 8), tmp_path / "straight")

    interrupted, interrupted_opt = make_run()
    with pytest.raises(KeyboardInterrupt):
        run_train(interrupted, interrupted_opt, ShardsDisconnectAfter(shards, 4, 8, batches_before_disconnect=12),
                  tmp_path / "resumed")

    resumed, resumed_opt = make_run(seed=123)
    run_train(resumed, resumed_opt, ShardLoader(shards, 4, 8), tmp_path / "resumed")
    assert same_weights(straight, resumed)


def test_last_pt_is_saved_on_schedule_and_at_the_end(tmp_path, monkeypatch):
    saved_steps = []
    real_save_last = loop_module.save_last
    monkeypatch.setattr(loop_module, "save_last", lambda run_dir, model, optimizer, step, *rest: (
        saved_steps.append(step), real_save_last(run_dir, model, optimizer, step, *rest)))
    model, optimizer = make_run()
    run_train(model, optimizer, RandomWindows(DATA, 4, 8), tmp_path / "run", num_steps=7, checkpoint_interval=3)
    assert saved_steps == [2, 5, 6]                         # every 3 completed steps, plus the end


def test_rerunning_a_finished_run_does_no_extra_training_and_says_so(tmp_path, capsys):
    model, optimizer = make_run()
    first = run_train(model, optimizer, RandomWindows(DATA, 4, 8), tmp_path / "run", num_steps=6)
    finished = [p.clone() for p in model.parameters()]
    capsys.readouterr()

    model, optimizer = make_run(seed=5)
    again = run_train(model, optimizer, RandomWindows(DATA, 4, 8), tmp_path / "run", num_steps=6)
    printed = capsys.readouterr().out
    assert again == first
    assert all(torch.equal(a, b) for a, b in zip(finished, model.parameters()))
    assert "already finished all 6 steps" in printed and "resuming" not in printed


def test_a_last_pt_saved_before_mps_state_existed_still_resumes(tmp_path):
    model, optimizer = make_run()
    batches = RandomWindows(DATA, 4, 8)
    save_last(tmp_path, model, optimizer, 0, batches, dict(EMPTY_HISTORY, steps=[0]), settings_for(batches))
    old_style = torch.load(tmp_path / "last.pt")
    del old_style["mps_rng_state"]
    torch.save(old_style, tmp_path / "last.pt")
    step, _ = load_last(tmp_path, model, optimizer, batches, settings_for(batches))
    assert step == 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a Mac GPU")
def test_mac_gpu_dropout_masks_are_restored(tmp_path):
    # on a Mac GPU, dropout's random masks come from the GPU's own generator -- a resume must restore it
    model, optimizer = make_run()
    model.to("mps")
    optimizer = AdamW(model.parameters())
    batches = RandomWindows(DATA, 4, 8, device="mps")
    save_last(tmp_path, model, optimizer, 0, batches, EMPTY_HISTORY, settings_for(batches))
    masks_after_save = torch.nn.functional.dropout(torch.ones(1000, device="mps"), p=0.5)

    torch.mps.manual_seed(999)                              # scramble the Mac GPU generator, as a restart would
    load_last(tmp_path, model, optimizer, batches, settings_for(batches))
    masks_after_load = torch.nn.functional.dropout(torch.ones(1000, device="mps"), p=0.5)
    assert torch.equal(masks_after_save.cpu(), masks_after_load.cpu())


# ---------- resuming: refusing anything that would quietly change the run ----------

def train_a_short_run(tmp_path, **make_options):
    model, optimizer = make_run(**make_options)
    run_train(model, optimizer, RandomWindows(DATA, 4, 8), tmp_path / "run", num_steps=10)
    return tmp_path / "run"


def test_resume_refuses_different_training_settings(tmp_path):
    run = train_a_short_run(tmp_path)
    model, optimizer = make_run()
    with pytest.raises(ValueError, match="can't resume"):
        run_train(model, optimizer, RandomWindows(DATA, 4, 8), run, num_steps=50)


def test_resume_refuses_a_different_model(tmp_path):
    run = train_a_short_run(tmp_path, dropout=0.1)
    model, optimizer = make_run(dropout=0.0)                # same shape, different dropout
    with pytest.raises(ValueError, match="can't resume"):
        run_train(model, optimizer, RandomWindows(DATA, 4, 8), run, num_steps=10)


def test_resume_refuses_different_optimizer_settings(tmp_path):
    run = train_a_short_run(tmp_path, weight_decay=0.1)
    model, optimizer = make_run(weight_decay=0.0)
    with pytest.raises(ValueError, match="quietly change how the weights move"):
        run_train(model, optimizer, RandomWindows(DATA, 4, 8), run, num_steps=10)


def test_resume_refuses_when_the_training_text_changed(tmp_path):
    run = train_a_short_run(tmp_path)
    model, optimizer = make_run()
    with pytest.raises(ValueError, match="training text changed"):
        run_train(model, optimizer, RandomWindows(DATA[:300], 4, 8), run, num_steps=10)


@pytest.mark.parametrize("what_changed", ["training settings", "optimizer settings", "training text"])
def test_a_refused_resume_changes_nothing(tmp_path, what_changed):
    # every check runs BEFORE anything is loaded, so the caller's model and optimizer are untouched --
    # whichever check is the one that refuses
    run = train_a_short_run(tmp_path)
    model, optimizer = make_run(seed=42, weight_decay=0.0 if what_changed == "optimizer settings" else 0.1)
    data = DATA[:300] if what_changed == "training text" else DATA
    num_steps = 50 if what_changed == "training settings" else 10
    before = [p.clone() for p in model.parameters()]
    with pytest.raises(ValueError):
        run_train(model, optimizer, RandomWindows(data, 4, 8), run, num_steps=num_steps)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert optimizer.step_count == 0


def test_shard_resume_finds_its_file_by_name_and_refuses_a_missing_one(tmp_path):
    for i in range(3):
        write_shard(tmp_path / f"train_00000{i}.bin", list(range(i * 100, i * 100 + 30)))
    loader = ShardLoader(sorted(tmp_path.glob("train_*.bin")), batch_size=2, context_length=3)
    for _ in range(5):
        loader.next_batch()                                 # now partway into train_000001
    saved = loader.state()
    expected = loader.next_batch()[0]

    # a shard ADDED in front shifts every list position -- the file name still finds the right place
    write_shard(tmp_path / "train_0000000.bin", list(range(900, 930)))
    shifted = ShardLoader(sorted(tmp_path.glob("train_*.bin")), batch_size=2, context_length=3)
    shifted.load_state(saved)
    assert torch.equal(shifted.next_batch()[0], expected)

    # the saved shard is MISSING: refuse instead of reading some other file
    (tmp_path / saved["file"]).unlink()
    missing = ShardLoader(sorted(tmp_path.glob("train_*.bin")), batch_size=2, context_length=3)
    with pytest.raises(ValueError, match="isn't among"):
        missing.load_state(saved)


def test_shard_resume_refuses_a_position_past_the_end_of_its_file(tmp_path):
    write_shard(tmp_path / "train_000000.bin", list(range(30)))
    loader = ShardLoader([tmp_path / "train_000000.bin"], batch_size=2, context_length=3)
    with pytest.raises(ValueError, match="outside"):
        loader.load_state({"file": "train_000000.bin", "position": 31})


# ---------- best.pt ----------

def test_best_pt_holds_the_lowest_validation_loss_and_its_step(tmp_path):
    model, optimizer = make_run(dropout=0.0)
    history = run_train(model, optimizer, RandomWindows(DATA, 4, 8), tmp_path / "run", num_steps=30,
                        max_learning_rate=3e-2, eval_interval=3)
    best = torch.load(tmp_path / "run" / "best.pt")
    lowest = min(range(len(history["val_loss"])), key=history["val_loss"].__getitem__)
    assert lowest > 0                                       # only meaningful if the best isn't the first check
    assert best["val_loss"] == history["val_loss"][lowest]
    assert best["step"] == history["val_steps"][lowest]


def test_the_final_step_is_always_validated(tmp_path):
    # 14 steps, checks every 5: steps 0, 5, 10 -- and 13, the end, so the finished model is scored
    model, optimizer = make_run()
    history = run_train(model, optimizer, RandomWindows(DATA, 4, 8), tmp_path / "run", num_steps=14)
    assert history["val_steps"] == [0, 5, 10, 13]


def test_resume_never_overwrites_a_better_best_pt(tmp_path):
    # best.pt can be NEWER than last.pt (a new best right before a disconnect). The resumed run must
    # trust best.pt's own record, not last.pt's older history, or it would overwrite it with a worse model.
    run = tmp_path / "run"
    model, optimizer = make_run()
    batches = RandomWindows(DATA, 4, 8)
    history = {"steps": [0], "train_loss": [4.0], "val_steps": [0], "val_loss": [9.0]}
    save_last(run, model, optimizer, 0, batches, history, settings_for(batches, num_steps=6))   # older: best 9.0
    save_best(run, model, step=3, val_loss=0.5)                                                  # newer: 0.5

    model, optimizer = make_run()
    run_train(model, optimizer, RandomWindows(DATA, 4, 8), run, num_steps=6, eval_interval=1)
    assert torch.load(run / "best.pt")["val_loss"] == 0.5


def test_a_fresh_start_never_overwrites_a_better_best_pt(tmp_path):
    # a disconnect BEFORE the first last.pt leaves a best.pt but no last.pt; starting again in that
    # folder must still respect best.pt's record
    run = tmp_path / "run"
    model, _ = make_run()
    save_best(run, model, step=3, val_loss=0.5)
    model, optimizer = make_run()
    run_train(model, optimizer, RandomWindows(DATA, 4, 8), run, num_steps=6, eval_interval=1)
    assert torch.load(run / "best.pt")["val_loss"] == 0.5


def test_a_best_pt_from_a_different_model_is_refused(tmp_path):
    run = tmp_path / "run"
    other, _ = make_run(dropout=0.0)
    save_best(run, other, step=3, val_loss=0.5)
    model, _ = make_run(dropout=0.1)
    with pytest.raises(ValueError, match="belongs to a different model"):
        best_val_loss_so_far(run, model)


def test_train_without_a_run_dir_saves_nothing(monkeypatch):
    def must_not_be_called(*args, **kwargs):
        raise AssertionError("saved a checkpoint with run_dir=None")
    monkeypatch.setattr(loop_module, "save_last", must_not_be_called)
    monkeypatch.setattr(loop_module, "save_best", must_not_be_called)
    model, optimizer = make_run()
    run_train(model, optimizer, RandomWindows(DATA, 4, 8), run_dir=None, num_steps=6)


# ---------- more ways a resume or a save could quietly go wrong ----------

def test_new_run_dir_keeps_counting_past_two_collisions(tmp_path, monkeypatch):
    model, _ = make_run()
    monkeypatch.setattr(checkpoint_module, "run_name", lambda model, config: "same_name")
    names = [new_run_dir(model, model.config, tmp_path).name for _ in range(3)]
    assert names == ["same_name", "same_name-2", "same_name-3"]


def test_every_save_is_flushed_to_disk_before_the_rename(tmp_path, monkeypatch):
    # the ORDER matters: the file's bytes must be forced onto the disk before it's renamed into place,
    # or a hard stop right after the rename could leave an empty last.pt
    events = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda descriptor: (events.append("flush"), real_fsync(descriptor)))
    monkeypatch.setattr(os, "replace", lambda source, target: (events.append("rename"), real_replace(source, target)))
    save_atomic({"x": torch.tensor([1])}, tmp_path / "last.pt")
    assert "rename" in events and "flush" in events[: events.index("rename")]


def test_load_checkpoint_hands_back_a_model_ready_to_query(tmp_path):
    model, _ = make_run()
    save_best(tmp_path, model, step=0, val_loss=1.0)
    assert not load_checkpoint(tmp_path).training           # dropout off


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_saves_refuse_a_bad_value_anywhere_in_the_weights(tmp_path, bad):
    # the LAST number of the LAST tensor -- a check that only looked at the first value would miss it
    model, optimizer = make_run()
    with torch.no_grad():
        list(model.parameters())[-1].view(-1)[-1] = bad
    with pytest.raises(RuntimeError, match="refusing to save"):
        save_best(tmp_path, model, step=0, val_loss=1.0)
    with pytest.raises(RuntimeError, match="refusing to save"):
        save_last(tmp_path, model, optimizer, 0, RandomWindows(DATA, 4, 8), EMPTY_HISTORY, {})


@pytest.mark.parametrize("average", ["momentum", "bumpiness"])
def test_save_last_refuses_a_broken_optimizer_state(tmp_path, average):
    model, optimizer = make_run()
    getattr(optimizer, average)[-1].view(-1)[-1] = float("inf")
    with pytest.raises(RuntimeError, match="optimizer's state"):
        save_last(tmp_path, model, optimizer, 0, RandomWindows(DATA, 4, 8), EMPTY_HISTORY, {})


@pytest.mark.parametrize("change", [
    dict(max_learning_rate=2e-2), dict(warmup_steps=4), dict(min_learning_rate=2e-3),
    dict(max_gradient_norm=0.5), dict(accumulation_steps=2), dict(batch_size=2), dict(context_length=4),
])
def test_resume_refuses_each_changed_setting(tmp_path, change):
    run = train_a_short_run(tmp_path)
    change = dict(change)
    batches = RandomWindows(DATA, change.pop("batch_size", 4), change.pop("context_length", 8))
    model, optimizer = make_run()
    with pytest.raises(ValueError, match="can't resume"):
        run_train(model, optimizer, batches, run, num_steps=10, **change)


def test_a_refused_resume_leaves_the_random_state_and_the_loader_alone(tmp_path):
    model, optimizer = make_run(weight_decay=0.1)
    batches = RandomWindows(DATA, 4, 8, seed=1)
    batches.next_batch()
    save_last(tmp_path, model, optimizer, 0, batches, EMPTY_HISTORY, settings_for(batches, num_steps=10))
    _, other_optimizer = make_run(weight_decay=0.0)          # refused: different optimizer settings
    fresh = RandomWindows(DATA, 4, 8, seed=1)
    torch.manual_seed(999)                                   # a restart's random state differs from the saved one
    loader_before, random_before = fresh.generator.get_state(), torch.get_rng_state()
    with pytest.raises(ValueError):
        load_last(tmp_path, model, other_optimizer, fresh, settings_for(fresh, num_steps=10))
    assert torch.equal(fresh.generator.get_state(), loader_before)
    assert torch.equal(torch.get_rng_state(), random_before)


def test_a_best_pt_from_a_different_model_stops_a_resume_before_anything_changes(tmp_path):
    run = train_a_short_run(tmp_path, dropout=0.1)
    other, _ = make_run(dropout=0.0)
    save_best(run, other, step=3, val_loss=0.01)             # a mixed-up folder
    model, optimizer = make_run(seed=42, dropout=0.1)
    before = [p.clone() for p in model.parameters()]
    with pytest.raises(ValueError, match="belongs to a different model"):
        run_train(model, optimizer, RandomWindows(DATA, 4, 8), run, num_steps=10)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_settings_given_as_numpy_numbers_still_resume(tmp_path):
    # numpy numbers save fine but torch.load's safe mode refuses them -- the settings are stored as
    # plain Python numbers so a run doesn't fail only at resume time, hours in
    model, optimizer = make_run()
    with pytest.raises(KeyboardInterrupt):
        run_train(model, optimizer, DisconnectAfter(DATA, 4, 8, batches_before_disconnect=7), tmp_path / "run",
                  num_steps=10, max_learning_rate=np.float64(1e-2), warmup_steps=np.int64(3))
    model, optimizer = make_run()
    history = run_train(model, optimizer, RandomWindows(DATA, 4, 8), tmp_path / "run",
                        num_steps=10, max_learning_rate=np.float64(1e-2), warmup_steps=np.int64(3))
    assert history["steps"] == list(range(10))


@pytest.mark.parametrize("random_state", ["rng_state", "cuda_rng_state", "mps_rng_state"])
@pytest.mark.parametrize("damage", ["not a tensor", torch.zeros(16)])   # the wrong kind, or the wrong number type
def test_a_damaged_random_state_is_refused_before_anything_changes(tmp_path, random_state, damage):
    run = train_a_short_run(tmp_path)
    state = torch.load(run / "last.pt")
    state[random_state] = damage
    torch.save(state, run / "last.pt")
    model, optimizer = make_run(seed=42)                # different weights from the saved ones
    before, random_before = [p.clone() for p in model.parameters()], torch.get_rng_state()
    with pytest.raises(ValueError, match="damaged"):
        run_train(model, optimizer, RandomWindows(DATA, 4, 8), run, num_steps=10)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert optimizer.step_count == 0
    assert torch.equal(torch.get_rng_state(), random_before)


def test_a_missing_cpu_random_state_is_refused(tmp_path):
    run = train_a_short_run(tmp_path)
    state = torch.load(run / "last.pt")
    del state["rng_state"]
    torch.save(state, run / "last.pt")
    model, optimizer = make_run(seed=42)
    with pytest.raises(ValueError, match="CPU random state is missing"):
        run_train(model, optimizer, RandomWindows(DATA, 4, 8), run, num_steps=10)
    assert optimizer.step_count == 0


def test_a_model_size_that_isnt_a_whole_number_is_refused():
    # 16.9 must not quietly become 16
    with pytest.raises(TypeError):
        GPTConfig(vocab_size=50, context_length=8, embed_dim=16.9, num_layers=2, num_heads=2)


def test_an_older_per_gpu_list_of_random_states_is_accepted(tmp_path):
    model, optimizer = make_run()
    batches = RandomWindows(DATA, 4, 8)
    save_last(tmp_path, model, optimizer, 0, batches, EMPTY_HISTORY, settings_for(batches))
    state = torch.load(tmp_path / "last.pt")
    state["cuda_rng_state"] = [torch.zeros(16, dtype=torch.uint8)]      # the old get_rng_state_all() format
    torch.save(state, tmp_path / "last.pt")
    step, _ = load_last(tmp_path, model, optimizer, batches, settings_for(batches))
    assert step == 0


def test_shard_and_in_memory_states_refuse_each_other(tmp_path):
    write_shard(tmp_path / "train_000000.bin", list(range(30)))
    shards = ShardLoader([tmp_path / "train_000000.bin"], batch_size=2, context_length=3)
    windows = RandomWindows(DATA, 2, 3)
    for shard_state in (shards.state(), {"shard": 0, "position": 6}):  # today's shard format, or an old one
        with pytest.raises(ValueError, match="reading shard files"):
            windows.check_state(shard_state)
    for not_a_shard_state in ({}, {"shard": 0, "position": 6}):       # in-memory, or an old shard format
        with pytest.raises(ValueError, match="wasn't saved reading shard files"):
            shards.check_state(not_a_shard_state)


def test_a_shard_regenerated_under_the_same_name_is_refused(tmp_path):
    write_shard(tmp_path / "train_000000.bin", list(range(30)))
    loader = ShardLoader([tmp_path / "train_000000.bin"], batch_size=2, context_length=3)
    loader.next_batch()
    saved = loader.state()
    write_shard(tmp_path / "train_000000.bin", list(range(100, 140)))    # same name, new contents
    fresh = ShardLoader([tmp_path / "train_000000.bin"], batch_size=2, context_length=3)
    with pytest.raises(ValueError, match="regenerated"):
        fresh.check_state(saved)


def test_a_shard_regenerated_at_the_same_length_is_refused_too(tmp_path):
    # re-running data prep with the same shard size gives every shard the same length -- only the
    # contents tell them apart
    write_shard(tmp_path / "train_000000.bin", list(range(30)))
    loader = ShardLoader([tmp_path / "train_000000.bin"], batch_size=2, context_length=3)
    loader.next_batch()
    saved = loader.state()
    write_shard(tmp_path / "train_000000.bin", list(range(100, 130)))    # same name, same length, new text
    fresh = ShardLoader([tmp_path / "train_000000.bin"], batch_size=2, context_length=3)
    with pytest.raises(ValueError, match="regenerated"):
        fresh.check_state(saved)


def test_a_shard_state_saved_before_fingerprints_still_resumes(tmp_path):
    write_shard(tmp_path / "train_000000.bin", list(range(30)))
    loader = ShardLoader([tmp_path / "train_000000.bin"], batch_size=2, context_length=3)
    loader.next_batch()
    saved = loader.state()
    del saved["file_fingerprint"]
    fresh = ShardLoader([tmp_path / "train_000000.bin"], batch_size=2, context_length=3)
    fresh.check_state(saved)
    fresh.load_state(saved)
    assert fresh.state()["position"] == saved["position"]


def test_a_shard_state_saved_before_fingerprints_still_catches_a_length_change(tmp_path):
    write_shard(tmp_path / "train_000000.bin", list(range(30)))
    loader = ShardLoader([tmp_path / "train_000000.bin"], batch_size=2, context_length=3)
    loader.next_batch()
    saved = loader.state()
    del saved["file_fingerprint"]                                          # the older format
    write_shard(tmp_path / "train_000000.bin", list(range(100, 140)))     # regenerated, longer
    fresh = ShardLoader([tmp_path / "train_000000.bin"], batch_size=2, context_length=3)
    with pytest.raises(ValueError, match="had 30 tokens when saved"):
        fresh.check_state(saved)


def test_a_fractional_warmup_is_stored_as_given():
    # "10% of the run" is often fractional -- rounding it down would quietly change the schedule
    assert run_settings(RandomWindows(DATA, 4, 8), 415, 1e-2, 41.5, 1e-3, 1.0, 1)["warmup_steps"] == 41.5


def test_a_model_and_optimizer_built_from_numpy_numbers_still_save_and_resume(tmp_path):
    # numpy numbers save fine but torch.load's safe mode refuses them: the model config and the
    # optimizer's settings are stored as plain Python numbers, so this can't fail hours into a run
    config = GPTConfig(vocab_size=np.int64(50), context_length=np.int64(8), embed_dim=np.int64(16),
                       num_layers=np.int64(2), num_heads=np.int64(2), dropout=np.float64(0.1))
    torch.manual_seed(0)
    model = GPT(config)
    optimizer = AdamW(model.parameters(), momentum_decay=np.float64(0.9), bumpiness_decay=np.float64(0.999),
                      epsilon=np.float64(1e-8), weight_decay=np.float64(0.1))
    batches = RandomWindows(DATA, 4, 8)
    run_dir = new_run_dir(model, config, tmp_path)
    save_last(run_dir, model, optimizer, 0, batches, EMPTY_HISTORY, settings_for(batches))
    step, _ = load_last(run_dir, model, optimizer, batches, settings_for(batches))
    assert step == 0
    assert json.loads((run_dir / "config.json").read_text())["embed_dim"] == 16


def test_shard_resume_checks_the_position_against_the_saved_file(tmp_path):
    write_shard(tmp_path / "train_000000.bin", list(range(100)))
    write_shard(tmp_path / "train_000001.bin", list(range(30)))
    loader = ShardLoader(sorted(tmp_path.glob("train_*.bin")), batch_size=2, context_length=3)
    for position in (-1, 31):
        with pytest.raises(ValueError, match="outside"):
            loader.check_state({"file": "train_000001.bin", "position": position})


def test_train_creates_the_run_folder_before_any_training(tmp_path):
    class FailsImmediately(RandomWindows):
        def next_batch(self):
            raise RuntimeError("stop before any training")
    model, optimizer = make_run()
    with pytest.raises(RuntimeError, match="stop before"):
        run_train(model, optimizer, FailsImmediately(DATA, 4, 8), tmp_path / "deep" / "run", num_steps=3)
    assert (tmp_path / "deep" / "run").is_dir()              # made up front -- a bad path fails in the first second


def test_a_compiled_run_trains_the_same_and_saves_checkpoints_a_plain_model_resumes(tmp_path, monkeypatch):
    # torch.compile's "eager" backend: the same wrapper, without minutes of compiling in a test
    real_compile, compiled_models = torch.compile, []
    class Counted(torch.nn.Module):             # counts the forward passes that go through the compiled copy
        def __init__(self, model):
            super().__init__()
            self.compiled, self.calls = real_compile(model, backend="eager"), 0
        def forward(self, *inputs):
            self.calls += 1
            return self.compiled(*inputs)
    def eager_compile(model):
        compiled_models.append(Counted(model))
        return compiled_models[-1]
    monkeypatch.setattr(torch, "compile", eager_compile)
    model, optimizer = make_run(dropout=0.0)
    plain = run_train(model, optimizer, RandomWindows(DATA, 4, 8), tmp_path / "plain", num_steps=10)
    model, optimizer = make_run(dropout=0.0)
    with pytest.raises(KeyboardInterrupt):
        run_train(model, optimizer, DisconnectAfter(DATA, 4, 8, batches_before_disconnect=7), tmp_path / "compiled",
                  num_steps=10, compile_model=True)
    # the compiled run really compiled it, and ran everything through it: 7 training steps, and validation
    # at steps 0 and 5 (20 batches each)
    assert [counted.compiled._orig_mod for counted in compiled_models] == [model]
    assert compiled_models[0].calls == 7 + 2 * 20
    saved = torch.load(tmp_path / "compiled" / "last.pt")["model_state"]
    assert not any(name.startswith("_orig_mod") for name in saved)        # the plain names, not the wrapper's
    model, optimizer = make_run(dropout=0.0)                               # a plain model picks it up
    compiled = run_train(model, optimizer, RandomWindows(DATA, 4, 8), tmp_path / "compiled", num_steps=10)
    assert compiled == plain
