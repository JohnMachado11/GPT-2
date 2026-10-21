import re

import pytest
import torch
from gpt2.model.config import GPTConfig
from gpt2.model.gpt import GPT
from gpt2.train.optimizer import AdamW
import gpt2.train.loop as loop_module
from gpt2.train.loop import duration, estimate_loss, train
from gpt2.train.schedule import learning_rate_at
from gpt2.train.data import RandomWindows, ShardLoader
from gpt2.train.shards import write_shard


def make_model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=50, context_length=16, embed_dim=32, num_layers=2, num_heads=2))


def test_bf16_forward_is_close_to_32bit():
    # bf16 keeps fewer digits, so the loss shifts a little -- but only a little
    model = make_model()
    token_ids = torch.randint(0, 50, (4, 16))
    targets = torch.randint(0, 50, (4, 16))

    _, loss_32bit = model(token_ids, targets)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _, loss_bf16 = model(token_ids, targets)

    assert abs(loss_32bit.item() - loss_bf16.item()) < 0.05


def test_train_with_bf16_learns_and_keeps_weights_32bit():
    # a repeating pattern is easy to learn, so the loss must clearly fall --
    # and autocast only changes the math, never the stored weights
    model = make_model()
    data = torch.arange(20).repeat(100)
    history = train(
        model, AdamW(model.parameters()), RandomWindows(data, batch_size=8, context_length=16),
        num_steps=60, max_learning_rate=1e-2, warmup_steps=5, min_learning_rate=1e-3,
        max_gradient_norm=1.0, device="cpu", use_bf16=True,
    )

    assert history["train_loss"][-1] < history["train_loss"][0] - 1.0
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_train_learns_from_shards_on_disk(tmp_path):
    # the same easy pattern, but read off shard files by ShardLoader instead of from memory
    write_shard(tmp_path / "train_000000.bin", list(range(20)) * 60)
    write_shard(tmp_path / "train_000001.bin", list(range(20)) * 60)
    write_shard(tmp_path / "val_000000.bin", list(range(20)) * 10)
    model = make_model()
    history = train(
        model, AdamW(model.parameters()),
        ShardLoader(sorted(tmp_path.glob("train_*.bin")), batch_size=8, context_length=16),
        num_steps=60, max_learning_rate=1e-2, warmup_steps=5, min_learning_rate=1e-3,
        max_gradient_norm=1.0, val_batches=ShardLoader([tmp_path / "val_000000.bin"], batch_size=8, context_length=16),
        eval_interval=20, device="cpu",
    )
    assert history["train_loss"][-1] < history["train_loss"][0] - 1.0
    assert history["val_loss"][-1] < history["val_loss"][0] - 1.0


def test_val_loss_from_shards_is_repeatable(tmp_path):
    # estimate_loss resets the loader first, so the same model gets the same number every time
    write_shard(tmp_path / "val_000000.bin", torch.randint(0, 50, (2000,)).tolist())
    val = ShardLoader([tmp_path / "val_000000.bin"], batch_size=4, context_length=16)
    model = make_model()
    first = estimate_loss(model, val, device="cpu", num_batches=5)
    second = estimate_loss(model, val, device="cpu", num_batches=5)
    assert first == second


class FixedBatches:
    # hands out exactly these batches, in order -- so two runs see identical data
    def __init__(self, batches):
        self.batches = batches
        self.index = 0
        self.batch_size, self.context_length = batches[0][0].shape     # like the real batch sources

    def next_batch(self):
        batch = self.batches[self.index]
        self.index += 1
        return batch

    def reset(self):
        self.index = 0


def run_one_step(batches, accumulation_steps):
    model = make_model()                      # same seed -> identical starting weights every call
    history = train(
        model, AdamW(model.parameters()), FixedBatches(batches),
        num_steps=1, max_learning_rate=1e-2, warmup_steps=1, min_learning_rate=1e-3,
        max_gradient_norm=1e9,                # clipping off, so it can't hide a gradient that's too big
        device="cpu", accumulation_steps=accumulation_steps,
    )
    return model, history


def test_accumulation_matches_one_big_batch():
    # 4 micro-batches of 8 must give the SAME gradients as 1 batch of 32 holding the same rows.
    # (Compared as gradients, not weights: AdamW's first step barely depends on gradient size,
    # so a missing "/ accumulation_steps" -- gradients 4x too big -- could hide in the weights.)
    torch.manual_seed(1)
    token_ids = torch.randint(0, 50, (32, 16))
    targets = torch.randint(0, 50, (32, 16))
    micro = [(token_ids[i : i + 8], targets[i : i + 8]) for i in range(0, 32, 8)]

    accumulated, accumulated_history = run_one_step(micro, accumulation_steps=4)
    big, big_history = run_one_step([(token_ids, targets)], accumulation_steps=1)

    for a, b in zip(accumulated.parameters(), big.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-6)
    # the recorded loss is the step's true average, not just the last micro-batch's
    assert abs(accumulated_history["train_loss"][0] - big_history["train_loss"][0]) < 1e-5


def test_train_rejects_zero_accumulation_steps():
    model = make_model()
    with pytest.raises(ValueError):
        train(model, AdamW(model.parameters()), RandomWindows(torch.arange(200) % 50, 2, 16),
              num_steps=1, max_learning_rate=1e-3, warmup_steps=1, min_learning_rate=1e-4,
              max_gradient_norm=1.0, accumulation_steps=0)


def test_train_stops_when_the_loss_diverges():
    # an absurd learning rate blows the weights up after the first step; training must stop
    # loudly instead of grinding on with a NaN loss
    model = make_model()
    with pytest.raises(RuntimeError, match="loss became"):     # the LOSS check, specifically
        train(model, AdamW(model.parameters()), RandomWindows(torch.arange(200) % 50, 2, 16),
              num_steps=5, max_learning_rate=1e30, warmup_steps=1, min_learning_rate=1e30,
              max_gradient_norm=1.0)


# ---------- the training step, piece by piece ----------

def short_run(model, batches, **overrides):
    arguments = dict(num_steps=1, max_learning_rate=1e-2, warmup_steps=1, min_learning_rate=1e-3,
                     max_gradient_norm=1.0, device="cpu")
    arguments.update(overrides)
    return train(model, AdamW(model.parameters()), batches, **arguments)


EASY = torch.arange(20).repeat(100)


def test_gradients_are_clipped_to_max_gradient_norm():
    # gradients survive after the last step, so their total size can be measured directly
    model = make_model()
    short_run(model, RandomWindows(EASY, 8, 16), max_gradient_norm=0.01)
    total = torch.norm(torch.stack([p.grad.norm() for p in model.parameters()]))
    assert total <= 0.01 * (1 + 1e-5)


def test_bf16_autocast_is_actually_used():
    # not just "training still works" -- the matrix math inside really runs in bf16
    model = make_model()
    dtypes = []
    model.blocks[0].attn.qkv.register_forward_hook(lambda module, inputs, output: dtypes.append(output.dtype))
    short_run(model, RandomWindows(EASY, 8, 16), use_bf16=True)
    assert dtypes == [torch.bfloat16]


def test_validation_runs_every_eval_interval_steps_and_at_the_end():
    model = make_model()
    history = short_run(model, RandomWindows(EASY, 8, 16), num_steps=12, warmup_steps=3,
                        val_batches=RandomWindows(EASY, 8, 16, seed=0), eval_interval=5)
    assert history["val_steps"] == [0, 5, 10, 11]


@pytest.mark.parametrize("bad", [{"eval_interval": 0}, {"checkpoint_interval": 0}, {"accumulation_steps": 0},
                                 {"print_interval": 0}])
def test_train_rejects_impossible_intervals_before_training(bad):
    model = make_model()
    before = [p.clone() for p in model.parameters()]
    with pytest.raises(ValueError, match="must be at least 1"):
        short_run(model, RandomWindows(EASY, 8, 16), **bad)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))     # refused up front


def test_compiling_on_a_mac_gpu_is_refused_before_training():
    # torch.compile gives wrong gradients on a Mac GPU (pytorch#201455): refused up front, on any machine
    model = make_model()
    before = [p.clone() for p in model.parameters()]
    with pytest.raises(ValueError, match="Mac GPU"):
        short_run(model, RandomWindows(EASY, 8, 16), device="mps", compile_model=True)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_a_warmup_longer_than_the_run_is_warned_about(capsys):
    short_run(make_model(), RandomWindows(EASY, 8, 16), num_steps=2, warmup_steps=50)
    assert "longer than the run" in capsys.readouterr().out


def test_estimate_loss_switches_dropout_off_and_hands_the_model_back_training(tmp_path):
    write_shard(tmp_path / "val_000000.bin", (torch.arange(400) % 50).tolist())
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=50, context_length=8, embed_dim=16, num_layers=2, num_heads=2, dropout=0.5))
    val = ShardLoader([tmp_path / "val_000000.bin"], 4, 8)
    first = estimate_loss(model, val, "cpu", num_batches=3)
    second = estimate_loss(model, val, "cpu", num_batches=3)
    assert first == second                  # with dropout 0.5 left ON, these would differ
    assert model.training                   # and the model is back in training mode afterwards


def test_seeded_validation_windows_give_the_same_number_every_check():
    model = make_model()
    val = RandomWindows(EASY, 8, 16, seed=0)
    assert estimate_loss(model, val, "cpu", num_batches=3) == estimate_loss(model, val, "cpu", num_batches=3)


def test_validation_checks_never_change_which_windows_training_draws():
    # seeded validation uses its own generator, so how OFTEN you validate can't change the training run
    def train_with(eval_interval):
        torch.manual_seed(0)
        model = GPT(GPTConfig(vocab_size=50, context_length=16, embed_dim=32, num_layers=2, num_heads=2))
        short_run(model, RandomWindows(EASY, 8, 16), num_steps=8, warmup_steps=2,
                  val_batches=RandomWindows(EASY, 8, 16, seed=0), eval_interval=eval_interval)
        return model

    assert all(torch.equal(a, b) for a, b in zip(train_with(1).parameters(), train_with(4).parameters()))


def test_train_hands_the_scheduled_learning_rate_to_the_optimizer_every_step():
    model = make_model()
    optimizer = AdamW(model.parameters())
    seen = []
    real_step = optimizer.step
    optimizer.step = lambda: (seen.append(optimizer.learning_rate), real_step())
    train(model, optimizer, RandomWindows(EASY, 8, 16), num_steps=12, max_learning_rate=1e-2, warmup_steps=3,
          min_learning_rate=1e-3, max_gradient_norm=1.0)
    assert seen == [learning_rate_at(step, 1e-2, 3, 12, 1e-3) for step in range(12)]


def test_an_infinite_gradient_also_stops_training_before_the_update():
    model = make_model()
    model.token_embedding.weight.register_hook(lambda gradient: gradient.clone().fill_(float("inf")))
    with pytest.raises(RuntimeError, match="gradients became"):
        short_run(model, RandomWindows(EASY, 8, 16))
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_a_warmup_exactly_as_long_as_the_run_is_not_warned_about(capsys):
    short_run(make_model(), RandomWindows(EASY, 8, 16), num_steps=3, warmup_steps=3)
    assert "longer than the run" not in capsys.readouterr().out


def test_unseeded_validation_windows_are_warned_about(capsys):
    short_run(make_model(), RandomWindows(EASY, 8, 16), val_batches=RandomWindows(EASY, 8, 16))
    assert "has no seed" in capsys.readouterr().out


def test_validation_read_from_shard_files_is_not_warned_about(tmp_path, capsys):
    # a ShardLoader reads the same windows in order every time -- it needs no seed
    write_shard(tmp_path / "val_000000.bin", list(range(20)) * 40)
    short_run(make_model(), RandomWindows(EASY, 8, 16), val_batches=ShardLoader([tmp_path / "val_000000.bin"], 8, 16))
    assert "has no seed" not in capsys.readouterr().out


def test_validation_uses_bf16_when_training_does():
    model = make_model()
    dtypes = []
    model.blocks[0].attn.qkv.register_forward_hook(lambda module, inputs, output: dtypes.append(output.dtype))
    short_run(model, RandomWindows(EASY, 8, 16), use_bf16=True, val_batches=RandomWindows(EASY, 8, 16, seed=0))
    assert dtypes and all(dtype == torch.bfloat16 for dtype in dtypes)     # 1 training forward + 20 validation


# ---------- the progress line ----------

def fake_clock(monkeypatch, model, seconds_per_forward, first_forward=None):
    # a clock that moves only while the model runs, so speeds and times come out exact. It starts at an
    # arbitrary time, as a real clock does; first_forward: the first forward's seconds (a slow warm-up)
    clock = {"now": 1000.0, "forwards": 0}
    def tick(*args):
        clock["forwards"] += 1
        clock["now"] += first_forward if clock["forwards"] == 1 and first_forward else seconds_per_forward
    monkeypatch.setattr(loop_module.time, "perf_counter", lambda: clock["now"])
    model.register_forward_hook(tick)


def progress_lines(capsys):
    # the progress lines, with the loss and learning rate (checked elsewhere) left out
    lines = [line for line in capsys.readouterr().out.splitlines() if " tokens/s " in line]
    return [re.sub(r" \| loss \S+ \| lr \S+", "", line) for line in lines]


def test_the_progress_line_shows_the_speed_and_the_time_done_and_left(monkeypatch, capsys):
    model = make_model()
    fake_clock(monkeypatch, model, seconds_per_forward=2.0)
    short_run(model, RandomWindows(EASY, 8, 16), num_steps=4)
    # 8 x 16 = 128 tokens a step, in 2 seconds: 64 tokens/s. After the first of 4 steps: 2s done, 3 x 2s left
    assert progress_lines(capsys) == [
        "step 0/4 | 64 tokens/s | 2s done, ~6s left",
        "step 1/4 | 64 tokens/s | 4s done, ~4s left",
        "step 2/4 | 64 tokens/s | 6s done, ~2s left",
        "step 3/4 | 64 tokens/s | 8s done, ~0s left",
    ]


def test_the_speed_counts_every_micro_batch_and_leaves_out_validation(monkeypatch, capsys):
    # 2 micro-batches a step: 256 tokens in 2 x 2s -- still 64 tokens/s. Validation (20 forwards, every 2
    # steps) takes time too: it's left out of the speed, but counted in the time left
    model = make_model()
    fake_clock(monkeypatch, model, seconds_per_forward=2.0)
    short_run(model, RandomWindows(EASY, 8, 16), num_steps=4, accumulation_steps=2,
              val_batches=RandomWindows(EASY, 8, 16, seed=0), eval_interval=2)
    lines = progress_lines(capsys)
    assert all(" 64 tokens/s " in line for line in lines)
    assert lines[1] == "step 1/4 | 64 tokens/s | 48s done, ~48s left"    # 4s + 40s validation + 4s


def test_the_speed_is_for_the_steps_since_the_last_line_not_the_whole_run(monkeypatch, capsys):
    # the first step is slow (warming up), the next two fast, printed together: the second line must show
    # the speed of BOTH fast steps -- and not of the slow one
    model = make_model()
    fake_clock(monkeypatch, model, seconds_per_forward=2.0, first_forward=8.0)
    short_run(model, RandomWindows(EASY, 8, 16), num_steps=3, print_interval=2)
    assert progress_lines(capsys) == [
        "step 0/3 | 16 tokens/s | 8s done, ~16s left",       # 128 tokens in 8s; 8s a step so far, 2 to go
        "step 2/3 | 64 tokens/s | 12s done, ~0s left",       # 256 tokens in 2 x 2s
    ]


def test_print_interval_prints_every_nth_step_and_the_last(capsys):
    short_run(make_model(), RandomWindows(EASY, 8, 16), num_steps=25, print_interval=10)
    assert [line.split("/")[0] for line in progress_lines(capsys)] == ["step  0", "step 10", "step 20", "step 24"]


def test_the_time_left_after_a_resume_counts_from_the_restart(monkeypatch, capsys, tmp_path):
    class StopsAfter(RandomWindows):
        def __init__(self, *args, batches):
            super().__init__(*args)
            self.batches = batches
        def next_batch(self):
            self.batches -= 1
            if self.batches < 0:
                raise KeyboardInterrupt("disconnected")
            return super().next_batch()
    with pytest.raises(KeyboardInterrupt):
        short_run(make_model(), StopsAfter(EASY, 8, 16, batches=10), num_steps=20, run_dir=tmp_path, checkpoint_interval=5)
    capsys.readouterr()
    model = make_model()
    fake_clock(monkeypatch, model, seconds_per_forward=2.0)
    short_run(model, RandomWindows(EASY, 8, 16), num_steps=20, run_dir=tmp_path, checkpoint_interval=5)
    # resumed at step 10: one step done in this session, 9 to go
    assert progress_lines(capsys)[0] == "step 10/20 | 64 tokens/s | 2s done, ~18s left"


@pytest.mark.parametrize("seconds, text", [(42, "42s"), (720, "12m"), (9360, "2.6h"), (0, "0s")])
def test_duration_reads_as_seconds_minutes_or_hours(seconds, text):
    assert duration(seconds) == text
