import argparse
import importlib.util
import re
import types
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from gpt2.checkpoint import load_checkpoint, save_best
from gpt2.model.config import GPTConfig
from gpt2.model.gpt import GPT
from gpt2.train.data import ShardLoader
from gpt2.train.loop import estimate_loss
from gpt2.train.shards import prepare, shard_paths
from run import settings_gpt2_small


TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
SRC = Path(__file__).parents[1]
PAGES = [f"Web page {i} says the Martians landed near town number {i * 7}." for i in range(150)]
SPAN = 2 * 16       # tokens in one tiny batch: batch_size x context_length


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SRC / "run" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_gpt2_small = load_script("train_gpt2_small")
compare_to_openai = load_script("compare_to_openai")


@pytest.fixture
def tiny(monkeypatch):
    # the real recipe and scripts, shrunk to a model the CPU trains in seconds: 2 x 2 x 16 = 64 tokens a step
    for name, value in dict(context_length=16, embed_dim=32, num_layers=1, num_heads=2, batch_size=2,
                            accumulation_steps=2, warmup_steps=3, eval_interval=5, checkpoint_interval=8,
                            print_interval=1, fast_attention=False, compile_model=False, max_new_tokens=3).items():
        monkeypatch.setattr(settings_gpt2_small, name, value)
    for script in (train_gpt2_small, compare_to_openai):
        monkeypatch.setattr(script, "pick_device", lambda: "cpu")
    # the tokenizer this file already loaded (loading it again asks Hugging Face online every time)
    monkeypatch.setattr(train_gpt2_small, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda name: TOKENIZER))


def prep(out):
    # a real prepare_data-style prep (whole pages, 1 in 10 to val), in small shards
    prepare(iter(PAGES), TOKENIZER, out, train_fraction=0.9, whole_documents=True, shard_size=1000,
            val_shard_size=10_000, source="pages")
    return out


def run(data, run_dir):
    train_gpt2_small.main(argparse.Namespace(data=str(data), run_dir=str(run_dir)))


def whole_batches(data, name):
    # how many tiny batches the shards hold: each one reads SPAN tokens plus the last target (2 bytes a token)
    return sum((len(path.read_bytes()) // 2 - 1) // SPAN for path in shard_paths(data, name))


def steps_in(output):
    return int(re.search(r"([\d,]+) steps of 64 tokens", output).group(1).replace(",", ""))


def test_trains_exactly_one_pass_over_the_train_shards(tmp_path, tiny, capsys):
    data = prep(tmp_path / "data")
    run(data, tmp_path / "run")
    output = capsys.readouterr().out
    num_steps = steps_in(output)

    assert num_steps == whole_batches(data, "train") // 2 > 10     # every whole batch, 2 per step
    last = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    assert last["step"] == num_steps - 1
    # one pass and no wrap-around: it stopped in the LAST train shard, not back at the first
    assert last["loader_state"]["file"] == shard_paths(data, "train")[-1].name
    assert (tmp_path / "run" / "best.pt").exists() and (tmp_path / "run" / "loss.png").exists()
    assert "--- sample after training ---" in output

    # the validation checks scored the VAL shards: the final model, scored there again, gets the last val loss
    model = load_checkpoint(tmp_path / "run" / "last.pt")
    val = ShardLoader(shard_paths(data, "val"), batch_size=2, context_length=16)
    assert estimate_loss(model, val, "cpu") == pytest.approx(last["history"]["val_loss"][-1], abs=1e-6)


def test_running_it_again_resumes_and_ends_exactly_as_a_straight_run(tmp_path, tiny, capsys, monkeypatch):
    data = prep(tmp_path / "data")
    run(data, tmp_path / "straight")

    # a disconnect partway: the 23rd batch never comes (step 11; the last save was after step 7)
    real_train = train_gpt2_small.train
    def disconnected(model, optimizer, train_batches, **options):
        real_next, calls = train_batches.next_batch, [0]
        def next_batch():
            calls[0] += 1
            if calls[0] == 23:
                raise KeyboardInterrupt("disconnected")
            return real_next()
        train_batches.next_batch = next_batch
        return real_train(model, optimizer, train_batches, **options)
    monkeypatch.setattr(train_gpt2_small, "train", disconnected)
    with pytest.raises(KeyboardInterrupt):
        run(data, tmp_path / "resumed")
    monkeypatch.setattr(train_gpt2_small, "train", real_train)
    capsys.readouterr()

    run(data, tmp_path / "resumed")                       # the SAME command again
    assert "from step 8" in capsys.readouterr().out
    straight = torch.load(tmp_path / "straight" / "last.pt", weights_only=False)
    resumed = torch.load(tmp_path / "resumed" / "last.pt", weights_only=False)
    assert all(torch.equal(straight["model_state"][k], resumed["model_state"][k]) for k in straight["model_state"])
    assert straight["history"] == resumed["history"]

    run(data, tmp_path / "resumed")                       # and once more: it's finished
    assert "already finished" in capsys.readouterr().out


@pytest.mark.parametrize("damage", ["missing", "short"])
def test_refuses_shards_that_arent_all_there(tmp_path, tiny, damage):
    data = prep(tmp_path / "data")
    shard = shard_paths(data, "train")[1]
    if damage == "missing":
        shard.unlink()
    else:
        shard.write_bytes(shard.read_bytes()[:-2])        # the copy off Google Drive stopped one token short
    with pytest.raises(SystemExit, match="finished prep"):
        run(data, tmp_path / "run")
    assert not (tmp_path / "run").exists()                # refused before anything was made


def test_refuses_a_prep_too_small_for_one_step(tmp_path, tiny, monkeypatch):
    monkeypatch.setattr(settings_gpt2_small, "accumulation_steps", 10_000)
    with pytest.raises(SystemExit, match="too few train tokens"):
        run(prep(tmp_path / "data"), tmp_path / "run")


def compare(data, run_dir):
    compare_to_openai.main(argparse.Namespace(data=str(data), run_dir=str(run_dir)))


def tiny_model(seed):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=50257, context_length=16, embed_dim=32, num_layers=1, num_heads=2)).eval()


def saved_model(run_dir):
    # a run folder with a best.pt, as a training run leaves it -- here an untrained tiny model is enough
    save_best(run_dir, tiny_model(seed=0), step=0, val_loss=10.0)
    return run_dir


def scores(output):
    return (float(re.search(r"your model +([\d.]+)", output).group(1)),
            float(re.search(r"OpenAI's GPT-2 small +([\d.]+)", output).group(1)))


def test_compare_scores_both_models_on_the_same_windows(tmp_path, tiny, capsys, monkeypatch):
    data, run_dir = prep(tmp_path / "data"), saved_model(tmp_path / "run")
    # "OpenAI's" model is the same model loaded again -- so the same windows must give the same score
    monkeypatch.setattr(compare_to_openai, "load_gpt2", lambda name: load_checkpoint(run_dir))
    capsys.readouterr()
    compare(data, run_dir)
    output = capsys.readouterr().out
    yours, openai = scores(output)
    assert yours == openai and 0 < yours < 11


def test_compare_prints_each_models_own_score(tmp_path, tiny, capsys, monkeypatch):
    data, run_dir = prep(tmp_path / "data"), saved_model(tmp_path / "run")
    other = tiny_model(seed=1)                             # a different model in "OpenAI's" place
    monkeypatch.setattr(compare_to_openai, "load_gpt2", lambda name: other)
    capsys.readouterr()
    compare(data, run_dir)
    yours, openai = scores(capsys.readouterr().out)
    val = ShardLoader(shard_paths(data, "val"), batch_size=2, context_length=16)
    every = whole_batches(data, "val")
    assert yours == pytest.approx(estimate_loss(load_checkpoint(run_dir), val, "cpu", num_batches=every), abs=5e-5)
    assert openai == pytest.approx(estimate_loss(other, val, "cpu", num_batches=every), abs=5e-5)
    assert yours != openai


def test_compare_scores_all_of_val_once(tmp_path, tiny, capsys, monkeypatch):
    data, run_dir = prep(tmp_path / "data"), saved_model(tmp_path / "run")
    monkeypatch.setattr(compare_to_openai, "load_gpt2", lambda name: tiny_model(seed=1))
    scored = []
    real_estimate = compare_to_openai.estimate_loss
    def recording(model, batches, device, num_batches, use_bf16):
        scored.append(num_batches)
        return real_estimate(model, batches, device, num_batches=num_batches, use_bf16=use_bf16)
    monkeypatch.setattr(compare_to_openai, "estimate_loss", recording)
    capsys.readouterr()
    compare(data, run_dir)
    every = whole_batches(data, "val")
    assert scored == [every, every]                        # both models: every val batch, none twice
    assert f"the same {every * SPAN:,} validation tokens" in capsys.readouterr().out


def test_compare_refuses_a_folder_that_isnt_a_finished_prep(tmp_path, tiny, monkeypatch):
    monkeypatch.setattr(compare_to_openai, "load_gpt2", lambda name: tiny_model(seed=1))
    with pytest.raises(SystemExit, match="finished prep"):
        compare(tmp_path / "nothing here", saved_model(tmp_path / "run"))
