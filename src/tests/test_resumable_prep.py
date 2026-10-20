import argparse
import errno
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from gpt2.train.resumable_prep import Checkpoints, remove_checkpoints, resume_point
from gpt2.train.shards import PrepRefused, ShardWriter, prepare, shard_paths, text_shards


TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
SRC = Path(__file__).parents[1]


def file_hashes(out):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())}


def data_files(out):
    # the shards and info.json -- what training reads -- leaving out the kept checkpoints
    return {name: digest for name, digest in file_hashes(out).items() if not name.startswith(("checkpoint", "buffer_"))}


# ---------- an interrupted prep resumes, and ends exactly as an uninterrupted one would ----------

PAGES = [f"Web page {i} says the Martians landed near town number {i * 7}." for i in range(3500)]   # 4 batches


class Stream:
    # a stand-in for FineWeb's stream: it can say where it is and restart from there
    def __init__(self, pages, disconnect_after=None):
        self.pages, self.position, self.disconnect_after = pages, 0, disconnect_after

    def __iter__(self):
        while self.position < len(self.pages):
            if self.position == self.disconnect_after:
                raise KeyboardInterrupt("disconnected")
            self.position += 1
            yield self.pages[self.position - 1]

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


def settings(max_tokens=None, train_fraction=0.99, shard_size=1500, val_shard_size=40):
    return {"source": "pages", "train_fraction": train_fraction, "whole_documents": True,
            "shard_size": shard_size, "val_shard_size": val_shard_size, "max_tokens": max_tokens}


def prep(out, stream, resume_from=None, **options):
    return prepare(iter(stream), TOKENIZER, out, **settings(**options),
                   checkpoints=Checkpoints(out, stream.state_dict, resume_from))


def resume(out, disconnect_after=None, **options):
    checkpoint = resume_point(out, settings(**options))
    stream = Stream(PAGES, disconnect_after)
    if checkpoint is not None:
        stream.load_state_dict(checkpoint["stream"])
    prep(out, stream, resume_from=checkpoint, **options)


@pytest.mark.parametrize("disconnect_after, options", [
    (0, {}), (50, {}), (999, {}), (1000, {}), (1001, {}), (2500, {}), (3499, {}),
    (300, {"max_tokens": 40_000}), (1500, {"max_tokens": 40_000}), (2100, {"max_tokens": 40_000}),  # ends at page 2,543
    # 1 page in 7 to val: the checkpoints (every 1,000 pages) don't fall on a multiple of 7, so a resume that
    # lost count of the pages would send the wrong ones to val
    (2500, {"train_fraction": 6 / 7}),
])
def test_an_interrupted_prep_resumes_to_exactly_the_same_files(tmp_path, disconnect_after, options):
    prep(tmp_path / "straight", Stream(PAGES), **options)
    with pytest.raises(KeyboardInterrupt):
        prep(tmp_path / "interrupted", Stream(PAGES, disconnect_after), **options)
    resume(tmp_path / "interrupted", **options)
    assert file_hashes(tmp_path / "interrupted") == file_hashes(tmp_path / "straight")


def test_checkpoints_advance_and_resume_starts_from_the_newest(tmp_path):
    # a checkpoint at the start, then after each batch of 1,000 pages that finished a train shard
    with pytest.raises(KeyboardInterrupt):
        prep(tmp_path / "out", Stream(PAGES, disconnect_after=2500))
    assert resume_point(tmp_path / "out", settings())["documents_done"] == 2000
    assert json.loads((tmp_path / "out" / "checkpoint.previous.json").read_text())["documents_done"] == 1000
    # only the two checkpoints' half-filled buffers are kept, not every one ever saved
    buffers = sorted(p.name for p in (tmp_path / "out").glob("buffer_*.bin"))
    named = sorted(json.loads((tmp_path / "out" / checkpoint_file).read_text())[name]["buffer_file"]
                   for checkpoint_file in ("checkpoint.json", "checkpoint.previous.json") for name in ("train", "val"))
    assert buffers == named


def test_a_checkpoint_is_saved_only_after_a_batch_that_finished_a_train_shard(tmp_path):
    # with 20,000-token train shards, the first batch of 1,000 pages (~16,000 train tokens) finishes none --
    # so no checkpoint after it: on FineWeb each one copies the half-filled buffers
    with pytest.raises(KeyboardInterrupt):
        prep(tmp_path / "out", Stream(PAGES, disconnect_after=1500), shard_size=20_000)
    assert resume_point(tmp_path / "out", settings(shard_size=20_000))["documents_done"] == 0


def test_a_kill_after_the_first_shard_but_before_its_checkpoint_still_resumes(tmp_path, monkeypatch):
    # the first shards are written partway through the first batch; the checkpoint comes at its end
    prep(tmp_path / "straight", Stream(PAGES))
    real_write = ShardWriter.write
    writes = []
    def write_then_die(self):
        real_write(self)
        writes.append(self.name)
        if len(writes) == 3:
            raise KeyboardInterrupt("killed")
    monkeypatch.setattr(ShardWriter, "write", write_then_die)
    with pytest.raises(KeyboardInterrupt):
        prep(tmp_path / "out", Stream(PAGES))
    monkeypatch.undo()
    assert list((tmp_path / "out").glob("train_0*.bin"))          # shards were written, but no checkpoint counts them
    resume(tmp_path / "out")
    assert file_hashes(tmp_path / "out") == file_hashes(tmp_path / "straight")


def test_a_finished_prep_keeps_its_last_checkpoints_until_removed(tmp_path):
    # on Google Drive the last files may not have arrived yet: the end must stay redoable until checked
    out = tmp_path / "out"
    prep(out, Stream(PAGES))
    assert (out / "info.json").exists() and (out / "checkpoint.json").exists()
    # the kept buffers can't be mistaken for shards, even by a plain "train_*.bin" search
    for name in ("train", "val"):
        assert sorted(out.glob(f"{name}_*.bin")) == shard_paths(out, name)
    (out / "info.json.tmp").write_text("{")                         # left by a kill mid-write
    remove_checkpoints(out)
    assert not [p for p in out.iterdir() if p.name.startswith(("checkpoint", "buffer_")) or p.suffix == ".tmp"]


def test_without_checkpoints_none_are_saved(tmp_path):
    # train_on_text's quick builds of .txt files never need resuming
    prepare(iter(PAGES), TOKENIZER, tmp_path / "out", train_fraction=0.99, whole_documents=True, shard_size=1500)
    assert not list((tmp_path / "out").glob("checkpoint*"))


def cut_newest_train_shard_short(out):
    # as if its upload to Google Drive never finished
    newest = json.loads((out / "checkpoint.json").read_text())
    shard = out / f"train_{newest['train']['shard_index'] - 1:06d}.bin"
    shard.write_bytes(shard.read_bytes()[:100])


def interrupted_prep_with(tmp_path, damage, **options):
    # a prep stopped after its checkpoints at pages 1,000 and 2,000, then damaged
    out = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        prep(out, Stream(PAGES, disconnect_after=2500), **options)
    newest = json.loads((out / "checkpoint.json").read_text())
    if damage in ("newest shard", "everything"):
        cut_newest_train_shard_short(out)
    if damage == "everything":
        (out / json.loads((out / "checkpoint.previous.json").read_text())["val"]["buffer_file"]).unlink()
    if damage == "newest checkpoint file":
        (out / "checkpoint.json").write_text('{"settings": {"tr')        # cut off mid-write
    if damage == "newest checkpoint file, the wrong shape":
        (out / "checkpoint.json").write_text('{"settings": null}')
    if damage == "newest checkpoint file missing a part":
        del newest["train"]
        (out / "checkpoint.json").write_text(json.dumps(newest))
    if damage == "newest checkpoint file missing":
        (out / "checkpoint.json").unlink()                              # only the older one arrived
    if damage == "newest half-filled buffer":
        buffer_file = out / newest["val"]["buffer_file"]
        buffer_file.write_bytes(buffer_file.read_bytes()[:10])
    return out


@pytest.mark.parametrize("damage, options", [
    ("newest shard", {}), ("newest checkpoint file", {}), ("newest checkpoint file, the wrong shape", {}),
    ("newest checkpoint file missing a part", {}),
    ("newest checkpoint file missing", {}), ("newest half-filled buffer", {}),
    # FineWeb's proportions: val's shards are big, so val's shard number stays put from checkpoint to
    # checkpoint -- each checkpoint's buffer files must still be its own
    ("newest shard", {"val_shard_size": 10_000}),
])
def test_resume_falls_back_to_the_previous_checkpoint_when_the_newest_files_are_incomplete(tmp_path, damage, options):
    # a disconnect can stop the newest files from reaching the disk -- or Google Drive -- in full
    prep(tmp_path / "straight", Stream(PAGES), **options)
    out = interrupted_prep_with(tmp_path, damage, **options)
    resume(out, **options)
    assert file_hashes(out) == file_hashes(tmp_path / "straight")


@pytest.mark.parametrize("damage", ["newest shard", "newest checkpoint file"])
def test_after_falling_back_the_good_fallback_is_kept(tmp_path, damage):
    # resume past a damaged newest checkpoint, then get interrupted again right after the next checkpoint
    # with ITS newest shard incomplete: the older, good checkpoint must still be there to fall back to
    prep(tmp_path / "straight", Stream(PAGES))
    out = interrupted_prep_with(tmp_path, damage)
    with pytest.raises(KeyboardInterrupt):
        resume(out, disconnect_after=2500)
    cut_newest_train_shard_short(out)
    resume(out)
    assert file_hashes(out) == file_hashes(tmp_path / "straight")


def test_a_stale_incomplete_checkpoint_file_is_never_kept_as_the_fallback(tmp_path):
    # Google Drive can deliver an OLDER, incomplete copy of checkpoint.json. It's skipped on resume -- and
    # the next checkpoint must not make it the fallback, even though it counts fewer pages
    prep(tmp_path / "straight", Stream(PAGES))
    out = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt):
        prep(out, Stream(PAGES, disconnect_after=2500))
    stale = json.loads((out / "checkpoint.json").read_text())
    stale["documents_done"] = 1500
    stale["train"]["buffer_file"] = "buffer_train_999999.bin"          # never arrived
    (out / "checkpoint.json").write_text(json.dumps(stale))
    with pytest.raises(KeyboardInterrupt):
        resume(out, disconnect_after=2500)
    cut_newest_train_shard_short(out)
    resume(out)
    assert file_hashes(out) == file_hashes(tmp_path / "straight")


def test_resume_refuses_when_no_checkpoint_is_complete(tmp_path):
    out = interrupted_prep_with(tmp_path, "everything")
    with pytest.raises(PrepRefused, match="incomplete"):
        resume_point(out, settings())


def test_a_damaged_checkpoint_file_with_nothing_older_is_refused_not_restarted(tmp_path):
    # only the starting checkpoint exists, and it's damaged: say so -- don't silently start over
    with pytest.raises(KeyboardInterrupt):
        prep(tmp_path / "out", Stream(PAGES, disconnect_after=50))
    (tmp_path / "out" / "checkpoint.json").write_text("{")
    with pytest.raises(PrepRefused, match="incomplete"):
        resume_point(tmp_path / "out", settings())


def test_resume_refuses_different_settings(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        prep(tmp_path / "out", Stream(PAGES, disconnect_after=700))
    with pytest.raises(PrepRefused, match="different settings"):
        resume_point(tmp_path / "out", settings(shard_size=3000))


def test_a_fresh_prep_refuses_a_folder_holding_an_unfinished_one(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        prep(tmp_path / "out", Stream(PAGES, disconnect_after=700))
    with pytest.raises(PrepRefused, match="unfinished prep"):
        prep(tmp_path / "out", Stream(PAGES))


def test_text_shards_never_deletes_an_unfinished_whole_document_prep(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "story.txt").write_text("The first story. " * 10)
    with pytest.raises(KeyboardInterrupt):
        prep(tmp_path / "fineweb", Stream(PAGES, disconnect_after=700))
    before = file_hashes(tmp_path / "fineweb")
    with pytest.raises(PrepRefused, match="whole documents"):
        text_shards(tmp_path / "docs", TOKENIZER, tmp_path / "fineweb", 0.9)
    assert file_hashes(tmp_path / "fineweb") == before


# ---------- prepare_data.py: resumes when run again, checks a finished prep, refuses what it can't do ----------

spec = importlib.util.spec_from_file_location("prepare_data", SRC / "run" / "prepare_data.py")
prepare_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_data)


class FakeFineWeb(Stream):
    # FineWeb, without the download: the same pages, as rows, failing partway if asked to
    def __init__(self, fail_at_page=None, error=ConnectionError):
        super().__init__(PAGES)
        self.fail_at_page, self.error = fail_at_page, error

    def __iter__(self):
        for text in super().__iter__():
            if self.position - 1 == self.fail_at_page:
                raise self.error("simulated: the connection dropped, or a downloaded chunk was corrupt")
            yield {"text": text}


def run(monkeypatch, out, fail_at_page=None, error=ConnectionError, shard_size=1500, max_tokens=None):
    # prepare_data.py's main(), as the command line would call it -- with the fake FineWeb, pinned to a version
    monkeypatch.setattr(prepare_data, "load_dataset", lambda *args, revision, **kwargs: FakeFineWeb(fail_at_page, error))
    prepare_data.main(argparse.Namespace(out=str(out), shard_size=shard_size, max_tokens=max_tokens))


@pytest.fixture(scope="module")
def straight(tmp_path_factory):
    # one uninterrupted prep to compare against -- copy it, never change it
    out = tmp_path_factory.mktemp("straight") / "out"
    with pytest.MonkeyPatch.context() as monkeypatch:
        run(monkeypatch, out)
    return out


@pytest.mark.parametrize("error", [ConnectionError, ValueError])     # a corrupt download chunk is a ValueError
def test_prepare_data_resumes_when_run_again_then_checks_every_file(tmp_path, monkeypatch, capsys, straight, error):
    out = tmp_path / "out"
    with pytest.raises(error):
        run(monkeypatch, out, fail_at_page=2500, error=error)
    run(monkeypatch, out)
    assert "resuming" in capsys.readouterr().out
    assert data_files(out) == data_files(straight)
    # once more: every file checked, then the kept checkpoints -- and any temporary file a kill left -- go
    (out / "buffer_val_000003.bin.tmp").write_bytes(b"half")
    run(monkeypatch, out)
    assert "every file is there in full" in capsys.readouterr().out
    assert file_hashes(out) == data_files(straight)


@pytest.mark.parametrize("lost", ["the last train shard", "the val shard", "info.json and the last train shard"])
def test_prepare_data_redoes_the_end_when_a_finished_prep_is_missing_a_file(tmp_path, monkeypatch, capsys, straight, lost):
    # on Google Drive, info.json can arrive before the last big shard has finished uploading
    out = tmp_path / "out"
    shutil.copytree(straight, out)
    if "train" in lost:
        max(out.glob("train_0*.bin")).unlink()
    if "val" in lost:
        max(out.glob("val_0*.bin")).unlink()
    if "info.json" in lost:
        (out / "info.json").unlink()
    run(monkeypatch, out)
    if "info.json" not in lost:
        assert "a file is missing or short" in capsys.readouterr().out
    assert data_files(out) == data_files(straight)


def test_prepare_data_refuses_a_checked_prep_that_later_lost_a_file(tmp_path, monkeypatch, capsys, straight):
    # once checked, a prep's checkpoints are gone -- a file deleted after that can't be redone
    out = tmp_path / "out"
    shutil.copytree(straight, out)
    run(monkeypatch, out)
    assert "every file is there in full" in capsys.readouterr().out
    max(out.glob("train_0*.bin")).unlink()
    with pytest.raises(PrepRefused, match="no checkpoint to redo them from"):
        run(monkeypatch, out)


def test_prepare_data_refuses_an_unfinished_prep_started_with_other_settings(tmp_path, monkeypatch, capsys):
    with pytest.raises(ConnectionError):
        run(monkeypatch, tmp_path / "out", fail_at_page=2500)
    with pytest.raises(PrepRefused, match="run it with the same options"):
        run(monkeypatch, tmp_path / "out", max_tokens=20_000.0)
    assert "resuming" not in capsys.readouterr().out                 # refused before saying it's resuming


def test_prepare_data_refuses_a_finished_prep_made_with_other_settings(tmp_path, monkeypatch):
    # e.g. a quick 10M-token check made in the folder the full prep is about to use
    run(monkeypatch, tmp_path / "out", max_tokens=20_000.0)
    with pytest.raises(PrepRefused, match="finished prep made with different settings"):
        run(monkeypatch, tmp_path / "out")


@pytest.mark.parametrize("bad", [dict(shard_size=0), dict(max_tokens=0.5), dict(max_tokens=math.nan),
                                 dict(max_tokens=math.inf)])
def test_prepare_data_refuses_bad_options_before_downloading_anything(tmp_path, monkeypatch, bad):
    def no_download(*args, **kwargs):
        raise AssertionError("FineWeb was opened before the options were checked")
    monkeypatch.setattr(prepare_data, "load_dataset", no_download)
    options = {"out": str(tmp_path / "out"), "shard_size": 1500, "max_tokens": None} | bad
    with pytest.raises(PrepRefused, match="must be"):
        prepare_data.main(argparse.Namespace(**options))


def test_prepare_data_says_what_to_do_after_each_kind_of_stop():
    assert prepare_data.stop_message(PrepRefused("different settings")).startswith("prep refused")
    assert "out of disk space" in prepare_data.stop_message(OSError(errno.ENOSPC, "No space left on device"))
    for interruption in (ConnectionError("dropped"), ValueError("corrupt chunk"), KeyboardInterrupt()):
        assert "run the same command again" in prepare_data.stop_message(interruption)


# what only `python prepare_data.py` itself does: the exit code, and the message (traceback or not)
FAKE_COMMAND = """
import os, runpy, sys
import datasets

class FakeFineWeb:
    position = 0
    def __iter__(self):
        while self.position < 3500:
            if self.position == int(os.environ["FAIL_AT_PAGE"]):
                raise ValueError("simulated: a downloaded chunk was corrupt")
            self.position += 1
            yield {"text": f"Web page {self.position - 1} says the Martians landed near town number {(self.position - 1) * 7}."}
    def state_dict(self):
        return {"position": self.position}
    def load_state_dict(self, state):
        self.position = state["position"]

datasets.load_dataset = lambda *args, revision, **kwargs: FakeFineWeb()
sys.argv = sys.argv[1:]                     # as if run as: python prepare_data.py --out ...
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def test_the_command_exits_with_the_right_code_and_message(tmp_path):
    script = tmp_path / "fake_command.py"
    script.write_text(FAKE_COMMAND)

    def command(fail_at_page, *options):
        env = dict(os.environ, PYTHONPATH=str(SRC), HF_HUB_OFFLINE="1", FAIL_AT_PAGE=str(fail_at_page))
        return subprocess.run([sys.executable, str(script), str(SRC / "run" / "prepare_data.py"), "--shard-size", "1500",
                               "--out", str(tmp_path / "out"), *options], capture_output=True, text=True, env=env,
                              timeout=120)

    stopped = command(2500)
    assert stopped.returncode == 1
    assert "Traceback" in stopped.stderr and "run the same command again" in stopped.stderr
    refused = command(-1, "--max-tokens", "2e4")                   # an unfinished prep, with other settings
    assert refused.returncode == 1
    assert "prep refused" in refused.stderr and "Traceback" not in refused.stderr
    assert "run the same command again" not in refused.stderr
    resumed = command(-1)
    assert resumed.returncode == 0 and "resuming" in resumed.stdout


@pytest.mark.parametrize("output", ["open", "closed"])       # closed: e.g. piped into `head`, which has quit
def test_quit_now_doesnt_wait_for_background_threads_or_leave_a_warning(tmp_path, output):
    # like the FineWeb downloader: a thread that never finishes, and a semaphore a normal exit would clean
    # up. (A "spawn" lock, so the warning would show on Linux too -- there, by default, no warning is possible.)
    script = tmp_path / "hang.py"
    close_output = 'print("unflushed", end=""); os.close(1)' if output == "closed" else ""
    script.write_text(f"""
import multiprocessing, os, runpy, threading, time
quit_now = runpy.run_path({str(SRC / "run" / "prepare_data.py")!r})["quit_now"]
threading.Thread(target=time.sleep, args=(3600,)).start()
lock = multiprocessing.get_context("spawn").Lock()
{close_output}
quit_now(3)
""")
    env = dict(os.environ, PYTHONPATH=str(SRC), HF_HUB_OFFLINE="1")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 3
    assert "leaked semaphore" not in result.stderr
