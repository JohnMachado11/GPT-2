# Tokenize FineWeb-Edu ONCE into uint16 shard files (train_*.bin + val_*.bin), for training on data too
# big for memory. (Your own .txt files don't need this: train_on_text.py makes their shards itself.)
#
#   # all of it -- ~10B tokens (~20 GB of shards) -> src/run/shards/fineweb/
#   .venv/bin/python src/run/prepare_data.py
#
#   # quick check first -- 10M tokens (well under a minute) in small shards, in a separate folder
#   .venv/bin/python src/run/prepare_data.py --max-tokens 1e7 --shard-size 2500000 --out src/run/shards/fineweb-check
#
# The split is 99/1: every 100th web page goes to val, whole (about 100M val tokens out of 10B).
# Interrupted (a disconnect, Ctrl-C, an error)? Run the SAME command again: it picks up from the last
# finished train shard and ends with exactly the shards an uninterrupted run would have written. (A resume
# can be quiet for a few minutes first: it re-reads part of a FineWeb file to find its place.)
# Finished? Running it once more, from a new session, checks that every file is there in full (on Google
# Drive, the last ones can still be uploading when Colab disconnects) and redoes the end if one isn't --
# then it removes the checkpoints it kept for that.
# Run with --help to see every option.

import argparse
import errno
import math
import multiprocessing.util
import os
import signal
import sys
import traceback
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import logging as transformers_logging

from gpt2.train.resumable_prep import Checkpoints, remove_checkpoints, resume_point
from gpt2.train.shards import PrepRefused, is_finished_build_of, prepare, read_json


here = Path(__file__).parent
# One exact version of FineWeb-Edu, so a prep resumed days later -- after a new release -- reads the same pages
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SOURCE = f"HuggingFaceFW/fineweb-edu sample-10BT @ {REVISION}"
# Val gets smaller shards than train: its half-filled buffer is saved with every checkpoint, and val fills
# slowly (1% of the pages), so a 100M-token val buffer would mean copying up to 200 MB at each checkpoint.
VAL_SHARD_SIZE = 10_000_000


def main(args):
    out_dir = Path(args.out)
    if args.shard_size < 1:
        raise PrepRefused(f"--shard-size must be at least 1, got {args.shard_size}")
    if args.max_tokens is not None and not 1 <= args.max_tokens < math.inf:      # (also catches nan)
        raise PrepRefused(f"--max-tokens must be a number of at least 1, got {args.max_tokens}")
    settings = {"source": SOURCE,
                "train_fraction": 0.99, "whole_documents": True,    # many short web pages: every 100th to val, whole
                "shard_size": args.shard_size, "val_shard_size": VAL_SHARD_SIZE,
                "max_tokens": int(args.max_tokens) if args.max_tokens is not None else None}

    # 1. finished before? Then just check that every file is there
    info = read_json(out_dir / "info.json")
    if info is not None:
        found = {key: info.get(key) for key in settings}
        if found != settings:
            raise PrepRefused(f"{out_dir} holds a finished prep made with different settings ({found}) -- "
                              f"choose another --out, or delete it")
        if is_finished_build_of(out_dir, settings):
            remove_checkpoints(out_dir)
            print(f"{out_dir} is a finished prep, and every file is there in full -- nothing to do")
            return

    # 2. where to start: fresh, or from the newest complete checkpoint
    checkpoint = resume_point(out_dir, settings)

    # 3. it says it's finished, but a file is missing or short: redo the end from the kept checkpoint
    if info is not None:
        if checkpoint is None:
            raise PrepRefused(f"{out_dir} is missing files, and has no checkpoint to redo them from -- delete it "
                              f"and start again")
        print(f"{out_dir} says it's finished, but a file is missing or short (Google Drive hadn't finished "
              f"uploading?) -- redoing the end of the prep")

    transformers_logging.set_verbosity_error()      # hide the harmless ">1024 tokens" warning
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # FineWeb-Edu's 10B-token sample, streamed: rows arrive as you read them, so nothing is downloaded up
    # front -- and the stream can say where it is, so an interrupted prep restarts it right there
    rows = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True,
                        revision=REVISION)
    if checkpoint is not None:
        rows.load_state_dict(checkpoint["stream"])
        print(f"resuming {out_dir} after {checkpoint['documents_done']:,} pages "
              f"({checkpoint['train']['shard_index']} train shards were already written)")
    documents = (row["text"] for row in rows)

    train, val = prepare(documents, tokenizer, out_dir, **settings,
                         checkpoints=Checkpoints(out_dir, rows.state_dict, resume_from=checkpoint))

    print(f"\ndone -> {out_dir}")
    print(f"  train: {train.total_tokens:,} tokens in {train.shard_index} shard(s)")
    print(f"  val:   {val.total_tokens:,} tokens in {val.shard_index} shard(s)")
    print("(on Google Drive: after drive.flush_and_unmount(), or from a new session, run the same command "
          "once more to check every file arrived)")


def stop_message(error):
    # what to say when prep stops early
    if isinstance(error, PrepRefused):              # something a rerun can't fix: the message says what to do
        return f"prep refused: {error}"
    if isinstance(error, OSError) and error.errno in (errno.ENOSPC, errno.EDQUOT):
        return ("prep stopped: out of disk space (on Google Drive: storage is full -- empty its Trash too). "
                "Free some, then run the same command again to pick up from the last finished train shard")
    return "prep stopped -- run the same command again to pick up from the last finished train shard"


def quit_now(exit_code):
    # Quit at once, without waiting for background threads. When prep stops before the end of FineWeb (a
    # token budget, an error, Ctrl-C), the streaming library is still downloading in a background thread,
    # and a normal exit waits on it forever. Before that, the cleanup a normal exit would do (without it,
    # Python warns about a "leaked semaphore"). The finally makes sure the exit happens no matter what --
    # a second Ctrl-C, or output that can't be flushed because its reader went away.
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        multiprocessing.util._run_finalizers()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(exit_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize FineWeb-Edu once and save it as uint16 shard files.")
    parser.add_argument("--out", default=str(here / "shards" / "fineweb"), help="where to write the shard files")
    parser.add_argument("--shard-size", type=int, default=100_000_000, help="tokens per train shard")
    parser.add_argument("--max-tokens", type=float, default=None, help="token budget, e.g. 2.5e9 (default: everything)")
    args = parser.parse_args()

    exit_code = 1
    try:
        main(args)
        exit_code = 0
    except BaseException as error:          # a refusal, or an interruption: a dropped connection, Ctrl-C, a bug
        if not isinstance(error, PrepRefused):
            traceback.print_exc()
        print(f"\n{stop_message(error)}", file=sys.stderr)
    finally:
        quit_now(exit_code)
