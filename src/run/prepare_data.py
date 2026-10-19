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
# Val pages are spread through the whole stream, so val_000000.bin is written near the end.
# Run with --help to see every option.

import argparse
import os
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import logging as transformers_logging

from gpt2.train.shards import prepare


here = Path(__file__).parent


def documents_from_fineweb():
    # FineWeb-Edu's 10B-token sample, streamed: rows arrive as you read them,
    # so nothing is downloaded up front
    rows = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for row in rows:
        yield row["text"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize FineWeb-Edu once and save it as uint16 shard files.")
    parser.add_argument("--out", default=str(here / "shards" / "fineweb"), help="where to write the shard files")
    parser.add_argument("--shard-size", type=int, default=100_000_000, help="tokens per shard")
    parser.add_argument("--max-tokens", type=float, default=None, help="token budget, e.g. 2.5e9 (default: everything)")
    args = parser.parse_args()

    transformers_logging.set_verbosity_error()      # hide the harmless ">1024 tokens" warning
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    documents = documents_from_fineweb()
    max_tokens = int(args.max_tokens) if args.max_tokens is not None else None

    # many short web pages: every 100th goes to val whole, so val is spread across the whole dataset
    train, val = prepare(documents, tokenizer, args.out, train_fraction=0.99, whole_documents=True,
                         shard_size=args.shard_size, max_tokens=max_tokens,
                         source="HuggingFaceFW/fineweb-edu sample-10BT")
    documents.close()   # done reading: stop the document stream

    print(f"\ndone -> {args.out}")
    print(f"  train: {train.total_tokens:,} tokens in {train.shard_index} shard(s)")
    print(f"  val:   {val.total_tokens:,} tokens in {val.shard_index} shard(s)")

    sys.stdout.flush()
    os._exit(0)     # all shards are written; don't wait for the downloader's background threads
