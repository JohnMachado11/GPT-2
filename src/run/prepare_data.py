# Tokenize text ONCE into uint16 shard files, for training on data too big for memory.
# (For a book or a few .txt files, use train_on_text.py instead -- no prep step needed.)
#
#   # FineWeb-Edu, the real data -- all ~10B tokens (~20 GB of shards)
#   .venv/bin/python src/run/prepare_data.py --source fineweb
#
#   # quick FineWeb check first -- 1M tokens in small shards
#   .venv/bin/python src/run/prepare_data.py --source fineweb --max-tokens 1e6 --shard-size 250000
#
#   # test the pipeline locally on the .txt files in src/run/data/
#   .venv/bin/python src/run/prepare_data.py --source folder --shard-size 20000
#
# Shard 0 is validation, the rest are training. Run with --help to see every option.

import argparse
import os
import sys
from itertools import batched
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from transformers import logging as transformers_logging

from gpt2.train.shards import TOKEN_DTYPE, write_shard


here = Path(__file__).parent


def documents_from_folder(folder):
    # every .txt file in the folder is one document
    for path in sorted(Path(folder).glob("*.txt")):
        yield path.read_text()


def documents_from_fineweb():
    # FineWeb-Edu's 10B-token sample, streamed: rows arrive as you read them,
    # so nothing is downloaded up front
    from datasets import load_dataset       # imported here so the folder path never needs it
    rows = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for row in rows:
        yield row["text"]


class ShardWriter:
    # Fills a fixed-size buffer with token ids; every time it's full, it's written out
    # as the next shard. Shard 0 is validation, every shard after it is training.

    def __init__(self, out_dir, shard_size):
        self.out_dir = Path(out_dir)
        existing = sorted(self.out_dir.glob("*.bin"))
        if existing:
            raise FileExistsError(
                f"{self.out_dir} already has {len(existing)} shard file(s). "
                f"Delete them or choose another --out, so old and new shards can't mix."
            )
        self.buffer = np.empty(shard_size, dtype=TOKEN_DTYPE)
        self.filled = 0         # how many slots of the buffer hold real tokens
        self.shard_index = 0
        self.total_tokens = 0

    def add(self, token_ids):
        while token_ids:
            space = len(self.buffer) - self.filled
            chunk, token_ids = token_ids[:space], token_ids[space:]     # what fits now, what spills over
            self.buffer[self.filled : self.filled + len(chunk)] = chunk
            self.filled += len(chunk)
            self.total_tokens += len(chunk)
            if self.filled == len(self.buffer):
                self.write()

    def write(self):
        if self.filled == 0:
            return
        if self.shard_index == 0:
            name = "val_000000.bin"
        else:
            name = f"train_{self.shard_index - 1:06d}.bin"
        write_shard(self.out_dir / name, self.buffer[: self.filled])
        print(f"  wrote {name}: {self.filled:,} tokens")
        self.shard_index += 1
        self.filled = 0


def tokenized(documents, tokenizer, batch_size=1000):
    # tokenize many documents at once -- much faster than one at a time.
    # split_special_tokens: a literal "<|endoftext|>" in the text stays plain text
    for batch in batched(documents, batch_size):
        yield from tokenizer(list(batch), split_special_tokens=True)["input_ids"]


def prepare(documents, out_dir, shard_size, max_tokens=None):
    transformers_logging.set_verbosity_error()      # hide the harmless ">1024 tokens" warning
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    endoftext_id = tokenizer.eos_token_id
    writer = ShardWriter(out_dir, shard_size)

    for token_ids in tokenized(documents, tokenizer):
        token_ids = token_ids + [endoftext_id]                          # mark where this document ends
        if max_tokens is not None:
            token_ids = token_ids[: max_tokens - writer.total_tokens]   # don't overshoot the budget
        writer.add(token_ids)
        if max_tokens is not None and writer.total_tokens >= max_tokens:
            break

    writer.write()      # the last, partly filled shard
    return writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize text once and save it as uint16 shard files.")
    parser.add_argument("--source", choices=["folder", "fineweb"], default="folder",
                        help="folder = the .txt files in src/run/data; fineweb = FineWeb-Edu, streamed")
    parser.add_argument("--out", default=str(here / "shards"), help="where to write the shard files")
    parser.add_argument("--shard-size", type=int, default=100_000_000, help="tokens per shard")
    parser.add_argument("--max-tokens", type=float, default=None, help="token budget, e.g. 2.5e9 (default: everything)")
    args = parser.parse_args()

    if args.source == "folder":
        documents = documents_from_folder(here / "data")
    else:
        documents = documents_from_fineweb()

    max_tokens = int(args.max_tokens) if args.max_tokens is not None else None
    writer = prepare(documents, args.out, args.shard_size, max_tokens)
    documents.close()   # done reading: stop the document stream

    print(f"\ndone: {writer.total_tokens:,} tokens in {writer.shard_index} shard(s) -> {args.out}")
    if writer.shard_index < 2:
        print("warning: everything fit in the validation shard -- lower --shard-size to get training shards")

    sys.stdout.flush()
    os._exit(0)     # all shards are written; don't wait for the downloader's background threads