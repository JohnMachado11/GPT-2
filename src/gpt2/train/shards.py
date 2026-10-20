import hashlib
import json
import math
import os
from itertools import batched
from pathlib import Path

import numpy as np
import torch


# A shard is a flat file of token ids, 2 bytes each (uint16), no header.
# uint16 holds 0..65,535 -- GPT-2's largest id is 50,256, so every id fits.
TOKEN_DTYPE = np.uint16


class PrepRefused(Exception):
    # A prep that can't go on as asked -- the folder holds something else, the settings differ, a file is
    # missing... The message says what to do; running the same command again won't help.
    pass


def write_shard(path, token_ids):
    token_ids = np.asarray(token_ids)
    if token_ids.size == 0:
        raise ValueError("no token ids to write")
    if token_ids.min() < 0 or token_ids.max() >= 2 ** 16:
        raise ValueError(f"token ids must be 0..65,535 to fit in uint16 (got {token_ids.min()}..{token_ids.max()})")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    token_ids.astype(TOKEN_DTYPE).tofile(path)


def read_shard(path):
    # memory-map the file: the OS reads only the parts you actually touch,
    # so a 100M-token shard costs almost no RAM to "open"
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def shard_paths(folder, name):
    # one set's shard files, in order -- exactly <name>_NNNNNN.bin, so a stray copy such as
    # "train_000000 copy.bin" is never mistaken for a shard
    return sorted(Path(folder).glob(f"{name}_[0-9][0-9][0-9][0-9][0-9][0-9].bin"))


class ShardWriter:
    # Fills a fixed-size buffer with token ids; every time it's full, it's written out as the
    # next shard of this set: train_000000.bin, train_000001.bin, ... One writer per set.
    #   fresh=False: the folder already holds this set's shards (a resumed prep -- see resumable_prep.py)

    def __init__(self, out_dir, name, shard_size, fresh=True):
        if shard_size < 1:
            raise PrepRefused(f"shard_size must be at least 1, got {shard_size}")
        self.out_dir = Path(out_dir)
        self.name = name                    # "train" or "val"
        existing = shard_paths(self.out_dir, name)
        if fresh and existing:
            raise PrepRefused(
                f"{self.out_dir} already has {len(existing)} {name} shard file(s). "
                f"Delete the folder or choose another one, so old and new shards can't mix."
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
        path = self.out_dir / f"{self.name}_{self.shard_index:06d}.bin"
        write_shard(path, self.buffer[: self.filled])
        print(f"  wrote {path.name}: {self.filled:,} tokens")
        self.shard_index += 1
        self.filled = 0


def load_tokens(folder, name):
    # every shard of one set ("train" or "val"), joined into one tensor -- for data small enough to hold
    # in memory, like a book, read with RandomWindows. (Big data stays on disk, read with ShardLoader.)
    paths = shard_paths(folder, name)
    if not paths:
        raise FileNotFoundError(f"no {name}_*.bin shards in {folder} -- delete the folder to rebuild it")
    return torch.from_numpy(np.concatenate([read_shard(path) for path in paths]).astype(np.int64))


def documents_from_folder(folder):
    # every .txt file in the folder is one document
    for path in sorted(Path(folder).glob("*.txt")):
        try:
            yield path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise PrepRefused(f"{path} isn't UTF-8 text -- re-save it as UTF-8") from error


def prepare(documents, tokenizer, out_dir, train_fraction, whole_documents, shard_size=100_000_000,
            val_shard_size=None, max_tokens=None, source=None, checkpoints=None):
    # Tokenize documents ONCE and write them to out_dir as train_*.bin + val_*.bin. The split into
    # train and val happens here, in one of two ways:
    #   whole_documents=False -> for a few LONG documents (books): the last (1 - train_fraction) of EACH
    #                            document goes to val, so every document shows up in both sets
    #   whole_documents=True  -> for many SHORT documents (web pages): every Nth document goes to val,
    #                            whole -- N = 1 / (1 - train_fraction), so 0.99 means every 100th
    # val_shard_size: tokens per val shard (default: the same as a train shard)
    # checkpoints:    for a long prep that may be interrupted (FineWeb, on Colab) -- a Checkpoints object
    #                 from resumable_prep.py, which saves checkpoints along the way and can resume from one
    check_train_fraction(train_fraction, whole_documents)
    val_every = round(1 / (1 - train_fraction))
    if max_tokens is not None and max_tokens < 1:
        raise PrepRefused(f"max_tokens must be at least 1, got {max_tokens}")
    if val_shard_size is None:
        val_shard_size = shard_size
    out_dir = Path(out_dir)
    settings = {"train_fraction": train_fraction, "whole_documents": whole_documents, "shard_size": shard_size,
                "val_shard_size": val_shard_size, "max_tokens": max_tokens, "source": source}
    if checkpoints is None:
        documents_done = 0
        train = ShardWriter(out_dir, "train", shard_size)
        val = ShardWriter(out_dir, "val", val_shard_size)
    else:
        documents_done, train, val = checkpoints.start(settings)     # fresh, or from where a checkpoint left off
    endoftext_id = tokenizer.eos_token_id
    info_path = out_dir / "info.json"
    info_path.unlink(missing_ok=True)       # an old info.json would make this half-built folder look finished

    def add(writer, token_ids):
        if max_tokens is not None:
            token_ids = token_ids[: max_tokens - train.total_tokens - val.total_tokens]   # don't overshoot
        writer.add(token_ids)

    # Tokenize 1000 short documents per call -- much faster than one at a time -- but long documents one
    # at a time: many big files in one call would all be held in memory at once.
    # split_special_tokens: a literal "<|endoftext|>" in the text stays plain text.
    for batch in batched(documents, 1000 if whole_documents else 1):
        batch_token_ids = tokenizer(list(batch), split_special_tokens=True, return_attention_mask=False)["input_ids"]
        train_shards_before = train.shard_index
        for token_ids in batch_token_ids:
            if whole_documents:
                add(val if documents_done % val_every == 0 else train, token_ids + [endoftext_id])
            else:
                split = int(train_fraction * len(token_ids))
                add(train, token_ids[:split] + [endoftext_id])     # <|endoftext|> marks where each piece ends
                add(val, token_ids[split:] + [endoftext_id])
            documents_done += 1
        if max_tokens is not None and train.total_tokens + val.total_tokens >= max_tokens:
            break       # (pages after the budget in this batch added nothing: add() trims them to zero)
        # checkpoint right after a train shard was written: the train buffer is nearly empty, so it's cheap
        if checkpoints is not None and train.shard_index > train_shards_before:
            checkpoints.save(documents_done, train, val)

    train.write()       # the last, partly filled shard of each set
    val.write()
    if train.total_tokens == 0 or val.total_tokens == 0:
        raise PrepRefused(f"the split left {train.total_tokens} train and {val.total_tokens} val tokens -- "
                          f"both need some: delete {out_dir}, then give it more documents or a bigger max_tokens")

    # written LAST (whole, via a rename), so a folder with info.json is a finished one
    write_json(info_path, dict(settings, train_tokens=train.total_tokens, val_tokens=val.total_tokens,
                               train_shards=train.shard_index, val_shards=val.shard_index))
    return train, val


def write_json(path, data):
    # write the whole file or nothing: to a temporary file first, then renamed into place
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def read_json(path):
    # a JSON file as a dict -- or None if it's missing or damaged
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def check_train_fraction(train_fraction, whole_documents):
    if not 0 < train_fraction < 1:
        raise PrepRefused(f"train_fraction must be between 0 and 1, got {train_fraction}")
    if whole_documents and not math.isclose(1 / (1 - train_fraction), round(1 / (1 - train_fraction))):
        raise PrepRefused(f"with whole documents, 1 in every N goes to val, so train_fraction must be 1 - 1/N "
                          f"(0.9, 0.95, 0.99, ...), got {train_fraction}")


def text_shards(folder, tokenizer, out_dir, train_fraction):
    # The shards for a folder of .txt files (what train_on_text.py trains on): built the first time, then
    # reused until the text or the split changes -- adding, editing or removing a file means a rebuild.
    # Returns the train and val tokens, each as one tensor.
    out_dir = Path(out_dir)
    files = sorted(Path(folder).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no .txt files found in {folder} -- add some text to train on")
    check_train_fraction(train_fraction, whole_documents=False)    # before anything is cleared out
    wanted = {"source": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in files},
              "train_fraction": train_fraction, "whole_documents": False}
    if is_finished_build_of(out_dir, wanted):
        print(f"using the shards in {out_dir} (the text hasn't changed)")
    else:
        info = read_json(out_dir / "info.json") or {}
        unfinished = any(out_dir.glob("checkpoint*.json"))      # a long prep's checkpoints (resumable_prep.py)
        if unfinished or info.get("whole_documents"):          # never delete, say, a FineWeb build
            raise PrepRefused(f"{out_dir} holds shards built from whole documents (like FineWeb's), not from "
                              f".txt files -- point text_shards at another folder")
        print(f"tokenizing {', '.join(f.name for f in files)} -> {out_dir}")
        for old in [out_dir / "info.json", *shard_paths(out_dir, "train"), *shard_paths(out_dir, "val")]:
            old.unlink(missing_ok=True)
        prepare(documents_from_folder(folder), tokenizer, out_dir, train_fraction, whole_documents=False,
                source=wanted["source"])
    return load_tokens(out_dir, "train"), load_tokens(out_dir, "val")


def is_finished_build_of(out_dir, wanted):
    # True only if out_dir holds a FINISHED build (info.json is written last) of exactly what's wanted,
    # with every shard still there in full. A missing, damaged or different info.json means no.
    info = read_json(Path(out_dir) / "info.json")
    if info is None or any(info.get(key) != value for key, value in wanted.items()):
        return False
    bytes_per_token = np.dtype(TOKEN_DTYPE).itemsize
    for name in ("train", "val"):
        paths, tokens = shard_paths(out_dir, name), info.get(f"{name}_tokens")
        if (len(paths) != info.get(f"{name}_shards") or not isinstance(tokens, int)
                or sum(path.stat().st_size for path in paths) != tokens * bytes_per_token):
            return False
    return True
