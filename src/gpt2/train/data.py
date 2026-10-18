import hashlib
from pathlib import Path

import numpy as np
import torch

from gpt2.train.shards import read_shard


def shard_fingerprint(tokens):
    # a cheap content check for a shard: its first and last 4,096 tokens, hashed. A shard regenerated
    # under the same name -- even with the same length -- almost certainly gets a different fingerprint.
    return hashlib.sha256(np.ascontiguousarray(tokens[:4096]).tobytes() +
                          np.ascontiguousarray(tokens[-4096:]).tobytes()).hexdigest()[:16]


def get_batch(token_data, batch_size, context_length, device="cpu", generator=None):
    if len(token_data) < context_length + 1:
        raise ValueError(
            f"need at least context_length+1 = {context_length + 1} tokens, "
            f"but this stream has {len(token_data)}. Add more text or lower context_length."
        )

    highest_start = len(token_data) - context_length - 1
    starts = torch.randint(0, highest_start + 1, (batch_size,), generator=generator)     # generator=None: torch's global one
    token_ids = torch.stack([token_data[s : s + context_length] for s in starts])
    targets = torch.stack([token_data[s + 1 : s + 1 + context_length] for s in starts])
    return token_ids.to(device), targets.to(device)


class RandomWindows:
    # Random windows from a token stream held in memory -- the right fit for book-sized runs.
    # It offers the same things as ShardLoader, so train() can take either one:
    #   batch_size, context_length, next_batch(), reset(), state(), check_state(), load_state()
    #
    #   seed=None   -> windows come from torch's GLOBAL random generator. Use this for training:
    #                  a resume restores that generator, so the same windows come next.
    #   seed=<int>  -> windows come from this object's OWN generator, and reset() rewinds it, so every
    #                  pass after reset() sees the exact same windows. Use this for validation: the
    #                  numbers are then comparable from one check to the next, and checking never
    #                  disturbs which windows training draws.

    def __init__(self, token_data, batch_size, context_length, device="cpu", seed=None):
        self.token_data = token_data
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.seed = seed
        self.generator = None if seed is None else torch.Generator().manual_seed(seed)

    def next_batch(self):
        return get_batch(self.token_data, self.batch_size, self.context_length, self.device, self.generator)

    def reset(self):
        if self.generator is not None:
            self.generator.manual_seed(self.seed)     # back to the first window of the sequence

    def state(self):
        # the token count lets a resume notice that the training text changed underneath it
        state = {"num_tokens": len(self.token_data)}
        if self.generator is not None:
            state["generator"] = self.generator.get_state()
        return state

    def check_state(self, state):
        if "file" in state or "shard" in state:
            raise ValueError("can't resume: this run was saved reading shard files (ShardLoader), "
                             "but it's being resumed with text held in memory (RandomWindows)")
        saved_tokens = state.get("num_tokens")         # missing in checkpoints from before this check
        if saved_tokens is not None and saved_tokens != len(self.token_data):
            raise ValueError(f"can't resume: the training text changed ({saved_tokens:,} tokens when saved, "
                             f"{len(self.token_data):,} now)")

    def load_state(self, state):
        self.check_state(state)
        if self.generator is not None and "generator" in state:
            self.generator.set_state(state["generator"])


class ShardLoader:
    # Reads batches straight off the shard files on disk, IN ORDER: shard after shard,
    # position after position -- so one pass sees every token once.

    def __init__(self, shard_paths, batch_size, context_length, device="cpu"):
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

        # keep only shards with enough tokens for at least one batch
        # (a budget-trimmed last shard can be tiny)
        self.shard_paths = []
        for path in sorted(shard_paths):        # sorted: the same reading order every run
            if len(read_shard(path)) >= batch_size * context_length + 1:
                self.shard_paths.append(path)
            else:
                print(f"skipping {Path(path).name}: too short for even one batch")
        if not self.shard_paths:
            raise ValueError("no shard has enough tokens for one batch")

        self.reset()

    def reset(self):
        # back to the very first token of the first shard
        self.shard = 0
        self.position = 0
        self.tokens = read_shard(self.shard_paths[0])

    def next_batch(self):
        span = self.batch_size * self.context_length
        # one batch reads span + 1 tokens (the +1 is the final target); if this shard
        # doesn't have that many left, move to the next one (after the last, wrap around)
        if self.position + span + 1 > len(self.tokens):
            self.shard = (self.shard + 1) % len(self.shard_paths)
            self.position = 0
            self.tokens = read_shard(self.shard_paths[self.shard])

        chunk = torch.from_numpy(self.tokens[self.position : self.position + span + 1].astype(np.int64))
        token_ids = chunk[:-1].view(self.batch_size, self.context_length)     # every token but the last
        targets = chunk[1:].view(self.batch_size, self.context_length)        # every token but the first
        self.position += span
        return token_ids.to(self.device), targets.to(self.device)

    def state(self):
        # exactly where the loader is -- a checkpoint saves this. The shard is recorded by FILE NAME,
        # not by its number in the list, so a resume can't silently land in a different file if the
        # set of shard files has changed (a shard added, missing, or not yet synced).
        # The file's token count and a fingerprint of its first and last 4,096 tokens are saved too, so a
        # shard regenerated under the same name is noticed -- even at the same length.
        return {"file": Path(self.shard_paths[self.shard]).name, "position": self.position,
                "file_tokens": len(self.tokens), "file_fingerprint": shard_fingerprint(self.tokens)}

    def check_state(self, state):
        # raise if this saved position can't be restored here -- without changing anything
        if "file" not in state:
            raise ValueError("can't resume: this run wasn't saved reading shard files (or was saved by an "
                             "older version that recorded shards by number) -- start a fresh run")
        names = [Path(path).name for path in self.shard_paths]
        if state["file"] not in names:
            raise ValueError(f"can't resume: the saved position is in {state['file']}, which isn't among "
                             f"this run's shards ({len(names)} found)")
        tokens = read_shard(self.shard_paths[names.index(state["file"])])
        length = len(tokens)
        if "file_tokens" in state and state["file_tokens"] != length:
            raise ValueError(f"can't resume: {state['file']} had {state['file_tokens']:,} tokens when saved "
                             f"and has {length:,} now -- the shard was regenerated")
        if "file_fingerprint" in state and state["file_fingerprint"] != shard_fingerprint(tokens):
            raise ValueError(f"can't resume: {state['file']}'s contents changed since it was saved "
                             f"-- the shard was regenerated")
        if not 0 <= state["position"] <= length:
            raise ValueError(f"can't resume: saved position {state['position']:,} is outside "
                             f"{state['file']} ({length:,} tokens) -- did the file change?")

    def load_state(self, state):
        # ...and a resumed run puts it back, continuing from the very next token
        self.check_state(state)
        self.shard = [Path(path).name for path in self.shard_paths].index(state["file"])
        self.position = state["position"]
        self.tokens = read_shard(self.shard_paths[self.shard])