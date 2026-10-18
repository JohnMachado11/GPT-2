from pathlib import Path

import numpy as np
import torch

from gpt2.train.shards import read_shard


def get_batch(token_data, batch_size, context_length, device="cpu"):
    if len(token_data) < context_length + 1:
        raise ValueError(
            f"need at least context_length+1 = {context_length + 1} tokens, "
            f"but this stream has {len(token_data)}. Add more text or lower context_length."
        )

    highest_start = len(token_data) - context_length - 1
    starts = torch.randint(0, highest_start + 1, (batch_size,))
    token_ids = torch.stack([token_data[s : s + context_length] for s in starts])
    targets = torch.stack([token_data[s + 1 : s + 1 + context_length] for s in starts])
    return token_ids.to(device), targets.to(device)


class RandomWindows:
    # Random windows from a token stream held in memory -- the right fit for book-sized runs.
    # Same next_batch() / reset() as ShardLoader, so train() can take either one.

    def __init__(self, token_data, batch_size, context_length, device="cpu"):
        self.token_data = token_data
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

    def next_batch(self):
        return get_batch(self.token_data, self.batch_size, self.context_length, self.device)

    def reset(self):
        pass    # random windows have no position to rewind


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
        for path in sorted(shard_paths):        # sorted: same order every run, so a saved position means the same file
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
        # exactly where the loader is -- a checkpoint saves this...
        return {"shard": self.shard, "position": self.position}

    def load_state(self, state):
        # ...and a resumed run puts it back, continuing from the very next token
        self.shard = state["shard"]
        self.position = state["position"]
        self.tokens = read_shard(self.shard_paths[self.shard])