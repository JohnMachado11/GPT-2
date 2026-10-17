from pathlib import Path

import numpy as np


# A shard is a flat file of token ids, 2 bytes each (uint16), no header.
# uint16 holds 0..65,535 -- GPT-2's largest id is 50,256, so every id fits.
TOKEN_DTYPE = np.uint16


def write_shard(path, token_ids):
    token_ids = np.asarray(token_ids)
    if token_ids.max() >= 2 ** 16:
        raise ValueError(f"token id {token_ids.max()} doesn't fit in uint16")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    token_ids.astype(TOKEN_DTYPE).tofile(path)


def read_shard(path):
    # memory-map the file: the OS reads only the parts you actually touch,
    # so a 100M-token shard costs almost no RAM to "open"
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")