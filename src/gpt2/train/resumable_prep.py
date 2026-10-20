# Lets a long prep survive being interrupted -- FineWeb takes hours, and Google Colab disconnects -- and
# copes with Google Drive finishing uploads late. Nothing here is needed for a book: train_on_text.py's
# preps take seconds, and go straight through shards.py.
#
#   checkpoints: saved at the start and after each batch of documents that finished a train shard --
#                where the document stream is, the counters, and each writer's half-filled buffer
#   resuming:    run the prep again; it picks up from the newest complete checkpoint, and the shards come
#                out exactly as an uninterrupted prep would have written them
#   Drive:       the checkpoint before the newest is kept too, in case the newest files never fully
#                arrived; and a finished prep keeps its last checkpoints until a later run has checked
#                every file (prepare_data.py does that)

import os
from pathlib import Path

import numpy as np

from gpt2.train.shards import TOKEN_DTYPE, PrepRefused, ShardWriter, read_json, write_json


NEWEST = "checkpoint.json"                  # the newest checkpoint
PREVIOUS = "checkpoint.previous.json"       # the one before it: the fallback


class Checkpoints:
    # Hand this to prepare() as checkpoints=...: it starts the prep -- fresh, or from a checkpoint -- and
    # saves a checkpoint whenever prepare() asks.
    #   stream_position: a function that says where the document stream is right now
    #   resume_from:     the checkpoint to pick up from (from resume_point), or None for a fresh start

    def __init__(self, out_dir, stream_position, resume_from=None):
        self.out_dir = Path(out_dir)
        self.stream_position = stream_position
        self.resume_from = resume_from
        self.settings = None                # set by start()

    def start(self, settings):
        self.settings = settings
        if self.resume_from is not None:
            train = restored_writer(self.out_dir, "train", settings["shard_size"], self.resume_from["train"])
            val = restored_writer(self.out_dir, "val", settings["val_shard_size"], self.resume_from["val"])
            return self.resume_from["documents_done"], train, val
        if has_checkpoints(self.out_dir):
            raise PrepRefused(f"{self.out_dir} holds an unfinished prep -- resume it (prepare_data.py does "
                              f"that on its own) or delete the folder")
        train = ShardWriter(self.out_dir, "train", settings["shard_size"])
        val = ShardWriter(self.out_dir, "val", settings["val_shard_size"])
        # a checkpoint of "nothing done yet": a kill before the first real one then still resumes,
        # instead of being refused for the shards already on disk
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.save(0, train, val)
        return 0, train, val

    def save(self, documents_done, train, val):
        number = train.shard_index
        checkpoint = {"settings": self.settings, "documents_done": documents_done, "stream": self.stream_position(),
                      "train": save_writer(train, number), "val": save_writer(val, number)}
        # the newest becomes the fallback -- unless it's damaged, or stale (one a resume just skipped):
        # then the good fallback already there stays
        newest = read_json(self.out_dir / NEWEST)
        if (newest is not None and checkpoint_files_complete(self.out_dir, newest)
                and newest["documents_done"] < documents_done):
            os.replace(self.out_dir / NEWEST, self.out_dir / PREVIOUS)
        write_json(self.out_dir / NEWEST, checkpoint)
        remove_unneeded_buffers(self.out_dir)


def save_writer(writer, number):
    # save a writer's half-filled buffer, and say where the writer is: everything needed to put it back
    # exactly here. Each checkpoint's buffer gets its own file, so the one a resume may need is never
    # overwritten.
    buffer_file = f"buffer_{writer.name}_{number:06d}.bin"      # never matches train_*.bin or val_*.bin
    temporary = writer.out_dir / (buffer_file + ".tmp")
    writer.buffer[: writer.filled].tofile(temporary)
    os.replace(temporary, writer.out_dir / buffer_file)
    return {"shard_index": writer.shard_index, "total_tokens": writer.total_tokens, "filled": writer.filled,
            "buffer_file": buffer_file}


def restored_writer(out_dir, name, shard_size, state):
    # a ShardWriter put back exactly where a checkpoint left it. (Any shard written after the checkpoint
    # gets written again, identically, as the prep goes on.)
    writer = ShardWriter(out_dir, name, shard_size, fresh=False)
    writer.shard_index = state["shard_index"]
    writer.total_tokens = state["total_tokens"]
    writer.filled = state["filled"]
    writer.buffer[: writer.filled] = np.fromfile(Path(out_dir) / state["buffer_file"], dtype=TOKEN_DTYPE)
    return writer


def has_checkpoints(out_dir):
    return (Path(out_dir) / NEWEST).exists() or (Path(out_dir) / PREVIOUS).exists()


def resume_point(out_dir, settings):
    # Where a prep in out_dir picks up: its newest checkpoint if all that checkpoint's files are there in
    # full, else the one before -- or None for a fresh start. Refuses a prep it can't continue: one whose
    # checkpoints are all incomplete, or one started with other settings.
    out_dir = Path(out_dir)
    if not has_checkpoints(out_dir):
        return None
    for checkpoint in (read_json(out_dir / NEWEST), read_json(out_dir / PREVIOUS)):
        if checkpoint is not None and checkpoint_files_complete(out_dir, checkpoint):
            if checkpoint["settings"] != settings:
                raise PrepRefused(f"{out_dir} holds an unfinished prep started with different settings "
                                  f"({checkpoint['settings']}) -- run it with the same options, or choose "
                                  f"another --out")
            return checkpoint
    raise PrepRefused(f"can't resume the prep in {out_dir}: its saved files are incomplete (a disconnect before "
                      f"they reached the disk or Google Drive?) -- delete the folder and start again")


def checkpoint_files_complete(out_dir, checkpoint):
    # every shard the checkpoint counts as written is there at full size, and so is each half-filled buffer
    bytes_per_token = np.dtype(TOKEN_DTYPE).itemsize
    try:
        for name, shard_size in (("train", checkpoint["settings"]["shard_size"]),
                                 ("val", checkpoint["settings"]["val_shard_size"])):
            state = checkpoint[name]
            for index in range(state["shard_index"]):
                shard = out_dir / f"{name}_{index:06d}.bin"
                if not shard.exists() or shard.stat().st_size != shard_size * bytes_per_token:
                    return False
            buffer_file = out_dir / state["buffer_file"]
            if not buffer_file.exists() or buffer_file.stat().st_size != state["filled"] * bytes_per_token:
                return False
    except (KeyError, TypeError):           # a checkpoint file that isn't shaped like one
        return False
    return True


def remove_unneeded_buffers(out_dir):
    # delete every half-filled buffer except those of the checkpoints a resume could still use
    still_needed = set()
    for checkpoint in (read_json(out_dir / NEWEST), read_json(out_dir / PREVIOUS)):
        if checkpoint is not None and checkpoint_files_complete(out_dir, checkpoint):
            still_needed |= {checkpoint["train"]["buffer_file"], checkpoint["val"]["buffer_file"]}
    for buffer_file in Path(out_dir).glob("buffer_*_[0-9][0-9][0-9][0-9][0-9][0-9].bin"):
        if buffer_file.name not in still_needed:
            buffer_file.unlink()


def remove_checkpoints(out_dir):
    # a finished, checked prep doesn't need its checkpoints any more -- nor any temporary file a kill left
    for checkpoint_file in (NEWEST, PREVIOUS):
        (Path(out_dir) / checkpoint_file).unlink(missing_ok=True)
    remove_unneeded_buffers(Path(out_dir))
    for temporary in Path(out_dir).glob("*.tmp"):
        temporary.unlink()
