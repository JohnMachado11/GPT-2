import contextlib
import json
import os
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

import torch

from gpt2.model.config import GPTConfig
from gpt2.model.gpt import GPT


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    if path.is_dir():
        best, last = path / "best.pt", path / "last.pt"
        if best.exists():
            path = best             # a run folder: load its best model --
        elif last.exists():
            path = last             # or, if no validation ever ran, its latest
        else:
            raise FileNotFoundError(f"{path} has no best.pt or last.pt yet -- nothing has been saved in this run")
    checkpoint = torch.load(path, map_location=device)
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def run_name(model, config):
    # architecture, size and a to-the-second timestamp: gpt2_L2_D128_6.8M_20261018-101500
    param_count = sum(p.numel() for p in model.parameters())
    params = f"{param_count / 1e6:.1f}M"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"gpt2_L{config.num_layers}_D{config.embed_dim}_{params}_{timestamp}"


def new_run_dir(model, config, checkpoints_dir):
    # One folder per training run, e.g. gpt2_L2_D128_6.8M_20261018-101500/
    # It never reuses an existing folder, so two runs can't overwrite each other.
    run_dir = Path(checkpoints_dir) / run_name(model, config)
    original = run_dir
    counter = 2
    while run_dir.exists():
        run_dir = original.with_name(f"{original.name}-{counter}")
        counter += 1
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    print(f"run folder -> {run_dir}")
    return run_dir


def save_atomic(state, path):
    # Write to a temporary file, then rename it over the real one. A rename is all-or-nothing,
    # so a crash or disconnect mid-save leaves the previous file intact instead of half-written.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)      # create the run folder if it doesn't exist yet
    temporary = path.with_name(path.name + ".tmp")
    try:
        with open(temporary, "wb") as file:
            torch.save(state, file)
            file.flush()
            os.fsync(file.fileno())     # force the bytes onto the disk BEFORE the rename: a power cut or
                                        # hard stop right after it then can't leave an empty file behind
                                        # (on Linux, e.g. the cloud; a Mac's fsync can leave them in the
                                        # drive's own cache a moment longer)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):  # (a failed delete mustn't hide the error that stopped the save)
            temporary.unlink(missing_ok=True)   # don't leave a half-written temp file (up to ~1.5 GB) behind
        raise
    try:                            # also flush the folder entry, so the rename itself survives a hard stop
        folder = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(folder)
        finally:
            os.close(folder)
    except OSError:
        pass                        # some filesystems can't flush a folder; the file itself is already safe


def ensure_finite(model, optimizer=None):
    # Never write a checkpoint holding NaN or inf -- in the weights, or in the optimizer's running averages
    # a resume would continue from. It would overwrite a good checkpoint with a broken one. Raising here
    # keeps the previous checkpoint on disk untouched.
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f"refusing to save: {name} contains NaN or inf -- the previous checkpoint is kept")
    if optimizer is not None:
        for averages in (optimizer.momentum, optimizer.bumpiness):
            if not all(torch.isfinite(tensor).all() for tensor in averages):
                raise RuntimeError("refusing to save: the optimizer's state contains NaN or inf -- "
                                   "the previous checkpoint is kept")


def save_best(run_dir, model, step, val_loss):
    # The lowest-validation-loss model so far -- the one you keep and query.
    # It holds config + model_state, the two keys load_checkpoint reads.
    ensure_finite(model)
    save_atomic(
        {"config": asdict(model.config), "model_state": model.state_dict(), "step": step, "val_loss": val_loss},
        Path(run_dir) / "best.pt",
    )


def save_last(run_dir, model, optimizer, step, train_batches, history, settings):
    # EVERYTHING needed to continue the run from this step.
    ensure_finite(model, optimizer)
    save_atomic(
        {
            "config": asdict(model.config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),       # momentum, bumpiness, step count, its settings
            "step": step,                                    # the last COMPLETED step
            "loader_state": train_batches.state(),           # where the data reader is
            "rng_state": torch.get_rng_state(),              # so random windows / dropout continue the same sequence
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,   # one GPU
            "mps_rng_state": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
            "history": history,                              # so the loss curve covers the whole run
            "settings": settings,                            # checked on resume: same schedule, or refuse
        },
        Path(run_dir) / "last.pt",
    )


def load_last(run_dir, model, optimizer, train_batches, settings):
    # Put a saved run back in place, and say where to pick up from. EVERY check runs before anything
    # changes, so a refused resume leaves the model, optimizer and data reader exactly as they were.
    state = torch.load(Path(run_dir) / "last.pt", map_location="cpu")

    if state["config"] != asdict(model.config):
        raise ValueError(f"can't resume: last.pt is a {state['config']} model, not {asdict(model.config)}")
    if state["settings"] != settings:
        raise ValueError(f"can't resume {run_dir}: it was trained with {state['settings']}, not {settings}")
    optimizer.check_state_dict(state["optimizer_state"])
    train_batches.check_state(state["loader_state"])
    best_val_loss_so_far(run_dir, model)                # refuses a best.pt that belongs to a different model
    cuda_state = state.get("cuda_rng_state")
    if isinstance(cuda_state, (list, tuple)):           # older checkpoints saved one state per GPU
        cuda_state = cuda_state[0] if cuda_state else None
    random_states = {"CPU": state.get("rng_state"), "NVIDIA GPU": cuda_state, "Mac GPU": state.get("mps_rng_state")}
    for device, random_state in random_states.items():
        if random_state is None and device != "CPU":
            continue                                    # that GPU wasn't in use when the run was saved
        if not (isinstance(random_state, torch.Tensor) and random_state.dtype == torch.uint8):
            raise ValueError(f"can't resume: last.pt's {device} random state is missing or damaged")

    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    train_batches.load_state(state["loader_state"])
    torch.set_rng_state(state["rng_state"])

    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_state)                # the NVIDIA GPU's generator: dropout masks on cuda
    if state.get("mps_rng_state") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps_rng_state"])         # the Mac GPU's generator: dropout masks on mps

    return state["step"], state["history"]


def best_val_loss_so_far(run_dir, model):
    # best.pt's own record of its validation loss (inf if there's no best.pt yet). best.pt can be NEWER
    # than last.pt -- a new best right before a disconnect -- so this, not last.pt's history, is what a
    # new best has to beat. Checked on every start, fresh or resumed.
    best_path = Path(run_dir) / "best.pt"
    if not best_path.exists():
        return float("inf")
    best = torch.load(best_path, map_location="cpu", mmap=True)     # mmap: reads the one number, not all the weights
    if best["config"] != asdict(model.config):
        raise ValueError(f"{best_path} belongs to a different model ({best['config']}) -- use a new run folder")
    return best["val_loss"]