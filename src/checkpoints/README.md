# checkpoints/

Every training run gets **its own folder** here. Runs never overwrite each other: the folder name carries
a to-the-second timestamp, and if two names ever collided the newer one gets a `-2`, `-3`, … suffix.

```
gpt2_L2_D128_6.8M_20261018-122043/
  config.json   the model's shape -- readable without loading anything
  best.pt       weights at the LOWEST validation loss so far  -> the model to keep and query
  last.pt       everything needed to continue the run         -> what a resume loads
  loss.png      the training-vs-validation loss curve, drawn at the end
```

**`last.pt`** holds the weights, the optimizer's state (momentum, bumpiness, step count), the training
step, the data reader's position, the random-number state, the loss history, and the training settings.
It's saved every `checkpoint_interval` steps and always at the end of the run.

**`best.pt`** holds the weights and the model's config, plus the step and validation loss they came
from. It's saved whenever validation loss hits a new low (validation runs every `eval_interval` steps and
at the final step), so a run that overfits at the end still keeps its best version. A run trained
without validation has no `best.pt`; its `last.pt` is used instead.

Every `best.pt` / `last.pt` save writes a temporary file first and then renames it into place, so a
crash or disconnect in the middle of a save leaves the previous checkpoint intact. And a save is
refused outright if the weights (or, for `last.pt`, the optimizer's running averages) contain NaN or
inf, so a broken run can never overwrite a good one. A save stopped by an error or Ctrl-C deletes its
temporary file; a hard kill (out of memory, a preempted cloud machine, a power cut) can leave a
`last.pt.tmp` behind, which is harmless -- nothing loads it, and the next save overwrites it.

The all-or-nothing rename needs the run folder on the machine's own disk. A network-mounted folder
(Google Drive, a cloud bucket) may not rename all-or-nothing, so train into a local folder and copy the
run folder off.

> **Nothing in this folder is committed** except this README (see `.gitignore`). Run folders get big:
> for GPT-2 small, `best.pt` is ~0.5 GB and `last.pt` ~1.5 GB (the optimizer keeps two extra numbers per
> weight). Your trained models stay on your machine.

## How to read a folder name

```
gpt2_L2_D128_6.8M_20261018-122043
│    │  │    │    │
│    │  │    │    └─ timestamp: 2026-10-18, 12:20:43  (YYYYMMDD-HHMMSS -- sorts oldest→newest)
│    │  │    └────── parameter count: ~6.8 million weights
│    │  └─────────── D128 = embed_dim 128  (width of the residual stream)
│    └────────────── L2   = 2 transformer layers  (num_layers)
└─────────────────── gpt2 = the architecture
```

So at a glance you can tell **what shape** the model is (`L`ayers × `D`imension), **how big** it is
(param count), and **when** you trained it (timestamp). Among runs of the same shape, alphabetical order
is oldest → newest, because the timestamp is `YYYYMMDD-HHMMSS`. (Different shapes group by shape first:
`gpt2_L12_…` sorts before `gpt2_L2_…` whatever the dates.)

## Resuming an interrupted run

In `src/run/settings.py`, set `resume_run` to the run's folder name, then run `src/run/train_on_text.py` again:

```python
resume_run = "gpt2_L2_D128_6.8M_20261018-122043"
```

`train()` loads that folder's `last.pt` and continues from the step after the last save -- saves happen
every `checkpoint_interval` steps, so at most that many steps are redone -- with the same data order,
learning-rate schedule and optimizer state, so it finishes as the same run.

It refuses to resume, before changing anything, if any of these differ from the saved run:

- the model's config (shape, dropout, …)
- the schedule settings: steps, learning rates, warmup, gradient cap, accumulation, batch size, context length
- the optimizer's settings: weight decay, the momentum and bumpiness decay rates, epsilon
- the data: for text held in memory, its token count; for shard files, the exact file the run stopped
  in, plus that file's token count and a fingerprint of its first and last 4,096 tokens (so a shard
  regenerated under the same name is caught, even at the same length)

It can't detect text of exactly the same length being swapped in, an edit only in the middle of a
shard, changes to the *other* shard files, a `seed=` argument added to or removed from the training
`RandomWindows`, or a change to the validation data, `eval_interval` or `use_bf16` -- so keep those the
same mid-run. Set `resume_run` back to `None` for a fresh run.

**If a run diverges** (it stops with "training diverged"), resuming won't get past it: the resume replays
the exact same batches, so it hits the same spike at the same step, and a lower learning rate is refused
as a settings change. Start a fresh run with a lower `max_learning_rate` (or more warmup) instead. The
checkpoint from before the spike is still good, so nothing on disk is lost.

## Loading a model

`src/queries/query_trained.py` loads the run folder saved to most recently (its `best.pt`, or `last.pt`
if the run had no validation) automatically. To load
a specific one in your own code, pass the folder:

```python
from gpt2.checkpoint import load_checkpoint
model = load_checkpoint("src/checkpoints/gpt2_L2_D128_6.8M_20261018-122043")   # its best.pt
```
