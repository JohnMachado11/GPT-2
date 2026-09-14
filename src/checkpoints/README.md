# checkpoints/

Trained models are saved here. Each training run (`src/run/train_on_text.py`) writes a **new,
uniquely-named** `.pt` file — runs **never** overwrite each other, so you keep every model you train.
(The name carries a to-the-second timestamp; in the rare case two names still collided, the saver adds
a `-2`, `-3`, … suffix rather than clobbering the existing file.)

Each run also drops a matching `..._loss.png` next to the checkpoint — the training-vs-validation loss
curve for that run.

> The `.pt` weight files **and the `_loss.png` plots** are **git-ignored** (weights are large; plots are
> regenerated each run). Only this README is committed, so the folder has a home in the repo. Your
> trained models and plots stay local.

## How to read a model name

```
gpt2_L4_D128_7.2M_20260914-115430.pt
│    │  │    │    │
│    │  │    │    └─ timestamp: 2026-09-14, 11:54:30  (YYYYMMDD-HHMMSS — sorts oldest→newest)
│    │  │    └────── parameter count: ~7.2 million weights
│    │  └─────────── D128 = embed_dim 128  (width of the residual stream)
│    └────────────── L4   = 4 transformer layers  (num_layers)
└─────────────────── gpt2 = the architecture
```

So at a glance you can tell **what shape** the model is (`L`ayers × `D`imension), **how big** it is
(param count), and **when** you trained it (timestamp).

- Two models trained on different days differ by the timestamp.
- Two models of different sizes differ by `L`, `D`, and the param count.
- Because the timestamp is `YYYYMMDD-HHMMSS`, plain alphabetical sorting puts the **newest last**.

## Loading one

`src/queries/query_trained.py` loads the **most recent** checkpoint automatically (by modification
time), so you don't need to type a filename. To load a specific one in your own code:

```python
from gpt2.checkpoint import load_checkpoint
model = load_checkpoint("src/checkpoints/gpt2_L4_D128_7.2M_20260914-115430.pt")
```
