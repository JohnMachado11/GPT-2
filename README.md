# gpt2 — a readable GPT-2 you can train from scratch

A small, from-scratch implementation of **GPT-2** written for clarity: every component is a hand-written
module with a plain, spelled-out name (`token_embedding`, `context_length`, `learning_rate` — no `wte`,
`ctx`, `lr`). Think *nanoGPT, but every weight and every step is named and easy to follow* — plus a
**verified bridge to the real Hugging Face GPT-2 weights**.

It's meant to be a **template / base** for three things:

1. **Train a GPT-2 from scratch** on your own text, quickly, on a laptop.
2. **Instantiate the real pretrained GPT-2** (from Hugging Face) inside this readable model — the
   forward pass matches Hugging Face's to **~1e-13** (float64), so you can trust it and explore it.
3. **A foundation for experiments** — mechanistic interpretability, activation patching, fine-tuning,
   and other LLM / neural-network research. Every weight is a named `nn.Parameter` and the forward
   pass is plain PyTorch you can hook into.

---

## What's inside

```
src/gpt2/
  model/         config, layernorm, attention, feedforward, block, gpt  (the architecture)
  train/         optimizer (AdamW), schedule (warmup+cosine), loop, data, shards (the trainer)
  from_hf.py     load the real GPT-2 weights into this model            (the HF bridge)
  checkpoint.py  run folders: save, load, resume                        (checkpointing)
src/run/         train_on_text.py, count_tokens.py, scaling_config.py   (train + size the model)
                 prepare_data.py                                        (big datasets -> token shards)
                 + data/*.txt                                           (your training text)
src/checkpoints/ one folder per training run + a README                 (git-ignored)
src/queries/     query_gpt2.py (real GPT-2) · query_trained.py (yours)  (prompt a model)
src/playgrounds/ small scripts to poke the core layers                  (learning aids)
src/tests/       the test suite                                         (pytest)
```

Everything is hand-written PyTorch: token + position embeddings → pre-norm transformer blocks (causal
multi-head attention + feed-forward) → final LayerNorm → tied unembedding. GPT-2's tanh-GELU and weight
initialization are matched so the pretrained weights load cleanly.

---

## Where the learnable weights live

Every learnable weight in the model, and the file to open for each. `[i]` = block index, `0`–`11`
(there are 12 identical blocks). This is exactly the set of tensors `loss.backward()` fills and the
optimizer updates.

| Weight | Reach it in code | Defined in | What it does |
|---|---|---|---|
| Token embedding | `model.token_embedding.weight` | `model/gpt.py` | maps each token id → a 768-vector (50,257 × 768) |
| Position embedding | `model.position_embedding.weight` | `model/gpt.py` | adds "where in the sequence" (1,024 × 768) |
| Attention Q/K/V | `model.blocks[i].attn.qkv` | `model/attention.py` | makes query, key, value (768 → 2,304) |
| Attention output | `model.blocks[i].attn.out_proj` | `model/attention.py` | recombines the heads (768 → 768) |
| Feed-forward — expand | `model.blocks[i].ff.expand` | `model/feedforward.py` | widen 4× (768 → 3,072) |
| Feed-forward — contract | `model.blocks[i].ff.contract` | `model/feedforward.py` | narrow back (3,072 → 768) |
| LayerNorms (2 per block) | `model.blocks[i].norm_before_attn`, `.norm_before_ff` | `model/layernorm.py` | scale + shift to keep the stream in range (768 each) |
| Final LayerNorm | `model.final_norm` | `model/layernorm.py` | one last scale + shift before the unembedding |
| Unembedding | `model.unembed.weight` | `model/gpt.py` | stream → vocab scores — **tied to the token embedding** (0 new params) |

`model/block.py` doesn't own any weights of its own — it just *wires* the attention, feed-forward, and
LayerNorm pieces above into one block. By parameter count, the 12 blocks are ~68% of the model and the
token embedding ~31%.

### How backprop passes through them

Forward runs bottom-up to the loss; **`loss.backward()` runs the exact reverse**, and the gradient
touches every weight in the table on the way down (then AdamW, `train/optimizer.py`, nudges each one).

```
  loss                                                  ▲  loss.backward() starts here
  unembed → logits            gpt.py                    │  → token_embedding grad (tied: filled here too)
  final LayerNorm             layernorm.py              │  → final_norm scale/shift grad
  Block 11  ┐                 attention.py              │  → attn.qkv, attn.out_proj
  Block ..  │  ×12            feedforward.py            │  → ff.expand, ff.contract
  Block 0   ┘                 layernorm.py              │  → 2× LayerNorm      (per block)
  token + position embeddings gpt.py                    │  → embedding grads
                                                        ▲  gradient has reached the first layer
```

**The residual stream is the gradient highway:** because each block *adds* to the stream
(`x = x + sublayer(x)`), the `+` passes the gradient straight through, so it reaches Block 0 without
fading — that's why a deep transformer trains at all. And thanks to weight tying, the token embedding
collects gradient from **both ends** (as the output projection *and* the input lookup).

### Not every weight is treated the same

Every weight above gets a gradient, but **weight decay** (AdamW's steady pull toward zero, which keeps
weights from growing large and overfitting) is applied **only to the 2-D weight matrices** — the
embeddings and the attention / feed-forward weights. The **1-D** tensors — every `bias` and every
LayerNorm `scale` / `shift` — are **skipped**, because shrinking them toward zero just fights what
they're for: a LayerNorm scale of 0 would erase the signal it's meant to re-scale.

This is the standard split for GPT-style training — it's what nanoGPT's GPT-2 reproduction does — and
here it's a one-line rule in `train/optimizer.py`:

```python
self.decayed = [parameter.dim() >= 2 for parameter in self.parameters]
```

On the example model (embed 128, 2 layers) that's **6,834,304 decayed** parameters across 10 tensors
versus **3,584 skipped** across 18 — a tiny slice of the weights, deliberately left alone. The decay
strength is **0.1**, the `AdamW` default here: it's the value stated in the GPT-3 paper, and the one
nanoGPT uses when it retrains GPT-2 from scratch (the GPT-2 paper itself never published one).

---

## Setup

Requires Python 3.12. From the repo root:

```bash
# 1. create and populate a virtual environment
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 2. make the local `gpt2` package importable (no pip install of this repo needed).
#    This writes one .pth file into the venv that puts src/ on the import path.
.venv/bin/python -c "import sysconfig, os; open(os.path.join(sysconfig.get_paths()['purelib'], 'gpt2.pth'), 'w').write(os.path.abspath('src'))"

# 3. verify everything works
.venv/bin/python -m pytest src/tests/ -q      # -> all tests pass (the first run downloads GPT-2's
                                              #    tokenizer from Hugging Face, a few MB, once)
```

After step 2, `import gpt2` works for anything the venv's Python runs (tests, scripts, your own code).

---

## Use 1 — Train a GPT-2 from scratch on your own text

**The whole flow, in order:**

```bash
# 1. add text: put .txt files in src/run/data/   (The War of the Worlds is already there)

# 2. size the model to your text -- prints each setting AND where it goes in train_on_text.py
.venv/bin/python src/run/count_tokens.py

# 3. copy those values into src/run/train_on_text.py, then train
#    (interrupted? set resume_run in train_on_text.py to the run's folder and run this again)
.venv/bin/python src/run/train_on_text.py

# 4. talk to the model you just trained
.venv/bin/python src/queries/query_trained.py
```

Each step is explained below.

**Where the data goes:** the example reads **every `.txt` file in `src/run/data/`** and tokenizes
each one. To train on your own text, just **drop more `.txt` files into that folder** — a book,
scraped articles, your notes, a web page's text — no code changes needed. Each file is split
**90% train / 10% validation on its own**, then the train pieces and the validation pieces are joined,
so every file shows up in both sets (no matter how the files sort or how small one is).
Every piece ends with GPT-2's document-boundary token, `<|endoftext|>` (id 50256), so the model
learns where one text stops and the next begins instead of inventing a transition between files.

**Prerequisite — size the model to your data first.** Before training, run the token counter. It reads
every `.txt` in `src/run/data/`, prints how many tokens you have, and (based on
`src/run/scaling_config.py`) prints a **suggested config**, laid out the same way as the code in
`train_on_text.py`, so you can see exactly which line each value belongs on:

```bash
.venv/bin/python src/run/count_tokens.py
```

```
Suggested config  (mode: minimum_viable)
----------------------------------------------------
set these values in src/run/train_on_text.py:

  config = GPTConfig(
      ...
      context_length=64,
      embed_dim=128,
      num_layers=2,
      num_heads=2,
      dropout=0.1,
  )

  train_batches = RandomWindows(train_data, batch_size=32, ...)
  val_batches = RandomWindows(val_data, batch_size=32, ...)

  history = train(
      ...
      num_steps=415,
      ...
  )
```

`scaling_config.py` has one switch, `MODE`, with two philosophies:

- **`"minimum_viable"`** — the smallest *usable* model, a fixed size whatever your text: embed
  `minimum_embed_dim` (128), with one layer and one head per 64 of width (so 128 · 2 layers · 2 heads),
  over-trained a fixed number of epochs (`minimum_viable_epochs`). Best when your text is small: you get
  coherent samples without a needlessly large model.
- **`"chinchilla_exact"`** — the *compute-optimal* size for your token count (the Chinchilla rule of
  ~20 training tokens per parameter, trained about one epoch). It **grows with your data** and recovers
  real architectures — e.g. ~1.7B tokens suggests embed 768 · 12 layers · 12 heads, GPT-2 small's *shape*.
  That's the most efficient size for that much data, not how much data OpenAI trained it on: a model
  this size keeps improving on far more (reproductions of GPT-2 small use ~10B tokens).

Copy the suggested `context_length` / `embed_dim` / `num_layers` / `num_heads` / `dropout` into the
`GPTConfig`, the `batch_size` into the two `RandomWindows(...)` lines, and `num_steps` into the
`train(...)` call, then train.

Run the bundled end-to-end example:

```bash
.venv/bin/python src/run/train_on_text.py
```

You'll see training loss fall from ~10.8 (random guessing over GPT-2's 50k vocab), a **validation
loss** printed every 100 steps and at the last step (measured on the same held-out windows each time,
text the model never trains on — your overfitting check), checkpoints saved into a new run folder (`best.pt` and `last.pt`, see below), and finally a
generated sample that echoes the training text.

Adding data needs no code edits — just put `.txt` files in `src/run/data/`. To change the model
size, edit the `GPTConfig(...)` in `src/run/train_on_text.py` (just after the data is loaded):

```python
config = GPTConfig(
    vocab_size=tokenizer.vocab_size,        # 50257 (GPT-2 BPE)
    context_length=64,                      # tokens per window
    embed_dim=128,                          # model size: grow these three for more capacity
    num_layers=2,
    num_heads=2,
    dropout=0.1,                            # GPT-2's value; regularizes against overfitting
)
```

plus the `batch_size` in the two `RandomWindows(...)` lines that cut the text into batches, and the
`train(...)` call's settings: `num_steps`, `max_learning_rate`, `warmup_steps`, `min_learning_rate`,
`max_gradient_norm`, and `checkpoint_interval` (how often `last.pt` is saved). Two more settings aren't in
the call, so they use their defaults: `eval_interval` (validate every N steps; default 100) and
`accumulation_steps`, which adds up the gradients of several batches before each update, for an
effectively bigger batch than fits in memory (add e.g. `accumulation_steps=4,`; default 1 = off).
That's the whole recipe: **text files → tokenize → build model → `train(...)` →
save → `generate(...)`.**

**Every run gets its own folder** in `src/checkpoints/`, e.g. `gpt2_L2_D128_6.8M_20261018-122043/`
(architecture · param count · to-the-second timestamp), so runs never overwrite each other. Inside:
`best.pt` (the weights at the lowest validation loss), `last.pt` (everything needed to resume),
`config.json`, and `loss.png`. Nothing in that folder is committed to git -- the weights are large.

**Resuming an interrupted run:** set `resume_run` (just above the `train(...)` call in
`train_on_text.py`) to the run's folder name and run it again. It continues from the step after the
last save in `last.pt` -- saves happen every `checkpoint_interval` steps (100 here), so at most that many
steps are redone -- with the same data order, learning-rate schedule and optimizer state, so it finishes
as the same run. It refuses to resume if the model, the schedule or optimizer settings, or the training
text's length changed (the full list is in `src/checkpoints/README.md`). Set `resume_run` back to `None`
for a fresh run.

**Talk to the model you trained:**

```bash
.venv/bin/python src/queries/query_trained.py
```

It auto-loads the run folder saved to most recently -- its `best.pt`, or `last.pt` if the run had no
validation -- and generates from a prompt. A tiny model
trained on a little text will echo and remix that text — that's expected; train a bigger model on more
data for more coherent output.

---

## Use 2 — Load the real pretrained GPT-2

```python
from transformers import AutoTokenizer
from gpt2.from_hf import load_gpt2

model = load_gpt2("gpt2")                       # real GPT-2 weights in this readable model
tokenizer = AutoTokenizer.from_pretrained("gpt2")

ids = tokenizer.encode("The capital of France is", return_tensors="pt")
out = model.generate(ids, max_new_tokens=10, top_k=1)
print(tokenizer.decode(out[0]))
```

`src/queries/query_gpt2.py` is a ready-made script for this. The loaded model's logits match Hugging
Face's exactly (argmax at every position; ~1e-13 in float64), so it's the *same* model — just readable.

---

## Use 3 — A base for experiments

Because the model is plain, named PyTorch, it's easy to build on:

- Every weight is reachable by name, e.g. `model.blocks[3].attn.qkv.weight`.
- The forward pass is hand-written, so you can insert hooks to capture or swap intermediate
  activations (the residual stream, attention patterns, etc.).
- Load real GPT-2 (Use 2) as the substrate, or train a tiny model (Use 1) you fully control.

This makes it a natural starting point for mechanistic-interpretability tooling, activation patching,
fine-tuning experiments, and similar work — built in separate projects that import this one.

---

## Why it looks the way it does

Names are spelled out, comments explain the *why*, most pieces have a test, and the core layers have a
runnable playground.
The goal is a codebase you can read top-to-bottom and actually understand — and then trust, because it
reproduces the real GPT-2 to floating-point rounding.

---

## References

- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — Jay Alammar's visual
  walkthrough of the GPT-2 architecture. An excellent companion for the intuition behind the pieces
  built here.
