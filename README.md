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
  train/         optimizer (AdamW), schedule (warmup+cosine), loop, data (the trainer)
  from_hf.py     load the real GPT-2 weights into this model            (the HF bridge)
  checkpoint.py  save / load / auto-name trained models                 (checkpointing)
src/run/         train_on_text.py + data/*.txt                          (train → saves a checkpoint)
src/checkpoints/ your trained models + a how-to-read README             (.pt weights git-ignored)
src/queries/     query_gpt2.py (real GPT-2) · query_trained.py (yours)  (prompt a model)
src/playgrounds/ small scripts to poke each component                   (learning aids)
src/tests/       the test suite                                         (23 tests)
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
.venv/bin/python -m pytest src/tests/ -q      # -> 23 passed
```

After step 2, `import gpt2` works for anything the venv's Python runs (tests, scripts, your own code).

---

## Use 1 — Train a GPT-2 from scratch on your own text

**Where the data goes:** the example reads **every `.txt` file in `src/run/data/`** and tokenizes
them into one training stream. To train on your own text, just **drop more `.txt` files into that
folder** — a book, scraped articles, your notes, a web page's text — no code changes needed; they're
concatenated automatically.

Run the bundled end-to-end example:

```bash
.venv/bin/python src/run/train_on_text.py
```

You'll see training loss fall from ~10.8 (random guessing over GPT-2's 50k vocab), a **validation
loss** printed every 100 steps (measured on held-out text the model never trains on — your overfitting
check), a saved checkpoint, and finally a generated sample that echoes the training text.

Adding data needs no code edits — just put `.txt` files in `src/run/data/`. To change the model
size, edit the config at the top of `src/run/train_on_text.py`:

```python
config = GPTConfig(
    vocab_size=tokenizer.vocab_size,        # 50257 (GPT-2 BPE)
    context_length=64,                      # tokens per window
    embed_dim=128, num_layers=4, num_heads=4,  # model size — grow for more capacity
    dropout=0.0,
)
```

and the `train(...)` call's knobs (`batch_size`, `num_steps`, `max_learning_rate`, `warmup_steps`,
`min_learning_rate`). That's the whole recipe: **text files → tokenize → build model → `train(...)` →
save → `generate(...)`.**

**Every run saves the model.** Training writes a uniquely-named checkpoint to `src/checkpoints/` — e.g.
`gpt2_L4_D128_7.2M_20260914-115430.pt` (architecture · param count · to-the-second timestamp), so runs
never overwrite each other. `src/checkpoints/README.md` explains how to read the names. (The `.pt`
weight files are git-ignored — they're large.)

**Talk to the model you trained:**

```bash
.venv/bin/python src/queries/query_trained.py
```

It auto-loads your **newest** checkpoint and generates from a prompt. A tiny model trained on a little
text will echo and remix that text — that's expected; train a bigger model on more data for more
coherent output.

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

Names are spelled out, comments explain the *why*, and each piece has a runnable playground and a test.
The goal is a codebase you can read top-to-bottom and actually understand — and then trust, because it
reproduces the real GPT-2 to floating-point rounding.

---

## References

- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — Jay Alammar's visual
  walkthrough of the GPT-2 architecture. An excellent companion for the intuition behind the pieces
  built here.
