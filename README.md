# gpt2

A readable GPT-2 built from scratch in PyTorch. Train it on your own text, or load OpenAI's weights into it
and get the same outputs as the original.

Every part is a hand-written module with a plain, spelled-out name (`token_embedding`, `context_length`,
`learning_rate` — no `wte`, `ctx`, `lr`). Think *nanoGPT, but every weight and every step is named and easy
to follow*.

**This code trained GPT-2 small from scratch, and it beats OpenAI's:** validation loss **3.05 vs 3.30** on the same
99.6M held-out tokens of FineWeb-Edu, after 15.6 hours on one Colab A100. [The numbers ↓](#train-gpt-2-small-from-scratch)

- **Train GPT-2 small (124M) from scratch** on 10B tokens of FineWeb-Edu, on one cloud GPU, and score it against OpenAI's.
- **Train your own** on any `.txt` files. One book trains in about a minute on an M1 Pro.
- **Load OpenAI's GPT-2** (124M) into the same code. Its outputs match Hugging Face's.
- **Read every piece.** One small file per part, with tests.

## Setup

Needs Python 3.12. From the repo root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import sysconfig, os; open(os.path.join(sysconfig.get_paths()['purelib'], 'gpt2.pth'), 'w').write(os.path.abspath('src'))"
.venv/bin/python -m pytest src/tests -q      # the first run downloads GPT-2's tokenizer (a few MB)
```

## Run it

| Step | Command | What you'll see |
|---|---|---|
| **1. Set up** | the commands under *Setup*, above | all tests pass |
| **2. Add text** | put `.txt` files in `src/run/data/` | *The War of the Worlds* is already there |
| **3. Size the model** | `.venv/bin/python src/run/count_tokens.py` | four values to put in `src/run/settings.py` |
| **4. Train** | `.venv/bin/python src/run/train_on_text.py` | with the bundled book: val loss falling from 10.8 to about 5.9, in about a minute on an M1 Pro |
| **5. Talk to it** | `.venv/bin/python src/queries/query_trained.py` | a continuation that remixes your books' words |

Stopped partway? Put the run's folder name (from `src/checkpoints/`) in `resume_run` in `settings.py`, and run
step 4 again. Set it back to `None` for a new run.

## Train GPT-2 small from scratch

GPT-2 small's shape (124M parameters, 12 layers, 768 wide, context 1024), trained on one pass over FineWeb-Edu:

| | |
|---|---|
| **Data** | 9.85B tokens of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`), read once |
| **Hardware** | one NVIDIA A100 40GB, on Google Colab Pro+ |
| **Time** | about 2 h to prepare the data, then 15.6 h of training (~176k tokens/s); about 115 Colab compute units in all |

| Validation loss on the same 99.6M held-out FineWeb-Edu tokens (lower is better) | |
|---|---|
| **GPT-2 small, trained from scratch with this repo** | **3.05** |
| OpenAI's GPT-2 small | 3.30 |

Both models score the exact same windows of text. (OpenAI's model trained on other web text, so FineWeb-Edu is
home turf for this one.)

Three commands, with every setting in `src/run/settings_gpt2_small.py`:

```bash
.venv/bin/python src/run/prepare_data.py --out SHARDS                                # FineWeb-Edu -> 20 GB of token shards
.venv/bin/python src/run/train_gpt2_small.py --data SHARDS --run-dir RUN             # one pass; run it again to resume
.venv/bin/python src/run/compare_to_openai.py --data SHARDS --run-dir RUN            # yours vs OpenAI's, on the same text
```

On Colab (its Python 3.13 works): clone the repo, `pip install transformers==5.14.1 matplotlib==3.11.1
datasets==5.1.0` (keep Colab's own torch), run each command as `!PYTHONPATH=src python …`, and keep `RUN` on Google
Drive so the checkpoints outlive the runtime. Disconnected? Run the same command again: it resumes from the last save.

## Load OpenAI's GPT-2

The first run downloads OpenAI's weights from Hugging Face (about 550 MB).

```python
from transformers import AutoTokenizer
from gpt2.from_hf import load_gpt2

model = load_gpt2("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

ids = tokenizer.encode("France: Paris\nItaly: Rome\nSpain:", return_tensors="pt")
print(tokenizer.decode(model.generate(ids, max_new_tokens=1, top_k=1)[0]))   # ... Spain: Madrid
```

## Where things are

| Folder | What's in it |
|---|---|
| `src/gpt2/model/` | the model, one file per part: attention, feed-forward, LayerNorm, block, GPT |
| `src/gpt2/train/` | the optimizer, learning-rate schedule, training loop and data |
| `src/run/` | the scripts you run, and their settings (`settings.py`, `settings_gpt2_small.py`) |
| `src/queries/` | prompt OpenAI's GPT-2, or the model you trained |
| `src/checkpoints/` | one folder per training run: **`best.pt` is the model**; `last.pt` adds the optimizer's state, to resume training |
| `src/playgrounds/` | small scripts to poke the core layers |
| `src/tests/` | the tests |

## How it works

Every part, step by step, with pictures: **gpt2.dev** (coming soon).

MIT License
