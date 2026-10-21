# gpt2

A readable GPT-2 built from scratch in PyTorch. Train it on your own text, or load OpenAI's weights into it
and get the same outputs as the original.

Every part is a hand-written module with a plain, spelled-out name (`token_embedding`, `context_length`,
`learning_rate` — no `wte`, `ctx`, `lr`). Think *nanoGPT, but every weight and every step is named and easy
to follow*.

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
| `src/run/` | the scripts you run, and `settings.py` |
| `src/queries/` | prompt OpenAI's GPT-2, or the model you trained |
| `src/checkpoints/` | one folder per training run |
| `src/playgrounds/` | small scripts to poke the core layers |
| `src/tests/` | the tests |

## How it works

Every part, step by step, with pictures: **gpt2.dev** (coming soon).

MIT License
