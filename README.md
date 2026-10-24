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

<details>
<summary><b>Exactly how this run was done on Colab, step by step</b></summary>

**What you need:** Colab Pro+ for the A100 (the whole job used about 115 of its 500 compute units), about 40 GB
free on Google Drive (more if you don't empty Drive's Trash: each checkpoint save sends the old copy there), and,
optionally, a free Hugging Face account for a read token (fewer download limits).

Make a new notebook, then **Runtime → Change runtime type → A100 GPU** (leave High-RAM off). Run each cell with its
▶ button, in order. Don't press "Run all".

**Cell 1 — setup.** The first cell of every session, and again after any restart or disconnect.

```python
from google.colab import drive
drive.mount("/content/drive")
!git clone -q https://github.com/JohnMachado11/GPT-2.git /content/GPT-2 || true
%cd /content/GPT-2
!git fetch -q && git reset -q --hard 20a81c9      # the exact code this run used (it also undoes step 2's edit)
!pip install -q transformers==5.14.1 matplotlib==3.11.1 datasets==5.1.0    # not torch: keep Colab's GPU build
!python -c "import sys, torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '|', torch.cuda.get_device_name() if torch.cuda.is_available() else 'no GPU')"
```

It should end with something like `python 3.13.15 | torch 2.11.0+cu130 | NVIDIA A100-SXM4-40GB`. A red pip
"dependency resolver" error about `cudf` and `pyarrow` is harmless. (Optional: run the tests on Colab's Python with
`!pip install -q pytest==9.1.1` and `!PYTHONPATH=src python -m pytest src/tests -q`: `296 passed, 2 skipped` --
the two skipped tests need a Mac GPU.)

**Step 1 — check the GPU** (about a minute). FlashAttention must really run, and `torch.compile` must give the same
gradients as the plain model (on a Mac GPU it doesn't).

```python
%%bash
PYTHONPATH=src python - <<'EOF'
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from gpt2.model.config import GPTConfig
from gpt2.model.gpt import GPT
torch.manual_seed(0)
model = GPT(GPTConfig(vocab_size=50257, context_length=1024, embed_dim=768, num_layers=12, num_heads=12),
            fast_attention=True).cuda()
x = torch.randint(0, 50257, (4, 1024), device="cuda")
with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autocast("cuda", dtype=torch.bfloat16):
    model(x, x)[1].backward()                     # errors if FlashAttention can't run
print("FlashAttention: OK")
def grads(forward):
    model.zero_grad()
    forward(x, x)[1].backward()
    return {name: p.grad.clone() for name, p in model.named_parameters()}
plain, compiled = grads(model), grads(torch.compile(model))
worst = max(plain, key=lambda n: ((compiled[n] - plain[n]).norm() / plain[n].norm()).item())
print(f"compiled vs plain, worst gradient difference: {worst} "
      f"{((compiled[worst] - plain[worst]).norm() / plain[worst].norm()).item():.1e}  (fine: below 1e-3)")
EOF
```

It printed `FlashAttention: OK` and a difference of `1.7e-06`.

**Step 2 — a smoke test** (about 20 minutes): a small prep and a short run, to see the speed and the memory, and
that a restart picks up where it stopped. First the small prep (100M tokens, about a minute):

```python
!PYTHONPATH=src python src/run/prepare_data.py --max-tokens 1e8 --shard-size 25000000 --out /content/fineweb-small
```

Then the short run, saving every 50 steps for this test:

```python
!sed -i 's/^checkpoint_interval = 1000/checkpoint_interval = 50/' src/run/settings_gpt2_small.py
!PYTHONPATH=src python src/run/train_gpt2_small.py --data /content/fineweb-small --run-dir /content/drive/MyDrive/gpt2-smoke
```

Once `step 110/188` prints, press Stop, then **Runtime → Restart session**, run Cell 1, and run this cell again.
It must print `resuming gpt2-smoke from step 100`, with the same losses at steps 100 and 110 as before. Let it
finish. Here it ran at about 176,000 tokens a second, using 14 GB of the GPU's 40 GB. Then delete the test run:
`!rm -rf /content/drive/MyDrive/gpt2-smoke`.

**Step 3 — prepare the data** (about 2 hours on the A100 machine's 12 CPU cores; 5–9 hours on a CPU-only runtime).
Optional first: save a Hugging Face read token in Colab's Secrets (the key icon) as `HF_TOKEN`.

```python
import os
from google.colab import userdata
try:
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    print("no HF_TOKEN secret -- downloading without one")
!PYTHONPATH=src python src/run/prepare_data.py --out /content/drive/MyDrive/gpt2-data
drive.flush_and_unmount()
```

It ended with `train: 9,854,302,149 tokens in 99 shard(s)` and `val: 99,687,195 tokens in 10 shard(s)`.
Interrupted? Run Cell 1 and this cell again: it carries on from the last finished shard.

**Step 4 — check that every shard reached Drive.** Run Cell 1, then:

```python
!PYTHONPATH=src python src/run/prepare_data.py --out /content/drive/MyDrive/gpt2-data
```

It must say `... is a finished prep, and every file is there in full -- nothing to do`.

**Step 5 — train** (15.6 hours):

```python
!rsync -a --info=progress2 /content/drive/MyDrive/gpt2-data/ /content/fineweb/
!PYTHONPATH=src python src/run/train_gpt2_small.py --data /content/fineweb --run-dir /content/drive/MyDrive/gpt2-run
drive.flush_and_unmount()
```

The first line copies the shards to the runtime's own disk (20 GB in under a minute): reading them off Drive
mid-run can stall. It prints `18,794 steps of 524,288 tokens`, starts at loss 11.0 and ends near 3.0, saving
`best.pt` and `last.pt` to `gpt2-run` on Drive as it goes. Disconnected? Reconnect to an A100, run Cell 1, then
this cell again: it prints `resuming gpt2-run from step …`. A "Saving failed" notice about the notebook is harmless:
the script writes the checkpoints itself.

**Step 6 — compare with OpenAI's** (27 minutes):

```python
!PYTHONPATH=src python src/run/compare_to_openai.py --data /content/fineweb --run-dir /content/drive/MyDrive/gpt2-run
```

On a new runtime, use `--data /content/drive/MyDrive/gpt2-data` instead. It printed `your model 3.0548`,
`OpenAI's GPT-2 small 3.2975`, `difference -0.2427`.

**Finally:** Runtime → **Disconnect and delete runtime** (an idle A100 still uses units), download
`gpt2-run/best.pt` (0.5 GB, the model) from Google Drive, and empty Drive's Trash.

</details>

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
