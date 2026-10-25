# gpt2

A readable GPT-2 built from scratch in PyTorch. Train it on your own text, or load OpenAI's weights into it
and get the same outputs as the original.

Every part is a hand-written module with a plain, spelled-out name (`token_embedding`, `context_length`,
`learning_rate` — no `wte`, `ctx`, `lr`). Think *nanoGPT, but every weight and every step is named and easy
to follow*.

**This code trained GPT-2 small from scratch, and it beats OpenAI's:** validation loss **3.05 vs 3.30** on the same
99.6M held-out tokens of FineWeb-Edu, after 15.6 hours on one Colab A100. [The numbers ↓](#train-gpt-2-small-from-scratch) ·
[The model on Hugging Face](https://huggingface.co/John-Machado/gpt2-small-fineweb-edu)

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
# the first run downloads GPT-2's tokenizer (a few MB)
.venv/bin/python -m pytest src/tests -q
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

**The result**, scored on 99.6 million tokens of FineWeb-Edu that neither model saw in training:

| Model | Trained on | Validation loss (lower is better) |
|---|---|---|
| **GPT-2 small, trained with this repo** | FineWeb-Edu, 9.85B tokens | **3.05** |
| OpenAI's GPT-2 small (2019) | WebText, OpenAI's own web data | 3.30 |

Both models score exactly the same text. FineWeb-Edu is the kind of text this model trained on, so it has a home
advantage here.

**Download it:** [John-Machado/gpt2-small-fineweb-edu](https://huggingface.co/John-Machado/gpt2-small-fineweb-edu)
on Hugging Face. It loads into this repo's readable model just like OpenAI's:

```python
from gpt2.from_hf import load_gpt2
model = load_gpt2("John-Machado/gpt2-small-fineweb-edu")
```

**To train it yourself:** all the training settings -- model size, learning rate, batch size and so on -- are in
`src/run/settings_gpt2_small.py`, already set to this run's values, so there's nothing to change. The steps below
are for Google Colab. On your own machine with an NVIDIA GPU, run the same commands without the leading `!` and
`PYTHONPATH=src` (after the Setup above, they aren't needed), and with your own folders instead of `/content/...`.

<details>
<summary><b>Train it on Google Colab, step by step</b></summary>

This is exactly how this repo's GPT-2 small was trained: a 124M-parameter model, from random weights, on 9.85 billion
tokens of FineWeb-Edu -- ending at validation loss 3.05, vs 3.30 for OpenAI's GPT-2 small on the same text.

#### Before you start

**What it costs**

| | | |
|---|---|---|
| **Colab Pro+** | about $50 a month | for the A100 GPU. This run used about **115 of the 500 compute units** it includes (about $12 worth) |
| **Google Drive space** | at least 40 GB free | the free 15 GB isn't enough. With 100 GB+ free you never have to empty Drive's Trash during the run |
| **Hugging Face account** | free, optional | a read token avoids download limits in step 4 |

**How long it takes:** about **19 hours**, but only about 20 minutes of it is you doing anything. The rest is waiting.

**Six things to know about Colab first**

1. **Run the cells one at a time** with their ▶ button. Never press "Run all".
2. **Two kinds of disk.** Files under `/content/drive/MyDrive/...` are on your **Google Drive**: they stay forever.
   Everything else under `/content/...` is on the **Colab machine**: it disappears when the machine is gone.
3. **"Restart session"** restarts Python but keeps the machine and its files. **"Disconnect and delete runtime"**
   throws the machine away (your Drive is untouched).
4. **One GPU per notebook.** Use an **A100** (an H100 works too, and is faster). Don't use a T4 or L4: too slow,
   and the batch may not fit.
5. **"Saving failed" at the top** is about saving the notebook page itself, not your model. Ignore it.
6. **Keep the tab open and your computer awake** during the long steps (on a Mac: run `caffeinate -d` in Terminal).

**The folders this guide uses**

| folder | on | what's in it |
|---|---|---|
| `/content/GPT-2` | the Colab machine | the code |
| `/content/fineweb-small` | the Colab machine | a small slice of data, for the practice run |
| `/content/drive/MyDrive/gpt2-smoke` | Drive | the practice run (deleted at the end of step 3) |
| `/content/drive/MyDrive/gpt2-data` | Drive | the full training data, 20 GB |
| `/content/fineweb` | the Colab machine | a copy of the training data, for speed |
| `/content/drive/MyDrive/gpt2-run` | Drive | **the trained model** |

**Retraining later?** You can reuse `gpt2-data` (step 4 just checks it and finishes in a minute). But give the new
model a **new folder name**, like `gpt2-run-2`, in steps 6 and 7: the training script continues whatever is already in
its folder, so with the old one it would just say `already finished all 18794 steps`.

---

#### Step 1 -- Setup *(about 2 minutes)*

**Make a new notebook**, then **Runtime → Change runtime type → A100 GPU → Save** (leave High-RAM off).

**This step** connects your Google Drive, downloads the code, installs three libraries, and shows what you're running
on. **Run it first in every session**, and again whenever a later step says so.

```python
# connect your Google Drive (where the data and the model will be kept)
from google.colab import drive
drive.mount("/content/drive")

# download the code onto the Colab machine ("|| true": fine if it's already there)
!git clone -q https://github.com/JohnMachado11/GPT-2.git /content/GPT-2 || true
%cd /content/GPT-2

# get the newest version of the code from GitHub
!git pull -q

# install the three libraries the code needs (not torch: Colab already has the GPU version)
!pip install -q transformers==5.14.1 matplotlib==3.11.1 datasets==5.1.0

# show the Python version, the torch version and the GPU
!python --version
!python -c "import torch; print(torch.__version__, torch.cuda.get_device_name())"
```

When Colab asks for access to your Google Drive, allow it.

**You should see:** `Python 3.13...`, then a torch version and `NVIDIA A100-SXM4-40GB`.
- A red pip error about `cudf` and `pyarrow` is harmless.
- An error about no GPU means the runtime isn't a GPU one: **Runtime → Change runtime type**.

*Optional: run the tests (2 minutes).* In a new cell:

```python
!pip install -q pytest==9.1.1
!PYTHONPATH=src python -m pytest src/tests -q
```

**You should see:** everything `passed`, and 2 `skipped` (those two need a Mac GPU).

---

#### Step 2 -- Check the GPU *(about 1 minute)*

**This step** makes sure the two speed features work on this GPU before you spend hours on it:
- **FlashAttention**, the fast attention, must really be used;
- **`torch.compile`** must give the same results as the plain model. (On a Mac GPU it gives wrong ones.)

**Put this in its own cell.** `%%bash` only works as the cell's very first line.

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

# FlashAttention must really run (this errors if it can't)
with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autocast("cuda", dtype=torch.bfloat16):
    model(x, x)[1].backward()
print("FlashAttention: OK")

# torch.compile must give the same gradients as the plain model
def grads(forward):
    model.zero_grad()
    forward(x, x)[1].backward()
    return {name: p.grad.clone() for name, p in model.named_parameters()}
plain, compiled = grads(model), grads(torch.compile(model))
worst = max(plain, key=lambda n: ((compiled[n] - plain[n]).norm() / plain[n].norm()).item())
print("compiled vs plain, worst difference:", ((compiled[worst] - plain[worst]).norm() / plain[worst].norm()).item())
EOF
```

**You should see:** `FlashAttention: OK`, and a difference below `0.001` (this run's: `1.7e-06`).
- A warning about TensorFloat32 is harmless.
- An error or a bigger difference: stop here, and don't train on this GPU.

---

#### Step 3 -- Practice run *(about 20 minutes)*

**This step** trains for a few minutes on a small slice of the data. In 20 minutes it shows how fast the GPU trains,
how much memory it uses, and that a stopped run picks up where it left off -- instead of finding a problem 10 hours
into the real run.

**3a. Get a small slice of the data** (100 million tokens, about a minute):

```python
!PYTHONPATH=src python src/run/prepare_data.py --max-tokens 1e8 --shard-size 25000000 --out /content/fineweb-small
```

**You should see:** `done -> /content/fineweb-small`, with about 99 million train tokens.

**3b. Practice training.** For this test only, the first line makes it save a checkpoint every 50 steps instead of
every 1000, so step 3c has something to resume from (3d puts it back):

```python
!sed -i 's/^checkpoint_interval = 1000/checkpoint_interval = 50/' src/run/settings_gpt2_small.py
!PYTHONPATH=src python src/run/train_gpt2_small.py --data /content/fineweb-small --run-dir /content/drive/MyDrive/gpt2-smoke
```

**You should see:**
- `device -> cuda | bf16 on | fast attention on | compile on`, then `... 188 steps of 524,288 tokens`;
- a warning that warmup is longer than the run (normal for this short test);
- about a minute of silence (compiling), then `step 0/188 | loss 11.0...`, then a line every 10 steps;
- once it's going, about **176,000 tokens/s**, and about **14 GB** of GPU RAM in Colab's **Resources** panel (top right).

**3c. Test that it resumes:**

1. When `step 110/188` appears, press **Stop** (the ■ next to the cell).
2. **Runtime → Restart session.** Run **step 1**, then the **3b cell** again.
3. **You should see:** `resuming gpt2-smoke from step 100`, with the same loss at steps 100 and 110 as before.
   Let it finish (step 187), which prints a short sample of text.

**3d. Clean up:** delete the practice run, and put the setting back from 50 to 1000 for the real training:

```python
# delete the practice run
!rm -rf /content/drive/MyDrive/gpt2-smoke

# put the setting back from 50 to 1000, and print it to check
!sed -i 's/^checkpoint_interval = 50/checkpoint_interval = 1000/' src/run/settings_gpt2_small.py
!grep "^checkpoint_interval" src/run/settings_gpt2_small.py
```

**You should see:** `checkpoint_interval = 1000`. **Don't go on if it says 50**: the real run would then save a
1.5 GB file every 50 steps.

---

#### Step 4 -- Prepare the data *(about 2 hours)*

**This step** downloads FineWeb-Edu's 10-billion-token sample, turns the text into GPT-2's tokens, and saves them on
your Drive as 109 files (20 GB): 99 for training and 10 held back for scoring. It's done once; every later run reuses it.

**Optional first, a Hugging Face token:** in Colab's left sidebar, click the **key icon** (Secrets) → **Add new secret**:
name `HF_TOKEN`, value your Hugging Face read token, and turn on **Notebook access**.

```python
# use your Hugging Face token if you saved one (fewer download limits)
import os
from google.colab import userdata
try:
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    print("no HF_TOKEN secret -- downloading without one")

# download and tokenize FineWeb-Edu into 20 GB of files on your Drive
!PYTHONPATH=src python src/run/prepare_data.py --out /content/drive/MyDrive/gpt2-data

# make sure every file has finished uploading to Drive
drive.flush_and_unmount()
```

**You should see:** a line per 100 million tokens (`wrote train_000000.bin: 100,000,000 tokens`, ...), about 100 of
them, ending with `train: 9,854,302,149 tokens in 99 shard(s)` and `val: 99,687,195 tokens in 10 shard(s)`.

**Stopped partway?** Run step 1, then this cell again. It picks up from the last finished file (it can be quiet for a
few minutes first, while it finds its place).

---

#### Step 5 -- Check the data reached Drive *(about 1 minute)*

**This step** checks that all 109 files are on your Drive, whole. (The last ones can still be uploading when a step
ends.) Run **step 1** first -- step 4 disconnected Drive at the end -- then:

```python
!PYTHONPATH=src python src/run/prepare_data.py --out /content/drive/MyDrive/gpt2-data
```

**You should see:** `... is a finished prep, and every file is there in full -- nothing to do`.
If it says a file is missing, it redoes the end by itself; run it once more afterwards.

---

#### Step 6 -- Train *(about 15.6 hours)*

**This step** is the real training: one pass over all 9.85 billion training tokens, 18,794 steps. It saves the model
to your Drive as it goes, so a disconnect loses at most about 50 minutes.

```python
# copy the data from Drive to the machine's own disk (20 GB, about a minute): reading off Drive mid-run can stall
!rsync -a --info=progress2 /content/drive/MyDrive/gpt2-data/ /content/fineweb/

# train, saving into gpt2-run on your Drive (for a later retrain, use a new folder name here and in step 7)
!PYTHONPATH=src python src/run/train_gpt2_small.py --data /content/fineweb --run-dir /content/drive/MyDrive/gpt2-run

# make sure the last saves have finished uploading to Drive
drive.flush_and_unmount()
```

**You should see:** `124M parameters | 9,854,302,149 train tokens ... | 18,794 steps of 524,288 tokens`, about a minute
of compiling, then a line every 10 steps. The loss falls from about 11.0 to about 3.0, and "left" settles around 15.5 h.

**While it trains:**

- **Leave the tab open, and don't refresh it.** The page may say "Saving failed", or even "disconnected", while the
  training keeps running on the GPU and saving to Drive. Refreshing can lose the live log.
- **To check on it**, open Colab's **Terminal** (bottom left) and run:

  ```bash
  # the GPU near 99% busy, with a python3 process: it's still training
  nvidia-smi

  # best.pt and last.pt should have recent times (Terminal times are UTC)
  ls -la /content/drive/MyDrive/gpt2-run
  ```

  `best.pt` (the model, 0.5 GB) is replaced at each new lowest validation loss, checked every 250 steps.
  `last.pt` (everything needed to resume, 1.5 GB) is replaced every 1000 steps, about every 50 minutes.
- **Drive's Trash fills up:** each replaced file goes there -- about 65 GB over the whole run. With less than ~100 GB
  free, empty it every few hours (drive.google.com → Trash → Empty trash).
- **If it really stopped** (no `python3` in `nvidia-smi`, or the runtime is gone): reconnect to an A100, run step 1,
  then this cell again. **You should see** `resuming gpt2-run from step ...`.

**When it's done:** the last lines show the final validation loss (this run: `3.0113`), `saved loss curve`, and a short
sample of text.

---

#### Step 7 -- Compare with OpenAI's GPT-2 *(about 27 minutes)*

**This step** scores your model and OpenAI's GPT-2 small on the same 99.6 million held-back tokens. The first time,
it downloads OpenAI's model (about 550 MB). Run **step 1** first -- step 6 disconnected Drive -- then:

```python
!PYTHONPATH=src python src/run/compare_to_openai.py --data /content/fineweb --run-dir /content/drive/MyDrive/gpt2-run
```

On a new Colab machine, `/content/fineweb` is gone: use `--data /content/drive/MyDrive/gpt2-data` instead.

**You should see** (this run's numbers -- lower is better):

```
scoring both on the same 99,631,104 validation tokens

  your model            3.0548
  OpenAI's GPT-2 small  3.2975
  difference           -0.2427   (below zero: yours is better)
```

---

#### Step 8 -- Finish

1. **Runtime → Disconnect and delete runtime.** An idle A100 still uses compute units.
2. From drive.google.com, **download `gpt2-run/best.pt`** (0.5 GB) -- the model. (`last.pt` is only needed to keep
   training; `loss.png` is the loss curve.)
3. **Empty Drive's Trash.** Delete `gpt2-data` (20 GB) too, unless you'll train again.

**To use the model on your computer:** put `best.pt` in a folder like `src/checkpoints/gpt2-run/`, then run
`.venv/bin/python src/queries/query_trained.py`. It loads the newest run folder; its prompt is set in
`src/run/settings.py`.

---

#### Step 9 -- Share it on Hugging Face *(optional, on your computer)*

1. **Convert it** to Hugging Face's format (it checks the outputs match before saving):
   ```bash
   .venv/bin/python src/run/export_to_hf.py --run-dir src/checkpoints/gpt2-run
   ```
2. **Make a Hugging Face token that can write:** huggingface.co → Settings → Access Tokens → Create new token → **Write**.
3. **Log in**, and paste the token when asked:
   ```bash
   .venv/bin/hf auth login
   ```
4. **Upload** (this creates a public model page):
   ```bash
   .venv/bin/hf upload <your-name>/<model-name> src/checkpoints/gpt2-run/hf .
   ```

---

#### If something goes wrong

| you see | what it means | what to do |
|---|---|---|
| "Saving failed" at the top | the notebook page can't save -- not your model | nothing |
| "Reconnect" / disconnected, mid-training | maybe only the page lost contact | check with `nvidia-smi` in the Terminal (step 6) |
| `already finished all 18794 steps` | that run folder already holds a finished run | use a new `--run-dir` folder name |
| `can't resume: ...` | the run folder was made with different settings | use a new `--run-dir`, or put the settings back |
| `training diverged` | the loss blew up; it stopped before damaging the saved model | don't rerun as-is (it would hit the same spot) -- start a new run folder with a lower `max_learning_rate` in `src/run/settings_gpt2_small.py` |
| `CUDA out of memory` | the batch doesn't fit this GPU | for a new run: in `settings_gpt2_small.py`, halve `batch_size` and double `accumulation_steps` |
| only T4 or L4 offered | no A100 free right now | try again later, or pick an H100 |
| prep stopped with an error | usually a download hiccup | run step 1 and step 4 again -- it resumes |

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
