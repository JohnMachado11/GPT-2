from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers import logging as transformers_logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gpt2.model.config import GPTConfig
from gpt2.model.gpt import GPT
from gpt2.train.optimizer import AdamW
from gpt2.train.loop import train
from gpt2.checkpoint import new_run_dir
from gpt2.train.data import RandomWindows
from gpt2.train.device import pick_device, supports_bf16
from gpt2.train.shards import text_shards


here = Path(__file__).parent
data_dir = here / "data"

# Fix every random draw: starting weights, batch windows, dropout, sampling.
# Same seed + same settings = same model.
seed = 1337
torch.manual_seed(seed)

transformers_logging.set_verbosity_error()   # hide the harmless ">1024 tokens" warning

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# GPT-2's one special token: <|endoftext|> (id 50256). It marks a document
# boundary -- an ending behind it, a beginning in front of it.
endoftext_id = tokenizer.eos_token_id

# Hold out the last 10% of EACH FILE for validation (the model never trains on it), so we can
# spot overfitting -- if train loss keeps falling while val loss rises, it's memorizing.
# Splitting per file (not once over everything) means every file shows up in BOTH sets --
# otherwise whichever file sorts last lands entirely (or mostly) in val and is barely trained on.
# 0.9 = 90% train / 10% val is a good ratio for a SMALL dataset like this. With a much
# bigger corpus you'd raise it -- the FineWeb run uses 0.99 (prepare_data.py).
train_fraction = 0.9

# EVERY .txt in the data folder is one document -- drop in as many files as you like, no code
# changes needed. The text is tokenized ONCE into shard files (the same train_*.bin / val_*.bin
# format the FineWeb run uses) in src/run/shards/text/. Later runs reuse them; adding, editing or
# removing a .txt file, or changing train_fraction, rebuilds them. info.json records what they
# were built from.
train_data, val_data = text_shards(data_dir, tokenizer, here / "shards" / "text", train_fraction)
print(f"\nsplit -> {len(train_data)} train tokens, {len(val_data)} val tokens\n")

# Show where the documents meet: the tail of one document with <|endoftext|>
# inline, then the first few words of the next one.
boundaries = (train_data == endoftext_id).nonzero().flatten().tolist()
print(f"{len(boundaries)} <|endoftext|> in the train stream, at {boundaries}")
for position in boundaries:
    ending = tokenizer.decode(train_data[max(0, position - 8):position + 1].tolist())
    next_ids = train_data[position + 1:position + 5].tolist()
    print(f"\n  position {position}")
    print(f"    {' '.join(ending.split())}")
    if next_ids:
        print(f"    next doc: {' '.join(tokenizer.decode(next_ids).split())}")
    else:
        print("    next doc: (end of stream)")
print()

config = GPTConfig(
    vocab_size=tokenizer.vocab_size,        # 50257 (GPT-2 BPE)
    context_length=64,                      # tokens per window
    embed_dim=128,                          # model size: grow these three for more capacity
    num_layers=2,
    num_heads=2,
    dropout=0.1,                            # GPT-2's value; regularizes against overfitting
)

model = GPT(config)

device = pick_device()                  # an NVIDIA GPU, else a Mac's GPU, else the CPU
use_bf16 = supports_bf16(device)        # only on NVIDIA GPUs that do bf16 in hardware (not free Colab's T4)

if device == "cuda":
    torch.set_float32_matmul_precision("high")    # let NVIDIA GPUs use TF32 for 32-bit matrix math

print(f"device -> {device} | bf16 {'on' if use_bf16 else 'off'}\n")

model.to(device)

optimizer = AdamW(model.parameters(), weight_decay=0.1)   # GPT-3 paper's value; nanoGPT's GPT-2 retrain

train_batches = RandomWindows(train_data, batch_size=32, context_length=config.context_length, device=device)
# validation gets its own seeded generator: every check then scores the SAME windows, so the numbers
# are comparable (and best.pt isn't picked on noise), and checking never changes what training draws
val_batches = RandomWindows(val_data, batch_size=32, context_length=config.context_length, device=device, seed=0)

# Where this run's checkpoints go. To continue an interrupted run, put its folder name here
# instead of None (e.g. "gpt2_L2_D128_6.8M_20261018-101500") -- train() picks up from its last.pt.
resume_run = None
checkpoints_dir = here.parent / "checkpoints"
if resume_run is None:
    run_dir = new_run_dir(model, config, checkpoints_dir)
else:
    run_dir = checkpoints_dir / resume_run
    if not (run_dir / "last.pt").exists():
        raise SystemExit(f"can't resume: {run_dir} has no last.pt")

history = train(
    model,
    optimizer,
    train_batches,
    num_steps=415,
    max_learning_rate=1e-3,
    warmup_steps=50,
    min_learning_rate=1e-4,
    max_gradient_norm=1.0,                  # GPT-3 paper's value; nanoGPT's GPT-2 retrain uses it
    val_batches=val_batches,
    device=device,
    use_bf16=use_bf16,
    run_dir=run_dir,
    checkpoint_interval=100,                # save last.pt every 100 steps (and always at the end)
)

plt.plot(history["steps"], history["train_loss"], label="train")
plt.plot(history["val_steps"], history["val_loss"], label="val", marker="o")
plt.xlabel("step")
plt.ylabel("loss")
plt.title("training vs validation loss")
plt.legend()
plot_path = run_dir / "loss.png"
plt.savefig(plot_path)
print(f"saved loss curve -> {plot_path}")

model.eval()
prompt = "The Martians"
prompt_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
output = model.generate(prompt_ids, max_new_tokens=40, top_k=20)
print("\n--- sample after training ---")
print(tokenizer.decode(output[0]))