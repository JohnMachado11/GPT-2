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
from run import settings        # every setting is in src/run/settings.py -- change them there


here = Path(__file__).parent
data_dir = here / "data"

torch.manual_seed(settings.seed)

transformers_logging.set_verbosity_error()   # hide the harmless ">1024 tokens" warning

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# GPT-2's one special token: <|endoftext|> (id 50256). It marks a document
# boundary -- an ending behind it, a beginning in front of it.
endoftext_id = tokenizer.eos_token_id

# EVERY .txt in the data folder is one document -- drop in as many files as you like, no code
# changes needed. The text is tokenized ONCE into shard files (the same train_*.bin / val_*.bin
# format the FineWeb run uses) in src/run/shards/text/. Later runs reuse them; adding, editing or
# removing a .txt file, or changing train_fraction, rebuilds them. info.json records what they
# were built from.
# Each file is split into train and val on its own (settings.train_fraction), not once over everything,
# so every file shows up in BOTH sets -- otherwise whichever file sorts last lands entirely (or mostly) in
# val and is barely trained on.
train_data, val_data = text_shards(data_dir, tokenizer, here / "shards" / "text", settings.train_fraction)
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
    context_length=settings.context_length,
    embed_dim=settings.embed_dim,
    num_layers=settings.num_layers,
    num_heads=settings.num_heads,
    dropout=settings.dropout,
)

model = GPT(config, fast_attention=settings.fast_attention)

device = pick_device()                  # an NVIDIA GPU, else a Mac's GPU, else the CPU
use_bf16 = supports_bf16(device)        # only on NVIDIA GPUs that do bf16 in hardware (not free Colab's T4)

if device == "cuda":
    torch.set_float32_matmul_precision("high")    # let NVIDIA GPUs use TF32 for 32-bit matrix math

print(f"device -> {device} | bf16 {'on' if use_bf16 else 'off'} | "
      f"fast attention {'on' if settings.fast_attention else 'off'} | "
      f"compile {'on' if settings.compile_model else 'off'}\n")

model.to(device)

optimizer = AdamW(
    model.parameters(),
    momentum_decay=settings.momentum_decay,
    bumpiness_decay=settings.bumpiness_decay,
    weight_decay=settings.weight_decay,
)

train_batches = RandomWindows(train_data, batch_size=settings.batch_size, context_length=config.context_length,
                              device=device)
# validation gets its own seeded generator: every check then scores the SAME windows, so the numbers
# are comparable (and best.pt isn't picked on noise), and checking never changes what training draws
val_batches = RandomWindows(val_data, batch_size=settings.batch_size, context_length=config.context_length,
                            device=device, seed=0)

# Where this run's checkpoints go: a new run folder, or the one settings.resume_run names
checkpoints_dir = here.parent / "checkpoints"
if settings.resume_run is None:
    run_dir = new_run_dir(model, config, checkpoints_dir)
else:
    run_dir = checkpoints_dir / settings.resume_run
    if not (run_dir / "last.pt").exists():
        raise SystemExit(f"can't resume: {run_dir} has no last.pt")

history = train(
    model,
    optimizer,
    train_batches,
    num_steps=settings.num_steps,
    max_learning_rate=settings.max_learning_rate,
    warmup_steps=settings.warmup_steps,
    min_learning_rate=settings.min_learning_rate,
    max_gradient_norm=settings.max_gradient_norm,
    accumulation_steps=settings.accumulation_steps,
    val_batches=val_batches,
    eval_interval=settings.eval_interval,
    device=device,
    use_bf16=use_bf16,
    run_dir=run_dir,
    checkpoint_interval=settings.checkpoint_interval,
    print_interval=settings.print_interval,
    compile_model=settings.compile_model,
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
prompt_ids = torch.tensor(tokenizer.encode(settings.prompt), device=device).unsqueeze(0)
output = model.generate(prompt_ids, max_new_tokens=settings.max_new_tokens, top_k=settings.top_k)
print("\n--- sample after training ---")
print(tokenizer.decode(output[0]))
