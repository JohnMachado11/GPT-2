from pathlib import Path

import torch
from transformers import AutoTokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gpt2.model.config import GPTConfig
from gpt2.model.gpt import GPT
from gpt2.train.optimizer import AdamW
from gpt2.train.loop import train
from gpt2.checkpoint import save_checkpoint, checkpoint_name


here = Path(__file__).parent
data_dir = here / "data"

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Read EVERY .txt in the data folder and tokenize each one.
# Drop in as many files as you like -- no code changes needed.
files = sorted(data_dir.glob("*.txt"))
if not files:
    raise SystemExit(f"no .txt files found in {data_dir} -- add some text to train on")

# Hold out the last 10% of EACH FILE for validation (the model never trains on it), so we can
# spot overfitting -- if train loss keeps falling while val loss rises, it's memorizing.
# Splitting per file (not once over everything) means every file shows up in BOTH sets --
# otherwise whichever file sorts last lands entirely (or mostly) in val and is barely trained on.
# 0.9 = 90% train / 10% val is a good ratio for a SMALL dataset like this. With a much
# bigger corpus you'd raise it (e.g. 0.99) -- a tiny fraction is already plenty of val tokens.
train_parts = []
val_parts = []
for f in files:
    file_tokens = torch.tensor(tokenizer.encode(f.read_text()))
    split = int(0.9 * len(file_tokens))
    train_parts.append(file_tokens[:split])
    val_parts.append(file_tokens[split:])
    print(f"  - {f.name}: {len(file_tokens)} tokens -> {split} train / {len(file_tokens) - split} val")

# Join the per-file pieces: all train pieces into one stream, all val pieces into another.
train_data = torch.cat(train_parts)
val_data = torch.cat(val_parts)
print(f"split -> {len(train_data)} train tokens, {len(val_data)} val tokens")

config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    context_length=64,
    embed_dim=128,
    num_layers=2,
    num_heads=2,
    dropout=0.1,
)

model = GPT(config)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

optimizer = AdamW(model.parameters(), weight_decay=0.01)

history = train(
    model,
    optimizer,
    train_data,
    batch_size=32,
    context_length=config.context_length,
    num_steps=830,
    max_learning_rate=1e-3,
    warmup_steps=50,
    min_learning_rate=1e-4,
    max_gradient_norm=1.0,                  # GPT-3 paper's value; nanoGPT uses it for GPT-2 too
    val_data=val_data,
    device=device,
)

saved_path = save_checkpoint(model, config, here.parent / "checkpoints" / checkpoint_name(model, config))

plt.plot(history["steps"], history["train_loss"], label="train")
plt.plot(history["val_steps"], history["val_loss"], label="val", marker="o")
plt.xlabel("step")
plt.ylabel("loss")
plt.title("training vs validation loss")
plt.legend()
plot_path = saved_path.with_name(saved_path.stem + "_loss.png")
plt.savefig(plot_path)
print(f"saved loss curve -> {plot_path}")

model.eval()
prompt = "The Martians"
prompt_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
output = model.generate(prompt_ids, max_new_tokens=40, top_k=20)
print("\n--- sample after training ---")
print(tokenizer.decode(output[0]))