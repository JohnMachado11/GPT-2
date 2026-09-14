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

# Read EVERY .txt in the data folder and tokenize it into one long token stream.
# Drop in as many files as you like -- they're concatenated, no code changes needed.
files = sorted(data_dir.glob("*.txt"))
if not files:
    raise SystemExit(f"no .txt files found in {data_dir} -- add some text to train on")
token_data = torch.cat([torch.tensor(tokenizer.encode(f.read_text())) for f in files])

print(f"loaded {len(files)} file(s) from {data_dir.name}/ -> {len(token_data)} tokens")
for f in files:
    print(f"  - {f.name}")

# Hold out 10% of the tokens for validation (the model never trains on them), so we can
# spot overfitting -- if train loss keeps falling while val loss rises, it's memorizing.
# 0.9 = 90% train / 10% val is a good ratio for a SMALL dataset like this (sample.txt). With a much
# bigger corpus you'd raise it (e.g. 0.99) -- a tiny fraction is already plenty of val tokens.
split = int(0.9 * len(token_data))
train_data, val_data = token_data[:split], token_data[split:]
print(f"split -> {len(train_data)} train tokens, {len(val_data)} val tokens")

config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    context_length=64,
    embed_dim=128,
    num_layers=4,
    num_heads=4,
    dropout=0.0,
)

model = GPT(config)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

optimizer = AdamW(model.parameters(), weight_decay=0.01)

history = train(
    model,
    optimizer,
    train_data,
    batch_size=16,
    context_length=config.context_length,
    num_steps=500,
    max_learning_rate=1e-3,
    warmup_steps=50,
    min_learning_rate=1e-4,
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
prompt = "The lighthouse"
prompt_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
output = model.generate(prompt_ids, max_new_tokens=40, top_k=20)
print("\n--- sample after training ---")
print(tokenizer.decode(output[0]))