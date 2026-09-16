from pathlib import Path

from transformers import AutoTokenizer
from transformers import logging as transformers_logging

import scaling_config


def round_to_multiple_of_64(value):
    # model widths are multiples of 64 (head_dim = 64); never below 64
    return max(64, round(value / 64) * 64)


here = Path(__file__).parent
data_dir = here / "data"

transformers_logging.set_verbosity_error()   # hide the harmless ">1024 tokens" warning when counting
tokenizer = AutoTokenizer.from_pretrained("gpt2")

files = sorted(data_dir.glob("*.txt"))
if not files:
    raise SystemExit(f"no .txt files found in {data_dir}")

total_tokens = 0
total_chars = 0

print(f"\n{'file':<28}{'chars':>12}{'tokens':>12}")
print("-" * 52)
for f in files:
    text = f.read_text()
    n_tokens = len(tokenizer.encode(text))
    total_chars += len(text)
    total_tokens += n_tokens
    print(f"{f.name:<28}{len(text):>12,}{n_tokens:>12,}")
print("-" * 52)
print(f"{'TOTAL':<28}{total_chars:>12,}{total_tokens:>12,}")


# --- suggested model size, from the token count + scaling_config.MODE ---
head_dim = 64

if scaling_config.MODE == "chinchilla_exact":
    # compute-optimal: ~20 tokens per parameter, trained about one epoch
    embed_dim = round_to_multiple_of_64((total_tokens / 3.75) ** (1 / 3))
    num_layers = num_heads = embed_dim // head_dim
    num_steps = round(total_tokens / (scaling_config.batch_size * scaling_config.context_length))
elif scaling_config.MODE == "minimum_viable":
    # smallest usable model, over-trained a fixed number of epochs
    embed_dim = scaling_config.minimum_embed_dim
    num_layers = num_heads = embed_dim // head_dim
    num_steps = round(
        scaling_config.minimum_viable_epochs
        * total_tokens
        / (scaling_config.batch_size * scaling_config.context_length)
    )
else:
    raise SystemExit(f"unknown MODE in scaling_config.py: {scaling_config.MODE!r}")

print()
print(f"Suggested config  (mode: {scaling_config.MODE})")
print("-" * 52)
print(f"  embed_dim       {embed_dim}")
print(f"  num_layers      {num_layers}")
print(f"  num_heads       {num_heads}")
print(f"  dropout         {scaling_config.dropout}")
print(f"  batch_size      {scaling_config.batch_size}")
print(f"  context_length  {scaling_config.context_length}")
print(f"  num_steps       {num_steps:,}")

optimal_params = round(total_tokens / 20)
if optimal_params >= 1_000_000:
    optimal_params_text = f"{optimal_params / 1_000_000:.1f}M"
else:
    optimal_params_text = f"{optimal_params:,}"
print("-" * 52)
print(f"  Chinchilla optimal ~= {optimal_params_text} params (tokens / 20)")

if scaling_config.MODE == "minimum_viable":
    print(f"  note: smallest usable model, over-trained ~{scaling_config.minimum_viable_epochs} epochs.")
    print("        add more text to justify a bigger model.")
elif scaling_config.MODE == "chinchilla_exact" and embed_dim < 128:
    print('  note: below a usable size for this little text — try MODE = "minimum_viable".')

print()
print("copy these into the GPTConfig + train(...) call in train_on_text.py\n")
