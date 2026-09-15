from pathlib import Path

from transformers import AutoTokenizer


here = Path(__file__).parent
data_dir = here / "data"

tokenizer = AutoTokenizer.from_pretrained("gpt2")

files = sorted(data_dir.glob("*.txt"))
if not files:
    raise SystemExit(f"no .txt files found in {data_dir}")

total_tokens = 0
total_chars = 0

print(f"{'file':<24}{'chars':>12}{'tokens':>12}")
print("-" * 48)
for f in files:
    text = f.read_text()
    n_tokens = len(tokenizer.encode(text))
    total_chars += len(text)
    total_tokens += n_tokens
    print(f"{f.name:<24}{len(text):>12,}{n_tokens:>12,}")
print("-" * 48)
print(f"{'TOTAL':<24}{total_chars:>12,}{total_tokens:>12,}")

