from pathlib import Path

from transformers import AutoTokenizer

from gpt2.checkpoint import load_checkpoint


checkpoints_dir = Path(__file__).parent.parent / "checkpoints"
checkpoints = list(checkpoints_dir.glob("*.pt"))
if not checkpoints:
    raise SystemExit(f"no checkpoints in {checkpoints_dir} -- run src/run/train_on_text.py first")

latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
print(f"loading {latest.name}")

model = load_checkpoint(latest)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = "The Martians"
token_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(token_ids, max_new_tokens=40, top_k=20)
print("\n--- your model's continuation ---")
print(tokenizer.decode(output[0]))