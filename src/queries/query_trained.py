from pathlib import Path

from transformers import AutoTokenizer

from gpt2.checkpoint import load_checkpoint
from run import settings        # the sample's prompt, length and top_k: src/run/settings.py


checkpoints_dir = Path(__file__).parent.parent / "checkpoints"

# every run folder that has saved a model (best.pt, or last.pt if no validation ran)
runs = [
    folder for folder in checkpoints_dir.glob("*")
    if folder.is_dir() and ((folder / "best.pt").exists() or (folder / "last.pt").exists())
]
if not runs:
    raise SystemExit(f"no trained models in {checkpoints_dir} -- run src/run/train_on_text.py first")

def last_saved(folder):
    # when this run last saved a model -- not the folder's own time, which any new file (even a
    # .DS_Store) changes
    return max(file.stat().st_mtime for file in (folder / "best.pt", folder / "last.pt") if file.exists())


latest = max(runs, key=last_saved)
print(f"loading {latest.name}")

model = load_checkpoint(latest)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

token_ids = tokenizer.encode(settings.prompt, return_tensors="pt")
output = model.generate(token_ids, max_new_tokens=settings.max_new_tokens, top_k=settings.top_k,
                        stop_at_token=tokenizer.eos_token_id)
print("\n--- your model's continuation ---")
print(tokenizer.decode(output[0]))