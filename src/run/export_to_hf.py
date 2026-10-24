# Save a trained run's model in Hugging Face's GPT-2 format -- model.safetensors, config.json and the GPT-2
# tokenizer -- ready to upload. Anyone can then load it with transformers' from_pretrained(), or with this repo's
# load_gpt2(). It first checks that the converted model gives the same outputs as yours, and saves only if it does.
#
#   .venv/bin/python src/run/export_to_hf.py --run-dir src/checkpoints/gpt2-run
#
# Then upload the folder it prints (after `.venv/bin/hf auth login` with a Hugging Face token that can write):
#   .venv/bin/hf upload <your-name>/<model-name> src/checkpoints/gpt2-run/hf .

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from gpt2.checkpoint import load_checkpoint
from gpt2.from_hf import to_hugging_face_model


def main(args):
    model = load_checkpoint(args.run_dir)          # the run's best.pt (or last.pt), in eval mode
    hugging_face_model = to_hugging_face_model(model)

    # the same outputs, before anything is saved
    token_ids = torch.randint(0, model.config.vocab_size, (2, min(64, model.config.context_length)))
    with torch.no_grad():
        difference = (hugging_face_model(token_ids).logits - model(token_ids)[0]).abs().max().item()
    if difference > 1e-3:
        raise SystemExit(f"the converted model's outputs differ by up to {difference:.1e} -- not saving")
    print(f"same outputs as your model (largest difference {difference:.1e})")

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else (run_dir if run_dir.is_dir() else run_dir.parent) / "hf"
    hugging_face_model.save_pretrained(out)
    AutoTokenizer.from_pretrained("gpt2").save_pretrained(out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save a trained run's model in Hugging Face's GPT-2 format.")
    parser.add_argument("--run-dir", required=True,
                        help="the run folder (its best.pt is exported, or last.pt if there's none), or a .pt file")
    parser.add_argument("--out", help="where to save (default: an hf folder inside the run folder)")
    main(parser.parse_args())
