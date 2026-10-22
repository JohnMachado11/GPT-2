# Score your GPT-2 small and OpenAI's GPT-2 small on the SAME held-out FineWeb-Edu text -- all of the val shards,
# once: the "did I match OpenAI?" number. It's the average loss per token: lower is better, and random guessing
# scores about 10.8. (Your model trained on FineWeb-Edu and OpenAI's didn't, so on this text yours can come out
# ahead.) On Colab, after the training cell (same folders):
#
#   !PYTHONPATH=src python src/run/compare_to_openai.py --data /content/fineweb --run-dir /content/drive/MyDrive/gpt2-run
#
# It scores the run's best.pt (or its last.pt, if no validation ran yet), so it works mid-run too.
# The first run downloads OpenAI's weights from Hugging Face (about 550 MB).

import argparse
from pathlib import Path

from gpt2.checkpoint import load_checkpoint
from gpt2.from_hf import load_gpt2
from gpt2.train.data import ShardLoader
from gpt2.train.device import pick_device, supports_bf16
from gpt2.train.loop import estimate_loss
from gpt2.train.shards import is_finished_build_of, shard_paths
from run import settings_gpt2_small as settings     # batch_size: src/run/settings_gpt2_small.py


def main(args):
    if not is_finished_build_of(args.data, {}):
        raise SystemExit(f"{args.data} isn't a finished prep with every shard there in full")
    device = pick_device()
    use_bf16 = supports_bf16(device)

    yours = load_checkpoint(args.run_dir, device)
    openai = load_gpt2("gpt2").to(device)

    # the same windows for both: estimate_loss rewinds the loader before it scores. One pass over ALL of val,
    # so every window counts, once
    context_length = yours.config.context_length
    val_batches = ShardLoader(shard_paths(args.data, "val"), settings.batch_size, context_length, device)
    num_batches = val_batches.batches_per_pass()
    print(f"scoring both on the same {num_batches * settings.batch_size * context_length:,} validation tokens\n")

    your_loss = estimate_loss(yours, val_batches, device, num_batches=num_batches, use_bf16=use_bf16)
    openai_loss = estimate_loss(openai, val_batches, device, num_batches=num_batches, use_bf16=use_bf16)
    print(f"  your model            {your_loss:.4f}   ({Path(args.run_dir).name})")
    print(f"  OpenAI's GPT-2 small  {openai_loss:.4f}")
    print(f"  difference            {your_loss - openai_loss:+.4f}   (below zero: yours is better)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score your GPT-2 small and OpenAI's on the same validation text.")
    parser.add_argument("--data", required=True, help="the folder prepare_data.py wrote")
    parser.add_argument("--run-dir", required=True, help="the run folder train_gpt2_small.py saved into")
    main(parser.parse_args())
