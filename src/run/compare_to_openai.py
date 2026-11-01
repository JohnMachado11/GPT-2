# Score your GPT-2 and OpenAI's GPT-2 of the same size on the SAME held-out FineWeb-Edu text -- all of the val
# shards, once: the "did I match OpenAI?" number. It's the average loss per token: lower is better, and random
# guessing scores about 10.8. (Your model trained on FineWeb-Edu and OpenAI's didn't, so on this text yours can come
# out ahead.) On Colab, after the training cell (the same --size and folders):
#
#   !PYTHONPATH=src python src/run/compare_to_openai.py --size small --data /content/fineweb --run-dir /content/drive/MyDrive/gpt2-run
#
# It scores the run's best.pt (or its last.pt, if no validation ran yet), so it works mid-run too.
# The first run downloads OpenAI's weights from Hugging Face (about 550 MB for small, 1.5 GB for medium).

import argparse
from pathlib import Path

from gpt2.checkpoint import load_checkpoint
from gpt2.from_hf import load_gpt2
from gpt2.train.data import ShardLoader
from gpt2.train.device import pick_device, supports_bf16
from gpt2.train.loop import estimate_loss
from gpt2.train.shards import is_finished_build_of, shard_paths
from run import settings_gpt2_small, settings_gpt2_medium

SETTINGS = {"small": settings_gpt2_small, "medium": settings_gpt2_medium}     # --size picks one


def main(args):
    settings = SETTINGS[args.size]          # batch_size, the shape and openai_model: src/run/settings_gpt2_<size>.py
    if not is_finished_build_of(args.data, {}):
        raise SystemExit(f"{args.data} isn't a finished prep with every shard there in full")
    device = pick_device()
    use_bf16 = supports_bf16(device)

    yours = load_checkpoint(args.run_dir, device)
    # a run of another size would be scored against the wrong OpenAI model -- refuse before downloading it
    your_shape = (yours.config.embed_dim, yours.config.num_layers, yours.config.num_heads)
    if your_shape != (settings.embed_dim, settings.num_layers, settings.num_heads):
        raise SystemExit(f"{args.run_dir} isn't a GPT-2 {args.size} run -- pass the --size it was trained with")
    openai = load_gpt2(settings.openai_model).to(device)

    # the same windows for both: estimate_loss rewinds the loader before it scores. One pass over ALL of val,
    # so every window counts, once
    context_length = yours.config.context_length
    val_batches = ShardLoader(shard_paths(args.data, "val"), settings.batch_size, context_length, device)
    num_batches = val_batches.batches_per_pass()
    print(f"scoring both on the same {num_batches * settings.batch_size * context_length:,} validation tokens\n")

    your_loss = estimate_loss(yours, val_batches, device, num_batches=num_batches, use_bf16=use_bf16)
    openai_loss = estimate_loss(openai, val_batches, device, num_batches=num_batches, use_bf16=use_bf16)
    print(f"  your model            {your_loss:.4f}   ({Path(args.run_dir).name})")
    print(f"  OpenAI's GPT-2 {args.size:<7}{openai_loss:.4f}")
    print(f"  difference            {your_loss - openai_loss:+.4f}   (below zero: yours is better)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score your GPT-2 and OpenAI's of the same size on the same text.")
    parser.add_argument("--size", required=True, choices=SETTINGS, help="which GPT-2 the run trained: small or medium")
    parser.add_argument("--data", required=True, help="the folder prepare_data.py wrote")
    parser.add_argument("--run-dir", required=True, help="the run folder train_gpt2.py saved into")
    main(parser.parse_args())
