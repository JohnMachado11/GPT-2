# Train GPT-2 small (124M) from scratch on the FineWeb-Edu shards prepare_data.py makes: one pass over them.
# Every setting is in src/run/settings_gpt2_small.py. The two folders are flags, since they differ per machine.
# On Colab, with the shards copied from Google Drive to the runtime's own disk first (reading them off Drive
# mid-run can stall), and the run folder on Drive so the checkpoints outlive the runtime:
#
#   !PYTHONPATH=src python src/run/train_gpt2_small.py --data /content/fineweb --run-dir /content/drive/MyDrive/gpt2-run
#
# Interrupted? Run the SAME command again: if --run-dir holds a last.pt, it picks up from there. When it's done,
# drive.flush_and_unmount() makes sure the last checkpoints finish uploading.
# (A test run on a Mac: set compile_model = False first -- it isn't allowed on a Mac GPU.)
# Then score it against OpenAI's GPT-2 with compare_to_openai.py.

import argparse
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
from gpt2.train.data import ShardLoader
from gpt2.train.device import pick_device, supports_bf16
from gpt2.train.shards import is_finished_build_of, read_json, shard_paths
from run import settings_gpt2_small as settings     # every setting: src/run/settings_gpt2_small.py


def main(args):
    data_dir, run_dir = Path(args.data), Path(args.run_dir)

    # every shard there in full, before any training starts
    if not is_finished_build_of(data_dir, {}):
        raise SystemExit(f"{data_dir} isn't a finished prep with every shard there in full -- let prepare_data.py "
                         f"(or the copy off Google Drive) finish first")
    info = read_json(data_dir / "info.json")

    torch.manual_seed(settings.seed)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

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
          f"compile {'on' if settings.compile_model else 'off'}")
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        momentum_decay=settings.momentum_decay,
        bumpiness_decay=settings.bumpiness_decay,
        weight_decay=settings.weight_decay,
    )

    # training reads the train shards in order; every validation check scores the same first windows of val
    train_batches = ShardLoader(shard_paths(data_dir, "train"), settings.batch_size, config.context_length, device)
    val_batches = ShardLoader(shard_paths(data_dir, "val"), settings.batch_size, config.context_length, device)

    # ONE pass over the training shards: as many steps as their whole batches fill
    num_steps = train_batches.batches_per_pass() // settings.accumulation_steps
    if num_steps < 1:
        raise SystemExit(f"{data_dir} has too few train tokens for even one step -- prep more")
    tokens_per_step = settings.batch_size * settings.accumulation_steps * config.context_length
    params = sum(p.numel() for p in model.parameters())
    print(f"{params / 1e6:.0f}M parameters | {info['train_tokens']:,} train tokens, {info['val_tokens']:,} val tokens | "
          f"{num_steps:,} steps of {tokens_per_step:,} tokens\n")

    history = train(
        model,
        optimizer,
        train_batches,
        num_steps=num_steps,
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

    prompt_ids = torch.tensor(tokenizer.encode(settings.prompt), device=device).unsqueeze(0)
    output = model.generate(prompt_ids, max_new_tokens=settings.max_new_tokens, top_k=settings.top_k,
                            stop_at_token=tokenizer.eos_token_id)
    print("\n--- sample after training ---")
    print(tokenizer.decode(output[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT-2 small from scratch on prepare_data.py's shards.")
    parser.add_argument("--data", required=True,
                        help="the folder prepare_data.py wrote (on Colab: a copy on the runtime's own disk)")
    parser.add_argument("--run-dir", required=True,
                        help="where the checkpoints go; run again with the same folder to resume")
    main(parser.parse_args())
