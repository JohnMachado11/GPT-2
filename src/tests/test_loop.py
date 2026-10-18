import torch
from gpt2.model.config import GPTConfig
from gpt2.model.gpt import GPT
from gpt2.train.optimizer import AdamW
from gpt2.train.loop import estimate_loss, train
from gpt2.train.data import RandomWindows, ShardLoader
from gpt2.train.shards import write_shard


def make_model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=50, context_length=16, embed_dim=32, num_layers=2, num_heads=2))


def test_bf16_forward_is_close_to_32bit():
    # bf16 keeps fewer digits, so the loss shifts a little -- but only a little
    model = make_model()
    token_ids = torch.randint(0, 50, (4, 16))
    targets = torch.randint(0, 50, (4, 16))

    _, loss_32bit = model(token_ids, targets)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _, loss_bf16 = model(token_ids, targets)

    assert abs(loss_32bit.item() - loss_bf16.item()) < 0.05


def test_train_with_bf16_learns_and_keeps_weights_32bit():
    # a repeating pattern is easy to learn, so the loss must clearly fall --
    # and autocast only changes the math, never the stored weights
    model = make_model()
    data = torch.arange(20).repeat(100)
    history = train(
        model, AdamW(model.parameters()), RandomWindows(data, batch_size=8, context_length=16),
        num_steps=60, max_learning_rate=1e-2, warmup_steps=5, min_learning_rate=1e-3,
        max_gradient_norm=1.0, device="cpu", use_bf16=True,
    )

    assert history["train_loss"][-1] < history["train_loss"][0] - 1.0
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_train_learns_from_shards_on_disk(tmp_path):
    # the same easy pattern, but read off shard files by ShardLoader instead of from memory
    write_shard(tmp_path / "train_000000.bin", list(range(20)) * 60)
    write_shard(tmp_path / "train_000001.bin", list(range(20)) * 60)
    write_shard(tmp_path / "val_000000.bin", list(range(20)) * 10)
    model = make_model()
    history = train(
        model, AdamW(model.parameters()),
        ShardLoader(sorted(tmp_path.glob("train_*.bin")), batch_size=8, context_length=16),
        num_steps=60, max_learning_rate=1e-2, warmup_steps=5, min_learning_rate=1e-3,
        max_gradient_norm=1.0, val_batches=ShardLoader([tmp_path / "val_000000.bin"], batch_size=8, context_length=16),
        eval_interval=20, device="cpu",
    )
    assert history["train_loss"][-1] < history["train_loss"][0] - 1.0
    assert history["val_loss"][-1] < history["val_loss"][0] - 1.0


def test_val_loss_from_shards_is_repeatable(tmp_path):
    # estimate_loss resets the loader first, so the same model gets the same number every time
    write_shard(tmp_path / "val_000000.bin", torch.randint(0, 50, (2000,)).tolist())
    val = ShardLoader([tmp_path / "val_000000.bin"], batch_size=4, context_length=16)
    model = make_model()
    first = estimate_loss(model, val, device="cpu", num_batches=5)
    second = estimate_loss(model, val, device="cpu", num_batches=5)
    assert first == second
