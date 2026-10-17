import torch
from gpt2.model.config import GPTConfig
from gpt2.model.gpt import GPT
from gpt2.train.optimizer import AdamW
from gpt2.train.loop import train


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
        model, AdamW(model.parameters()), data,
        batch_size=8, context_length=16, num_steps=60,
        max_learning_rate=1e-2, warmup_steps=5, min_learning_rate=1e-3,
        max_gradient_norm=1.0, device="cpu", use_bf16=True,
    )

    assert history["train_loss"][-1] < history["train_loss"][0] - 1.0
    assert all(p.dtype == torch.float32 for p in model.parameters())
