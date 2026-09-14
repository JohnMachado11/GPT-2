import torch
from gpt2.train.optimizer import AdamW


def test_matches_pytorch_adamw_over_100_steps():
    # two identical toy models (same seed + copied weights -> identical start)
    torch.manual_seed(0)
    mine = torch.nn.Linear(4, 3)
    torch.manual_seed(0)
    reference = torch.nn.Linear(4, 3)
    reference.load_state_dict(mine.state_dict())

    my_opt = AdamW(mine.parameters(), learning_rate=1e-2, weight_decay=0.01)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=1e-2,
                                betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    torch.manual_seed(1)
    x = torch.randn(8, 4)
    target = torch.randn(8, 3)
    loss_fn = torch.nn.MSELoss()

    for _ in range(100):
        my_opt.zero_grad()
        loss_fn(mine(x), target).backward()
        my_opt.step()

        ref_opt.zero_grad()
        loss_fn(reference(x), target).backward()
        ref_opt.step()

    # after 100 steps the weights should match PyTorch's to float-rounding
    max_diff = max((a - b).abs().max().item()
                   for a, b in zip(mine.parameters(), reference.parameters()))
    assert max_diff < 1e-5


def test_zero_grad_clears_gradients():
    layer = torch.nn.Linear(3, 2)
    opt = AdamW(layer.parameters())
    layer(torch.randn(4, 3)).sum().backward()
    assert layer.weight.grad is not None      # backward filled it
    opt.zero_grad()
    assert layer.weight.grad is None          # cleared


def test_step_actually_lowers_the_loss():
    torch.manual_seed(0)
    layer = torch.nn.Linear(4, 2)
    opt = AdamW(layer.parameters(), learning_rate=1e-2)
    x = torch.randn(16, 4)
    target = torch.randn(16, 2)
    loss_fn = torch.nn.MSELoss()

    first_loss = loss_fn(layer(x), target).item()
    for _ in range(50):
        opt.zero_grad()
        loss_fn(layer(x), target).backward()
        opt.step()
    last_loss = loss_fn(layer(x), target).item()
    assert last_loss < first_loss
