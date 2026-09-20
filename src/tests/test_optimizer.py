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
    # Our AdamW decays only the 2-D weight matrices, so PyTorch has to be given the
    # SAME split -- two param groups -- or the bias alone would make them disagree.
    ref_opt = torch.optim.AdamW(
        [
            {"params": [p for p in reference.parameters() if p.dim() >= 2], "weight_decay": 0.01},
            {"params": [p for p in reference.parameters() if p.dim() < 2], "weight_decay": 0.0},
        ],
        lr=1e-2, betas=(0.9, 0.999), eps=1e-8,
    )

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


def test_weight_decay_skips_biases_and_layernorm():
    # Standard GPT practice: decay only the 2-D weight matrices. A Linear's weight is 2-D (decayed);
    # its bias and a LayerNorm's scale/shift are 1-D (skipped).
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    opt = AdamW(model.parameters())
    decayed = [p for p, yes in zip(opt.parameters, opt.decayed) if yes]
    skipped = [p for p, yes in zip(opt.parameters, opt.decayed) if not yes]
    assert [p.dim() for p in decayed] == [2]        # just the Linear's weight
    assert [p.dim() for p in skipped] == [1, 1, 1]  # Linear bias + LayerNorm scale & shift


def test_decay_shrinks_only_the_2d_weights():
    # with NO gradients at all, decay is the only thing that can move a parameter:
    # the 2-D weight must shrink, the 1-D bias must not budge.
    layer = torch.nn.Linear(4, 4)
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.bias.fill_(1.0)
    layer.weight.grad = torch.zeros_like(layer.weight)
    layer.bias.grad = torch.zeros_like(layer.bias)

    opt = AdamW(layer.parameters(), learning_rate=1e-2, weight_decay=0.1)
    opt.step()

    assert layer.weight.max().item() < 1.0        # shrunk by weight decay
    assert layer.bias.min().item() == 1.0         # untouched
