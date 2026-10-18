import pytest
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


def test_state_dict_round_trip_resumes_exactly(tmp_path):
    # train 10 steps straight, vs 5 steps -> save to disk -> load into a FRESH model and
    # optimizer -> 5 more. The weights must come out identical.
    def make():
        torch.manual_seed(0)
        layer = torch.nn.Linear(4, 3)
        return layer, AdamW(layer.parameters(), learning_rate=1e-2)

    torch.manual_seed(1)
    x, target = torch.randn(8, 4), torch.randn(8, 3)

    def train_steps(layer, opt, n):
        for _ in range(n):
            opt.zero_grad()
            torch.nn.functional.mse_loss(layer(x), target).backward()
            opt.step()

    straight, straight_opt = make()
    train_steps(straight, straight_opt, 10)

    first, first_opt = make()
    train_steps(first, first_opt, 5)
    torch.save({"model": first.state_dict(), "optimizer": first_opt.state_dict()}, tmp_path / "state.pt")

    resumed, resumed_opt = make()                       # brand new, as after a restart
    saved = torch.load(tmp_path / "state.pt")
    resumed.load_state_dict(saved["model"])
    resumed_opt.load_state_dict(saved["optimizer"])
    assert resumed_opt.step_count == 5
    train_steps(resumed, resumed_opt, 5)

    for a, b in zip(straight.parameters(), resumed.parameters()):
        assert torch.equal(a, b)


def test_load_state_dict_rejects_state_from_a_different_model():
    small = AdamW(torch.nn.Linear(4, 3).parameters())                       # 2 tensors
    bigger = AdamW(torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2)).parameters())   # 4
    with pytest.raises(ValueError):
        bigger.load_state_dict(small.state_dict())


def test_state_dict_records_the_optimizer_settings():
    opt = AdamW(torch.nn.Linear(4, 3).parameters(), momentum_decay=0.8, bumpiness_decay=0.95, epsilon=1e-6,
                weight_decay=0.2)
    assert opt.state_dict()["hyperparameters"] == {
        "momentum_decay": 0.8, "bumpiness_decay": 0.95, "epsilon": 1e-6, "weight_decay": 0.2,
    }


def test_load_state_dict_rejects_different_optimizer_settings():
    saved = AdamW(torch.nn.Linear(4, 3).parameters(), weight_decay=0.1).state_dict()
    other = AdamW(torch.nn.Linear(4, 3).parameters(), weight_decay=0.0)
    with pytest.raises(ValueError, match="quietly change how the weights move"):
        other.load_state_dict(saved)


def test_load_state_dict_rejects_same_count_but_different_shapes():
    saved = AdamW(torch.nn.Linear(4, 3).parameters()).state_dict()           # weight (3, 4), bias (3,)
    other = AdamW(torch.nn.Linear(5, 3).parameters())                        # weight (3, 5), bias (3,)
    with pytest.raises(ValueError, match="different model"):
        other.load_state_dict(saved)


def test_a_failed_load_changes_nothing():
    saved = AdamW(torch.nn.Linear(4, 3).parameters(), weight_decay=0.1).state_dict()
    saved["step_count"] = 99
    other = AdamW(torch.nn.Linear(4, 3).parameters(), weight_decay=0.0)
    with pytest.raises(ValueError):
        other.load_state_dict(saved)
    assert other.step_count == 0


def test_state_saved_before_settings_were_recorded_still_loads():
    old_style = AdamW(torch.nn.Linear(4, 3).parameters()).state_dict()
    del old_style["hyperparameters"]
    opt = AdamW(torch.nn.Linear(4, 3).parameters())
    opt.load_state_dict(old_style)
    assert opt.step_count == 0


def test_matches_pytorch_adamw_with_tiny_gradients():
    # gradients around 1e-6 make epsilon's exact placement matter -- the big-gradient test can't tell
    torch.manual_seed(0)
    mine, reference = torch.nn.Linear(4, 3), torch.nn.Linear(4, 3)
    reference.load_state_dict(mine.state_dict())
    my_opt = AdamW(mine.parameters(), learning_rate=1e-2, weight_decay=0.0)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=1e-2, weight_decay=0.0, eps=1e-8)
    x, target = torch.randn(8, 4), torch.randn(8, 3)
    for _ in range(20):
        for model, opt in ((mine, my_opt), (reference, ref_opt)):
            opt.zero_grad()
            (1e-6 * torch.nn.functional.mse_loss(model(x), target)).backward()
            opt.step()
    assert max((a - b).abs().max().item() for a, b in zip(mine.parameters(), reference.parameters())) < 1e-6


@pytest.mark.parametrize("setting", ["momentum_decay", "bumpiness_decay", "epsilon", "weight_decay"])
def test_load_state_dict_rejects_each_changed_setting(setting):
    defaults = {"momentum_decay": 0.9, "bumpiness_decay": 0.999, "epsilon": 1e-8, "weight_decay": 0.1}
    saved = AdamW(torch.nn.Linear(4, 3).parameters(), **defaults).state_dict()
    changed = dict(defaults, **{setting: defaults[setting] * 0.5})
    with pytest.raises(ValueError, match="quietly change how the weights move"):
        AdamW(torch.nn.Linear(4, 3).parameters(), **changed).load_state_dict(saved)


def test_load_state_dict_checks_the_bumpiness_tensors_too():
    saved = AdamW(torch.nn.Linear(4, 3).parameters()).state_dict()
    saved["bumpiness"] = [torch.zeros(9, 9), saved["bumpiness"][1]]   # damaged: wrong shape
    with pytest.raises(ValueError, match="different model"):
        AdamW(torch.nn.Linear(4, 3).parameters()).load_state_dict(saved)


def test_load_state_dict_checks_the_bumpiness_count_too():
    saved = AdamW(torch.nn.Linear(4, 3).parameters()).state_dict()
    saved["bumpiness"] = saved["bumpiness"][:1]                         # damaged: one tensor missing
    with pytest.raises(ValueError, match="bumpiness tensors"):
        AdamW(torch.nn.Linear(4, 3).parameters()).load_state_dict(saved)
