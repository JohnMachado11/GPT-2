import torch
import torch.nn.functional as F
from gpt2.model.feedforward import FeedForward


def test_feedforward_wiring():
    torch.manual_seed(0)
    ff = FeedForward(embed_dim=16)
    x = torch.randn(2, 4, 16)

    # rebuild the forward by hand: expand -> GELU(tanh) -> contract
    manual = ff.contract(F.gelu(ff.expand(x), approximate="tanh"))

    assert torch.allclose(ff(x), manual, atol=1e-6)  # right order, right activation
    assert ff(x).shape == x.shape