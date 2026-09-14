import torch
from gpt2.model.attention import CausalSelfAttention


def test_causal_no_future_leak():
    torch.manual_seed(0)
    attn = CausalSelfAttention(embed_dim=32, num_heads=4)

    x = torch.randn(1, 6, 32)               # 1 sequence, 6 positions, 32 numbers each
    out = attn(x)

    # change ONLY the last position's input, then rerun
    x_changed = x.clone()
    x_changed[:, -1, :] = torch.randn(32)
    out_changed = attn(x_changed)

    # positions BEFORE the last must be identical — they can't see the future
    assert torch.allclose(out[:, :-1, :], out_changed[:, :-1, :], atol=1e-6)
    # the last position MUST change — it saw its own new input
    assert not torch.allclose(out[:, -1, :], out_changed[:, -1, :])
