import torch
from gpt2.model.layernorm import LayerNorm


def test_matches_pytorch():
    torch.manual_seed(0)
    embed_dim = 128
    x = torch.randn(2, 5, embed_dim)

    mine = LayerNorm(embed_dim)
    reference = torch.nn.LayerNorm(embed_dim)

    with torch.no_grad():
        weight = torch.randn(embed_dim)
        bias = torch.randn(embed_dim)
        mine.scale.copy_(weight)
        mine.shift.copy_(bias)
        reference.weight.copy_(weight)
        reference.bias.copy_(bias)
    
    assert torch.allclose(mine(x), reference(x), atol=1e-5)