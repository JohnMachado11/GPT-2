import torch
from gpt2.model.config import GPTConfig
from gpt2.model.block import Block


def test_residual_passthrough():
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=26, context_length=64, embed_dim=128, num_layers=2, num_heads=4)
    block = Block(config)

    # zero both sublayers so each contributes nothing
    for p in list(block.attn.parameters()) + list(block.ff.parameters()):
        torch.nn.init.zeros_(p)
    
    x = torch.randn(2, 5, 128)
    # with sublayers outputting 0, the residual highway must hand x straight through
    assert torch.allclose(block(x), x, atol=1e-6)