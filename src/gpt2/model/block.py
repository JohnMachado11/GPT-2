import torch.nn as nn

from gpt2.model.layernorm import LayerNorm
from gpt2.model.attention import CausalSelfAttention
from gpt2.model.feedforward import FeedForward


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_before_attn = LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config.embed_dim, config.num_heads, config.bias, config.dropout)
        self.norm_before_ff = LayerNorm(config.embed_dim)
        self.ff = FeedForward(config.embed_dim, config.bias, config.dropout)
    
    def forward(self, x):
        x = x + self.attn(self.norm_before_attn(x))
        x = x + self.ff(self.norm_before_ff(x))
        return x