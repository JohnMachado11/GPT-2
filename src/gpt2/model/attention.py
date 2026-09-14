import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # produces query, key, and value for every position, all in one shot
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        # mixes the heads' outputs back together at the very end
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)     # randomly drops attention weights
        self.resid_dropout = nn.Dropout(dropout)    # drops the output before it rejoins the stream
    
    def forward(self, x):
        batch, positions, _ = x.shape

        qkv = self.qkv(x)                                       # (batch, positions, 3 * embed_dim)
        query, key, value = qkv.split(self.embed_dim, dim=-1)   # three of (batch, positions, embed_dim)

        # split each into heads: (batch, positions, embed_dim) -> (batch, num_heads, positions, head_dim)
        query = query.view(batch, positions, self.num_heads, self.head_dim).transpose(1, 2)
        key   = key.view(batch, positions, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, positions, self.num_heads, self.head_dim).transpose(1, 2)

        # score every query against every key
        scores = (query @ key.transpose(-2, -1)) / (self.head_dim ** 0.5)   # (batch, num_heads, positions, positions)

        # causal mask: block each position from seeing the future
        mask = torch.tril(torch.ones(positions, positions, device=x.device)).bool()
        scores = scores.masked_fill(~mask, float("-inf"))

        # softmax turns each row of scores into weights that sum to 1
        weights = torch.softmax(scores, dim=-1)     # (batch, num_heads, positions, position)
        weights = self.attn_dropout(weights)

        # blend the values using the attention weights
        out = weights @ value                       # (batch, num_heads, positions, head_dim)

        # recombine the heads: (batch, num_heads, positions, head_dim) -> (batch, positions, embed_dim)
        out = out.transpose(1, 2).contiguous().view(batch, positions, self.embed_dim)

        # final learned mix across the combined heads
        out = self.out_proj(out)                    # (batch, positions, embed_dim)
        out = self.resid_dropout(out)
        return out
