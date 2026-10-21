import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    # fast=False: attention spelled out step by step (below) -- the readable version, and the reference
    # fast=True:  the same math in one call to PyTorch's fused attention -- faster, and on NVIDIA GPUs far
    #             less memory (its dropout draws its randomness differently, so runs differ slightly)
    def __init__(self, embed_dim, num_heads, bias=True, dropout=0.0, context_length=1024, fast=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.fast = fast

        # produces query, key, and value for every position, all in one shot
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        # mixes the heads' outputs back together at the very end
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)     # randomly drops attention weights
        self.resid_dropout = nn.Dropout(dropout)    # drops the output before it rejoins the stream

        # causal mask: position i may look at positions 0..i only. Built ONCE, for the longest input the
        # model takes; each forward uses its top-left corner. (persistent=False: it isn't learned, so it's
        # not saved in checkpoints.)
        causal_mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x):
        batch, positions, _ = x.shape

        qkv = self.qkv(x)                                       # (batch, positions, 3 * embed_dim)
        query, key, value = qkv.split(self.embed_dim, dim=-1)   # three of (batch, positions, embed_dim)

        # split each into heads: (batch, positions, embed_dim) -> (batch, num_heads, positions, head_dim)
        query = query.view(batch, positions, self.num_heads, self.head_dim).transpose(1, 2)
        key   = key.view(batch, positions, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, positions, self.num_heads, self.head_dim).transpose(1, 2)

        if self.fast:
            # the five steps below in ONE call. PyTorch's fused attention (FlashAttention on recent NVIDIA
            # GPUs) works through the scores in small tiles and never stores the full positions x positions
            # table -- the same result, much faster
            out = F.scaled_dot_product_attention(query, key, value, is_causal=True,
                                                 dropout_p=self.attn_dropout.p if self.training else 0.0)
        else:
            # score every query against every key
            scores = (query @ key.transpose(-2, -1)) / (self.head_dim ** 0.5)   # (batch, num_heads, positions, positions)

            # causal mask: block each position from seeing the future
            # (this used to rebuild the mask on every call: torch.tril(torch.ones(positions, positions)) --
            # the same mask, built again for every layer at every step)
            mask = self.causal_mask[:positions, :positions]
            scores = scores.masked_fill(~mask, float("-inf"))

            # softmax turns each row of scores into weights that sum to 1
            weights = torch.softmax(scores, dim=-1)     # (batch, num_heads, positions, positions)
            weights = self.attn_dropout(weights)

            # blend the values using the attention weights
            out = weights @ value                       # (batch, num_heads, positions, head_dim)

        # recombine the heads: (batch, num_heads, positions, head_dim) -> (batch, positions, embed_dim)
        out = out.transpose(1, 2).contiguous().view(batch, positions, self.embed_dim)

        # final learned mix across the combined heads
        out = self.out_proj(out)                    # (batch, positions, embed_dim)
        out = self.resid_dropout(out)
        return out
