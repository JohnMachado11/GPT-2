"""Watch ONE small input flow through every stage of attention, with real numbers.

Small on purpose (embed_dim=8, 2 heads, 4 tokens) so every value is readable. This
re-creates each stage of CausalSelfAttention.forward step by step and prints it, then
confirms the hand-walk matches the real forward() at the end.
"""
import torch
from gpt2.model.attention import CausalSelfAttention

torch.manual_seed(0)
torch.set_printoptions(precision=2, sci_mode=False)

embed_dim, num_heads, positions = 8, 2, 4
head_dim = embed_dim // num_heads          # 4

attn = CausalSelfAttention(embed_dim, num_heads)
x = torch.randn(1, positions, embed_dim)   # batch=1, 4 tokens, 8 numbers each


def show(label, t):
    print(f"{label}   shape={tuple(t.shape)}")
    print(t)
    print()


print("=" * 64)
print("STEP 0 — input: 1 sequence, 4 tokens, 8 numbers each")
show("x", x)

print("=" * 64)
print("STEP 1 — qkv projection (8 -> 24), then split into query/key/value")
qkv = attn.qkv(x)
query, key, value = qkv.split(embed_dim, dim=-1)
show("qkv(x)", qkv)
show("query", query)

print("=" * 64)
print("STEP 2 — split each into heads: (1,4,8) -> (1,2,4,4) = [batch, heads, positions, head_dim]")
query = query.view(1, positions, num_heads, head_dim).transpose(1, 2)
key   = key.view(1, positions, num_heads, head_dim).transpose(1, 2)
value = value.view(1, positions, num_heads, head_dim).transpose(1, 2)
show("query (per head)", query)

print("=" * 64)
print("STEP 3 — scores = query . key^T / sqrt(head_dim)   (per head: a 4x4 grid)")
scores = (query @ key.transpose(-2, -1)) / (head_dim ** 0.5)
show("scores, head 0 (row=query pos, col=key pos)", scores[0, 0])

print("=" * 64)
print("STEP 4 — mask the future, then softmax -> attention weights")
mask = torch.tril(torch.ones(positions, positions)).bool()
scores = scores.masked_fill(~mask, float("-inf"))
weights = torch.softmax(scores, dim=-1)
show("weights, head 0 (lower-triangular, rows sum to 1)", weights[0, 0])
print("row sums:", weights[0, 0].sum(dim=-1), "\n")

print("=" * 64)
print("STEP 5 — weights @ value: each position becomes a weighted blend of the values")
out = weights @ value
show("blended (per head)", out)

print("=" * 64)
print("STEP 6 — recombine heads (1,2,4,4)->(1,4,8), then out_proj")
out = out.transpose(1, 2).contiguous().view(1, positions, embed_dim)
out = attn.out_proj(out)
show("final output", out)

print("=" * 64)
print("does this hand-walk match the real forward()? ->", torch.allclose(out, attn(x)))
