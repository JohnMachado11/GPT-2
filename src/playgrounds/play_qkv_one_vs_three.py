"""Is the fused qkv net (one Linear) the same as three separate Linears?

Two demos, both with tiny readable numbers (embed_dim=4):

  PART A — the three separate nets and the one fused net produce the SAME q/k/v,
           because the fused weight matrix is just the three glued side by side.
  PART B — during backprop, the gradient stays inside its own block. Make the loss
           depend on ONLY the query and the key/value weights get exactly zero.
           Proof that "one net" does not blend the learning of q/k/v together.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_printoptions(precision=2, sci_mode=False)

embed_dim = 4
x = torch.randn(1, 2, embed_dim)   # batch=1, 2 tokens, 4 numbers each


print("=" * 64)
print("PART A — three separate nets vs one fused net: same numbers?")
print("=" * 64)

# three separate nets (the video's picture / the old code)
W_q = nn.Linear(embed_dim, embed_dim, bias=False)
W_k = nn.Linear(embed_dim, embed_dim, bias=False)
W_v = nn.Linear(embed_dim, embed_dim, bias=False)

q_a, k_a, v_a = W_q(x), W_k(x), W_v(x)

# one fused net (the new code). glue the three weight matrices together so the
# fused net IS the three nets. nn.Linear weight is (out, in), so stack on dim 0.
W_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
with torch.no_grad():
    W_qkv.weight.copy_(torch.cat([W_q.weight, W_k.weight, W_v.weight], dim=0))

qkv = W_qkv(x)
q_b, k_b, v_b = qkv.split(embed_dim, dim=-1)

print("query matches:", torch.allclose(q_a, q_b))
print("key   matches:", torch.allclose(k_a, k_b))
print("value matches:", torch.allclose(v_a, v_b))
print("\nquery from 3 nets:\n", q_a)
print("query from split :\n", q_b)


print()
print("=" * 64)
print("PART B — backprop: does the gradient stay in its own block?")
print("=" * 64)

# fresh fused net, this time we care about the gradient
W_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
qkv = W_qkv(x)
q, k, v = qkv.split(embed_dim, dim=-1)

# pretend the loss depends on ONLY the query (ignore key & value entirely)
loss = q.sum()
loss.backward()

# weight is (12, 4): rows 0-3 = query, 4-7 = key, 8-11 = value
g = W_qkv.weight.grad
print("grad on QUERY block (rows 0-3)  -> real numbers, these weights update:")
print(g[0:4])
print("\ngrad on KEY block (rows 4-7)    -> all zero, untouched:")
print(g[4:8])
print("\ngrad on VALUE block (rows 8-11) -> all zero, untouched:")
print(g[8:12])
print("\nkey/value grads all zero? ->", bool((g[4:12] == 0).all()))
