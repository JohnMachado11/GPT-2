import torch
import torch.nn as nn


torch.manual_seed(0)
torch.set_printoptions(precision=2, sci_mode=False)

embed_dim = 4
qkv = nn.Linear(embed_dim, 3 * embed_dim)   # maps 4 -> 12

x = torch.tensor([1.0, 2.0, 3.0, 4.0])
out = qkv(x)
query, key, value = out.split(embed_dim, dim=-1)

print("INPUT    (one token, 4 numbers):", x)
print("OUTPUT of qkv (12 numbers):  ", out)
print("     query:", query)
print("     key:", key)
print("     value:", value)