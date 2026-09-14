import torch
import torch.nn.functional as F
from gpt2.model.feedforward import FeedForward


torch.set_printoptions(precision=3, sci_mode=False)

# GELU's "gate" on a spread of values
vals = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
print("input:", vals)
print("GELU: ", F.gelu(vals, approximate="tanh"))
print()

# the whole feed-forward: same shape in, same shape out
ff = FeedForward(embed_dim=8)
x = torch.randn(1, 4, 8)
print("x shape:  ", tuple(x.shape))
print("out shape:", tuple(ff(x).shape), " <- same as input, ready for the residual stream")