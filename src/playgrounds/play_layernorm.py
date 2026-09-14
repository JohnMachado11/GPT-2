import torch
from gpt2.model.layernorm import LayerNorm


# print tensors cleanly: 3 decimals, no scientific notation
torch.set_printoptions(precision=3, sci_mode=False)

def show(label, t):
    print(label)
    print("     values:", t.squeeze())
    print(f"     mean = {t.mean():.3f}   std = {t.std(unbiased=False):.3f}\n")

# one small, readable row (8 numbers) so every value is visible
x = torch.tensor([[2.0, 4.0, 4.0, 6.0, 10.0, 0.0, 8.0, 2.0]])

print("=== LayerNorm forward, stage by stage ===\n")
show("1) input row", x)

mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)

centered = x - mean
show("2) after (x - mean)", centered)

normalized = centered / torch.sqrt(var + 1e-5)
show("3) after / sqrt(var + eps)", normalized)

# stage 4: the learnable knobs. at defaults (scale=1, shift=0) they change nothing,
# so set them to visible values to see what they actually do:
ln = LayerNorm(embed_dim=8)
with torch.no_grad():
    ln.scale.fill_(2.0)     # stretch the spread by 2
    ln.shift.fill_(10.0)    # then move the center up to 10
show("4) * scale + shift    (scale=2, shift=10)", ln(x))