import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, embed_dim, epsilon=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))
        self.epsilon = epsilon
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.epsilon)
        return normalized * self.scale + self.shift