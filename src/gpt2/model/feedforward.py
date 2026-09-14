import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embed_dim, bias=True, dropout=0.0):
        super().__init__()
        self.expand = nn.Linear(embed_dim, 4 * embed_dim, bias=bias)
        self.contract = nn.Linear(4 * embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.expand(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        x = self.contract(x)
        x = self.dropout(x)
        return x