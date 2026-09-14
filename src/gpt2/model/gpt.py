import torch
import torch.nn as nn
from gpt2.model.block import Block
from gpt2.model.layernorm import LayerNorm
import torch.nn.functional as F
import math


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embed_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.final_norm = LayerNorm(config.embed_dim)
        self.unembed = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.unembed.weight = self.token_embedding.weight   # weight tying
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("contract.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):                        # a "multiply + add" layer (qkv, out_proj, feed-forward, unembed)
            nn.init.normal_(module.weight, mean=0.0, std=0.02)   # small random weights so logits start small
            if module.bias is not None:                          # unembed has no bias, so guard for it
                nn.init.zeros_(module.bias)                      # start the offset term at zero
        elif isinstance(module, nn.Embedding):                   # a lookup table (token & position embeddings)
            nn.init.normal_(module.weight, mean=0.0, std=0.02)   # same small random start

    def forward(self, token_ids, targets=None):
        batch, positions = token_ids.shape
        position_indices = torch.arange(positions, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(position_indices)    # start the residual stream
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x)                                                # each block reads & adds
        x = self.final_norm(x)
        logits = self.unembed(x)                                        # (batch, positions, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            cropped_ids = token_ids[:, -self.config.context_length:]
            logits, _ = self(cropped_ids)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat((token_ids, next_id), dim=1)
        return token_ids
