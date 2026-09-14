from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int         # how many different tokens exist (50257 for GPT-2's BPE vocabulary)
    context_length: int     # the longest sequence the model can read, in tokens
    embed_dim: int          # the width of the residual stream: how many numbers represent each token
    num_layers: int         # how many blocks are stacked on top of each other (the model's depth)
    num_heads: int          # how many attention "viewpoints" each block uses at once
    dropout: float = 0.0    # during training, randomly ignore this fraction of the numbers (0.0 = off)
    bias: bool = True       # include the "+ offset" term in the model's linear steps?

    def __post_init__(self):
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must divide evenly by num_heads ({self.num_heads})"
            )