from gpt2.model.config import GPTConfig


# our little letters model
config = GPTConfig(
    vocab_size=26,
    context_length=64,
    embed_dim=128,
    num_layers=2,
    num_heads=4
)

print(config)
print("width per head:", config.embed_dim // config.num_heads)