# How count_tokens.py suggests a model size from your token count.
#
# Two philosophies — set MODE to pick one:
#
#   "minimum_viable"   -> the smallest usable model, over-trained a fixed
#                         number of epochs. Best when your text is small.
#
#   "chinchilla_exact" -> the compute-optimal size for your token count
#                         (~20 training tokens per parameter), trained about
#                         one epoch. Grows with your data; recovers GPT-2
#                         small (768 / 12 / 12) at ~1.7B tokens.

MODE = "minimum_viable"

# Training-window assumptions the suggester uses to turn a token budget into
# a number of steps. These are the source of truth — copy them into train_on_text.py.
batch_size = 32
context_length = 64

# Dropout - GPT-2's actual value (applied at all three dropout spots).
dropout = 0.1

# --- minimum_viable settings ---
minimum_embed_dim = 128       # the floor width (a multiple of 64)
minimum_viable_epochs = 20    # how many times to read through your whole text
