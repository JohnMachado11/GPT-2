# Every setting for training GPT-2 small (124M) from scratch on FineWeb-Edu, in one place.
# train_gpt2_small.py and compare_to_openai.py read this file -- change values here, not in the scripts.
# (The two folders -- where the shards are, and where checkpoints go -- are command-line flags instead,
# since they differ from machine to machine.)
# The recipe: GPT-2 small's shape, trained with the GPT-3 paper's settings for its 125M model.

# --- the model: GPT-2 small ---
context_length = 1024
embed_dim = 768
num_layers = 12
num_heads = 12
dropout = 0.0               # one pass over 10B fresh tokens never shows the model the same text twice, so
                            # there's little to overfit; nanoGPT and llm.c use 0.0 here (GPT-1 used 0.1; the
                            # GPT-2 and GPT-3 papers don't give a value)

# --- the batch: 524,288 tokens per step (the GPT-3 paper's, for 125M) ---
#   tokens per step = batch_size x accumulation_steps x context_length = 16 x 32 x 1024 = 524,288
batch_size = 16             # windows per micro-batch: 16 fits in an A100 40GB's memory.
accumulation_steps = 32     # Out of memory? Halve batch_size and double this, to keep 524,288

# --- training ---
# There's no num_steps: the run is ONE pass over the training shards, worked out from the data
# (about 18,900 steps for FineWeb-Edu's 10B tokens).
# Fix the random draws: the starting weights and the sample. (On an NVIDIA GPU, two runs can still differ
# in the last decimals.)
seed = 1337
max_learning_rate = 6e-4    # the learning rate climbs to this over warmup_steps (GPT-3 paper, 125M),
warmup_steps = 715          # = GPT-3's 375M warmup tokens / 524,288 tokens per step
min_learning_rate = 6e-5    # then eases down to this, 10% of the max, by the last step
max_gradient_norm = 1.0     # cap the total gradient size; GPT-3 paper's value
weight_decay = 0.1          # GPT-3 paper's value
momentum_decay = 0.9        # AdamW's "beta1"
bumpiness_decay = 0.95      # AdamW's "beta2": the GPT-3 paper's 0.95 (book runs use the default, 0.999)

# --- how often ---
# Each save replaces the file before it: last.pt (to resume) is ~1.5 GB, best.pt (the weights) ~0.5 GB. On Google
# Drive the replaced copies go to Drive's Trash -- ~65 GB over a full 10B run -- so empty it now and then.
eval_interval = 250         # check the validation loss every this many steps (and at the last one);
                            # best.pt is saved whenever it's the lowest so far
checkpoint_interval = 1000  # save last.pt every this many steps (and at the end): a disconnect loses at most this many
print_interval = 10         # print a progress line every this many steps (and at the last one)

# --- two speed switches: they change how fast it trains, not what it learns ---
fast_attention = True       # True: the same math, computed faster -- FlashAttention on recent NVIDIA GPUs
compile_model = True        # True: torch.compile -- faster on NVIDIA GPUs, after ~1 min of compiling (and a bit
                            # more at the first validation check). Must be False on a Mac GPU: PyTorch's
                            # compiler gives wrong gradients there

# --- the sample written after training ---
prompt = "In this lesson, we will learn"
max_new_tokens = 60
top_k = 50                  # pick each next token from the top_k likeliest
