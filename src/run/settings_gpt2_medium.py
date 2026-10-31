# Every setting for training GPT-2 medium (355M) from scratch on FineWeb-Edu, in one place.
# train_gpt2.py and compare_to_openai.py read this file when given --size medium -- change values here, not in the scripts.
# (The two folders -- where the shards are, and where checkpoints go -- are command-line flags instead,
# since they differ from machine to machine.)
# The recipe: GPT-2 medium's shape, trained with the GPT-3 paper's settings for its 350M model, on the same data as
# small. Only the shape, the learning rates, the batch split (8 x 64, not 16 x 32), how often to check and save, and
# the OpenAI model it's compared against differ from settings_gpt2_small.py.

# --- the model: GPT-2 medium ---
context_length = 1024
embed_dim = 1024
num_layers = 24
num_heads = 16
dropout = 0.0               # one pass over 10B fresh tokens never shows the model the same text twice, so
                            # there's little to overfit; nanoGPT and llm.c use 0.0 here (GPT-1 used 0.1; the
                            # GPT-2 and GPT-3 papers don't give a value)

# --- the batch: 524,288 tokens per step (the GPT-3 paper's, for 350M too) ---
#   tokens per step = batch_size x accumulation_steps x context_length = 8 x 64 x 1024 = 524,288
batch_size = 8              # windows per micro-batch: half of small's 16, since each window takes more memory in the
                            # bigger model. Expected to fit an A100 40GB, not yet tried there.
accumulation_steps = 64     # Out of memory? Halve batch_size and double this, to keep 524,288

# --- training ---
# There's no num_steps: the run is ONE pass over the training shards, worked out from the data
# (18,795 steps for FineWeb-Edu's 10B tokens, one more than small's 18,794, because the smaller micro-batches leave
# fewer tokens unused at the end of each shard).
# Fix the random draws: the starting weights and the sample. (On an NVIDIA GPU, two runs can still differ
# in the last decimals.)
seed = 1337
max_learning_rate = 3e-4    # the learning rate climbs to this over warmup_steps (GPT-3 paper, 350M: half of
                            # small's 6e-4, since bigger models need gentler steps),
warmup_steps = 715          # = GPT-3's 375M warmup tokens / 524,288 tokens per step (the same for every size)
min_learning_rate = 3e-5    # then eases down to this, 10% of the max, by the last step
max_gradient_norm = 1.0     # cap the total gradient size; GPT-3 paper's value
weight_decay = 0.1          # GPT-3 paper's value
momentum_decay = 0.9        # AdamW's "beta1"
bumpiness_decay = 0.95      # AdamW's "beta2": the GPT-3 paper's 0.95 (book runs use the default, 0.999)

# --- how often ---
# Each save replaces the file before it: last.pt (to resume) is ~4.3 GB, best.pt (the weights) ~1.4 GB. On Google
# Drive the replaced copies go to Drive's Trash -- up to ~210 GB over a full 10B run, about 5 GB an hour -- so empty it
# every few hours.
eval_interval = 500         # check the validation loss every this many steps (and at the last one);
                            # best.pt is saved whenever it's the lowest so far. Half as often as small, for half as
                            # many best.pt saves
checkpoint_interval = 500   # save last.pt every this many steps (and at the end): a disconnect loses at most this many.
                            # Half of small's 1000, since medium's steps take longer (500 is roughly an hour on an A100)
print_interval = 10         # print a progress line every this many steps (and at the last one)

# --- two speed switches: they change how fast it trains, not what it learns ---
fast_attention = True       # True: the same math, computed faster -- FlashAttention on recent NVIDIA GPUs
compile_model = True        # True: torch.compile -- faster on NVIDIA GPUs, after some compiling (~1 min for small,
                            # likely longer for medium's 24 layers; a bit more at the first validation check).
                            # Must be False on a Mac GPU: PyTorch's compiler gives wrong gradients there

# --- the sample written after training ---
prompt = "In this lesson, we will learn"
max_new_tokens = 60
top_k = 50                  # pick each next token from the top_k likeliest

# --- the comparison ---
openai_model = "gpt2-medium"    # OpenAI's model of this size, on Hugging Face: compare_to_openai.py scores against it
