# Every setting for training a model on your own text, in one place. train_on_text.py, count_tokens.py
# and query_trained.py all read this file -- change values here, not in the scripts.
# (The FineWeb prep keeps its own settings in prepare_data.py: they're baked into its shards.)

# --- the model ---
context_length = 64         # tokens per window: the longest text the model reads at once
embed_dim = 128             # the model's size: grow these three for more capacity
num_layers = 2              # (count_tokens.py suggests them for your amount of text)
num_heads = 2
dropout = 0.1               # regularizes against overfitting. 0.1 is GPT-1's value, at its three dropout spots
                            # (residual, embedding, attention); the GPT-2 paper gives no value of its own, saying
                            # it largely follows GPT-1

# --- the data ---
# Hold out the last 10% of EACH .txt file for validation (the model never trains on it), so we can spot
# overfitting -- if train loss keeps falling while val loss rises, it's memorizing. 0.9 = 90% train / 10% val
# is a good ratio for a SMALL dataset like a few books; with a much bigger corpus you'd raise it -- the FineWeb run
# uses 0.99.
train_fraction = 0.9
batch_size = 32             # windows per batch, for training and for validation

# --- training ---
# Fix every random draw in train_on_text.py: starting weights, batch windows, dropout, the sample.
# Same seed + same settings = same model.
seed = 1337
num_steps = 415             # count_tokens.py suggests this too
max_learning_rate = 1e-3    # the learning rate climbs to this over warmup_steps,
warmup_steps = 50
min_learning_rate = 1e-4    # then eases down to this by the last step
max_gradient_norm = 1.0     # cap the total gradient size; GPT-3 paper's value, nanoGPT's GPT-2 retrain uses it
weight_decay = 0.1          # GPT-3 paper's value; nanoGPT's GPT-2 retrain
momentum_decay = 0.9        # AdamW's "beta1"
bumpiness_decay = 0.999     # AdamW's "beta2"
accumulation_steps = 1      # add up this many batches' gradients before each update, for a bigger batch
                            # than fits in memory (1 = off)

# --- how often ---
eval_interval = 100         # check the validation loss every this many steps (and at the last one)
checkpoint_interval = 100   # save last.pt every this many steps (and at the end)
print_interval = 1          # print a progress line every this many steps (and at the last one)

# --- two speed switches: they change how fast it trains, not what it learns ---
fast_attention = False      # True: the same math, computed faster (see attention.py)
compile_model = False       # True: torch.compile -- faster on NVIDIA GPUs, after ~1 min of compiling (not on a Mac GPU)

# --- continuing a run ---
# None = a new run. To continue an interrupted run, put its folder name here instead
# (e.g. "gpt2_L2_D128_6.8M_20261018-101500") -- train() picks up from its last.pt.
resume_run = None

# --- the sample written after training, and by query_trained.py ---
prompt = "The Martians"
max_new_tokens = 40
top_k = 20                  # pick each next token from the top_k likeliest

# --- only count_tokens.py reads these: how it suggests a model size from your token count ---
# Two philosophies -- set MODE to pick one:
#
#   "minimum_viable"   -> the smallest usable model, over-trained a fixed number of epochs.
#                         Best when your text is small.
#
#   "chinchilla_exact" -> the compute-optimal size for your token count (~20 training tokens per
#                         parameter), trained about one epoch. Grows with your data; recovers GPT-2
#                         small's shape (768 / 12 / 12) at ~1.7B tokens -- the efficient size for that
#                         data, not GPT-2's actual training amount.
MODE = "minimum_viable"

minimum_embed_dim = 128     # minimum_viable's width (a multiple of 64); layers = heads = width / 64

# minimum_viable_epochs: how many times to read through your whole text. In minimum_viable, this sets
# the suggested step count:
#
#   num_steps = minimum_viable_epochs x total_tokens / tokens per step
#             =          10           x    84,996    / (batch_size x accumulation_steps x context_length)
#             =          10           x    84,996    / (    32     x         1          x       64      )
#             =          10           x    84,996    /                    2,048
#             =   415 steps            (~41 steps = the book's 85k tokens once; training reads
#                                         only the 90% train split, so that's ~11 passes over it)
minimum_viable_epochs = 10
