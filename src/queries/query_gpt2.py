from transformers import AutoTokenizer
from gpt2.from_hf import load_gpt2
import torch


# GPT-2 comes in four sizes -- swap the name below to load a bigger one.
# load_gpt2 reads the shape from the checkpoint, so any of these just works:
#   "gpt2"         124M   12 layers,  768 dim, 12 heads   (smallest; weakest -- current)
#   "gpt2-medium"  355M   24 layers, 1024 dim, 16 heads
#   "gpt2-large"   774M   36 layers, 1280 dim, 20 heads
#   "gpt2-xl"      1.5B   48 layers, 1600 dim, 25 heads   (largest; heavy on 16GB)
# Bigger = slower to load but noticeably better answers.
model = load_gpt2("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")   # same BPE tokenizer for all four sizes -- no need to change this

# A catalog of prompts for probing GPT-2 small (124M), grouped by what each one
# REVEALS. The "-> ..." notes are the GREEDY (top_k=1) continuation.
# Remember: GPT-2 is a BASE model -- it CONTINUES text, it doesn't answer
# questions -- so format and pattern matter far more than stored facts.
#
# Uncomment exactly ONE prompt to run it, and use the GREEDY generate call at the
# bottom (the active one) -- every prompt here is a "what comes next?" test, and
# every "-> " note was produced with greedy. Only switch to the sampling call for
# open-ended creative writing.

# ============================================================================
# GIVES A CORRECT / GOOD ANSWER  --  the "it actually works" demos
# ============================================================================
# Few-shot format locks the model into "capital lookup" and it nails it.
# This is the best default: a clean, correct answer right off the bat.
prompt = "France: Paris\nItaly: Rome\nSpain:"           # -> " Madrid" (then "Sweden: Stockholm")

# Pure pattern continuation -- foolproof, unmistakably right.
# prompt = "1, 2, 3, 4, 5, 6,"                           # -> " 7, 8, 9,"

# The one plain factual-recall phrasing that lands (very common wording).
# prompt = "The Eiffel Tower is located in the city of"  # -> " Paris, France."

# Completes the "birds fly, fish ___" analogy correctly.
# prompt = "Birds can fly, fish can"                      # -> " swim" (then rambles)

# Induction / copying: continues the run by copying the earlier pattern.
# prompt = "apple orange banana apple orange"             # -> " banana" (then degenerates)

# ============================================================================
# WRONG, BUT INSTRUCTIVELY WRONG  --  shows GPT-2 small's limits
# ============================================================================
# --- factual recall is weak: the model dodges or invents ---
# prompt = "The capital of France is"                     # -> " the capital of the French Republic" (dodges)
# prompt = "The largest planet in the solar system is"    # -> " about 1.5 billion light" (drifts, wrong)
# prompt = "The chemical symbol for gold is"              # -> ' the "G" in the' (wrong; it's Au)
# prompt = "The first President of the United States was"  # -> " born in 1836. He" (wrong; Washington)
# prompt = "Water freezes at a temperature of"            # -> " about -40 degrees Fahrenheit" (wrong; 32F)

# --- question form is the WORST framing for a base model ---
# prompt = "Brazilians speak what language? "             # -> "They are not fluent in" (fails)

# --- arithmetic fails all along the ladder (the classic GPT-2 wall) ---
# prompt = "2 + 2 ="                                      # -> " 3.5 + 3."
# prompt = "7 + 5 ="                                      # -> " 1.5"
# prompt = "12 + 15 ="                                    # -> " 0.5"
# prompt = "47 + 58 ="                                    # -> " 0.9"
# prompt = "128 + 371 ="                                  # -> " 0.9"

# --- words instead of digits doesn't rescue it ---
# prompt = "Two plus two equals"                          # -> " one." (wrong)

# ============================================================================
# PARTIAL  --  the FORMAT survives but the CONTENT is wrong
# ============================================================================
# Few-shot sums: it copies the "N + N =" layout but not the actual math.
# prompt = "2 + 3 = 5\n4 + 1 = 5\n6 + 2 ="               # -> " 5" (6 + 2 != 5)
# prompt = "12 + 15 = 27\n31 + 24 = 55\n47 + 58 ="       # -> " 55" (copies a prior answer)

# ============================================================================
# WEAK / REDUNDANT  --  kept for reference; the weakest of the set
# ============================================================================
# --- analogies read clever but mostly come out wrong (greedy) ---
# prompt = "Paris is to France as Tokyo is to"           # -> " London." (should be Japan)
# prompt = "A puppy is a young dog. A kitten is a young"  # -> " dog." (should be cat!)
# prompt = "Hot is to cold as up is to"                  # -> " hot." (should be down)
# prompt = "Dogs are furry, fish are"                    # -> " fish, and dogs are dogs" (degenerate)

# --- same arithmetic failure as the + ladder, so redundant with it ---
# prompt = "3 x 4 ="                                      # -> " 1.5 x 4 ="
# prompt = "10 - 4 ="                                     # -> " 1.5"

# --- other Brazil framings: both dodge (neither says "Portuguese") ---
# prompt = "Brazilians speak the language of"             # -> " the people, and the people"
# prompt = "The language spoken in Brazil is called"      # -> ' "Brazilian" and is' (wrong)

# --- pure non-answer (was the old default -- swapped out) ---
# prompt = "half of 10 is"                                # -> " a good sign."

token_ids = tokenizer.encode(prompt, return_tensors="pt")

print("=" * 64)
print("INPUT PROMPT")
print("=" * 64)
print(prompt)
print("\ntoken ids:", token_ids[0].tolist())
print("tokens:   ", [tokenizer.decode([t]) for t in token_ids[0].tolist()])

with torch.no_grad():
    # USE THIS (greedy): the single most likely next token each step. Deterministic,
    # and what every "-> " note above was made with. Best for all prompts here.
    output = model.generate(token_ids, max_new_tokens=10, top_k=1)

    # ALTERNATIVE (sampling): random, more creative continuations -- only for
    # open-ended writing (a story, a paragraph), NOT for checking a right answer.
    # To use it, comment out the line above and uncomment this one:
    #   output = model.generate(token_ids, max_new_tokens=30, temperature=0.8, top_k=40)

print("\n" + "=" * 64)
print("OUTPUT FROM MODEL")
print("=" * 64)
print("token ids:", output[0].tolist())

prompt_length = token_ids.shape[1]
generated_ids = output[0][prompt_length:].tolist()
first_new_text = tokenizer.decode([generated_ids[0]])

print("\ntext  (>>> marks where the model starts; [ ] is its first new token):")
print(
    tokenizer.decode(output[0][:prompt_length])
    + ">>>[" + first_new_text + "]"
    + tokenizer.decode(generated_ids[1:])
)
print("\nfirst new token: id", generated_ids[0], "->", repr(first_new_text))

print("\nnewly generated only (what the model ADDED, minus the echoed prompt):")
print("token ids:", generated_ids)
print("text:     ", repr(tokenizer.decode(generated_ids)))
