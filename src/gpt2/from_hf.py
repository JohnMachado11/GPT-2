"""
Load a real, pretrained GPT-2 from Hugging Face into MY hand-built GPT.

Why this file exists:
  1. Proof: pour GPT-2's trained weights into my modules and the outputs match (~1e-13 in float64,
     ~1e-4 in float32), showing my from-scratch architecture is structurally identical to real GPT-2.
  2. A real, capable model whose every weight & intermediate I can read/edit later (Part 2 surgery).

The only fiddly bits are (a) GPT-2's cryptic names -> my clear names, and (b) four weight matrices
per block that GPT-2 stores transposed (it uses a Conv1D layer, shape (in, out)); my nn.Linear
stores weights as (out, in), so I flip those with .t() when copying.
"""

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from gpt2.model.config import GPTConfig
from gpt2.model.gpt import GPT


_TRANSPOSED = (".attn.c_attn.weight", ".attn.c_proj.weight", ".mlp.c_fc.weight", ".mlp.c_proj.weight")
# what each maps to in my model (i = block number):
#   .attn.c_attn.weight  ->  blocks[i].attn.qkv.weight       (fused query/key/value)
#   .attn.c_proj.weight  ->  blocks[i].attn.out_proj.weight  (attention output mix)
#   .mlp.c_fc.weight     ->  blocks[i].ff.expand.weight      (feed-forward widen)
#   .mlp.c_proj.weight   ->  blocks[i].ff.contract.weight    (feed-forward shrink)

def _to_my_name(gpt2_name):
    """Translate one Hugging Face GPT-2 weight name into the match name in my model."""

    if gpt2_name == "lm_head.weight":
        return None
    name = gpt2_name
    name = name.replace("transformer.wte.weight", "token_embedding.weight")
    name = name.replace("transformer.wpe.weight", "position_embedding.weight")
    name = name.replace("transformer.ln_f.weight", "final_norm.scale")
    name = name.replace("transformer.ln_f.bias", "final_norm.shift")
    # everything below renames what's INSIDE a block (transformer.h.{i}...)
    name = name.replace("transformer.h.", "blocks.")
    name = name.replace(".ln_1.weight", ".norm_before_attn.scale")
    name = name.replace(".ln_1.bias", ".norm_before_attn.shift")
    name = name.replace(".ln_2.weight", ".norm_before_ff.scale")
    name = name.replace(".ln_2.bias", ".norm_before_ff.shift")
    name = name.replace(".attn.c_attn.", ".attn.qkv.")
    name = name.replace(".attn.c_proj.", ".attn.out_proj.")
    name = name.replace(".mlp.c_fc.", ".ff.expand.")
    name = name.replace(".mlp.c_proj.", ".ff.contract.")
    return name


def load_gpt2(model_name="gpt2"):
    """Build my GPT at GPT-2's size and copy the pretrained weights in. Returns my GPT in eval mode."""
    return from_hugging_face_model(GPT2LMHeadModel.from_pretrained(model_name))


def from_hugging_face_model(hugging_face_model):
    """Build my GPT with the same shape as a Hugging Face GPT-2 model and copy its weights in.
    Split out of load_gpt2 so the copy can be tested offline, on a small randomly-initialized model."""
    pretrained_weights = hugging_face_model.state_dict()

    config = GPTConfig(
        vocab_size=hugging_face_model.config.vocab_size,          # 50257
        context_length=hugging_face_model.config.n_positions,     # 1024
        embed_dim=hugging_face_model.config.n_embd,               # 768
        num_layers=hugging_face_model.config.n_layer,             # 12
        num_heads=hugging_face_model.config.n_head                # 12
    )
    model = GPT(config)
    my_params = dict(model.named_parameters())

    copied = 0
    with torch.no_grad():
        for gpt2_name, tensor in pretrained_weights.items():
            my_key = _to_my_name(gpt2_name)
            if my_key is None:
                continue
            if any(gpt2_name.endswith(suffix) for suffix in _TRANSPOSED):
                tensor = tensor.t()
            my_params[my_key].copy_(tensor)
            copied += 1

    assert copied == len(my_params), f"copied {copied}, but my model has {len(my_params)} params"
    model.eval()
    return model


def to_hugging_face_model(model):
    """The other direction: a Hugging Face GPT2LMHeadModel holding MY model's weights -- the same names table and
    the same four flipped matrices, run backwards. Its save_pretrained() writes a model anyone can load with
    transformers' from_pretrained(), or with load_gpt2() above."""
    hugging_face_model = GPT2LMHeadModel(GPT2Config(
        vocab_size=model.config.vocab_size,
        n_positions=model.config.context_length,
        n_embd=model.config.embed_dim,
        n_layer=model.config.num_layers,
        n_head=model.config.num_heads,
        activation_function="gelu_new",         # GPT-2's tanh GELU, as in feedforward.py
        resid_pdrop=model.config.dropout, embd_pdrop=model.config.dropout, attn_pdrop=model.config.dropout,
    ))
    my_weights = model.state_dict()

    copied = 0
    with torch.no_grad():
        for gpt2_name, tensor in hugging_face_model.state_dict().items():
            my_key = _to_my_name(gpt2_name)
            if my_key is None:
                continue                        # lm_head: tied to the token embedding, so it's already filled
            mine = my_weights[my_key]
            if any(gpt2_name.endswith(suffix) for suffix in _TRANSPOSED):
                mine = mine.t()
            tensor.copy_(mine)
            copied += 1

    assert copied == len(dict(model.named_parameters())), f"copied {copied} of my model's weights"
    hugging_face_model.eval()
    return hugging_face_model
