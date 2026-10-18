import torch
from transformers import GPT2Config, GPT2LMHeadModel

from gpt2.from_hf import from_hugging_face_model


def test_my_model_matches_hugging_faces_gpt2_exactly():
    # The README's headline claim, tested without downloading anything: build a SMALL, randomly
    # initialized Hugging Face GPT-2, copy its weights into my model with the same code load_gpt2 uses,
    # and compare outputs in float64. Any difference in the architecture -- attention scaling, the
    # causal mask, LayerNorm, GELU, weight tying, a transposed weight -- would show up far above 1e-10.
    torch.manual_seed(0)
    hugging_face_model = GPT2LMHeadModel(GPT2Config(vocab_size=100, n_positions=32, n_embd=32, n_layer=2, n_head=4))
    with torch.no_grad():
        for parameter in hugging_face_model.parameters():
            parameter.normal_(0.0, 0.2)     # EVERY weight random -- Hugging Face's own init sets each LayerNorm
                                            # to exactly 1/0 and each bias to 0, which would hide a mix-up
    hugging_face_model.eval()
    mine = from_hugging_face_model(hugging_face_model)

    hugging_face_model.double()
    mine.double()
    token_ids = torch.randint(0, 100, (2, 16))
    with torch.no_grad():
        expected = hugging_face_model(token_ids).logits
        mine_logits, _ = mine(token_ids)

    assert (mine_logits - expected).abs().max().item() < 1e-10
    assert torch.equal(mine_logits.argmax(-1), expected.argmax(-1))


def test_every_weight_is_copied():
    torch.manual_seed(0)
    hugging_face_model = GPT2LMHeadModel(GPT2Config(vocab_size=100, n_positions=32, n_embd=32, n_layer=2, n_head=4))
    mine = from_hugging_face_model(hugging_face_model)
    # the token embedding (and the tied unembedding) must be Hugging Face's, not fresh random weights
    assert torch.equal(mine.token_embedding.weight, hugging_face_model.transformer.wte.weight)
    assert mine.unembed.weight is mine.token_embedding.weight
