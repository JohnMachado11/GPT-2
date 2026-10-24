import argparse
import importlib.util
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from gpt2.checkpoint import save_best
from gpt2.from_hf import from_hugging_face_model, load_gpt2, to_hugging_face_model
from gpt2.model.config import GPTConfig
from gpt2.model.gpt import GPT


@pytest.mark.parametrize("fast_attention", [False, True])
def test_my_model_matches_hugging_faces_gpt2_exactly(fast_attention):
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
    for block in mine.blocks:
        block.attn.fast = fast_attention

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


def random_tiny_model():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=100, context_length=32, embed_dim=32, num_layers=2, num_heads=4))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.2)     # every weight random, so a mix-up can't hide behind 1s and 0s
    return model.eval()


def test_my_weights_come_back_unchanged_from_hugging_face_format():
    # the other direction (to_hugging_face_model, which export_to_hf.py uses), then back again
    mine = random_tiny_model()
    back = from_hugging_face_model(to_hugging_face_model(mine))
    assert all(torch.equal(a, b) for a, b in zip(mine.state_dict().values(), back.state_dict().values()))


def test_hugging_face_format_gives_my_models_outputs():
    mine = random_tiny_model().double()
    hugging_face_model = to_hugging_face_model(mine).double()
    token_ids = torch.randint(0, 100, (2, 16))
    with torch.no_grad():
        assert (hugging_face_model(token_ids).logits - mine(token_ids)[0]).abs().max().item() < 1e-10
    assert hugging_face_model.lm_head.weight.data_ptr() == hugging_face_model.transformer.wte.weight.data_ptr()


def test_export_saves_a_model_transformers_can_load(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("export_to_hf", Path(__file__).parents[1] / "run" / "export_to_hf.py")
    export_to_hf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_to_hf)

    mine = random_tiny_model()
    save_best(tmp_path / "run", mine, step=0, val_loss=1.0)
    export_to_hf.main(argparse.Namespace(run_dir=str(tmp_path / "run"), out=None))
    assert "same outputs as your model" in capsys.readouterr().out

    loaded = GPT2LMHeadModel.from_pretrained(tmp_path / "run" / "hf").eval()      # plain transformers
    token_ids = torch.randint(0, 100, (2, 16))
    with torch.no_grad():
        assert (loaded(token_ids).logits - mine(token_ids)[0]).abs().max().item() < 1e-4
    assert (tmp_path / "run" / "hf" / "tokenizer.json").exists()
    # and back into this repo's readable model, as the README says
    assert all(torch.equal(a, b) for a, b in zip(mine.state_dict().values(), load_gpt2(tmp_path / "run" / "hf").state_dict().values()))


def test_export_accepts_the_pt_file_itself(tmp_path):
    spec = importlib.util.spec_from_file_location("export_to_hf", Path(__file__).parents[1] / "run" / "export_to_hf.py")
    export_to_hf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_to_hf)
    save_best(tmp_path / "run", random_tiny_model(), step=0, val_loss=1.0)
    export_to_hf.main(argparse.Namespace(run_dir=str(tmp_path / "run" / "best.pt"), out=None))
    assert (tmp_path / "run" / "hf" / "model.safetensors").exists()      # saved beside the file, not inside it
