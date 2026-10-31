import ast
from pathlib import Path

import pytest

from run import settings, settings_gpt2_small, settings_gpt2_medium

src = Path(__file__).parent.parent
train_on_text = src / "run" / "train_on_text.py"
count_tokens = src / "run" / "count_tokens.py"
query_trained = src / "queries" / "query_trained.py"
train_gpt2 = src / "run" / "train_gpt2.py"
compare_to_openai = src / "run" / "compare_to_openai.py"

# count_tokens.py's own settings: they shape its suggestions, never a run
suggestion_only = {"MODE", "minimum_embed_dim", "minimum_viable_epochs"}


def defined(module=settings):
    return {name for name in vars(module) if not name.startswith("_")}


def read_by(*scripts):
    # every settings.<name> these scripts use
    return {node.attr for script in scripts for node in ast.walk(ast.parse(script.read_text()))
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "settings"}


def test_every_setting_reaches_a_run():
    # a setting train_on_text.py / query_trained.py don't read (say, a value typed back into the script)
    # would quietly do nothing when you change it
    assert defined() - read_by(train_on_text, query_trained) == suggestion_only


def test_every_setting_a_script_reads_exists():
    # a misspelled one would only fail when the script reaches it -- for the sample, after the whole training run
    assert read_by(train_on_text, count_tokens, query_trained) - defined() == set()


@pytest.mark.parametrize("gpt2_settings", [settings_gpt2_small, settings_gpt2_medium])
def test_every_gpt2_setting_is_read_and_exists(gpt2_settings):
    # each GPT-2 settings file, both checks at once: every setting is read by the two scripts, and they read
    # nothing it doesn't define -- so either size runs to the end without a missing setting
    assert defined(gpt2_settings) == read_by(train_gpt2, compare_to_openai)


@pytest.mark.parametrize("gpt2_settings", [settings_gpt2_small, settings_gpt2_medium])
def test_every_gpt2_size_keeps_524288_tokens_per_step(gpt2_settings):
    # the GPT-3 paper's batch for both sizes; "halve batch_size and double accumulation_steps" must keep it
    assert gpt2_settings.batch_size * gpt2_settings.accumulation_steps * gpt2_settings.context_length == 524_288
