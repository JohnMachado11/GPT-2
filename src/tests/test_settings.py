import ast
from pathlib import Path

from run import settings

src = Path(__file__).parent.parent
train_on_text = src / "run" / "train_on_text.py"
count_tokens = src / "run" / "count_tokens.py"
query_trained = src / "queries" / "query_trained.py"

# count_tokens.py's own settings: they shape its suggestions, never a run
suggestion_only = {"MODE", "minimum_embed_dim", "minimum_viable_epochs"}


def defined():
    return {name for name in vars(settings) if not name.startswith("_")}


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
