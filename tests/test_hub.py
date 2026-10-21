"""Hub fetches must pin revisions; nothing may track an implicit branch."""

import pytest

from jev_qwen import tokenize as T
from jev_qwen.data import DATASET_NAME, DATASET_REVISION, load_questions
from jev_qwen.hub import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    resolve_dataset_revision,
    resolve_model_revision,
)


def test_default_revisions_are_commit_shas():
    assert len(DEFAULT_MODEL_REVISION) == 40
    assert len(DATASET_REVISION) == 40
    int(DEFAULT_MODEL_REVISION, 16)
    int(DATASET_REVISION, 16)


def test_unknown_model_requires_explicit_revision():
    with pytest.raises(ValueError, match="no pinned revision"):
        resolve_model_revision("someone/other-model")
    assert resolve_model_revision("someone/other-model", "a" * 40) == "a" * 40


def test_unknown_dataset_requires_explicit_revision():
    with pytest.raises(ValueError, match="no pinned revision"):
        resolve_dataset_revision("someone/other-dataset")


def test_tokenizer_is_fetched_at_the_pinned_revision(monkeypatch):
    import transformers

    seen = {}

    def fake_from_pretrained(name, revision=None, **kwargs):
        seen["name"], seen["revision"] = name, revision
        return object()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    old_tok, old_key = T._tokenizer, T._tokenizer_key
    T._tokenizer, T._tokenizer_key = None, None
    try:
        T.get_tokenizer(DEFAULT_MODEL_NAME)
        assert seen["name"] == DEFAULT_MODEL_NAME
        assert seen["revision"] == DEFAULT_MODEL_REVISION
    finally:
        T._tokenizer, T._tokenizer_key = old_tok, old_key


def test_dataset_is_loaded_at_the_pinned_revision(monkeypatch):
    import sys
    import types

    seen = {}

    class FakeDS:
        column_names = []

        def __len__(self):
            return 100

        def shuffle(self, seed=0):
            return self

        def select(self, indices):
            return self

        def map(self, *args, **kwargs):
            return self

    def fake_load_dataset(name, subset=None, split=None, revision=None, **kwargs):
        seen.update(name=name, revision=revision)
        return FakeDS()

    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_module)
    load_questions("train", 1, seed=0, subset="all")
    assert seen["name"] == DATASET_NAME
    assert seen["revision"] == DATASET_REVISION


def test_engine_loads_tokenizer_from_checkpoint_revision(monkeypatch):
    import torch

    from jev_qwen import infer

    class FakeNet:
        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeTok:
        pad_token_id = None
        pad_token = None
        eos_token = None

    seen = {}

    def fake_load(ckpt, device):
        return FakeNet(), {
            "model_name": DEFAULT_MODEL_NAME,
            "model_revision": "b" * 40,
        }

    def fake_get(name, revision=None):
        seen["name"], seen["revision"] = name, revision
        return FakeTok()

    monkeypatch.setattr(infer, "load_checkpoint", fake_load)
    monkeypatch.setattr(infer, "get_tokenizer", fake_get)
    infer.DecisionEngine(checkpoint="unused", device=torch.device("cpu"))
    assert seen == {"name": DEFAULT_MODEL_NAME, "revision": "b" * 40}
