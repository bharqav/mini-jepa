"""Checkpoint format security: safetensors only, explicit legacy migration."""

import importlib.util
from pathlib import Path

import pytest
import torch

from jev_qwen.run_logger import (
    HEAD_WEIGHTS,
    LEGACY_HEAD_WEIGHTS,
    load_head_weights,
    save_head_weights,
)


def _converter_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "convert_checkpoints.py"
    spec = importlib.util.spec_from_file_location("convert_checkpoints", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_head_weights_roundtrip_is_safetensors(tmp_path):
    state = {"a": torch.arange(4, dtype=torch.float32), "b": torch.ones(2)}
    save_head_weights(state, tmp_path / HEAD_WEIGHTS)
    assert (tmp_path / HEAD_WEIGHTS).exists()
    assert not (tmp_path / LEGACY_HEAD_WEIGHTS).exists()
    loaded = load_head_weights(tmp_path)
    assert set(loaded) == set(state)
    assert torch.equal(loaded["a"], state["a"])


def test_loader_refuses_legacy_pickle_without_fallback(tmp_path, monkeypatch):
    torch.save({"a": torch.ones(2)}, tmp_path / LEGACY_HEAD_WEIGHTS)

    def explode(*args, **kwargs):
        raise AssertionError("torch.load must never be used to load a checkpoint")

    monkeypatch.setattr(torch, "load", explode)
    with pytest.raises(RuntimeError, match="convert_checkpoints.py"):
        load_head_weights(tmp_path)


def test_loader_never_invokes_torch_load(tmp_path, monkeypatch):
    save_head_weights({"a": torch.ones(2)}, tmp_path / HEAD_WEIGHTS)

    def explode(*args, **kwargs):
        raise AssertionError("torch.load must never be used to load a checkpoint")

    monkeypatch.setattr(torch, "load", explode)
    assert torch.equal(load_head_weights(tmp_path)["a"], torch.ones(2))


def test_conversion_script_writes_safetensors_and_keeps_old_file(tmp_path):
    mod = _converter_module()
    old = {"w": torch.arange(3, dtype=torch.float32)}
    torch.save(old, tmp_path / LEGACY_HEAD_WEIGHTS)
    out = mod.convert(tmp_path)
    assert out.name == HEAD_WEIGHTS
    assert (tmp_path / LEGACY_HEAD_WEIGHTS).exists()
    assert torch.equal(load_head_weights(tmp_path)["w"], old["w"])

    with pytest.raises(FileExistsError):
        mod.convert(tmp_path)


def test_conversion_script_can_delete_old_pickle(tmp_path):
    mod = _converter_module()
    torch.save({"w": torch.ones(1)}, tmp_path / LEGACY_HEAD_WEIGHTS)
    mod.convert(tmp_path, delete_old=True)
    assert not (tmp_path / LEGACY_HEAD_WEIGHTS).exists()
    assert (tmp_path / HEAD_WEIGHTS).exists()
