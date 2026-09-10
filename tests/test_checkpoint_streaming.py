"""逐张量保存与标准safetensors、旧格式及原子失败的兼容性。"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from gemma4_posttrain_jax import checkpoint


def test_streamed_state_is_readable_by_standard_library_and_legacy_loader(tmp_path):
    state = {
        "bf16": np.asarray([0.25, -1.5, 3], dtype=ml_dtypes.bfloat16),
        "empty": np.empty((3, 0, 7), dtype=np.float32),
        "bool": np.asarray([True, False]),
        "scalar": np.asarray(9, dtype=np.int32),
        "unicode键": np.asfortranarray(np.arange(20, dtype=np.float32).reshape(4, 5)),
        "slice": np.arange(21, dtype=np.int16)[::3],
    }
    target = checkpoint.save_train_state(state, tmp_path / "streamed", metadata={"completed_step": 9})
    names, leaves, _ = checkpoint._flatten_named(state)
    reference = {name: np.asarray(value, order="C") for name, value in zip(names, leaves, strict=True)}
    loaded = load_file(target / "state.safetensors")
    assert loaded.keys() == reference.keys()
    for name in names:
        np.testing.assert_array_equal(loaded[name], reference[name])
        assert loaded[name].dtype == reference[name].dtype
        assert loaded[name].shape == reference[name].shape

    # 旧库生成的文件继续由同一个恢复入口读取；新writer不引入格式版本变更。
    save_file(reference, target / "state.safetensors")
    template = jax.tree.map(lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype), state)
    restored = checkpoint.load_train_state(target, template)
    for expected, actual in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert checkpoint.read_checkpoint_metadata(target)["version"] == 1


def test_save_does_not_delete_or_change_live_jax_state(tmp_path):
    state = {"matrix": jnp.arange(96, dtype=jnp.float32).reshape(8, 12), "step": jnp.asarray(3, dtype=jnp.int32)}
    path = checkpoint.save_train_state(state, tmp_path / "state", metadata={})
    template = jax.tree.map(lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype), state)
    restored = checkpoint.load_train_state(path, template)
    for name in state:
        assert not state[name].is_deleted()
        np.testing.assert_array_equal(jax.jit(lambda x: x + 1)(state[name]), restored[name] + 1)


def test_write_failure_keeps_checkpoint_unpublished_and_existing_state_intact(tmp_path, monkeypatch):
    original = checkpoint._write_leaf
    calls = 0

    def fail_during_second_tensor(file, leaf):
        nonlocal calls
        calls += 1
        if calls == 2:
            file.write(b"partial")
            raise OSError("simulated full filesystem")
        original(file, leaf)

    state = {"a": jnp.arange(8), "b": jnp.arange(8) + 1}
    monkeypatch.setattr(checkpoint, "_write_leaf", fail_during_second_tensor)
    with pytest.raises(OSError, match="simulated"):
        checkpoint.save_train_state(state, tmp_path / "failed", metadata={})
    assert list(tmp_path.iterdir()) == []
    assert not any(value.is_deleted() for value in state.values())

    monkeypatch.setattr(checkpoint, "_write_leaf", original)
    target = checkpoint.save_train_state(state, tmp_path / "complete", metadata={"kept": True})
    original_bytes = (target / "state.safetensors").read_bytes()
    with pytest.raises(FileExistsError):
        checkpoint.save_train_state(state, target, metadata={"kept": False})
    assert (target / "state.safetensors").read_bytes() == original_bytes
    assert json.loads((target / "meta.json").read_text())["metadata"] == {"kept": True}
