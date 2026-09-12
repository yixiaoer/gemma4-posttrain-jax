"""逐张量保存与标准safetensors、旧格式及原子失败的兼容性。"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from safetensors import SafetensorError
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


@pytest.mark.parametrize("corruption", ["shape", "dtype", "duplicate", "missing", "sharding", "truncated"])
def test_restore_rejects_invalid_checkpoint_before_device_allocation(tmp_path, monkeypatch, corruption):
    state = {"a": np.arange(8, dtype=np.float32), "b": np.asarray(2, np.int32)}
    path = checkpoint.save_train_state(state, tmp_path / "state", metadata={})
    template = jax.tree.map(lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype), state)
    document = checkpoint.read_checkpoint_metadata(path)
    shardings = None
    if corruption == "shape":
        document["leaves"][-1]["shape"] = [1]
    elif corruption == "dtype":
        document["leaves"][-1]["dtype"] = "float32"
    elif corruption == "duplicate":
        document["leaves"].append(document["leaves"][0])
    elif corruption == "missing":
        save_file({"['a']": state["a"]}, path / "state.safetensors")
    elif corruption == "sharding":
        shardings = (jax.sharding.SingleDeviceSharding(jax.devices()[0]),)
    else:
        data = (path / "state.safetensors").read_bytes()
        (path / "state.safetensors").write_bytes(data[:-1])
    (path / "meta.json").write_text(json.dumps(document))

    def no_allocation(*args, **kwargs):
        pytest.fail("invalid checkpoint reached device allocation")

    monkeypatch.setattr(checkpoint.jnp, "asarray", no_allocation)
    monkeypatch.setattr(checkpoint.jax, "device_put", no_allocation)
    with pytest.raises(SafetensorError if corruption == "truncated" else ValueError) as caught:
        checkpoint.load_train_state(path, template, shardings=shardings)
    assert "invalid checkpoint reached" not in str(caught.value)


@pytest.mark.parametrize("format_name", [checkpoint.FORMAT_NAME, *checkpoint.LEGACY_FORMAT_NAMES, "unknown-format"])
def test_current_writer_and_explicit_legacy_reader(tmp_path, format_name):
    state = {"weight": jnp.asarray([1.0, -0.0]), "step": jnp.asarray(3)}
    path = checkpoint.save_train_state(state, tmp_path / "state", metadata={})
    document = checkpoint.read_checkpoint_metadata(path)
    assert document["format"] == "gemma4_posttrain_jax-train-state"
    document["format"] = format_name
    (path / "meta.json").write_text(json.dumps(document))
    if format_name == "unknown-format":
        with pytest.raises(ValueError, match="unsupported checkpoint"):
            checkpoint.load_train_state(path, state)
    else:
        restored = checkpoint.load_train_state(path, state)
        for expected, actual in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
            assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


def test_restore_releases_current_array_when_transfer_wait_fails(tmp_path, monkeypatch):
    state = {"a": np.arange(4, dtype=np.float32), "b": np.arange(6, dtype=np.float32)}
    path = checkpoint.save_train_state(state, tmp_path / "state", metadata={})
    created = []

    class Transfer:
        deleted = False

        def block_until_ready(self):
            if len(created) == 2 and self is created[1]:
                raise RuntimeError("transfer fence failed")
            return self

        def delete(self):
            self.deleted = True

    def put(value):
        result = Transfer()
        created.append(result)
        return result

    # 首个传输成功，第二个设备数组已创建，但在等待结束时失败。
    monkeypatch.setattr(checkpoint.jnp, "asarray", put)
    with pytest.raises(RuntimeError, match="transfer fence failed"):
        checkpoint.load_train_state(path, state)
    assert len(created) == 2
    assert all(value.deleted for value in created)


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.uint64])
def test_restore_rejects_silent_64bit_downcast(tmp_path, monkeypatch, dtype):
    state = {"weight": np.asarray([2**40 + 1], dtype=dtype)}
    path = checkpoint.save_train_state(state, tmp_path / "state", metadata={})

    def no_allocation(*args, **kwargs):
        pytest.fail("unsupported precision reached device allocation")

    with jax.enable_x64(False), monkeypatch.context() as patch:
        patch.setattr(checkpoint.jnp, "asarray", no_allocation)
        patch.setattr(checkpoint.jax, "device_put", no_allocation)
        with pytest.raises(ValueError, match="JAX_ENABLE_X64"):
            checkpoint.load_train_state(path, state)
    with jax.enable_x64(True):
        restored = checkpoint.load_train_state(path, state)
        assert restored["weight"].dtype == np.dtype(dtype)
        assert np.asarray(restored["weight"]).tobytes() == state["weight"].tobytes()
