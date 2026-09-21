"""检查分块保存的完整字节、任意分片方向和读取内存上限。"""

from __future__ import annotations

import io

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from safetensors.numpy import load_file

from gemma4_posttrain_jax import checkpoint


@pytest.mark.parametrize("shape", [(8, 12), (2, 8, 12)])
@pytest.mark.parametrize("layout", ["rows", "columns", "both", "replicated"])
def test_sharded_chunks_preserve_file_bytes_and_bound_device_reads(tmp_path, monkeypatch, shape, layout):
    if len(jax.devices()) < 4:
        pytest.skip("requires four devices")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    tail = {
        "rows": (("x", "y"), None),
        "columns": (None, ("x", "y")),
        "both": ("x", "y"),
        "replicated": (None, None),
    }[layout]
    sharding = NamedSharding(mesh, P(*((None,) * (len(shape) - 2)), *tail))
    expected = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    value = jax.device_put(expected, sharding)
    original_cache = value._npy_value
    get = jax.device_get
    read_sizes = []

    def observe(array):
        read_sizes.append(array.nbytes)
        return get(array)

    monkeypatch.setattr(checkpoint, "_CHECKPOINT_CHUNK_BYTES", 80)
    monkeypatch.setattr(checkpoint.jax, "device_get", observe)
    path = checkpoint.save_train_state({"weight": value}, tmp_path / "chunked", metadata={"step": 2})
    assert read_sizes and max(read_sizes) <= 80
    assert value._npy_value is original_cache
    np.testing.assert_array_equal(load_file(path / "state.safetensors")["['weight']"], expected)
    assert not value.is_deleted()
    np.testing.assert_array_equal(jax.jit(lambda x: x + 1)(value), expected + 1)
    monkeypatch.setattr(checkpoint, "_CHECKPOINT_CHUNK_BYTES", 2**30)
    reference = checkpoint.save_train_state({"weight": value}, tmp_path / "whole", metadata={"step": 2})
    assert (path / "state.safetensors").read_bytes() == (reference / "state.safetensors").read_bytes()
    assert (path / "meta.json").read_bytes() == (reference / "meta.json").read_bytes()


def test_chunks_preserve_all_bfloat16_encodings(tmp_path, monkeypatch):
    bits = np.arange(2**16, dtype=np.uint16).reshape(256, 256)
    value = jax.device_put(bits.view(ml_dtypes.bfloat16))
    monkeypatch.setattr(checkpoint, "_CHECKPOINT_CHUNK_BYTES", 4096)
    path = checkpoint.save_train_state({"bits": value}, tmp_path / "bf16", metadata={})
    loaded = load_file(path / "state.safetensors")["['bits']"]
    np.testing.assert_array_equal(loaded.view(np.uint16), bits)


@pytest.mark.parametrize("dtype", [np.int32, np.bool_, np.complex64])
def test_chunks_preserve_other_supported_device_dtypes(monkeypatch, dtype):
    expected = np.arange(96).reshape(8, 12).astype(dtype)
    value = jax.device_put(expected)
    monkeypatch.setattr(checkpoint, "_CHECKPOINT_CHUNK_BYTES", 64)
    file = io.BytesIO()
    checkpoint._write_leaf(file, value)
    assert file.getvalue() == expected.tobytes(order="C")


def test_chunks_preserve_noncontiguous_host_input(monkeypatch):
    expected = np.arange(600, dtype=np.float32).reshape(20, 30).T[:, ::2]
    monkeypatch.setattr(checkpoint, "_CHECKPOINT_CHUNK_BYTES", 64)
    file = io.BytesIO()
    checkpoint._write_leaf(file, expected)
    assert file.getvalue() == expected.tobytes(order="C")


def test_partial_chunk_write_does_not_publish_checkpoint(tmp_path, monkeypatch):
    original = checkpoint._checkpoint_slice
    count = 0

    def fail(*args):
        nonlocal count
        count += 1
        if count == 3:
            raise OSError("interrupted chunk read")
        return original(*args)

    monkeypatch.setattr(checkpoint, "_CHECKPOINT_CHUNK_BYTES", 64)
    monkeypatch.setattr(checkpoint, "_checkpoint_slice", fail)
    value = jnp.arange(96, dtype=jnp.float32).reshape(8, 12)
    with pytest.raises(OSError, match="interrupted chunk"):
        checkpoint.save_train_state({"weight": value}, tmp_path / "failed", metadata={})
    assert list(tmp_path.iterdir()) == []
    np.testing.assert_array_equal(value, np.arange(96, dtype=np.float32).reshape(8, 12))
