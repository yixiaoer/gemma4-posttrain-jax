"""恢复摘要必须覆盖每个参数/Adam叶，且不能漏掉单叶改变和非有限状态。"""

import numpy as np

from gemma4_posttrain_jax.state_audit import summarize_train_state


def test_complete_state_content_identity_and_nonfinite_detection():
    first = {"params": np.array([[1, 2]], np.float32), "adam": (np.array([0.5], np.float32), np.array(2, np.int32))}
    reference = summarize_train_state(first)
    assert reference["all_finite"] and reference["leaf_count"] == 3
    assert reference["logical_bytes"] == 16
    copied = {"params": first["params"].copy(), "adam": tuple(x.copy() for x in first["adam"])}
    assert summarize_train_state(copied)["content_sha256"] == reference["content_sha256"]
    copied["adam"][0][0] += 0.25
    assert summarize_train_state(copied)["content_sha256"] != reference["content_sha256"]
    copied["adam"][0][0] = np.nan
    assert not summarize_train_state(copied)["all_finite"]


def test_device_summary_reads_temporary_copies_and_preserves_live_state(monkeypatch):
    import jax
    import jax.numpy as jnp

    expected = {"matrix": np.arange(48, dtype=np.float32).reshape(6, 8), "step": np.asarray(2, np.int32)}
    state = jax.tree.map(jnp.asarray, expected)
    source = list(state.values())
    read = []
    original = jax.device_get

    def record(value):
        assert all(value is not leaf for leaf in source)
        read.append(value)
        return original(value)

    with monkeypatch.context() as patch:
        patch.setattr(jax, "device_get", record)
        result = summarize_train_state(state)
    assert result == summarize_train_state(expected)
    assert len(read) == 2 and all(value.is_deleted() for value in read)
    assert all(not value.is_deleted() for value in source)
    np.testing.assert_array_equal(jax.jit(lambda x: x + 1)(state["matrix"]), expected["matrix"] + 1)


def test_summary_checks_tail_blocks_and_empty_arrays():
    values = np.zeros(2**20 + 3, np.float32)
    values[-1] = np.nan
    result = summarize_train_state({"large": values, "empty": np.empty((2, 0), np.float32)})
    assert not result["all_finite"] and result["leaf_count"] == 2
    assert result["leaves"][0]["bytes"] == 0
    assert result["leaves"][1]["nonzero_elements"] == 1


def test_bf16_summary_keeps_signed_zero_and_subnormal_bytes():
    import hashlib

    import ml_dtypes

    bits = np.asarray([0, 0x8000, 1, 0x8001, 0x3F80], np.uint16)
    result = summarize_train_state({"bf16": bits.view(ml_dtypes.bfloat16)})
    record = result["leaves"][0]
    assert record["sha256"] == hashlib.sha256(bits.tobytes()).hexdigest()
    assert record["finite"] and record["dtype"] == "bfloat16" and record["nonzero_elements"] == 3


def test_chunked_summary_matches_full_numpy_content_with_sharding(monkeypatch):
    import jax
    import pytest
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from gemma4_posttrain_jax import checkpoint

    if len(jax.devices()) < 4:
        pytest.skip("requires four devices")
    host = np.arange(192, dtype=np.float32).reshape(2, 8, 12)
    host[0, 0, :4] = [0.0, -0.0, np.inf, np.nan]
    host[-1, -1, -1] = np.nextafter(np.float32(0), np.float32(1))
    expected = summarize_train_state({"params": host, "step": np.asarray(17, np.int32)})
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    value = jax.device_put(host, NamedSharding(mesh, PartitionSpec(None, "x", "y")))
    before_cache = value._npy_value
    original = jax.device_get
    reads = []

    def bounded_read(array):
        assert array is not value and array.nbytes <= 80
        reads.append(array)
        return original(array)

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint, "_CHECKPOINT_CHUNK_BYTES", 80)
        patch.setattr(jax, "device_get", bounded_read)
        actual = summarize_train_state({"params": value, "step": np.asarray(17, np.int32)})
    assert actual == expected
    assert value._npy_value is before_cache and not value.is_deleted()
    assert all(array.is_deleted() for array in reads if isinstance(array, jax.Array))


def test_chunked_summary_keeps_all_bfloat16_encodings(monkeypatch):
    import hashlib

    import jax
    import ml_dtypes

    from gemma4_posttrain_jax import checkpoint

    bits = np.arange(2**16, dtype=np.uint16).reshape(256, 256)
    host = bits.view(ml_dtypes.bfloat16)
    expected = summarize_train_state({"values": host})
    value = jax.device_put(host)
    monkeypatch.setattr(checkpoint, "_CHECKPOINT_CHUNK_BYTES", 4096)
    result = summarize_train_state({"values": value})
    assert result == expected and not result["all_finite"]
    assert result["leaves"][0]["sha256"] == hashlib.sha256(bits.tobytes()).hexdigest()


def test_host_iterator_releases_temporary_on_early_close(monkeypatch):
    import jax
    import jax.numpy as jnp

    from gemma4_posttrain_jax import checkpoint

    value = jnp.arange(12, dtype=jnp.float32)
    original = jax.device_put
    copies = []

    def record(*args, **kwargs):
        array = original(*args, **kwargs)
        copies.append(array)
        return array

    monkeypatch.setattr(jax, "device_put", record)
    blocks = checkpoint.iter_host_array_blocks(value)
    first = next(blocks)
    blocks.close()
    assert len(copies) == 1 and copies[0].is_deleted()
    assert not value.is_deleted()
    np.testing.assert_array_equal(first, np.arange(12, dtype=np.float32))


def test_update_witness_uses_actual_trainable_values_and_survives_mutation():
    from gemma4_posttrain_jax.state_audit import compare_parameter_witness, snapshot_parameter_witness

    params = {"frozen": np.array([5.0], np.float32), "norm": np.array([1.0, 2.0], np.float32)}
    mask = {"frozen": False, "norm": True}
    first = snapshot_parameter_witness(params, mask)
    unchanged = compare_parameter_witness(first, snapshot_parameter_witness(params, mask))
    assert unchanged["all_finite"] and unchanged["changed_elements"] == 0
    params["norm"][1] += 0.25
    params["frozen"][0] += 4
    changed = compare_parameter_witness(first, snapshot_parameter_witness(params, mask))
    assert changed["leaf_count"] == 1 and changed["changed_elements"] == 1
    assert changed["leaves"][0]["max_abs_delta"] == 0.25
    params["norm"][0] = np.nan
    bad = compare_parameter_witness(first, snapshot_parameter_witness(params, mask))
    assert not bad["all_finite"] and bad["leaves"][0]["max_abs_delta"] is None


def test_update_witness_rejects_incompatible_or_absent_observations():
    import pytest

    from gemma4_posttrain_jax.state_audit import compare_parameter_witness, snapshot_parameter_witness

    params = {"x": np.ones(4, np.float32)}
    with pytest.raises(ValueError, match="没有"):
        snapshot_parameter_witness(params, {"x": False})
    with pytest.raises(ValueError, match="参数结构一致"):
        snapshot_parameter_witness(params, {"y": True})
    with pytest.raises(ValueError, match="没有"):
        snapshot_parameter_witness(params, None, max_elements=2)
    with pytest.raises(ValueError, match="形状或类型"):
        compare_parameter_witness({"x": np.ones(2, np.float32)}, {"x": np.ones(2, np.float64)})
