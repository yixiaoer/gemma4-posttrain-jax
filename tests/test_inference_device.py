"""设备权重转换、跨设备内容和失败条件；运行前设置四个 CPU 设备或使用 TPU。"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from ml_dtypes import bfloat16

from gemma4_posttrain_jax.ici_transfer import transfer_array_ici, transfer_large_replicated_weight
from gemma4_posttrain_jax.inference_device import device_comparator, device_converter, transfer_device_weight
from gemma4_posttrain_jax.inference_weights import _host_bf16, _Mapping


def test_rounding_boundaries_and_all_bf16_encodings():
    # 包括 BF16 全部有限编码及各个舍入边界两侧；FP32 非有限值另行拒绝。
    high = np.arange(65536, dtype=np.uint32)
    high = high[(high & 0x7F80) != 0x7F80]
    bits = ((high[:, None] << 16) | np.array([0, 0x7FFF, 0x8000, 0x8001, 0xFFFF], np.uint32)).ravel()
    value = bits.view(np.float32)
    output, finite = device_converter("identity", value.shape, 1)(jax.device_put(value))
    with np.errstate(over="ignore"):
        expected = value.astype(bfloat16)
    np.testing.assert_array_equal(np.asarray(output).view(np.uint16), expected.view(np.uint16))
    assert bool(finite) == bool(np.isfinite(expected).all())


@pytest.mark.parametrize(
    "layout,shapes,target,parts",
    [
        ("identity", [(32, 64)], (32, 64), 1),
        ("reshape", [(32, 2, 4)], (32, 8), 1),
        ("q_heads", [(32, 2, 4, 8)], (32, 8, 8), 1),
        ("o_heads", [(2, 4, 8, 32)], (8, 8, 32), 1),
        ("gate_up_interleaved", [(32, 64), (32, 64)], (32, 128), 1),
        ("gate_up_interleaved", [(32, 64), (32, 64)], (32, 128), 2),
    ],
)
@pytest.mark.parametrize("replicated", [True, False])
def test_host_and_device_mapping_bits_and_master_unchanged(layout, shapes, target, parts, replicated):
    if len(jax.devices()) < 4:
        pytest.skip("需要四个设备；CPU 用 XLA_FLAGS=--xla_force_host_platform_device_count=4")
    src = Mesh(np.array(jax.devices()[:2]), ("src",))
    dst = Mesh(np.array(jax.devices()[2:4]), ("dst",))
    rng = np.random.default_rng(43)
    host = [rng.normal(size=shape).astype(np.float32) for shape in shapes]
    values = [jax.device_put(a, NamedSharding(src, P("src"))) for a in host]
    item = _Mapping("test", target, tuple((str(i), v) for i, v in enumerate(values)), layout, parts)
    expected = _host_bf16(_Mapping("host", target, tuple((str(i), a) for i, a in enumerate(host)), layout, parts))
    output, report = transfer_device_weight(item, NamedSharding(dst, P() if replicated else P("dst")))
    np.testing.assert_array_equal(np.asarray(output).view(np.uint16), expected.view(np.uint16))
    assert report["roundtrip_bits_equal"] and report["host_payload_bytes"] == 0
    for original, actual in zip(host, values, strict=True):
        np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), original.view(np.uint32))
        assert actual.dtype == jnp.float32 and not actual.is_deleted()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, np.finfo(np.float32).max])
def test_reject_nonfinite_source_and_bf16_overflow(bad):
    src = NamedSharding(Mesh(np.array(jax.devices()[:2]), ("src",)), P())
    dst = NamedSharding(Mesh(np.array(jax.devices()[2:4]), ("dst",)), P())
    value = jax.device_put(np.array([bad], np.float32), src)
    item = _Mapping("bad", (1,), (("source", value),))
    with pytest.raises(ValueError, match="有限性"):
        transfer_device_weight(item, dst)
    assert not value.is_deleted()


def test_reject_host_array_as_device_transport():
    item = _Mapping("bad", (2,), (("source", np.ones(2, np.float32)),))
    with pytest.raises(ValueError, match="FP32 JAX"):
        transfer_device_weight(item, jax.sharding.SingleDeviceSharding(jax.devices()[0]))


@pytest.mark.parametrize("dtype", [np.uint16, bfloat16])
@pytest.mark.parametrize("source_axis", [None, 0, 1])
@pytest.mark.parametrize("target_axis", [None, 0, 1])
def test_collective_preserves_bits_without_host_transfer(source_axis, target_axis, dtype):
    source_mesh = Mesh(np.array(jax.devices()[:2]), ("src",))
    # 倒序 mesh 检查逻辑分片顺序，不能假设全局 device id 与 mesh index 相同。
    target_mesh = Mesh(np.array(jax.devices()[2:4][::-1]), ("dst",))
    src_spec = [None, None]
    dst_spec = [None, None]
    if source_axis is not None:
        src_spec[source_axis] = "src"
    if target_axis is not None:
        dst_spec[target_axis] = "dst"
    bits = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    if dtype == bfloat16:
        # BF16 权重只允许有限值；CPU collective 会规范化 NaN 的编码。
        bits[(bits & 0x7F80) == 0x7F80] = 0
    value = jax.device_put(bits.view(dtype), NamedSharding(source_mesh, P(*src_spec)))
    destination = NamedSharding(target_mesh, P(*dst_spec))
    # uint16 覆盖全部编码；BF16 覆盖全部有限编码，包括正负零和 subnormal。
    with (
        jax.transfer_guard_host_to_device("disallow_explicit"),
        jax.transfer_guard_device_to_host("disallow_explicit"),
    ):
        moved = transfer_array_ici(value, destination)
        moved.block_until_ready()
    np.testing.assert_array_equal(np.asarray(moved).view(np.uint16), bits)
    assert moved.sharding == destination
    assert getattr(value, "_npy_value", None) is None


def test_comparison_detects_corruption_in_second_replica():
    devices = jax.devices()[:2]
    mesh = Mesh(np.array(devices).reshape(2, 1), ("data", "model"))
    sharding = NamedSharding(mesh, P())
    host = np.ones((128, 128), dtype=bfloat16)
    expected = jax.device_put(host, sharding)
    corrupted = host.copy()
    corrupted.view(np.uint16)[-1, -1] ^= np.uint16(1)
    # 故意构造不一致副本，检查比较函数不会只相信 replicated 声明而忽略第二颗芯片。
    actual = jax.make_array_from_single_device_arrays(
        host.shape, sharding, [jax.device_put(host, devices[0]), jax.device_put(corrupted, devices[1])]
    )
    assert not bool(device_comparator(sharding)(expected, actual))
    assert bool(device_comparator(sharding)(expected, expected))


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("chunk_bytes", [512, 1536, 65536])
def test_chunked_copy_and_tail_preserve_all_target_replicas(axis, chunk_bytes):
    source_mesh = Mesh(np.array(jax.devices()[:2]), ("source",))
    target_mesh = Mesh(np.array(jax.devices()[2:]).reshape(2, 1), ("data", "model"))
    spec = P("source") if axis == 0 else P(None, "source")
    host = np.random.default_rng(38).normal(size=(64, 128)).astype(bfloat16)
    source = jax.device_put(host, NamedSharding(source_mesh, spec))
    target = NamedSharding(target_mesh, P())
    with jax.transfer_guard_device_to_host("disallow_explicit"):
        moved, passed, report = transfer_large_replicated_weight(source, target, chunk_bytes=chunk_bytes)
        jax.block_until_ready((moved, passed))
    assert bool(passed)
    assert report["chunks"] == (host.nbytes // 2 + chunk_bytes - 1) // chunk_bytes
    for shard in moved.addressable_shards:
        np.testing.assert_array_equal(np.asarray(shard.data).view(np.uint16), host.view(np.uint16))
    np.testing.assert_array_equal(np.asarray(source).view(np.uint16), host.view(np.uint16))


@pytest.mark.parametrize("shape", [(256, 256), (256, 4, 64)])
def test_row_chunks_preserve_all_finite_bf16_bits_with_unaligned_tail(shape):
    bits = np.arange(65536, dtype=np.uint16).reshape(shape)
    bits[(bits & 0x7F80) == 0x7F80] = 0
    src = NamedSharding(Mesh(np.array(jax.devices()[:2]), ("source",)), P("source"))
    dst = NamedSharding(Mesh(np.array(jax.devices()[2:4][::-1]), ("target",)), P())
    original = jax.device_put(bits.view(bfloat16), src)
    moved, passed, report = transfer_large_replicated_weight(original, dst, chunk_bytes=7 * 256 * 2)
    assert bool(passed)
    assert report["chunks"] == 19 and report["chunk_rows_per_source_chip"] == 7
    for shard in moved.addressable_shards:
        np.testing.assert_array_equal(np.asarray(shard.data).view(np.uint16), bits)
    np.testing.assert_array_equal(np.asarray(original).view(np.uint16), bits)
