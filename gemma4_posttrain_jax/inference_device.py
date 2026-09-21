"""在设备上转换、传输和比较推理权重；主机只接收检查结果。"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=256)
def device_converter(layout: str, shape: tuple[int, ...], shards: int) -> Any:
    """缓存计算函数，不缓存参数数组；按位实现 FP32 到 BF16 的 ties-to-even 舍入。"""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def convert(*values: Any) -> tuple[Any, Any]:
        arrays = []
        finite = jnp.array(True)
        for value in values:
            bits = jax.lax.bitcast_convert_type(value, jnp.uint32)
            finite = finite & jnp.all((bits & jnp.uint32(0x7F800000)) != jnp.uint32(0x7F800000))
            # 整数舍入保留正负零和 BF16 subnormal，避免 TPU 浮点转换的 flush-to-zero。
            rounded = ((bits + jnp.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(jnp.uint16)
            finite = finite & jnp.all((rounded & jnp.uint16(0x7F80)) != jnp.uint16(0x7F80))
            arrays.append(jax.lax.bitcast_convert_type(rounded, jnp.bfloat16))
        if layout == "gate_up_interleaved":
            gate, up = arrays
            m, width = gate.shape
            result = jnp.concatenate(
                (gate.reshape(m, shards, width // shards), up.reshape(m, shards, width // shards)), axis=-1
            ).reshape(shape)
        elif layout == "q_heads":
            result = arrays[0].transpose(0, 2, 1, 3).reshape(shape)
        elif layout == "o_heads":
            result = arrays[0].transpose(1, 0, 2, 3).reshape(shape)
        elif layout == "reshape":
            result = arrays[0].reshape(shape)
        elif layout == "identity":
            result = arrays[0]
        else:
            raise ValueError(f"未知参数布局: {layout}")
        return result, finite

    return convert


@lru_cache(maxsize=128)
def device_comparator(sharding: Any) -> Any:
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    @jax.shard_map(mesh=sharding.mesh, in_specs=(sharding.spec, sharding.spec), out_specs=P(), check_vma=False)
    def compare(expected: Any, actual: Any) -> Any:
        equal = jnp.all(
            jax.lax.bitcast_convert_type(expected, jnp.uint16) == jax.lax.bitcast_convert_type(actual, jnp.uint16)
        )
        # 即使逻辑参数为 replicated，也必须比较每个物理副本；不能只读取一个 replica 的结果。
        return jax.lax.pmin(equal.astype(jnp.int32), sharding.mesh.axis_names).astype(jnp.bool_)

    return jax.jit(compare)


def transfer_device_weight(item: Any, destination_sharding: Any) -> tuple[Any, dict[str, Any]]:
    """只用于同一 JAX 运行时可访问的设备；包含完整往返位比较，不读取权重到 CPU。"""
    import jax

    values = tuple(value for _, value in item.sources)
    if not values or any(not isinstance(v, jax.Array) or str(v.dtype) != "float32" for v in values):
        raise ValueError("设备直传要求设备上的 FP32 JAX 参数")
    source_devices = values[0].sharding.device_set
    if any(v.sharding.device_set != source_devices or not v.is_fully_addressable for v in values):
        raise ValueError("一个映射的源参数必须位于相同的可访问设备集合")
    # 多 CPU 设备仅用于独立测试，正式入口由 TPU 引擎控制。
    platforms = {d.platform for d in source_devices | destination_sharding.device_set}
    if platforms not in ({"tpu"}, {"cpu"}):
        raise ValueError("源和目标必须使用同一种支持的设备")
    if not destination_sharding.is_fully_addressable:
        raise ValueError("当前实现要求目标设备在本进程可访问；不支持独立运行时")
    started = time.perf_counter()
    expected, finite = device_converter(item.layout, item.shape, item.shards)(*values)
    jax.block_until_ready((expected, finite))
    converted = time.perf_counter()
    if expected.shape != item.shape:
        raise ValueError(f"设备转换后的形状不符: {item.name}")
    from gemma4_posttrain_jax.ici_transfer import (
        _reshard,
        destination_layout_on_source,
        transfer_array_ici,
        transfer_large_replicated_weight,
    )

    if source_devices == destination_sharding.device_set:
        raise ValueError("设备直传要求训练和推理使用不同芯片")
    chunk_report = None
    transfer_overhead_s = 0.0
    if expected.nbytes > 64 << 20:
        incoming, equal, chunk_report = transfer_large_replicated_weight(expected, destination_sharding)
        checked = time.perf_counter()
        forward_s = chunk_report["forward_s"]
        check_s = chunk_report["check_s"]
        # 分块中传输和检查交错执行；目标分配、重分片、标量控制和Python调度另列。
        transfer_overhead_s = max(0.0, checked - converted - forward_s - check_s)
    else:
        incoming = transfer_array_ici(expected, destination_sharding)
        jax.block_until_ready(incoming)
        transferred = time.perf_counter()
        # 比较目标设备实际返回的内容，不能将 incoming 与它自身比较。
        verification_sharding = destination_layout_on_source(expected, destination_sharding)
        returned = transfer_array_ici(incoming, verification_sharding)
        equal = device_comparator(verification_sharding)(_reshard(verification_sharding)(expected), returned)
        equal.block_until_ready()
        checked = time.perf_counter()
        forward_s = transferred - converted
        check_s = checked - transferred
    finite_host, equal_host = jax.device_get((finite, equal))
    finished = time.perf_counter()
    if not bool(finite_host) or not bool(equal_host):
        raise ValueError(f"设备同步的有限性或往返位比较失败: {item.name}")
    return incoming, {
        "verification": "ici-collective-all-replicas-bitwise-v2",
        "source_and_bf16_finite": True,
        "roundtrip_bits_equal": True,
        "all_destination_replicas_checked": True,
        "host_payload_bytes": 0,
        "host_check_bytes": 2,
        "chunked": chunk_report,
        "convert_and_finite_s": converted - started,
        "destination_transfer_s": forward_s,
        "roundtrip_check_s": check_s,
        "transfer_setup_and_control_s": transfer_overhead_s,
        "check_scalar_readback_s": finished - checked,
        "total_s": finished - started,
    }
