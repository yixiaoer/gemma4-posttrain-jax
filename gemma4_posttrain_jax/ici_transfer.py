"""用共同 mesh 上的 collective-permute 传递权重，避免跨 mesh 的主机暂存。"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=256)
def _zeros(shape: tuple[int, ...], dtype: Any, sharding: Any) -> Any:
    import jax
    import jax.numpy as jnp

    # 只缓存函数；不得缓存随模型大小增长的缓冲区。
    return jax.jit(lambda: jnp.zeros(shape, dtype), out_shardings=sharding)


@lru_cache(maxsize=256)
def _reshape(shape: tuple[int, ...], sharding: Any) -> Any:
    import jax

    return jax.jit(lambda x: x.reshape(shape), out_shardings=sharding)


@lru_cache(maxsize=128)
def _reshard(sharding: Any) -> Any:
    import jax

    return jax.jit(lambda x: x, out_shardings=sharding)


@lru_cache(maxsize=64)
def collective_function(mesh: Any) -> Any:
    import jax
    from jax.sharding import PartitionSpec as P

    count = mesh.size // 2
    pairs = tuple((i, i + count) for i in range(count))

    @jax.shard_map(mesh=mesh, in_specs=P("transfer"), out_specs=P("transfer"), check_vma=False)
    def move(value: Any) -> Any:
        return jax.lax.ppermute(value, "transfer", pairs)

    return jax.jit(move)


def destination_layout_on_source(value: Any, destination: Any) -> Any:
    """在源芯片上表达目标的分片方式，包含目标的每个物理副本。"""
    import numpy as np
    from jax.sharding import Mesh, NamedSharding

    source_devices = tuple(value.sharding.mesh.devices.flat)
    return NamedSharding(
        Mesh(np.array(source_devices).reshape(destination.mesh.devices.shape), destination.mesh.axis_names),
        destination.spec,
    )


def transfer_array_ici(value: Any, destination: Any) -> Any:
    """两侧芯片数相同、互不重叠；当前验证范围是同进程。

    对每个目标分片，先在训练 mesh 内形成对应分片，再由 collective 发往目标芯片。
    """
    import jax
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, SingleDeviceSharding
    from jax.sharding import PartitionSpec as P

    if not isinstance(value.sharding, NamedSharding) or not isinstance(destination, NamedSharding):
        raise ValueError("ICI 传输要求两侧使用 NamedSharding")
    source_devices = tuple(value.sharding.mesh.devices.flat)
    target_devices = tuple(destination.mesh.devices.flat)
    if len(source_devices) != len(target_devices) or set(source_devices) & set(target_devices):
        raise ValueError("ICI 传输要求两侧芯片数相同且互不重叠")
    if not value.is_fully_addressable or not destination.is_fully_addressable:
        raise ValueError("跨进程 TPU 运行时尚未通过验证；当前只支持同进程")
    mirror = destination_layout_on_source(value, destination)
    aligned = _reshard(mirror)(value)
    shard_shape = mirror.shard_shape(value.shape)
    mesh = Mesh(np.array(source_devices + target_devices), ("transfer",))
    packed_sharding = NamedSharding(mesh, P("transfer"))
    by_device = {s.device: s.data for s in aligned.addressable_shards}
    buffers = []
    for device in mesh.devices.flat:
        if device.process_index != jax.process_index():
            continue
        local = SingleDeviceSharding(device)
        if device in by_device:
            buffers.append(_reshape((1, *shard_shape), local)(by_device[device]))
        else:
            buffers.append(_zeros((1, *shard_shape), value.dtype, local)())
    packed = jax.make_array_from_single_device_arrays((mesh.size, *shard_shape), packed_sharding, buffers)
    moved = collective_function(mesh)(packed)
    moved_by_device = {s.device: s.data for s in moved.addressable_shards}
    target_buffers = [
        _reshape(shard_shape, SingleDeviceSharding(d))(moved_by_device[d])
        for d in target_devices
        if d.process_index == jax.process_index()
    ]
    return jax.make_array_from_single_device_arrays(value.shape, destination, target_buffers, dtype=value.dtype)


@lru_cache(maxsize=128)
def _source_tile(sharding: Any, length: int) -> Any:
    import jax
    from jax.sharding import PartitionSpec as P

    @jax.shard_map(mesh=sharding.mesh, in_specs=(sharding.spec, P()), out_specs=P(), check_vma=False)
    def read(value: Any, offset: Any) -> Any:
        local = jax.lax.dynamic_slice_in_dim(value, offset, length, axis=0)
        return jax.lax.all_gather(local, sharding.mesh.axis_names[0], axis=0, tiled=True)

    return jax.jit(read)


@lru_cache(maxsize=128)
def _write_tile(sharding: Any, count: int, local_rows: int) -> Any:
    import jax
    from jax.sharding import PartitionSpec as P

    @jax.shard_map(mesh=sharding.mesh, in_specs=(P(), P(), P()), out_specs=P(), check_vma=False)
    def write(value: Any, tile: Any, offset: Any) -> Any:
        rows = tile.shape[0] // count
        for i in range(count):
            value = jax.lax.dynamic_update_slice_in_dim(
                value, tile[i * rows : (i + 1) * rows], offset + i * local_rows, axis=0
            )
        return value

    # 只 donate 本函数分配的目标新数组，训练参数和引擎原权重均不参与 donation。
    return jax.jit(write, donate_argnums=(0,))


@lru_cache(maxsize=128)
def _target_tile(sharding: Any, count: int, local_rows: int, length: int) -> Any:
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    @jax.shard_map(mesh=sharding.mesh, in_specs=(P(), P()), out_specs=P(), check_vma=False)
    def read(value: Any, offset: Any) -> Any:
        blocks = [jax.lax.dynamic_slice_in_dim(value, offset + i * local_rows, length, axis=0) for i in range(count)]
        # TPU 的 BF16 concatenate 可降为浮点 maximum，冲掉 subnormal；按整数位拼接。
        bits = jnp.concatenate(
            jax.lax.optimization_barrier(tuple(jax.lax.bitcast_convert_type(block, jnp.uint16) for block in blocks)),
            axis=0,
        )
        return bits

    return jax.jit(read)


def transfer_large_replicated_weight(value: Any, destination: Any, *, chunk_bytes: int = 16 << 20) -> Any:
    """按源分片读取有限大小的块，在目标写入后独立读回检查每个副本。

    当前覆盖一维源 mesh 和目标全复制的大参数；源分片先统一到首维。
    """
    import time

    import jax
    import numpy as np
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from .inference_device import device_comparator

    source = value.sharding
    if (
        not isinstance(source, NamedSharding)
        or len(source.mesh.axis_names) != 1
        or not destination.is_fully_replicated
        or value.shape[0] % source.mesh.size
    ):
        raise ValueError("大权重分块 ICI 要求一维源 mesh、首维可分片、目标全复制")
    if chunk_bytes < value.dtype.itemsize:
        raise ValueError("chunk_bytes 必须至少容纳一个元素")
    count = source.mesh.size
    source = NamedSharding(source.mesh, P(source.mesh.axis_names[0]))
    value = _reshard(source)(value)
    local_rows = value.shape[0] // count
    if local_rows >= 2**31:
        raise ValueError("当前分块索引要求单个源分片少于 2**31 行")
    # 保留原始行形状；展平真实 PLE 会触发每块整参数的布局转换，抵消 ICI 收益。
    row_bytes = int(np.prod(value.shape[1:])) * value.dtype.itemsize
    rows = min(max(1, chunk_bytes // row_bytes), local_rows)
    source_rep = NamedSharding(source.mesh, P())
    target_rep = NamedSharding(destination.mesh, P())
    incoming = _zeros(value.shape, value.dtype, destination)()
    incoming.block_until_ready()
    passed = jax.device_put(np.array(True), source_rep)
    forward_s = check_s = 0.0
    chunks = 0
    for offset in range(0, local_rows, rows):
        length = min(rows, local_rows - offset)
        source_offset = jax.device_put(np.int32(offset), source_rep)
        target_offset = jax.device_put(np.int32(offset), target_rep)
        started = time.perf_counter()
        expected = _source_tile(source, length)(value, source_offset)
        sent = transfer_array_ici(expected, target_rep)
        incoming = _write_tile(destination, count, local_rows)(incoming, sent, target_offset)
        incoming.block_until_ready()
        written = time.perf_counter()
        # 与写入分开编译，避免编译器把 read-after-write 检查简化成比较输入本身。
        actual = _target_tile(destination, count, local_rows, length)(incoming, target_offset)
        returned = transfer_array_ici(actual, source_rep)
        passed = passed & device_comparator(source_rep)(expected, returned)
        passed.block_until_ready()
        forward_s += written - started
        check_s += time.perf_counter() - written
        chunks += 1
    return (
        incoming,
        passed,
        {
            "chunks": chunks,
            "chunk_bytes_per_source_chip": rows * row_bytes,
            "chunk_rows_per_source_chip": rows,
            "preserves_row_layout": True,
            "forward_s": forward_s,
            "check_s": check_s,
            "host_control_logical_bytes": 1 + chunks * 8,
        },
    )
