"""共同 JAX 运行时中的 2+2 权重传输；两侧须按相同顺序调用。

训练侧保留原 FP32 分片，单个参数在设备上转换成 BF16 位编码后发送。
本模块不创建进程或引擎；部署、事务提交及独立内容验证由调用方负责。
"""

from __future__ import annotations

import math
import time
from functools import lru_cache
from typing import Any, cast


@lru_cache(maxsize=256)
def mapped_weight_bits(layout: str, shape: tuple[int, ...], shards: int, sharding: Any) -> Any:
    """直接从训练参数生成按行分片的 uint16；排列和 padding 不经过浮点运算。"""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    def convert(*values: Any) -> tuple[Any, Any]:
        arrays = []
        finite = jnp.array(True)
        for value in values:
            bits = jax.lax.bitcast_convert_type(value, jnp.uint32)
            rounded = ((bits + jnp.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(jnp.uint16)
            finite &= jnp.all((bits & 0x7F800000) != 0x7F800000) & jnp.all((rounded & 0x7F80) != 0x7F80)
            arrays.append(rounded)
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
        elif layout in ("identity", "reshape"):
            result = arrays[0].reshape(shape)
        else:
            raise ValueError(f"未知参数布局: {layout}")
        result = jnp.pad(result, ((0, shape[0] % 2), *[(0, 0) for _ in shape[1:]]))
        return result, finite

    return jax.jit(convert, out_shardings=(sharding, NamedSharding(sharding.mesh, P())))


class SharedRuntimeWeightTransfer:
    """四颗芯片中，前两颗发送、后两颗各保留完整副本；不承担引擎版本事务。"""

    def __init__(self, devices: list[Any], *, chunk_bytes: int = 16 << 20, verify: bool = False):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P

        if len(devices) != 4 or len(set(devices)) != 4 or chunk_bytes <= 0:
            raise ValueError("共同运行时传输要求四个不同设备和正数分块预算")
        self.devices = devices
        self.local_ranks = [i for i, d in enumerate(devices) if d.process_index == jax.process_index()]
        if self.local_ranks not in ([0, 1], [2, 3], [0, 1, 2, 3]):
            raise ValueError("只支持同进程四芯，或两进程各持有连续的两芯")
        self.chunk_bytes = chunk_bytes
        self.verify = verify
        mesh = Mesh(np.array(devices), ("transfer",))
        self.sharding = NamedSharding(mesh, P("transfer"))
        self.source_sharding = NamedSharding(Mesh(np.array(devices[:2]), ("d",)), P("d"))

        @jax.shard_map(mesh=mesh, in_specs=P("transfer"), out_specs=P("transfer"), check_vma=False)
        def send(bits: Any) -> Any:
            replicated = jax.lax.all_gather(bits, "transfer", axis_index_groups=((0, 1), (2, 3)), axis=1, tiled=True)
            return jax.lax.ppermute(replicated, "transfer", ((0, 2), (1, 3)))

        self.send = jax.jit(send)

        @jax.shard_map(mesh=mesh, in_specs=(P("transfer"), P("transfer")), out_specs=P(), check_vma=False)
        def check_written(expected: Any, written: Any) -> Any:
            reference = jax.lax.all_gather(expected, "transfer", axis_index_groups=((0, 1), (2, 3)), axis=1, tiled=True)
            returned = jax.lax.ppermute(written, "transfer", ((2, 0), (3, 1)))
            same = (jax.lax.axis_index("transfer") >= 2) | jnp.all(reference == returned)
            return jax.lax.pmin(same.astype(jnp.int32), "transfer")

        self.check_written = jax.jit(check_written)
        self.dtype = jnp.uint16

    def map_source(self, item: Any) -> tuple[Any, Any]:
        import jax

        values = tuple(value for _, value in item.sources)
        if not self.has_source or any(
            not isinstance(v, jax.Array)
            or str(v.dtype) != "float32"
            or not v.is_fully_addressable
            or v.sharding.device_set != set(self.devices[:2])
            for v in values
        ):
            raise ValueError("发送侧要求两颗训练芯片上的可访问 FP32 参数")
        return cast(
            tuple[Any, Any], mapped_weight_bits(item.layout, item.shape, item.shards, self.source_sharding)(*values)
        )

    @property
    def has_source(self) -> bool:
        return 0 in self.local_ranks

    def transfer(self, shape: tuple[int, ...], source: Any | None) -> tuple[dict[int, Any], dict[str, Any]]:
        """接收结果只位于目标进程；完整权重不经过主机，结果尚未提交给引擎。"""
        import jax
        import numpy as np

        if not shape or any(d <= 0 for d in shape):
            raise ValueError("推理映射必须具有非空的正数维度")
        half_rows = (shape[0] + 1) // 2
        padded_shape = (2 * half_rows, *shape[1:])
        if self.has_source != (source is not None):
            raise ValueError("只有训练进程提供源参数")
        if source is not None and (
            source.shape != padded_shape or source.dtype != self.dtype or source.sharding != self.source_sharding
        ):
            raise ValueError("源参数必须已转换为按行分片的 uint16")
        local_source = (
            {self.devices.index(s.device): s.data for s in source.addressable_shards} if source is not None else {}
        )
        start = time.perf_counter()
        actual = {r: _empty(padded_shape, self.devices[r])() for r in self.local_ranks if r >= 2}
        chunk_rows = max(1, self.chunk_bytes // (math.prod(shape[1:]) * 2))
        chunks = 0
        for offset in range(0, half_rows, chunk_rows):
            length = min(chunk_rows, half_rows - offset)
            block_shape = (length, *shape[1:])
            buffers = [
                _read(length)(local_source[r], np.int32(offset))
                if r < 2
                else _empty((1, *block_shape), self.devices[r])()
                for r in self.local_ranks
            ]
            packed = jax.make_array_from_single_device_arrays((4, *block_shape), self.sharding, buffers)
            moved = self.send(packed)
            for shard in moved.addressable_shards:
                rank = self.devices.index(shard.device)
                if rank >= 2:
                    actual[rank] = _write(half_rows, length)(actual[rank], shard.data, np.int32(offset))
            if self.verify:
                written_buffers = [
                    _read_written(half_rows, length)(actual[r], np.int32(offset))
                    if r >= 2
                    else _empty((1, 2 * length, *shape[1:]), self.devices[r])()
                    for r in self.local_ranks
                ]
                written = jax.make_array_from_single_device_arrays(
                    (4, 2 * length, *shape[1:]), self.sharding, written_buffers
                )
                checked = self.check_written(packed, written)
                # 只读取复制的一个整数；完整目标分块经 ICI 返回发送侧比较。
                if int(checked.addressable_shards[0].data) != 1:
                    raise ValueError("ICI 接收数组的实际内容与训练源不同")
                del written, written_buffers, checked
            jax.block_until_ready((actual, moved))
            del packed, moved, buffers
            chunks += 1
        actual = {r: _view(shape)(value) for r, value in actual.items()}
        jax.block_until_ready(actual)
        return actual, {
            "transfer_s": time.perf_counter() - start,
            "chunks": chunks,
            "host_payload_bytes": 0,
            "all_replicas_bitwise_verified": self.verify,
        }


@lru_cache(maxsize=256)
def _empty(shape: tuple[int, ...], device: Any) -> Any:
    import jax
    import jax.numpy as jnp
    from jax.sharding import SingleDeviceSharding

    return jax.jit(lambda: jnp.zeros(shape, jnp.uint16), out_shardings=SingleDeviceSharding(device))


@lru_cache(maxsize=256)
def _read(length: int) -> Any:
    import jax

    return jax.jit(lambda x, offset: jax.lax.dynamic_slice_in_dim(x, offset, length, axis=0)[None])


@lru_cache(maxsize=256)
def _write(half_rows: int, length: int) -> Any:
    import jax

    @jax.jit(donate_argnums=(0,))
    def write(value: Any, bits: Any, offset: Any) -> Any:
        for i in range(2):
            value = jax.lax.dynamic_update_slice_in_dim(
                value, bits[0, i * length : (i + 1) * length], offset + i * half_rows, axis=0
            )
        return value

    return write


@lru_cache(maxsize=256)
def _read_written(half_rows: int, length: int) -> Any:
    import jax
    import jax.numpy as jnp

    @jax.jit
    def read(value: Any, offset: Any) -> Any:
        parts = [jax.lax.dynamic_slice_in_dim(value, offset + i * half_rows, length, axis=0) for i in range(2)]
        return jnp.concatenate(parts, axis=0)[None]

    return read


@lru_cache(maxsize=256)
def _view(shape: tuple[int, ...]) -> Any:
    import jax
    import jax.numpy as jnp

    return jax.jit(lambda x: jax.lax.bitcast_convert_type(x[: shape[0]].reshape(shape), jnp.bfloat16))
