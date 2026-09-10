"""固定调度批次的typed PRNG状态；不承诺逐请求独立随机流。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BatchRngState:
    implementation: str
    words: tuple[int, int]
    protocol: str = "engine-batch-rng-v1"

    def __post_init__(self) -> None:
        if self.protocol != "engine-batch-rng-v1" or self.implementation != "threefry2x32":
            raise ValueError("当前仅覆盖engine-batch-rng-v1/threefry2x32")
        if not isinstance(self.words, tuple) or len(self.words) != 2:
            raise ValueError("随机状态必须为两个uint32字的tuple")
        if any(type(word) is not int or not 0 <= word < 2**32 for word in self.words):
            raise ValueError("随机状态字必须为uint32范围的Python整数，不能为bool")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"words": list(self.words)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BatchRngState:
        if not isinstance(value, dict) or set(value) != {"implementation", "words", "protocol"}:
            raise ValueError("checkpoint随机状态字段缺失或包含未知字段")
        words = value["words"]
        if not isinstance(words, list) or len(words) != 2:
            raise ValueError("checkpoint随机状态必须为两个整数的list")
        return cls(value["implementation"], (words[0], words[1]), value["protocol"])


def read_runner_rng(runner: Any, expected_device_ids: tuple[int, ...]) -> tuple[BatchRngState, dict[str, Any]]:
    """读取实际采样key和放置；调用方必须先确认引擎无活动请求。"""
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec

    key = runner.rng_params_for_sampling
    if tuple(key.shape) != () or not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
        raise ValueError("采样状态不是标量typed PRNG key")
    implementation = str(jax.random.key_impl(key))
    if implementation != "threefry2x32":
        raise ValueError("runner实际PRNG实现不在本次资格范围内")
    sharding = key.sharding
    devices = sorted(int(device.id) for device in sharding.device_set)
    if (
        not isinstance(sharding, NamedSharding)
        or sharding.spec != PartitionSpec()
        or not sharding.is_fully_replicated
        or devices != sorted(expected_device_ids)
        or len(set(expected_device_ids)) != len(expected_device_ids)
    ):
        raise ValueError("runner key并非指定引擎mesh的完全复制状态")
    data = np.asarray(jax.device_get(jax.random.key_data(key)))
    if data.dtype != np.dtype("uint32") or data.shape != (2,):
        raise ValueError("typed key_data不是两个uint32字")
    state = BatchRngState(implementation, (int(data[0]), int(data[1])))
    record = {
        "shape": list(key.shape),
        "dtype": str(key.dtype),
        "device_ids": devices,
        "mesh_shape": dict(sharding.mesh.shape),
        "mesh_device_ids": [int(device.id) for device in sharding.mesh.devices.flat],
        "partition_spec": str(sharding.spec),
        "fully_replicated": sharding.is_fully_replicated,
    }
    return state, record


def install_runner_rng(runner: Any, state: BatchRngState, expected_device_ids: tuple[int, ...]) -> dict[str, Any]:
    """保留原有NamedSharding，运输完成且实际读回匹配后才返回。"""
    import jax
    import numpy as np

    if not isinstance(state, BatchRngState):
        raise TypeError("必须显式传递batch状态，不能用逐请求seed列表替代")
    before, placement = read_runner_rng(runner, expected_device_ids)
    old = runner.rng_params_for_sampling
    data = jax.device_put(np.asarray(state.words, dtype=np.uint32), old.sharding)
    key = jax.random.wrap_key_data(data, impl=state.implementation)
    key = jax.device_put(key, old.sharding)
    jax.block_until_ready(key)
    runner.rng_params_for_sampling = key
    after, actual_placement = read_runner_rng(runner, expected_device_ids)
    if after != state or actual_placement != placement:
        raise RuntimeError("实际采样key运输读回或放置不符合要求；调用方必须使后端失效")
    return {"previous": before.to_dict(), "installed": after.to_dict(), "placement": actual_placement}
