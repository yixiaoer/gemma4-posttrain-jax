"""一步滞后采样的完整恢复树；训练state与CPU行为快照分别保存。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import SingleDeviceSharding

from .checkpoint import load_train_state
from .losses import TrainState
from .model import Gemma4TextParams


class BehaviorSnapshot(NamedTuple):
    params_bf16: Gemma4TextParams
    step: Array


class LaggedTrainState(NamedTuple):
    train: TrainState
    behavior: BehaviorSnapshot


def load_lagged_train_state(path: Path, template: TrainState, shardings: Any) -> LaggedTrainState:
    """严格恢复训练树，并将采样快照放在CPU；不能用当前参数补造缺失快照。"""
    behavior = BehaviorSnapshot(
        cast(
            Gemma4TextParams,
            jax.tree.map(lambda value: jax.ShapeDtypeStruct(value.shape, jnp.bfloat16), template.params_f32),
        ),
        cast(Array, jax.ShapeDtypeStruct((), jnp.int32)),
    )
    host = SingleDeviceSharding(jax.devices("cpu")[0])
    restored = load_train_state(
        path,
        LaggedTrainState(template, behavior),
        shardings=LaggedTrainState(shardings, jax.tree.map(lambda _: host, behavior)),
    )
    if int(restored.train.step) < 1 or int(restored.behavior.step) != int(restored.train.step) - 1:
        raise ValueError("滞后checkpoint的行为快照必须来自最后一次Adam更新前")
    return restored
