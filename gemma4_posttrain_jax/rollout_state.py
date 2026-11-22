"""Save training state and a CPU behavior-policy snapshot for sampling one update behind."""

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
    """Restore full training state and the CPU behavior snapshot; current parameters cannot replace a missing
    snapshot.
    """
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
