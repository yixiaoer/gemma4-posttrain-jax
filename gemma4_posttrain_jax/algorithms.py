"""Configure GRPO-family objectives, fixed importance weights, and generation and loss masks."""

from __future__ import annotations

from typing import Literal, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from .losses import compute_advantages, compute_rloo_advantages
from .losses import truncated_importance_weights as truncated_importance_weights

AdvantageEstimator = Literal["standardized", "centered", "rloo"]
AlgorithmName = Literal["grpo", "drgrpo", "dapo", "gspo-token", "rloo"]


class ObjectiveConfig(NamedTuple):
    advantage_estimator: AdvantageEstimator = "standardized"
    ratio_level: Literal["token", "sequence"] = "token"
    eps_low: float = 0.2
    eps_high: float = 0.2
    clip_policy: bool = True
    agg_mode: str = "sequence-mean-token-mean"
    token_scale: float | None = None


def objective_for_algorithm(name: AlgorithmName, *, generation_budget: int) -> ObjectiveConfig:
    """Resolve advantages and loss settings; the host loop handles DAPO resampling and length filtering."""
    if generation_budget <= 0:
        raise ValueError("generation_budget必须为正")
    baseline = ObjectiveConfig()
    match name:
        case "grpo":
            return baseline
        case "drgrpo":
            return baseline._replace(
                advantage_estimator="centered",
                agg_mode="sequence-mean-token-scale",
                token_scale=float(generation_budget),
            )
        case "dapo":
            return baseline._replace(eps_high=0.28, agg_mode="token-mean")
        case "gspo-token":
            return baseline._replace(ratio_level="sequence", eps_low=3e-4, eps_high=4e-4)
        case "rloo":
            return baseline._replace(advantage_estimator="rloo", clip_policy=False, agg_mode="seq-mean-token-sum")
        case _:
            raise ValueError(f"未知算法：{name}")


def objective_advantages(rewards: Array, group_size: int, estimator: AdvantageEstimator) -> Array:
    if estimator == "rloo":
        return compute_rloo_advantages(rewards, group_size)
    if estimator not in ("standardized", "centered"):
        raise ValueError(f"未知优势估计：{estimator}")
    return compute_advantages(rewards, group_size, normalize_std=estimator == "standardized")


def completion_loss_mask(completion_mask: Array, truncated: Array, *, filter_truncated: bool) -> Array:
    """Select tokens for the loss while preserving the original generation mask for model context."""
    if completion_mask.ndim != 2 or truncated.shape != (completion_mask.shape[0],):
        raise ValueError("需要[B,N]生成mask与[B]截断标记")
    return completion_mask.astype(jnp.bool_) & (~truncated[:, None] if filter_truncated else True)


def validate_update_schedule(
    *, updates_per_rollout: int, max_steps: int, eval_every: int, save_every: int, start_step: int = 0
) -> None:
    """Evaluate and save after all updates for a rollout, so recovery needs no unfinished training batch."""
    if updates_per_rollout <= 0 or max_steps <= 0 or min(eval_every, save_every, start_step) < 0:
        raise ValueError("μ和目标步数必须为正，间隔和起始步数不能为负")
    if any(value % updates_per_rollout for value in (max_steps, eval_every, save_every, start_step)):
        raise ValueError("max-steps、eval/save间隔和checkpoint step必须位于完整rollout的μ次更新边界")


def mixed_success_groups(task_success: np.ndarray, group_size: int) -> np.ndarray:
    """Filter whole groups by binary task success, excluding format and length rewards from success rates."""
    success = np.asarray(task_success)
    if success.ndim != 1 or group_size < 2 or success.size % group_size:
        raise ValueError("task_success必须是一维连续G组")
    if not np.isin(success, (0, 1)).all():
        raise ValueError("动态采样只接受0/1 task_success")
    counts = success.reshape(-1, group_size).sum(axis=-1)
    return (counts > 0) & (counts < group_size)
