"""Host-side mathematical rewards for GSM8K and MATH GRPO rollouts."""

from __future__ import annotations

import multiprocessing
import re
from collections.abc import Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from typing import NamedTuple

import numpy as np

# 一个字符串可表示完整集合；元组表示题目允许任选一个完整答案。
Gold = str | tuple[str, ...]


class RewardConfig(NamedTuple):
    success_reward: float = 1.0
    failure_reward: float = 0.0
    format_bonus: float = 0.1
    max_completion_length: int | None = None
    overlong_buffer_length: int = 0
    overlong_penalty: float = 0.0


class RewardOutput(NamedTuple):
    task_success: np.ndarray
    base_reward: np.ndarray
    format_reward: np.ndarray
    length_penalty: np.ndarray
    score: np.ndarray


_BOXED = re.compile(r"\\boxed\s*\{")


def _length_penalty(length: int | None, config: RewardConfig) -> float:
    if length is None or config.max_completion_length is None or config.overlong_penalty == 0.0:
        return 0.0
    if config.overlong_buffer_length <= 0:
        raise ValueError("overlong_buffer_length must be positive when length penalty is enabled")
    penalty_start = config.max_completion_length - config.overlong_buffer_length
    excess = max(length - penalty_start, 0)
    return -config.overlong_penalty * min(excess / config.overlong_buffer_length, 1.0)


def _score_one(arguments: tuple[str, Gold, int | None, RewardConfig]) -> tuple[float, float, float, float, float]:
    completion, gold, length, config = arguments
    from math_verify import parse, verify

    alternatives = (gold,) if isinstance(gold, str) else gold
    if not alternatives or any(not isinstance(value, str) or not value.strip() for value in alternatives):
        raise ValueError("gold alternatives must be nonempty strings")
    predicted_expression = parse(completion)
    success = float(
        bool(predicted_expression)
        and any(bool(parsed := parse(value)) and verify(parsed, predicted_expression) for value in alternatives)
    )
    base = config.success_reward if success else config.failure_reward
    format_reward = config.format_bonus if _BOXED.search(completion) else 0.0
    length_reward = _length_penalty(length, config)
    return success, base, format_reward, length_reward, base + format_reward + length_reward


def score_completions(
    completions: Sequence[str],
    golds: Sequence[Gold],
    *,
    completion_lengths: Sequence[int] | None = None,
    config: RewardConfig | None = None,
    workers: int | None = None,
    executor: Executor | None = None,
) -> RewardOutput:
    """Score completions in worker processes and retain each reward component separately."""

    if len(completions) != len(golds):
        raise ValueError(f"completion/gold counts differ: {len(completions)} != {len(golds)}")
    if completion_lengths is not None and len(completion_lengths) != len(completions):
        raise ValueError("completion_lengths must match completions")
    if workers is not None and workers <= 0:
        raise ValueError("workers must be positive")
    if config is None:
        config = RewardConfig()
    lengths: Sequence[int | None]
    lengths = [None] * len(completions) if completion_lengths is None else completion_lengths
    arguments = list(zip(completions, golds, lengths, [config] * len(completions), strict=True))
    if executor is not None:
        rows = list(executor.map(_score_one, arguments))
    elif workers == 1:
        rows = list(map(_score_one, arguments))
    else:
        # JAX runtime 已有线程；fork 会复制不完整的线程/锁状态。
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            rows = list(pool.map(_score_one, arguments))
    if not rows:
        empty = np.empty((0,), dtype=np.float32)
        return RewardOutput(empty, empty.copy(), empty.copy(), empty.copy(), empty.copy())
    columns = np.asarray(rows, dtype=np.float32).T
    return RewardOutput(*(columns[index] for index in range(columns.shape[0])))
