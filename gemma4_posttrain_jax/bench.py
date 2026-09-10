"""Small, synchronised benchmark harness shared by TPU experiments."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any, NamedTuple

import jax


class BenchmarkResult(NamedTuple):
    compile_s: float
    steady_mean_ms: float
    steady_p50_ms: float
    iterations_ms: tuple[float, ...]
    peak_bytes_in_use: tuple[int | None, ...]


def _ready(value: Any) -> Any:
    return jax.block_until_ready(value)


def timed_call(fn: Callable[..., Any], *args: Any) -> tuple[Any, float]:
    """同步执行一次；适合不能额外 warmup/update 的真实训练循环。"""

    start = time.perf_counter()
    result = _ready(fn(*args))
    return result, time.perf_counter() - start


def device_peak_bytes() -> tuple[int | None, ...]:
    peaks: list[int | None] = []
    for device in jax.local_devices():
        stats = device.memory_stats() or {}
        peak = stats.get("peak_bytes_in_use")
        peaks.append(None if peak is None else int(peak))
    return tuple(peaks)


def benchmark(fn: Callable[..., Any], *args: Any, warmup: int = 2, iterations: int = 5) -> BenchmarkResult:
    """Measure first-call compile+execute separately from synchronised steady-state calls."""

    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    start = time.perf_counter()
    _ready(fn(*args))
    compile_s = time.perf_counter() - start
    for _ in range(warmup):
        _ready(fn(*args))
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ready(fn(*args))
        samples.append((time.perf_counter() - start) * 1000.0)
    return BenchmarkResult(
        compile_s=compile_s,
        steady_mean_ms=statistics.fmean(samples),
        steady_p50_ms=statistics.median(samples),
        iterations_ms=tuple(samples),
        peak_bytes_in_use=device_peak_bytes(),
    )


def benchmark_stateful(
    fn: Callable[..., tuple[Any, Any]], state: Any, *args: Any, warmup: int = 2, iterations: int = 5
) -> tuple[BenchmarkResult, Any]:
    """Benchmark a donated-state step by feeding each returned state into the next invocation."""

    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    start = time.perf_counter()
    state, output = fn(state, *args)
    _ready((state, output))
    compile_s = time.perf_counter() - start
    for _ in range(warmup):
        state, output = fn(state, *args)
        _ready((state, output))
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        state, output = fn(state, *args)
        _ready((state, output))
        samples.append((time.perf_counter() - start) * 1000.0)
    return (
        BenchmarkResult(
            compile_s=compile_s,
            steady_mean_ms=statistics.fmean(samples),
            steady_p50_ms=statistics.median(samples),
            iterations_ms=tuple(samples),
            peak_bytes_in_use=device_peak_bytes(),
        ),
        state,
    )
