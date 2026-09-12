"""Host-side numerical diagnostics shared by rollout and training startup checks."""

from __future__ import annotations

import importlib.metadata
import subprocess
from pathlib import Path
from typing import Any

import jax
import numpy as np


def optional_package_version(name: str) -> str | None:
    """记录可选运行库；CPU环境没有libtpu时写null，不伪造TPU版本。"""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def source_git_state(source_root: Path) -> tuple[str | None, str | None]:
    """Git元数据可缺省；未提交目录或源码包仍由调用方记录实际源码SHA。"""

    try:
        root = subprocess.run(
            ("git", "rev-parse", "--show-toplevel"),
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        # 独立源码快照可能放在另一个仓库内，不能继承那个仓库的提交身份。
        if Path(root).resolve() != source_root.resolve():
            return None, None
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=source_root, check=True, capture_output=True, text=True, timeout=5
        ).stdout.strip()
        diff = subprocess.run(
            ("git", "diff", "HEAD"), cwd=source_root, check=True, capture_output=True, text=True, timeout=5
        ).stdout
        return revision, diff
    except (FileNotFoundError, subprocess.SubprocessError):
        return None, None


def _masked_rows(actual: Any, expected: Any, mask: Any) -> tuple[np.ndarray, np.ndarray]:
    actual_array = np.asarray(jax.device_get(actual))
    expected_array = np.asarray(jax.device_get(expected))
    mask_array = np.asarray(jax.device_get(mask), dtype=np.bool_)
    if actual_array.shape != expected_array.shape:
        raise ValueError(f"actual/expected shape mismatch: {actual_array.shape} != {expected_array.shape}")
    if actual_array.shape[: mask_array.ndim] != mask_array.shape:
        raise ValueError(f"mask {mask_array.shape} is not a prefix of values {actual_array.shape}")
    if not np.any(mask_array):
        raise ValueError("parity diagnostics require at least one selected row")
    return actual_array[mask_array].astype(np.float64), expected_array[mask_array].astype(np.float64)


def array_drift_metrics(actual: Any, expected: Any, mask: Any) -> dict[str, float | int]:
    """Summarize selected tensor rows without hiding direction or scale."""

    actual_rows, expected_rows = _masked_rows(actual, expected, mask)
    delta = actual_rows - expected_rows
    actual_flat = actual_rows.reshape(-1)
    expected_flat = expected_rows.reshape(-1)
    numerator = float(np.linalg.norm(delta.reshape(-1)))
    expected_norm = float(np.linalg.norm(expected_flat))
    actual_norm = float(np.linalg.norm(actual_flat))
    denominator = max(actual_norm * expected_norm, np.finfo(np.float64).tiny)
    return {
        "rows": int(actual_rows.shape[0]),
        "elements": int(actual_rows.size),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "max_abs_delta": float(np.max(np.abs(delta))),
        "relative_l2": numerator / max(expected_norm, np.finfo(np.float64).tiny),
        "cosine": float(np.vdot(actual_flat, expected_flat).real / denominator),
        "norm_ratio": actual_norm / max(expected_norm, np.finfo(np.float64).tiny),
    }


def logprob_drift_metrics(
    actual: Any,
    expected: Any,
    mask: Any,
    *,
    delta_definition: str,
) -> dict[str, float | int | str]:
    """Report log-probability drift and ``exp(actual - expected)`` importance ratios."""

    actual_rows, expected_rows = _masked_rows(actual, expected, mask)
    delta = actual_rows - expected_rows
    ratio = np.exp(delta)
    return {
        "delta_definition": delta_definition,
        "count": int(delta.size),
        "mean_delta": float(np.mean(delta)),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "max_abs_delta": float(np.max(np.abs(delta))),
        "ratio_mean": float(np.mean(ratio)),
        "ratio_min": float(np.min(ratio)),
        "ratio_max": float(np.max(ratio)),
        "ratio_outside_10pct_fraction": float(np.mean((ratio < 0.9) | (ratio > 1.1))),
        "ratio_p01": float(np.quantile(ratio, 0.01)),
        "ratio_p50": float(np.quantile(ratio, 0.50)),
        "ratio_p99": float(np.quantile(ratio, 0.99)),
    }
