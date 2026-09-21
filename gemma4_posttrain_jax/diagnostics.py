"""Host-side numerical diagnostics shared by rollout and training startup checks."""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
from pathlib import Path
from typing import Any

import jax
import numpy as np


def checkpoint_storage(path: Path | None, *, mountinfo: str | None = None) -> dict[str, Any] | None:
    """按实际挂载点识别内存文件系统；非内存文件系统也不代表已有持久备份。"""
    if path is None:
        return None
    resolved = path.resolve()
    result: dict[str, Any] = {"path": str(resolved), "mount_point": None, "filesystem": None, "memory_backed": None}
    if mountinfo is None:
        try:
            mountinfo = Path("/proc/self/mountinfo").read_text()
        except OSError:
            return result
    longest = -1
    for line in mountinfo.splitlines():
        before, separator, after = line.partition(" - ")
        fields, filesystem = before.split(), after.split()
        if not separator or len(fields) < 6 or not filesystem:
            continue
        mount = Path(re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), fields[4]))
        if resolved.is_relative_to(mount) and len(mount.parts) > longest:
            longest = len(mount.parts)
            result.update(
                mount_point=str(mount), filesystem=filesystem[0], memory_backed=filesystem[0] in {"tmpfs", "ramfs"}
            )
    return result


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


def logprob_drift_gate(
    metrics: dict[str, float | int | str],
    *,
    start_step: int,
    ratio_lower: float = 0.9,
    ratio_upper: float = 1.1,
) -> dict[str, float | int | bool | str]:
    """Apply the rollout/trainer admission gate only to a run starting at step zero.

    A resumed process observes a different generated batch than the original run's
    first batch.  Its drift remains useful evidence, but making it a new admission
    gate would give continuous and resumed execution different control flow.
    """

    if start_step < 0:
        raise ValueError("start_step must be nonnegative")
    ratio_p01 = float(metrics["ratio_p01"])
    ratio_p99 = float(metrics["ratio_p99"])
    passed = ratio_p01 >= ratio_lower and ratio_p99 <= ratio_upper
    enforced = start_step == 0
    return {
        "protocol": "startup-logprob-drift-gate-v2",
        "start_step": start_step,
        "ratio_lower": ratio_lower,
        "ratio_upper": ratio_upper,
        "ratio_p01": ratio_p01,
        "ratio_p99": ratio_p99,
        "passed": passed,
        "enforced": enforced,
        "action": "continue" if passed or not enforced else "stop-before-update",
        "reason": "initial-backend-admission" if enforced else "recovery-observation-only",
    }
