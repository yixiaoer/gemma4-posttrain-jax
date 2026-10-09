"""Optional host-side metric tracking for training scripts."""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol


class Tracker(Protocol):
    """Small interface kept outside every JAX-transformed function."""

    def log(self, metrics: Mapping[str, int | float | None], *, step: int) -> None: ...

    def finish(self) -> None: ...


class NullTracker:
    """No-op tracker used when optional reporting is disabled."""

    def log(self, metrics: Mapping[str, int | float | None], *, step: int) -> None:
        del metrics, step

    def finish(self) -> None:
        return None


class WandbTracker:
    """Adapter around the small subset of a W&B run that the host loop needs."""

    def __init__(self, run: Any):
        self._run = run

    def log(self, metrics: Mapping[str, int | float | None], *, step: int) -> None:
        try:
            self._run.log(dict(metrics), step=step)
        except Exception as error:
            raise RuntimeError(f"W&B failed while logging step {step}") from error

    def finish(self) -> None:
        try:
            self._run.finish()
        except Exception as error:
            raise RuntimeError("W&B failed while finishing the run") from error


def init_tracker(
    *,
    enabled: bool,
    project: str,
    run_name: str | None,
    tags: Sequence[str],
    config: Mapping[str, Any],
    wandb_module: Any | None = None,
    settings: Mapping[str, Any] | None = None,
    group: str | None = None,
    job_type: str | None = None,
) -> Tracker:
    """Create a disabled tracker or initialise W&B with explicit failures.

    ``wandb_module`` is injectable so unit tests never import W&B or access the
    network. Normal runs import it lazily only when ``enabled`` is true.
    ``group`` and ``job_type`` let one experiment keep its continuous run, resumed
    run and reload evaluation together while each stays a separate W&B run.
    """

    if not enabled:
        return NullTracker()
    if wandb_module is None:
        try:
            wandb_module = importlib.import_module("wandb")
        except ModuleNotFoundError as error:
            raise RuntimeError("W&B tracking requested; install the optional dependency with `.[tracking]`") from error
    try:
        options: dict[str, Any] = {} if settings is None else {"settings": dict(settings)}
        if group is not None:
            options["group"] = group
        if job_type is not None:
            options["job_type"] = job_type
        run = wandb_module.init(project=project, name=run_name, tags=list(tags), config=dict(config), **options)
    except Exception as error:
        raise RuntimeError("W&B initialisation failed; check login/network or set WANDB_MODE=offline") from error
    if run is None:
        raise RuntimeError("W&B initialisation returned no run")
    return WandbTracker(run)


def grpo_tracking_metrics(
    train: Mapping[str, Any] | None = None, evaluation: Mapping[str, Any] | None = None
) -> dict[str, int | float]:
    """Host scalars only; training and evaluation of one step are merged into a single log call."""
    result: dict[str, int | float] = {}
    for prefix, values in (("train", train), ("eval", evaluation)):
        for key, value in (values or {}).items():
            if key in {"step", "dataset_indices"} or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(value):
                raise ValueError(f"non-finite W&B metric: {prefix}_{key}={value}")
            group = "time" if prefix == "train" and key.endswith("_s") and not key.endswith("_per_s") else prefix
            result[f"{group}_{key}"] = value
    return result


def compile_tracking_metrics(directory: Path) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for path in sorted(directory.glob("*_compile.json")):
        item = json.loads(path.read_text())
        name = path.name.removesuffix("_compile.json")
        result[f"compile_{name}_s"] = float(item["compile_s"])
        for key, value in item["memory"].items():
            if isinstance(value, (int, float)):
                result[f"compile_{name}_{key}"] = value
    return result
