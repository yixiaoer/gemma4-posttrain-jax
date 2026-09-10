"""Optional host-side metric tracking for training scripts."""

from __future__ import annotations

import csv
import hashlib
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
) -> Tracker:
    """Create a disabled tracker or initialise W&B with explicit failures.

    ``wandb_module`` is injectable so unit tests never import W&B or access the
    network. Normal runs import it lazily only when ``enabled`` is true.
    """

    if not enabled:
        return NullTracker()
    if wandb_module is None:
        try:
            wandb_module = importlib.import_module("wandb")
        except ModuleNotFoundError as error:
            raise RuntimeError("W&B tracking requested; install the optional dependency with `.[tracking]`") from error
    try:
        options = {} if settings is None else {"settings": dict(settings)}
        run = wandb_module.init(project=project, name=run_name, tags=list(tags), config=dict(config), **options)
    except Exception as error:
        raise RuntimeError("W&B initialisation failed; check login/network or set WANDB_MODE=offline") from error
    if run is None:
        raise RuntimeError("W&B initialisation returned no run")
    return WandbTracker(run)


def grpo_tracking_metrics(
    train: Mapping[str, Any] | None = None, evaluation: Mapping[str, Any] | None = None
) -> dict[str, int | float]:
    """只记录 host 标量；同一步的训练和评估合并后一次提交，不回写 W&B 历史。"""
    result: dict[str, int | float] = {}
    for prefix, values in (("train", train), ("eval", evaluation)):
        for key, value in (values or {}).items():
            if key in {"step", "dataset_indices"} or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(value):
                raise ValueError(f"非有限 W&B 指标：{prefix}_{key}={value}")
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


def load_grpo_history(directory: Path) -> tuple[dict[str, Any], list[tuple[int, dict[str, int | float]]]]:
    """只读 CSV/summary/meta；不读权重、逐题文本或修改原始日志。"""
    meta_path = directory / "meta.json"
    metadata = json.loads(meta_path.read_text())
    sources = [meta_path]
    rows: dict[int, dict[str, int | float]] = {}
    csv_path = directory / "metrics.csv"
    if csv_path.is_file():
        sources.append(csv_path)
        with csv_path.open() as file:
            for row in csv.DictReader(file):
                step = int(row.pop("step"))
                if step < 0 or step in rows:
                    raise ValueError(f"重复或非法 global step：{step}")
                rows[step] = grpo_tracking_metrics({key: float(value) for key, value in row.items() if value != ""})
        for path in sorted((directory / "eval").glob("step_*/summary.json")):
            step = int(path.parent.name.removeprefix("step_"))
            rows.setdefault(step, {}).update(grpo_tracking_metrics(evaluation=json.loads(path.read_text())))
            sources.append(path)
    elif (directory / "evaluation/summary.json").is_file():
        path = directory / "summary.json"
        rows[int(metadata["step"])] = grpo_tracking_metrics(evaluation=json.loads(path.read_text()))
        sources.append(path)
    else:
        raise ValueError("目录不是 GRPO metrics.csv 或独立 GSM8K evaluation 的产物")
    if not rows:
        raise ValueError("没有可导入的历史指标")
    # 编译摘要放在首个已记录的训练步；不要伪造原日志没有的耗时或训练更新。
    training_steps = [step for step, row in rows.items() if any(key.startswith("train_") for key in row)]
    compile_step = min(training_steps or list(rows))
    rows[compile_step].update(compile_tracking_metrics(directory))
    sources.extend(sorted(directory.glob("*_compile.json")))
    if (directory / "source.diff").is_file():
        sources.append(directory / "source.diff")
    config = {
        "logging_mode": "historical_import",
        "source_directory": str(directory.resolve()),
        "source_metadata": metadata,
        "source_sha256": {
            str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
        "historical_timestamps_available": False,
    }
    return config, sorted(rows.items())
