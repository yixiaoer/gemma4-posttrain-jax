#!/usr/bin/env python3
"""在CPU上逐块检查完整训练状态的manifest、shape/dtype、有限值和步数。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
from safetensors import safe_open


def check_checkpoint(path: Path, expected_step: int, *, require_nonzero_adam: bool = False) -> dict[str, Any]:
    document = json.loads((path / "meta.json").read_text())
    if document["format"] != "gemma4-rl-jax-train-state" or document["version"] != 1:
        raise ValueError("不支持的checkpoint格式")
    entries = {entry["name"]: entry for entry in document["leaves"]}
    if len(entries) != len(document["leaves"]):
        raise ValueError("manifest包含重复叶子")
    lag = document["metadata"].get("run_config", {}).get("rollout_lag_updates", 0)
    if lag not in (0, 1):
        raise ValueError("只支持同步或一次Adam滞后checkpoint")
    prefix = ".train" if lag else ""
    moment_nonzero = {"mu": 0, "nu": 0}
    elements = 0
    array_bytes = 0
    with safe_open(path / "state.safetensors", framework="numpy") as stored:
        if set(stored.keys()) != entries.keys():
            raise ValueError("manifest与safetensors的叶子集合不一致")
        for name, entry in entries.items():
            view = stored.get_slice(name)
            shape = tuple(view.get_shape())
            if shape != tuple(entry["shape"]):
                raise ValueError(f"shape不一致：{name}")
            dtype = np.dtype(ml_dtypes.bfloat16 if entry["dtype"] == "bfloat16" else entry["dtype"])
            elements += math.prod(shape)
            array_bytes += math.prod(shape) * dtype.itemsize
            moment = next(
                (key for key in moment_nonzero if name.startswith(f"{prefix}.opt_state") and f".{key}." in name), None
            )
            # 按首轴切片，避免一次加载最大的PLE表或整份Adam状态。
            rows = max(1, 8 * 2**20 // (math.prod(shape[1:]) * dtype.itemsize)) if shape else 1
            for start in range(0, shape[0] if shape else 1, rows):
                value = view[start : min(start + rows, shape[0])] if shape else stored.get_tensor(name)
                if value.dtype != dtype or not np.isfinite(value).all():
                    raise ValueError(f"dtype不一致或含非有限值：{name}，首轴起点{start}")
                if moment is not None:
                    moment_nonzero[moment] += int(np.count_nonzero(value))
        step = int(stored.get_tensor(f"{prefix}.step"))
        if step != expected_step:
            raise ValueError(f"checkpoint step {step} != 预期{expected_step}")
        optimizer_counts = {
            name: int(stored.get_tensor(name))
            for name in entries
            if name.startswith(f"{prefix}.opt_state") and name.endswith(".count") and not entries[name]["shape"]
        }
        behavior_step = None
        if lag:
            behavior_step = int(stored.get_tensor(".behavior.step"))
            if step < 1 or behavior_step != step - 1:
                raise ValueError("行为快照版本必须是最后一次Adam前的step")
            training = {
                name.removeprefix(".train.params_f32"): entry["shape"]
                for name, entry in entries.items()
                if name.startswith(".train.params_f32")
            }
            behavior = {
                name.removeprefix(".behavior.params_bf16"): entry["shape"]
                for name, entry in entries.items()
                if name.startswith(".behavior.params_bf16")
            }
            if (
                not training
                or training != behavior
                or any(
                    entry["dtype"] != "bfloat16"
                    for name, entry in entries.items()
                    if name.startswith(".behavior.params_bf16")
                )
            ):
                raise ValueError("行为快照必须是完整同shape的BF16参数树")
        if not optimizer_counts or any(count != step for count in optimizer_counts.values()):
            raise ValueError(f"Adam count与全局step不一致：{optimizer_counts} / {step}")
    if require_nonzero_adam and not all(moment_nonzero.values()):
        raise ValueError(f"需要非零Adam一阶和二阶矩：{moment_nonzero}")
    return {
        "path": str(path),
        "step": step,
        "leaf_count": len(entries),
        "elements": elements,
        "array_bytes": array_bytes,
        "file_bytes": (path / "state.safetensors").stat().st_size,
        "all_finite": True,
        "adam_nonzero_elements": moment_nonzero,
        "optimizer_counts": optimizer_counts,
        "behavior_policy_step": behavior_step,
        "metadata": document["metadata"],
        "scope": "CPU逐块读取验证；不替代加载到TPU后的eval/续训检查。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-nonzero-adam", action="store_true")
    args = parser.parse_args()
    result = check_checkpoint(args.checkpoint, args.expected_step, require_nonzero_adam=args.require_nonzero_adam)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
