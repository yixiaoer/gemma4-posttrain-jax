#!/usr/bin/env python3
"""Summarize a GRPO run: the evaluation curve, per-question changes, and where the time went.

Reads only small files in the run directory (metrics.csv, iteration_timing.jsonl, eval/, summary.json);
no model and no TPU. With the stages.jsonl written by the example script it also splits the time of
the whole process.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from gemma4_posttrain_jax.jsonl import read_jsonl

# Same as accounted_fields in train_grpo.py: together with unattributed_s these add up to the
# iteration time before metrics are logged.
STAGE_FIELDS = (
    "reshard_s",
    "prompt_prepare_s",
    "rollout_s",
    "rollout_device_get_s",
    "rollout_audit_s",
    "text_decode_s",
    "reward_s",
    "rollout_record_s",
    "batch_prepare_s",
    "startup_logps_s",
    "reference_transfer_s",
    "reference_logps_s",
    "behavior_snapshot_s",
    "update_s",
    "update_audit_s",
    "checkpoint_s",
    "eval_s",
)


def distribution(values: list[float]) -> dict[str, float | int]:
    """Mean, median and linearly interpolated percentiles over steps (different inputs, not repeated measurements)."""
    ordered = sorted(values)
    if not ordered or not all(math.isfinite(value) for value in ordered):
        raise ValueError("statistics input is empty or non-finite")

    def percentile(q: float) -> float:
        position = q * (len(ordered) - 1)
        lower, upper = math.floor(position), math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "count": len(ordered),
        "sum_s": math.fsum(ordered),
        "mean_s": statistics.fmean(ordered),
        "median_s": statistics.median(ordered),
        "min_s": ordered[0],
        "p90_s": percentile(0.9),
        "max_s": ordered[-1],
    }


def paired_change(initial: list[dict[str, Any]], current: list[dict[str, Any]]) -> dict[str, Any]:
    """Questions that went from wrong to right, right to wrong or stayed the same, with a 95% bootstrap interval
    over questions."""
    before = {int(row["dataset_index"]): int(row["task_success"]) for row in initial}
    after = {int(row["dataset_index"]): int(row["task_success"]) for row in current}
    if not before or len(before) != len(initial) or len(after) != len(current) or before.keys() != after.keys():
        raise ValueError("both evaluations must cover the same unique questions")
    if not set(before.values()) | set(after.values()) <= {0, 1}:
        raise ValueError("task_success must be 0 or 1")
    differences = np.asarray([after[key] - before[key] for key in before])
    counts = np.asarray([(differences == value).sum() for value in (-1, 0, 1)])
    draws = np.random.default_rng(0).multinomial(len(differences), counts / len(differences), size=20_000)
    interval = np.quantile((draws[:, 2] - draws[:, 0]) / len(differences), [0.025, 0.975])
    return {
        "count": len(differences),
        "wrong_to_right": int(counts[2]),
        "right_to_wrong": int(counts[0]),
        "unchanged": int(counts[1]),
        "accuracy_change": float(differences.mean()),
        "bootstrap_95pct": interval.tolist(),
    }


def evaluation_curve(run_dir: Path) -> list[dict[str, Any]]:
    paths = sorted((run_dir / "eval").glob("step_*/summary.json"))
    if not paths:
        return []
    initial = read_jsonl(paths[0].with_name("predictions.jsonl"))
    curve = []
    for path in paths:
        summary = json.loads(path.read_text())
        predictions = read_jsonl(path.with_name("predictions.jsonl"))
        correct = sum(int(row["task_success"]) for row in predictions)
        if len(predictions) != summary["count"] or not math.isclose(correct / len(predictions), summary["accuracy"]):
            raise ValueError(f"evaluation summary does not match its predictions: {path}")
        curve.append(
            {
                "step": int(path.parent.name.removeprefix("step_")),
                "correct": correct,
                "count": len(predictions),
                "accuracy": correct / len(predictions),
                "mean_length": summary["mean_length"],
                "truncated": sum(bool(row["truncated"]) for row in predictions),
                "change_from_first_eval": paired_change(initial, predictions),
            }
        )
    return curve


def read_metrics(run_dir: Path) -> list[dict[str, float]]:
    with (run_dir / "metrics.csv").open(newline="") as file:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(file)]
    steps = [int(row["step"]) for row in rows]
    if not rows or steps != list(range(steps[0], steps[0] + len(steps))):
        raise ValueError("metrics.csv is empty or has missing or repeated steps")
    return rows


def cost_breakdown(run_dir: Path, metrics: list[dict[str, float]], process_wall_s: float | None) -> dict[str, Any]:
    timings = read_jsonl(run_dir / "iteration_timing.jsonl")
    summary = json.loads((run_dir / "summary.json").read_text())
    if [int(row["step"]) for row in timings] != [int(row["step"]) for row in metrics]:
        raise ValueError("iteration_timing.jsonl and metrics.csv cover different steps")
    full = [float(row["iteration_after_metrics_log_s"]) for row in timings]
    stage_sums = {field: math.fsum(row.get(field, 0.0) for row in metrics) for field in STAGE_FIELDS}
    stage_sums["unattributed_s"] = math.fsum(row["unattributed_s"] for row in metrics)
    stage_sums["metrics_logging_s"] = math.fsum(float(row["metrics_logging_s"]) for row in timings)
    periodic = [i for i, row in enumerate(metrics) if row["checkpoint_s"] or row["eval_s"]]
    later = list(range(1, len(metrics)))
    without_periodic = [full[i] - metrics[i]["checkpoint_s"] - metrics[i]["eval_s"] for i in later]
    later_total = math.fsum(without_periodic)
    tokens = sum(int(metrics[i]["generated_tokens"]) for i in later)
    compile_records = {
        path.name.removesuffix("_compile.json"): float(json.loads(path.read_text())["compile_s"])
        for path in sorted(run_dir.glob("*_compile.json"))
    }
    result: dict[str, Any] = {
        "iterations": {
            "all_steps": distribution(full),
            "after_first_step": distribution([full[i] for i in later]) if later else None,
            "after_first_step_minus_save_and_eval": distribution(without_periodic) if later else None,
            "after_first_step_without_save_or_eval": distribution([full[i] for i in later if i not in periodic])
            if [i for i in later if i not in periodic]
            else None,
        },
        "steps_with_save_or_eval": [int(metrics[i]["step"]) for i in periodic],
        "stage_sums_s": stage_sums,
        "stage_sum_residual_s": math.fsum(full) - math.fsum(stage_sums.values()),
        "after_first_step_shares_percent": {
            "rollout": 100 * math.fsum(metrics[i]["rollout_s"] for i in later) / later_total,
            "update": 100 * math.fsum(metrics[i]["update_s"] for i in later) / later_total,
        }
        if later
        else None,
        "after_first_step_generated_tokens_per_s": tokens / later_total if later else None,
        "first_step": {
            "iteration_s": full[0],
            "rollout_s": metrics[0]["rollout_s"],
            "update_s": metrics[0]["update_s"],
            "unattributed_s": metrics[0]["unattributed_s"],
        },
        "compile_s": compile_records,
        "training_session_s": summary["training_session_s"],
        "initial_eval_s": summary["initial_eval_s"],
        "cleanup_s": summary["cleanup_s"],
        "notes": [
            "Step 1 includes the first compilation of rollout, startup_logps and update and is listed separately.",
            "'minus save and eval' subtracts the measured save/eval time; it is not a run without them.",
            "Each step has different questions and answer lengths, so these are samples along one trajectory.",
        ],
    }
    if process_wall_s is not None:
        evaluations = (summary["initial_eval_s"] or 0.0) + stage_sums["eval_s"]
        result["process"] = {
            "wall_s": process_wall_s,
            "outside_training_session_s": process_wall_s - summary["training_session_s"],
            "shares_percent": {
                name: 100 * value / process_wall_s
                for name, value in {
                    "rollout": stage_sums["rollout_s"],
                    "update": stage_sums["update_s"],
                    "evaluations_including_initial": evaluations,
                    "checkpoints": stage_sums["checkpoint_s"],
                    "first_step_compiles": math.fsum(
                        compile_records.get(name, 0.0) for name in ("rollout", "startup_logps", "update")
                    ),
                    "cleanup": summary["cleanup_s"],
                }.items()
            },
        }
    return result


def stage_wall_s(stages_path: Path, stage: str) -> float:
    rows = [row for row in read_jsonl(stages_path) if row["stage"] == stage]
    if len(rows) != 1 or rows[0]["exit_code"] != 0:
        raise ValueError(f"{stages_path} has no single successful stage named {stage}")
    return (rows[0]["ended_unix_ns"] - rows[0]["started_unix_ns"]) / 1e9


def summarize_run(run_dir: Path, *, process_wall_s: float | None = None) -> dict[str, Any]:
    meta = json.loads((run_dir / "meta.json").read_text())
    metrics = read_metrics(run_dir)
    config = meta["run_config"]
    rows_per_update = int(config["prompt_batch_size"]) * int(config["group_size"])
    truncated = [row["truncated_fraction"] * rows_per_update for row in metrics]
    if any(abs(value - round(value)) > 1e-6 for value in truncated):
        raise ValueError("truncated_fraction does not map to whole rows; the configuration may not match the records")
    return {
        "run_dir": str(run_dir.resolve()),
        "git_commit": meta.get("git_commit"),
        "uncommitted_changes": meta.get("uncommitted_changes"),
        "first_step": int(metrics[0]["step"]),
        "final_step": int(metrics[-1]["step"]),
        "training": {
            "updates": len(metrics),
            "generated_rows": int(sum(row["generated_rows"] for row in metrics)),
            "generated_tokens": int(sum(row["generated_tokens"] for row in metrics)),
            "truncated_training_rows": int(sum(round(value) for value in truncated)),
            "mean_task_success": statistics.fmean(row["task_success"] for row in metrics),
            "mean_mixed_success_group_fraction": statistics.fmean(
                row["mixed_success_group_fraction"] for row in metrics
            ),
        },
        "evaluation_curve": evaluation_curve(run_dir),
        "cost": cost_breakdown(run_dir, metrics, process_wall_s),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--stages", type=Path, help="stages.jsonl written by the example script, for the whole-process time"
    )
    parser.add_argument("--stage", help="stage name of run_dir in stages.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.stages is None) != (args.stage is None):
        parser.error("--stages and --stage must be given together")
    if args.output.exists():
        raise FileExistsError(args.output)
    wall = None if args.stages is None else stage_wall_s(args.stages, args.stage)
    result = summarize_run(args.run_dir, process_wall_s=wall)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for point in result["evaluation_curve"]:
        change = point["change_from_first_eval"]
        print(
            f"step {point['step']:>5}: {point['correct']}/{point['count']} = {point['accuracy']:.2%}"
            f"  (wrong->right {change['wrong_to_right']}, right->wrong {change['right_to_wrong']})"
        )
    later = result["cost"]["iterations"]["after_first_step_minus_save_and_eval"]
    if later is not None:
        shares = result["cost"]["after_first_step_shares_percent"]
        print(
            f"from step 2: {later['mean_s']:.3f} s per iteration without save/eval, "
            f"rollout {shares['rollout']:.1f}%, update {shares['update']:.1f}%"
        )


if __name__ == "__main__":
    main()
