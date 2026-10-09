#!/usr/bin/env python3
"""Check that a run resumed in a new process continues the original run, and check the reload evaluation.

Meant for runs of a few hundred steps. Keeping two full checkpoints side by side is expensive, so the
continuous run does not save its final state; instead every parameter and Adam array gets a checksum
at the end of training (--audit-final-state). This script compares those records, then the training
metrics, training answers and evaluation outputs of every resumed step.

The four-step example examples/e2b_grpo_recovery.sh keeps all checkpoints and compares the raw arrays
of every rollout batch; that deeper check is scripts/compare_recovery.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

# Timings, throughput, peak memory and the first-call flag naturally differ between processes.
VOLATILE_METRICS = frozenset({"first_call", "update_mfu_estimate", "peak_hbm_gib"})


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def same_source(left: Path, right: Path) -> bool:
    """Both runs come from the same git commit with identical uncommitted changes."""
    commits = [read_json(path / "meta.json").get("git_commit") for path in (left, right)]
    diffs = [(path / "source.diff").read_text() for path in (left, right)]
    return commits[0] == commits[1] and diffs[0] == diffs[1]


def comparable_metrics(path: Path, steps: range) -> dict[int, dict[str, str]]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    require([int(row["step"]) for row in rows] == list(steps), f"{path} has missing or repeated steps")
    return {
        int(row["step"]): {
            key: value for key, value in row.items() if not key.endswith("_s") and key not in VOLATILE_METRICS
        }
        for row in rows
    }


def state_arrays(path: Path) -> list[dict[str, Any]]:
    record = read_json(path)
    require(record["complete"] and record["all_finite"], f"{path} is incomplete or the state has non-finite values")
    require(record["leaf_count"] == len(record["leaves"]) > 0, f"{path} has an inconsistent array count")
    return record["leaves"]


def compare_runs(continuous: Path, resumed: Path) -> dict[str, Any]:
    metas = [read_json(path / "meta.json") for path in (continuous, resumed)]
    require(metas[0]["run_config"] == metas[1]["run_config"], "the two runs have different configurations")
    require(same_source(continuous, resumed), "the two runs used different source code")
    require(
        read_json(continuous / "data_split.json") == read_json(resumed / "data_split.json"),
        "the two runs have different data splits",
    )
    summaries = [read_json(path / "summary.json") for path in (continuous, resumed)]
    start, resume_step, final = summaries[0]["start_step"], summaries[1]["start_step"], summaries[0]["final_step"]
    require(start < resume_step < final, "the resumed run must start inside the continuous run")
    require(summaries[1]["final_step"] == final, "the two runs did not stop at the same step")
    require(
        summaries[0]["final_data_cursor"] == summaries[1]["final_data_cursor"],
        "the two runs ended at different data cursors",
    )

    arrays = [state_arrays(path / "final_state_audit.json") for path in (continuous, resumed)]
    require(arrays[0] == arrays[1], "final parameters or Adam state differ")

    metrics = [
        comparable_metrics(continuous / "metrics.csv", range(start + 1, final + 1)),
        comparable_metrics(resumed / "metrics.csv", range(resume_step + 1, final + 1)),
    ]
    for step in range(resume_step + 1, final + 1):
        require(metrics[0][step] == metrics[1][step], f"training metrics differ at step {step}")

    samples = read_rows(resumed / "rollouts.jsonl")
    original = [row for row in read_rows(continuous / "rollouts.jsonl") if row["step"] > resume_step]
    require(
        bool(samples) and samples == original, "training answers, rewards or questions differ after the resume step"
    )

    evaluations = []
    for path in sorted((resumed / "eval").glob("step_*/predictions.jsonl")):
        step = int(path.parent.name.removeprefix("step_"))
        other = continuous / "eval" / path.parent.name / "predictions.jsonl"
        require(other.is_file(), f"the continuous run has no evaluation at step {step}")
        require(other.read_bytes() == path.read_bytes(), f"evaluation outputs differ at step {step}")
        rows = read_rows(path)
        evaluations.append({"step": step, "count": len(rows), "correct": int(sum(r["task_success"] for r in rows))})
    require(bool(evaluations), "the resumed run has no evaluations to compare")

    layout = None
    if all((path / "execution_layout.json").is_file() for path in (continuous, resumed)):
        records = [read_json(path / "execution_layout.json") for path in (continuous, resumed)]
        require(all(record["process_count"] == 1 for record in records), "training did not run in a single process")
        for key in ("mesh", "selected_backends", "requested_rollout_layout", "training_state", "rollout_params"):
            require(records[0][key] == records[1][key], f"the resumed run used different chips or shardings: {key}")
        layout = {
            "devices": [device["id"] for device in records[0]["mesh"]["devices"]],
            "training_arrays": len(records[0]["training_state"]),
        }
    return {
        "start_step": start,
        "resume_step": resume_step,
        "final_step": final,
        "state_arrays": len(arrays[0]),
        "state_bytes": sum(row["bytes"] for row in arrays[0]),
        "compared_metric_steps": final - resume_step,
        "compared_training_rows": len(samples),
        "compared_evaluations": evaluations,
        "execution_layout": layout,
        "git_commit": metas[0].get("git_commit"),
        "uncommitted_changes": metas[0].get("uncommitted_changes"),
        "run_config": metas[0]["run_config"],
    }


def check_reload(reload_dir: Path, continuous: Path, resumed: Path, final_step: int) -> dict[str, Any]:
    meta = read_json(reload_dir / "meta.json")
    require(meta["step"] == final_step, "the reload evaluation did not load the final step")
    require(
        meta["checkpoint_metadata"]["run_config"] == read_json(resumed / "meta.json")["run_config"],
        "the loaded checkpoint belongs to another run",
    )
    require(same_source(resumed, reload_dir), "the reload evaluation used different source code")
    loaded = state_arrays(reload_dir / "loaded_state_audit.json")
    require(
        loaded == state_arrays(resumed / "final_state_audit.json"),
        "the reloaded state differs from the final training state",
    )
    predictions = reload_dir / "evaluation/predictions.jsonl"
    in_training = continuous / "eval" / f"step_{final_step:08d}" / "predictions.jsonl"
    require(in_training.is_file(), "the continuous run did not evaluate at its final step")
    require(
        predictions.read_bytes() == in_training.read_bytes(),
        "reload evaluation outputs differ from the evaluation during training",
    )
    rows = read_rows(predictions)
    return {"step": final_step, "count": len(rows), "correct": int(sum(row["task_success"] for row in rows))}


def check_data_report(report_path: Path, run_config: dict[str, Any]) -> dict[str, Any]:
    report = read_json(report_path)
    require(report["passed"], "the data check failed")
    require(
        (report["dev_size"], report["dev_seed"]) == (run_config["dev_size"], run_config["dev_seed"]),
        "the data check used a different dev split",
    )
    partitions = report["partitions"]
    require(
        partitions["train"]["count"] == run_config["train_eligible_size"],
        "the data check counted a different number of training questions",
    )
    return {name: partitions[name]["count"] for name in partitions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--continuous", type=Path, required=True, help="output directory of the continuous run")
    parser.add_argument("--resumed", type=Path, required=True, help="output directory of the resumed run")
    parser.add_argument("--reload-eval", type=Path, help="output directory of the reload evaluation")
    parser.add_argument("--data-check", type=Path, help="output of scripts/check_grpo_data.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result: dict[str, Any] = {"complete": False}
    try:
        result.update(compare_runs(args.continuous, args.resumed))
        if args.reload_eval is not None:
            result["reload_eval"] = check_reload(args.reload_eval, args.continuous, args.resumed, result["final_step"])
        if args.data_check is not None:
            result["data_check"] = check_data_report(args.data_check, result["run_config"])
        result["complete"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(
        f"resumed run matches from step {result['resume_step']} to {result['final_step']}: "
        f"{result['state_arrays']} state arrays, {result['compared_metric_steps']} steps of metrics, "
        f"{result['compared_training_rows']} training answers, {len(result['compared_evaluations'])} evaluations"
    )
    if "reload_eval" in result:
        reload = result["reload_eval"]
        print(f"reload evaluation: {reload['correct']}/{reload['count']} correct, same as during training")


if __name__ == "__main__":
    main()
