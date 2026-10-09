#!/usr/bin/env python3
"""Compare full training state, rollout records and reload evaluations from the four-step resume example."""

from __future__ import annotations

import argparse
import filecmp
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .check_checkpoint import check_checkpoint
else:
    from check_checkpoint import check_checkpoint

from gemma4_posttrain_jax.checkpoint import read_checkpoint_metadata

CANDIDATE_FIELDS = frozenset(
    {
        "protocol",
        "backend",
        "data_cursor",
        "policy_step",
        "target_old_policy_step",
        "dataset_indices",
        "key_data",
        "prompt_ids",
        "prompt_mask",
        "completion_ids",
        "completion_mask",
        "lengths",
        "raw_logprobs",
        "proposal_logprobs",
        "proposal_known",
        "temperature",
        "top_k",
        "top_p",
        "eos_ids",
    }
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def same_source(left: Path, right: Path) -> bool:
    """Both runs come from the same git commit with identical uncommitted changes."""
    commits = [read_json(path / "meta.json").get("git_commit") for path in (left, right)]
    diffs = [(path / "source.diff").read_text() for path in (left, right)]
    return commits[0] == commits[1] and diffs[0] == diffs[1]


def check_training_sampling_protocol(config: dict[str, Any]) -> str:
    """Check the training backend's sampling protocol; reload evaluations are checked against the native sampler."""
    protocols = {"jax": "fp32-exp-log-highest-v1", "tpu-inference-separate-process-v1": "engine-v1"}
    # Native train_grpo records omit this field; engine records use the protocol name, not the CLI option.
    backend = config.get("rollout_backend", "jax")
    if not isinstance(backend, str) or backend not in protocols:
        raise ValueError("resume comparison does not support this training backend")
    expected = protocols[backend]
    require(config.get("sampling_math_protocol") == expected, "training backend and sampling protocol do not match")
    return expected


def compare_candidate_files(baseline: Path, resumed: Path, *, expected_backend: str | None = None) -> dict[str, Any]:
    """Compare arrays by dtype, shape and bytes, including signed zeros, random state and discarded samples."""
    with np.load(baseline, allow_pickle=False) as left, np.load(resumed, allow_pickle=False) as right:
        require(set(left.files) == set(right.files), f"candidate fields differ: {resumed.name}")
        require(
            set(left.files) >= CANDIDATE_FIELDS,
            f"candidate is missing fields: {sorted(CANDIDATE_FIELDS - set(left.files))}",
        )
        require(left["protocol"].item() == "rollout-candidate-arrays-v1", "candidate record protocol does not match")
        if expected_backend is not None:
            require(
                left["backend"].item() == expected_backend,
                "candidate backend does not match the training configuration",
            )
        require(
            left["completion_ids"].ndim == 2 and len(left["completion_ids"]) > 0,
            "candidate samples are empty or have the wrong shape",
        )
        for field in left.files:
            a, b = left[field], right[field]
            require(
                a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
                f"candidate arrays differ: {resumed.name}/{field}",
            )
        return {"file": resumed.name, "fields": sorted(left.files), "rows": len(left["completion_ids"])}


def check_evaluation_records(
    root: Path,
    train_meta: dict[str, Any],
    final_cursor: int,
    *,
    expected_count: int = 4,
    expected_max_new_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Check the greedy reload evaluations of the final models against the expected question count and length."""
    require(type(expected_count) is int and expected_count > 0, "evaluation question count must be a positive integer")
    require(
        expected_max_new_tokens is None or (type(expected_max_new_tokens) is int and expected_max_new_tokens > 0),
        "evaluation length must be a positive integer",
    )
    paths = [root / f"eval-{run}" for run in ("baseline", "resume")]
    prediction_files = [path / "evaluation/predictions.jsonl" for path in paths]
    require(prediction_files[0].read_bytes() == prediction_files[1].read_bytes(), "the two reload evaluations differ")
    predictions = read_rows(prediction_files[0])
    indices = [row["dataset_index"] for row in predictions]
    require(
        len(indices) == expected_count and len(set(indices)) == expected_count,
        "evaluation question count is not the expected one, or has duplicates",
    )
    training = set(read_json(root / "baseline/data_split.json")["training_indices"])
    require(not training.intersection(indices), "evaluation questions overlap the training set")
    metas = [read_json(path / "meta.json") for path in paths]
    for meta, path in zip(metas, paths, strict=True):
        require(meta["step"] == 4, "evaluation did not load the step-4 state")
        require(
            meta["checkpoint_metadata"]["run_config"] == train_meta["run_config"],
            "evaluation model has a different training configuration",
        )
        require(
            meta["checkpoint_metadata"]["data_cursor"] == final_cursor, "evaluation state has a different data cursor"
        )
        require(
            meta["split"] == "train" and meta["subset"] == "train-dev",
            "evaluation set is not the held-out train-dev set",
        )
        require(
            meta["dataset_fingerprint"] == train_meta["run_config"]["dataset_fingerprint"],
            "evaluation data comes from a different dataset",
        )
        require(
            read_json(path / "data_selection.json")["indices"] == indices,
            "evaluation question IDs or order differ from the selection record",
        )
        require(meta["prediction_record_schema"] == "raw-token-ids-v1", "evaluation lacks the raw token record schema")
        require(meta["sampling_math_protocol"] == "fp32-exp-log-highest-v1", "evaluation sampling protocol differs")
        require(meta["sampler"].get("temperature") == 0, "evaluation must use greedy decoding")
        if expected_max_new_tokens is not None:
            require(
                meta["sampler"].get("max_new_tokens") == expected_max_new_tokens,
                "evaluation generation length differs from the expected one",
            )
        summary = read_json(path / "summary.json")
        require(summary["step"] == 4 and summary["count"] == expected_count, "evaluation did not finish")
        require(summary["dataset_indices"] == indices, "evaluation summary lists different questions")
        require(same_source(root / "baseline", path), "evaluation used different source code from training")
    for field in ("sampler", "jax_default_matmul_precision", "git_commit"):
        require(metas[0][field] == metas[1][field], f"the two evaluation settings differ: {field}")
    for row in predictions:
        require(
            len(row["completion_token_ids"]) == row["length"],
            "evaluation answer token count differs from its recorded length",
        )
        require(row["length"] > 0, "evaluation answer has no tokens")
        if expected_max_new_tokens is not None:
            require(row["length"] <= expected_max_new_tokens, "evaluation answer is longer than the expected length")
        require(bool(row["prompt_token_ids"]), "evaluation record has no prompt tokens")
    return predictions


def check_engine_records(root: Path) -> dict[str, Any]:
    """Check the resumed engine's random state, complete weight commits and both process exits."""

    def canonical(value: Any) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    def generation(row: dict[str, Any]) -> dict[str, Any]:
        fields = (
            "data_cursor",
            "step",
            "rows",
            "policy_version",
            "rng_protocol",
            "batch_rng",
            "proposal_logprobs_available",
            "stops",
        )
        return {key: row[key] for key in fields} | {
            "after_key": row["engine"]["batch_rng_after"],
            "installed_key": row["engine"]["batch_rng_install"]["installed"],
            "sampling": row["engine"]["sampling"],
        }

    baseline, resume = root / "baseline", root / "resume"
    original = [generation(row) for row in read_rows(baseline / "inference.jsonl") if row["step"] >= 2]
    restored = [generation(row) for row in read_rows(resume / "inference.jsonl")]
    require(
        bool(restored) and canonical(original) == canonical(restored),
        "engine generations or random state differ after resume",
    )
    for version in (2, 3):
        reports = [read_json(path / f"inference_weights_{version:06d}.json") for path in (baseline, resume)]
        targets = []
        for report in reports:
            require(
                report["complete"] and report["host_transport"]["device_commit"],
                "engine weights were not committed to the device",
            )
            updated = report["mapping"]["updated"]
            mapped = {row["name"]: row["host_sha256"] for row in updated}
            require(len(updated) == len(mapped) == 505, "E2B target weight record is incomplete or has duplicate names")
            require(len(report["host_transport"]["leaves"]) == 540, "E2B source parameter record is incomplete")
            require(sum(len(row["addressable_shards"]) for row in updated) == 1010, "engine shard check is incomplete")
            require(
                all(shard["exact"] for row in updated for shard in row["addressable_shards"]),
                "engine target weight contents differ",
            )
            targets.append(mapped)
        require(targets[0] == targets[1], "BF16 target weights differ after resume")
        require(
            canonical(reports[0]["host_transport"]["leaves"]) == canonical(reports[1]["host_transport"]["leaves"]),
            "FP32 source parameters differ after resume",
        )
    workers = []
    for path, versions in ((baseline, [0, 1, 2, 3]), (resume, [2, 3])):
        close = read_json(path / "inference_close.json")
        require(
            close["complete"] and close["worker_stopped"] and close["worker_exit_code"] == 0,
            "engine shutdown did not complete",
        )
        require(
            not close["worker_wait"]["timed_out"] and not close["forced_group_termination"],
            "engine shutdown exceeded its wait budget",
        )
        group = close["owned_group_cleanup"]
        require(
            group["complete"] and not group["remaining_live_members"] and not group["signals"],
            "engine process group did not exit on its own",
        )
        require(
            close["remote_close"]["scheduler_shutdown"]["complete"]
            and not close["remote_close"]["global_core_shutdown_used"],
            "engine scheduler shutdown did not meet the isolation requirement",
        )
        worker = read_json(path / "inference_worker.json")
        requests = worker["requests"]
        require(worker["complete"] and not worker["errors"] and bool(requests), "engine request record is incomplete")
        require(all(row["complete"] for row in requests), "some engine requests did not complete")
        require(
            [row["request_id"] for row in requests] == list(range(len(requests))),
            "engine request IDs are not consecutive",
        )
        require(
            [row["version"] for row in requests if row["operation"] == "update"] == versions,
            "engine weight versions are in a different order",
        )
        require(requests[-1]["operation"] == "close", "engine requests did not end with close")
        require(
            sum(row["operation"] == "generate" for row in requests) == len(read_rows(path / "inference.jsonl")),
            "engine generation records are missing",
        )
        require(
            read_json(path / "inference_initialization.json")["metadata"]["environment_identity_verified"],
            "inference environment identity was not verified",
        )
        workers.append(worker)
    require(workers[0]["pid"] != workers[1]["pid"], "resume did not start a new engine process")
    require(
        workers[0]["tpu_bootstrap"] == workers[1]["tpu_bootstrap"],
        "engine runtime libraries or device settings differ after resume",
    )
    return {
        "complete": True,
        "resumed_generations": len(restored),
        "compared_weight_versions": [2, 3],
        "worker_pids": [row["pid"] for row in workers],
    }


def check_backend_records(root: Path, config: dict[str, Any]) -> dict[str, Any] | None:
    """Only the validated native configuration may skip the engine lifecycle records."""
    protocol = check_training_sampling_protocol(config)
    return check_engine_records(root) if protocol == "engine-v1" else None


def compare_recovery(
    root: Path,
    *,
    rescore: bool = False,
    evaluation_count: int = 4,
    evaluation_max_new_tokens: int | None = None,
) -> dict[str, Any]:
    metas = [read_json(root / run / "meta.json") for run in ("baseline", "resume")]
    require(metas[0]["run_config"] == metas[1]["run_config"], "resumed run has a different configuration")
    require(same_source(root / "baseline", root / "resume"), "resumed run used different source code")
    sampling_protocol = check_training_sampling_protocol(metas[0]["run_config"])
    checkpoints = {}
    for run, step in [("baseline", 2), ("baseline", 4), ("resume", 4)]:
        path = root / f"{run}-checkpoints" / f"step_{step:08d}"
        inspection = check_checkpoint(path, step, require_nonzero_adam=True)
        checkpoints[f"{run}-step{step}"] = {
            "file_bytes": inspection["file_bytes"],
            "leaf_count": inspection["leaf_count"],
            "all_finite": inspection["all_finite"],
            "adam_nonzero_elements": inspection["adam_nonzero_elements"],
            "data_cursor": inspection["metadata"]["data_cursor"],
            "format": read_checkpoint_metadata(path)["format"],
        }
    require(
        filecmp.cmp(
            root / "baseline-checkpoints/step_00000004/state.safetensors",
            root / "resume-checkpoints/step_00000004/state.safetensors",
            shallow=False,
        ),
        "final state files differ",
    )
    summaries = [read_json(root / run / "summary.json") for run in ("baseline", "resume")]
    require(summaries[0]["start_step"] == 0 and summaries[1]["start_step"] == 2, "unexpected start steps")
    require(all(s["final_step"] == 4 for s in summaries), "training did not reach step 4")
    start, end = checkpoints["baseline-step2"]["data_cursor"], checkpoints["baseline-step4"]["data_cursor"]
    require(
        type(start) is int and type(end) is int and 0 <= start < end,
        "data cursor did not advance during the resumed run",
    )
    require(checkpoints["resume-step4"]["data_cursor"] == end, "resumed run ended at a different data cursor")
    names = [f"batch_{index:08d}.npz" for index in range(start, end)]
    require(
        sorted(p.name for p in (root / "resume/rollout_batches").glob("*.npz")) == names,
        "resumed candidate batches are incomplete",
    )
    candidates = [
        compare_candidate_files(
            root / "baseline/rollout_batches" / name,
            root / "resume/rollout_batches" / name,
            expected_backend="inference-process" if sampling_protocol == "engine-v1" else "jax",
        )
        for name in names
    ]
    original_rows = [row for row in read_rows(root / "baseline/rollouts.jsonl") if row["step"] > 2]
    require(original_rows == read_rows(root / "resume/rollouts.jsonl"), "resumed answers, rewards or selections differ")
    for run, steps in [("baseline", range(1, 5)), ("resume", range(3, 5))]:
        for step in steps:
            witness = read_json(root / run / f"update_witness_{step:08d}.json")
            require(
                witness["all_finite"] and witness["changed_elements"] > 0,
                "no valid non-zero parameter update was observed",
            )
            require(
                witness["before_step"] == step - 1 and witness["after_step"] == step,
                "update witness has mismatched steps",
            )
        for name in ("startup_drift.json", "joint_old_drift.json"):
            drift = read_json(root / run / name)
            require(
                drift["ratio_p01"] >= 0.9 and drift["ratio_p99"] <= 1.1,
                "probability ratio quantiles are outside the allowed range",
            )
    for name in ("final_state_audit.json", "update_witness_00000003.json", "update_witness_00000004.json"):
        require(
            read_json(root / "baseline" / name) == read_json(root / "resume" / name),
            f"state check records differ: {name}",
        )
    predictions = check_evaluation_records(
        root, metas[0], end, expected_count=evaluation_count, expected_max_new_tokens=evaluation_max_new_tokens
    )
    engine = check_backend_records(root, metas[0]["run_config"])
    if rescore:
        from transformers import AutoTokenizer

        from gemma4_posttrain_jax.rewards import score_completions

        tokenizer = AutoTokenizer.from_pretrained(metas[0]["run_config"]["model_path"], local_files_only=True)
        decoded = tokenizer.batch_decode([row["completion_token_ids"] for row in predictions], skip_special_tokens=True)
        require(
            decoded == [row["completion"] for row in predictions], "re-decoded tokens do not match the recorded text"
        )
        scores = score_completions(
            decoded,
            [row["gold"] for row in predictions],
            completion_lengths=[row["length"] for row in predictions],
            workers=1,
        )
        require(
            np.array_equal(scores.task_success, [row["task_success"] for row in predictions]),
            "re-scored task success differs",
        )
        require(
            np.array_equal(scores.format_reward, [row["format_reward"] for row in predictions]),
            "re-scored format reward differs",
        )
    return {
        "complete": True,
        "scope": f"E2B four-step recovery and {evaluation_count}-question reload evaluation; engine checked separately",
        "evaluation_protocol": {
            "expected_count": evaluation_count,
            "expected_max_new_tokens": evaluation_max_new_tokens,
            "greedy_temperature": 0,
        },
        "training_sampling_math_protocol": sampling_protocol,
        "evaluation_sampling_math_protocol": "fp32-exp-log-highest-v1",
        "checkpoints": checkpoints,
        "engine": engine,
        "resumed_candidates": candidates,
        "evaluation_rows": len(predictions),
        "evaluation_successes": sum(row["task_success"] for row in predictions),
        "tokens_decoded_and_rescored": rescore,
        "git_commit": metas[0]["git_commit"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="re-decode and re-score the recorded tokens with the local tokenizer (needs the hf extra)",
    )
    parser.add_argument(
        "--output", type=Path, help="defaults to ROOT/recovery_check.json; an existing report is never overwritten"
    )
    parser.add_argument(
        "--evaluation-count", type=int, default=4, help="expected number of evaluation questions (4 in the example)"
    )
    parser.add_argument(
        "--evaluation-max-new-tokens",
        type=int,
        help="also check the configured generation length and the actual answer lengths",
    )
    args = parser.parse_args()
    output = args.output or args.root / "recovery_check.json"
    if output.exists():
        raise FileExistsError(output)
    result = compare_recovery(
        args.root,
        rescore=args.rescore,
        evaluation_count=args.evaluation_count,
        evaluation_max_new_tokens=args.evaluation_max_new_tokens,
    )
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "resumed_batches": len(result["resumed_candidates"]), "output": str(output)}))


if __name__ == "__main__":
    main()
