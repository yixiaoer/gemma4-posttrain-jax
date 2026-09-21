#!/usr/bin/env python3
"""比较恢复示例中的完整训练状态、生成样本和独立评估结果。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .check_checkpoint import check_checkpoint
else:
    from check_checkpoint import check_checkpoint

from gemma4_posttrain_jax.checkpoint import read_checkpoint_metadata
from gemma4_posttrain_jax.evaluation import indices_sha256

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


def file_sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def check_training_sampling_protocol(config: dict[str, Any]) -> str:
    """训练按实际后端核协议；两边重载后的共同评估仍单独核原生采样器。"""
    protocols = {"jax": "fp32-exp-log-highest-v1", "tpu-inference-separate-process-v1": "engine-v1"}
    # train_grpo 原生记录省略此字段；引擎记录使用协议名称，不是CLI选项名称。
    backend = config.get("rollout_backend", "jax")
    if not isinstance(backend, str) or backend not in protocols:
        raise ValueError("恢复比较尚未支持该训练后端")
    expected = protocols[backend]
    require(config.get("sampling_math_protocol") == expected, "训练后端与采样协议不一致")
    return expected


def compare_candidate_files(baseline: Path, resumed: Path, *, expected_backend: str | None = None) -> dict[str, Any]:
    """按类型、形状和字节比较数组，包括正负零、随机状态和被丢弃的样本。"""
    with np.load(baseline, allow_pickle=False) as left, np.load(resumed, allow_pickle=False) as right:
        require(set(left.files) == set(right.files), f"候选字段不同: {resumed.name}")
        require(set(left.files) >= CANDIDATE_FIELDS, f"候选缺少必需字段: {sorted(CANDIDATE_FIELDS - set(left.files))}")
        require(left["protocol"].item() == "rollout-candidate-arrays-v1", "候选记录协议不符")
        if expected_backend is not None:
            require(left["backend"].item() == expected_backend, "候选后端与训练配置不一致")
        require(left["completion_ids"].ndim == 2 and len(left["completion_ids"]) > 0, "候选样本为空或形状错误")
        for field in left.files:
            a, b = left[field], right[field]
            require(
                a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
                f"候选数组不同: {resumed.name}/{field}",
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
    """按登记的题数和可选长度检查末步模型的共同greedy评估。"""
    require(type(expected_count) is int and expected_count > 0, "评估题数必须为正整数")
    require(
        expected_max_new_tokens is None or (type(expected_max_new_tokens) is int and expected_max_new_tokens > 0),
        "评估长度必须为正整数",
    )
    paths = [root / f"eval-{run}" for run in ("baseline", "resume")]
    prediction_files = [path / "evaluation/predictions.jsonl" for path in paths]
    require(prediction_files[0].read_bytes() == prediction_files[1].read_bytes(), "独立重载评估不同")
    predictions = read_rows(prediction_files[0])
    indices = [row["dataset_index"] for row in predictions]
    require(len(indices) == expected_count and len(set(indices)) == expected_count, "评估题数与登记不符或存在重复")
    training = set(read_json(root / "baseline/data_split.json")["training_indices"])
    require(not training.intersection(indices), "评估与训练重叠")
    metas = [read_json(path / "meta.json") for path in paths]
    for meta, path in zip(metas, paths, strict=True):
        require(meta["step"] == 4, "评估未加载第4步状态")
        require(meta["checkpoint_metadata"]["run_config"] == train_meta["run_config"], "评估模型训练配置不同")
        require(meta["checkpoint_metadata"]["data_cursor"] == final_cursor, "评估状态数据游标不同")
        require(meta["split"] == "train" and meta["subset"] == "train-dev", "评估集合不是预留训练验证集")
        require(meta["dataset_fingerprint"] == train_meta["run_config"]["dataset_fingerprint"], "评估数据来源不同")
        require(meta["indices_sha256"] == indices_sha256(indices), "评估题号或顺序与记录不同")
        require(meta["prediction_record_schema"] == "raw-token-ids-v1", "评估缺少实际token记录协议")
        require(meta["sampling_math_protocol"] == "fp32-exp-log-highest-v1", "评估采样协议不同")
        require(meta["sampler"].get("temperature") == 0, "评估必须使用登记的greedy采样")
        if expected_max_new_tokens is not None:
            require(
                meta["sampler"].get("max_new_tokens") == expected_max_new_tokens,
                "评估生成长度与登记不符",
            )
        summary = read_json(path / "summary.json")
        require(summary["step"] == 4 and summary["count"] == expected_count, "评估未完整结束")
        require(summary["dataset_indices"] == indices, "评估汇总题号不同")
        for name, digest in train_meta["source_files_sha256"].items():
            if name.startswith("gemma4_posttrain_jax/"):
                require(meta["source_files_sha256"].get(name) == digest, f"评估使用了不同的源码: {name}")
    for field in ("sampler", "jax_default_matmul_precision", "source_files_sha256"):
        require(metas[0][field] == metas[1][field], f"两次评估设置不同: {field}")
    for row in predictions:
        require(len(row["completion_token_ids"]) == row["length"], "实际token数与长度不同")
        require(row["length"] > 0, "评估回答没有实际token")
        if expected_max_new_tokens is not None:
            require(row["length"] <= expected_max_new_tokens, "评估实际token数超过登记长度")
        require(bool(row["prompt_token_ids"]), "缺少实际prompt token")
    return predictions


def check_engine_records(root: Path) -> dict[str, Any]:
    """检查恢复后的引擎随机状态、完整权重提交和两次进程退出。"""

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
    require(bool(restored) and canonical(original) == canonical(restored), "引擎生成或随机状态恢复不同")
    for version in (2, 3):
        reports = [read_json(path / f"inference_weights_{version:06d}.json") for path in (baseline, resume)]
        targets = []
        for report in reports:
            require(report["complete"] and report["host_transport"]["device_commit"], "引擎权重未完成设备提交")
            updated = report["mapping"]["updated"]
            mapped = {row["name"]: row["host_sha256"] for row in updated}
            require(len(updated) == len(mapped) == 505, "E2B目标权重记录不完整或名称重复")
            require(len(report["host_transport"]["leaves"]) == 540, "E2B源参数记录不完整")
            require(sum(len(row["addressable_shards"]) for row in updated) == 1010, "引擎分片检查不完整")
            require(
                all(shard["exact"] for row in updated for shard in row["addressable_shards"]), "引擎目标权重内容不同"
            )
            targets.append(mapped)
        require(targets[0] == targets[1], "恢复后的BF16目标权重不同")
        require(
            canonical(reports[0]["host_transport"]["leaves"]) == canonical(reports[1]["host_transport"]["leaves"]),
            "恢复后的FP32源参数不同",
        )
    workers = []
    for path, versions in ((baseline, [0, 1, 2, 3]), (resume, [2, 3])):
        close = read_json(path / "inference_close.json")
        require(close["complete"] and close["worker_stopped"] and close["worker_exit_code"] == 0, "引擎关闭未完成")
        require(
            not close["worker_wait"]["timed_out"] and not close["forced_group_termination"], "引擎关闭超过原等待预算"
        )
        group = close["owned_group_cleanup"]
        require(
            group["complete"] and not group["remaining_live_members"] and not group["signals"], "引擎进程组未自然退出"
        )
        require(
            close["remote_close"]["scheduler_shutdown"]["complete"]
            and not close["remote_close"]["global_core_shutdown_used"],
            "引擎调度器退出不符合隔离要求",
        )
        worker = read_json(path / "inference_worker.json")
        requests = worker["requests"]
        require(worker["complete"] and not worker["errors"] and bool(requests), "引擎请求记录不完整")
        require(all(row["complete"] for row in requests), "存在未完成的引擎请求")
        require([row["request_id"] for row in requests] == list(range(len(requests))), "引擎请求顺序不连续")
        require(
            [row["version"] for row in requests if row["operation"] == "update"] == versions, "引擎权重版本顺序不同"
        )
        require(requests[-1]["operation"] == "close", "引擎请求没有正常结束")
        require(
            sum(row["operation"] == "generate" for row in requests) == len(read_rows(path / "inference.jsonl")),
            "引擎生成记录缺失",
        )
        require(
            read_json(path / "inference_initialization.json")["metadata"]["environment_identity_verified"],
            "推理环境身份未确认",
        )
        workers.append(worker)
    require(workers[0]["pid"] != workers[1]["pid"], "恢复未启动新的引擎进程")
    require(workers[0]["tpu_bootstrap"] == workers[1]["tpu_bootstrap"], "恢复后的引擎运行库或设备设置不同")
    return {
        "complete": True,
        "resumed_generations": len(restored),
        "compared_weight_versions": [2, 3],
        "worker_pids": [row["pid"] for row in workers],
    }


def check_backend_records(root: Path, config: dict[str, Any]) -> dict[str, Any] | None:
    """只有已验证的原生配置可以跳过引擎生命周期记录。"""
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
    require(metas[0]["run_config"] == metas[1]["run_config"], "续训配置不同")
    require(metas[0]["source_files_sha256"] == metas[1]["source_files_sha256"], "续训源码不同")
    require(bool(metas[0]["source_files_sha256"]), "缺少续训源码记录")
    sampling_protocol = check_training_sampling_protocol(metas[0]["run_config"])
    checkpoints = {}
    for run, step in [("baseline", 2), ("baseline", 4), ("resume", 4)]:
        path = root / f"{run}-checkpoints" / f"step_{step:08d}"
        inspection = check_checkpoint(path, step, require_nonzero_adam=True)
        checkpoints[f"{run}-step{step}"] = {
            "sha256": file_sha256(path / "state.safetensors"),
            "file_bytes": inspection["file_bytes"],
            "leaf_count": inspection["leaf_count"],
            "all_finite": inspection["all_finite"],
            "adam_nonzero_elements": inspection["adam_nonzero_elements"],
            "data_cursor": inspection["metadata"]["data_cursor"],
            "format": read_checkpoint_metadata(path)["format"],
        }
    require(checkpoints["baseline-step4"]["sha256"] == checkpoints["resume-step4"]["sha256"], "末步状态文件不同")
    summaries = [read_json(root / run / "summary.json") for run in ("baseline", "resume")]
    require(summaries[0]["start_step"] == 0 and summaries[1]["start_step"] == 2, "起始步数不同")
    require(all(s["final_step"] == 4 for s in summaries), "训练未到达第 4 步")
    start, end = checkpoints["baseline-step2"]["data_cursor"], checkpoints["baseline-step4"]["data_cursor"]
    require(type(start) is int and type(end) is int and 0 <= start < end, "续训数据游标没有前进")
    require(checkpoints["resume-step4"]["data_cursor"] == end, "续训数据游标不同")
    names = [f"batch_{index:08d}.npz" for index in range(start, end)]
    require(sorted(p.name for p in (root / "resume/rollout_batches").glob("*.npz")) == names, "续训候选不完整")
    candidates = [
        compare_candidate_files(
            root / "baseline/rollout_batches" / name,
            root / "resume/rollout_batches" / name,
            expected_backend="inference-process" if sampling_protocol == "engine-v1" else "jax",
        )
        for name in names
    ]
    original_rows = [row for row in read_rows(root / "baseline/rollouts.jsonl") if row["step"] > 2]
    require(original_rows == read_rows(root / "resume/rollouts.jsonl"), "续训文本/奖励/选择不同")
    for run, steps in [("baseline", range(1, 5)), ("resume", range(3, 5))]:
        for step in steps:
            witness = read_json(root / run / f"update_witness_{step:08d}.json")
            require(witness["all_finite"] and witness["changed_elements"] > 0, "缺少有效的非零更新")
            require(witness["before_step"] == step - 1 and witness["after_step"] == step, "更新版本错位")
        for name in ("startup_drift.json", "joint_old_drift.json"):
            drift = read_json(root / run / name)
            require(drift["ratio_p01"] >= 0.9 and drift["ratio_p99"] <= 1.1, "概率比值的分位数超出允许范围")
    for name in ("final_state_audit.json", "update_witness_00000003.json", "update_witness_00000004.json"):
        require(read_json(root / "baseline" / name) == read_json(root / "resume" / name), f"状态检查记录不同: {name}")
    predictions = check_evaluation_records(
        root, metas[0], end, expected_count=evaluation_count, expected_max_new_tokens=evaluation_max_new_tokens
    )
    engine = check_backend_records(root, metas[0]["run_config"])
    if rescore:
        from transformers import AutoTokenizer

        from gemma4_posttrain_jax.rewards import score_completions

        tokenizer = AutoTokenizer.from_pretrained(metas[0]["run_config"]["model_path"], local_files_only=True)
        decoded = tokenizer.batch_decode([row["completion_token_ids"] for row in predictions], skip_special_tokens=True)
        require(decoded == [row["completion"] for row in predictions], "实际token重新解码不同")
        scores = score_completions(
            decoded,
            [row["gold"] for row in predictions],
            completion_lengths=[row["length"] for row in predictions],
            workers=1,
        )
        require(np.array_equal(scores.task_success, [row["task_success"] for row in predictions]), "重新评分不同")
        require(np.array_equal(scores.format_reward, [row["format_reward"] for row in predictions]), "格式评分不同")
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
        "source_files_sha256": metas[0]["source_files_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--rescore", action="store_true", help="用本地tokenizer从实际token重新解码与评分，需要hf依赖")
    parser.add_argument("--output", type=Path, help="默认root/recovery_check.json；拒绝覆盖已有报告")
    parser.add_argument("--evaluation-count", type=int, default=4, help="实验预先登记的评估题数，默认保留四题示例")
    parser.add_argument("--evaluation-max-new-tokens", type=int, help="同时核对登记长度和实际输出token数")
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
