#!/usr/bin/env python3
"""比较恢复示例中的完整训练状态、生成样本和独立评估结果。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
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


def compare_candidate_files(baseline: Path, resumed: Path) -> dict[str, Any]:
    """按类型、形状和字节比较数组，包括正负零、随机状态和被丢弃的样本。"""
    with np.load(baseline, allow_pickle=False) as left, np.load(resumed, allow_pickle=False) as right:
        require(set(left.files) == set(right.files), f"候选字段不同: {resumed.name}")
        require(set(left.files) >= CANDIDATE_FIELDS, f"候选缺少必需字段: {sorted(CANDIDATE_FIELDS - set(left.files))}")
        require(left["protocol"].item() == "rollout-candidate-arrays-v1", "候选记录协议不符")
        require(left["completion_ids"].ndim == 2 and len(left["completion_ids"]) > 0, "候选样本为空或形状错误")
        for field in left.files:
            a, b = left[field], right[field]
            require(
                a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
                f"候选数组不同: {resumed.name}/{field}",
            )
        return {"file": resumed.name, "fields": sorted(left.files), "rows": len(left["completion_ids"])}


def check_evaluation_records(root: Path, train_meta: dict[str, Any], final_cursor: int) -> list[dict[str, Any]]:
    """核对四题评估实际对应末步模型、同一套题目及同一采样实现。"""
    paths = [root / f"eval-{run}" for run in ("baseline", "resume")]
    prediction_files = [path / "evaluation/predictions.jsonl" for path in paths]
    require(prediction_files[0].read_bytes() == prediction_files[1].read_bytes(), "独立重载评估不同")
    predictions = read_rows(prediction_files[0])
    indices = [row["dataset_index"] for row in predictions]
    require(len(indices) == 4 and len(set(indices)) == 4, "恢复示例要求四道不同的评估题")
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
        summary = read_json(path / "summary.json")
        require(summary["step"] == 4 and summary["count"] == 4, "评估未完整结束")
        require(summary["dataset_indices"] == indices, "评估汇总题号不同")
        for name, digest in train_meta["source_files_sha256"].items():
            if name.startswith("gemma4_posttrain_jax/"):
                require(meta["source_files_sha256"].get(name) == digest, f"评估使用了不同的源码: {name}")
    for field in ("sampler", "jax_default_matmul_precision", "source_files_sha256"):
        require(metas[0][field] == metas[1][field], f"两次评估设置不同: {field}")
    for row in predictions:
        require(len(row["completion_token_ids"]) == row["length"], "实际token数与长度不同")
        require(bool(row["prompt_token_ids"]), "缺少实际prompt token")
    return predictions


def compare_recovery(root: Path, *, rescore: bool = False) -> dict[str, Any]:
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
    metas = [read_json(root / run / "meta.json") for run in ("baseline", "resume")]
    require(metas[0]["run_config"] == metas[1]["run_config"], "续训配置不同")
    require(metas[0]["source_files_sha256"] == metas[1]["source_files_sha256"], "续训源码不同")
    require(bool(metas[0]["source_files_sha256"]), "缺少续训源码记录")
    require(metas[0]["run_config"]["sampling_math_protocol"] == "fp32-exp-log-highest-v1", "采样协议不同")
    summaries = [read_json(root / run / "summary.json") for run in ("baseline", "resume")]
    require(summaries[0]["start_step"] == 0 and summaries[1]["start_step"] == 2, "起始步数不同")
    require(all(s["final_step"] == 4 for s in summaries), "训练未到达第 4 步")
    start, end = checkpoints["baseline-step2"]["data_cursor"], checkpoints["baseline-step4"]["data_cursor"]
    require(type(start) is int and type(end) is int and 0 <= start < end, "续训数据游标没有前进")
    require(checkpoints["resume-step4"]["data_cursor"] == end, "续训数据游标不同")
    names = [f"batch_{index:08d}.npz" for index in range(start, end)]
    require(sorted(p.name for p in (root / "resume/rollout_batches").glob("*.npz")) == names, "续训候选不完整")
    candidates = [
        compare_candidate_files(root / "baseline/rollout_batches" / name, root / "resume/rollout_batches" / name)
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
    predictions = check_evaluation_records(root, metas[0], end)
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
        "scope": "E2B native four-update recovery and small reload evaluation; not a learning-quality benchmark",
        "checkpoints": checkpoints,
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
    args = parser.parse_args()
    output = args.output or args.root / "recovery_check.json"
    if output.exists():
        raise FileExistsError(output)
    result = compare_recovery(args.root, rescore=args.rescore)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "resumed_batches": len(result["resumed_candidates"]), "output": str(output)}))


if __name__ == "__main__":
    main()
