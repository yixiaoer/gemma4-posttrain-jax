"""固定 GSM8K/MATH 留出集评估；补齐行不进入指标，不改变训练状态或随机流。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor
from numbers import Integral
from pathlib import Path
from typing import Any, NamedTuple

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding

from .bench import timed_call
from .data import eval_subset, example_gold, format_prompt, load_gsm8k, split_holdout_indices
from .math_data import load_math500_data, load_math_training_data
from .model import Gemma4TextConfig
from .rewards import Gold, score_completions
from .sampler import RolloutBatch, SamplerConfig, generate
from .sharding import batch_spec, named_shardings, param_specs_rollout


class EvalBatch(NamedTuple):
    prompt_ids: np.ndarray
    prompt_mask: np.ndarray
    indices: tuple[int, ...]  # 只包含有效行；设备 batch 的尾部重复不计入结果
    golds: tuple[Gold, ...]
    source_ids: tuple[str, ...] | None = None


class EvaluationData(NamedTuple):
    dataset: Any
    indices: list[int]
    provenance: dict[str, Any]
    split: str
    subset: str


def indices_sha256(indices: Sequence[int]) -> str:
    """与训练 run_config 相同的 little-endian int64 题号身份。"""
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def load_evaluation_data(
    dataset_name: str = "gsm8k",
    *,
    subset: str | None = None,
    size: int = 500,
    seed: int = 0,
    dev_size: int = 500,
    dev_seed: int = 0,
    checkpoint_run_config: Mapping[str, Any] | None = None,
) -> EvaluationData:
    """在加载模型前固定题目；train-dev 必须与 checkpoint 的实际排除集合一致。"""
    if dataset_name not in ("gsm8k", "math"):
        raise ValueError("评估 dataset 只能是 gsm8k 或 math")
    subset = subset or ("math500" if dataset_name == "math" else "test-monitor")
    allowed = ("train-dev", "math500") if dataset_name == "math" else ("train-dev", "test-monitor", "test-unused")
    if subset not in allowed or size <= 0:
        raise ValueError("评估 subset 与 dataset 不匹配，或 size 非正")
    split = "train" if subset == "train-dev" else "test"
    if dataset_name == "math":
        data = load_math_training_data() if split == "train" else load_math500_data()
        dataset, provenance = data
    else:
        dataset = load_gsm8k(split)
        provenance = {"source": "openai/gsm8k", "configuration": "main", "split": split}
    if subset == "train-dev":
        train, held_out = split_holdout_indices(len(dataset), dev_size, seed=dev_seed)
        if checkpoint_run_config is not None:
            expected = {
                "dataset_fingerprint": dataset._fingerprint,
                "dataset_size": len(dataset),
                "dev_size": dev_size,
                "dev_seed": dev_seed,
                "train_indices_sha256": indices_sha256(train),
                "dev_indices_sha256": indices_sha256(held_out),
            }
            # 旧 GSM8K checkpoint 尚无 dataset 字段；其它身份项不能靠默认值补造。
            if checkpoint_run_config.get("dataset", "gsm8k") != dataset_name or any(
                checkpoint_run_config.get(key) != value for key, value in expected.items()
            ):
                raise ValueError("train-dev 数据身份/排除集合与 checkpoint 不一致")
            if dataset_name == "math" and checkpoint_run_config.get("data_provenance") != provenance:
                raise ValueError("MATH train-dev 清洗或 gold 协议与 checkpoint 不一致")
        indices = held_out[:size]
        if len(indices) != size:
            raise ValueError("评估 size 超过 train-dev 留出子集")
    elif subset == "test-unused":
        # 排除 Phase4 已监测的固定 test500/seed0；eval seed 只改变剩余题顺序。
        remaining, _ = split_holdout_indices(len(dataset), 500, seed=0)
        indices = eval_subset(remaining, size=size, seed=seed)
    else:
        indices = eval_subset(list(range(len(dataset))), size=size, seed=seed)
    return EvaluationData(dataset, indices, provenance, split, subset)


def resolve_eval_prompt_style(requested: str | None, run_config: Mapping[str, Any] | None) -> str:
    """省略参数时继承checkpoint；旧checkpoint固定chat，拒绝改变已训练模板。"""
    saved = None if run_config is None else run_config.get("prompt_style", "chat")
    if saved is not None and saved not in ("chat", "plain"):
        raise ValueError("checkpoint的prompt style不受支持")
    if saved == "plain" and run_config is not None and run_config.get("prompt_protocol") != "plain-bos-instruction-v2":
        raise ValueError("checkpoint的plain prompt协议不受支持")
    if requested is not None and requested not in ("chat", "plain"):
        raise ValueError("prompt style只能为chat或plain")
    if requested is not None and saved is not None and requested != saved:
        raise ValueError("评估prompt style与checkpoint不一致")
    return requested or saved or "chat"


def resolve_eval_rollout_layout(requested: str | None, run_config: Mapping[str, Any] | None) -> str:
    """默认继承训练布局；显式覆盖用于登记后的布局系统对照。"""
    saved = "replicated" if run_config is None else run_config.get("rollout_layout", "replicated")
    if saved not in ("replicated", "fsdp") or requested not in (None, "replicated", "fsdp"):
        raise ValueError("rollout layout只能为replicated或fsdp")
    return requested or saved


def make_eval_sampler(
    config: Gemma4TextConfig, sampler_config: SamplerConfig, mesh: Mesh, *, rollout_layout: str = "replicated"
) -> Any:
    if sampler_config.temperature != 0:
        raise ValueError("当前固定评估使用 greedy；不要把随机采样与 greedy 准确率混为一谈")
    batch = NamedSharding(mesh, batch_spec(2))
    return jax.jit(
        lambda params, ids, mask, key: generate(
            params, config, ids, mask, sampler_config=sampler_config, key=key, mesh=mesh
        ),
        in_shardings=(
            named_shardings(param_specs_rollout(config, layout=rollout_layout, num_devices=mesh.size), mesh),
            batch,
            batch,
            None,
        ),
    )


def prepare_eval_batches(
    dataset: Sequence[Any],
    tokenizer: Any,
    *,
    size: int = 500,
    seed: int = 0,
    batch_size: int = 32,
    max_prompt_len: int = 512,
    pad_token_id: int = 0,
    indices: Sequence[int] | None = None,
    chat: bool = True,
) -> list[EvalBatch]:
    if batch_size <= 0 or max_prompt_len <= 0:
        raise ValueError("评估 batch size 和 prompt 长度必须为正")
    if indices is None:
        indices = eval_subset(list(range(len(dataset))), size=size, seed=seed)
    elif (
        size <= 0
        or len(indices) != size
        or len(set(indices)) != size
        or any(not isinstance(index, Integral) or index < 0 or index >= len(dataset) for index in indices)
    ):
        raise ValueError("显式评估题号必须与size相等、非空、唯一且在原始dataset范围内")
    source_ids = [dataset[index].get("source_id") for index in indices]
    has_source_ids = any(value is not None for value in source_ids)
    if has_source_ids and (
        any(not isinstance(value, str) or not value.strip() for value in source_ids) or len(set(source_ids)) != size
    ):
        raise ValueError("评估 source_id 必须完整、非空且唯一")
    batches = []
    for start in range(0, size, batch_size):
        selected = indices[start : start + batch_size]
        ids = np.full((batch_size, max_prompt_len), pad_token_id, np.int32)
        mask = np.zeros_like(ids, dtype=np.bool_)
        golds = []
        for row, index in enumerate(selected):
            tokens = format_prompt(str(dataset[index]["question"]), tokenizer, chat=chat)
            if not tokens or len(tokens) > max_prompt_len:
                raise ValueError(f"评估题目 {index} 的 prompt 长度 {len(tokens)} 不在 [1,{max_prompt_len}]")
            ids[row, -len(tokens) :] = tokens
            mask[row, -len(tokens) :] = True
            golds.append(example_gold(dataset[index]))
        # 使用合法 prompt 补齐设备形状，返回的 indices/golds 仍只有有效行。
        ids[len(selected) :] = ids[len(selected) - 1]
        mask[len(selected) :] = mask[len(selected) - 1]
        selected_sources = tuple(str(value) for value in source_ids[start : start + len(selected)])
        batches.append(
            EvalBatch(ids, mask, tuple(selected), tuple(golds), selected_sources if has_source_ids else None)
        )
    return batches


def evaluate_batches(
    generate_batch: Callable[[Any, np.ndarray, np.ndarray, jax.Array], RolloutBatch],
    params: Any,
    batches: Sequence[EvalBatch],
    tokenizer: Any,
    *,
    output_dir: Path,
    eos_ids: tuple[int, ...],
    seed: int = 0,
    executor: Executor | None = None,
) -> dict[str, Any]:
    """调用预编译的 greedy sampler；不接收 TrainState，也不复用训练 key。"""

    if not batches:
        raise ValueError("评估 batch 不能为空")
    output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    elapsed = 0.0
    with (output_dir / "predictions.jsonl").open("w") as file:
        for batch_index, batch in enumerate(batches):
            key = jax.random.fold_in(jax.random.PRNGKey(seed), batch_index)
            rollout, seconds = timed_call(generate_batch, params, batch.prompt_ids, batch.prompt_mask, key)
            elapsed += seconds
            ids, lengths = jax.device_get((rollout.completion_ids, rollout.lengths))
            count = len(batch.indices)
            ids, lengths = np.asarray(ids)[:count], np.asarray(lengths)[:count]
            completions = tokenizer.batch_decode(
                [tokens[:length] for tokens, length in zip(ids, lengths, strict=True)],
                skip_special_tokens=True,
            )
            rewards = score_completions(
                completions,
                batch.golds,
                completion_lengths=lengths.tolist(),
                workers=1 if executor is None else None,
                executor=executor,
            )
            for row, text in enumerate(completions):
                record = {
                    "dataset_index": batch.indices[row],
                    "gold": batch.golds[row],
                    "completion": text,
                    # 从实际输入/输出取ID；保留有效EOS及pad值，不能由解码文本重新分词补造。
                    "prompt_token_ids": batch.prompt_ids[row, batch.prompt_mask[row]].tolist(),
                    "completion_token_ids": ids[row, : int(lengths[row])].tolist(),
                    "length": int(lengths[row]),
                    "task_success": float(rewards.task_success[row]),
                    "format_reward": float(rewards.format_reward[row]),
                    "truncated": bool(lengths[row] == ids.shape[1] and int(ids[row, -1]) not in eos_ids),
                }
                if batch.source_ids is not None:
                    record["source_id"] = batch.source_ids[row]
                records.append(record)
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
    correct = sum(record["task_success"] for record in records)
    summary = {
        "count": len(records),
        "correct": int(correct),
        "accuracy": correct / len(records),
        "truncated_fraction": float(np.mean([record["truncated"] for record in records])),
        "mean_length": float(np.mean([record["length"] for record in records])),
        "generated_tokens": sum(record["length"] for record in records),
        "generation_s": elapsed,
        "dataset_indices": [record["dataset_index"] for record in records],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
