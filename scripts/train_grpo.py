#!/usr/bin/env python3
"""Gemma 4 E2B 的同步 GRPO族多轮更新：真实 GSM8K rollout、奖励、更新与恢复。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from gemma4_posttrain_jax.algorithms import (
    completion_loss_mask,
    mixed_success_groups,
    objective_advantages,
    objective_for_algorithm,
    truncated_importance_weights,
    validate_update_schedule,
)
from gemma4_posttrain_jax.bench import device_peak_bytes, timed_call
from gemma4_posttrain_jax.checkpoint import load_train_state, read_checkpoint_metadata, save_train_state
from gemma4_posttrain_jax.data import (
    PromptBatch,
    collate_grpo_prompts,
    grpo_prompt_indices,
    load_gsm8k,
    split_holdout_indices,
)
from gemma4_posttrain_jax.diagnostics import logprob_drift_metrics, optional_package_version, source_git_state
from gemma4_posttrain_jax.evaluation import evaluate_batches, make_eval_sampler, prepare_eval_batches
from gemma4_posttrain_jax.lora import LoRAConfig, check_lora_config, init_lora_params, prepare_lora_params
from gemma4_posttrain_jax.lora_training import LoRAPolicyParams, lora_base_identity, make_lora_optimizer
from gemma4_posttrain_jax.losses import (
    TrainState,
    cast_floating_tree,
    grpo_train_step,
    init_train_state,
    make_optimizer,
    trainer_completion_logps,
)
from gemma4_posttrain_jax.math_data import load_math_training_data
from gemma4_posttrain_jax.model import Gemma4TextParams
from gemma4_posttrain_jax.rewards import Gold, RewardConfig, RewardOutput, score_completions
from gemma4_posttrain_jax.rollout_state import BehaviorSnapshot, LaggedTrainState, load_lagged_train_state
from gemma4_posttrain_jax.sampler import RolloutBatch, SamplerConfig, generate
from gemma4_posttrain_jax.sharding import (
    batch_spec,
    make_mesh,
    named_shardings,
    param_specs_rollout,
    replicate_scalars,
    reshard_for_rollout,
    shard_batch,
    shard_gemma4_text_params,
    tree_shardings,
)
from gemma4_posttrain_jax.tracking import compile_tracking_metrics, grpo_tracking_metrics, init_tracker
from gemma4_posttrain_jax.weights import load_hf_eos_token_ids, load_hf_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dataset", choices=("gsm8k", "math"), default="gsm8k")
    parser.add_argument(
        "--prompt-style", choices=("chat", "plain"), default="chat", help="base模型使用plain；恢复必须保持模板"
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="新的运行目录，不覆盖历史产物")
    parser.add_argument("--rollout-layout", choices=("replicated", "fsdp"), default="replicated")
    parser.add_argument("--rollout-backend", choices=("jax", "inference", "inference-process"), default="jax")
    parser.add_argument("--training-device-ids", type=int, nargs="+", help="显式训练mesh设备ID；省略保持全部设备")
    parser.add_argument(
        "--inference-device-ids", type=int, nargs="+", help="引擎本地设备：同进程默认2、3；独立进程固定0、1"
    )
    parser.add_argument("--inference-python", type=Path, help="独立进程使用的固定推理环境解释器")
    parser.add_argument("--inference-memory-fraction", type=float, default=0.5)
    parser.add_argument("--lora-rank", type=int, help="只训练独立FP32 attention适配器；首个入口需要FSDP/β0/lag0")
    parser.add_argument("--lora-alpha", type=float, help="默认2*rank，训练和保存使用未缩放A/B")
    parser.add_argument("--lora-targets", nargs="+", choices=("q_proj", "k_proj", "v_proj", "o_proj"))
    parser.add_argument("--prompt-batch-size", type=int, default=8, help="不同题目数；实际生成行数为此值乘 G")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument(
        "--algorithm",
        choices=("grpo", "drgrpo", "dapo", "dapo-loss", "gspo-token", "rloo"),
        default="grpo",
        help="dapo默认启用动态补采/超长过滤/软长度惩罚；dapo-loss只选择目标",
    )
    parser.add_argument("--updates-per-rollout", type=int, default=1, help="同一完整rollout batch执行μ次Adam更新")
    parser.add_argument(
        "--loss-aggregation",
        choices=("sequence-mean-token-mean", "token-mean", "sequence-mean-token-scale", "seq-mean-token-sum"),
        help="显式聚合消融；覆盖算法默认值",
    )
    parser.add_argument("--token-scale", type=float, help="常数尺度聚合的分母；省略时使用生成预算N")
    parser.add_argument("--sampler-is-cap", type=float, help="冻结old-trainer/behavior逐token修正的上限")
    parser.add_argument(
        "--rollout-lag-updates",
        type=int,
        choices=(0, 1),
        default=0,
        help="1: 顺序执行的一次Adam权重滞后对照；不表示训练与生成并行",
    )
    parser.add_argument(
        "--filter-truncated-loss",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="超长行不参与loss，保留生成上下文和原始奖励",
    )
    parser.add_argument("--dynamic-sampling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-sampling-attempts", type=int, default=16, help="补足一个有效batch的最大候选batch次数")
    parser.add_argument("--overlong-buffer-length", type=int, default=None)
    parser.add_argument("--overlong-penalty", type=float, default=None)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--microbatch-size", type=int, help="训练/参考前向的展开行数；完整 rollout 后累积一次更新")
    parser.add_argument("--max-steps", type=int, default=30, help="目标全局更新步数")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument(
        "--truncated-reward",
        choices=("keep", "zero"),
        default="keep",
        help="zero: 达长度上限但未 EOS 的行使用零训练奖励，保留原始评分",
    )
    parser.add_argument("--format-bonus", type=float, default=0.1)
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--remat", action="store_true")
    parser.add_argument("--vocab-chunk", type=int, default=8192)
    parser.add_argument("--sequence-chunk", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument(
        "--save-at-steps", type=int, nargs="+", help="只在指定完整rollout更新边界保存；与save-every互斥"
    )
    parser.add_argument(
        "--audit-final-state", action="store_true", help="逐叶保存最终master/Adam内容摘要，供独立恢复核对"
    )
    parser.add_argument("--audit-update-state", action="store_true", help="每步记录少量可训练参数的真实变化")
    parser.add_argument(
        "--audit-rollout-batches", action="store_true", help="保留全部候选batch的原始token/logprob/掩码和随机key"
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save-hlo", action="store_true", help="保存各 JIT 边界的 StableHLO、优化 HLO 和内存分析")
    parser.add_argument("--eval-every", type=int, default=0, help="0 关闭；否则在起始步和每 N 个更新后评估")
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-max-new-tokens", type=int, default=1024)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--dev-size", type=int, default=0, help="从所选训练数据中排除并固定保留的dev题数")
    parser.add_argument("--dev-seed", type=int, default=0)
    parser.add_argument("--eval-split", choices=("test", "train-dev"), default="test")
    parser.add_argument("--wandb", action="store_true", help="启用 host 侧 W&B；离线设 WANDB_MODE=offline")
    parser.add_argument("--wandb-project", default="gemma4-posttrain-jax")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-tags", nargs="*", default=())
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def compile_call(name: str, fn: Any, arguments: tuple[Any, ...], output_dir: Path, save_hlo: bool) -> Any:
    start = time.perf_counter()
    lowered = fn.lower(*arguments)
    lower_s = time.perf_counter() - start
    stablehlo_dump_s = 0.0
    if save_hlo:
        dump_start = time.perf_counter()
        (output_dir / f"{name}.stablehlo.txt").write_text(lowered.as_text())
        stablehlo_dump_s = time.perf_counter() - dump_start
    backend_start = time.perf_counter()
    try:
        compiled = lowered.compile()
    except Exception as error:
        backend_compile_s = time.perf_counter() - backend_start
        record = {
            "name": name,
            "status": "failed",
            "stage": "backend_compile",
            "compile_s": lower_s + backend_compile_s,
            "lower_s": lower_s,
            "backend_compile_s": backend_compile_s,
            "stablehlo_dump_s": stablehlo_dump_s,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        write_json(output_dir / f"{name}_compile.json", record)
        print(json.dumps(record), flush=True)
        raise
    backend_compile_s = time.perf_counter() - backend_start
    memory = compiled.memory_analysis()
    record = {
        "name": name,
        "status": "complete",
        "compile_s": lower_s + backend_compile_s,
        "lower_s": lower_s,
        "backend_compile_s": backend_compile_s,
        "stablehlo_dump_s": stablehlo_dump_s,
        "memory": {
            field: int(getattr(memory, field))
            for field in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "alias_size_in_bytes",
                "temp_size_in_bytes",
            )
            if getattr(memory, field, None) is not None
        },
    }
    write_json(output_dir / f"{name}_compile.json", record)
    if save_hlo:
        (output_dir / f"{name}.hlo.txt").write_text(compiled.as_text())
    print(json.dumps(record), flush=True)
    return compiled


def main() -> None:
    args = parse_args()
    process_inference = args.rollout_backend == "inference-process"
    inference_identity = None
    if args.rollout_backend != "jax":
        if args.lora_rank is not None or args.rollout_lag_updates or args.eval_every:
            raise ValueError("首个引擎闭环不支持LoRA、滞后或进程内周期评估；独立评估可读完整训练状态")
        if args.sampler_is_cap is None:
            raise ValueError("引擎闭环需要显式sampler-is-cap，记录trainer/behavior分布修正")
        if not 0 < args.inference_memory_fraction < 1:
            raise ValueError("推理内存比例必须在0和1之间")
        if args.training_device_ids is None:
            args.training_device_ids = [0, 1]
        if args.inference_device_ids is None:
            args.inference_device_ids = [0, 1] if process_inference else [2, 3]
        if process_inference:
            from gemma4_posttrain_jax.inference_remote import inspect_engine_environment

            if os.environ.get("TPU_VISIBLE_CHIPS") != "0,1" or args.training_device_ids != [0, 1]:
                raise ValueError("独立进程模式要求启动前绑定训练物理chips0,1，并使用本地devices0,1")
            if args.inference_device_ids != [0, 1] or args.inference_python is None or args.beta:
                raise ValueError("独立进程首版固定引擎本地devices0,1、β0，并显式指定inference-python")
            inference_identity = inspect_engine_environment(args.inference_python)
        elif args.inference_python is not None:
            raise ValueError("inference-python仅用于独立进程模式")
        # 引擎方法使用独立DEFAULT上下文，训练仍按原HIGHEST精度编译。
        jax.config.update("jax_default_matmul_precision", "highest")
    elif (
        args.inference_device_ids is not None
        or args.inference_memory_fraction != 0.5
        or args.inference_python is not None
    ):
        raise ValueError("推理设备和内存参数仅用于inference后端")
    lora_config = None
    if args.lora_rank is not None:
        lora_config = LoRAConfig(
            args.lora_rank,
            2.0 * args.lora_rank if args.lora_alpha is None else args.lora_alpha,
            tuple(args.lora_targets or LoRAConfig().targets),
        )
        check_lora_config(lora_config)
        if args.beta or args.rollout_lag_updates or args.rollout_layout != "fsdp":
            raise ValueError("首个LoRA入口固定β0/lag0和FSDP基础权重；其它组合尚未验收")
        args.freeze_embeddings = True
    elif args.lora_alpha is not None or args.lora_targets is not None:
        raise ValueError("lora-alpha/targets需要显式lora-rank")
    if args.dataset == "math" and args.eval_every and args.eval_split != "train-dev":
        raise ValueError("MATH周期评估使用train-dev；MATH500保留给独立最终评估")
    if min(args.prompt_batch_size, args.max_prompt_len, args.max_new_tokens, args.max_steps, args.reward_workers) <= 0:
        raise ValueError("batch、长度、步数和 reward workers 必须为正")
    if args.group_size < 2 or args.beta < 0 or args.save_every < 0 or args.eval_every < 0:
        raise ValueError("需要 G >= 2、beta >= 0、save-every/eval-every >= 0")
    validate_update_schedule(
        updates_per_rollout=args.updates_per_rollout,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )
    if args.save_at_steps is not None:
        if args.save_every or len(set(args.save_at_steps)) != len(args.save_at_steps):
            raise ValueError("save-at-steps不能与save-every并用，且不允许重复步")
        if any(step <= 0 or step > args.max_steps or step % args.updates_per_rollout for step in args.save_at_steps):
            raise ValueError("save-at-steps必须为预算内完整rollout的正更新步")
    if args.sampler_is_cap is not None and (not np.isfinite(args.sampler_is_cap) or args.sampler_is_cap <= 0):
        raise ValueError("sampler-is-cap必须为正的有限数")
    if args.dynamic_sampling is None:
        args.dynamic_sampling = args.algorithm == "dapo"
    if args.filter_truncated_loss is None:
        args.filter_truncated_loss = args.algorithm == "dapo"
    if args.overlong_buffer_length is None:
        args.overlong_buffer_length = min(128, args.max_new_tokens) if args.algorithm == "dapo" else 0
    if args.overlong_penalty is None:
        args.overlong_penalty = 1.0 if args.algorithm == "dapo" else 0.0
    if args.max_sampling_attempts <= 0 or not 0 <= args.overlong_buffer_length <= args.max_new_tokens:
        raise ValueError("sampling attempts必须为正，overlong buffer须在[0,N]内")
    if (
        not np.isfinite(args.overlong_penalty)
        or args.overlong_penalty < 0
        or (args.overlong_penalty and not args.overlong_buffer_length)
    ):
        raise ValueError("overlong penalty必须非负有限，启用时buffer必须为正")
    if args.algorithm == "rloo" and (
        args.updates_per_rollout != 1
        or args.beta
        or args.sampler_is_cap is not None
        or args.rollout_lag_updates
        or args.dynamic_sampling
        or args.filter_truncated_loss
    ):
        raise ValueError("首个RLOO入口固定μ1、β0、无IS/滞后/动态过滤；其他组合需要单独定义")
    objective = objective_for_algorithm(
        "dapo" if args.algorithm == "dapo-loss" else args.algorithm, generation_budget=args.max_new_tokens
    )
    if args.loss_aggregation is not None:
        objective = objective._replace(agg_mode=args.loss_aggregation)
    if args.token_scale is not None and (not np.isfinite(args.token_scale) or args.token_scale <= 0):
        raise ValueError("token-scale必须为正的有限数")
    if objective.agg_mode == "sequence-mean-token-scale":
        objective = objective._replace(token_scale=args.token_scale or float(args.max_new_tokens))
    elif args.token_scale is not None:
        raise ValueError("token-scale只用于sequence-mean-token-scale聚合")
    else:
        objective = objective._replace(token_scale=None)
    if args.algorithm == "rloo" and objective.agg_mode != "seq-mean-token-sum":
        raise ValueError("首个RLOO基线固定轨迹token-sum；其他长度权重不冒充相同目标")
    objective_options = {key: value for key, value in objective._asdict().items() if key != "advantage_estimator"}
    needs_old_logps = args.updates_per_rollout > 1 or args.sampler_is_cap is not None
    if (args.save_every or args.save_at_steps) and args.checkpoint_dir is None:
        raise ValueError("启用保存时必须指定 --checkpoint-dir，并预留完整 FP32 master/Adam 状态空间")
    if args.model_path is None:
        snapshots = sorted(Path.home().glob(".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/*"))
        if len(snapshots) != 1:
            raise ValueError("需要明确 --model-path（未找到唯一的 E2B-it snapshot）")
        args.model_path = snapshots[0]
    args.model_path = args.model_path.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.dataset == "math":
        prepared = load_math_training_data()
        dataset, data_provenance = prepared.dataset, prepared.provenance
    else:
        dataset = load_gsm8k("train")
        data_provenance = {"source": "openai/gsm8k", "configuration": "main", "split": "train"}
    training_indices, dev_indices = split_holdout_indices(len(dataset), args.dev_size, seed=args.dev_seed)
    if args.prompt_batch_size > len(training_indices):
        raise ValueError("排除dev后剩余训练题不足一个prompt batch")
    if args.eval_split == "train-dev" and (not args.dev_size or args.eval_size > args.dev_size):
        raise ValueError("train-dev评估需要dev-size>0且eval-size不超过dev-size")
    write_json(
        args.output_dir / "data_split.json",
        {
            "dataset": args.dataset,
            "source_provenance": data_provenance,
            "dataset_fingerprint": dataset._fingerprint,
            "seed": args.dev_seed,
            "training_indices": training_indices,
            "dev_indices": dev_indices,
        },
    )
    devices_by_id = {int(device.id): device for device in jax.devices()}
    training_ids = list(devices_by_id) if args.training_device_ids is None else args.training_device_ids
    if not training_ids or len(set(training_ids)) != len(training_ids) or set(training_ids) - devices_by_id.keys():
        raise ValueError("训练设备ID必须非空、不重复且实际存在")
    if args.rollout_backend == "inference":
        inference_ids = args.inference_device_ids
        if (
            not inference_ids
            or len(set(inference_ids)) != len(inference_ids)
            or set(inference_ids) - devices_by_id.keys()
            or set(inference_ids) & set(training_ids)
        ):
            raise ValueError("推理设备须实际存在、不重复且与训练mesh分离")
    elif process_inference and sorted(devices_by_id) != [0, 1]:
        raise ValueError("独立进程训练必须实际只看到两个本地TPU设备")
    mesh = make_mesh([devices_by_id[index] for index in training_ids])
    rows = args.prompt_batch_size * args.group_size
    if rows % mesh.size:
        raise ValueError(f"生成行数 {rows} 必须能被设备数 {mesh.size} 整除")
    if args.microbatch_size is not None and (
        args.microbatch_size <= 0 or rows % args.microbatch_size or args.microbatch_size % mesh.size
    ):
        raise ValueError("microbatch-size 必须整除展开行数且能被设备数整除")
    eval_batches = []
    eval_fingerprint = None
    if args.eval_every:
        if args.eval_batch_size <= 0 or args.eval_batch_size % mesh.size or args.eval_max_new_tokens <= 0:
            raise ValueError("评估长度必须为正，eval-batch-size 必须能被设备数整除")
        eval_dataset = dataset if args.eval_split == "train-dev" else load_gsm8k("test")
        eval_fingerprint = eval_dataset._fingerprint
        eval_batches = prepare_eval_batches(
            eval_dataset,
            tokenizer,
            size=args.eval_size,
            seed=args.eval_seed,
            batch_size=args.eval_batch_size,
            max_prompt_len=args.max_prompt_len,
            pad_token_id=tokenizer.pad_token_id,
            indices=dev_indices[: args.eval_size] if args.eval_split == "train-dev" else None,
            chat=args.prompt_style == "chat",
        )
    # 首个闭环按完整策略采样；temperature/top-k 的 proposal correction 留给后续算法实验。
    run_config = {
        "algorithm": "grpo-family-sync-v4",
        "old_policy_protocol": "joint-forward-first-update-capture-v1",
        "objective_name": args.algorithm,
        "objective": objective._asdict(),
        "updates_per_rollout": args.updates_per_rollout,
        "sampler_is_cap": args.sampler_is_cap,
        "filter_truncated_loss": args.filter_truncated_loss,
        "dynamic_sampling": args.dynamic_sampling,
        "max_sampling_attempts": args.max_sampling_attempts,
        "overlong_buffer_length": args.overlong_buffer_length,
        "overlong_penalty": args.overlong_penalty,
        "advantage_centering": "relative-to-first-v1",
        "ple_lookup": "three-dimensional-ple-tangent-barrier-v1",
        "model_path": str(args.model_path),
        "dataset": args.dataset,
        "data_provenance": data_provenance,
        "dataset_fingerprint": dataset._fingerprint,
        "dataset_size": len(dataset),
        "train_eligible_size": len(training_indices),
        "dev_size": args.dev_size,
        "dev_seed": args.dev_seed,
        "train_indices_sha256": hashlib.sha256(np.asarray(training_indices, dtype="<i8").tobytes()).hexdigest(),
        "dev_indices_sha256": hashlib.sha256(np.asarray(dev_indices, dtype="<i8").tobytes()).hexdigest(),
        "prompt_batch_size": args.prompt_batch_size,
        "group_size": args.group_size,
        "max_prompt_len": args.max_prompt_len,
        "max_new_tokens": args.max_new_tokens,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "format_bonus": args.format_bonus,
        "success_reward": 1.0,
        "failure_reward": -1.0 if args.algorithm == "dapo" else 0.0,
        "freeze_embeddings": args.freeze_embeddings,
        "remat": args.remat,
        "vocab_chunk": args.vocab_chunk,
        "sequence_chunk": args.sequence_chunk,
        "seed": args.seed,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "sampling_math_protocol": "fp32-exp-log-highest-v1" if args.rollout_backend == "jax" else "engine-v1",
        "jax_default_matmul_precision": jax.config.jax_default_matmul_precision or "default",
        "compute_dtype": "bfloat16",
        "master_dtype": "float32",
        "optimizer": "AdamW; weight_decay=0; max_grad_norm=1",
        "mesh_size": mesh.size,
        "jax": jax.__version__,
        "jaxlib": importlib.metadata.version("jaxlib"),
        "libtpu": optional_package_version("libtpu"),
    }
    if args.training_device_ids is not None:
        run_config["training_device_ids"] = training_ids
    if args.rollout_backend != "jax":
        run_config.update(
            {
                "rollout_backend": "tpu-inference-separate-process-v1"
                if process_inference
                else "tpu-inference-same-process-v1",
                "inference_device_ids": args.inference_device_ids,
                "inference_precision": "default",
                "inference_kernel_route": "batched",
                "inference_memory_fraction": args.inference_memory_fraction,
                "sampling_rng_protocol": "engine-batch-rng-v1-threefry2x32-fold-in-seed-data-cursor",
                "inference_weight_protocol": "full-gemma4-params-state-leaves-kv-version-v1",
                "inference_proposal": "full-policy-temperature-one",
                "inference_max_logprobs": 128,
                "inference_async_scheduling": False,
                "inference_versions": inference_identity["versions"]
                if inference_identity is not None
                else {name: importlib.metadata.version(name) for name in ("tpu-inference", "vllm-tpu")},
                "inference_contract_sha256": {
                    name: hashlib.sha256(
                        (Path(__file__).resolve().parents[1] / "gemma4_posttrain_jax" / name).read_bytes()
                    ).hexdigest()
                    for name in (
                        "inference_rng.py",
                        "inference_runtime.py",
                        "inference_weights.py",
                        "inference_rollout.py",
                    )
                    + (
                        ("inference_wire.py", "inference_process.py", "inference_remote.py")
                        if process_inference
                        else ()
                    )
                },
                "inference_model_config_sha256": {
                    name: hashlib.sha256((args.model_path / name).read_bytes()).hexdigest()
                    for name in ("config.json", "generation_config.json")
                    if (args.model_path / name).exists()
                },
                "inference_environment": {
                    name: os.environ.get(name)
                    for name in (
                        "ATTN_BUCKETIZED_NUM_REQS",
                        "ATTN_CUSTOM_NUM_REQS_BUCKETS",
                        "MIN_TOKEN_BUCKET",
                        "VLLM_TPU_BUCKET_PADDING_GAP",
                        "DP_SCHED_BATCH_PREFILL",
                        "DP_SCHED_BATCH_PREFILL_FLUSH_TIMEOUT_MS",
                        "RPA_V3_DECODE_BLOCK_SIZES",
                        "RPA_V3_PREFILL_BLOCK_SIZES",
                        "RPA_V3_MIXED_BLOCK_SIZES",
                    )
                },
            }
        )
        if process_inference:
            run_config["inference_process_identity"] = inference_identity
            run_config["inference_weight_transport"] = "gemma4-host-fp32-v1"
            run_config["training_physical_chips"] = [0, 1]
            run_config["inference_physical_chips"] = [2, 3]
    # 默认chat保持原v4配置；plain的模板协议参与完整checkpoint身份校验。
    if args.rollout_layout != "replicated":
        run_config["rollout_layout"] = args.rollout_layout
    if args.prompt_style == "plain":
        run_config["prompt_style"] = "plain"
        run_config["prompt_protocol"] = "plain-bos-instruction-v2"
    if args.sampler_is_cap is not None:
        run_config["sampler_is_protocol"] = "old-behavior-exp-highest-v2"
    if args.rollout_lag_updates:
        run_config["algorithm"] = "grpo-family-lagged-v1"
        run_config["rollout_lag_updates"] = args.rollout_lag_updates
        run_config["rollout_execution_protocol"] = "sequential-one-update-lag-v1"
    if args.truncated_reward != "keep":
        run_config["truncated_reward"] = args.truncated_reward
    if args.microbatch_size is not None:
        run_config["microbatch_size"] = args.microbatch_size
    if lora_config is not None:
        run_config.update(
            {
                "training_mode": "lora-unscaled-fp32-v1",
                "lora": {**lora_config._asdict(), "targets": list(lora_config.targets)},
                "frozen_base_identity": lora_base_identity(args.model_path),
                "adapter_rng_protocol": "fold-in-seed-0x4c4f5241-v1",
                "sampling_rng_protocol": "fold-in-seed-data-cursor-v1",
                "adapter_sharding": "replicated",
            }
        )
    resume = None if args.resume is None else read_checkpoint_metadata(args.resume)["metadata"]
    if resume is not None and resume.get("run_config") != run_config:
        raise ValueError("恢复配置与 checkpoint 不一致（模型、数据、采样、优化器、精度、shape 或版本改变）")
    source_root = Path(__file__).resolve().parents[1]
    revision, diff = source_git_state(source_root)
    (args.output_dir / "source.diff").write_text(diff or "")
    source_files = [*sorted((source_root / "gemma4_posttrain_jax").rglob("*.py")), Path(__file__).resolve()]
    write_json(
        args.output_dir / "meta.json",
        {
            "run_config": run_config,
            "git_commit": revision,
            "source_diff_sha256": None if diff is None else hashlib.sha256(diff.encode()).hexdigest(),
            "source_files_sha256": {
                str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_files
            },
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "devices": [str(device) for device in jax.devices()],
            "eval_dataset_fingerprint": eval_fingerprint,
            "eval_dataset_split": "train" if args.eval_split == "train-dev" else "test",
            "eval_subset": args.eval_split,
        },
    )
    host_params, config = load_hf_params(
        str(args.model_path), dtype=jnp.float32 if lora_config is None else jnp.bfloat16
    )
    params = shard_gemma4_text_params(host_params, config, mesh)
    # 参考模型始终来自初始 checkpoint，不随策略更新；beta=0 时不建立此树。
    reference_host = None
    if args.beta:
        with jax.default_device(jax.devices("cpu")[0]):
            reference_host = jax.tree.map(lambda value: value.astype(jnp.bfloat16), host_params)
    del host_params
    frozen_base: Gemma4TextParams | None = None
    trainable: Gemma4TextParams | None
    state: TrainState[Any]
    if lora_config is None:
        optimizer, trainable = make_optimizer(
            params, learning_rate=args.learning_rate, freeze_embeddings=args.freeze_embeddings
        )
        state = replicate_scalars(init_train_state(params, optimizer), mesh)
    else:
        frozen_base = params
        # fold-in固定命名空间，与采样的seed/cursor流分开；恢复仍读完整未缩放A/B与Adam。
        adapters = init_lora_params(params, lora_config, jax.random.fold_in(jax.random.PRNGKey(args.seed), 0x4C4F5241))
        adapters = jax.device_put(adapters, NamedSharding(mesh, P()))
        optimizer, trainable = make_lora_optimizer(args.learning_rate), None
        state = replicate_scalars(init_train_state(adapters, optimizer), mesh)
    del params
    state_shardings = tree_shardings(state)
    behavior_snapshot = None
    if args.resume:
        # 只保留形状模板，先释放初始化态，再放置恢复态，避免两份 Adam/master 同时占 HBM。
        template = jax.eval_shape(lambda: state)
        del state
        if args.rollout_lag_updates:
            restored = load_lagged_train_state(args.resume, template, state_shardings)
            state, behavior_snapshot = restored
            del restored
        else:
            state = load_train_state(args.resume, template, shardings=state_shardings)
    jax.block_until_ready(state)
    start_step = int(state.step)
    validate_update_schedule(
        updates_per_rollout=args.updates_per_rollout,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        start_step=start_step,
    )
    data_cursor = 0 if resume is None else resume.get("data_cursor")
    if not isinstance(data_cursor, int) or data_cursor < start_step // args.updates_per_rollout:
        raise ValueError("checkpoint缺少有效的候选batch数据游标")
    if not args.dynamic_sampling and data_cursor != start_step // args.updates_per_rollout:
        raise ValueError("固定采样的数据游标与完整rollout边界不一致")
    if start_step >= args.max_steps:
        raise ValueError(f"checkpoint step {start_step} 已达到 max-steps {args.max_steps}")
    state_bytes = sum(leaf.size * leaf.dtype.itemsize for leaf in jax.tree.leaves(state))
    behavior_bytes = (
        sum(leaf.size * 2 for leaf in jax.tree.leaves(state.params_f32)) + 4 if args.rollout_lag_updates else 0
    )
    future_checkpoint = bool(
        args.save_every and (start_step // args.save_every + 1) * args.save_every <= args.max_steps
    ) or any(step > start_step for step in (args.save_at_steps or []))
    if future_checkpoint:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(args.checkpoint_dir).free < state_bytes + behavior_bytes + 2**30:
            raise ValueError(f"checkpoint 需要至少 {(state_bytes + behavior_bytes) / 2**30:.2f} GiB 加 1 GiB 余量")
    print(
        json.dumps(
            {
                "run_config": run_config,
                "start_step": start_step,
                "state_bytes": state_bytes,
                "behavior_host_bytes": behavior_bytes,
            }
        ),
        flush=True,
    )

    def policy_params() -> Any:
        return state.params_f32 if frozen_base is None else LoRAPolicyParams(frozen_base, state.params_f32)

    parameter_shardings = (
        state_shardings.params_f32
        if frozen_base is None
        else LoRAPolicyParams(tree_shardings(frozen_base), state_shardings.params_f32)
    )
    rollout_shardings = (
        named_shardings(param_specs_rollout(config, layout=args.rollout_layout, num_devices=mesh.size), mesh)
        if frozen_base is None
        else parameter_shardings
    )
    batch_sharding = NamedSharding(mesh, batch_spec(2))
    sampler_config = SamplerConfig(
        args.max_prompt_len, args.max_new_tokens, eos_ids=load_hf_eos_token_ids(str(args.model_path), config)
    )

    def sample_policy(current: Any, ids: Any, mask: Any, key: Any, sampler: SamplerConfig) -> RolloutBatch:
        if lora_config is None:
            return generate(current, config, ids, mask, sampler_config=sampler, key=key, mesh=mesh)
        return generate(
            current.frozen_base,
            config,
            ids,
            mask,
            sampler_config=sampler,
            key=key,
            mesh=mesh,
            lora=prepare_lora_params(current.adapters_f32, lora_config),
        )

    reshard_fn = jax.jit(
        lambda current: (
            reshard_for_rollout(current, config, mesh, layout=args.rollout_layout) if lora_config is None else current
        ),
        in_shardings=(parameter_shardings,),
        out_shardings=rollout_shardings,
    )
    generate_fn = jax.jit(
        lambda current, ids, mask, key: sample_policy(current, ids, mask, key, sampler_config),
        in_shardings=(rollout_shardings, batch_sharding, batch_sharding, None),
    )
    logps_fn = jax.jit(
        lambda current, ids, mask, completions, completion_mask: trainer_completion_logps(
            current if lora_config is None else current.frozen_base,
            ids,
            mask,
            completions,
            completion_mask,
            config=config,
            compute_dtype=jnp.bfloat16,
            vocab_chunk=args.vocab_chunk,
            sequence_chunk=args.sequence_chunk,
            microbatch_size=args.microbatch_size,
            remat_layers=args.remat,
            mesh=mesh,
            lora=None if lora_config is None else prepare_lora_params(current.adapters_f32, lora_config),
        ),
        in_shardings=(parameter_shardings,) + (batch_sharding,) * 4,
        out_shardings=batch_sharding,
    )

    def update_policy(
        current: TrainState[Any],
        ids: Any,
        mask: Any,
        completions: Any,
        cmask: Any,
        advantages: Any,
        old_logps: Any,
        ref_logps: Any,
        loss_mask: Any,
        behavior: Any,
        capture: Any,
        frozen: Gemma4TextParams | None = None,
    ) -> Any:
        return grpo_train_step(
            current,
            ids,
            mask,
            completions,
            cmask,
            advantages,
            old_logps,
            ref_logps,
            config=config,
            optimizer=optimizer,
            trainable_mask=trainable,
            compute_dtype=jnp.bfloat16,
            vocab_chunk=args.vocab_chunk,
            sequence_chunk=args.sequence_chunk,
            remat_layers=args.remat,
            mesh=mesh,
            beta=args.beta,
            microbatch_size=args.microbatch_size,
            loss_mask=loss_mask,
            use_current_policy_as_old=capture,
            behavior_logps=behavior,
            sampler_is_cap=args.sampler_is_cap,
            frozen_base=frozen,
            lora_config=lora_config,
            **objective_options,
        )

    update_fn = jax.jit(
        update_policy,
        donate_argnums=(0,),
        in_shardings=(state_shardings,)
        + (batch_sharding,) * 4
        + (
            NamedSharding(mesh, batch_spec(1)),
            batch_sharding if needs_old_logps else None,
            None if reference_host is None else batch_sharding,
            batch_sharding if args.filter_truncated_loss else None,
            batch_sharding if args.sampler_is_cap is not None else None,
            None,
        )
        + (() if frozen_base is None else (tree_shardings(frozen_base),)),
        out_shardings=(state_shardings, None),
    )
    update_extra = () if frozen_base is None else (frozen_base,)
    reshard_executable = (
        compile_call("reshard", reshard_fn, (policy_params(),), args.output_dir, args.save_hlo)
        if args.rollout_backend == "jax"
        else None
    )
    capture_executable = None
    if args.rollout_lag_updates:
        capture_fn = jax.jit(
            lambda current: cast_floating_tree(current, jnp.bfloat16),
            in_shardings=(parameter_shardings,),
            out_shardings=parameter_shardings,
        )
        capture_executable = compile_call(
            "behavior_capture", capture_fn, (state.params_f32,), args.output_dir, args.save_hlo
        )
    generate_executable = update_executable = logps_executable = reference_executable = None
    eval_executable = None
    eval_sampler_config = SamplerConfig(
        args.max_prompt_len,
        args.eval_max_new_tokens,
        temperature=0,
        eos_ids=sampler_config.eos_ids,
    )
    reward_config = RewardConfig(
        success_reward=run_config["success_reward"],
        failure_reward=run_config["failure_reward"],
        format_bonus=args.format_bonus,
        max_completion_length=args.max_new_tokens,
        overlong_buffer_length=args.overlong_buffer_length,
        overlong_penalty=args.overlong_penalty,
    )
    records: list[dict[str, Any]] = []
    base_key = jax.random.PRNGKey(args.seed)
    with (
        ExitStack() as stack,
        ProcessPoolExecutor(max_workers=args.reward_workers, mp_context=multiprocessing.get_context("spawn")) as pool,
        (args.output_dir / "metrics.csv").open("w", newline="") as csv_file,
        (args.output_dir / "rollouts.jsonl").open("w") as samples_file,
    ):
        inference_rollout = None
        inference_samples = None
        if args.rollout_backend != "jax":
            from gemma4_posttrain_jax.inference_remote import RemoteEngineRuntime
            from gemma4_posttrain_jax.inference_rollout import InferenceRollout
            from gemma4_posttrain_jax.inference_runtime import EngineConfig, EngineRuntime

            assert args.inference_device_ids is not None
            inference_dp = len(args.inference_device_ids)
            engine_init_started = time.perf_counter()
            engine_config = EngineConfig(
                model_path=str(args.model_path),
                device_indexes=tuple(args.inference_device_ids),
                dp_size=inference_dp,
                max_num_seqs_per_rank=(rows + inference_dp - 1) // inference_dp,
                token_budget_per_rank=max(512, ((rows + inference_dp - 1) // inference_dp) * args.max_prompt_len),
                max_model_len=args.max_prompt_len + args.max_new_tokens,
                memory_fraction=args.inference_memory_fraction,
                max_logprobs=128,
            )
            runtime: EngineRuntime | RemoteEngineRuntime
            if process_inference:
                assert args.inference_python is not None and inference_identity is not None
                runtime = RemoteEngineRuntime(
                    engine_config,
                    python_executable=args.inference_python,
                    record_path=args.output_dir / "inference_worker.json",
                    expected_environment=inference_identity,
                )
            else:
                runtime = EngineRuntime(engine_config)

            def close_inference() -> None:
                report = runtime.close()
                write_json(args.output_dir / "inference_close.json", report)
                if not report.get("complete"):
                    primary = sys.exception()
                    if primary is not None:
                        primary.add_note("独立引擎关闭也未完整通过，见inference_close.json；保留原始失败")
                    else:
                        raise RuntimeError("独立引擎关闭未完整通过，见inference_close.json")

            stack.callback(close_inference)
            write_json(
                args.output_dir / "inference_initialization.json",
                {
                    "load_and_initialize_s": time.perf_counter() - engine_init_started,
                    "first_generate_includes_jit": True,
                    "metadata": runtime.metadata,
                    "计时说明": "引擎首次生成包含按实际采样配置编译，不计为稳态；初始化和真实权重运输另列。",
                },
            )
            inference_rollout = InferenceRollout(runtime, config, sampler_config=sampler_config, mesh=mesh)
            inference_samples = stack.enter_context((args.output_dir / "inference.jsonl").open("w"))
        tracker = init_tracker(
            enabled=args.wandb,
            project=args.wandb_project,
            run_name=args.wandb_run_name,
            tags=args.wandb_tags,
            config={"logging_mode": "live", "source_metadata": json.loads((args.output_dir / "meta.json").read_text())},
        )
        stack.callback(tracker.finish)
        writer = None

        def run_eval(global_step: int) -> tuple[float, dict[str, Any]]:
            nonlocal eval_executable
            assert reshard_executable is not None
            start = time.perf_counter()
            eval_params, _ = timed_call(reshard_executable, policy_params())
            if eval_executable is None:
                first_batch = shard_batch((eval_batches[0].prompt_ids, eval_batches[0].prompt_mask), mesh)
                eval_executable = compile_call(
                    "eval",
                    make_eval_sampler(config, eval_sampler_config, mesh, rollout_layout=args.rollout_layout)
                    if lora_config is None
                    else jax.jit(
                        lambda current, ids, mask, key: sample_policy(current, ids, mask, key, eval_sampler_config),
                        in_shardings=(rollout_shardings, batch_sharding, batch_sharding, None),
                    ),
                    (eval_params, *first_batch, jax.random.PRNGKey(args.eval_seed)),
                    args.output_dir,
                    args.save_hlo,
                )
            summary = evaluate_batches(
                lambda current, prompt, valid, key: eval_executable(current, *shard_batch((prompt, valid), mesh), key),
                eval_params,
                eval_batches,
                tokenizer,
                output_dir=args.output_dir / "eval" / f"step_{global_step:08d}",
                eos_ids=sampler_config.eos_ids,
                seed=args.eval_seed,
                executor=pool,
            )
            del eval_params
            elapsed = time.perf_counter() - start
            print(json.dumps({"eval_step": global_step, "eval_s": elapsed, **summary}), flush=True)
            return elapsed, summary

        if eval_batches:
            initial_eval_s, initial_eval = run_eval(start_step)
            tracker.log(
                {
                    "time_eval_s": initial_eval_s,
                    **grpo_tracking_metrics(evaluation=initial_eval),
                    **compile_tracking_metrics(args.output_dir),
                },
                step=start_step,
            )
        for rollout_step in range(start_step // args.updates_per_rollout, args.max_steps // args.updates_per_rollout):
            step = rollout_step * args.updates_per_rollout
            step_start = time.perf_counter()
            behavior_step = step if behavior_snapshot is None else int(behavior_snapshot.step)
            if behavior_step != max(0, step - args.rollout_lag_updates):
                raise RuntimeError("采样权重没有保持声明的一次Adam滞后")
            if inference_rollout is not None:
                _, reshard_s = timed_call(partial(inference_rollout.update_params, version=step), policy_params())
                rollout_params = None
                write_json(args.output_dir / f"inference_weights_{step:06d}.json", inference_rollout.last_update)
            elif behavior_snapshot is None:
                assert reshard_executable is not None
                rollout_params, reshard_s = timed_call(reshard_executable, policy_params())
            else:
                rollout_params, reshard_s = timed_call(jax.device_put, behavior_snapshot.params_bf16, rollout_shardings)
            pieces: dict[str, list[np.ndarray]] = {}
            selected_questions: list[str] = []
            selected_golds: list[Gold] = []
            indices: list[int] = []
            accepted_groups = attempts = generated_tokens = 0
            rollout_s = reward_s = rollout_audit_s = 0.0
            while accepted_groups < args.prompt_batch_size:
                if attempts == args.max_sampling_attempts:
                    raise RuntimeError(
                        f"补采{attempts}批后仅保留{accepted_groups}/{args.prompt_batch_size}个有效组；未更新参数，生成证据已保存"
                    )
                positions = grpo_prompt_indices(
                    len(training_indices), args.prompt_batch_size, data_cursor, seed=args.seed
                )
                candidate_indices = [training_indices[position] for position in positions]
                prompts = collate_grpo_prompts(
                    [dataset[index] for index in candidate_indices],
                    tokenizer,
                    group_size=args.group_size,
                    max_prompt_len=args.max_prompt_len,
                    pad_token_id=config.pad_token_id,
                    chat=args.prompt_style == "chat",
                )
                ids, mask = shard_batch((prompts.prompt_ids, prompts.prompt_mask), mesh)
                key = jax.random.fold_in(base_key, data_cursor)
                data_cursor += 1
                attempts += 1
                if inference_rollout is not None:
                    rollout, elapsed = timed_call(partial(inference_rollout.generate, key=key), ids, mask)
                    if inference_rollout.last_proposal_logps is None:
                        raise RuntimeError("完整策略采样未返回实际proposal概率，不能接受训练batch")
                    assert inference_samples is not None and inference_rollout.last_generation is not None
                    inference_samples.write(
                        json.dumps(
                            {"data_cursor": data_cursor - 1, "step": step, **inference_rollout.last_generation},
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    inference_samples.flush()
                else:
                    if generate_executable is None:
                        generate_executable = compile_call(
                            "rollout", generate_fn, (rollout_params, ids, mask, key), args.output_dir, args.save_hlo
                        )
                    rollout, elapsed = timed_call(generate_executable, rollout_params, ids, mask, key)
                rollout_s += elapsed
                completion_ids, completion_mask, lengths, behavior_logps = jax.device_get(
                    (rollout.completion_ids, rollout.completion_mask, rollout.lengths, rollout.rollout_logps)
                )
                if args.audit_rollout_batches:
                    audit_start = time.perf_counter()
                    audit_dir = args.output_dir / "rollout_batches"
                    audit_dir.mkdir(exist_ok=True)
                    proposal_logps = (
                        np.asarray(jax.device_get(inference_rollout.last_proposal_logps))
                        if inference_rollout is not None
                        else behavior_logps
                        if sampler_config.temperature == 1 and sampler_config.top_k == 0 and sampler_config.top_p == 1
                        else None
                    )
                    # 原生非T1/full-vocab仅返回raw策略概率；明确缺失proposal，禁止伪造。
                    with (audit_dir / f"batch_{data_cursor - 1:08d}.npz").open("xb") as audit_file:
                        np.savez(
                            audit_file,
                            protocol=np.asarray("rollout-candidate-arrays-v1"),
                            backend=np.asarray(args.rollout_backend),
                            data_cursor=np.asarray(data_cursor - 1, np.int64),
                            policy_step=np.asarray(behavior_step, np.int64),
                            target_old_policy_step=np.asarray(step, np.int64),
                            dataset_indices=np.repeat(np.asarray(candidate_indices, np.int64), args.group_size),
                            key_data=np.asarray(jax.device_get(jax.random.key_data(key))),
                            prompt_ids=prompts.prompt_ids,
                            prompt_mask=prompts.prompt_mask,
                            completion_ids=completion_ids,
                            completion_mask=completion_mask,
                            lengths=lengths,
                            raw_logprobs=behavior_logps,
                            proposal_logprobs=np.empty((0,), np.float32) if proposal_logps is None else proposal_logps,
                            proposal_known=np.asarray(proposal_logps is not None),
                            temperature=np.asarray(sampler_config.temperature, np.float64),
                            top_k=np.asarray(sampler_config.top_k, np.int64),
                            top_p=np.asarray(sampler_config.top_p, np.float64),
                            eos_ids=np.asarray(sampler_config.eos_ids, np.int32),
                        )
                    rollout_audit_s += time.perf_counter() - audit_start
                generated_tokens += int(lengths.sum())
                completions = tokenizer.batch_decode(
                    [tokens[:length] for tokens, length in zip(completion_ids, lengths, strict=True)],
                    skip_special_tokens=True,
                )
                reward_start = time.perf_counter()
                reward = score_completions(
                    completions, prompts.golds, completion_lengths=lengths, config=reward_config, executor=pool
                )
                reward_s += time.perf_counter() - reward_start
                truncated = (lengths == args.max_new_tokens) & ~np.isin(completion_ids[:, -1], sampler_config.eos_ids)
                training_rewards = reward.score.copy()
                if args.truncated_reward == "zero":
                    training_rewards[truncated] = 0.0
                eligible = (
                    mixed_success_groups(reward.task_success, args.group_size)
                    if args.dynamic_sampling
                    else np.ones(args.prompt_batch_size, bool)
                )
                if args.dynamic_sampling and args.filter_truncated_loss:
                    eligible &= (~truncated).reshape(-1, args.group_size).any(axis=-1)
                keep = np.flatnonzero(eligible)[: args.prompt_batch_size - accepted_groups]
                kept_rows = (keep[:, None] * args.group_size + np.arange(args.group_size)).reshape(-1)
                training_rows = np.full(rows, -1, np.int32)
                training_rows[kept_rows] = accepted_groups * args.group_size + np.arange(len(kept_rows))
                for index, text in enumerate(completions):
                    samples_file.write(
                        json.dumps(
                            {
                                "step": step + 1,
                                "rollout_step": rollout_step + 1,
                                "generation_batch": data_cursor,
                                "behavior_policy_step": behavior_step,
                                "target_old_policy_step": step,
                                "row": index,
                                "training_row": int(training_rows[index]),
                                "selected_for_update": bool(training_rows[index] >= 0),
                                "dataset_index": candidate_indices[index // args.group_size],
                                "completion": text,
                                "length": int(lengths[index]),
                                "truncated": bool(truncated[index]),
                                "gold": prompts.golds[index],
                                "training_reward": float(training_rewards[index]),
                                **{name: float(array[index]) for name, array in reward._asdict().items()},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                samples_file.flush()
                if kept_rows.size:
                    candidates = {
                        "ids": prompts.prompt_ids,
                        "mask": prompts.prompt_mask,
                        "completion_ids": completion_ids,
                        "completion_mask": completion_mask,
                        "lengths": lengths,
                        "behavior_logps": behavior_logps,
                        "truncated": truncated,
                        "training_rewards": training_rewards,
                        **reward._asdict(),
                    }
                    for name, array in candidates.items():
                        pieces.setdefault(name, []).append(array[kept_rows])
                    selected_questions.extend(prompts.questions[index] for index in kept_rows)
                    selected_golds.extend(prompts.golds[index] for index in kept_rows)
                    indices.extend(candidate_indices[index] for index in keep)
                    accepted_groups += len(keep)
            del rollout_params
            combined = {name: np.concatenate(values) for name, values in pieces.items()}
            prompts = PromptBatch(combined["ids"], combined["mask"], tuple(selected_questions), tuple(selected_golds))
            ids, mask = shard_batch((prompts.prompt_ids, prompts.prompt_mask), mesh)
            completion_ids, completion_mask, lengths = (
                combined[name] for name in ("completion_ids", "completion_mask", "lengths")
            )
            truncated, training_rewards = combined["truncated"], combined["training_rewards"]
            reward = RewardOutput(*(combined[name] for name in RewardOutput._fields))
            rollout = RolloutBatch(
                ids, *shard_batch((completion_ids, completion_mask, combined["behavior_logps"], lengths), mesh)
            )
            advantages = shard_batch(
                objective_advantages(jnp.asarray(training_rewards), args.group_size, objective.advantage_estimator),
                mesh,
            )
            loss_mask = None
            if args.filter_truncated_loss:
                loss_mask = shard_batch(
                    completion_loss_mask(rollout.completion_mask, jnp.asarray(truncated), filter_truncated=True), mesh
                )
                if not np.asarray(loss_mask).any():
                    raise RuntimeError("整批loss mask为空，保留生成记录并停止，不执行无信号的Adam更新")
            token_arguments = (ids, mask, rollout.completion_ids, rollout.completion_mask)
            reference_logps = None
            reference_s = startup_logps_s = 0.0
            if reference_host is not None:
                # 参考模型与 rollout 复制工作区不能同时常驻 v4 HBM；只在本阶段放入 FSDP 分片。
                reference, reference_transfer_s = timed_call(
                    lambda: shard_gemma4_text_params(reference_host, config, mesh)
                )
                if reference_executable is None:
                    reference_executable = compile_call(
                        "reference", logps_fn, (reference, *token_arguments), args.output_dir, args.save_hlo
                    )
                reference_logps, reference_s = timed_call(reference_executable, reference, *token_arguments)
                reference_s += reference_transfer_s
                del reference
            # 独立前向仅作为首批sampler诊断；μ轮old从实际联合梯度前向捕获。
            old_logps_s = 0.0
            is_weights = None
            parity_arrays = None
            if step == start_step:
                logps_executable = compile_call(
                    "startup_logps", logps_fn, (policy_params(), *token_arguments), args.output_dir, args.save_hlo
                )
                trainer_logps, startup_logps_s = timed_call(logps_executable, policy_params(), *token_arguments)
                behavior_trainer_logps = trainer_logps
                if behavior_step != step:
                    assert behavior_snapshot is not None
                    behavior_params, transfer_s = timed_call(
                        shard_gemma4_text_params, behavior_snapshot.params_bf16, config, mesh
                    )
                    behavior_logps_executable = compile_call(
                        "startup_behavior_logps",
                        logps_fn,
                        (behavior_params, *token_arguments),
                        args.output_dir,
                        args.save_hlo,
                    )
                    behavior_trainer_logps, elapsed = timed_call(
                        behavior_logps_executable, behavior_params, *token_arguments
                    )
                    startup_logps_s += elapsed + transfer_s
                    del behavior_params, behavior_logps_executable
                    target_drift = logprob_drift_metrics(
                        trainer_logps,
                        rollout.rollout_logps,
                        rollout.completion_mask,
                        delta_definition="standalone_current_trainer_logp - lagged_behavior_rollout_logp",
                    )
                    write_json(args.output_dir / "startup_target_drift.json", target_drift)
                drift = logprob_drift_metrics(
                    behavior_trainer_logps,
                    rollout.rollout_logps,
                    rollout.completion_mask,
                    delta_definition="standalone_behavior_trainer_logp - behavior_rollout_full_policy_logp",
                )
                write_json(args.output_dir / "startup_drift.json", drift)
                print(json.dumps({"startup_drift": drift}), flush=True)
                parity_arrays = {
                    "behavior_policy_step": np.asarray(behavior_step),
                    "target_old_policy_step": np.asarray(step),
                    "behavior_trainer_logps": np.asarray(behavior_trainer_logps),
                    "prompt_ids": prompts.prompt_ids,
                    "prompt_mask": prompts.prompt_mask,
                    "completion_ids": completion_ids,
                    "completion_mask": completion_mask,
                    "loss_mask": completion_mask if loss_mask is None else np.asarray(loss_mask),
                    "sampler_is_weights": np.empty((0,), np.float32),
                    "objective_config_json": np.asarray(json.dumps(objective._asdict())),
                    "group_size": np.asarray(args.group_size),
                    "beta": np.asarray(args.beta),
                    "rewards": training_rewards,
                    "raw_rewards": reward.score,
                    "advantages": np.asarray(advantages),
                    "trainer_logps": np.asarray(trainer_logps),
                    "old_logps": np.asarray(trainer_logps),
                    "rollout_logps": np.asarray(rollout.rollout_logps),
                    "reference_logps": np.empty((0,), np.float32)
                    if reference_logps is None
                    else np.asarray(reference_logps),
                }
                filename = "startup_parity_batch.npz" if needs_old_logps else "parity_batch.npz"
                np.savez(args.output_dir / filename, allow_pickle=False, **parity_arrays)
                if not (float(drift["ratio_p01"]) >= 0.9 and float(drift["ratio_p99"]) <= 1.1):
                    raise RuntimeError("启动独立前向H2 drift gate 未通过；已保存固定输入，尚未更新参数")
                if not np.isfinite(np.asarray(trainer_logps)[completion_mask]).all():
                    raise RuntimeError("trainer logprob 含非有限值")
                del trainer_logps, behavior_trainer_logps, logps_executable
                logps_executable = None
            old_for_update = shard_batch(np.zeros(completion_ids.shape, np.float32), mesh) if needs_old_logps else None
            behavior_for_update = rollout.rollout_logps if args.sampler_is_cap is not None else None
            capture = jnp.asarray(True) if needs_old_logps else None
            update_batch = (
                *token_arguments,
                advantages,
                old_for_update,
                reference_logps,
                loss_mask,
                behavior_for_update,
                capture,
            )
            if update_executable is None:
                update_executable = compile_call(
                    "update",
                    update_fn,
                    (state, *update_batch, *update_extra),
                    args.output_dir,
                    args.save_hlo,
                )
            for iteration in range(args.updates_per_rollout):
                step = rollout_step * args.updates_per_rollout + iteration
                iteration_start = step_start if iteration == 0 else time.perf_counter()
                behavior_snapshot_s = 0.0
                if capture_executable is not None and iteration == args.updates_per_rollout - 1:
                    # 在最后一次Adam前捕获；下一轮恰好滞后一次更新，而不是滞后整个μ轮。
                    snapshot_start = time.perf_counter()
                    captured_params = capture_executable(state.params_f32)
                    behavior_snapshot = BehaviorSnapshot(
                        jax.device_put(captured_params, jax.devices("cpu")[0]),
                        jax.device_put(np.asarray(step, np.int32), jax.devices("cpu")[0]),
                    )
                    jax.block_until_ready(behavior_snapshot)
                    del captured_params
                    behavior_snapshot_s = time.perf_counter() - snapshot_start
                update_audit_s = 0.0
                if args.audit_update_state:
                    from gemma4_posttrain_jax.state_audit import compare_parameter_witness, snapshot_parameter_witness

                    audit_start = time.perf_counter()
                    witness_before_step = int(state.step)
                    witness_before = snapshot_parameter_witness(state.params_f32, trainable)
                    update_audit_s += time.perf_counter() - audit_start
                (state, metrics), update_s = timed_call(update_executable, state, *update_batch, *update_extra)
                if args.audit_update_state:
                    audit_start = time.perf_counter()
                    witness_after = snapshot_parameter_witness(state.params_f32, trainable)
                    witness = compare_parameter_witness(witness_before, witness_after)
                    witness.update(before_step=witness_before_step, after_step=int(state.step))
                    write_json(args.output_dir / f"update_witness_{step + 1:08d}.json", witness)
                    del witness_before, witness_after
                    if not witness["all_finite"] or witness["after_step"] != witness["before_step"] + 1:
                        raise RuntimeError("实际参数更新观察非有限或Adam步数未推进")
                    update_audit_s += time.perf_counter() - audit_start

                if needs_old_logps and iteration == 0:
                    if metrics.policy_logps is None:
                        raise RuntimeError("联合图没有返回轮内old logps")
                    old_for_update = shard_batch(metrics.policy_logps, mesh)
                    # 后续μ更新只读这份首次前向数组；不能被第二次current logps覆盖。
                    update_batch = (
                        *token_arguments,
                        advantages,
                        old_for_update,
                        reference_logps,
                        loss_mask,
                        behavior_for_update,
                        jnp.asarray(False),
                    )
                    if args.sampler_is_cap is not None:
                        is_weights = truncated_importance_weights(
                            old_for_update, rollout.rollout_logps, rollout.completion_mask, cap=args.sampler_is_cap
                        )
                    if parity_arrays is not None:
                        captured = np.asarray(old_for_update)
                        parity_arrays["standalone_trainer_logps"] = parity_arrays["trainer_logps"]
                        parity_arrays["trainer_logps"] = parity_arrays["old_logps"] = captured
                        parity_arrays["sampler_is_weights"] = (
                            np.empty((0,), np.float32) if is_weights is None else np.asarray(is_weights)
                        )
                        np.savez(args.output_dir / "parity_batch.npz", allow_pickle=False, **parity_arrays)
                        joint_drift = logprob_drift_metrics(
                            captured,
                            rollout.rollout_logps,
                            rollout.completion_mask,
                            delta_definition="joint_trainer_old_logp - rollout_full_policy_logp",
                        )
                        write_json(args.output_dir / "joint_old_drift.json", joint_drift)
                        print(json.dumps({"joint_old_drift": joint_drift}), flush=True)
                values = {name: float(value) for name, value in metrics.loss_metrics._asdict().items()}
                values["grad_norm"] = float(metrics.grad_norm)
                if not all(np.isfinite(value) for value in values.values()):
                    raise RuntimeError(f"step {step + 1} loss/gradient/metrics 含非有限值：{values}")
                if (args.updates_per_rollout == 1 or iteration == 0) and (values["ratio_min"], values["ratio_max"]) != (
                    1.0,
                    1.0,
                ):
                    raise RuntimeError("首个联合前向的old/current ratio不再为1")
                row = {
                    "step": step + 1,
                    "rollout_step": rollout_step + 1,
                    "update_in_rollout": iteration + 1,
                    "behavior_policy_step": behavior_step,
                    "target_old_policy_step": rollout_step * args.updates_per_rollout,
                    "current_policy_step": step,
                    "behavior_snapshot_s": behavior_snapshot_s,
                    "behavior_snapshot_bytes": behavior_bytes if behavior_snapshot_s else 0,
                    "generated_rows": rows * attempts if iteration == 0 else 0,
                    "generated_tokens": generated_tokens if iteration == 0 else 0,
                    "sampling_attempts": attempts if iteration == 0 else 0,
                    "data_cursor": data_cursor,
                    "retained_prompt_groups": accepted_groups if iteration == 0 else 0,
                    "first_call": int(step == start_step),
                    **values,
                    "reward_mean": float(training_rewards.mean()),
                    "raw_reward_mean": float(reward.score.mean()),
                    "task_success": float(reward.task_success.mean()),
                    "format_reward": float(reward.format_reward.mean()),
                    "length_mean": float(lengths.mean()),
                    "truncated_fraction": float(
                        np.mean(
                            ~np.isin(completion_ids[:, -1], sampler_config.eos_ids) & (lengths == args.max_new_tokens)
                        )
                    ),
                    "mixed_success_group_fraction": float(
                        np.mean(
                            (reward.task_success.reshape(-1, args.group_size).mean(axis=1) > 0)
                            & (reward.task_success.reshape(-1, args.group_size).mean(axis=1) < 1)
                        )
                    ),
                    "all_correct_group_fraction": float(
                        np.mean(reward.task_success.reshape(-1, args.group_size).mean(axis=1) == 1)
                    ),
                    "all_wrong_group_fraction": float(
                        np.mean(reward.task_success.reshape(-1, args.group_size).mean(axis=1) == 0)
                    ),
                    "nonzero_advantage_fraction": float(np.mean(np.asarray(advantages) != 0)),
                    "reshard_s": reshard_s if iteration == 0 else 0.0,
                    "rollout_s": rollout_s if iteration == 0 else 0.0,
                    "reward_s": reward_s if iteration == 0 else 0.0,
                    "startup_logps_s": startup_logps_s if iteration == 0 else 0.0,
                    "reference_s": reference_s if iteration == 0 else 0.0,
                    "update_s": update_s,
                    "update_audit_s": update_audit_s,
                    "old_logps_s": old_logps_s if iteration == 0 else 0.0,
                    "sampler_is_mean": 1.0
                    if is_weights is None
                    else float(np.asarray(is_weights)[completion_mask].mean()),
                    "generated_tokens_per_s": float(generated_tokens / rollout_s) if iteration == 0 else 0.0,
                    "completion_tokens_per_update_s": float(values["completion_tokens"] / update_s),
                    "update_mfu_estimate": 6
                    * 2.3e9
                    * rows
                    * (args.max_prompt_len + args.max_new_tokens)
                    / update_s
                    / (1.1e15 * mesh.size / 4),
                    "peak_hbm_gib": max(value or 0 for value in device_peak_bytes()) / 2**30,
                    "step_s": time.perf_counter() - iteration_start,
                    "checkpoint_s": 0.0,
                    "eval_s": 0.0,
                }
                if args.audit_rollout_batches:
                    row["rollout_audit_s"] = rollout_audit_s if iteration == 0 else 0.0
                if lora_config is not None:
                    # 原6*2.3B估计只适用于旧E2B全参数图，不用于冻结base的LoRA性能声明。
                    row.pop("update_mfu_estimate")
                if (args.save_every and (step + 1) % args.save_every == 0) or (step + 1) in (args.save_at_steps or []):
                    checkpoint_start = time.perf_counter()
                    checkpoint_path = args.checkpoint_dir / f"step_{step + 1:08d}"
                    checkpoint_state = (
                        state if behavior_snapshot is None else LaggedTrainState(state, behavior_snapshot)
                    )
                    save_train_state(
                        checkpoint_state,
                        checkpoint_path,
                        metadata={"run_config": run_config, "git_commit": revision, "data_cursor": data_cursor},
                    )
                    del checkpoint_state
                    row["checkpoint_s"] = time.perf_counter() - checkpoint_start
                    print(json.dumps({"checkpoint": str(checkpoint_path), "save_s": row["checkpoint_s"]}), flush=True)
                eval_summary = None
                if eval_batches and (step + 1) % args.eval_every == 0:
                    row["eval_s"], eval_summary = run_eval(step + 1)
                if writer is None:
                    writer = csv.DictWriter(csv_file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                csv_file.flush()
                records.append(row)
                print(json.dumps(row, allow_nan=False), flush=True)
                tracked = grpo_tracking_metrics(row, eval_summary)
                if step == start_step:
                    tracked.update(compile_tracking_metrics(args.output_dir))
                tracker.log(tracked, step=step + 1)
    if args.audit_final_state:
        from gemma4_posttrain_jax.state_audit import summarize_train_state

        final_audit = summarize_train_state(state)
        write_json(args.output_dir / "final_state_audit.json", final_audit)
        if not final_audit["all_finite"]:
            raise RuntimeError("最终训练状态存在非有限叶；完整摘要已保存")
    write_json(
        args.output_dir / "summary.json",
        {
            "start_step": start_step,
            "final_step": int(state.step),
            "final_data_cursor": data_cursor,
            "generated_rows": sum(row["generated_rows"] for row in records),
            "generated_tokens": sum(row["generated_tokens"] for row in records),
            "peak_bytes": device_peak_bytes(),
            "steady_steps": len(records[1:]),
            "steady_mean": {
                field: float(np.mean([row[field] for row in records[1:]]))
                for field in ("reshard_s", "rollout_s", "reward_s", "old_logps_s", "reference_s", "update_s", "step_s")
            }
            if len(records) > 1
            else {},
            "checkpoint_persistent": args.checkpoint_dir is not None
            and not str(args.checkpoint_dir).startswith("/dev/shm/"),
        },
    )


if __name__ == "__main__":
    main()
