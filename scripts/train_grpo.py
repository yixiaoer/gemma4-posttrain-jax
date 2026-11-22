#!/usr/bin/env python3
"""Synchronous GRPO-family training for Gemma 4 E2B: GSM8K rollouts, rewards, updates and resume."""

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
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import Any, Literal

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
from gemma4_posttrain_jax.diagnostics import (
    checkpoint_storage,
    execution_layout_record,
    logprob_drift_gate,
    logprob_drift_metrics,
    optional_package_version,
    source_git_state,
)
from gemma4_posttrain_jax.evaluation import evaluate_batches, make_eval_sampler, prepare_eval_batches
from gemma4_posttrain_jax.logprob_protocol import logprob_run_metadata
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
from gemma4_posttrain_jax.tp_rollout import resolve_allreduce
from gemma4_posttrain_jax.tracking import compile_tracking_metrics, grpo_tracking_metrics, init_tracker
from gemma4_posttrain_jax.weights import load_hf_eos_token_ids, load_hf_params

TIMING_PROTOCOL = "rl-iteration-stages-v3"
# Apply only to explicit TP rollout; replicated and FSDP paths retain their compiler settings.
UPDATE_COMPILER_OPTIONS = {"xla_tpu_scheduler_percent_shared_memory_limit": "75"}


def iteration_device_allocation(
    backend: str, training_ids: list[int], inference_ids: list[int] | None
) -> dict[str, int]:
    """Charge a full iteration by the devices it occupies; the same index in a separate runtime is not the same chip."""
    if not training_ids or len(set(training_ids)) != len(training_ids):
        raise ValueError("training devices must be non-empty and unique")
    if backend == "jax":
        if inference_ids:
            raise ValueError("native synchronous rollout reuses the training devices; do not assign inference devices")
        separate = 0
    elif backend in ("inference", "inference-process", "inference-distributed"):
        if not inference_ids or len(set(inference_ids)) != len(inference_ids):
            raise ValueError("inference devices must be non-empty and unique")
        if backend != "inference-process" and set(training_ids) & set(inference_ids):
            raise ValueError("training and inference devices in the same runtime must not overlap")
        separate = len(inference_ids)
    else:
        raise ValueError(f"unknown rollout backend: {backend}")
    return {
        "training_devices": len(training_ids),
        "separate_inference_devices": separate,
        "allocated_devices": len(training_ids) + separate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dataset", choices=("gsm8k", "math"), default="gsm8k")
    parser.add_argument(
        "--prompt-style",
        choices=("chat", "plain"),
        default="chat",
        help="use plain for base models; a resumed run must keep the same template",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="new run directory; existing outputs are never overwritten"
    )
    parser.add_argument("--rollout-layout", choices=("replicated", "fsdp", "tp"), default="replicated")
    parser.add_argument(
        "--tp-allreduce",
        choices=("auto", "xla", "one_shot_fused", "one_shot_fused_high_rows_local"),
        default="auto",
        help="TP 生成的归约方式；auto 根据批量选择已登记路径，恢复时继承 checkpoint 的实际方式",
    )
    parser.add_argument(
        "--rollout-backend", choices=("jax", "inference", "inference-process", "inference-distributed"), default="jax"
    )
    parser.add_argument(
        "--training-device-ids", type=int, nargs="+", help="explicit training mesh device IDs; all devices when omitted"
    )
    parser.add_argument(
        "--inference-device-ids",
        type=int,
        nargs="+",
        help="same process: 2 3 by default; separate runtime: local 0 1; shared runtime: chosen by the engine",
    )
    parser.add_argument(
        "--inference-python",
        type=Path,
        help="interpreter of the pinned inference environment used by the separate process",
    )
    parser.add_argument(
        "--inference-weight-transport",
        choices=("auto", "host", "device"),
        default="auto",
        help="auto (default): ICI for inference and inference-distributed, host transfer for inference-process",
    )
    parser.add_argument("--inference-memory-fraction", type=float, default=0.5)
    parser.add_argument(
        "--inference-source-read-mode",
        choices=("direct", "transient_copy"),
        default="direct",
        help="how training prepares the host copy of parameters for a separate inference process",
    )
    parser.add_argument(
        "--trim-training-host-allocator-after-transfer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="after a separate inference process receives the weights, return free host pages on the training side",
    )
    parser.add_argument(
        "--trim-inference-host-allocator-after-update",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="after the separate inference process applies an update, ask it to return free host pages",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        help="train only separate FP32 attention adapters; currently requires FSDP, beta 0 and lag 0",
    )
    parser.add_argument(
        "--lora-alpha", type=float, help="defaults to 2*rank; training and checkpoints keep the unscaled A/B"
    )
    parser.add_argument("--lora-targets", nargs="+", choices=("q_proj", "k_proj", "v_proj", "o_proj"))
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=8,
        help="number of distinct questions; generated rows are this value times G",
    )
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument(
        "--algorithm",
        choices=("grpo", "drgrpo", "dapo", "dapo-loss", "gspo-token", "rloo"),
        default="grpo",
        help="dapo also turns on dynamic sampling, overlong filtering and length penalty; dapo-loss does not",
    )
    parser.add_argument(
        "--updates-per-rollout", type=int, default=1, help="number of Adam updates (mu) on one complete rollout batch"
    )
    parser.add_argument(
        "--loss-aggregation",
        choices=("sequence-mean-token-mean", "token-mean", "sequence-mean-token-scale", "seq-mean-token-sum"),
        help="explicit loss aggregation; overrides the algorithm default",
    )
    parser.add_argument(
        "--token-scale",
        type=float,
        help="denominator of the constant-scale aggregation; defaults to the generation budget N",
    )
    parser.add_argument(
        "--sampler-is-cap", type=float, help="cap on the frozen per-token old-trainer/behavior correction"
    )
    parser.add_argument(
        "--rollout-lag-updates",
        type=int,
        choices=(0, 1),
        default=0,
        help="1: sample with weights one Adam update behind, run sequentially; training and generation do not overlap",
    )
    parser.add_argument(
        "--filter-truncated-loss",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="exclude overlong rows from the loss while keeping their context and raw reward",
    )
    parser.add_argument("--dynamic-sampling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--max-sampling-attempts", type=int, default=16, help="maximum candidate batches drawn to fill one valid batch"
    )
    parser.add_argument("--overlong-buffer-length", type=int, default=None)
    parser.add_argument("--overlong-penalty", type=float, default=None)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--microbatch-size",
        type=int,
        help="rows per training/reference forward; gradients accumulate into one update per rollout",
    )
    parser.add_argument("--max-steps", type=int, default=30, help="target global update step")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument(
        "--kl-clamp-value",
        type=float,
        help="optional per-token KL cap; enabling it changes the objective and the gradient above the cap",
    )
    parser.add_argument(
        "--truncated-reward",
        choices=("keep", "zero"),
        default="keep",
        help="zero: rows that hit the length limit without EOS get zero training reward; the raw score is kept",
    )
    parser.add_argument("--format-bonus", type=float, default=0.1)
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--remat", action="store_true")
    parser.add_argument("--vocab-chunk", type=int, default=8192)
    parser.add_argument("--sequence-chunk", type=int, default=256)
    parser.add_argument(
        "--logprob-backend",
        choices=("jax", "pallas"),
        default="jax",
        help="logprob backend; pallas is experimental (four TPU v4 chips, BF16, unresolved long-run differences)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument(
        "--save-at-steps",
        type=int,
        nargs="+",
        help="save only at these update steps (complete rollout boundaries); mutually exclusive with --save-every",
    )
    parser.add_argument(
        "--audit-final-state",
        action="store_true",
        help="record a checksum of every final parameter and Adam array for resume checks",
    )
    parser.add_argument(
        "--audit-execution-layout",
        action="store_true",
        help="record the processes, chips and shardings of training state and rollout weights before the first rollout",
    )
    parser.add_argument(
        "--audit-update-state",
        action="store_true",
        help="record the actual change of a few trainable arrays at every step",
    )
    parser.add_argument(
        "--audit-rollout-batches",
        action="store_true",
        help="keep the raw tokens, logprobs, masks and random keys of every candidate batch",
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--save-hlo",
        action="store_true",
        help="save StableHLO, optimized HLO and memory analysis for each JIT boundary",
    )
    parser.add_argument(
        "--eval-every", type=int, default=0, help="0 disables; otherwise evaluate at the start step and every N updates"
    )
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-max-new-tokens", type=int, default=1024)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument(
        "--dev-size", type=int, default=0, help="number of dev questions held out from the training data"
    )
    parser.add_argument("--dev-seed", type=int, default=0)
    parser.add_argument("--eval-split", choices=("test", "train-dev"), default="test")
    parser.add_argument(
        "--wandb", action="store_true", help="log to W&B from the host; set WANDB_MODE=offline to log offline"
    )
    parser.add_argument("--wandb-project", default="gemma4_posttrain_jax")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-tags", nargs="*", default=())
    parser.add_argument(
        "--wandb-group", help="W&B group shared by the continuous run, resumed run and evaluation of one experiment"
    )
    parser.add_argument("--wandb-job-type", help="W&B job type within the group, e.g. train or resume")
    from gemma4_posttrain_jax.inference_runtime import resolve_weight_sync_transport

    args = parser.parse_args()
    if args.logprob_backend == "pallas":
        if (args.vocab_chunk, args.sequence_chunk) != (8192, 256):
            parser.error("Pallas logprob requires vocab-chunk=8192 and sequence-chunk=256")
        if args.rollout_backend != "jax" or args.lora_rank is not None:
            parser.error(
                "Pallas logprob currently requires native rollout and full-model training on four TPU v4 chips"
            )
        if args.training_device_ids is not None and len(set(args.training_device_ids)) != 4:
            parser.error("Pallas logprob requires exactly four training devices")
    args.inference_weight_transport = resolve_weight_sync_transport(
        args.rollout_backend, args.inference_weight_transport
    )
    return args


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def compile_call(
    name: str,
    fn: Any,
    arguments: tuple[Any, ...],
    output_dir: Path,
    save_hlo: bool,
    compiler_options: dict[str, str] | None = None,
) -> Any:
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
        compiled = lowered.compile() if compiler_options is None else lowered.compile(compiler_options=compiler_options)
    except Exception as error:
        backend_compile_s = time.perf_counter() - backend_start
        record: dict[str, Any] = {
            "name": name,
            "status": "failed",
            "stage": "backend_compile",
            "compiler_options": compiler_options,
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
        "compiler_options": compiler_options,
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


def validate_resume_config(saved: Any, current: dict[str, Any]) -> None:
    """Check a resume configuration, filling in only old default fields that do not change training behavior."""
    if not isinstance(saved, dict):
        raise ValueError("checkpoint has no valid training configuration")
    restored = dict(saved)
    if "kl_estimator" not in restored:
        if restored.get("beta") != 0:
            raise ValueError(
                "the old checkpoint uses an unversioned KL formula and cannot be resumed exactly with the new K3 "
                "implementation; continue the original experiment with the code that saved it"
            )
        restored["kl_estimator"] = "none"
    restored.setdefault("kl_clamp_value", None)
    if restored != current:
        missing = object()
        changed = sorted(
            name
            for name in restored.keys() | current.keys()
            if restored.get(name, missing) != current.get(name, missing)
        )
        raise ValueError(f"resume configuration differs from the checkpoint: {', '.join(changed)}")


def main(*, cycle_observer: Callable[[int, Literal["start", "end"]], None] | None = None) -> None:
    args = parse_args()
    process_inference = args.rollout_backend == "inference-process"
    distributed_inference = args.rollout_backend == "inference-distributed"
    inference_identity = None
    memory_controls_enabled = (
        args.inference_source_read_mode != "direct"
        or args.trim_training_host_allocator_after_transfer
        or args.trim_inference_host_allocator_after_update
    )
    if memory_controls_enabled and not process_inference:
        raise ValueError("host-memory controls for weight sync apply only to the separate inference-process backend")
    if args.trim_training_host_allocator_after_transfer and args.inference_source_read_mode != "transient_copy":
        raise ValueError("training-side allocator release requires transient_copy")
    if args.rollout_backend != "jax":
        if args.lora_rank is not None or args.rollout_lag_updates or args.eval_every:
            raise ValueError(
                "the engine loop does not support LoRA, lag or periodic evaluation; evaluate checkpoints separately"
            )
        if args.sampler_is_cap is None:
            raise ValueError(
                "the engine loop requires an explicit --sampler-is-cap for the trainer/behavior correction"
            )
        if not 0 < args.inference_memory_fraction < 1:
            raise ValueError("inference memory fraction must be between 0 and 1")
        if distributed_inference:
            from gemma4_posttrain_jax.inference_shared import shared_devices

            ordered_devices = shared_devices(0)
            if not os.environ.get("GEMMA4_SHARED_ICI_SOCKET") or not os.environ.get("GEMMA4_SHARED_ICI_NONCE"):
                raise ValueError("the shared runtime needs the local control connection created by its launcher")
            if args.training_device_ids is None:
                args.training_device_ids = [int(d.id) for d in ordered_devices[:2]]
            if args.inference_device_ids is None:
                args.inference_device_ids = [int(d.id) for d in ordered_devices[2:]]
            if args.training_device_ids != [int(d.id) for d in ordered_devices[:2]] or args.inference_device_ids != [
                int(d.id) for d in ordered_devices[2:]
            ]:
                raise ValueError(
                    "the shared runtime uses rank 0 for training and rank 1 for inference, with two chips each"
                )
        if args.training_device_ids is None:
            args.training_device_ids = [0, 1]
        if args.inference_device_ids is None:
            args.inference_device_ids = [0, 1] if process_inference else [2, 3]
        if process_inference:
            from gemma4_posttrain_jax.inference_remote import inspect_engine_environment

            if os.environ.get("TPU_VISIBLE_CHIPS") != "0,1" or args.training_device_ids != [0, 1]:
                raise ValueError(
                    "separate-process mode requires training bound to physical chips 0,1 and using local devices 0,1"
                )
            if args.inference_device_ids != [0, 1] or args.inference_python is None or args.beta:
                raise ValueError(
                    "separate-process mode uses engine devices 0,1 and beta 0, and requires --inference-python"
                )
            inference_identity = inspect_engine_environment(args.inference_python)
        elif args.inference_python is not None:
            raise ValueError("--inference-python applies only to separate-process mode")
        # Engine calls use their own DEFAULT precision context; training still compiles at HIGHEST.
        jax.config.update("jax_default_matmul_precision", "highest")
    elif (
        args.inference_device_ids is not None
        or args.inference_memory_fraction != 0.5
        or args.inference_python is not None
        or memory_controls_enabled
    ):
        raise ValueError("inference device and memory options apply only to the inference backends")
    lora_config = None
    if args.lora_rank is not None:
        lora_config = LoRAConfig(
            args.lora_rank,
            2.0 * args.lora_rank if args.lora_alpha is None else args.lora_alpha,
            tuple(args.lora_targets or LoRAConfig().targets),
        )
        check_lora_config(lora_config)
        if args.beta or args.rollout_lag_updates or args.rollout_layout != "fsdp":
            raise ValueError(
                "LoRA currently requires beta 0, lag 0 and FSDP base weights; other combinations are not validated"
            )
        args.freeze_embeddings = True
    elif args.lora_alpha is not None or args.lora_targets is not None:
        raise ValueError("--lora-alpha and --lora-targets require --lora-rank")
    if args.dataset == "math" and args.eval_every and args.eval_split != "train-dev":
        raise ValueError(
            "periodic MATH evaluation uses train-dev; MATH500 is reserved for the separate final evaluation"
        )
    if min(args.prompt_batch_size, args.max_prompt_len, args.max_new_tokens, args.max_steps, args.reward_workers) <= 0:
        raise ValueError("batch sizes, lengths, steps and reward workers must be positive")
    if args.group_size < 2 or args.beta < 0 or args.save_every < 0 or args.eval_every < 0:
        raise ValueError("requires G >= 2, beta >= 0 and save-every/eval-every >= 0")
    validate_update_schedule(
        updates_per_rollout=args.updates_per_rollout,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )
    if args.save_at_steps is not None:
        if args.save_every or len(set(args.save_at_steps)) != len(args.save_at_steps):
            raise ValueError("--save-at-steps cannot be combined with --save-every or repeat a step")
        if any(step <= 0 or step > args.max_steps or step % args.updates_per_rollout for step in args.save_at_steps):
            raise ValueError(
                "--save-at-steps must be positive update steps at complete rollout boundaries within the budget"
            )
    if args.sampler_is_cap is not None and (not np.isfinite(args.sampler_is_cap) or args.sampler_is_cap <= 0):
        raise ValueError("--sampler-is-cap must be a positive finite number")
    if args.kl_clamp_value is not None and (
        not np.isfinite(args.kl_clamp_value)
        or args.kl_clamp_value <= 0
        or args.kl_clamp_value > float(np.finfo(np.float32).max)
        or not args.beta
    ):
        raise ValueError("--kl-clamp-value must be a positive finite FP32 value and requires beta > 0")
    if args.dynamic_sampling is None:
        args.dynamic_sampling = args.algorithm == "dapo"
    if args.filter_truncated_loss is None:
        args.filter_truncated_loss = args.algorithm == "dapo"
    if args.overlong_buffer_length is None:
        args.overlong_buffer_length = min(128, args.max_new_tokens) if args.algorithm == "dapo" else 0
    if args.overlong_penalty is None:
        args.overlong_penalty = 1.0 if args.algorithm == "dapo" else 0.0
    if args.max_sampling_attempts <= 0 or not 0 <= args.overlong_buffer_length <= args.max_new_tokens:
        raise ValueError("sampling attempts must be positive and the overlong buffer must be within [0, N]")
    if (
        not np.isfinite(args.overlong_penalty)
        or args.overlong_penalty < 0
        or (args.overlong_penalty and not args.overlong_buffer_length)
    ):
        raise ValueError(
            "the overlong penalty must be finite and non-negative, and needs a positive buffer when enabled"
        )
    if args.algorithm == "rloo" and (
        args.updates_per_rollout != 1
        or args.beta
        or args.sampler_is_cap is not None
        or args.rollout_lag_updates
        or args.dynamic_sampling
        or args.filter_truncated_loss
    ):
        raise ValueError("RLOO currently requires mu 1, beta 0 and no IS, lag or dynamic filtering")
    objective = objective_for_algorithm(
        "dapo" if args.algorithm == "dapo-loss" else args.algorithm, generation_budget=args.max_new_tokens
    )
    if args.loss_aggregation is not None:
        objective = objective._replace(agg_mode=args.loss_aggregation)
    if args.token_scale is not None and (not np.isfinite(args.token_scale) or args.token_scale <= 0):
        raise ValueError("--token-scale must be a positive finite number")
    if objective.agg_mode == "sequence-mean-token-scale":
        objective = objective._replace(token_scale=args.token_scale or float(args.max_new_tokens))
    elif args.token_scale is not None:
        raise ValueError("--token-scale applies only to sequence-mean-token-scale aggregation")
    else:
        objective = objective._replace(token_scale=None)
    if args.algorithm == "rloo" and objective.agg_mode != "seq-mean-token-sum":
        raise ValueError(
            "the RLOO baseline sums tokens per trajectory; other length weightings would be a different objective"
        )
    objective_options = {key: value for key, value in objective._asdict().items() if key != "advantage_estimator"}
    needs_old_logps = args.updates_per_rollout > 1 or args.sampler_is_cap is not None
    if (args.save_every or args.save_at_steps) and args.checkpoint_dir is None:
        raise ValueError("saving requires --checkpoint-dir with room for the full FP32 master/Adam state")
    if args.model_path is None:
        snapshots = sorted(Path.home().glob(".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/*"))
        if len(snapshots) != 1:
            raise ValueError("--model-path is required (no unique E2B-it snapshot was found)")
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
        raise ValueError("fewer training questions than one prompt batch remain after holding out dev")
    if args.eval_split == "train-dev" and (not args.dev_size or args.eval_size > args.dev_size):
        raise ValueError("train-dev evaluation requires dev-size > 0 and eval-size <= dev-size")
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
        raise ValueError("training device IDs must be non-empty, unique and present")
    if args.rollout_backend in ("inference", "inference-distributed"):
        inference_ids = args.inference_device_ids
        if (
            not inference_ids
            or len(set(inference_ids)) != len(inference_ids)
            or set(inference_ids) - devices_by_id.keys()
            or set(inference_ids) & set(training_ids)
        ):
            raise ValueError("inference devices must be present, unique and separate from the training mesh")
    elif process_inference and sorted(devices_by_id) != [0, 1]:
        raise ValueError("separate-process training must see exactly two local TPU devices")
    mesh = make_mesh([devices_by_id[index] for index in training_ids])
    if args.logprob_backend == "pallas" and (
        mesh.size != 4 or any("TPU v4" not in device.device_kind for device in mesh.devices.flat)
    ):
        raise ValueError("Pallas logprob requires four TPU v4 chips in the training mesh")
    device_allocation = iteration_device_allocation(args.rollout_backend, training_ids, args.inference_device_ids)
    rows = args.prompt_batch_size * args.group_size
    if rows % mesh.size:
        raise ValueError(f"generated rows ({rows}) must be divisible by the device count ({mesh.size})")
    if args.microbatch_size is not None and (
        args.microbatch_size <= 0 or rows % args.microbatch_size or args.microbatch_size % mesh.size
    ):
        raise ValueError("microbatch-size must divide the generated rows and be divisible by the device count")
    eval_batches = []
    eval_fingerprint = None
    if args.eval_every:
        if args.eval_batch_size <= 0 or args.eval_batch_size % mesh.size or args.eval_max_new_tokens <= 0:
            raise ValueError(
                "evaluation lengths must be positive and eval-batch-size must be divisible by the device count"
            )
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
    if args.rollout_layout == "tp" and (lora_config is not None or args.rollout_backend != "jax"):
        raise ValueError("TP rollout 目前只接入原生全参数同步训练")
    if args.rollout_layout == "tp" and args.rollout_lag_updates:
        raise ValueError("本次 TP 入口要求当前策略同步生成，暂不接入滞后策略")
    if args.rollout_layout != "tp" and args.tp_allreduce not in ("auto", "xla"):
        raise ValueError("tp-allreduce 需要 rollout-layout=tp")
    saved_rollout_config = (
        None if args.resume is None else read_checkpoint_metadata(args.resume)["metadata"].get("run_config")
    )
    args.tp_allreduce = resolve_allreduce(
        args.tp_allreduce,
        layout=args.rollout_layout,
        tp=mesh.size,
        rows=(args.prompt_batch_size * args.group_size, *(() if not args.eval_every else (args.eval_batch_size,))),
        saved=saved_rollout_config,
    )
    # Training samples from the full policy; proposal corrections for temperature or top-k are left to later work.
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
        "kl_estimator": "k3-stable-series-v2" if args.beta else "none",
        "kl_clamp_value": args.kl_clamp_value,
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
    run_config.update(logprob_run_metadata(args.logprob_backend))
    if args.training_device_ids is not None:
        run_config["training_device_ids"] = training_ids
    if args.rollout_backend != "jax":
        run_config.update(
            {
                "rollout_backend": "tpu-inference-shared-runtime-v1"
                if distributed_inference
                else "tpu-inference-separate-process-v1"
                if process_inference
                else "tpu-inference-same-process-v1",
                "inference_device_ids": args.inference_device_ids,
                "inference_precision": "default",
                "inference_kernel_route": "batched",
                "inference_memory_fraction": args.inference_memory_fraction,
                "inference_source_read_mode": args.inference_source_read_mode,
                "trim_training_host_allocator_after_transfer": (args.trim_training_host_allocator_after_transfer),
                "trim_inference_host_allocator_after_update": (args.trim_inference_host_allocator_after_update),
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
    # Chat keeps the original configuration; the plain template protocol is part of the checkpoint identity check.
    if distributed_inference:
        run_config["inference_weight_transport"] = "gemma4-shared-runtime-ici-verified-v1"
        run_config["inference_runtime_versions"] = {
            name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "libtpu")
        }
        for name in (
            "distributed_ici.py",
            "inference_shared.py",
            "inference_local_cpu.py",
            "inference_wire.py",
            "inference_remote.py",
        ):
            run_config["inference_contract_sha256"][name] = hashlib.sha256(
                (Path(__file__).resolve().parents[1] / "gemma4_posttrain_jax" / name).read_bytes()
            ).hexdigest()
    elif args.inference_weight_transport != "host":
        run_config["inference_weight_transport"] = "ici-collective-all-replicas-bitwise-v2"
        for name in ("inference_device.py", "ici_transfer.py"):
            run_config["inference_contract_sha256"][name] = hashlib.sha256(
                (Path(__file__).resolve().parents[1] / "gemma4_posttrain_jax" / name).read_bytes()
            ).hexdigest()
    if args.rollout_layout != "replicated":
        run_config["rollout_layout"] = args.rollout_layout
    if args.rollout_layout == "tp" and args.tp_allreduce != "xla":
        run_config["tp_allreduce"] = args.tp_allreduce
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
    if resume is not None:
        validate_resume_config(resume.get("run_config"), run_config)
    source_root = Path(__file__).resolve().parents[1]
    revision, diff = source_git_state(source_root)
    (args.output_dir / "source.diff").write_text(diff or "")
    write_json(
        args.output_dir / "meta.json",
        {
            "run_config": run_config,
            "timing_protocol": TIMING_PROTOCOL,
            "device_allocation": device_allocation,
            # Uncommitted changes are written to source.diff as is; both fields are null outside a git checkout.
            "git_commit": revision,
            "uncommitted_changes": None if diff is None else bool(diff),
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
    # The reference model always comes from the initial weights and is never updated; it is not loaded when beta=0.
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
        # fold_in uses a fixed namespace separate from the sampling seed/cursor stream;
        # resume reads the full unscaled A/B and Adam state.
        adapters = init_lora_params(params, lora_config, jax.random.fold_in(jax.random.PRNGKey(args.seed), 0x4C4F5241))
        adapters = jax.device_put(adapters, NamedSharding(mesh, P()))
        optimizer, trainable = make_lora_optimizer(args.learning_rate), None
        state = replicate_scalars(init_train_state(adapters, optimizer), mesh)
    del params
    state_shardings = tree_shardings(state)
    behavior_snapshot = None
    if args.resume:
        # Keep only a shape template and free the initial state before placing the restored one,
        # so two copies of master/Adam never share HBM.
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
        raise ValueError("checkpoint has no valid candidate-batch data cursor")
    if not args.dynamic_sampling and data_cursor != start_step // args.updates_per_rollout:
        raise ValueError("the data cursor does not fall on a complete rollout boundary")
    if start_step >= args.max_steps:
        raise ValueError(f"checkpoint step {start_step} has already reached max-steps {args.max_steps}")
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
            raise ValueError(
                f"checkpoint needs at least {(state_bytes + behavior_bytes) / 2**30:.2f} GiB plus 1 GiB of headroom"
            )
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
        if frozen_base is None and args.rollout_layout != "tp"
        else parameter_shardings
    )
    tp_mesh = tp_specs = generate_compiler_options = None
    if args.rollout_layout == "tp":
        from gemma4_posttrain_jax.tp_rollout import (
            attention_rows,
            check_tp_config,
            make_tp_mesh,
            tp_param_specs,
            tp_shardings,
        )
        from gemma4_posttrain_jax.tp_rollout import generate_compiler_options as tp_generate_options

        check_tp_config(config, mesh.size)
        tp_mesh = make_tp_mesh(1, mesh.size, list(mesh.devices.flat))
        tp_specs = tp_param_specs(state.params_f32, attention_rows=attention_rows(args.tp_allreduce))
        rollout_shardings = tp_shardings(tp_mesh, tp_specs)
        generate_compiler_options = tp_generate_options(args.tp_allreduce)
    batch_sharding = NamedSharding(mesh, batch_spec(2))
    sample_sharding = batch_sharding if tp_mesh is None else NamedSharding(tp_mesh, P())

    def shard_sample_inputs(values: tuple[np.ndarray, ...]) -> tuple[Any, ...]:
        if tp_mesh is None:
            return tuple(shard_batch(values, mesh))
        return tuple(jax.device_put(np.asarray(value), sample_sharding) for value in values)

    sampler_config = SamplerConfig(
        args.max_prompt_len, args.max_new_tokens, eos_ids=load_hf_eos_token_ids(str(args.model_path), config)
    )

    def sample_policy(current: Any, ids: Any, mask: Any, key: Any, sampler: SamplerConfig) -> RolloutBatch:
        if tp_mesh is not None:
            from gemma4_posttrain_jax.tp_rollout import tp_generate

            return tp_generate(
                current,
                config,
                ids,
                mask,
                sampler_config=sampler,
                key=key,
                mesh=tp_mesh,
                param_specs=tp_specs,
                allreduce=args.tp_allreduce,
            )
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

    def rollout_cast(current: Any) -> Any:
        if lora_config is not None:
            return current
        if tp_mesh is not None:
            return jax.tree.map(lambda value: value.astype(jnp.bfloat16), current)
        return reshard_for_rollout(current, config, mesh, layout=args.rollout_layout)

    reshard_fn = jax.jit(
        rollout_cast,
        in_shardings=(parameter_shardings,),
        out_shardings=rollout_shardings,
    )
    generate_fn = jax.jit(
        lambda current, ids, mask, key: sample_policy(current, ids, mask, key, sampler_config),
        in_shardings=(rollout_shardings, sample_sharding, sample_sharding, None),
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
            logprob_backend=args.logprob_backend,
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
            logprob_backend=args.logprob_backend,
            beta=args.beta,
            kl_clamp_value=args.kl_clamp_value,
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
    iteration_timing_records: list[dict[str, Any]] = []
    base_key = jax.random.PRNGKey(args.seed)
    training_finished = False
    training_session_start = time.perf_counter()
    tracker_init_s = initial_eval_s = cleanup_s = 0.0
    with (
        ExitStack() as stack,
        ProcessPoolExecutor(max_workers=args.reward_workers, mp_context=multiprocessing.get_context("spawn")) as pool,
        (args.output_dir / "metrics.csv").open("w", newline="") as csv_file,
        (args.output_dir / "rollouts.jsonl").open("w") as samples_file,
        (args.output_dir / "iteration_timing.jsonl").open("w") as iteration_timing_file,
    ):

        def record_final_state(_exc_type: Any, primary: BaseException | None, _traceback: Any) -> None:
            if not training_finished or not args.audit_final_state:
                return
            from gemma4_posttrain_jax.state_audit import summarize_train_state

            try:
                final_audit = summarize_train_state(state)
                write_json(args.output_dir / "final_state_audit.json", final_audit)
                if not final_audit["all_finite"]:
                    raise RuntimeError(
                        "the final training state contains non-finite values; the check result was saved"
                    )
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note(f"the final training state check also failed: {type(error).__name__}: {error}")

        # Registered first so it runs last: the engine is closed before the full readback,
        # so the two never hold memory at the same time.
        # ExitStack still runs this check when an earlier cleanup raises, and keeps the original exception.
        stack.push(record_final_state)
        inference_rollout = None
        inference_samples = None
        if args.rollout_backend != "jax":
            from gemma4_posttrain_jax.inference_remote import RemoteEngineRuntime
            from gemma4_posttrain_jax.inference_rollout import InferenceRollout
            from gemma4_posttrain_jax.inference_runtime import EngineConfig, EngineRuntime
            from gemma4_posttrain_jax.inference_shared import SharedEngineRuntime

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
                weight_sync_transport=args.inference_weight_transport,
                source_read_mode=args.inference_source_read_mode,
                trim_training_host_allocator_after_transfer=(args.trim_training_host_allocator_after_transfer),
                trim_host_allocator_after_update=args.trim_inference_host_allocator_after_update,
            )
            runtime: EngineRuntime | RemoteEngineRuntime
            if distributed_inference:
                runtime = SharedEngineRuntime(engine_config, path=Path(os.environ["GEMMA4_SHARED_ICI_SOCKET"]))
            elif process_inference:
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
                        primary.add_note("closing the separate engine also failed; see inference_close.json")
                    else:
                        raise RuntimeError("closing the separate engine did not complete; see inference_close.json")

            stack.callback(close_inference)
            write_json(
                args.output_dir / "inference_initialization.json",
                {
                    "load_and_initialize_s": time.perf_counter() - engine_init_started,
                    "first_generate_includes_jit": True,
                    "metadata": runtime.metadata,
                    "timing_note": (
                        "The first engine generate includes compilation and is not steady state; "
                        "initialization and weight transfer are listed separately."
                    ),
                },
            )
            inference_rollout = InferenceRollout(runtime, config, sampler_config=sampler_config, mesh=mesh)
            inference_samples = stack.enter_context((args.output_dir / "inference.jsonl").open("w"))
        tracker_init_start = time.perf_counter()
        tracker = init_tracker(
            enabled=args.wandb,
            project=args.wandb_project,
            run_name=args.wandb_run_name,
            tags=args.wandb_tags,
            config={"logging_mode": "live", "source_metadata": json.loads((args.output_dir / "meta.json").read_text())},
            group=args.wandb_group,
            job_type=args.wandb_job_type,
        )
        tracker_init_s = time.perf_counter() - tracker_init_start
        stack.callback(tracker.finish)
        writer = None

        def run_eval(global_step: int) -> tuple[float, dict[str, Any]]:
            nonlocal eval_executable
            assert reshard_executable is not None
            start = time.perf_counter()
            eval_params, _ = timed_call(reshard_executable, policy_params())
            if eval_executable is None:
                first_batch = shard_sample_inputs((eval_batches[0].prompt_ids, eval_batches[0].prompt_mask))
                eval_executable = compile_call(
                    "eval",
                    make_eval_sampler(
                        config,
                        eval_sampler_config,
                        mesh,
                        rollout_layout=args.rollout_layout,
                        param_tree=state.params_f32,
                        tp_allreduce=args.tp_allreduce,
                    )
                    if lora_config is None
                    else jax.jit(
                        lambda current, ids, mask, key: sample_policy(current, ids, mask, key, eval_sampler_config),
                        in_shardings=(rollout_shardings, sample_sharding, sample_sharding, None),
                    ),
                    (eval_params, *first_batch, jax.random.PRNGKey(args.eval_seed)),
                    args.output_dir,
                    args.save_hlo,
                    generate_compiler_options,
                )
            summary = evaluate_batches(
                lambda current, prompt, valid, key: eval_executable(
                    current, *shard_sample_inputs((prompt, valid)), key
                ),
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
            if cycle_observer is not None:
                cycle_observer(rollout_step + 1, "start")
            step = rollout_step * args.updates_per_rollout
            step_start = time.perf_counter()
            behavior_step = step if behavior_snapshot is None else int(behavior_snapshot.step)
            if behavior_step != max(0, step - args.rollout_lag_updates):
                raise RuntimeError("rollout weights did not keep the declared one-update lag")
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
            prompt_prepare_s = rollout_s = rollout_device_get_s = 0.0
            text_decode_s = reward_s = rollout_record_s = rollout_audit_s = 0.0
            if args.audit_execution_layout and step == start_step:
                # Both the training state and the rollout weights are in place before the first rollout. Only shard
                # attributes are read, and the time counts toward rollout_audit_s.
                layout_record_start = time.perf_counter()
                write_json(
                    args.output_dir / "execution_layout.json",
                    execution_layout_record(
                        state,
                        rollout_params,
                        mesh,
                        rollout_backend=args.rollout_backend,
                        rollout_layout=args.rollout_layout,
                        logprob_backend=args.logprob_backend,
                    ),
                )
                rollout_audit_s += time.perf_counter() - layout_record_start
            while accepted_groups < args.prompt_batch_size:
                if attempts == args.max_sampling_attempts:
                    raise RuntimeError(
                        f"only {accepted_groups}/{args.prompt_batch_size} valid groups after {attempts} attempts; "
                        "parameters were not updated and the generated batches were saved"
                    )
                prompt_prepare_start = time.perf_counter()
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
                ids, mask = shard_sample_inputs((prompts.prompt_ids, prompts.prompt_mask))
                key = jax.random.fold_in(base_key, data_cursor)
                data_cursor += 1
                attempts += 1
                prompt_prepare_s += time.perf_counter() - prompt_prepare_start
                if inference_rollout is not None:
                    rollout, elapsed = timed_call(partial(inference_rollout.generate, key=key), ids, mask)
                    if inference_rollout.last_proposal_logps is None:
                        raise RuntimeError(
                            "full-policy sampling returned no proposal probabilities; the training batch cannot be used"
                        )
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
                            "rollout",
                            generate_fn,
                            (rollout_params, ids, mask, key),
                            args.output_dir,
                            args.save_hlo,
                            generate_compiler_options,
                        )
                    rollout, elapsed = timed_call(generate_executable, rollout_params, ids, mask, key)
                rollout_s += elapsed
                transfer_start = time.perf_counter()
                completion_ids, completion_mask, lengths, behavior_logps = jax.device_get(
                    (rollout.completion_ids, rollout.completion_mask, rollout.lengths, rollout.rollout_logps)
                )
                rollout_device_get_s += time.perf_counter() - transfer_start
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
                    # Native sampling other than T=1 over the full vocabulary returns only raw policy probabilities;
                    # mark the proposal as missing.
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
                decode_start = time.perf_counter()
                generated_tokens += int(lengths.sum())
                completions = tokenizer.batch_decode(
                    [tokens[:length] for tokens, length in zip(completion_ids, lengths, strict=True)],
                    skip_special_tokens=True,
                )
                text_decode_s += time.perf_counter() - decode_start
                reward_start = time.perf_counter()
                reward = score_completions(
                    completions, prompts.golds, completion_lengths=lengths, config=reward_config, executor=pool
                )
                reward_s += time.perf_counter() - reward_start
                record_start = time.perf_counter()
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
                rollout_record_s += time.perf_counter() - record_start
            del rollout_params
            batch_prepare_start = time.perf_counter()
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
                    raise RuntimeError(
                        "the loss mask of the whole batch is empty; stopping before the update (rollouts are kept)"
                    )
            token_arguments = (ids, mask, rollout.completion_ids, rollout.completion_mask)
            reference_logps = None
            batch_prepare_s = time.perf_counter() - batch_prepare_start
            reference_transfer_s = reference_logps_s = startup_logps_s = 0.0
            if reference_host is not None:
                # The reference model and the replicated rollout workspace cannot both stay in v4 HBM;
                # its FSDP shards are placed only for this stage.
                reference, reference_transfer_s = timed_call(
                    lambda: shard_gemma4_text_params(reference_host, config, mesh)
                )
                if reference_executable is None:
                    reference_executable = compile_call(
                        "reference", logps_fn, (reference, *token_arguments), args.output_dir, args.save_hlo
                    )
                reference_logps, reference_logps_s = timed_call(reference_executable, reference, *token_arguments)
                del reference
            reference_s = reference_transfer_s + reference_logps_s
            # This separate forward is only a first-batch sampler diagnostic; the old logprobs
            # for the mu updates come from the joint gradient forward.
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
                drift_gate = logprob_drift_gate(drift, start_step=start_step)
                write_json(args.output_dir / "startup_drift_gate.json", drift_gate)
                print(json.dumps({"startup_drift_gate": drift_gate}), flush=True)
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
                if drift_gate["action"] == "stop-before-update":
                    raise RuntimeError(
                        "the startup sampler/trainer drift check failed; inputs were saved and no update was made"
                    )
                if not np.isfinite(np.asarray(trainer_logps)[completion_mask]).all():
                    raise RuntimeError("trainer logprobs contain non-finite values")
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
                    UPDATE_COMPILER_OPTIONS if tp_mesh is not None and jax.default_backend() == "tpu" else None,
                )
            for iteration in range(args.updates_per_rollout):
                step = rollout_step * args.updates_per_rollout + iteration
                iteration_start = step_start if iteration == 0 else time.perf_counter()
                behavior_snapshot_s = 0.0
                if capture_executable is not None and iteration == args.updates_per_rollout - 1:
                    # Capture before the last Adam update so the next rollout lags by exactly one update,
                    # not by the whole mu round.
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
                        raise RuntimeError(
                            "the observed parameter update is non-finite or the Adam step did not advance"
                        )
                    update_audit_s += time.perf_counter() - audit_start

                if needs_old_logps and iteration == 0:
                    if metrics.policy_logps is None:
                        raise RuntimeError("the joint graph did not return the old logprobs for this round")
                    old_for_update = shard_batch(metrics.policy_logps, mesh)
                    # Later updates in the round read only this first-forward array;
                    # later current logprobs must not overwrite it.
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
                if args.audit_update_state and needs_old_logps:
                    audit_start = time.perf_counter()
                    np.savez(
                        args.output_dir / f"update_probabilities_{step + 1:08d}.npz",
                        policy_logps=np.asarray(metrics.policy_logps),
                        old_logps=np.asarray(old_for_update),
                        reference_logps=np.empty((0,), np.float32)
                        if reference_logps is None
                        else np.asarray(reference_logps),
                        behavior_logps=np.asarray(rollout.rollout_logps),
                        completion_mask=completion_mask,
                        loss_mask=completion_mask if loss_mask is None else np.asarray(loss_mask),
                        advantages=np.asarray(advantages),
                        sampler_is_weights=np.empty((0,), np.float32) if is_weights is None else np.asarray(is_weights),
                        use_current_policy_as_old=np.asarray(iteration == 0),
                    )
                    update_audit_s += time.perf_counter() - audit_start
                if not all(np.isfinite(value) for value in values.values()):
                    raise RuntimeError(f"step {step + 1} loss/gradient/metrics contain non-finite values: {values}")
                if (args.updates_per_rollout == 1 or iteration == 0) and (values["ratio_min"], values["ratio_max"]) != (
                    1.0,
                    1.0,
                ):
                    raise RuntimeError("the old/current ratio of the first joint forward is not 1")
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
                    "prompt_prepare_s": prompt_prepare_s if iteration == 0 else 0.0,
                    "rollout_s": rollout_s if iteration == 0 else 0.0,
                    "rollout_device_get_s": rollout_device_get_s if iteration == 0 else 0.0,
                    "rollout_audit_s": rollout_audit_s if iteration == 0 else 0.0,
                    "text_decode_s": text_decode_s if iteration == 0 else 0.0,
                    "reward_s": reward_s if iteration == 0 else 0.0,
                    "rollout_record_s": rollout_record_s if iteration == 0 else 0.0,
                    "batch_prepare_s": batch_prepare_s if iteration == 0 else 0.0,
                    "startup_logps_s": startup_logps_s if iteration == 0 else 0.0,
                    "reference_s": reference_s if iteration == 0 else 0.0,
                    "reference_transfer_s": reference_transfer_s if iteration == 0 else 0.0,
                    "reference_logps_s": reference_logps_s if iteration == 0 else 0.0,
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
                    "iteration_before_metrics_log_s": 0.0,
                    "unattributed_s": 0.0,
                }
                if lora_config is not None:
                    # The 6*2.3B estimate only applies to the full-parameter E2B graph, not to LoRA with a frozen base.
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
                row["iteration_before_metrics_log_s"] = time.perf_counter() - iteration_start
                metrics_prepare_start = time.perf_counter()
                accounted_fields = (
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
                row["unattributed_s"] = row["iteration_before_metrics_log_s"] - sum(
                    float(row.get(field, 0.0)) for field in accounted_fields
                )
                metrics_prepare_s = time.perf_counter() - metrics_prepare_start
                metrics_csv_start = time.perf_counter()
                if writer is None:
                    writer = csv.DictWriter(csv_file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                csv_file.flush()
                metrics_csv_s = time.perf_counter() - metrics_csv_start
                records.append(row)
                metrics_stdout_start = time.perf_counter()
                print(json.dumps(row, allow_nan=False), flush=True)
                metrics_stdout_s = time.perf_counter() - metrics_stdout_start
                tracker_prepare_start = time.perf_counter()
                tracked = grpo_tracking_metrics(row, eval_summary)
                if step == start_step:
                    tracked.update(compile_tracking_metrics(args.output_dir))
                tracker_prepare_s = time.perf_counter() - tracker_prepare_start
                tracker_log_start = time.perf_counter()
                tracker.log(tracked, step=step + 1)
                tracker_log_s = time.perf_counter() - tracker_log_start
                iteration_after_metrics_log_s = time.perf_counter() - iteration_start
                measured_logging_s = (
                    metrics_prepare_s + metrics_csv_s + metrics_stdout_s + tracker_prepare_s + tracker_log_s
                )
                metrics_logging_overhead_s = max(
                    0.0,
                    iteration_after_metrics_log_s - row["iteration_before_metrics_log_s"] - measured_logging_s,
                )
                metrics_logging_s = measured_logging_s + metrics_logging_overhead_s
                known_device_idle_s = (
                    float(row["text_decode_s"])
                    + float(row["reward_s"])
                    + float(row["rollout_record_s"])
                    + metrics_logging_s
                )
                timing_record = {
                    "step": step + 1,
                    "first_call": int(step == start_step),
                    **device_allocation,
                    "iteration_before_metrics_log_s": row["iteration_before_metrics_log_s"],
                    "metrics_prepare_s": metrics_prepare_s,
                    "metrics_csv_s": metrics_csv_s,
                    "metrics_stdout_s": metrics_stdout_s,
                    "tracker_prepare_s": tracker_prepare_s,
                    "tracker_log_s": tracker_log_s,
                    "metrics_logging_overhead_s": metrics_logging_overhead_s,
                    "metrics_logging_s": metrics_logging_s,
                    "iteration_after_metrics_log_s": iteration_after_metrics_log_s,
                    "known_device_idle_s": known_device_idle_s,
                    "allocated_device_seconds": iteration_after_metrics_log_s * device_allocation["allocated_devices"],
                    "known_idle_device_seconds": known_device_idle_s * device_allocation["allocated_devices"],
                }
                iteration_timing_file.write(json.dumps(timing_record, allow_nan=False) + "\n")
                iteration_timing_file.flush()
                iteration_timing_records.append(timing_record)
            if cycle_observer is not None:
                cycle_observer(rollout_step + 1, "end")
        training_finished = True
        write_json(
            args.output_dir / "training_end.json",
            {
                "event": "training_finished",
                "cleanup_included": False,
                "start_step": start_step,
                "final_step": int(state.step),
                "final_data_cursor": data_cursor,
                "generated_rows": sum(row["generated_rows"] for row in records),
                "generated_tokens": sum(row["generated_tokens"] for row in records),
            },
        )
        cleanup_start = time.perf_counter()
    cleanup_s = time.perf_counter() - cleanup_start
    storage = checkpoint_storage(args.checkpoint_dir)
    write_json(
        args.output_dir / "summary.json",
        {
            "start_step": start_step,
            "final_step": int(state.step),
            "final_data_cursor": data_cursor,
            "generated_rows": sum(row["generated_rows"] for row in records),
            "generated_tokens": sum(row["generated_tokens"] for row in records),
            "peak_bytes": device_peak_bytes(),
            "timing_protocol": TIMING_PROTOCOL,
            "device_allocation": device_allocation,
            "training_session_s": time.perf_counter() - training_session_start,
            "tracker_init_s": tracker_init_s,
            "initial_eval_s": initial_eval_s,
            "cleanup_s": cleanup_s,
            "iteration_timing_record_write_excluded": True,
            "steady_steps": len(records[1:]),
            "steady_mean": {
                field: float(np.mean([row[field] for row in records[1:]]))
                for field in (
                    "reshard_s",
                    "prompt_prepare_s",
                    "rollout_s",
                    "rollout_device_get_s",
                    "text_decode_s",
                    "reward_s",
                    "rollout_record_s",
                    "batch_prepare_s",
                    "old_logps_s",
                    "startup_logps_s",
                    "reference_transfer_s",
                    "reference_logps_s",
                    "reference_s",
                    "update_s",
                    "step_s",
                    "checkpoint_s",
                    "eval_s",
                    "iteration_before_metrics_log_s",
                    "unattributed_s",
                )
            }
            if len(records) > 1
            else {},
            "steady_iteration_after_metrics_log_s": float(
                np.mean([row["iteration_after_metrics_log_s"] for row in iteration_timing_records[1:]])
            )
            if len(iteration_timing_records) > 1
            else None,
            "checkpoint_storage": storage,
            "checkpoint_persistent": False
            if storage is None
            else None
            if storage["memory_backed"] is None
            else not storage["memory_backed"],
        },
    )


if __name__ == "__main__":
    main()
