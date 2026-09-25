#!/usr/bin/env python3
"""在TPU v4上进行Gemma 4 FSDP SFT，支持梯度累积与完整状态恢复。"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import math
import os
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

from gemma4_posttrain_jax.bench import device_peak_bytes
from gemma4_posttrain_jax.checkpoint import load_train_state, read_checkpoint_metadata, save_train_state
from gemma4_posttrain_jax.data import GSM8KBatchStream, collate_sft, encode_sft_example, load_gsm8k
from gemma4_posttrain_jax.diagnostics import optional_package_version, source_git_state
from gemma4_posttrain_jax.logprob_protocol import logprob_run_metadata
from gemma4_posttrain_jax.losses import init_train_state, make_optimizer, train_step
from gemma4_posttrain_jax.sharding import (
    batch_spec,
    make_mesh,
    replicate_scalars,
    shard_batch,
    shard_gemma4_text_params,
    tree_shardings,
)
from gemma4_posttrain_jax.tracking import init_tracker
from gemma4_posttrain_jax.weights import load_hf_params

DEFAULT_SNAPSHOT_GLOB = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/*/")
OVERFIT_QUESTION = "A box has 3 rows of 4 red balls and 2 additional blue balls. How many balls are in the box?"
OVERFIT_ANSWER = "There are 3 times 4 = 12 red balls. Adding 2 blue balls gives 14 balls. The answer is 14."
E2B_PARAMETER_ESTIMATE = 2.3e9
TPU_V4_8_BF16_PEAK_FLOPS = 1.1e15
RESUME_CONFIG_FIELDS = (
    "mode",
    "model_path",
    "batch_size",
    "sequence_length",
    "learning_rate",
    "weight_decay",
    "max_grad_norm",
    "gradient_accumulation_steps",
    "freeze_embeddings",
    "compute_dtype",
    "vocab_chunk",
    "sequence_chunk",
    "vocab_parallel",
    "logprob_backend",
    "logprob_protocol",
    "remat_layers",
    "mesh_shape",
    "mesh_axis_names",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="HF snapshot directory; defaults to the cached Gemma 4 E2B-it")
    parser.add_argument("--overfit-one-batch", action="store_true", help="repeat one fixed example instead of GSM8K")
    parser.add_argument("--max-steps", type=int, default=None, help="defaults to 100 for overfit or 200 for GSM8K")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--freeze-embeddings", action="store_true", help="full-core FT: freeze token and PLE lookup tables"
    )
    parser.add_argument("--compute-dtype", choices=("bf16", "f32"), default="bf16")
    parser.add_argument(
        "--logprob-backend",
        choices=("jax", "pallas"),
        default="jax",
        help="logprob 后端；pallas 为四芯片 TPU v4/BF16 实验选项，默认使用 jax",
    )
    parser.add_argument("--vocab-chunk", type=int, default=8192)
    parser.add_argument("--sequence-chunk", type=int, default=256)
    parser.add_argument(
        "--vocab-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep the LM-head embedding vocabulary-sharded (default: enabled)",
    )
    parser.add_argument(
        "--remat",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="rematerialize decoder layers during backward (default: disabled)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--log-csv", default="outputs/logs/sft.csv")
    parser.add_argument("--wandb", action="store_true", help="enable optional host-side Weights & Biases logging")
    parser.add_argument("--wandb-project", default="gemma4_posttrain_jax")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-tags", nargs="*", default=())
    parser.add_argument("--save-every", type=int, default=0, help="save the full train state every N steps; 0 disables")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints/sft")
    parser.add_argument("--resume", help="resume from an exact step_N checkpoint directory")
    parser.add_argument(
        "--require-overfit-target", action="store_true", help="fail unless final overfit loss is below 0.05"
    )
    parser.add_argument(
        "--require-single-executable",
        action="store_true",
        help="fail if the JIT train step creates more than one in-process executable cache entry",
    )
    args = parser.parse_args()
    if args.logprob_backend == "pallas" and (
        not args.vocab_parallel
        or args.compute_dtype != "bf16"
        or (args.vocab_chunk, args.sequence_chunk) != (8192, 256)
    ):
        parser.error("Pallas logprob requires BF16, vocabulary parallelism, vocab-chunk=8192 and sequence-chunk=256")
    return args


def resolve_model_path(value: str | None) -> str:
    if value:
        return value
    snapshots = sorted(glob.glob(DEFAULT_SNAPSHOT_GLOB))
    if not snapshots:
        raise FileNotFoundError("Gemma 4 E2B-it is not cached; pass --model-path")
    return snapshots[0]


def jit_cache_size(jitted: Any) -> int | None:
    """Return JAX's in-process executable cache size when this version exposes it."""

    getter = getattr(jitted, "_cache_size", None)
    return None if getter is None else int(getter())


def git_commit() -> str | None:
    """按脚本源码目录记录提交；独立快照不继承运行目录或父仓库身份。"""

    return source_git_state(Path(__file__).resolve().parents[1])[0]


def estimated_mfu(*, tokens: int, elapsed_s: float) -> float:
    """Approximate training MFU using 6*N FLOPs/token and TPU v4-8 BF16 peak."""

    return 6.0 * E2B_PARAMETER_ESTIMATE * tokens / elapsed_s / TPU_V4_8_BF16_PEAK_FLOPS


def validate_resume_config(saved: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Refuse a continuation whose model, optimizer, data shape, or JIT controls changed."""

    # 未记录后端的旧 checkpoint 使用原生 JAX。
    saved = {"logprob_backend": "jax", **saved}
    current = {"logprob_backend": "jax", **current}
    fields = RESUME_CONFIG_FIELDS + (("seed",) if current.get("mode") == "gsm8k" else ())
    differences = {
        field: {"saved": saved.get(field), "current": current.get(field)}
        for field in fields
        if saved.get(field) != current.get(field)
    }
    if differences:
        raise ValueError(f"resume config differs from checkpoint: {json.dumps(differences, sort_keys=True)}")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.sequence_length < 2:
        raise ValueError("batch-size must be positive and sequence-length at least two")
    steps = args.max_steps if args.max_steps is not None else (100 if args.overfit_one_batch else 200)
    if steps <= 0:
        raise ValueError("max-steps must be positive")
    if args.save_every < 0:
        raise ValueError("save-every must be non-negative")
    resume_metadata: Mapping[str, Any] | None = None
    if args.resume:
        resume_metadata = read_checkpoint_metadata(args.resume)["metadata"]
    model_path = resolve_model_path(args.model_path)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pad_token_id = int(tokenizer.pad_token_id or 0)
    data_stream: GSM8KBatchStream | None = None
    batches: Iterator[Any]
    if args.overfit_one_batch:
        encoded = encode_sft_example(tokenizer, OVERFIT_QUESTION, OVERFIT_ANSWER, args.sequence_length)
        fixed_batch = collate_sft([encoded], args.batch_size, args.sequence_length, pad_token_id)
        batches = itertools.repeat(fixed_batch)
        mode = "overfit-one-batch"
    else:
        data_stream = GSM8KBatchStream(
            load_gsm8k("train"),
            tokenizer,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            pad_token_id=pad_token_id,
            seed=args.seed,
        )
        batches = iter(data_stream)
        mode = "gsm8k"
    if resume_metadata is not None and data_stream is not None:
        saved_data_state = resume_metadata.get("data_state")
        if not isinstance(saved_data_state, Mapping):
            raise ValueError("GSM8K checkpoint metadata has no data_state")
        data_stream.load_state_dict(saved_data_state)

    mesh = make_mesh()
    if args.batch_size % mesh.size:
        raise ValueError(f"global batch {args.batch_size} must be divisible by mesh size {mesh.size}")
    compute_dtype = jnp.bfloat16 if args.compute_dtype == "bf16" else jnp.float32
    run_config = {
        "mode": mode,
        "model_path": model_path,
        "steps": steps,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "freeze_embeddings": args.freeze_embeddings,
        "compute_dtype": args.compute_dtype,
        "vocab_chunk": args.vocab_chunk,
        "sequence_chunk": args.sequence_chunk,
        "vocab_parallel": args.vocab_parallel,
        "logprob_backend": args.logprob_backend,
        "remat_layers": args.remat,
        "seed": args.seed,
        "shape_policy": "fixed padded [batch_size, sequence_length], drop remainder before reshuffle",
        "require_single_executable": args.require_single_executable,
        "checkpoint_dir": args.checkpoint_dir,
        "save_every": args.save_every,
        "resume": args.resume,
        "mfu_estimate": "6 * 2.3B parameters * padded tokens / TPU v4-8 BF16 peak (1.1 PFLOP/s)",
        "git_commit": git_commit(),
        "jax": jax.__version__,
        "libtpu": optional_package_version("libtpu"),
        "backend": jax.default_backend(),
        "mesh_shape": list(mesh.devices.shape),
        "mesh_axis_names": list(mesh.axis_names),
        "devices": [device.device_kind for device in jax.devices()],
    }
    run_config.update(logprob_run_metadata(args.logprob_backend))
    load_start = time.perf_counter()
    host_params, config = load_hf_params(model_path, dtype=jnp.float32)
    params = shard_gemma4_text_params(host_params, config, mesh)
    del host_params
    optimizer, trainable_mask = make_optimizer(
        params,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        freeze_embeddings=args.freeze_embeddings,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    state = replicate_scalars(init_train_state(params, optimizer), mesh)
    del params
    state_shardings = tree_shardings(state)
    if args.resume:
        assert resume_metadata is not None
        saved_config = resume_metadata.get("run_config")
        if not isinstance(saved_config, dict):
            raise ValueError("checkpoint metadata has no run_config")
        validate_resume_config(saved_config, run_config)
        state = load_train_state(args.resume, state, shardings=state_shardings)
    jax.block_until_ready(state)
    start_step = int(state.step)
    if start_step >= steps:
        raise ValueError(f"checkpoint is already at step {start_step}, not below requested max-steps {steps}")
    run_config["start_step"] = start_step
    run_config["data_state_start"] = None if data_stream is None else data_stream.state_dict()
    print(json.dumps(run_config, ensure_ascii=False), flush=True)
    print(f"load_shard_init_s={time.perf_counter() - load_start:.3f} peak_bytes={device_peak_bytes()}", flush=True)

    batch_sharding = NamedSharding(mesh, batch_spec(2))
    compiled_step = jax.jit(
        lambda current, token_ids, token_labels, token_mask: train_step(
            current,
            token_ids,
            token_labels,
            token_mask,
            config=config,
            optimizer=optimizer,
            trainable_mask=trainable_mask,
            compute_dtype=compute_dtype,
            vocab_chunk=args.vocab_chunk,
            sequence_chunk=args.sequence_chunk,
            remat_layers=args.remat,
            mesh=mesh if args.vocab_parallel else None,
            logprob_backend=args.logprob_backend,
        ),
        donate_argnums=(0,),
        in_shardings=(state_shardings, batch_sharding, batch_sharding, batch_sharding),
        out_shardings=(state_shardings, None),
    )
    if args.require_single_executable and jit_cache_size(compiled_step) is None:
        raise RuntimeError("this JAX version does not expose the JIT cache size required by the guard")
    tracker = init_tracker(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        tags=args.wandb_tags,
        config=run_config,
    )
    os.makedirs(os.path.dirname(args.log_csv) or ".", exist_ok=True)
    first_loss = final_loss = float("nan")
    tokens_per_step = args.batch_size * args.sequence_length
    try:
        with open(args.log_csv, "w", newline="") as log_file:
            writer = csv.DictWriter(
                log_file,
                fieldnames=(
                    "step",
                    "loss",
                    "grad_norm",
                    "step_s",
                    "train_tokens_per_s",
                    "mfu",
                    "peak_hbm_bytes",
                    "peak_hbm_gib",
                    "jit_cache_size",
                    "learning_rate",
                    "checkpoint_s",
                ),
            )
            writer.writeheader()
            for step_index in range(start_step, steps):
                batch = shard_batch(next(batches), mesh)
                start = time.perf_counter()
                state, metrics = compiled_step(state, *batch)
                jax.block_until_ready((state, metrics))
                elapsed = time.perf_counter() - start
                final_loss = float(metrics.loss)
                first_loss = final_loss if math.isnan(first_loss) else first_loss
                cache_size = jit_cache_size(compiled_step)
                peak_hbm_bytes = max(value or 0 for value in device_peak_bytes())
                row = {
                    "step": step_index + 1,
                    "loss": final_loss,
                    "grad_norm": float(metrics.grad_norm),
                    "step_s": elapsed,
                    "train_tokens_per_s": tokens_per_step / elapsed,
                    "mfu": estimated_mfu(tokens=tokens_per_step, elapsed_s=elapsed),
                    "peak_hbm_bytes": peak_hbm_bytes,
                    "peak_hbm_gib": peak_hbm_bytes / 2**30,
                    "jit_cache_size": cache_size,
                    "learning_rate": args.learning_rate,
                    "checkpoint_s": None,
                }
                if args.save_every and (step_index + 1) % args.save_every == 0:
                    checkpoint_path = Path(args.checkpoint_dir) / f"step_{step_index + 1:08d}"
                    checkpoint_start = time.perf_counter()
                    save_train_state(
                        state,
                        checkpoint_path,
                        metadata={
                            "completed_step": step_index + 1,
                            "run_config": run_config,
                            "data_state": None if data_stream is None else data_stream.state_dict(),
                        },
                    )
                    row["checkpoint_s"] = time.perf_counter() - checkpoint_start
                    print(
                        json.dumps(
                            {
                                "checkpoint": str(checkpoint_path),
                                "step": step_index + 1,
                                "save_s": row["checkpoint_s"],
                            }
                        ),
                        flush=True,
                    )
                writer.writerow(row)
                log_file.flush()
                tracker.log({key: value for key, value in row.items() if key != "step"}, step=step_index + 1)
                if step_index == 0 or (step_index + 1) % args.log_every == 0:
                    print(json.dumps(row), flush=True)
                if args.require_single_executable and cache_size != 1:
                    raise RuntimeError(
                        f"train-step JIT cache contains {cache_size} executables at step {step_index + 1}; "
                        "a shape, dtype, static value, or sharding constraint changed"
                    )
    finally:
        tracker.finish()

    summary = {
        "mode": mode,
        "steps": steps,
        "start_step": start_step,
        "initial_loss": first_loss,
        "final_loss": final_loss,
        "target_met": (not args.overfit_one_batch) or final_loss < 0.05,
        "peak_bytes": device_peak_bytes(),
        "jit_cache_size": jit_cache_size(compiled_step),
        "log_csv": args.log_csv,
    }
    print(json.dumps(summary), flush=True)
    if args.require_overfit_target and args.overfit_one_batch and final_loss >= 0.05:
        raise SystemExit(f"overfit target missed: final loss {final_loss:.6f} >= 0.05")


if __name__ == "__main__":
    main()
