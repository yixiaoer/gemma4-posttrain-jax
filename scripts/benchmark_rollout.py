#!/usr/bin/env python3
"""在独立进程中测量Gemma 4 E2B纯JAX批量rollout的吞吐和内存。"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.metadata
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from gemma4_posttrain_jax.data import format_prompt, load_gsm8k
from gemma4_posttrain_jax.sampler import RolloutBatch, SamplerConfig, generate
from gemma4_posttrain_jax.sharding import (
    batch_spec,
    make_mesh,
    named_shardings,
    param_specs_replicated,
    reshard_for_rollout,
    shard_batch,
    shard_gemma4_text_params,
    tree_shardings,
)
from gemma4_posttrain_jax.weights import load_hf_eos_token_ids, load_hf_params

DEFAULT_SNAPSHOT_GLOB = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/*/")
TPU_V4_8_BF16_PEAK_FLOPS = 1.1e15
TPU_V4_HBM_BANDWIDTH_BYTES_S = 1.2e12


class Iteration(NamedTuple):
    execute_ms: float
    decode_steps: int
    slot_tokens: int
    actual_tokens: int
    utilization: float
    slot_tokens_per_s: float
    actual_tokens_per_s: float
    amortized_step_ms: float
    estimated_model_tflops_s: float
    estimated_mfu: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=(32, 64, 128, 256))
    parser.add_argument("--new-token-counts", nargs="+", type=int, default=(256, 512))
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", default="outputs/benchmarks/rollout.json")
    parser.add_argument("--case-output-dir", default="outputs/benchmarks/rollout_cases")
    parser.add_argument("--require-pass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--case-batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--case-new-tokens", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def _git_state() -> dict[str, Any]:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.splitlines()
    return {"commit": revision or None, "status_short": status}


def _memory() -> list[dict[str, float | int | None]]:
    result: list[dict[str, float | int | None]] = []
    for device in jax.local_devices():
        stats = device.memory_stats() or {}
        used = stats.get("bytes_in_use")
        peak = stats.get("peak_bytes_in_use")
        result.append(
            {
                "device": device.id,
                "bytes_in_use": None if used is None else int(used),
                "gib_in_use": None if used is None else float(used / 2**30),
                "peak_bytes_in_use": None if peak is None else int(peak),
                "peak_gib": None if peak is None else float(peak / 2**30),
            }
        )
    return result


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _prompts(
    model_path: str, batch_size: int, max_prompt_len: int, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer has no pad token")
    dataset = load_gsm8k("train")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    indices = indices[:batch_size]
    encoded = [format_prompt(str(dataset[index]["question"]), tokenizer) for index in indices]
    lengths = [len(tokens) for tokens in encoded]
    if max(lengths) > max_prompt_len:
        raise ValueError(f"prompt length {max(lengths)} exceeds max_prompt_len={max_prompt_len}")
    ids = np.full((batch_size, max_prompt_len), int(tokenizer.pad_token_id), np.int32)
    mask = np.zeros((batch_size, max_prompt_len), np.bool_)
    for row, tokens in enumerate(encoded):
        ids[row, -len(tokens) :] = tokens
        mask[row, -len(tokens) :] = True
    return (
        ids,
        mask,
        {
            "indices": indices,
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            "prompt_length_min": min(lengths),
            "prompt_length_mean": statistics.fmean(lengths),
            "prompt_length_max": max(lengths),
            "sha256": _hash_arrays(ids, mask),
        },
    )


def _rollout_shardings(mesh: Mesh) -> RolloutBatch:
    batch_2d = NamedSharding(mesh, batch_spec(2))
    batch_1d = NamedSharding(mesh, batch_spec(1))
    constructor: Any = RolloutBatch
    return cast(
        RolloutBatch,
        constructor(batch_2d, batch_2d, batch_2d, batch_2d, batch_1d),
    )


def _memory_analysis(compiled: Any) -> dict[str, int | None]:
    analysis = compiled.memory_analysis()
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_temp_size_in_bytes",
        "generated_code_size_in_bytes",
    )
    return {
        name: None if analysis is None or getattr(analysis, name, None) is None else int(getattr(analysis, name))
        for name in names
    }


def _cost_analysis(compiled: Any) -> dict[str, float]:
    analysis = compiled.cost_analysis()
    if isinstance(analysis, list):
        analysis = analysis[0] if analysis else {}
    if not isinstance(analysis, dict):
        return {}
    selected = {"flops", "bytes accessed", "transcendentals", "optimal_seconds"}
    return {
        str(key): float(value)
        for key, value in analysis.items()
        if key in selected and isinstance(value, int | float) and math.isfinite(float(value))
    }


def _summarize_iterations(rows: list[Iteration]) -> dict[str, Any]:
    fields = Iteration._fields
    result: dict[str, Any] = {"samples": [row._asdict() for row in rows]}
    for field in fields:
        values = [float(getattr(row, field)) for row in rows]
        result[f"{field}_mean"] = statistics.fmean(values)
        result[f"{field}_p50"] = statistics.median(values)
    lengths = [row.decode_steps for row in rows]
    result["decode_steps_stable"] = len(set(lengths)) == 1
    return result


def _run_case(args: argparse.Namespace) -> None:
    batch_size = args.case_batch_size
    max_new_tokens = args.case_new_tokens
    if batch_size is None or max_new_tokens is None:
        raise ValueError("isolated case needs both hidden case arguments")
    if batch_size <= 0 or max_new_tokens <= 0 or args.max_prompt_len <= 0:
        raise ValueError("B/P/N must be positive")
    if args.warmup < 1 or args.iterations <= 0:
        raise ValueError("warmup must be >=1 and iterations positive")
    snapshots = sorted(glob.glob(DEFAULT_SNAPSHOT_GLOB))
    model_path = args.model_path or (snapshots[0] if snapshots else None)
    if model_path is None:
        raise FileNotFoundError("Gemma 4 E2B-it is not cached; pass --model-path")
    mesh = make_mesh()
    if mesh.size != 4 or batch_size % mesh.size:
        raise ValueError("each B4 case requires four devices and B divisible by four")

    ids, mask, prompt_record = _prompts(model_path, batch_size, args.max_prompt_len, args.seed)
    record: dict[str, Any] = {
        "environment": {
            "git": _git_state(),
            "jax": jax.__version__,
            "jaxlib": importlib.metadata.version("jaxlib"),
            "libtpu": importlib.metadata.version("libtpu"),
            "devices": [device.device_kind for device in jax.devices()],
            "command": " ".join(sys.argv),
        },
        "config": {
            "batch_size": batch_size,
            "max_prompt_len": args.max_prompt_len,
            "max_new_tokens": max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "seed": args.seed,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "prompt": prompt_record,
        "memory": {"start": _memory()},
    }
    load_started = time.perf_counter()
    host_params, config = load_hf_params(model_path, dtype=jnp.float32)
    fsdp_params = shard_gemma4_text_params(host_params, config, mesh)
    del host_params
    jax.block_until_ready(fsdp_params)
    record["load_and_fsdp_s"] = time.perf_counter() - load_started
    record["memory"]["fsdp_loaded"] = _memory()

    replicated_params = named_shardings(param_specs_replicated(config), mesh)
    to_rollout = jax.jit(
        lambda current: reshard_for_rollout(current, config, mesh),
        in_shardings=(tree_shardings(fsdp_params),),
        out_shardings=replicated_params,
    )
    started = time.perf_counter()
    compiled_reshard = to_rollout.lower(fsdp_params).compile()
    record["reshard_compile_s"] = time.perf_counter() - started
    started = time.perf_counter()
    rollout_params = compiled_reshard(fsdp_params)
    jax.block_until_ready(rollout_params)
    record["reshard_execute_ms"] = (time.perf_counter() - started) * 1000.0
    record["memory"]["rollout_loaded"] = _memory()

    eos_ids = load_hf_eos_token_ids(model_path, config)
    sampler_config = SamplerConfig(
        max_prompt_len=args.max_prompt_len,
        max_new_tokens=max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_ids=eos_ids,
        seed=args.seed,
    )
    batch_2d = NamedSharding(mesh, batch_spec(2))
    replicated = NamedSharding(mesh, P())
    device_ids, device_mask = shard_batch((ids, mask), mesh)
    generate_fn = jax.jit(
        lambda current, prompt_ids, prompt_mask, key: generate(
            current,
            config,
            prompt_ids,
            prompt_mask,
            sampler_config=sampler_config,
            key=key,
            mesh=mesh,
        ),
        in_shardings=(replicated_params, batch_2d, batch_2d, replicated),
        out_shardings=_rollout_shardings(mesh),
    )
    compile_key = jax.device_put(jax.random.PRNGKey(args.seed), replicated)
    started = time.perf_counter()
    compiled = generate_fn.lower(rollout_params, device_ids, device_mask, compile_key).compile()
    record["compile_s"] = time.perf_counter() - started
    record["compiled_memory"] = _memory_analysis(compiled)
    record["compiled_cost_analysis"] = _cost_analysis(compiled)

    warmup_ms: list[float] = []
    for index in range(args.warmup):
        key = jax.device_put(jax.random.fold_in(jax.random.PRNGKey(args.seed), index), replicated)
        started = time.perf_counter()
        warmup_output = compiled(rollout_params, device_ids, device_mask, key)
        jax.block_until_ready(warmup_output)
        warmup_ms.append((time.perf_counter() - started) * 1000.0)
    del warmup_output

    parameter_count = sum(leaf.size for leaf in jax.tree.leaves(rollout_params))
    bf16_weight_bytes = parameter_count * 2
    iterations: list[Iteration] = []
    completion_length_summaries: list[dict[str, float | int]] = []
    output_semantics: list[bool] = []
    for index in range(args.iterations):
        key = jax.device_put(
            jax.random.fold_in(jax.random.PRNGKey(args.seed), args.warmup + index),
            replicated,
        )
        started = time.perf_counter()
        output = compiled(rollout_params, device_ids, device_mask, key)
        jax.block_until_ready(output)
        elapsed_s = time.perf_counter() - started
        lengths = np.asarray(jax.device_get(output.lengths), dtype=np.int64)
        completion_mask = np.asarray(jax.device_get(output.completion_mask), dtype=np.bool_)
        completion_ids = np.asarray(jax.device_get(output.completion_ids))
        rollout_logps = np.asarray(jax.device_get(output.rollout_logps))
        expected_mask = np.arange(max_new_tokens)[None, :] < lengths[:, None]
        output_semantics.append(
            bool(
                np.array_equal(completion_mask, expected_mask)
                and np.all(completion_ids[~completion_mask] == config.pad_token_id)
                and np.all(rollout_logps[~completion_mask] == 0.0)
                and np.all(np.isfinite(rollout_logps[completion_mask]))
            )
        )
        decode_steps = int(lengths.max())
        actual_tokens = int(lengths.sum())
        slot_tokens = batch_size * decode_steps
        utilization = actual_tokens / slot_tokens
        slot_tokens_per_s = slot_tokens / elapsed_s
        actual_tokens_per_s = actual_tokens / elapsed_s
        estimated_model_tflops_s = 2.0 * parameter_count * slot_tokens_per_s / 1e12
        iterations.append(
            Iteration(
                execute_ms=elapsed_s * 1000.0,
                decode_steps=decode_steps,
                slot_tokens=slot_tokens,
                actual_tokens=actual_tokens,
                utilization=utilization,
                slot_tokens_per_s=slot_tokens_per_s,
                actual_tokens_per_s=actual_tokens_per_s,
                amortized_step_ms=elapsed_s * 1000.0 / decode_steps,
                estimated_model_tflops_s=estimated_model_tflops_s,
                estimated_mfu=estimated_model_tflops_s * 1e12 / TPU_V4_8_BF16_PEAK_FLOPS,
            )
        )
        completion_length_summaries.append(
            {
                "min": int(lengths.min()),
                "mean": statistics.fmean(int(value) for value in lengths),
                "p50": float(np.quantile(lengths, 0.50)),
                "p90": float(np.quantile(lengths, 0.90)),
                "max": decode_steps,
            }
        )
        del output

    record["parameter_count"] = parameter_count
    record["bf16_weight_bytes_per_chip"] = bf16_weight_bytes
    record["bf16_weight_gib_per_chip"] = bf16_weight_bytes / 2**30
    record["weight_bandwidth_lower_bound_ms"] = bf16_weight_bytes / TPU_V4_HBM_BANDWIDTH_BYTES_S * 1000.0
    record["warmup_ms"] = warmup_ms
    record["throughput"] = _summarize_iterations(iterations)
    record["completion_lengths"] = completion_length_summaries
    record["memory"]["final"] = _memory()
    throughput = record["throughput"]
    record["gates"] = {
        "finite_positive_time": all(math.isfinite(row.execute_ms) and row.execute_ms > 0 for row in iterations),
        "nonzero_decode_steps": all(row.decode_steps > 0 for row in iterations),
        "length_within_static_capacity": all(row.decode_steps <= max_new_tokens for row in iterations),
        "utilization_in_range": all(0.0 < row.utilization <= 1.0 for row in iterations),
        "output_mask_pad_logprob_semantics": all(output_semantics),
    }
    record["passed"] = all(record["gates"].values())
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "case": f"b{batch_size}_p{args.max_prompt_len}_n{max_new_tokens}",
                "passed": record["passed"],
                "compile_s": record["compile_s"],
                "execute_ms_mean": throughput["execute_ms_mean"],
                "slot_tokens_per_s_mean": throughput["slot_tokens_per_s_mean"],
                "actual_tokens_per_s_mean": throughput["actual_tokens_per_s_mean"],
                "utilization_mean": throughput["utilization_mean"],
                "amortized_step_ms_mean": throughput["amortized_step_ms_mean"],
                "peak_gib": [row["peak_gib"] for row in record["memory"]["final"]],
            }
        ),
        flush=True,
    )
    if args.require_pass and not record["passed"]:
        raise AssertionError(record["gates"])


def _run_matrix(args: argparse.Namespace) -> None:
    if args.warmup < 1 or args.iterations <= 0:
        raise ValueError("warmup must be >=1 and iterations positive")
    if any(batch <= 0 for batch in args.batch_sizes) or any(tokens <= 0 for tokens in args.new_token_counts):
        raise ValueError("all matrix dimensions must be positive")
    output_path = Path(args.output)
    case_dir = Path(args.case_output_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        for max_new_tokens in args.new_token_counts:
            case_path = case_dir / f"b{batch_size}_p{args.max_prompt_len}_n{max_new_tokens}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--case-batch-size",
                str(batch_size),
                "--case-new-tokens",
                str(max_new_tokens),
                "--max-prompt-len",
                str(args.max_prompt_len),
                "--temperature",
                str(args.temperature),
                "--top-k",
                str(args.top_k),
                "--top-p",
                str(args.top_p),
                "--seed",
                str(args.seed),
                "--warmup",
                str(args.warmup),
                "--iterations",
                str(args.iterations),
                "--output",
                str(case_path),
            ]
            if args.model_path:
                command.extend(("--model-path", args.model_path))
            result = subprocess.run(command, check=False)
            if result.returncode != 0 or not case_path.is_file():
                failure = {
                    "batch_size": batch_size,
                    "max_new_tokens": max_new_tokens,
                    "returncode": result.returncode,
                    "output": str(case_path),
                }
                failures.append(failure)
                print(json.dumps({"failed_case": failure}), flush=True)
                continue
            case = json.loads(case_path.read_text())
            cases.append(case)
    record = {
        "environment": {
            "git": _git_state(),
            "command": " ".join(sys.argv),
            "isolation": "one fresh Python/JAX process per B/N case so peak HBM and compile time do not leak",
        },
        "matrix": {
            "batch_sizes": args.batch_sizes,
            "new_token_counts": args.new_token_counts,
            "max_prompt_len": args.max_prompt_len,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "cases": cases,
        "failures": failures,
        "passed": not failures and len(cases) == len(args.batch_sizes) * len(args.new_token_counts),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"matrix_output": str(output_path), "cases": len(cases), "failures": failures}), flush=True)
    if args.require_pass and not record["passed"]:
        raise AssertionError(f"rollout matrix incomplete: {failures}")


def main() -> None:
    args = parse_args()
    if (args.case_batch_size is None) != (args.case_new_tokens is None):
        raise ValueError("case-batch-size and case-new-tokens must be provided together")
    if args.case_batch_size is not None:
        _run_case(args)
    else:
        _run_matrix(args)


if __name__ == "__main__":
    main()
