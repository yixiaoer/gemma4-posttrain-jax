#!/usr/bin/env python3
"""Greedy evaluation of the initial HF model or a full training checkpoint on a fixed GSM8K/MATH subset."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from train_grpo import compile_call, write_json

from gemma4_posttrain_jax.bench import device_peak_bytes
from gemma4_posttrain_jax.checkpoint import load_train_state, read_checkpoint_metadata
from gemma4_posttrain_jax.diagnostics import optional_package_version, source_git_state
from gemma4_posttrain_jax.evaluation import (
    evaluate_batches,
    load_evaluation_data,
    make_eval_sampler,
    prepare_eval_batches,
    resolve_eval_prompt_style,
    resolve_eval_rollout_layout,
)
from gemma4_posttrain_jax.lora import init_lora_params, prepare_lora_params
from gemma4_posttrain_jax.lora_training import LoRAPolicyParams, checkpoint_lora_config, make_lora_optimizer
from gemma4_posttrain_jax.losses import TrainState, init_train_state, make_optimizer
from gemma4_posttrain_jax.rollout_state import load_lagged_train_state
from gemma4_posttrain_jax.sampler import SamplerConfig, generate
from gemma4_posttrain_jax.sharding import (
    batch_spec,
    make_mesh,
    replicate_scalars,
    reshard_for_rollout,
    shard_batch,
    shard_gemma4_text_params,
    tree_shardings,
)
from gemma4_posttrain_jax.tp_rollout import resolve_allreduce
from gemma4_posttrain_jax.tracking import compile_tracking_metrics, grpo_tracking_metrics, init_tracker
from gemma4_posttrain_jax.weights import load_hf_eos_token_ids, load_hf_params


def record_loaded_state(state: TrainState[Any], output: Path) -> None:
    """Read the whole loaded parameter and Adam state and record a checksum per array.

    The record is compared with the one written at the end of training.
    """
    from gemma4_posttrain_jax.state_audit import summarize_train_state

    record = summarize_train_state(state)
    write_json(output, record)
    if not record["all_finite"]:
        raise ValueError("the loaded state contains non-finite values; the full record was saved")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--audit-loaded-state",
        action="store_true",
        help="before evaluating, read the whole loaded parameter and Adam state and record a checksum per array (slow)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-prompt-len", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", choices=("gsm8k", "math"), default="gsm8k")
    parser.add_argument(
        "--prompt-style",
        choices=("chat", "plain"),
        help="inherited from the checkpoint when omitted; chat for the initial HF model",
    )
    parser.add_argument(
        "--rollout-layout", choices=("replicated", "fsdp", "tp"), help="inherited from the checkpoint when omitted"
    )
    parser.add_argument(
        "--tp-allreduce",
        choices=("auto", "xla", "one_shot_fused", "one_shot_fused_high_rows_local"),
        default="auto",
        help="TP reduction mode; auto inherits the checkpoint's resolved mode",
    )
    parser.add_argument("--subset", choices=("test-monitor", "train-dev", "test-unused", "math500"))
    parser.add_argument("--dev-size", type=int, default=500)
    parser.add_argument("--dev-seed", type=int, default=0)
    parser.add_argument("--reward-workers", type=int, default=4)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="gemma4_posttrain_jax")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-tags", nargs="*", default=())
    parser.add_argument("--wandb-group", help="W&B group shared with the corresponding training runs")
    parser.add_argument("--wandb-job-type", help="W&B job type, e.g. evaluate")
    args = parser.parse_args()
    if args.audit_loaded_state and args.checkpoint is None:
        raise ValueError("--audit-loaded-state requires --checkpoint")
    if args.max_prompt_len is None:
        args.max_prompt_len = 2048 if args.dataset == "math" else 512
    if min(args.batch_size, args.max_prompt_len, args.max_new_tokens, args.reward_workers) <= 0:
        raise ValueError("evaluation batch size, lengths and workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_meta = None if args.checkpoint is None else read_checkpoint_metadata(args.checkpoint)["metadata"]
    args.prompt_style = resolve_eval_prompt_style(
        args.prompt_style, None if checkpoint_meta is None else checkpoint_meta["run_config"]
    )
    args.rollout_layout = resolve_eval_rollout_layout(
        args.rollout_layout, None if checkpoint_meta is None else checkpoint_meta["run_config"]
    )
    if args.model_path is None:
        if checkpoint_meta is not None:
            args.model_path = Path(checkpoint_meta["run_config"]["model_path"])
        else:
            snapshots = sorted(Path.home().glob(".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/*"))
            if len(snapshots) != 1:
                raise ValueError("--model-path is required")
            args.model_path = snapshots[0]
    if checkpoint_meta is not None and str(args.model_path.resolve()) != checkpoint_meta["run_config"]["model_path"]:
        raise ValueError("the evaluation config/tokenizer must come from the checkpoint's HF snapshot")
    lora_config = checkpoint_lora_config(
        None if checkpoint_meta is None else checkpoint_meta["run_config"], args.model_path
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    selected = load_evaluation_data(
        args.dataset,
        subset=args.subset,
        size=args.size,
        seed=args.seed,
        dev_size=args.dev_size,
        dev_seed=args.dev_seed,
        checkpoint_run_config=None if checkpoint_meta is None else checkpoint_meta["run_config"],
    )
    dataset, indices, provenance, split, args.subset = selected
    write_json(
        args.output_dir / "data_selection.json",
        {
            "dataset": args.dataset,
            "dataset_size": len(dataset),
            "dataset_fingerprint": dataset._fingerprint,
            "data_provenance": provenance,
            "split": split,
            "subset": args.subset,
            "size": len(indices),
            "seed": args.seed,
            "dev_size": args.dev_size if args.subset == "train-dev" else None,
            "dev_seed": args.dev_seed if args.subset == "train-dev" else None,
            "indices": indices,
            "prompt_style": args.prompt_style,
            "source_ids": [dataset[index].get("source_id") for index in indices],
        },
    )
    host, config = load_hf_params(str(args.model_path), dtype=jnp.float32 if lora_config is None else jnp.bfloat16)
    mesh = make_mesh()
    if args.batch_size <= 0 or args.batch_size % mesh.size or args.max_new_tokens <= 0 or args.reward_workers <= 0:
        raise ValueError(
            "evaluation lengths and workers must be positive, and the batch size must be divisible by the device count"
        )
    master = shard_gemma4_text_params(host, config, mesh)
    del host
    step = 0
    state: TrainState[Any]
    layout_state: TrainState[Any]
    if args.checkpoint and lora_config is None:
        assert checkpoint_meta is not None
        optimizer, _ = make_optimizer(
            master,
            learning_rate=checkpoint_meta["run_config"]["learning_rate"],
            freeze_embeddings=checkpoint_meta["run_config"]["freeze_embeddings"],
        )
        # Derive the full state layout from the parameter layout, then free the initial weights;
        # the optimizer is only needed for that layout.
        layout_state = replicate_scalars(init_train_state(master, optimizer), mesh)
        template = jax.eval_shape(lambda current: current, layout_state)
        layouts = tree_shardings(layout_state)
        del layout_state, master
        if checkpoint_meta["run_config"].get("rollout_lag_updates", 0):
            restored = load_lagged_train_state(args.checkpoint, template, layouts)
            state = restored.train
            del restored
        else:
            state = load_train_state(args.checkpoint, template, shardings=layouts)
        if args.audit_loaded_state:
            record_loaded_state(state, args.output_dir / "loaded_state_audit.json")
        master, step = state.params_f32, int(state.step)
        del state
    rollout_params: Any
    saved_config = None if checkpoint_meta is None else checkpoint_meta["run_config"]
    args.tp_allreduce = resolve_allreduce(
        args.tp_allreduce,
        layout=args.rollout_layout,
        tp=mesh.size,
        rows=(args.batch_size,),
        saved=saved_config,
    )
    tp_mesh = generate_options = None
    if args.rollout_layout == "tp":
        if lora_config is not None:
            raise ValueError("TP evaluation currently requires a full-parameter checkpoint")
        from gemma4_posttrain_jax.tp_rollout import (
            attention_rows,
            check_tp_config,
            generate_compiler_options,
            make_tp_mesh,
            prepare_tp_params,
        )

        check_tp_config(config, mesh.size)
        tp_mesh = make_tp_mesh(1, mesh.size, list(mesh.devices.flat))
        rollout_params, _ = prepare_tp_params(
            master,
            tp_mesh,
            attention_rows=attention_rows(args.tp_allreduce),
        )
        generate_options = generate_compiler_options(args.tp_allreduce)
    elif lora_config is None:
        rollout_params = jax.jit(
            lambda current: reshard_for_rollout(current, config, mesh, layout=args.rollout_layout)
        )(master)
    else:
        assert args.checkpoint is not None and checkpoint_meta is not None
        seed = checkpoint_meta["run_config"]["seed"]
        adapters = init_lora_params(master, lora_config, jax.random.fold_in(jax.random.PRNGKey(seed), 0x4C4F5241))
        adapters = jax.device_put(adapters, NamedSharding(mesh, P()))
        optimizer = make_lora_optimizer(checkpoint_meta["run_config"]["learning_rate"])
        layout_state = replicate_scalars(init_train_state(adapters, optimizer), mesh)
        template = jax.eval_shape(lambda current: current, layout_state)
        layouts = tree_shardings(layout_state)
        del layout_state, adapters
        state = load_train_state(args.checkpoint, template, shardings=layouts)
        if args.audit_loaded_state:
            record_loaded_state(state, args.output_dir / "loaded_state_audit.json")
        rollout_params, step = LoRAPolicyParams(master, state.params_f32), int(state.step)
        del state
    jax.block_until_ready(rollout_params)
    del master

    def shard_sample_inputs(ids, mask):
        if tp_mesh is None:
            return shard_batch((ids, mask), mesh)
        placement = NamedSharding(tp_mesh, P())
        return tuple(jax.device_put(np.asarray(value), placement) for value in (ids, mask))

    batches = prepare_eval_batches(
        dataset,
        tokenizer,
        size=args.size,
        seed=args.seed,
        batch_size=args.batch_size,
        max_prompt_len=args.max_prompt_len,
        pad_token_id=config.pad_token_id,
        indices=indices,
        chat=args.prompt_style == "chat",
    )
    eos_ids = load_hf_eos_token_ids(str(args.model_path), config)
    sampler_config = SamplerConfig(args.max_prompt_len, args.max_new_tokens, temperature=0, eos_ids=eos_ids)
    if lora_config is None:
        sample = make_eval_sampler(
            config,
            sampler_config,
            mesh,
            rollout_layout=args.rollout_layout,
            param_tree=rollout_params,
            tp_allreduce=args.tp_allreduce,
        )
    else:
        sample = jax.jit(
            lambda current, ids, mask, key: generate(
                current.frozen_base,
                config,
                ids,
                mask,
                sampler_config=sampler_config,
                key=key,
                mesh=mesh,
                lora=prepare_lora_params(current.adapters_f32, lora_config),
            ),
            in_shardings=(tree_shardings(rollout_params),) + (NamedSharding(mesh, batch_spec(2)),) * 2 + (None,),
        )
    first = shard_sample_inputs(batches[0].prompt_ids, batches[0].prompt_mask)
    executable = compile_call(
        "eval",
        sample,
        (rollout_params, *first, jax.random.PRNGKey(args.seed)),
        args.output_dir,
        False,
        generate_options,
    )
    source_root = Path(__file__).resolve().parents[1]
    revision, diff = source_git_state(source_root)
    (args.output_dir / "source.diff").write_text(diff or "")
    write_json(
        args.output_dir / "meta.json",
        {
            "git_commit": revision,
            "uncommitted_changes": None if diff is None else bool(diff),
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "checkpoint_metadata": checkpoint_meta,
            "step": step,
            "split": split,
            "subset": args.subset,
            "dataset_fingerprint": dataset._fingerprint,
            "data_provenance": provenance,
            "sampler": sampler_config._asdict(),
            "sampling_math_protocol": "fp32-exp-log-highest-v1",
            "prediction_record_schema": "raw-token-ids-v1",
            "jax_default_matmul_precision": jax.config.jax_default_matmul_precision or "default",
            "jax": jax.__version__,
            "libtpu": optional_package_version("libtpu"),
        },
    )
    with ExitStack() as stack:
        tracker = init_tracker(
            enabled=args.wandb,
            project=args.wandb_project,
            run_name=args.wandb_run_name,
            tags=args.wandb_tags,
            config={"logging_mode": "live", "source_metadata": json.loads((args.output_dir / "meta.json").read_text())},
            group=args.wandb_group,
            job_type=args.wandb_job_type,
        )
        stack.callback(tracker.finish)
        start = time.perf_counter()
        with ProcessPoolExecutor(
            max_workers=args.reward_workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            summary = evaluate_batches(
                lambda params, ids, mask, key: executable(params, *shard_sample_inputs(ids, mask), key),
                rollout_params,
                batches,
                tokenizer,
                output_dir=args.output_dir / "evaluation",
                eos_ids=eos_ids,
                seed=args.seed,
                executor=pool,
            )
        summary.update(step=step, total_eval_s=time.perf_counter() - start, peak_bytes=device_peak_bytes())
        write_json(args.output_dir / "summary.json", summary)
        print(summary, flush=True)
        tracker.log(
            {**grpo_tracking_metrics(evaluation=summary), **compile_tracking_metrics(args.output_dir)}, step=step
        )


if __name__ == "__main__":
    main()
