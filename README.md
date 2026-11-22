# Gemma4 Posttrain JAX

**English** | [简体中文](README.zh-CN.md)

A JAX project for implementing and studying Gemma 4 post-training on TPUs. It includes the text model, text generation, supervised fine-tuning (SFT), GRPO and related reinforcement learning algorithms, and training checkpoint save and restore. Development and validation focus on Gemma 4 E2B and TPU v4.

- Readable code: parameters are explicit inputs to JAX functions, so model forward passes, probability calculations, and gradient updates can be followed directly in the source.
- Documentation grounded in experiments: the docs describe optimizations and measured improvements, including the causes of numerical differences, memory use, and runtime costs.

## Features and status

| Area | Current implementation |
|---|---|
| Text model and weights | Pure JAX Gemma 4 text model and Hugging Face weight conversion, including Per-Layer Embeddings (PLE), shared KV, and sliding-window and global attention |
| SFT and reinforcement learning | Training entry points for SFT, GRPO, Dr. GRPO, DAPO, GSPO-token, and RLOO; optional KL regularization, multiple updates per rollout, and DAPO dynamic resampling. See [GRPO and related algorithms](docs/grpo-and-related-algorithms.md) for supported combinations |
| Native text generation | Prefill and KV-cache decoding, with ring buffers for sliding-window layers. The general sampler supports greedy decoding, temperature sampling, top-k, and top-p. TP generation supports greedy decoding or full-vocabulary sampling for training, checkpoint recovery, and evaluation |
| Training logprobs | Chunking over tokens and vocabulary with online log-sum-exp and backward recomputation; explicit vocabulary parallelism avoids retaining the full three-dimensional logits tensor. GRPO computes logprobs only for completion hidden states |
| Sharding and memory | FSDP-style parameter, gradient, and optimizer-state sharding on a single host; FP32 master parameters with BF16 computation; microbatches, optional layer rematerialization, and buffer donation. Native generation supports replicated, FSDP, and TP layouts, with fused reductions and normalization for four-chip TP |
| LoRA | Adapters for attention Q/K/V/O projections, integrated with native GRPO training, checkpoint save and restore, and independent evaluation |
| tpu-inference | Experimental generation backend. The recommended integration runs in the same process and synchronizes weights to inference chips over ICI. Two-process implementations, using host transfer or a shared distributed runtime, remain available as research controls; see the [design overview](docs/design-overview.md) |
| Checkpoint save and restore | Chunked device-array writes for parameters and optimizer state, plus random configuration and data position needed to resume training; loading into the target sharding, with scripts comparing uninterrupted training against recovery in a new process |
| Complete experiment | A 200-step GSM8K GRPO experiment with data checks, uninterrupted training, recovery in a new process, and independent checkpoint evaluation; automatic comparison of resumed results, evaluation curves, and timings, with optional W&B logging |
| Data, evaluation, and diagnostics | GSM8K and MATH data processing and rewards; independent and in-training evaluation; training metrics, probability differences, compilation and memory diagnostics, and optional W&B integration |
| Pallas LM-head/logprob | Optional BF16 training backend for four TPU v4 chips. It retains the native forward pass and hidden-state gradients while computing part of the weight gradients with Pallas; still experimental |
| Single-chip Pallas attention | Standalone TPU v4 experiments with Splash forward and backward passes, work partitioning across both TensorCores, and storage optimization |

See the [design overview](docs/design-overview.md) for implementation details and [GRPO and related algorithms](docs/grpo-and-related-algorithms.md) for differences between objectives.

**Validation scope**

- Most validation uses E2B on a single-host TPU v4-8; E4B has partial experimental coverage.
- 12B LoRA still needs training and checkpoint-recovery validation on real TPUs.
- With native JAX generation, E2B has completed 200 GSM8K training steps. Resuming in a new process at step 100 matched uninterrupted training in all checked results, and independent checkpoint evaluation matched in-training evaluation. Dev accuracy rose from 84.8% to 89.1% for one seed; see the [complete experiment results](examples/e2b_grpo_gsm8k.md) (Chinese).
- ICI weight synchronization for tpu-inference has passed TPU array tests and has been used in short synchronous training benchmarks. Full E2B checkpoint save and restore, and long training runs, still require separate validation.
- In two synchronous GRPO configurations, B8/N1024 and B16/N2048, this project and Tunix each processed the same responses using native generation and the same tpu-inference engine. With native generation, this project's throughput was **1.69–1.79×** that of the optimized Tunix baseline; with tpu-inference on both sides, it was **1.03–1.14×**.
- Chunked logprobs and vocabulary parallelism are used in training. Pallas LM-head/logprob requires explicit opt-in, and numerical differences in long GRPO runs remain unresolved. The local Pallas attention results still need validation in the full model.

The synchronous benchmarks use E2B-it on the same four TPU v4 chips. B is the number of prompts per batch, with four responses per prompt; N is the maximum response length. Fixed response sequences keep the workload equal. Each framework/backend combination runs in two independent processes, and throughput ratios use each process's median iteration time over iterations 7–12. Timing includes weight preparation, generation, reward scoring, and parameter updates, but excludes initial compilation, periodic checkpointing, and evaluation. These results measure execution efficiency under the stated configurations; they do not establish better learning quality under natural sampling.

Times below are **seconds per complete iteration**. Each cell lists the medians from two independent runs; lower is faster.

| Configuration | This project + native rollout | Tunix + native rollout | This project + tpu-inference | Tunix + tpu-inference |
|---|---:|---:|---:|---:|
| B8 / N1024 | **4.095 / 4.098** | 7.177 / 7.350 | 9.754 / 9.623 | 11.084 / 10.825 |
| B16 / N2048 | **15.188 / 15.177** | 25.703 / 25.843 | 25.280 / 25.018 | 26.064 / 25.937 |

See [TPU performance optimization](docs/tpu-performance-optimization.md) for the comparison method, the scope of the conclusions, and directions for further optimization.

**Known limitations**

- SFT gradient accumulation currently weights microbatches equally. When microbatches contain different numbers of supervised tokens, this differs from averaging over all supervised tokens in the full batch.

**Inference backend integration**

In addition to native rollout:

- tpu-inference requires a pinned engine environment. A complete comparison of learning quality and cost is still in progress.
- SGLang-JAX has passed integration tests but has not run through the full workflow. It is not included in the current public release.

## Installation

Use Python 3.12 and run the following from the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[tpu,hf,dev]"
.venv/bin/python scripts/check_env.py --backend tpu
```

**CPU development environment**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[hf,dev]"
JAX_PLATFORMS=cpu .venv/bin/python scripts/check_env.py --backend cpu
```

## Examples

Prepare a Hugging Face E2B model directory as described in the [quickstart](docs/quickstart.md). The examples use `/path/to/e2b-snapshot` for this directory. The second argument to each training example is a new output directory.

### Generate a response

```bash
.venv/bin/python scripts/generate.py \
  --model /path/to/e2b-snapshot --chat \
  --prompt "Calculate 3 times 4, then add 2." --max-new-tokens 64
```

### SFT

Train on one built-in question-answer pair for four steps. Each step computes gradients from the loss and applies an Adam update, providing a short check of the training workflow.

```bash
bash examples/e2b_sft_smoke.sh /path/to/e2b-snapshot /path/to/new-sft
```

### GRPO checkpoint recovery

Train and save the state, then resume in a new process and compare the results against uninterrupted training.

This example creates three complete state files totaling about 94 GiB. Check that the output filesystem has enough space before running it.

```bash
bash examples/e2b_grpo_recovery.sh /path/to/e2b-snapshot /path/to/new-grpo
```

### Complete GRPO experiment

Train on GSM8K for 200 steps, resume in a new process at step 100, and independently load the final state for evaluation. The script checks whether resumed training exactly matches uninterrupted training. A run takes more than an hour on TPU v4-8 and retains at most two complete training states at once, about 52 GiB each.

```bash
WANDB_PROJECT=gemma4_posttrain_jax \
  bash examples/e2b_grpo_gsm8k.sh /path/to/e2b-snapshot /path/to/new-run
```

Leave `WANDB_PROJECT` unset to run without W&B. See the [experiment guide](examples/e2b_grpo_gsm8k.md) (Chinese) for configuration, checks, and outputs.

The [quickstart](docs/quickstart.md) covers further options and checkpoint save and restore commands.

## Documentation

| Document | What it covers |
|---|---|
| [Quickstart](docs/quickstart.md) | Installation, model preparation, generation, training, and resuming from a checkpoint |
| [Design overview](docs/design-overview.md) | How data flows through generation and training, and how modules manage parameters, devices, and persistent state |
| [GRPO and related algorithms](docs/grpo-and-related-algorithms.md) | How rewards become gradients, how clipping and Adam affect updates, and what each algorithm changes |
| [TPU performance optimization](docs/tpu-performance-optimization.md) | Synchronous GRPO comparisons under two configurations, and optimization across frameworks, compilers, and instruction execution |

`gemma4_posttrain_jax/` contains the Python package, `scripts/` provides command-line entry points, `examples/` combines training stages, and `tests/` contains tests. Generated data, logs, and compiled graphs go to the output directories specified by the user.

The source code is licensed under [MIT](LICENSE). Models, tokenizers, datasets, and external dependencies retain their respective licenses.
