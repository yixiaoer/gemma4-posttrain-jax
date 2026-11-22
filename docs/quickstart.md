# Quickstart

**English** | [简体中文](quickstart.zh-CN.md)

This guide covers installation, text generation, SFT, GRPO, checkpoint save and restore, the complete GRPO experiment, and the scope of the tests. Run all commands from the repository root. The training examples use Gemma 4 E2B and four TPU v4 JAX devices.

## 1. Set up the environment

Python 3.12 is required. Create a dedicated environment for the project:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[tpu,hf,dev]'
.venv/bin/python -m pip check
.venv/bin/python scripts/check_env.py --backend tpu
```

For CPU development and testing, use the installation extra `.[hf,dev]` and select the CPU backend when checking the environment:

```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/check_env.py --backend cpu
```

The CPU check verifies versions and matrix computation. The TPU check also runs a small Pallas kernel.

These installation commands are for the native JAX implementation. The tpu-inference adapter depends on pinned versions and source code; do not run these upgrade commands directly in an existing engine environment. The recommended engine configuration runs in the same process and synchronizes weights over ICI. Two-process and host-transfer configurations remain available as research controls. The integration is experimental; see the [design overview](design-overview.md) for the differences and supported scope.

**Model:** Download `google/gemma-4-E2B-it` to a local directory containing its configuration, tokenizer, and safetensors weights. Replace `/path/to/e2b-snapshot` below with that directory.

**Data:** The short SFT test includes its own question-answer pair. GRPO uses GSM8K and downloads it on the first run. With an existing cache, set `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` to run offline.

## 2. Generate responses

For greedy generation:

```bash
.venv/bin/python scripts/generate.py \
  --model /path/to/e2b-snapshot --chat \
  --prompt "Calculate 3 times 4, then add 2." --max-new-tokens 64
```

To sample, or pass multiple prompts in one call:

```bash
.venv/bin/python scripts/generate.py \
  --model /path/to/e2b-snapshot --chat \
  --prompt "Explain what a gradient is." --prompt "Calculate 3 times 4, then add 2." \
  --temperature 0.8 --top-k 50 --top-p 0.9 --seed 42 \
  --max-new-tokens 128
```

Use `--top-k 0 --top-p 1` to disable candidate truncation. `--compare-hf` also runs Transformers on the CPU for comparison; this requires additional memory, and the two frameworks may produce different samples with the same seed because their random-number implementations differ.

## 3. Run SFT

```bash
bash examples/e2b_sft_smoke.sh /path/to/e2b-snapshot /path/to/new-sft
```

This is a short smoke test of the training workflow.

**What it does:** The script uses a built-in counting question: three rows of four red balls plus two blue balls gives 14 balls. It makes four copies for a batch size of four, matching the four TPUs, pads sequences to 128 tokens, and trains on the same batch for four steps. Each step:

1. Runs the forward pass and computes loss on the answer tokens.
2. Backpropagates to obtain gradients for the trainable parameters.
3. Updates parameters using Adam and its accumulated state.

Keeping the input fixed makes loss changes easier to inspect. Running several steps checks that parameters and optimizer state carry forward correctly and that later steps reuse the compilation cache. Four is simply the chosen test length.

**Configuration:** Learning rate `1e-4`, frozen embeddings, and no rematerialization. Results are written to `metrics.csv`, including each step's loss, gradient norm, and elapsed time.

**Train on GSM8K:** Run `scripts/train_sft.py` directly without `--overfit-one-batch`.

**Save and resume:** The smoke test does not save checkpoints by default. Use `--save-every`, `--checkpoint-dir`, and `--resume` with `scripts/train_sft.py` when needed.

**Choose a Python interpreter:** The SFT and recovery shell examples default to `.venv/bin/python`. Override this with `PYTHON_BIN=/absolute/path/to/python`.

## 4. Run GRPO and verify checkpoint recovery

```bash
bash examples/e2b_grpo_recovery.sh /path/to/e2b-snapshot /path/to/new-grpo
```

This example checks whether training resumed after an interruption exactly matches uninterrupted training.

**Workflow:**

1. Train the initial model for four steps, saving complete states at steps 2 and 4.
2. Start a new process, load the step-2 state, and continue to step 4.
3. Use separate processes to load the final states from uninterrupted and resumed training, then generate responses to the same four questions.
4. Compare the final parameters, Adam state, random state, data position, generated samples, and evaluation results.

**Training configuration:** Two prompts per iteration, four responses each, prompts up to 512 tokens, responses up to 256 tokens, and microbatch size 4. The learning rate is `1e-6`, the KL coefficient is 0, embeddings are frozen, rematerialization is enabled, and the seed is 0. Dynamic resampling allows up to 16 attempts. Samples excluded from updates are also recorded so that random-number use and data order can be checked.

**Evaluation:** Four questions from a held-out set of 500 check that the model was saved and restored correctly. This is too small to assess training quality.

### KL computation

The default `--beta 0` disables KL. Set a positive value, such as `--beta 0.04`, to enable it.

The implementation uses K3: with `d = reference_logprob - policy_logprob`, KL is `exp(d) - d - 1`. Near zero, a polynomial approximation reduces cancellation; elsewhere, a high-precision `expm1` reduces error from the TPU's default approximation.

The optional `--kl-clamp-value 10000` caps each token's KL contribution. Its KL gradient becomes zero above the cap, and the implementation handles the limit before exponential overflow. This cap is independent of the numerical correction near zero. It must be positive and finite, no larger than the maximum FP32 value, and used with a positive `beta`.

**Older checkpoints:** New checkpoints record the K3 version and cap. An older `beta=0` checkpoint without these fields continues with KL disabled. Resuming an older checkpoint with nonzero KL is rejected if the version is missing or mismatched; use the code that created that checkpoint to continue the original experiment.

### Disk and memory requirements

The output directory must not already exist. The three state files total about 100.59 GB (93.69 GiB); allow at least 100 GiB including other outputs. Files on tmpfs also consume host memory. The measured process-group memory peak was about 186.77 GiB, with the actual requirement depending on the model, input shapes, and environment.

### Output structure

```
new-grpo/
  baseline/                   Uninterrupted training: configuration, metrics, samples, parameter changes
  baseline-checkpoints/       Complete states at steps 2 and 4
  resume/                     Resumed training records for steps 3 and 4
  resume-checkpoints/         Resumed state at step 4
  eval-baseline/evaluation/   Per-question predictions and scores from uninterrupted training
  eval-resume/evaluation/     Per-question predictions and scores from resumed training
  recovery_check.json         Comparison results (complete: true when all checks pass)
```

If a run is interrupted, keep its outputs for diagnosis and use a new output directory when rerunning the complete example.

### Repeat the comparison or resume manually

To compare an already completed example again:

```bash
.venv/bin/python scripts/compare_recovery.py /path/to/new-grpo \
  --rescore --output /path/to/new-recovery-check.json
```

`--rescore` decodes recorded token IDs with the local tokenizer and scores the resulting text again. The comparison script does not retrain the model, but it does read all three large state files.

For manual recovery, point `--resume` to a specific step directory, such as `step_00000002`. `--max-steps 4` sets the global step limit; it does not request four additional steps after recovery. The script checks that the model, data, sampling, and training configuration match.

The SFT and GRPO examples each start from the initial model. Continuing from SFT into GRPO requires manual weight and state conversion; the examples do not perform that conversion automatically.

## 5. Choose a generation layout and run the complete experiment

### Tensor-parallel generation

Synchronous GRPO with full-parameter updates supports a tensor-parallel (TP) generation layout. The following command samples responses from the current policy for training:

```bash
JAX_DEFAULT_MATMUL_PRECISION=highest .venv/bin/python scripts/train_grpo.py \
  --model-path /path/to/e2b-snapshot --output-dir /path/to/new-tp-run \
  --rollout-layout tp --tp-allreduce auto \
  --prompt-batch-size 8 --group-size 4 --max-prompt-len 512 --max-new-tokens 1024 \
  --microbatch-size 4 --max-steps 12 --updates-per-rollout 1 \
  --learning-rate 1e-6 --beta 0 --remat --no-dynamic-sampling --no-filter-truncated-loss
```

When resuming, keep `--rollout-layout tp --tp-allreduce auto` and the other training settings unchanged. `auto` inherits the reduction mode actually used by the checkpoint. Independent evaluation also inherits the checkpoint's layout and reduction mode by default. CPU TP numerical checks use `--tp-allreduce xla`; the fused Pallas implementation requires a TPU.

TP generation uses the same checkpoint and evaluation mechanisms as the other native generation layouts. Keep the model, inputs, sampling parameters, and timing boundaries consistent when comparing layouts.

### Complete GSM8K experiment

The recovery example above runs four steps to make it practical to save and compare all state. The complete experiment below trains for 200 GSM8K steps, resumes in a new process at step 100, independently evaluates the final checkpoint, and automatically checks resumed results against uninterrupted training.

```bash
WANDB_PROJECT=gemma4_posttrain_jax \
  bash examples/e2b_grpo_gsm8k.sh /path/to/e2b-snapshot /path/to/new-run /path/to/checkpoints
```

A complete run takes more than an hour. First try `MAX_STEPS=4 RESUME_STEP=2 EVAL_EVERY=2 EVAL_SIZE=8` for a shorter check. See the [experiment guide](../examples/e2b_grpo_gsm8k.md) (Chinese) for configuration, stages, comparison results, and W&B curves.

---

## 6. Run independent evaluation

For example, evaluate the GRPO step-4 checkpoint:

```bash
.venv/bin/python scripts/evaluate_gsm8k.py \
  --checkpoint /path/to/new-grpo/baseline-checkpoints/step_00000004 \
  --output-dir /path/to/new-evaluation --subset train-dev --size 4 \
  --dev-size 500 --dev-seed 0 --batch-size 4 \
  --max-prompt-len 512 --max-new-tokens 256 --reward-workers 2
```

Outputs include each question's generated answer, reference answer, score, generated length, and token IDs. Evaluation also supports the initial model and MATH data; see `--help`.

With `--audit-loaded-state`, evaluation first reads every loaded parameter and Adam-state array and writes their checksums to `loaded_state_audit.json`. Comparing these against the records from the end of training checks that the saved state was fully read back. This complete read takes additional time, so the option is disabled by default.

## 7. Tests and validation scope

After installing `.[hf,dev]`, run the core CPU tests with:

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  .venv/bin/python -m pytest -m 'not tpu and not full_model'
```

Four virtual CPU devices check sharding rules and consistency across devices; they do not measure TPU performance. Some Linux process tests are skipped on other systems. RNG tests requiring exactly two devices can be run separately:

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  .venv/bin/python -m pytest tests/test_inference_rng.py
```

The CPU tests for stale rollout and candidate-batch recording assume a single device and skip themselves with four virtual devices. Run them separately with `XLA_FLAGS` unset:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -m 'not tpu' \
  tests/test_stale_rollout.py tests/test_rollout_batch_audit.py
```

ICI weight synchronization, BF16 conversion, and chunked checkpoint writes have passed 59 array checks on four TPU v4 chips. These cover transferred contents, floating-point conversion, sharding, saved bytes, and failure conditions. They do not load E2B. Run them with:

```bash
JAX_PLATFORMS=tpu,cpu .venv/bin/python -m pytest \
  tests/test_inference_device.py tests/test_checkpoint_chunks.py
```

Full E2B training, checkpointing, and recovery in a separate process require the recovery example or complete experiment above. Historical experiments do not automatically validate the reorganized public code, particularly nonzero-KL training, which needs verification against the current mathematical implementation.
