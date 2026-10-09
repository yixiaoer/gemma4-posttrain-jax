#!/usr/bin/env bash
# A complete GRPO experiment on GSM8K: data check -> continuous training -> resume from a mid-run
# checkpoint in a new process -> reload the final state and evaluate -> compare and summarize.
#
# The default is the full 200-step run: checkpoint at step 100, evaluate 256 dev questions every 50 steps.
# For a short trial run (a few minutes of training) use
#   MAX_STEPS=4 RESUME_STEP=2 EVAL_EVERY=2 EVAL_SIZE=8 bash examples/e2b_grpo_gsm8k.sh ...
# With WANDB_PROJECT set (optionally WANDB_ENTITY and WANDB_GROUP), the three training/evaluation processes
# log to one W&B group.
# At most two full training states (resume point and final state, about 52 GiB each) exist at once in CHECKPOINT_DIR.
set -euo pipefail
model_path=${1:?usage: e2b_grpo_gsm8k.sh HF_SNAPSHOT NEW_OUTPUT_DIR [CHECKPOINT_DIR]}
output_root=${2:?a new output directory is required}
checkpoint_root=${3:-$output_root/checkpoints}
max_steps=${MAX_STEPS:-200}
resume_step=${RESUME_STEP:-100}
eval_every=${EVAL_EVERY:-50}
eval_size=${EVAL_SIZE:-256}
seed=${SEED:-0}
if ((resume_step <= 0 || resume_step >= max_steps || resume_step % eval_every || max_steps % eval_every)); then
  echo "need 0 < RESUME_STEP < MAX_STEPS, both multiples of EVAL_EVERY, so evaluations line up before and after resume" >&2
  exit 2
fi
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-"$repo_root/.venv/bin/python"}
# The evaluation script requires the exact model path stored in the checkpoint, so resolve it first.
model_path=$(cd -- "$model_path" && pwd -P)
mkdir -- "$output_root"
output_root=$(cd -- "$output_root" && pwd)
mkdir -p -- "$checkpoint_root"
checkpoint_root=$(cd -- "$checkpoint_root" && pwd)
cd -- "$repo_root"
export JAX_PLATFORMS=${JAX_PLATFORMS:-tpu,cpu}
export JAX_DEFAULT_MATMUL_PRECISION=highest

group=${WANDB_GROUP:-grpo-gsm8k-seed$seed-$(date +%Y%m%d-%H%M%S)}
wandb_args() {
  if [[ -n ${WANDB_PROJECT:-} ]]; then
    printf '%s\n' --wandb --wandb-project "$WANDB_PROJECT" --wandb-group "$group" \
      --wandb-job-type "$1" --wandb-run-name "$group-$1"
  fi
}

# Start and end times of each stage go to stages.jsonl; the summary uses them to split the process time.
run_stage() {
  local name=$1 started ended code=0
  shift
  started=$(date +%s%N)
  "$@" || code=$?
  ended=$(date +%s%N)
  printf '{"stage": "%s", "started_unix_ns": %s, "ended_unix_ns": %s, "exit_code": %s}\n' \
    "$name" "$started" "$ended" "$code" >>"$output_root/stages.jsonl"
  return "$code"
}

common=(
  --model-path "$model_path" --dataset gsm8k --prompt-style chat
  --rollout-backend jax --rollout-layout fsdp --logprob-backend jax --training-device-ids 0 1 2 3
  --algorithm grpo --updates-per-rollout 1 --rollout-lag-updates 0
  --prompt-batch-size 2 --group-size 4 --microbatch-size 4
  --max-prompt-len 512 --max-new-tokens 1024 --max-steps "$max_steps"
  --beta 0 --learning-rate 1e-6 --seed "$seed" --dev-size 256 --dev-seed 0
  --no-dynamic-sampling --no-filter-truncated-loss --max-sampling-attempts 16
  --overlong-buffer-length 0 --overlong-penalty 0 --truncated-reward keep --format-bonus 0.1
  --vocab-chunk 8192 --sequence-chunk 256 --reward-workers 4
  --eval-every "$eval_every" --eval-split train-dev --eval-size "$eval_size" --eval-batch-size 8
  --eval-max-new-tokens 1024 --eval-seed 0
  --audit-final-state --audit-execution-layout
)
resume_from=$(printf '%s/continuous/step_%08d' "$checkpoint_root" "$resume_step")
final_checkpoint=$(printf '%s/resume/step_%08d' "$checkpoint_root" "$max_steps")

run_stage data-check env JAX_PLATFORMS=cpu "$python_bin" scripts/check_grpo_data.py \
  --model-path "$model_path" --dev-size 256 --dev-seed 0 --max-prompt-len 512 \
  --output "$output_root/data_check.json"

mapfile -t tracking < <(wandb_args train)
run_stage continuous "$python_bin" -u scripts/train_grpo.py "${common[@]}" "${tracking[@]}" \
  --output-dir "$output_root/continuous" \
  --checkpoint-dir "$checkpoint_root/continuous" --save-at-steps "$resume_step"

mapfile -t tracking < <(wandb_args resume)
run_stage resume "$python_bin" -u scripts/train_grpo.py "${common[@]}" "${tracking[@]}" \
  --resume "$resume_from" --output-dir "$output_root/resume" \
  --checkpoint-dir "$checkpoint_root/resume" --save-at-steps "$max_steps"

mapfile -t tracking < <(wandb_args evaluate)
run_stage reload-eval "$python_bin" -u scripts/evaluate_gsm8k.py "${tracking[@]}" \
  --model-path "$model_path" --checkpoint "$final_checkpoint" --audit-loaded-state \
  --output-dir "$output_root/reload-eval" --dataset gsm8k --subset train-dev --dev-size 256 --dev-seed 0 \
  --size "$eval_size" --seed 0 --batch-size 8 --max-prompt-len 512 --max-new-tokens 1024 \
  --prompt-style chat --rollout-layout fsdp --reward-workers 4

JAX_PLATFORMS=cpu "$python_bin" scripts/compare_resumed_run.py \
  --continuous "$output_root/continuous" --resumed "$output_root/resume" \
  --reload-eval "$output_root/reload-eval" --data-check "$output_root/data_check.json" \
  --output "$output_root/resume_check.json"
JAX_PLATFORMS=cpu "$python_bin" scripts/summarize_grpo_run.py "$output_root/continuous" \
  --stages "$output_root/stages.jsonl" --stage continuous --output "$output_root/continuous_summary.json"
