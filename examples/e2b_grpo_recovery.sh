#!/usr/bin/env bash
# 比较连续训练4步与从第2步恢复后的结果；保存三个训练状态文件，约94 GiB。
set -euo pipefail
model_path=${1:?用法: e2b_grpo_recovery.sh HF_SNAPSHOT NEW_OUTPUT_DIR}
output_root=${2:?需要新的输出目录}
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-"$repo_root/.venv/bin/python"}
mkdir -- "$output_root"
output_root=$(cd -- "$output_root" && pwd)
cd -- "$repo_root"
export JAX_PLATFORMS=${JAX_PLATFORMS:-tpu,cpu}
export JAX_DEFAULT_MATMUL_PRECISION=highest
common=(
  --model-path "$model_path" --rollout-backend jax
  --training-device-ids 0 1 2 3 --dev-size 500 --dev-seed 0
  --prompt-batch-size 2 --group-size 4 --max-prompt-len 512
  --max-new-tokens 256 --microbatch-size 4 --max-steps 4
  --learning-rate 1e-6 --beta 0 --freeze-embeddings --remat
  --vocab-chunk 8192 --sequence-chunk 256 --seed 0 --reward-workers 2
  --sampler-is-cap 2 --dynamic-sampling --max-sampling-attempts 16
  --save-at-steps 2 4 --audit-final-state --audit-update-state --audit-rollout-batches
)
"$python_bin" -u scripts/train_grpo.py "${common[@]}" \
  --output-dir "$output_root/baseline" --checkpoint-dir "$output_root/baseline-checkpoints"
"$python_bin" -u scripts/train_grpo.py "${common[@]}" \
  --resume "$output_root/baseline-checkpoints/step_00000002" \
  --output-dir "$output_root/resume" --checkpoint-dir "$output_root/resume-checkpoints"
for run in baseline resume; do
  "$python_bin" -u scripts/evaluate_gsm8k.py \
    --checkpoint "$output_root/$run-checkpoints/step_00000004" \
    --output-dir "$output_root/eval-$run" --subset train-dev --size 4 \
    --dev-size 500 --dev-seed 0 --batch-size 4 \
    --max-prompt-len 512 --max-new-tokens 256 --reward-workers 2
done
"$python_bin" scripts/compare_recovery.py "$output_root" --rescore
