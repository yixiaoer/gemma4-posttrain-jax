#!/usr/bin/env bash
# Run four updates on a built-in question and answer to check gradient computation and Adam state updates.
set -euo pipefail
model_path=${1:?用法: e2b_sft_smoke.sh HF_SNAPSHOT NEW_OUTPUT_DIR}
output_root=${2:?需要新的输出目录}
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-"$repo_root/.venv/bin/python"}
mkdir -- "$output_root"
output_root=$(cd -- "$output_root" && pwd)
cd -- "$repo_root"
export JAX_PLATFORMS=${JAX_PLATFORMS:-tpu,cpu}
export JAX_DEFAULT_MATMUL_PRECISION=highest
"$python_bin" -u scripts/train_sft.py \
  --model-path "$model_path" --overfit-one-batch --batch-size 4 \
  --sequence-length 128 --max-steps 4 --learning-rate 1e-4 \
  --freeze-embeddings --no-remat --log-every 1 \
  --log-csv "$output_root/metrics.csv" --require-single-executable
