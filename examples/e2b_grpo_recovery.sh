#!/usr/bin/env bash
# 兼容原示例路径；实验配置和流程集中在 experiments/grpo_resume。
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$repo_root/experiments/05-grpo-checkpoint-resume/run.sh" "$@"
