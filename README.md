# Gemma4 Posttrain JAX

在 TPU 上实现和研究 Gemma 4 后训练的 JAX 项目，包含文本模型、监督微调（SFT）、GRPO 等强化学习算法、文本生成，以及训练状态的保存和恢复。

主要使用 Gemma 4 E2B 和单机 TPU v4。代码将模型参数显式传给 JAX 函数，便于阅读模型前向、概率计算和梯度更新的实现；文档记录实际实验结果，解释数值误差、内存占用和运行速度的原因。

## 安装和运行

使用 Python 3.12，在仓库根目录安装：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[tpu,hf,dev]"
.venv/bin/python scripts/check_env.py --backend tpu
```

CPU 开发环境安装 `.[hf,dev]`，检查环境时使用 `JAX_PLATFORMS=cpu` 和 `--backend cpu`。发行包名是 `gemma4-posttrain-jax`，Python 导入名是 `gemma4_posttrain_jax`。

准备好 E2B 的 Hugging Face 模型目录后，可以生成回答或运行训练示例：

```bash
.venv/bin/python scripts/generate.py \
  --model /path/to/e2b-snapshot --chat \
  --prompt "计算3乘4再加2。" --max-new-tokens 64

bash examples/e2b_sft_smoke.sh /path/to/e2b-snapshot /path/to/new-sft
bash examples/e2b_grpo_recovery.sh /path/to/e2b-snapshot /path/to/new-grpo
```

SFT 示例用一道内置问答检查训练流程：每一步根据 loss 计算梯度，再由 Adam 修改模型参数，连续运行 4 步。GRPO 示例会保存训练状态，再启动新进程继续训练，检查结果是否与连续训练一致；它会生成三个完整状态文件，合计约 **94 GiB**。示例的目的、具体步骤和输出说明见[安装与使用](docs/quickstart.md)。

## 目前能做什么

| 功能 | 已完成的范围 |
|---|---|
| E2B 文本模型、SFT、原生 GRPO | 已在 TPU v4 上运行，包含生成、训练、保存、恢复和独立评估 |
| Dr. GRPO、DAPO、GSPO-token、RLOO 等 | 已有实现和限定配置下的实验，尚未覆盖所有模型与参数组合 |
| TopK、Top-p 生成 | 已有 CPU/TPU 测试；RL 入口目前使用温度 1 的完整词表采样 |
| tpu-inference | 实验性接入；同进程默认通过 ICI 同步，另有主机中转及共同分布式运行时的双进程对照；完整质量与成本比较仍在进行 |
| E4B、12B LoRA | E4B 有独立实验；12B 仍需完成真实 TPU 训练和恢复验证 |

当前 ICI 和分块保存代码已通过 TPU v4 数组测试，迁移后的完整 E2B 训练、保存和独立续训仍需重新验证。引擎使用单独的固定依赖环境，适用范围见[整体设计](docs/design-overview.md)。实验报告将在后续整理发布。

## 使用、设计与算法

| 文档 | 阅读后可以了解什么 |
|---|---|
| [快速开始](docs/quickstart.md) | 怎样安装、准备模型、运行生成和训练，以及保存后如何继续 |
| [整体设计](docs/design-overview.md) | 数据如何经过模型、生成与训练，各模块如何管理参数、设备和持久状态 |
| [GRPO 及相关算法](docs/grpo-and-related-algorithms.md) | 奖励怎样变成梯度，概率比裁剪和 Adam 怎样影响参数，各算法具体改了哪一步 |


`gemma4_posttrain_jax/` 是 Python 包，`scripts/` 提供运行命令，`examples/` 组合训练步骤，`tests/` 保存测试。运行生成的数据、日志和编译图保存在使用者指定的输出目录。

源码采用[MIT](LICENSE)。模型、tokenizer、数据集和外部依赖的许可分别适用。
