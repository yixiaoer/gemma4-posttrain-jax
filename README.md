# Gemma4 Posttrain JAX

在 TPU 上实现和研究 Gemma 4 后训练的 JAX 项目，涵盖文本模型、文本生成、监督微调（SFT）、GRPO 等强化学习算法，以及训练状态的保存与恢复。主要针对 Gemma 4 E2B 和 TPU v4。

特点：

- 代码易读：模型参数显式传给 JAX 函数，前向计算、概率计算和梯度更新都能在代码里直接看清。
- 文档基于实测：记录实际实验优化过程，提升结果，并解释数值误差、内存占用和运行速度的成因。

## 功能与进度

| 部分 | 当前实现 |
|---|---|
| 文本模型与权重 | Gemma 4 文本模型的纯 JAX 实现和 Hugging Face 权重转换，包含逐层输入嵌入（PLE）、共享 KV、滑窗与全局 attention |
| SFT 与强化学习 | 提供 SFT、GRPO、Dr. GRPO、DAPO、GSPO-token 和 RLOO 训练入口；支持可选 KL 正则、多次更新和 DAPO 动态补采，组合限制见[GRPO 及相关算法](docs/grpo-and-related-algorithms.md) |
| 原生文本生成 | prefill 和 KV cache 解码，滑窗层使用环形缓存；支持 greedy、温度采样、top-k 和 top-p。RL 入口使用温度 1 的完整词表采样 |
| 训练 logprob | 沿 token 和词表分块，在线累计 log-sum-exp，反向重计算；支持显式词表并行，无需保存完整的三维 logits。GRPO 只对回答对应的 hidden 计算 logprob |
| 分片与内存管理 | 单主机 FSDP 风格的参数、梯度和优化器状态分片；FP32 主参数、BF16 计算；支持微批次、可选层重计算和 buffer donation。原生生成权重可复制或分片，重分片与计算所需的跨芯片通信使用 ICI |
| LoRA | 实现 attention Q/K/V/O 投影的适配器，接入原生 GRPO 训练、保存、恢复和独立评估 |
| tpu-inference | 实验性生成后端，推荐同进程接入并通过 ICI 向推理芯片同步权重；双进程方案（经主机中转或共用分布式运行时）仅保留为开发对照，区别与配置见[整体设计](docs/design-overview.md) |
| 保存与恢复 | 分块写出设备数组，保存参数、优化器状态及续训所需的随机配置和数据位置；按目标分片加载，提供连续训练与新进程续训的比较脚本 |
| 数据、评估与诊断 | GSM8K、MATH 数据处理与奖励计算，独立评估和训练期间评估；记录训练指标、概率差异、编译与内存信息，可选接入 W&B |
| Pallas LM-head/logprob | 四芯片 TPU v4、BF16 的可选训练后端；保留原生前向和 hidden 梯度，用 Pallas 计算部分权重梯度，目前仍为实验选项|
| 单 chip Pallas attention | 针对 TPU v4 的 Splash 前向、反向、双 TensorCore 分工与存储优化；目前是独立实验实现 |

各部分的实现细节见[整体设计](docs/design-overview.md)，各算法的区别见[GRPO 及相关算法](docs/grpo-and-related-algorithms.md)。

**适用范围**

- 主要验证环境是 E2B 和单机 TPU v4-8；E4B 有部分实验。
- 12B LoRA 尚需在真实 TPU 上完成训练与恢复验证。
- tpu-inference 的 ICI 权重同步和 checkpoint 分块保存已通过 TPU 数组测试，但这不能替代完整 E2B 训练、保存和新进程续训的验证（后者仍待完成）。
- 分块 logprob 和词表并行已用于训练；Pallas LM-head/logprob 需显式启用，长期 GRPO 数值差异尚未解决。Pallas attention 的局部结果仍需在完整模型中验证。

**已知限制**

- SFT 梯度累积目前按微批次等权平均。各微批次的监督 token 数不同时，结果不等价于整个批次按 token 平均的目标。

**推理后端接入**

除原生 rollout 外，推理后端的接入进度如下：

- tpu-inference：依赖版本固定的引擎环境；质量与成本的完整比较仍在进行。
- SGLang-JAX：已完成接入测试，尚未在完整流程中运行；当前公开版本尚未包含该后端。

## 安装和运行

使用 Python 3.12，在仓库根目录安装：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[tpu,hf,dev]"
.venv/bin/python scripts/check_env.py --backend tpu
```

**CPU（开发环境）**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[hf,dev]"
JAX_PLATFORMS=cpu .venv/bin/python scripts/check_env.py --backend cpu
```

## 运行示例

先准备好 E2B 的 Hugging Face 模型目录（方法见[快速开始](docs/quickstart.md)），下文记作 `/path/to/e2b-snapshot`。两个训练示例的第二个参数是输出目录。

### 生成回答

```bash
.venv/bin/python scripts/generate.py \
  --model /path/to/e2b-snapshot --chat \
  --prompt "计算3乘4再加2。" --max-new-tokens 64
```

### SFT

用一道内置问答连续训练 4 步（每步由 loss 计算梯度，再用 Adam 更新参数），检查训练流程是否正常。

```bash
bash examples/e2b_sft_smoke.sh /path/to/e2b-snapshot /path/to/new-sft
```

### GRPO 断点续训

训练并保存状态后，启动新进程接着训练，检查结果是否与不间断训练一致。

该示例会生成 3 个完整的状态文件，合计约 94 GiB。运行前请确认输出目录所在磁盘空间充足。

```bash
bash examples/e2b_grpo_recovery.sh /path/to/e2b-snapshot /path/to/new-grpo
```

各示例的目的、具体步骤和输出说明同样见[快速开始](docs/quickstart.md)。

## 文档

| 文档 | 阅读后可以了解什么 |
|---|---|
| [快速开始](docs/quickstart.md) | 怎样安装、准备模型、运行生成和训练，以及保存后如何继续 |
| [整体设计](docs/design-overview.md) | 数据如何经过模型、生成与训练，各模块如何管理参数、设备和持久状态 |
| [GRPO 及相关算法](docs/grpo-and-related-algorithms.md) | 奖励怎样变成梯度，概率比裁剪和 Adam 怎样影响参数，各算法具体改了哪一步 |


`gemma4_posttrain_jax/` 是 Python 包，`scripts/` 提供运行命令，`examples/` 组合训练步骤，`tests/` 保存测试。运行生成的数据、日志和编译图保存在使用者指定的输出目录。

源码采用[MIT](LICENSE)。模型、tokenizer、数据集和外部依赖的许可分别适用。
