# 安装与使用

这篇文档从安装开始，依次介绍文本生成、SFT、GRPO、保存和继续训练，以及数值与性能实验。所有命令都在仓库根目录运行。训练示例使用 Gemma 4 E2B 和四个 TPU v4 JAX 设备。

## 准备环境和模型

需要 Python 3.12。建议为项目创建独立环境：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[tpu,hf,dev]'
.venv/bin/python -m pip check
.venv/bin/python scripts/check_env.py --backend tpu
```

只在 CPU 上开发或运行测试时，将安装选项换成 `.[hf,dev]`，然后执行：

```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/check_env.py --backend cpu
```

CPU 模式检查版本和矩阵计算；TPU 模式还会运行一个小型 Pallas kernel。脚本支持 `--help`，查看帮助不会执行设备任务。

训练需要下载好的 `google/gemma-4-E2B-it` 模型目录，其中应包含配置、tokenizer 和 safetensors 权重。下面用 `/path/to/e2b-snapshot` 代指这个目录，运行前请替换成实际的绝对路径。

SFT 短测试使用脚本内置的一道问答，无需下载训练数据；GRPO 示例使用 GSM8K。首次读取缺少的数据时，Hugging Face 工具可能联网下载；已有完整缓存时，可以设置 `HF_HUB_OFFLINE=1` 和 `HF_DATASETS_OFFLINE=1`。本仓库不包含模型权重。

## 生成回答

```bash
.venv/bin/python scripts/generate.py \
  --model /path/to/e2b-snapshot --chat \
  --prompt "计算3乘4再加2。" --max-new-tokens 64
```

默认使用 greedy，即每一步选择分数最高的 token。随机生成可以设置温度、TopK、Top-p 和随机种子；重复传入 `--prompt` 可以一次生成多条回答：

```bash
.venv/bin/python scripts/generate.py \
  --model /path/to/e2b-snapshot --chat \
  --prompt "解释什么是梯度。" --prompt "计算3乘4再加2。" \
  --temperature 0.8 --top-k 50 --top-p 0.9 --seed 42 \
  --max-new-tokens 128
```

`--top-k 0 --top-p 1` 关闭候选截断。`--compare-hf` 会同时用 Transformers 在 CPU 上加载模型并显示结果，需要额外的内存。两个框架的随机数实现不同，即使用相同的随机种子，随机生成的回答也可能不同。

## 运行 SFT

```bash
bash examples/e2b_sft_smoke.sh /path/to/e2b-snapshot /path/to/new-sft
```

这个示例用于快速检查训练能否正常执行。脚本内置了一道计算球数的问答：三排红球，每排四个，再加两个蓝球，答案是 14。它将这道问答重复四份，组成 batch size 为 4 的输入，适配四个 TPU 设备；每行长度补齐或截断到 128 token。

这里的一个“训练步”包括以下操作：

1. 用当前模型参数计算答案的 loss，衡量模型对给定答案的预测误差。
2. 对 loss 求梯度，得到各个可训练参数的调整依据。
3. Adam 根据梯度和之前保存的优化器状态修改参数，供下一步使用。

脚本对同一批输入连续执行 4 个训练步。固定输入便于检查 loss 的变化；连续执行几步可以检查参数和优化器状态能否继续使用，以及后续步骤是否复用已有的编译结果。4 是短测试选定的步数，没有特殊的算法含义。

配置使用学习率 `1e-4`，冻结 embedding，不使用重计算。结果写入 `metrics.csv`，包含每一步的 loss、梯度范数和耗时。需要在 GSM8K 上训练时，直接使用 `scripts/train_sft.py` 并省略 `--overfit-one-batch`。

这个短测试默认不保存训练状态。需要保存或继续训练时，在 `scripts/train_sft.py` 中设置 `--save-every`、`--checkpoint-dir` 和 `--resume`。

两个 shell 示例默认使用仓库的 `.venv/bin/python`。如果已有其他 Python 环境，可以通过 `PYTHON_BIN=/absolute/path/to/python` 指定解释器。

## 运行 GRPO，并检查保存和恢复

```bash
bash examples/e2b_grpo_recovery.sh /path/to/e2b-snapshot /path/to/new-grpo
```

这个示例检查中断后继续训练能否得到与连续训练一致的结果。每个训练步仍以 Adam 完成一次参数修改为结束；与 SFT 不同，GRPO 的训练数据来自模型生成的回答及其奖励。示例执行以下步骤：

1. 从初始模型训练到第 4 步，在第 2、4 步保存完整状态。
2. 启动新的 Python 进程，加载第 2 步状态，继续训练到第 4 步。
3. 在另外两个进程中，分别加载连续训练和恢复训练的末步状态，在相同的四道题上生成回答。
4. 比较最终参数、Adam 状态、随机状态、数据读取位置，以及恢复后生成的所有样本和评估结果。

这里每轮取 2 道题，每题生成 4 个回答；prompt 最长 512 token，回答最多 256 token，训练 microbatch 为 4。学习率为 `1e-6`，KL 系数为 0，冻结 embedding，开启重计算，使用随机种子 0。动态补采最多尝试 16 次；未进入更新的样本也会记录，用于检查恢复后随机数和数据顺序是否一致。

评估题来自预先划出的 500 道题，与训练题分开。四题评估用于检查模型是否正确保存和恢复，不能据此判断训练质量。

### 空间与输出

输出目录必须不存在。三个状态文件合计约 100.59 GB，也就是 93.69 GiB；加上其他输出，建议预留至少 100 GiB 可写空间。如果输出位于 tmpfs，文件还会占用主机内存。这次实际运行的进程组内存峰值约 186.77 GiB，具体需要多少内存取决于模型、输入形状和运行环境。

```text
new-grpo/
  baseline/                   连续训练的配置、指标、样本和参数变化记录
  baseline-checkpoints/       第 2、4 步的完整状态
  resume/                     恢复后第 3、4 步的训练记录
  resume-checkpoints/         恢复训练的第 4 步状态
  eval-baseline/evaluation/   连续训练模型的逐题预测和评分
  eval-resume/evaluation/     恢复训练模型的逐题预测和评分
  recovery_check.json         两次训练的比较结果
```

`recovery_check.json` 只有在所有检查通过后才会写出 `complete: true`。如果运行中断，保留已有输出供排查，再次执行整个示例时使用新的目录。

### 检查或继续已有训练

可以在完成的示例上重新比较结果，将新报告写到其他路径：

```bash
.venv/bin/python scripts/compare_recovery.py /path/to/new-grpo \
  --rescore --output /path/to/new-recovery-check.json
```

比较还会核对生成记录的必需字段、四道评估题的身份与完整性，以及评估是否加载了第 4 步模型。`--rescore` 使用本地 tokenizer，从记录的实际 token 重新解码和评分。比较脚本不会重新训练模型，但会读取三个大状态文件。

手动使用训练脚本时，`--resume` 指向某一步的完整目录，例如 `step_00000002`。`--max-steps 4` 表示训练到全局第 4 步，而不是恢复后额外训练四步。恢复时需要保持模型、数据选择、采样方式和训练设置一致；脚本会检查这些配置。

SFT 和 GRPO 示例分别从初始模型开始。将 SFT 的训练结果作为 RL 起点，需要处理模型权重和任务状态的转换，当前示例没有自动完成这一步。

## 单独评估模型

例如，评估刚才得到的第 4 步状态：

```bash
.venv/bin/python scripts/evaluate_gsm8k.py \
  --checkpoint /path/to/new-grpo/baseline-checkpoints/step_00000004 \
  --output-dir /path/to/new-evaluation --subset train-dev --size 4 \
  --dev-size 500 --dev-seed 0 --batch-size 4 \
  --max-prompt-len 512 --max-new-tokens 256 --reward-workers 2
```

评估输出包含逐题回答、标准答案、评分、生成长度和实际 token ID。脚本还支持初始模型与 MATH 数据，完整参数可通过 `--help` 查看。

## 运行数值与性能实验

以下三个实验使用脚本生成的小型输入，不需要下载模型：

| 脚本 | 检查内容 |
|---|---|
| `scripts/probe_vocab_logprob.py` | 自动分片与词表并行的 loss、梯度和设备通信 |
| `scripts/probe_sampling_math.py` | 默认归一化与高精度归一化的 logprob 误差 |
| `scripts/probe_weight_checks.py` | BF16 数值检查与位检查的行为和耗时 |

例如，在 CPU 上比较两种采样概率计算方法：

```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/probe_sampling_math.py \
  --backend cpu --save-ir --output-dir outputs/sampling-cpu
```

结果写入指定目录的 `summary.json`；`--save-ir` 还会导出 StableHLO 和优化后的 HLO，用于分析编译器如何处理这些计算。输出目录必须不存在，便于保留不同运行的结果。

另外两个脚本的完整命令、实验设计和结果解释见[实验结果与分析](development-results.md)的对应章节。词表并行小实验需要四个 CPU 虚拟设备或四个 TPU 设备；BF16 检查仅使用 CPU。
