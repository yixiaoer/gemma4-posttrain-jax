# QuickStart

这篇文档从安装开始，依次介绍文本生成、SFT、GRPO、保存和继续训练，以及测试与验证范围。所有命令都在仓库根目录运行。训练示例使用 Gemma 4 E2B 和四个 TPU v4 JAX 设备。

## 1. 搭环境

需要 Python 3.12。建议为项目创建独立环境：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[tpu,hf,dev]'
.venv/bin/python -m pip check
.venv/bin/python scripts/check_env.py --backend tpu
```

如果只在 CPU 上开发或跑测试，把安装选项换成 `.[hf,dev]`，检查时加前缀：

```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/check_env.py --backend cpu
```

CPU 模式只检查版本和矩阵计算；TPU 模式还会跑一个小 Pallas kernel。

以上安装用于原生 JAX 路径。tpu-inference 适配层依赖固定版本与源码，不能直接在引擎环境执行这组升级命令。接入引擎时推荐同进程执行、通过 ICI 同步权重；双进程和主机中转仅保留为开发对照。该接入仍属于实验配置，具体区别与适用范围见[整体设计](design-overview.md)。

**模型准备：** 下载 `google/gemma-4-E2B-it` 到本地目录（需包含配置、tokenizer、safetensors 权重）。下文统一用 `/path/to/e2b-snapshot` 指代，运行前替换成实际路径。

**数据：** SFT 短测试自带内置问答，不需要额外数据。GRPO 用 GSM8K，首次运行会自动下载；如果已有缓存，可设 `HF_HUB_OFFLINE=1` 和 `HF_DATASETS_OFFLINE=1` 离线运行。

## 2. 生成回答

最简单的 greedy 生成：

```bash
.venv/bin/python scripts/generate.py \
  --model /path/to/e2b-snapshot --chat \
  --prompt "计算3乘4再加2。" --max-new-tokens 64
```

加上采样参数，或一次传多条 prompt：

```bash
.venv/bin/python scripts/generate.py \
  --model /path/to/e2b-snapshot --chat \
  --prompt "解释什么是梯度。" --prompt "计算3乘4再加2。" \
  --temperature 0.8 --top-k 50 --top-p 0.9 --seed 42 \
  --max-new-tokens 128
```

几个实用选项：`--top-k 0 --top-p 1` 关闭候选截断；`--compare-hf` 同时用 Transformers 在 CPU 上跑一遍对比（需要额外内存，且两个框架随机数实现不同，同种子结果可能不一样）。

## 3. 运行 SFT

```bash
bash examples/e2b_sft_smoke.sh /path/to/e2b-snapshot /path/to/new-sft
```

这是一个快速示例测试，用来确认训练流程能跑通。

**它做了什么：** 脚本内置一道球数计算题（三排红球×4 + 两个蓝球 = 14），复制 4 份凑成 batch size 4（对应 4 块 TPU），序列长度统一到 128 token，然后在这同一批数据上连跑 4 步。

**每一步的含义：**
1. 前向传播，算出答案部分的 loss（模型预测与正确答案的差距）。
2. 反向传播，得到各可训练参数的梯度。
3. Adam 优化器根据梯度和历史状态更新参数。

固定输入是为了观察 loss 变化；跑多步是为了验证参数/优化器状态能正确传递，以及后续步骤复用编译缓存。4 步只是测试选的数字，没有特殊含义。

**配置：** 学习率 `1e-4`，冻结 embedding，不用重计算。结果写入 `metrics.csv`（含每步 loss、梯度范数、耗时）。

**正式训练 GSM8K：** 直接用 `scripts/train_sft.py`，去掉 `--overfit-one-batch`。

**保存和续训：** 冒烟测试默认不保存。需要时在 `scripts/train_sft.py` 里加 `--save-every`、`--checkpoint-dir`、`--resume`。

**换 Python 解释器：** 两个 shell 示例默认用 `.venv/bin/python`，可通过 `PYTHON_BIN=/absolute/path/to/python` 覆盖。


## 4. 运行 GRPO + 断点恢复验证

```bash
bash examples/e2b_grpo_recovery.sh /path/to/e2b-snapshot /path/to/new-grpo
```

这个示例的核心目的是验证：中断后恢复训练，结果与连续训练完全一致。

**流程：**
1. 从初始模型训练 4 步，在第 2、4 步保存完整状态。
2. 新进程加载第 2 步状态，继续训练到第 4 步。
3. 两个进程分别加载连续训练和恢复训练的末步状态，在相同 4 道题上生成回答。
4. 逐项比较：最终参数、Adam 状态、随机状态、数据读取位置、生成样本、评估结果。

**训练配置：** 每轮 2 道题 × 4 个回答，prompt ≤ 512 token，回答 ≤ 256 token，microbatch 4。学习率 `1e-6`，KL 系数 0，冻结 embedding，开启重计算，seed 0。动态补采最多 16 次；未参与更新的样本也会记录（用于校验随机数和数据顺序）。

**评估：** 用预先划出的 500 题，取 4 题检查模型是否正确保存和恢复——不能据此判断训练质量。

### KL 计算

默认 `--beta 0`，不算 KL。需要时设正数，如 `--beta 0.04`。

公式采用 K3：令 `d = reference_logprob - policy_logprob`，KL = `exp(d) - d - 1`。数值处理上，接近零时用多项式近似，范围外用高精度 `expm1`，以减小相减误差和 TPU 默认近似的偏差。

可选 `--kl-clamp-value 10000` 对每个 token 的 KL 设上限（超限后该 token 的 KL 梯度为零，指数溢出前就会处理）。这和接近零的数值修正是两个独立选项。上限必须为正有限数、不超过 FP32 最大值，且只能搭配正 `beta` 使用。

**旧状态兼容：** 新状态会记录 K3 版本和上限值。旧 `beta=0` 状态缺这两个字段时，按原逻辑（不用 KL）继续训练。旧的非零 KL 状态如果版本缺失或不匹配，会拒绝续训——要继续原实验，请用保存该状态的代码。

### 磁盘和内存

输出目录必须不存在。三个状态文件合计约 100.59 GB（93.69 GiB），加上其他输出，建议预留 ≥ 100 GiB。如果输出在 tmpfs 上，文件还会占主机内存。实测进程组内存峰值约 186.77 GiB（具体取决于模型、输入形状和环境）。

### 输出结构

```
new-grpo/
  baseline/                   连续训练记录（配置、指标、样本、参数变化）
  baseline-checkpoints/       第 2、4 步完整状态
  resume/                     恢复后第 3、4 步训练记录
  resume-checkpoints/         恢复训练第 4 步状态
  eval-baseline/evaluation/   连续训练模型的逐题预测和评分
  eval-resume/evaluation/     恢复训练模型的逐题预测和评分
  recovery_check.json         比较结果（所有检查通过时 complete: true）
```

运行中断的话，保留已有输出排查问题，再跑用新目录。

### 重新比较 / 手动续训

对已完成的示例重新跑比较：

```bash
.venv/bin/python scripts/compare_recovery.py /path/to/new-grpo \
  --rescore --output /path/to/new-recovery-check.json
```

`--rescore` 会用本地 tokenizer 从记录的 token ID 重新解码评分。比较脚本不会重新训练，但会读取三个大状态文件。

手动续训时，`--resume` 指向具体步骤目录（如 `step_00000002`），`--max-steps 4` 是全局步数上限（不是恢复后再跑 4 步）。恢复要求模型、数据、采样和训练配置一致，脚本会检查。

**注意：** SFT 和 GRPO 示例各自从初始模型开始。把 SFT 结果接到 GRPO 需要手动处理权重和状态转换，示例没有自动完成这一步。

---

## 5. 单独评估

例如评估 GRPO 第 4 步：

```bash
.venv/bin/python scripts/evaluate_gsm8k.py \
  --checkpoint /path/to/new-grpo/baseline-checkpoints/step_00000004 \
  --output-dir /path/to/new-evaluation --subset train-dev --size 4 \
  --dev-size 500 --dev-seed 0 --batch-size 4 \
  --max-prompt-len 512 --max-new-tokens 256 --reward-workers 2
```

输出包含逐题回答、标准答案、评分、生成长度和 token ID。也支持初始模型和 MATH 数据，详见 `--help`。

---

## 6. 测试与验证范围

安装 `.[hf,dev]` 后，可以在 CPU 上运行核心测试：

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  .venv/bin/python -m pytest -m 'not tpu and not full_model'
```

四个 CPU 虚拟设备用于检查分片规则和设备间数据一致性，不能代表 TPU 性能。部分 Linux 进程测试在其他系统上跳过；要求恰好两个设备的 RNG 测试可单独运行：

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  .venv/bin/python -m pytest tests/test_inference_rng.py
```

本批迁移后的 ICI、BF16 内容与分块保存已在真实四芯 TPU v4 上通过 59 项数组检查。它们覆盖传输内容、浮点转换、分片方向、保存字节和失败条件，不加载 E2B。运行相同检查：

```bash
JAX_PLATFORMS=tpu,cpu .venv/bin/python -m pytest \
  tests/test_inference_device.py tests/test_checkpoint_chunks.py
```

完整 E2B 训练、保存与独立续训应另外运行前面的恢复示例。已有历史实验不自动覆盖迁移后的代码，尤其非零 KL 的训练需要按当前数学实现重新验证。
