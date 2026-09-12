# 测试方法与运行记录

测试分为 CPU、TPU 和真实模型三类。CPU 测试可以检查公式、数据处理和接口；设备分片、内存和真实模型恢复需要在 TPU 上运行。下面记录实际测试环境、命令和已覆盖的范围。

## 运行测试

使用 Python 3.12，先安装 `.[hf,dev]`。小模型由测试直接构造，无需下载真实权重。部分测试需要两个或四个 CPU 设备，应在启动 Python 前设置设备数量：

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=1 \
  .venv/bin/python -m pytest -q -m 'not tpu and not full_model' \
  --ignore=tests/test_grpo_host_loop.py

JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  .venv/bin/python -m pytest -q \
  tests/test_grpo_host_loop.py tests/test_lora_training.py

JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  .venv/bin/python -m pytest -q \
  tests/test_inference_remote.py tests/test_inference_rng.py
```

进程管理测试使用 Linux `/proc`，在 macOS 上会跳过。`tpu` 标记要求真实 TPU，`full_model` 标记还要求准备模型权重。跳过的测试需要在相应环境中另外执行。

代码格式和类型检查：

```bash
.venv/bin/ruff check gemma4_posttrain_jax scripts tests
.venv/bin/ruff format --check gemma4_posttrain_jax scripts tests
.venv/bin/mypy gemma4_posttrain_jax scripts
```

## 已执行的测试

2026-09-13 在 macOS 上执行了以下测试：

| 环境与分组 | 通过 | 跳过 | 主要内容 |
|---|---:|---:|---|
| 单 CPU | 308 | 10 | 模型、算法、采样、数据和状态保存；另有 19 项设备或真实模型测试未选取 |
| 四 CPU | 18 | 0 | 小模型 GRPO/LoRA 训练、恢复和设备布局 |
| 双 CPU | 10 | 5 | 随机状态传递及可在 macOS 运行的进程测试 |

各组包含重复用例，通过数不能直接相加。单 CPU 跳过项要求指定设备数或 Linux；双 CPU 的五项跳过均要求 Linux。

此前在 2026-09-12 的 Linux 与 TPU 环境中，双 CPU 进程和随机状态测试 15 项通过，四 CPU LoRA 测试 4 项通过，TPU v4 的采样、保存、生成、分片、rollout 布局及 PLE 梯度测试 61 项通过。这些结果对应当时的代码；9 月 13 日的恢复边界修复只进行了 CPU 验证，没有重跑 TPU。

macOS 使用 Python 3.12.13、JAX/jaxlib 0.11.1、NumPy 2.5.3、Optax 0.2.8、safetensors 0.8.0、Torch 2.14.0、Transformers 5.17.0、datasets 5.0.1 和 math-verify 0.9.0。TPU 环境使用 JAX/jaxlib 0.11.1、libtpu 0.0.46、NumPy 2.5.2 和 Transformers 5.16.1。

## 保存和恢复检查

已修正并测试以下边界情况：传输在等待完成时失败，64 位状态因 JAX 设置而被降成 32 位，空数组的形状和类型检查，以及恢复记录两边同时缺少字段。恢复比较还会检查评估是否完成、是否加载末步模型、题号是否重复，以及采样设置和源码是否一致。

2026-09-12 的真实 E2B 实验完成了 greedy 生成、四步 SFT，以及 GRPO 保存和恢复示例。GRPO 连续训练到第 4 步，再用新进程从第 2 步恢复到第 4 步，最后分别重载两个模型进行评估。

两次训练的最终状态文件完全相同，1,548 个数组均为有限值，Adam 状态非零。恢复期间的九批、72 行生成数据，以及随机状态、概率、mask 和参数变化记录一致。四道训练集以外的题目，预测、实际 token 重新解码和评分也一致。结束后没有遗留 TPU 持有进程。

2026-09-13 加强比较规则后，此前保存的九批生成记录和四题评估仍通过检查。该结果覆盖指定 E2B 配置的保存和继续训练，四题评估仅用于检查恢复后的模型能否正确工作。

## 安装和实验入口

已构建并安装普通 wheel，在仓库之外导入全部 28 个模块，确认使用的是安装包。只安装核心依赖的环境不含 Torch、Transformers、datasets、W&B 或推理引擎，也通过了模块导入和三个小型实验。

三个实验分别检查采样精度、词表并行和 BF16 权重检查：六个采样配置完成旧版与当前函数对照；四设备词表并行的 embedding/hidden 梯度相对误差均低于 `3.2e-7`；BF16 检查覆盖全部 65,536 种编码、正负零、NaN、单 bit 改变和 C/F 布局。这些实验均在 CPU 上执行。

当前安装包与源码包使用 MIT，保留 11 个命令入口。格式、类型、模块导入、命令帮助、文档链接和 shell 命令语法均已检查。运行结果与日志写入本地输出目录，文档只保留实验方法和结果说明。

## 尚未覆盖的部分

外部引擎的完整质量和成本比较、故障退出、SGL-JAX 真实模型接入、12B LoRA 训练，以及真正并行的训练和生成尚未全部完成。已有短测试和历史结果不能覆盖这些组合，其他模型、采样设置与硬件也需要分别验证。
