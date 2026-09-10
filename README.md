# Gemma4 Posttrain JAX

在单机TPU v4上进行Gemma 4后训练，包含纯JAX的函数式模型实现、SFT、GRPO家族、原生rollout，以及可选的vLLM tpu 版本的tpu-inference适配。

进行实验开发和逆向的主要模型使用Gemma 4 E2B-it；E4B和LoRA保留已有实现，但还没有对Gemma 4所有模型、算法、后端的不同组合进行完整测试。

## 目录

```text
gemma4_posttrain_jax/   模型、算法、采样、分片与状态管理
scripts/               生成、训练、评估、检查和吞吐测量
tests/                 核心功能及运行接口的回归测试
docs/                  设计说明与实验结果
pyproject.toml         依赖和打包配置
```

package直接位于仓库根目录，导入名为`gemma4_posttrain_jax`。

## 安装

在已配置的TPU v4环境中使用Python 3.12：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[tpu,hf,dev]"
.venv/bin/python scripts/check_env.py
```

CPU开发环境安装`.[hf,dev]`。详细依赖见[pyproject.toml](pyproject.toml)；模型权重和数据集需要另行下载，TPU运行库通过`.[tpu]`安装。模型路径指向已下载的HF snapshot，其中应包含config、tokenizer和safetensors。

## 运行入口

| 脚本 | 用途 |
|---|---|
| [generate.py](scripts/generate.py) | 单设备batch生成，支持greedy、temperature、TopK、Top-p及组合采样；可选与Hugging Face Transformers的生成结果比较 |
| [train_sft.py](scripts/train_sft.py) | FSDP SFT、梯度累积、保存与恢复 |
| [train_grpo.py](scripts/train_grpo.py) | GRPO、Dr.GRPO、DAPO、GSPO-token、RLOO及LoRA训练入口 |
| [evaluate_gsm8k.py](scripts/evaluate_gsm8k.py) | GSM8K或MATH评估，支持初始模型与训练checkpoint |
| [check_checkpoint.py](scripts/check_checkpoint.py) | 检查完整状态、步数、有限性和Adam状态 |
| [check_env.py](scripts/check_env.py) | 检查JAX、TPU设备及基础Pallas环境 |
| [benchmark_rollout.py](scripts/benchmark_rollout.py) | 原生batch rollout的独立进程吞吐测量 |

例如，已有E2B模型可以这样生成：

```bash
.venv/bin/python scripts/generate.py \
  --model /path/to/gemma4-snapshot \
  --chat --prompt "计算3乘4再加2。" --max-new-tokens 64
```

SFT和RL入口均可用`--help`查看配置，训练时显式提供模型路径、数据/形状与新的输出目录：

```bash
.venv/bin/python scripts/train_sft.py --help
.venv/bin/python scripts/train_grpo.py --help
.venv/bin/python scripts/evaluate_gsm8k.py --help
```

`train_grpo.py --algorithm dapo`包含动态补采等行为，`dapo-loss`只选择目标函数。`--updates-per-rollout`表示同一生成batch执行几次Adam更新；prompt题数乘`--group-size`才是展开采样行数，`--microbatch-size`是训练microbatch。

使用`--save-every`启用保存，`--resume`指定完整step目录继续训练。恢复检查包括训练配置、优化器与随机/数据状态；旧checkpoint的格式标识保留，以免仅因项目改名破坏读取。历史引擎checkpoint还包含源码SHA，更名后的跨版本续训兼容性需要单独验证。RL与评估脚本会记录Git版本和实际源码SHA；没有Git HEAD时版本记录为`null`，仍可运行。

## 实现范围

模型和训练使用显式参数树、JAX函数与Optax。默认训练保留FP32 master/Adam，计算通常使用BF16；精度、remat、冻结embedding和分片配置会影响容量与数值行为，具体以运行参数为准。

原生rollout在设备循环中执行batch decode。tpu-inference同进程／独立进程适配仍是实验入口，要求匹配源码中锁定的独立环境与运行库身份，普通安装不能替代该环境。两进程不表示训练与生成已异步重叠；引擎共同配置下的恢复、故障退出、跨后端对齐和正式多seed质量仍有待办。

## 测试

CPU测试需要`hf`与`dev`依赖；tiny模型直接构造，无需下载真实权重。

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q -m 'not tpu and not full_model'
.venv/bin/ruff check gemma4_posttrain_jax scripts tests
.venv/bin/ruff format --check gemma4_posttrain_jax scripts tests
```

TPU sharding测试使用`tpu`标记，真实模型测试使用`full_model`标记，须在相应设备和模型条件下单独运行。当前版本的验证范围与实验结论见[设计与实验结果](docs/development-results.md)。

源码采用[MIT](LICENSE)。模型、tokenizer、数据集和外部依赖的许可分别适用。
