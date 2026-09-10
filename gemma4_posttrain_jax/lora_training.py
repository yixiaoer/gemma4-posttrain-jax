"""LoRA训练的冻结基础模型身份与小型Adam；复用原GRPO/完整状态格式。"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import jax
import optax

from .lora import Gemma4LoRAParams, LoRAConfig, check_lora_config
from .model import Gemma4TextParams


class LoRAPolicyParams(NamedTuple):
    frozen_base: Gemma4TextParams
    adapters_f32: Gemma4LoRAParams


def make_lora_optimizer(learning_rate: float, *, max_grad_norm: float = 1.0) -> optax.GradientTransformation:
    if not all(math.isfinite(x) and x > 0 for x in (learning_rate, max_grad_norm)):
        raise ValueError("LoRA learning rate和gradient clip必须是正的有限数")
    return optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adamw(learning_rate, weight_decay=0.0))


def lora_base_identity(model_path: Path) -> dict[str, dict[str, str | int]]:
    """保存全文件字节身份；仅路径相同不能证明恢复时冻结base或tokenizer未变。"""
    weights = sorted(model_path.glob("*.safetensors"))
    if not weights or not (model_path / "config.json").is_file():
        raise ValueError("LoRA需要明确的HF config与safetensors基础权重")
    names = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
    ]
    files = [*weights, *(model_path / name for name in names if (model_path / name).is_file())]
    result: dict[str, dict[str, str | int]] = {}
    for path in files:
        with path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        result[path.name] = {"size": path.stat().st_size, "sha256": digest}
    return result


def checkpoint_lora_config(run_config: Mapping[str, Any] | None, model_path: Path) -> LoRAConfig | None:
    """独立评估加载前核对LoRA格式、缩放/精度协议及全部冻结base字节。"""
    if run_config is None or "training_mode" not in run_config:
        return None
    expected = {
        "training_mode": "lora-unscaled-fp32-v1",
        "adapter_rng_protocol": "fold-in-seed-0x4c4f5241-v1",
        "sampling_rng_protocol": "fold-in-seed-data-cursor-v1",
        "adapter_sharding": "replicated",
        "rollout_layout": "fsdp",
        "compute_dtype": "bfloat16",
        "master_dtype": "float32",
        "freeze_embeddings": True,
        "beta": 0.0,
        "jax_default_matmul_precision": jax.config.jax_default_matmul_precision or "default",
    }
    if any(run_config.get(k) != v for k, v in expected.items()) or run_config.get("rollout_lag_updates", 0):
        raise ValueError("LoRA checkpoint格式、缩放、布局或精度协议不匹配")
    raw = run_config.get("lora")
    if not isinstance(raw, Mapping) or set(raw) != {"rank", "alpha", "targets"}:
        raise ValueError("LoRA checkpoint缺少完整适配器配置")
    if type(raw["rank"]) is not int or not isinstance(raw["targets"], list):
        raise ValueError("LoRA checkpoint的rank/targets类型不正确")
    config = LoRAConfig(raw["rank"], raw["alpha"], tuple(raw["targets"]))
    check_lora_config(config)
    if run_config.get("frozen_base_identity") != lora_base_identity(model_path):
        raise ValueError("LoRA checkpoint的冻结基础模型或tokenizer字节改变")
    return config
