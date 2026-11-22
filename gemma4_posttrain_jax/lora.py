"""Store attention LoRA adapters separately from frozen base weights; scale and merge them with pure functions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from gemma4_posttrain_jax.model import Gemma4TextConfig, Gemma4TextParams


class LoRAConfig(NamedTuple):
    rank: int = 8
    alpha: float = 16.0
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


class LowRankParams(NamedTuple):
    a: Array  # [input, rank]
    b: Array  # [rank, output]


class AttentionLoRAParams(NamedTuple):
    q_proj: LowRankParams | None
    k_proj: LowRankParams | None
    v_proj: LowRankParams | None
    o_proj: LowRankParams | None


class Gemma4LoRAParams(NamedTuple):
    layers: tuple[AttentionLoRAParams, ...]


def check_lora_config(config: LoRAConfig) -> None:
    if config.rank <= 0 or not math.isfinite(config.alpha) or config.alpha <= 0:
        raise ValueError("LoRA rank和alpha必须是有限正值")
    if not config.targets or len(set(config.targets)) != len(config.targets):
        raise ValueError("LoRA targets必须非空且不能重复")
    if set(config.targets) - set(AttentionLoRAParams._fields):
        raise ValueError("当前LoRA只支持attention的q/k/v/o投影")


def projection_dimensions(weight: Array, name: str) -> tuple[int, int]:
    if name == "o_proj":
        return math.prod(weight.shape[:-1]), weight.shape[-1]
    return weight.shape[0], math.prod(weight.shape[1:])


def init_lora_params(base: Gemma4TextParams, config: LoRAConfig, key: Array) -> Gemma4LoRAParams:
    """Initialize only the adapters, with Gaussian A and zero B, without constructing or replacing base weights."""
    check_lora_config(config)
    layers = []
    for layer in base.layers:
        values: list[LowRankParams | None] = []
        for name in AttentionLoRAParams._fields:
            weight = getattr(layer.attention, name)
            if name not in config.targets or weight is None:
                # Shared-KV layers add no K/V adapters; 12B K=V layers modify only the existing K projection.
                values.append(None)
                continue
            inputs, outputs = projection_dimensions(weight, name)
            key, subkey = jax.random.split(key)
            a = jax.random.normal(subkey, (inputs, config.rank), dtype=jnp.float32) / math.sqrt(inputs)
            values.append(LowRankParams(a, jnp.zeros((config.rank, outputs), dtype=jnp.float32)))
        layers.append(AttentionLoRAParams(*values))
    result = Gemma4LoRAParams(tuple(layers))
    if not jax.tree.leaves(result):
        raise ValueError("所选targets没有对应的实际权重")
    return result


def check_lora_params(base: Gemma4TextParams, params: Gemma4LoRAParams, config: LoRAConfig) -> None:
    check_lora_config(config)
    if len(params.layers) != len(base.layers):
        raise ValueError("LoRA层数与HF基础模型不同")
    for base_layer, layer in zip(base.layers, params.layers, strict=True):
        for name in AttentionLoRAParams._fields:
            weight, adapter = getattr(base_layer.attention, name), getattr(layer, name)
            if (adapter is not None) != (weight is not None and name in config.targets):
                raise ValueError(f"LoRA target存在性错误：{name}")
            if adapter is not None:
                inputs, outputs = projection_dimensions(weight, name)
                if adapter.a.shape != (inputs, config.rank) or adapter.b.shape != (config.rank, outputs):
                    raise ValueError(f"LoRA矩阵shape错误：{name}")
                if adapter.a.dtype != jnp.float32 or adapter.b.dtype != jnp.float32:
                    raise ValueError("LoRA master参数必须是FP32")


def prepare_lora_params(params: Gemma4LoRAParams, config: LoRAConfig) -> Gemma4LoRAParams:
    """Apply alpha/r to B inside the differentiable graph while the optimizer stores unscaled FP32 A and B."""
    check_lora_config(config)
    scale = config.alpha / config.rank
    return Gemma4LoRAParams(
        tuple(
            AttentionLoRAParams(
                *(None if value is None else LowRankParams(value.a, value.b * scale) for value in layer)
            )
            for layer in params.layers
        )
    )


def forward_lora_projection(params: LowRankParams | None, x: Array, base_output: Array) -> Array:
    """Use leading batch/sequence axes and separate base and LoRA projections to avoid rounding small updates into
    BF16.
    """
    if params is None:
        return base_output
    flat = x.astype(params.a.dtype).reshape(x.shape[0], x.shape[1], -1)
    update = ((flat @ params.a) @ params.b).reshape(base_output.shape)
    return (base_output.astype(update.dtype) + update).astype(base_output.dtype)


def merge_lora_params(
    base: Gemma4TextParams, params: Gemma4LoRAParams, config: LoRAConfig, *, dtype: Any = jnp.float32
) -> Gemma4TextParams:
    """Merge (alpha/r)AB for export and return Gemma 4 parameters; assess BF16 export rounding separately."""
    check_lora_params(base, params, config)
    prepared = prepare_lora_params(params, config)
    layers = []
    for base_layer, layer in zip(base.layers, prepared.layers, strict=True):
        replacements = {}
        for name in AttentionLoRAParams._fields:
            adapter = getattr(layer, name)
            if adapter is not None:
                weight = getattr(base_layer.attention, name)
                delta = (adapter.a @ adapter.b).reshape(weight.shape)
                replacements[name] = (weight.astype(jnp.float32) + delta).astype(dtype)
        layers.append(base_layer._replace(attention=base_layer.attention._replace(**replacements)))
    return base._replace(layers=tuple(layers))


def convert_back_lora_params(
    base: Gemma4TextParams, params: Gemma4LoRAParams, lora_config: LoRAConfig, model_config: Gemma4TextConfig
) -> dict[str, Any]:
    """Use the existing HF convert_back path, merging only for export and keeping the training branches separate."""
    from gemma4_posttrain_jax.weights import convert_back_gemma4_text_params

    return convert_back_gemma4_text_params(merge_lora_params(base, params, lora_config), model_config)
