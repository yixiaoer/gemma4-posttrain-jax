"""Hugging Face checkpoint <-> :class:`Gemma4TextParams` conversion.

Only the text decoder is converted. Vision/audio towers, the dead ``k_proj``/``v_proj``/``k_norm``
weights that the checkpoint still stores for kv-shared layers, and everything else outside the
language model are skipped, mirroring what ``Gemma4TextModel`` loads.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .model import (
    AttentionParams,
    DecoderLayerParams,
    Gemma4TextConfig,
    Gemma4TextParams,
    MLPParams,
    PerLayerInputParams,
    RMSNormParams,
    config_from_hf,
    is_kv_shared,
    layer_has_v_proj,
    layer_head_dim,
    layer_intermediate_size,
    layer_kv_heads,
)

COMPOSITE_PREFIX = "model.language_model."  # Gemma4ForConditionalGeneration checkpoints
CAUSAL_LM_PREFIX = "model."  # Gemma4ForCausalLM state dicts


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):  # torch.Tensor, possibly bf16
        import torch

        return np.asarray(value.detach().to(torch.float32).cpu().numpy())
    return np.asarray(value)


def _norm(get: Callable[[str], np.ndarray], key: str, dtype: Any) -> RMSNormParams:
    return RMSNormParams(jnp.asarray(get(key), dtype))


def _attention(
    get: Callable[[str], np.ndarray], key: str, config: Gemma4TextConfig, layer_index: int, dtype: Any
) -> AttentionParams:
    M = config.hidden_size
    D = layer_head_dim(config, layer_index)
    H = layer_kv_heads(config, layer_index)
    R = config.num_attention_heads // H
    q = get(f"{key}q_proj.weight").T.reshape(M, H, R, D).transpose(0, 2, 1, 3)  # HF q head index = h * R + r
    o = get(f"{key}o_proj.weight").T.reshape(H, R, D, M).transpose(1, 0, 2, 3)
    q_norm = _norm(get, f"{key}q_norm.weight", dtype)
    if is_kv_shared(config, layer_index):
        return AttentionParams(jnp.asarray(q, dtype), None, None, jnp.asarray(o, dtype), q_norm, None)
    k = get(f"{key}k_proj.weight").T.reshape(M, H, D)
    v = get(f"{key}v_proj.weight").T.reshape(M, H, D) if layer_has_v_proj(config, layer_index) else None
    return AttentionParams(
        jnp.asarray(q, dtype),
        jnp.asarray(k, dtype),
        None if v is None else jnp.asarray(v, dtype),
        jnp.asarray(o, dtype),
        q_norm,
        _norm(get, f"{key}k_norm.weight", dtype),
    )


def _decoder_layer(
    get: Callable[[str], np.ndarray], key: str, config: Gemma4TextConfig, layer_index: int, dtype: Any
) -> DecoderLayerParams:
    mlp = MLPParams(
        jnp.asarray(get(f"{key}mlp.gate_proj.weight").T, dtype),
        jnp.asarray(get(f"{key}mlp.up_proj.weight").T, dtype),
        jnp.asarray(get(f"{key}mlp.down_proj.weight").T, dtype),
    )
    per_layer_input = None
    if config.hidden_size_per_layer_input:
        per_layer_input = PerLayerInputParams(
            jnp.asarray(get(f"{key}per_layer_input_gate.weight").T, dtype),
            jnp.asarray(get(f"{key}per_layer_projection.weight").T, dtype),
            _norm(get, f"{key}post_per_layer_input_norm.weight", dtype),
        )
    return DecoderLayerParams(
        input_norm=_norm(get, f"{key}input_layernorm.weight", dtype),
        attention=_attention(get, f"{key}self_attn.", config, layer_index, dtype),
        post_attention_norm=_norm(get, f"{key}post_attention_layernorm.weight", dtype),
        pre_feedforward_norm=_norm(get, f"{key}pre_feedforward_layernorm.weight", dtype),
        mlp=mlp,
        post_feedforward_norm=_norm(get, f"{key}post_feedforward_layernorm.weight", dtype),
        per_layer_input=per_layer_input,
        layer_scalar=jnp.asarray(get(f"{key}layer_scalar").reshape(()), dtype),
    )


def convert_gemma4_text_params(
    get_tensor: Callable[[str], Any],
    config: Gemma4TextConfig,
    *,
    prefix: str = CAUSAL_LM_PREFIX,
    dtype: Any = jnp.float32,
) -> Gemma4TextParams:
    """Build params from a ``key -> tensor`` accessor (state dict ``__getitem__`` or safetensors)."""

    def get(key: str) -> np.ndarray:
        return _to_numpy(get_tensor(prefix + key))

    L, P = config.num_hidden_layers, config.hidden_size_per_layer_input
    layers = tuple(_decoder_layer(get, f"layers.{i}.", config, i, dtype) for i in range(L))
    embed_tokens_per_layer = per_layer_model_projection = per_layer_projection_norm = None
    if P:
        embed_tokens_per_layer = jnp.asarray(get("embed_tokens_per_layer.weight").reshape(-1, L, P), dtype)
        per_layer_model_projection = jnp.asarray(
            get("per_layer_model_projection.weight").T.reshape(config.hidden_size, L, P), dtype
        )
        per_layer_projection_norm = _norm(get, "per_layer_projection_norm.weight", dtype)
    return Gemma4TextParams(
        embed_tokens=jnp.asarray(get("embed_tokens.weight"), dtype),
        embed_tokens_per_layer=embed_tokens_per_layer,
        per_layer_model_projection=per_layer_model_projection,
        per_layer_projection_norm=per_layer_projection_norm,
        layers=layers,
        final_norm=_norm(get, "norm.weight", dtype),
    )


def convert_hf_model(model: Any, *, dtype: Any = jnp.float32) -> tuple[Gemma4TextParams, Gemma4TextConfig]:
    """Convert a ``Gemma4ForCausalLM`` (or its ``Gemma4TextModel``) instance."""

    hf_config = getattr(model.config, "text_config", model.config)
    config = config_from_hf(_hf_config_dict(hf_config))
    state = model.state_dict()
    prefix = CAUSAL_LM_PREFIX if any(k.startswith(CAUSAL_LM_PREFIX + "layers.") for k in state) else ""
    return convert_gemma4_text_params(state.__getitem__, config, prefix=prefix, dtype=dtype), config


def _hf_config_dict(hf_config: Any) -> dict[str, Any]:
    d = dict(hf_config.to_dict())
    # ``to_dict`` drops the constructor-only kwargs; recover them from ``per_layer_config``.
    per_layer = getattr(hf_config, "per_layer_config", None) or {}
    for layer_index, layer_type in enumerate(hf_config.layer_types):
        if layer_type != "full_attention":
            continue
        override = per_layer.get(layer_index) or per_layer.get(str(layer_index)) or {}
        if hasattr(override, "to_dict"):
            override = override.to_dict()
        if "head_dim" in override:
            d["global_head_dim"] = override["head_dim"]
        if "num_key_value_heads" in override:
            d["num_global_key_value_heads"] = override["num_key_value_heads"]
        break
    return d


def convert_back_gemma4_text_params(
    params: Gemma4TextParams, config: Gemma4TextConfig, *, prefix: str = CAUSAL_LM_PREFIX
) -> dict[str, np.ndarray]:
    """Inverse of :func:`convert_gemma4_text_params` (float32 numpy, Hugging Face key names)."""

    out: dict[str, np.ndarray] = {}

    def put(key: str, value: Array) -> None:
        out[prefix + key] = np.asarray(jnp.asarray(value, jnp.float32))

    M, L, P = config.hidden_size, config.num_hidden_layers, config.hidden_size_per_layer_input
    put("embed_tokens.weight", params.embed_tokens)
    put("norm.weight", params.final_norm.weight)
    if P:
        assert params.embed_tokens_per_layer is not None and params.per_layer_model_projection is not None
        assert params.per_layer_projection_norm is not None
        put("embed_tokens_per_layer.weight", params.embed_tokens_per_layer.reshape(-1, L * P))
        put("per_layer_model_projection.weight", params.per_layer_model_projection.reshape(M, L * P).T)
        put("per_layer_projection_norm.weight", params.per_layer_projection_norm.weight)
    for i, layer in enumerate(params.layers):
        key = f"layers.{i}."
        a = layer.attention
        D, H = layer_head_dim(config, i), layer_kv_heads(config, i)
        R = config.num_attention_heads // H
        put(f"{key}self_attn.q_proj.weight", a.q_proj.transpose(0, 2, 1, 3).reshape(M, H * R * D).T)
        put(f"{key}self_attn.o_proj.weight", a.o_proj.transpose(1, 0, 2, 3).reshape(H * R * D, M).T)
        put(f"{key}self_attn.q_norm.weight", a.q_norm.weight)
        if a.k_proj is not None:
            put(f"{key}self_attn.k_proj.weight", a.k_proj.reshape(M, H * D).T)
            assert a.k_norm is not None
            put(f"{key}self_attn.k_norm.weight", a.k_norm.weight)
        if a.v_proj is not None:
            put(f"{key}self_attn.v_proj.weight", a.v_proj.reshape(M, H * D).T)
        put(f"{key}mlp.gate_proj.weight", layer.mlp.gate_proj.T)
        put(f"{key}mlp.up_proj.weight", layer.mlp.up_proj.T)
        put(f"{key}mlp.down_proj.weight", layer.mlp.down_proj.T)
        put(f"{key}input_layernorm.weight", layer.input_norm.weight)
        put(f"{key}post_attention_layernorm.weight", layer.post_attention_norm.weight)
        put(f"{key}pre_feedforward_layernorm.weight", layer.pre_feedforward_norm.weight)
        put(f"{key}post_feedforward_layernorm.weight", layer.post_feedforward_norm.weight)
        put(f"{key}layer_scalar", layer.layer_scalar.reshape(1))
        if layer.per_layer_input is not None:
            put(f"{key}per_layer_input_gate.weight", layer.per_layer_input.gate.T)
            put(f"{key}per_layer_projection.weight", layer.per_layer_input.projection.T)
            put(f"{key}post_per_layer_input_norm.weight", layer.per_layer_input.post_norm.weight)
    return out


def check_gemma4_text_params(params: Gemma4TextParams, config: Gemma4TextConfig) -> None:
    """Assert every leaf has the shape implied by ``config``."""

    M, V, L, P = config.hidden_size, config.vocab_size, config.num_hidden_layers, config.hidden_size_per_layer_input
    assert params.embed_tokens.shape == (V, M)
    assert params.final_norm.weight.shape == (M,)
    if P:
        assert params.embed_tokens_per_layer is not None and params.embed_tokens_per_layer.shape == (
            config.vocab_size_per_layer_input,
            L,
            P,
        )
        assert params.per_layer_model_projection is not None and params.per_layer_model_projection.shape == (M, L, P)
        assert params.per_layer_projection_norm is not None and params.per_layer_projection_norm.weight.shape == (P,)
    else:
        assert params.embed_tokens_per_layer is None and params.per_layer_model_projection is None
    assert len(params.layers) == L
    for i, layer in enumerate(params.layers):
        D, H, F = layer_head_dim(config, i), layer_kv_heads(config, i), layer_intermediate_size(config, i)
        R = config.num_attention_heads // H
        a = layer.attention
        assert a.q_proj.shape == (M, R, H, D), (i, a.q_proj.shape)
        assert a.o_proj.shape == (R, H, D, M), (i, a.o_proj.shape)
        assert a.q_norm.weight.shape == (D,)
        if is_kv_shared(config, i):
            assert a.k_proj is None and a.v_proj is None and a.k_norm is None
        else:
            assert a.k_proj is not None and a.k_proj.shape == (M, H, D)
            assert a.k_norm is not None and a.k_norm.weight.shape == (D,)
            assert (a.v_proj is None) == (not layer_has_v_proj(config, i))
            if a.v_proj is not None:
                assert a.v_proj.shape == (M, H, D)
        assert (
            layer.mlp.gate_proj.shape == (M, F)
            and layer.mlp.up_proj.shape == (M, F)
            and layer.mlp.down_proj.shape == (F, M)
        )
        assert layer.layer_scalar.shape == ()
        for norm in (
            layer.input_norm,
            layer.post_attention_norm,
            layer.pre_feedforward_norm,
            layer.post_feedforward_norm,
        ):
            assert norm.weight.shape == (M,)
        if P:
            assert layer.per_layer_input is not None
            assert layer.per_layer_input.gate.shape == (M, P) and layer.per_layer_input.projection.shape == (P, M)
            assert layer.per_layer_input.post_norm.weight.shape == (M,)
        else:
            assert layer.per_layer_input is None


def _safetensor_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(path, name) for name in files]
    single = os.path.join(path, "model.safetensors")
    if os.path.exists(single):
        return [single]
    raise FileNotFoundError(f"no safetensors found under {path}")


def load_hf_params(
    path: str,
    *,
    dtype: Any = jnp.bfloat16,
    config: Gemma4TextConfig | None = None,
    device: Any = None,
) -> tuple[Gemma4TextParams, Gemma4TextConfig]:
    """Load a Hugging Face snapshot directory (config.json + safetensors) into params.

    Arrays are created on ``device`` (default: the first CPU device) so multi-GB checkpoints never
    touch the accelerator before being sharded explicitly.
    """

    from safetensors import safe_open

    if config is None:
        with open(os.path.join(path, "config.json")) as f:
            config = config_from_hf(json.load(f))
    handles = [safe_open(name, framework="pt") for name in _safetensor_files(path)]
    key_to_handle = {key: h for h in handles for key in list(h.keys())}
    prefix = COMPOSITE_PREFIX if any(k.startswith(COMPOSITE_PREFIX) for k in key_to_handle) else CAUSAL_LM_PREFIX

    def get_tensor(key: str) -> Any:
        return key_to_handle[key].get_tensor(key)

    device = jax.devices("cpu")[0] if device is None else device
    with jax.default_device(device):
        params = convert_gemma4_text_params(get_tensor, config, prefix=prefix, dtype=dtype)
    return params, config


def load_hf_eos_token_ids(path: str, config: Gemma4TextConfig) -> tuple[int, ...]:
    """EOS ids used by ``generate``: ``generation_config.json`` (e.g. ``<eos>``, ``<turn|>``) or the model's."""

    gen_path = os.path.join(path, "generation_config.json")
    if not os.path.exists(gen_path):
        return (config.eos_token_id,)
    with open(gen_path) as f:
        eos = json.load(f).get("eos_token_id", config.eos_token_id)
    return tuple(eos) if isinstance(eos, list) else (int(eos),)


def iter_language_model_keys(keys: Iterable[str], prefix: str = COMPOSITE_PREFIX) -> list[str]:
    return sorted(k for k in keys if k.startswith(prefix))
