"""Functional JAX implementation of the Gemma 4 text decoder.

Numerical semantics follow transformers 5.16.1 ``models/gemma4/modeling_gemma4.py``
(``Gemma4TextModel`` / ``Gemma4ForCausalLM``). Only the text stack is implemented; the vision
and audio towers are out of scope. Parameters are converted from the official checkpoint by
:mod:`gemma4_posttrain_jax.weights`; there are no random-initialisation functions.

einops axis names used throughout:
``B`` batch, ``S`` query length, ``T`` key length (cache capacity when a cache is used),
``M`` hidden size, ``H`` kv heads, ``R`` query heads per kv head, ``D`` head dim
(``head_dim`` on sliding layers, ``global_head_dim`` on full-attention layers), ``F`` MLP width,
``L`` layers, ``P`` per-layer-input width, ``V`` vocabulary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal, NamedTuple

import einops as op
import jax
import jax.numpy as jnp
from jax import Array, lax

from gemma4_posttrain_jax.lora import AttentionLoRAParams, Gemma4LoRAParams, forward_lora_projection

SLIDING = "sliding_attention"
FULL = "full_attention"


class Gemma4TextConfig(NamedTuple):
    """Static architecture values of the Gemma 4 text decoder (see ``config_from_hf``)."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    global_head_dim: int
    num_global_key_value_heads: int | None
    layer_types: tuple[str, ...]
    sliding_window: int
    num_kv_shared_layers: int
    use_double_wide_mlp: bool
    attention_k_eq_v: bool
    hidden_size_per_layer_input: int
    vocab_size_per_layer_input: int
    rms_norm_eps: float
    rope_theta_sliding: float
    rope_theta_full: float
    full_partial_rotary_factor: float
    final_logit_softcapping: float | None
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int


def _hf_attention_dimensions(text: Mapping[str, Any], layer_types: list[str]) -> tuple[int, int, int, int | None]:
    """把HF稀疏逐层配置还原为本模型支持的两类attention维度；拒绝不能表示的异构覆盖。"""

    head_dim = int(text.get("head_dim", 256))
    kv_heads = int(text["num_key_value_heads"])
    global_dim = int(text.get("global_head_dim", 512))
    global_kv = text.get("num_global_key_value_heads")
    if "per_layer_config" not in text:
        return head_dim, global_dim, kv_heads, global_kv
    raw = text["per_layer_config"]
    overrides: dict[int, Mapping[str, Any]] = {}
    if raw is not None:
        if not isinstance(raw, Mapping):
            raise ValueError("per_layer_config必须是按层索引的映射")
        for key, value in raw.items():
            index = int(key)
            if str(index) != str(key) or index < 0 or index >= len(layer_types) or index in overrides:
                raise ValueError(f"无效或重复的per_layer_config层索引：{key}")
            if not isinstance(value, Mapping) or set(value) - {"head_dim", "num_key_value_heads"}:
                raise ValueError(f"不支持的per_layer_config覆盖：{value}")
            overrides[index] = value
    dimensions: dict[str, set[tuple[int, int]]] = {SLIDING: set(), FULL: set()}
    for index, layer_type in enumerate(layer_types):
        value = overrides.get(index, {})
        dim = int(value.get("head_dim", head_dim))
        heads = int(value.get("num_key_value_heads", kv_heads))
        if dim <= 0 or heads <= 0 or int(text["num_attention_heads"]) % heads:
            raise ValueError("逐层attention维度必须为正，且KV头数须整除query头数")
        dimensions[layer_type].add((dim, heads))
    if any(len(values) > 1 for values in dimensions.values()):
        raise ValueError("同一种 attention 在不同层的维度不一致，当前参数结构不支持这种配置")
    head_dim, kv_heads = next(iter(dimensions[SLIDING]), (head_dim, kv_heads))
    global_dim, full_kv = next(iter(dimensions[FULL]), (head_dim, kv_heads))
    if full_kv != kv_heads and not text.get("attention_k_eq_v", False):
        raise ValueError("当前实现仅在 K=V 的全局 attention 中支持单独设置 KV 头数")
    global_kv = full_kv if full_kv != kv_heads else None
    return head_dim, global_dim, kv_heads, global_kv


def config_from_hf(hf: Mapping[str, Any]) -> Gemma4TextConfig:
    """Build the config from a Hugging Face ``config.json`` dict (composite or text-only).

    Supports legacy global attention fields and the sparse ``per_layer_config`` emitted by
    transformers 5.16.1. Explicit per-layer overrides take precedence as in Hugging Face.
    """

    text = hf.get("text_config", hf)
    num_layers = int(text["num_hidden_layers"])
    layer_types = list(text["layer_types"])
    if len(layer_types) != num_layers:
        raise ValueError(f"layer_types has {len(layer_types)} entries for {num_layers} layers")
    layer_types[-1] = FULL
    head_dim, global_dim, kv_heads, global_kv = _hf_attention_dimensions(text, layer_types)
    rope = text.get("rope_parameters") or {}
    sliding_rope = rope.get(SLIDING) or {}
    full_rope = rope.get(FULL) or {}
    if (
        sliding_rope.get("rope_type", "default") != "default"
        or full_rope.get("rope_type", "proportional") != "proportional"
    ):
        raise ValueError(f"unsupported rope_parameters: {rope}")
    return Gemma4TextConfig(
        vocab_size=int(text["vocab_size"]),
        hidden_size=int(text["hidden_size"]),
        intermediate_size=int(text["intermediate_size"]),
        num_hidden_layers=num_layers,
        num_attention_heads=int(text["num_attention_heads"]),
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        global_head_dim=global_dim,
        num_global_key_value_heads=global_kv,
        layer_types=tuple(layer_types),
        sliding_window=int(text.get("sliding_window", 512)),
        num_kv_shared_layers=int(text.get("num_kv_shared_layers", 0)),
        use_double_wide_mlp=bool(text.get("use_double_wide_mlp", False)),
        attention_k_eq_v=bool(text.get("attention_k_eq_v", False)),
        hidden_size_per_layer_input=int(text.get("hidden_size_per_layer_input", 256)),
        vocab_size_per_layer_input=int(text.get("vocab_size_per_layer_input", 262144)),
        rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
        rope_theta_sliding=float(sliding_rope.get("rope_theta", 10_000.0)),
        rope_theta_full=float(full_rope.get("rope_theta", 1_000_000.0)),
        full_partial_rotary_factor=float(full_rope.get("partial_rotary_factor", 0.25)),
        final_logit_softcapping=text.get("final_logit_softcapping"),
        bos_token_id=int(text.get("bos_token_id", 2)),
        eos_token_id=int(text.get("eos_token_id", 1)),
        pad_token_id=int(text.get("pad_token_id", 0)),
    )


# --- static per-layer derivations ---------------------------------------------------------------


def is_kv_shared(config: Gemma4TextConfig, layer_index: int) -> bool:
    first_shared = config.num_hidden_layers - config.num_kv_shared_layers
    return config.num_kv_shared_layers > 0 and layer_index >= first_shared


def kv_source_layer(config: Gemma4TextConfig, layer_index: int) -> int:
    """Index of the last non-shared layer of the same type; the shared layer reuses its K/V."""

    if not is_kv_shared(config, layer_index):
        return layer_index
    first_shared = config.num_hidden_layers - config.num_kv_shared_layers
    layer_type = config.layer_types[layer_index]
    for j in range(first_shared - 1, -1, -1):
        if config.layer_types[j] == layer_type:
            return j
    raise ValueError(f"no non-shared {layer_type} layer before layer {layer_index}")


def layer_head_dim(config: Gemma4TextConfig, layer_index: int) -> int:
    return config.global_head_dim if config.layer_types[layer_index] == FULL else config.head_dim


def layer_kv_heads(config: Gemma4TextConfig, layer_index: int) -> int:
    if config.layer_types[layer_index] == FULL and config.attention_k_eq_v and config.num_global_key_value_heads:
        return int(config.num_global_key_value_heads)
    return config.num_key_value_heads


def layer_has_v_proj(config: Gemma4TextConfig, layer_index: int) -> bool:
    return not (config.attention_k_eq_v and config.layer_types[layer_index] == FULL)


def layer_intermediate_size(config: Gemma4TextConfig, layer_index: int) -> int:
    double = config.use_double_wide_mlp and is_kv_shared(config, layer_index)
    return config.intermediate_size * (2 if double else 1)


# --- parameters -----------------------------------------------------------------------------------


class RMSNormParams(NamedTuple):
    weight: Array  # [dim]


class AttentionParams(NamedTuple):
    q_proj: Array  # [M, R, H, D]
    k_proj: Array | None  # [M, H, D]; None on kv-shared layers
    v_proj: Array | None  # [M, H, D]; None on kv-shared layers and on K=V layers
    o_proj: Array  # [R, H, D, M]
    q_norm: RMSNormParams  # [D]
    k_norm: RMSNormParams | None  # [D]; None on kv-shared layers


class MLPParams(NamedTuple):
    gate_proj: Array  # [M, F]
    up_proj: Array  # [M, F]
    down_proj: Array  # [F, M]


class PerLayerInputParams(NamedTuple):
    gate: Array  # [M, P]
    projection: Array  # [P, M]
    post_norm: RMSNormParams  # [M]


class DecoderLayerParams(NamedTuple):
    input_norm: RMSNormParams
    attention: AttentionParams
    post_attention_norm: RMSNormParams
    pre_feedforward_norm: RMSNormParams
    mlp: MLPParams
    post_feedforward_norm: RMSNormParams
    per_layer_input: PerLayerInputParams | None
    layer_scalar: Array  # [] residual-stream scale applied at the end of the layer


class Gemma4TextParams(NamedTuple):
    embed_tokens: Array  # [V, M]; also the (tied) LM head
    embed_tokens_per_layer: Array | None  # [Vp, L, P]
    per_layer_model_projection: Array | None  # [M, L, P]
    per_layer_projection_norm: RMSNormParams | None  # [P]
    layers: tuple[DecoderLayerParams, ...]
    final_norm: RMSNormParams  # [M]


class LayerKVCache(NamedTuple):
    key: Array  # [B, C, H, D]
    value: Array  # [B, C, H, D]


class KVCache(NamedTuple):
    """Static-shape decode cache; ``layers[i]`` is ``None`` on KV-shared layers.

    Full-attention layers use the full generation capacity. Sliding-attention layers use a
    circular buffer with only ``min(sliding_window, capacity)`` physical slots. ``key_mask``
    deliberately remains full-capacity: it records which absolute positions are real tokens,
    while a sliding layer maps its circular-buffer slots back to those positions. This cache
    data structure is unrelated to the distributed algorithm called Ring Attention.
    """

    layers: tuple[LayerKVCache | None, ...]
    key_mask: Array  # [B, C] bool; True where a real (non-pad) token has been written
    length: Array  # [] int32; number of positions written so far


class RotaryValues(NamedTuple):
    cos: Array  # [B, S, D] float32
    sin: Array  # [B, S, D] float32


class Gemma4Output(NamedTuple):
    logits: Array  # [B, S', V]
    hidden: Array  # [B, S, M] after the final norm
    kv_cache: KVCache | None


# --- primitives -----------------------------------------------------------------------------------


def _scale(x: Array, factor: float) -> Array:
    """Multiply by a Python float the way torch does for bf16 tensors (f32 opmath, then round)."""

    return (x.astype(jnp.float32) * factor).astype(x.dtype)


def forward_rms_norm(params: RMSNormParams | None, x: Array, eps: float) -> Array:
    """``Gemma4RMSNorm``: f32 statistics, ``x * (mean(x^2) + eps)^-0.5 * w``; ``None`` means no scale."""

    x32 = x.astype(jnp.float32)
    y = x32 * jnp.power(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + eps, -0.5)
    if params is not None:
        y = y * params.weight.astype(jnp.float32)
    return y.astype(x.dtype)


def _rotary_from_inv_freq(position_ids: Array, inv_freq: Array) -> RotaryValues:
    freqs = position_ids.astype(jnp.float32)[..., None] * inv_freq[None, None, :]  # [B, S, D/2]
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return RotaryValues(jnp.cos(emb), jnp.sin(emb))


def make_rotary_values(config: Gemma4TextConfig, position_ids: Array) -> dict[str, RotaryValues]:
    """cos/sin for both layer types; sliding layers use default RoPE, full layers ``proportional``."""

    dim = config.head_dim
    inv_sliding = 1.0 / (config.rope_theta_sliding ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    dim = config.global_head_dim
    rope_angles = int(config.full_partial_rotary_factor * dim // 2)
    inv_rotated = 1.0 / (config.rope_theta_full ** (jnp.arange(0, 2 * rope_angles, 2, dtype=jnp.float32) / dim))
    inv_full = jnp.concatenate([inv_rotated, jnp.zeros(dim // 2 - rope_angles, jnp.float32)])
    return {
        SLIDING: _rotary_from_inv_freq(position_ids, inv_sliding),
        FULL: _rotary_from_inv_freq(position_ids, inv_full),
    }


def forward_rotary_embedding(x: Array, rotary_values: RotaryValues) -> Array:
    """``x * cos + rotate_half(x) * sin`` on the last axis; ``x`` is ``[B, S, ..., D]``."""

    shape = x.shape[:2] + (1,) * (x.ndim - 3) + x.shape[-1:]
    cos = rotary_values.cos.reshape(shape).astype(x.dtype)
    sin = rotary_values.sin.reshape(shape).astype(x.dtype)
    half = x.shape[-1] // 2
    rotated = jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rotated * sin


def make_qk_masks(
    key_mask: Array, sliding_window: int, *, query_start: Array | int, num_queries: int
) -> dict[str, Array]:
    """Boolean ``[B, 1, 1, S, T]`` masks for both layer types.

    Query ``s`` sits at absolute index ``query_start + s``; keys at ``0..T-1``. Following the Hugging
    Face mask functions, causal is ``k <= q`` and the sliding overlay is ``k > q - sliding_window``;
    padding indices count towards the window.
    """

    T = key_mask.shape[1]
    q_idx = (jnp.arange(num_queries) + query_start)[:, None]
    k_idx = jnp.arange(T)[None, :]
    full = (k_idx <= q_idx)[None] & key_mask[:, None, :]
    sliding = full & (k_idx > q_idx - sliding_window)[None]
    return {FULL: full[:, None, None], SLIDING: sliding[:, None, None]}


def make_cached_qk_masks(
    key_mask: Array, sliding_window: int, *, query_start: Array, num_queries: int
) -> dict[str, Array]:
    """Masks for a full-capacity global cache and a sliding-window circular buffer.

    The append path is used for one-token decode. At absolute length ``n``, circular-buffer slot
    ``r`` contains the greatest position ``p <= n`` for which ``p % window == r``. Attention is
    order-independent along the key axis, so the physical slot order need not be chronological.
    """

    capacity = key_mask.shape[1]
    window_capacity = min(sliding_window, capacity)
    q_abs = (jnp.arange(num_queries, dtype=jnp.int32) + query_start)[:, None]

    full_k_abs = jnp.arange(capacity, dtype=jnp.int32)[None, :]
    full = (full_k_abs <= q_abs)[None] & key_mask[:, None, :]

    last_written = query_start + num_queries - 1
    slots = jnp.arange(window_capacity, dtype=jnp.int32)
    sliding_k_abs = slots + jnp.floor_divide(last_written - slots, window_capacity) * window_capacity
    sliding_valid = sliding_k_abs >= 0
    gather_abs = jnp.clip(sliding_k_abs, 0, capacity - 1)
    sliding_real = jnp.take(key_mask, gather_abs, axis=1) & sliding_valid[None, :]
    sliding = (
        (sliding_k_abs[None, :] <= q_abs) & (sliding_k_abs[None, :] > q_abs - sliding_window) & sliding_valid[None, :]
    )
    sliding = sliding[None] & sliding_real[:, None, :]
    return {FULL: full[:, None, None], SLIDING: sliding[:, None, None]}


def _update_layer_cache(cache: LayerKVCache, key: Array, value: Array, *, start: Array, circular: bool) -> LayerKVCache:
    """Write a prefill chunk or one decode token into a static layer cache."""

    key = key.astype(cache.key.dtype)
    value = value.astype(cache.value.dtype)
    if not circular:
        index = (0, start, 0, 0)
        return LayerKVCache(
            lax.dynamic_update_slice(cache.key, key, index),
            lax.dynamic_update_slice(cache.value, value, index),
        )

    window_capacity = cache.key.shape[1]
    # A prompt may be longer than the circular buffer. Only its final ``window_capacity`` tokens survive,
    # and those indices are unique modulo the buffer size, avoiding scatter semantics with duplicate destinations.
    keep = min(key.shape[1], window_capacity)
    source_start = key.shape[1] - keep
    absolute = jnp.arange(source_start, key.shape[1], dtype=jnp.int32) + start
    slots = jnp.mod(absolute, window_capacity)
    return LayerKVCache(
        cache.key.at[:, slots].set(key[:, source_start:]),
        cache.value.at[:, slots].set(value[:, source_start:]),
    )


def forward_mlp(params: MLPParams, x: Array) -> Array:
    """``down(gelu_tanh(gate(x)) * up(x))``."""

    return (jax.nn.gelu(x @ params.gate_proj, approximate=True) * (x @ params.up_proj)) @ params.down_proj


@jax.custom_jvp
def _lookup_per_layer_tokens(table: Array, input_ids: Array) -> Array:
    """保持三维前向与切线查表，以局部屏障隔离错误scatter外移。"""
    return jnp.take(table, input_ids, axis=0)


@_lookup_per_layer_tokens.defjvp
def _lookup_per_layer_tokens_jvp(primals: tuple[Array, Array], tangents: tuple[Array, Array]) -> tuple[Array, Array]:
    table, input_ids = primals
    table_tangent, _ = tangents
    # 转置后的屏障隔离scatter与微批carry，保持原来的三维查表布局。
    tangent = jnp.take(jax.lax.optimization_barrier(table_tangent), input_ids, axis=0, fill_value=0)
    return _lookup_per_layer_tokens(table, input_ids), tangent


def forward_per_layer_inputs(
    params: Gemma4TextParams, config: Gemma4TextConfig, input_ids: Array, inputs_embeds: Array
) -> Array:
    """Per-layer embeddings ``[B, S, L, P]`` = ``(rmsnorm(proj(x) / sqrt(M)) + table[ids] * sqrt(P)) / sqrt(2)``."""

    assert params.embed_tokens_per_layer is not None and params.per_layer_model_projection is not None
    P = config.hidden_size_per_layer_input
    token_part = _lookup_per_layer_tokens(params.embed_tokens_per_layer, input_ids)
    token_part = token_part * jnp.asarray(math.sqrt(P), token_part.dtype)
    context_part = op.einsum(inputs_embeds, params.per_layer_model_projection, "B S M, M L P -> B S L P")
    context_part = _scale(context_part, config.hidden_size**-0.5)
    context_part = forward_rms_norm(params.per_layer_projection_norm, context_part, config.rms_norm_eps)
    return _scale(context_part + token_part, 2.0**-0.5)


def forward_per_layer_input(params: PerLayerInputParams, seq: Array, per_layer_input: Array, eps: float) -> Array:
    h = jax.nn.gelu(seq @ params.gate, approximate=True) * per_layer_input
    h = h @ params.projection
    return forward_rms_norm(params.post_norm, h, eps)


# --- attention / decoder layer / model --------------------------------------------------------------


def forward_attention(
    params: AttentionParams,
    seq: Array,
    qk_mask: Array,
    rotary_values: RotaryValues,
    *,
    layer_index: int,
    config: Gemma4TextConfig,
    shared_kv: dict[str, tuple[Array, Array]],
    kv_cache: LayerKVCache | None = None,
    cache_length: Array | None = None,
    cache_mode: Literal["prefill", "append"] = "prefill",
    lora: AttentionLoRAParams | None = None,
) -> tuple[Array, LayerKVCache | None]:
    """Gemma 4 attention with q/k/v norms, RoPE, scaling 1.0, KV sharing and an optional static cache.

    ``shared_kv`` is filled by non-shared layers and read by shared layers, keyed by layer type.
    With a cache, new keys are written at ``cache_length``. ``prefill`` attends directly to the
    complete prompt K/V before retaining only the sliding-window tail; ``append`` attends to the
    updated full/circular cache and is intentionally the one-token decode path.
    """

    layer_type = config.layer_types[layer_index]
    eps = config.rms_norm_eps

    q = op.einsum(seq, params.q_proj, "B S M, M R H D -> B S R H D")
    q = forward_lora_projection(None if lora is None else lora.q_proj, seq, q)
    q = forward_rms_norm(params.q_norm, q, eps)
    q = forward_rotary_embedding(q, rotary_values)

    new_cache: LayerKVCache | None = None
    if is_kv_shared(config, layer_index):
        k, v = shared_kv[layer_type]
    else:
        assert params.k_proj is not None
        k = op.einsum(seq, params.k_proj, "B T M, M H D -> B T H D")
        k = forward_lora_projection(None if lora is None else lora.k_proj, seq, k)
        if params.v_proj is not None:
            v = op.einsum(seq, params.v_proj, "B T M, M H D -> B T H D")
            v = forward_lora_projection(None if lora is None else lora.v_proj, seq, v)
        else:
            v = k
        k = forward_rms_norm(params.k_norm, k, eps)
        k = forward_rotary_embedding(k, rotary_values)
        v = forward_rms_norm(None, v, eps)
        if kv_cache is not None:
            assert cache_length is not None
            new_cache = _update_layer_cache(
                kv_cache,
                k,
                v,
                start=cache_length,
                circular=layer_type == SLIDING,
            )
            if cache_mode == "append":
                k, v = new_cache
        shared_kv[layer_type] = (k, v)

    scores = op.einsum(q, k, "B S R H D, B T H D -> B R H S T")  # scaling is 1.0 in Gemma 4
    scores = jnp.where(qk_mask, scores, jnp.finfo(scores.dtype).min)
    probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q.dtype)
    out = op.einsum(probs, v, "B R H S T, B T H D -> B S R H D")
    projected = op.einsum(out, params.o_proj, "B S R H D, R H D M -> B S M")
    out = forward_lora_projection(None if lora is None else lora.o_proj, out, projected)
    return out, new_cache


def forward_decoder_layer(
    params: DecoderLayerParams,
    seq: Array,
    per_layer_input: Array | None,
    qk_masks: Mapping[str, Array],
    rotary_values: Mapping[str, RotaryValues],
    *,
    layer_index: int,
    config: Gemma4TextConfig,
    shared_kv: dict[str, tuple[Array, Array]],
    kv_cache: LayerKVCache | None = None,
    cache_length: Array | None = None,
    cache_mode: Literal["prefill", "append"] = "prefill",
    lora: AttentionLoRAParams | None = None,
) -> tuple[Array, LayerKVCache | None]:
    layer_type = config.layer_types[layer_index]
    eps = config.rms_norm_eps

    with jax.named_scope(f"layer_{layer_index:02d}/attention"):
        residual = seq
        h = forward_rms_norm(params.input_norm, seq, eps)
        h, new_cache = forward_attention(
            params.attention,
            h,
            qk_masks[layer_type],
            rotary_values[layer_type],
            layer_index=layer_index,
            config=config,
            shared_kv=shared_kv,
            kv_cache=kv_cache,
            cache_length=cache_length,
            cache_mode=cache_mode,
            lora=lora,
        )
        h = forward_rms_norm(params.post_attention_norm, h, eps)
        seq = residual + h

    with jax.named_scope(f"layer_{layer_index:02d}/feed_forward"):
        residual = seq
        h = forward_rms_norm(params.pre_feedforward_norm, seq, eps)
        h = forward_mlp(params.mlp, h)
        h = forward_rms_norm(params.post_feedforward_norm, h, eps)
        seq = residual + h

    if params.per_layer_input is not None:
        assert per_layer_input is not None
        with jax.named_scope(f"layer_{layer_index:02d}/per_layer_input"):
            seq = seq + forward_per_layer_input(params.per_layer_input, seq, per_layer_input, eps)

    return seq * params.layer_scalar.astype(seq.dtype), new_cache


def init_kv_cache(config: Gemma4TextConfig, batch_size: int, capacity: int, dtype: Any) -> KVCache:
    """Allocate full caches for global layers and bounded circular buffers for sliding layers."""

    layers: list[LayerKVCache | None] = []
    for i in range(config.num_hidden_layers):
        if is_kv_shared(config, i):
            layers.append(None)
        else:
            layer_capacity = min(config.sliding_window, capacity) if config.layer_types[i] == SLIDING else capacity
            shape = (batch_size, layer_capacity, layer_kv_heads(config, i), layer_head_dim(config, i))
            layers.append(LayerKVCache(jnp.zeros(shape, dtype), jnp.zeros(shape, dtype)))
    return KVCache(tuple(layers), jnp.zeros((batch_size, capacity), jnp.bool_), jnp.zeros((), jnp.int32))


def forward_gemma4_text(
    params: Gemma4TextParams,
    input_ids: Array,
    position_ids: Array,
    *,
    config: Gemma4TextConfig,
    attention_mask: Array | None = None,
    kv_cache: KVCache | None = None,
    cache_mode: Literal["prefill", "append"] = "prefill",
    layer_outputs: list[Array] | None = None,
    remat_layers: bool = False,
    lora: Gemma4LoRAParams | None = None,
) -> tuple[Array, KVCache | None]:
    """Run the decoder and return ``(hidden after final norm, updated cache)``.

    ``attention_mask`` (``[B, S]`` bool, True = real token) covers the new tokens only; with a cache
    the earlier tokens' mask lives in ``kv_cache.key_mask``. A fresh cache must use
    ``cache_mode="prefill"``; subsequent one-token calls use ``"append"``. ``layer_outputs`` is a
    debug hook that collects every decoder layer's output for parity tests.
    """

    batch_size, num_queries = input_ids.shape
    if lora is not None and len(lora.layers) != len(params.layers):
        raise ValueError("LoRA层数与基础模型不一致")
    if attention_mask is None:
        attention_mask = jnp.ones((batch_size, num_queries), jnp.bool_)
    attention_mask = attention_mask.astype(jnp.bool_)

    x = jnp.take(params.embed_tokens, input_ids, axis=0)
    x = x * jnp.asarray(math.sqrt(config.hidden_size), x.dtype)
    per_layer_inputs = (
        forward_per_layer_inputs(params, config, input_ids, x) if config.hidden_size_per_layer_input else None
    )

    if kv_cache is None:
        key_mask = attention_mask
        query_start: Array | int = 0
        cache_length = None
    else:
        key_mask = lax.dynamic_update_slice(kv_cache.key_mask, attention_mask, (0, kv_cache.length))
        query_start = kv_cache.length
        cache_length = kv_cache.length

    if kv_cache is not None and cache_mode == "append":
        if num_queries != 1:
            raise ValueError(f'cache_mode="append" requires one token, got {num_queries}')
        qk_masks = make_cached_qk_masks(
            key_mask, config.sliding_window, query_start=kv_cache.length, num_queries=num_queries
        )
    elif kv_cache is not None:
        # Prefill attention needs all prompt K/V. The circular buffer is populated as a side effect, but using
        # it here would lose early prompt positions before their hidden states are fully computed.
        qk_masks = make_qk_masks(attention_mask, config.sliding_window, query_start=0, num_queries=num_queries)
    else:
        qk_masks = make_qk_masks(key_mask, config.sliding_window, query_start=query_start, num_queries=num_queries)
    rotary_values = make_rotary_values(config, position_ids)
    shared_kv: dict[str, tuple[Array, Array]] = {}
    new_layer_caches: list[LayerKVCache | None] = []
    for i, layer in enumerate(params.layers):
        layer_input = per_layer_inputs[:, :, i] if per_layer_inputs is not None else None
        layer_kv_cache = kv_cache.layers[i] if kv_cache is not None else None

        def run_layer(
            layer_params: DecoderLayerParams,
            seq: Array,
            pli: Array | None,
            shared: dict[str, tuple[Array, Array]],
            layer_lora: AttentionLoRAParams | None,
            layer_index: int = i,
            current_kv_cache: LayerKVCache | None = layer_kv_cache,
        ) -> tuple[Array, LayerKVCache | None, dict[str, tuple[Array, Array]]]:
            # Treat the Python dictionary as an explicit pytree value so checkpointing also retains
            # gradient paths through the K/V tensors consumed by later shared layers.
            updated_shared = dict(shared)
            output, updated_cache = forward_decoder_layer(
                layer_params,
                seq,
                pli,
                qk_masks,
                rotary_values,
                layer_index=layer_index,
                config=config,
                shared_kv=updated_shared,
                kv_cache=current_kv_cache,
                cache_length=cache_length,
                cache_mode=cache_mode,
                lora=layer_lora,
            )
            return output, updated_cache, updated_shared

        # This is a Python layer loop, so keep checkpoint's CSE prevention enabled.  Disabling it is
        # useful for scan bodies, but here XLA can otherwise merge the backward recomputation with
        # the original forward and retain the activations that rematerialization should release.
        layer_fn = jax.checkpoint(run_layer) if remat_layers else run_layer
        x, layer_cache, shared_kv = layer_fn(layer, x, layer_input, shared_kv, None if lora is None else lora.layers[i])
        new_layer_caches.append(layer_cache)
        if layer_outputs is not None:
            layer_outputs.append(x)

    hidden = forward_rms_norm(params.final_norm, x, config.rms_norm_eps)
    if kv_cache is None:
        return hidden, None
    return hidden, KVCache(tuple(new_layer_caches), key_mask, kv_cache.length + num_queries)


def forward_lm_head(embed_tokens: Array, hidden: Array, softcap: float | None) -> Array:
    logits = op.einsum(hidden, embed_tokens, "B S M, V M -> B S V")
    if softcap is not None:
        logits = jnp.tanh(logits / softcap) * softcap
    return logits


def forward_gemma4_lm(
    params: Gemma4TextParams,
    input_ids: Array,
    position_ids: Array,
    *,
    config: Gemma4TextConfig,
    attention_mask: Array | None = None,
    kv_cache: KVCache | None = None,
    cache_mode: Literal["prefill", "append"] = "prefill",
    logits_to_keep: int = 0,
    lora: Gemma4LoRAParams | None = None,
) -> Gemma4Output:
    """Full causal LM forward; ``logits_to_keep > 0`` computes logits for the last positions only."""

    hidden, new_cache = forward_gemma4_text(
        params,
        input_ids,
        position_ids,
        config=config,
        attention_mask=attention_mask,
        kv_cache=kv_cache,
        cache_mode=cache_mode,
        lora=lora,
    )
    head_input = hidden[:, -logits_to_keep:] if logits_to_keep > 0 else hidden
    logits = forward_lm_head(params.embed_tokens, head_input, config.final_logit_softcapping)
    return Gemma4Output(logits, hidden, new_cache)
