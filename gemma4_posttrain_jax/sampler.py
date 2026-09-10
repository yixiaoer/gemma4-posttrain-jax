"""Pure-JAX prefill, token decode, and rollout sampling for Gemma 4.

The numerical core is functional: arrays enter, arrays leave, and the generation loop is a
``lax.while_loop``. ``PureJaxRollout`` is only a thin host-side holder implementing the interface
for swapping training parameters into the rollout backend.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.sharding import Mesh

from gemma4_posttrain_jax.lora import Gemma4LoRAParams
from gemma4_posttrain_jax.model import (
    Gemma4Output,
    Gemma4TextConfig,
    Gemma4TextParams,
    KVCache,
    forward_gemma4_lm,
    init_kv_cache,
)
from gemma4_posttrain_jax.sharding import constrain_batch, constrain_batch_tree


class SamplerConfig(NamedTuple):
    """生成时固定的采样配置。

    temperature<=0使用greedy，top_k<=0关闭TopK截断，top_p=1关闭Top-p截断。
    随机采样先应用temperature，再做TopK和Top-p。记录的logprob始终来自
    未调整温度、未截断的FP32策略分布，不能用它代替实际采样分布的概率。
    """

    max_prompt_len: int
    max_new_tokens: int
    temperature: float = 1.0
    top_k: int = 0
    eos_ids: tuple[int, ...] = ()
    seed: int = 0
    top_p: float = 1.0


class RolloutBatch(NamedTuple):
    prompt_ids: Array  # [B, P]
    completion_ids: Array  # [B, N]
    completion_mask: Array  # [B, N] bool; includes the first EOS, excludes every later slot
    rollout_logps: Array  # [B, N] float32; zero outside completion_mask
    lengths: Array  # [B] int32


class Rollout(Protocol):
    """Minimal rollout backend boundary used by the later GRPO trainer."""

    def generate(self, prompt_ids: Array, prompt_mask: Array, *, key: Array | None = None) -> RolloutBatch: ...

    def update_params(self, params: Gemma4TextParams) -> None: ...


def _position_ids(attention_mask: Array) -> Array:
    """HF-style logical positions: left padding stays at zero and real tokens start at zero."""

    positions = jnp.cumsum(attention_mask.astype(jnp.int32), axis=-1) - 1
    return jnp.maximum(positions, 0)


def prefill(
    params: Gemma4TextParams,
    config: Gemma4TextConfig,
    prompt_ids: Array,
    prompt_mask: Array,
    *,
    max_new_tokens: int,
    cache_dtype: Any | None = None,
    mesh: Mesh | None = None,
    lora: Gemma4LoRAParams | None = None,
) -> Gemma4Output:
    """Encode a fixed-width, left-padded prompt and populate full/sliding-window KV caches."""

    if prompt_ids.ndim != 2 or prompt_mask.shape != prompt_ids.shape:
        raise ValueError(f"prompt ids/mask must have the same [B, P] shape, got {prompt_ids.shape}/{prompt_mask.shape}")
    batch_size, prompt_width = prompt_ids.shape
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    dtype = params.embed_tokens.dtype if cache_dtype is None else cache_dtype
    if mesh is not None:
        prompt_ids = constrain_batch(prompt_ids, mesh)
        prompt_mask = constrain_batch(prompt_mask, mesh)
    cache = init_kv_cache(config, batch_size, prompt_width + max_new_tokens, dtype)
    if mesh is not None:
        cache = constrain_batch_tree(cache, mesh)
    with jax.named_scope("rollout_prefill"):
        return forward_gemma4_lm(
            params,
            prompt_ids,
            _position_ids(prompt_mask),
            config=config,
            attention_mask=prompt_mask,
            kv_cache=cache,
            cache_mode="prefill",
            logits_to_keep=1,
            lora=lora,
        )


def decode_step(
    params: Gemma4TextParams,
    config: Gemma4TextConfig,
    token_ids: Array,
    position_ids: Array,
    kv_cache: KVCache,
    *,
    attention_mask: Array | None = None,
    mesh: Mesh | None = None,
    lora: Gemma4LoRAParams | None = None,
) -> Gemma4Output:
    """Append one physical token slot per batch row and return next-token logits."""

    if token_ids.ndim != 1 or position_ids.shape != token_ids.shape:
        raise ValueError(f"token/position must have the same [B] shape, got {token_ids.shape}/{position_ids.shape}")
    if attention_mask is None:
        attention_mask = jnp.ones_like(token_ids, dtype=jnp.bool_)
    if mesh is not None:
        token_ids = constrain_batch(token_ids, mesh)
        position_ids = constrain_batch(position_ids, mesh)
        attention_mask = constrain_batch(attention_mask, mesh)
        kv_cache = constrain_batch_tree(kv_cache, mesh)
    with jax.named_scope("rollout_decode"):
        return forward_gemma4_lm(
            params,
            token_ids[:, None],
            position_ids[:, None],
            config=config,
            attention_mask=attention_mask[:, None],
            kv_cache=kv_cache,
            cache_mode="append",
            logits_to_keep=1,
            lora=lora,
        )


def _sample_token(
    logits: Array, key: Array, *, temperature: float, top_k: int, top_p: float = 1.0
) -> tuple[Array, Array]:
    """Sample from the configured proposal but score under the full f32 policy."""

    with jax.named_scope("rollout_policy"):
        logits32 = logits.astype(jnp.float32)
        # TPU默认exp/log近似曾使小词表logprob偏差达到3.65e-5。
        # matmul精度设置不会控制这些算子，这里显式要求高精度。
        shifted = logits32 - jnp.max(logits32, axis=-1, keepdims=True)
        normalizer = lax.log(
            jnp.sum(lax.exp(shifted, accuracy=lax.AccuracyMode.HIGHEST), axis=-1, keepdims=True),
            accuracy=lax.AccuracyMode.HIGHEST,
        )
        full_logps = shifted - normalizer
        vocab_size = logits.shape[-1]
        if temperature <= 0:
            token = jnp.argmax(logits32, axis=-1).astype(jnp.int32)
        else:
            scores = logits32 / temperature
            indices = None
            if 0 < top_k < vocab_size:
                scores, indices = lax.top_k(scores, top_k)
            elif top_p < 1:
                indices = jnp.argsort(scores, axis=-1, descending=True, stable=True)
                scores = jnp.take_along_axis(scores, indices, axis=-1)
            if top_p < 1:
                weights = lax.exp(scores - jnp.max(scores, axis=-1, keepdims=True), accuracy=lax.AccuracyMode.HIGHEST)
                probabilities = weights / jnp.sum(weights, axis=-1, keepdims=True)
                previous_mass = jnp.concatenate(
                    (jnp.zeros_like(probabilities[:, :1]), jnp.cumsum(probabilities, axis=-1)[:, :-1]), axis=-1
                )
                # 保留使累计概率达到p的那一项，并保证至少保留概率最高的token。
                keep = (previous_mass < top_p) | (jnp.arange(scores.shape[-1])[None, :] == 0)
                scores = jnp.where(keep, scores, -jnp.inf)
            selected = jax.random.categorical(key, scores, axis=-1).astype(jnp.int32)
            token = selected if indices is None else jnp.take_along_axis(indices, selected[:, None], axis=-1)[:, 0]
        logp = jnp.take_along_axis(full_logps, token[:, None], axis=-1)[:, 0]
        return token, logp


def _is_eos(token: Array, eos_ids: tuple[int, ...]) -> Array:
    result = jnp.zeros(token.shape, dtype=jnp.bool_)
    for eos_id in eos_ids:
        result |= token == eos_id
    return result


class _GenerationState(NamedTuple):
    step: Array
    key: Array
    active: Array
    logits: Array
    cache: KVCache
    completion_ids: Array
    completion_mask: Array
    rollout_logps: Array


def generate(
    params: Gemma4TextParams,
    config: Gemma4TextConfig,
    prompt_ids: Array,
    prompt_mask: Array,
    *,
    sampler_config: SamplerConfig,
    key: Array | None = None,
    mesh: Mesh | None = None,
    lora: Gemma4LoRAParams | None = None,
) -> RolloutBatch:
    """Generate a fixed-shape rollout, stopping the device loop once every row reaches EOS."""

    if prompt_ids.shape[1] != sampler_config.max_prompt_len:
        raise ValueError(
            f"prompt width {prompt_ids.shape[1]} does not match max_prompt_len={sampler_config.max_prompt_len}"
        )
    if sampler_config.max_prompt_len <= 0 or sampler_config.max_new_tokens <= 0:
        raise ValueError("max_prompt_len and max_new_tokens must both be positive")
    if sampler_config.top_k > config.vocab_size:
        raise ValueError(f"top_k={sampler_config.top_k} exceeds vocab_size={config.vocab_size}")
    if not math.isfinite(sampler_config.temperature):
        raise ValueError("temperature must be finite")
    if not math.isfinite(sampler_config.top_p) or not 0 < sampler_config.top_p <= 1:
        raise ValueError("top_p must be finite and in (0, 1]")

    prompt_mask = prompt_mask.astype(jnp.bool_)
    out = prefill(
        params,
        config,
        prompt_ids,
        prompt_mask,
        max_new_tokens=sampler_config.max_new_tokens,
        mesh=mesh,
        lora=lora,
    )
    assert out.kv_cache is not None
    batch_size = prompt_ids.shape[0]
    max_new = sampler_config.max_new_tokens
    prompt_lengths = jnp.sum(prompt_mask, axis=-1, dtype=jnp.int32)
    if key is None:
        key = jax.random.PRNGKey(sampler_config.seed)
    active = jnp.ones((batch_size,), dtype=jnp.bool_)
    completion_ids = jnp.full((batch_size, max_new), config.pad_token_id, dtype=prompt_ids.dtype)
    completion_mask = jnp.zeros((batch_size, max_new), dtype=jnp.bool_)
    rollout_logps = jnp.zeros((batch_size, max_new), dtype=jnp.float32)
    if mesh is not None:
        prompt_lengths = constrain_batch(prompt_lengths, mesh)
        active = constrain_batch(active, mesh)
        completion_ids = constrain_batch(completion_ids, mesh)
        completion_mask = constrain_batch(completion_mask, mesh)
        rollout_logps = constrain_batch(rollout_logps, mesh)
    state = _GenerationState(
        step=jnp.zeros((), dtype=jnp.int32),
        key=key,
        active=active,
        logits=out.logits[:, -1].astype(jnp.float32),
        cache=out.kv_cache,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        rollout_logps=rollout_logps,
    )

    def cond_fn(current: _GenerationState) -> Array:
        return (current.step < max_new) & jnp.any(current.active)

    def body_fn(current: _GenerationState) -> _GenerationState:
        next_key, sample_key = jax.random.split(current.key)
        sampled, logp = _sample_token(
            current.logits,
            sample_key,
            temperature=sampler_config.temperature,
            top_k=sampler_config.top_k,
            top_p=sampler_config.top_p,
        )
        step_mask = current.active
        token = jnp.where(step_mask, sampled, config.pad_token_id).astype(prompt_ids.dtype)
        logp = jnp.where(step_mask, logp, 0.0)
        completion_ids = current.completion_ids.at[:, current.step].set(token)
        completion_mask = current.completion_mask.at[:, current.step].set(step_mask)
        rollout_logps = current.rollout_logps.at[:, current.step].set(logp)
        next_active = step_mask & ~_is_eos(token, sampler_config.eos_ids)

        should_decode = (current.step + 1 < max_new) & jnp.any(next_active)

        def run_decode(_: None) -> tuple[Array, KVCache]:
            decoded = decode_step(
                params,
                config,
                token,
                prompt_lengths + current.step,
                current.cache,
                attention_mask=step_mask,
                mesh=mesh,
                lora=lora,
            )
            assert decoded.kv_cache is not None
            return decoded.logits[:, -1].astype(jnp.float32), decoded.kv_cache

        logits, cache = lax.cond(
            should_decode,
            run_decode,
            lambda _: (current.logits, current.cache),
            operand=None,
        )
        return _GenerationState(
            current.step + 1,
            next_key,
            next_active,
            logits,
            cache,
            completion_ids,
            completion_mask,
            rollout_logps,
        )

    state = lax.while_loop(cond_fn, body_fn, state)
    lengths = jnp.sum(state.completion_mask, axis=-1, dtype=jnp.int32)
    return RolloutBatch(prompt_ids, state.completion_ids, state.completion_mask, state.rollout_logps, lengths)


class PureJaxRollout:
    """Host-side parameter holder around the pure ``generate`` function."""

    def __init__(
        self,
        params: Gemma4TextParams,
        model_config: Gemma4TextConfig,
        sampler_config: SamplerConfig,
        mesh: Mesh | None = None,
    ) -> None:
        self.params = params
        self.model_config = model_config
        self.sampler_config = sampler_config
        self.mesh = mesh

    def update_params(self, params: Gemma4TextParams) -> None:
        self.params = params

    def generate(self, prompt_ids: Array, prompt_mask: Array, *, key: Array | None = None) -> RolloutBatch:
        return generate(
            self.params,
            self.model_config,
            prompt_ids,
            prompt_mask,
            sampler_config=self.sampler_config,
            key=key,
            mesh=self.mesh,
        )
