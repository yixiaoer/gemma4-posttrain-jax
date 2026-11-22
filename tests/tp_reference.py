"""Fixed reference history after a common prefill cache.

The experimental workload excludes prefill and categorical sampling and returns
selected log-probabilities plus the final cache. It is not complete generation.
This module does not compile, time, load a model, or choose a candidate.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

Decode = Callable[..., tuple[jax.Array, Any]]


def reference_length(completion_mask: np.ndarray) -> int:
    """Validate the frozen host mask and return its longest effective row.

    Run outside jit, once when preparing the reference workload. Every row in
    native generate starts active and therefore has at least one sampled token.
    The mask includes EOS itself and excludes every subsequent token.
    """
    mask = np.asarray(completion_mask)
    if mask.ndim != 2 or not all(mask.shape) or mask.dtype != np.dtype(np.bool_):
        raise ValueError("completion_mask must be a nonempty rank-2 bool array")
    lengths = np.sum(mask, axis=1, dtype=np.int64)
    if np.any(lengths < 1):
        raise ValueError("native generate requires at least one active token per row")
    expected = np.arange(mask.shape[1])[None, :] < lengths[:, None]
    if not np.array_equal(mask, expected):
        raise ValueError("completion_mask must contain contiguous active prefixes")
    return int(lengths.max())


def score_policy_targets(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Full-vocabulary raw policy score; same formula as current native sampler.

    This deliberately avoids jax.nn.log_softmax, whose historical implementation is a different
    numerical contract from sampler._sample_token.
    """
    logits32 = logits.astype(jnp.float32)
    shifted = logits32 - jnp.max(logits32, axis=-1, keepdims=True)
    normalizer = lax.log(
        jnp.sum(lax.exp(shifted, accuracy=lax.AccuracyMode.HIGHEST), axis=-1, keepdims=True),
        accuracy=lax.AccuracyMode.HIGHEST,
    )
    full_logps = shifted - normalizer
    return jnp.take_along_axis(full_logps, targets[..., None], axis=-1)[..., 0]


def fixed_history_decode(
    params: Any,
    initial_cache: Any,
    completion_ids: jax.Array,
    completion_mask: jax.Array,
    prompt_lengths: jax.Array,
    *,
    decode_fn: Decode,
    reference_length: int,
) -> tuple[jax.Array, Any]:
    """Return [B, L-1] selected logps and the cache after exactly L-1 appends.

    All array inputs, including model parameters, are explicit dynamic arguments.
    Only the callable and frozen integer L are static. decode_fn accepts
    (params, token_ids, positions, cache, attention_mask=...) and returns
    (rank-2 [B,V] logits, next_cache).

    Append t reads token[:,t], position=prompt_lengths+t, mask[:,t]. It scores
    token[:,t+1], masking the SCORE with mask[:,t+1]. In particular an EOS row
    still participates in its last append if other rows require another step.
    No first-token score (which belongs to prefill), categorical sampling,
    mutation, donation, or host callback is introduced here.

    The caller validates the frozen host mask with reference_length() and binds
    the returned L. The core intentionally does not device_get or branch on
    runtime mask values. Reusing its compiled shape with a different L is outside
    that frozen workload contract.
    """
    if type(reference_length) is not int or reference_length < 1:
        raise ValueError("reference_length must be a positive static integer")
    if completion_ids.ndim != 2:
        raise ValueError("completion_ids must have shape [batch, capacity]")
    batch, capacity = completion_ids.shape
    if not batch or not capacity or reference_length > capacity:
        raise ValueError("reference_length must fit a nonempty completion capacity")
    if completion_mask.shape != completion_ids.shape:
        raise ValueError("completion mask and token shapes must match")
    if prompt_lengths.shape != (batch,):
        raise ValueError("prompt_lengths must have shape [batch]")
    if completion_ids.dtype != jnp.int32 or prompt_lengths.dtype != jnp.int32:
        raise ValueError("tokens and logical prompt lengths must be int32")
    if completion_mask.dtype != jnp.bool_:
        raise ValueError("completion_mask must be bool")

    def body(cache: Any, step: jax.Array) -> tuple[Any, jax.Array]:
        tokens = lax.dynamic_slice_in_dim(completion_ids, step, 1, axis=1)[:, 0]
        step_mask = lax.dynamic_slice_in_dim(completion_mask, step, 1, axis=1)[:, 0]
        logits, next_cache = decode_fn(params, tokens, prompt_lengths + step, cache, attention_mask=step_mask)
        if logits.ndim != 2 or logits.shape[0] != batch:
            raise ValueError("decode_fn must return logits with shape [batch, vocabulary]")
        next_tokens = lax.dynamic_slice_in_dim(completion_ids, step + 1, 1, axis=1)[:, 0]
        next_mask = lax.dynamic_slice_in_dim(completion_mask, step + 1, 1, axis=1)[:, 0]
        scores = score_policy_targets(logits, next_tokens)
        return next_cache, jnp.where(next_mask, scores, jnp.float32(0.0))

    final_cache, scores = lax.scan(body, initial_cache, jnp.arange(reference_length - 1, dtype=jnp.int32))
    return scores.T, final_cache
