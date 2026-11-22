"""Prefill prompts in chunks within a fixed KV capacity, using a shared array interface for native and TP execution."""

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array, lax

from gemma4_posttrain_jax.model import KVCache


def chunked_prefill(
    forward: Callable[[Array, Array, Array, KVCache], tuple[Array, KVCache]],
    ids: Array,
    positions: Array,
    mask: Array,
    cache: KVCache,
    *,
    chunk_size: int,
) -> tuple[Array, KVCache]:
    """Call the forward function in extend mode, preserving positions from the full prompt.

    Process the final partial chunk separately without increasing capacity or writing extra padding. Each scan
    step needs attention workspace only for its current chunk. Return all hidden states to preserve the prefill
    interface; the compiler can discard unused positions when the caller needs only the last one. This function
    does not interleave host-side decoding with prefill.
    """

    if chunk_size <= 0:
        raise ValueError("prefill chunk_size 必须为正")
    batch, width = ids.shape
    chunk_size = min(chunk_size, width)
    chunks, remainder = divmod(width, chunk_size)
    stop = chunks * chunk_size

    def split(value: Array) -> Array:
        return value[:, :stop].reshape(batch, chunks, chunk_size).transpose(1, 0, 2)

    def body(current: KVCache, values: tuple[Array, Array, Array]) -> tuple[KVCache, Array]:
        tokens, logical_positions, real = values
        hidden, updated = forward(tokens, logical_positions, real, current)
        return updated, hidden

    cache, outputs = lax.scan(body, cache, (split(ids), split(positions), split(mask)))
    hidden = outputs.transpose(1, 0, 2, 3).reshape(batch, stop, outputs.shape[-1])
    if remainder:
        tail, cache = forward(ids[:, stop:], positions[:, stop:], mask[:, stop:], cache)
        hidden = jnp.concatenate((hidden, tail), axis=1)
    return hidden, cache
