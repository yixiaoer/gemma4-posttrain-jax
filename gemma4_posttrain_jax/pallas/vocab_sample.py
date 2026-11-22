"""Vocabulary projection fused with Gumbel-max sampling and the log-normalizer (prototype).

At a decode step the TP sampler computes this chip's logits ``hidden @ embed.T`` (softcapped), writes them to
HBM, then reads them again for the Gumbel noise, the argmax and the log-sum-exp. The projection streams the
chip's embedding slice from HBM (65536 x 1536 bf16, 201 MB), so the vector work of the sampler (the threefry
noise above all) can hide behind those copies. This kernel walks the slice in vocabulary tiles; per tile it
computes the logits on the MXU, adds the noise (the same threefry counters as ``tp_rollout.gumbel_slice``),
and updates per row the best noisy value, its index and logit, and an online log-sum-exp of the logits. The
logits never leave VMEM. The grid's first axis splits the slice between the two TensorCores.

Numerics: ``rounding`` chooses where the logits are rounded to the model dtype: ``"dot"`` (default: after the
matmul, as the bf16 output of the XLA einsum; the softcap then runs in float32), ``"none"`` (float32 throughout)
or ``"dot+softcap"``. The MXU accumulates in float32 in a different order than XLA's matmul, so a logit that sits
on a bf16 rounding boundary can come out one bf16 step away (0.25 near the softcap of 30): on random inputs 1-6%
of sampled tokens differ from the XLA sampler and log-probabilities differ by up to about 0.1. The noise and the
tie rule (first index of the maximum) are those of the TP sampler; the log-sum-exp is accumulated online over
the tiles.
"""

from __future__ import annotations

from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src.random.threefry2x32 import threefry2x32_p
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from gemma4_posttrain_jax.pallas.cost import estimate

DEFAULT = lax.Precision.DEFAULT
LANES = 128
# Output lanes per row: best noisy value, its index (as float32; < 2**24), its logit, the row's max logit, and
# the sum of exp(logit - max).
BEST, INDEX, LOGIT, MAX, SUMEXP = range(5)


class Partials(NamedTuple):
    """Per-row results over this chip's vocabulary slice (indices are local to the slice)."""

    best: jax.Array  # [B] float32, max of noise + logits / temperature (logits when temperature is 0)
    position: jax.Array  # [B] int32, first index of that maximum
    logit: jax.Array  # [B] float32, the logit at that index
    row_max: jax.Array  # [B] float32
    row_sumexp: jax.Array  # [B] float32, sum of exp(logit - row_max)


def _gumbel(key0: Any, key1: Any, low: Any) -> Any:
    """``-log(-log(uniform))`` from the threefry bits of ``low`` (as ``tp_rollout.gumbel_slice``)."""

    bits1, bits2 = threefry2x32_p.bind(key0, key1, jnp.zeros_like(low), low)
    bits = bits1 ^ bits2
    one = np.array(1.0, np.float32).view(np.uint32)
    float_bits = lax.bitwise_or(lax.shift_right_logical(bits, jnp.uint32(32 - 23)), jnp.uint32(one))
    floats = lax.bitcast_convert_type(float_bits, jnp.float32) - jnp.float32(1.0)
    tiny = jnp.float32(jnp.finfo(jnp.float32).tiny)
    uniform = lax.max(tiny, floats * (jnp.float32(1.0) - tiny) + tiny)
    return -jnp.log(-jnp.log(uniform))


ROUNDING = ("none", "dot", "dot+softcap")


def _logits(hidden: Any, embed: Any, softcap: float | None, rounding: str = "none") -> Any:
    logits = lax.dot_general(
        hidden, embed, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32, precision=DEFAULT
    )
    if rounding != "none":
        logits = logits.astype(hidden.dtype).astype(jnp.float32)
    if softcap is not None:
        logits = jnp.tanh(logits / jnp.float32(softcap)) * jnp.float32(softcap)
        if rounding == "dot+softcap":
            logits = logits.astype(hidden.dtype).astype(jnp.float32)
    return logits


def pick_tile(vocab: int, dim: int, *, cores: int = 2, itemsize: int = 2, budget: int = 8 * 2**20) -> int:
    """The largest of 2048/1024/512/256 rows per tile that splits the slice evenly and keeps two tiles in budget."""

    for tile in (2048, 1024, 512, 256):
        if vocab % (cores * tile) == 0 and 2 * tile * dim * itemsize <= budget:
            return tile
    if vocab % cores == 0 and vocab // cores < 256:  # small test vocabularies: one tile per core
        return vocab // cores
    raise ValueError(f"no tile for a vocabulary slice of {vocab}")


def vocab_sample_partials(
    hidden: jax.Array,
    embed: jax.Array,
    key_data: jax.Array,
    row_offset: jax.Array,
    column_offset: jax.Array,
    *,
    vocab_size: int,
    temperature: float,
    softcap: float | None,
    tile: int | None = None,
    cores: int = 2,
    interpret: bool | None = None,
    rounding: str = "dot",
) -> Partials:
    """Sampling partials of ``hidden [B, M] @ embed[V, M].T`` over this chip's slice of the vocabulary.

    ``key_data`` is the raw threefry key (2 uint32), ``row_offset`` the global index of row 0 and
    ``column_offset`` the global vocabulary index of ``embed[0]`` (the noise counters are
    ``row * vocab_size + column`` as in ``tp_rollout.gumbel_slice``).
    """
    batch, dim = hidden.shape
    vocab = embed.shape[0]
    if rounding not in ROUNDING:
        raise ValueError(f"rounding must be one of {ROUNDING}")
    if embed.shape[1] != dim:
        raise ValueError("hidden and embed must share the model dimension")
    tile = pick_tile(vocab, dim, cores=cores, itemsize=embed.dtype.itemsize) if tile is None else tile
    if vocab % (cores * tile):
        raise ValueError("the vocabulary slice must split into cores x tiles")
    tiles = vocab // (cores * tile)
    if interpret is None:
        interpret = jax.devices()[0].platform != "tpu"
    scalars = jnp.stack(
        [
            lax.bitcast_convert_type(key_data[0].astype(jnp.uint32), jnp.int32),
            lax.bitcast_convert_type(key_data[1].astype(jnp.uint32), jnp.int32),
            jnp.asarray(row_offset, jnp.int32),
            jnp.asarray(column_offset, jnp.int32),
        ]
    )

    def kernel(s: Any, h_ref: Any, e_ref: Any, o_ref: Any, state: Any) -> None:
        core, j = pl.program_id(0), pl.program_id(1)

        @pl.when(j == 0)
        def _init() -> None:
            lane = lax.broadcasted_iota(jnp.int32, (batch, LANES), 1)
            start = jnp.where(lane == MAX, -jnp.inf, 0.0).astype(jnp.float32)
            state[...] = jnp.where(lane == BEST, -jnp.inf, start).astype(jnp.float32)

        logits = _logits(h_ref[...], e_ref[...], softcap, rounding)  # [B, tile]
        first = (core * tiles + j) * tile
        local = first + lax.broadcasted_iota(jnp.int32, (batch, tile), 1)
        if temperature > 0:
            rows = (s[2] + lax.broadcasted_iota(jnp.int32, (batch, tile), 0)).astype(jnp.uint32)
            low = rows * jnp.uint32(vocab_size) + (s[3] + local).astype(jnp.uint32)
            key0 = jnp.full(low.shape, s[0], jnp.int32).astype(jnp.uint32)
            key1 = jnp.full(low.shape, s[1], jnp.int32).astype(jnp.uint32)
            noisy = _gumbel(key0, key1, low) + logits / jnp.float32(temperature)
        else:
            noisy = logits
        tile_best = jnp.max(noisy, axis=1, keepdims=True)
        hit = (noisy == tile_best) | jnp.isnan(noisy)
        tile_index = jnp.min(jnp.where(hit, local, jnp.int32(2**30)), axis=1, keepdims=True)
        tile_logit = jnp.sum(jnp.where(local == tile_index, logits, 0.0), axis=1, keepdims=True)
        tile_max = jnp.max(logits, axis=1, keepdims=True)

        current = state[...]
        best = current[:, BEST : BEST + 1]
        # strict: an earlier tile keeps a tie; a NaN wins as in jnp.argmax
        take = (tile_best > best) | (jnp.isnan(tile_best) & ~jnp.isnan(best))
        row_max = current[:, MAX : MAX + 1]
        new_max = jnp.maximum(row_max, tile_max)
        sumexp = current[:, SUMEXP : SUMEXP + 1] * jnp.exp(row_max - new_max) + jnp.sum(
            jnp.exp(logits - new_max), axis=1, keepdims=True
        )
        lane = lax.broadcasted_iota(jnp.int32, (batch, LANES), 1)
        update = {
            BEST: jnp.where(take, tile_best, best),
            INDEX: jnp.where(take, tile_index.astype(jnp.float32), current[:, INDEX : INDEX + 1]),
            LOGIT: jnp.where(take, tile_logit, current[:, LOGIT : LOGIT + 1]),
            MAX: new_max,
            SUMEXP: sumexp,
        }
        merged = current
        for position, value in update.items():
            merged = jnp.where(lane == position, value, merged)
        state[...] = merged

        @pl.when(j == tiles - 1)
        def _finish() -> None:
            o_ref[0] = state[...]

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(cores, tiles),
        in_specs=[
            pl.BlockSpec((batch, dim), lambda c, j, s: (0, 0)),
            pl.BlockSpec((tile, dim), lambda c, j, s: (c * tiles + j, 0)),
        ],
        out_specs=pl.BlockSpec((1, batch, LANES), lambda c, j, s: (c, 0, 0)),
        scratch_shapes=[pltpu.VMEM((batch, LANES), jnp.float32)],
    )
    out = pl.pallas_call(
        kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((cores, batch, LANES), jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        interpret=interpret,
        name="vocab_sample",
        cost_estimate=estimate(
            flops=2 * batch * vocab * dim,
            transcendentals=batch * vocab * (4 if temperature > 0 else 2),
            bytes_accessed=vocab * dim * embed.dtype.itemsize + batch * dim * hidden.dtype.itemsize,
        ),
    )(scalars, hidden, embed)
    return combine_cores(cast(jax.Array, out))


def combine_cores(out: jax.Array) -> Partials:
    """Merge the per-core results (core 0 holds the lower vocabulary indices, so it wins ties)."""

    best = out[0, :, BEST]
    index = out[0, :, INDEX]
    logit = out[0, :, LOGIT]
    row_max = out[0, :, MAX]
    sumexp = out[0, :, SUMEXP]
    for core in range(1, out.shape[0]):
        take = (out[core, :, BEST] > best) | (jnp.isnan(out[core, :, BEST]) & ~jnp.isnan(best))
        best = jnp.where(take, out[core, :, BEST], best)
        index = jnp.where(take, out[core, :, INDEX], index)
        logit = jnp.where(take, out[core, :, LOGIT], logit)
        new_max = jnp.maximum(row_max, out[core, :, MAX])
        sumexp = sumexp * jnp.exp(row_max - new_max) + out[core, :, SUMEXP] * jnp.exp(out[core, :, MAX] - new_max)
        row_max = new_max
    return Partials(best, index.astype(jnp.int32), logit, row_max, sumexp)


def reference_partials(
    hidden: jax.Array,
    embed: jax.Array,
    key_data: jax.Array,
    row_offset: jax.Array,
    column_offset: jax.Array,
    *,
    vocab_size: int,
    temperature: float,
    softcap: float | None,
    rounding: str = "dot",
) -> Partials:
    """The same quantities with XLA ops over the whole slice (two-pass log-sum-exp)."""

    logits = _logits(hidden, embed, softcap, rounding)
    batch, vocab = logits.shape
    local = lax.broadcasted_iota(jnp.int32, (batch, vocab), 1)
    if temperature > 0:
        rows = (jnp.asarray(row_offset, jnp.int32) + lax.broadcasted_iota(jnp.int32, (batch, vocab), 0)).astype(
            jnp.uint32
        )
        low = rows * jnp.uint32(vocab_size) + (jnp.asarray(column_offset, jnp.int32) + local).astype(jnp.uint32)
        noisy = _gumbel(key_data[0].astype(jnp.uint32), key_data[1].astype(jnp.uint32), low)
        noisy = noisy + logits / jnp.float32(temperature)
    else:
        noisy = logits
    best = jnp.max(noisy, axis=1)
    index = jnp.argmax(noisy, axis=1).astype(jnp.int32)
    logit = jnp.take_along_axis(logits, index[:, None], axis=1)[:, 0]
    row_max = jnp.max(logits, axis=1)
    sumexp = jnp.sum(jnp.exp(logits - row_max[:, None]), axis=1)
    return Partials(best, index, logit, row_max, sumexp)
