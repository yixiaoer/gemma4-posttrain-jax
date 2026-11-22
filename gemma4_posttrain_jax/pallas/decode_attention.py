"""Single-token decode attention that reads only the written part of a contiguous KV cache (prototype).

At a decode step every row has one new query token and attends to the cache slots
``[starts[b], length)``: prompts are left-padded (``starts[b]`` is the first real slot of row ``b``)
and every row has written the same number of slots ``length``. XLA's attention reads all ``C`` slots
and masks the rest. This kernel reads only the blocks of ``[min(starts), length)``, the union of the
rows' ranges (the starts differ by the prompt lengths, a few dozen to a hundred slots), and masks
inside the blocks.

Layout: the grid's first axis splits the rows between the two TensorCores (``parallel``); the second
walks the KV blocks. A block index past the union range repeats the last valid block, so the pipeline
issues no new copy, and the body is skipped. Per block, one MXU matmul computes the scores of all
rows of the core against all their queries (the K block streams through, the queries stay); each row
keeps its own ``[block, heads]`` columns. The softmax runs online over the blocks in float32;
``p @ V`` is accumulated on the vector unit, because loading a V block into the MXU for two query
rows would cost more than streaming it.

Numerics: scores and softmax in float32, probabilities rounded to the value dtype before ``p @ V``
(as the XLA path rounds the probabilities), float32 accumulation. The normalization happens at the
end (online softmax), so results differ from the XLA path by rounding.
"""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from gemma4_posttrain_jax.pallas.cost import estimate

DEFAULT = lax.Precision.DEFAULT
NEG_INF = -1e30


def decode_attention(
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    starts: jax.Array,
    length: jax.Array,
    *,
    block: int | None = None,
    cores: int = 2,
    interpret: bool | None = None,
) -> jax.Array:
    """``softmax(q k^T masked to [starts[b], length)) v`` for one query token per row.

    q: ``[B, R, D]`` (the query heads sharing one KV head), k_cache / v_cache: ``[B, C, D]``,
    starts: ``[B]`` int32, length: int32 scalar (the same for all rows). ``B`` must divide by
    ``cores``, ``C`` by ``block``, and ``B // cores * R`` by 8. Returns ``[B, R, D]`` in ``q.dtype``.
    """
    batch, heads, dim = q.shape
    capacity = k_cache.shape[1]
    block = (
        pick_block(capacity, rows=batch // cores, dim=dim, itemsize=k_cache.dtype.itemsize) if block is None else block
    )
    if k_cache.shape != (batch, capacity, dim) or v_cache.shape != k_cache.shape:
        raise ValueError("expects q [B, R, D] and caches [B, C, D]")
    if batch % cores or capacity % block or (batch // cores * heads) % 8:
        raise ValueError("batch must divide by cores, capacity by block, rows per core * heads by 8")
    rows = batch // cores
    blocks = capacity // block
    if interpret is None:
        interpret = jax.devices()[0].platform != "tpu"
    scalars = jnp.concatenate([starts.astype(jnp.int32).reshape(batch), jnp.reshape(length, (1,)).astype(jnp.int32)])
    # Queries and output as float32 [B * R, D]: a [rows, R, D] bf16 block would pad R to the bf16 tile.
    flat_q = q.astype(jnp.float32).reshape(batch * heads, dim)

    def first_block(core: Any, scalar_ref: Any) -> Any:
        lowest = scalar_ref[core * rows]
        for r in range(1, rows):
            lowest = jnp.minimum(lowest, scalar_ref[core * rows + r])
        return lowest // block

    def last_block(scalar_ref: Any) -> Any:
        return jnp.maximum(scalar_ref[batch] - 1, 0) // block

    def kv_index(core: Any, j: Any, scalar_ref: Any) -> tuple[Any, Any, Any]:
        return core, jnp.minimum(first_block(core, scalar_ref) + j, last_block(scalar_ref)), 0

    def kernel(scalar_ref: Any, q_ref: Any, k_ref: Any, v_ref: Any, o_ref: Any, m_ref: Any, l_ref: Any, acc_ref: Any):
        core, j = pl.program_id(0), pl.program_id(1)
        first = first_block(core, scalar_ref)
        count = last_block(scalar_ref) - first + 1
        length_value = scalar_ref[batch]

        @pl.when(j == 0)
        def _init() -> None:
            m_ref[...] = jnp.full(m_ref.shape, NEG_INF, jnp.float32)
            l_ref[...] = jnp.zeros(l_ref.shape, jnp.float32)
            acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

        @pl.when(j < count)
        def _block() -> None:
            keys = k_ref[...].reshape(rows * block, dim)
            queries = q_ref[...].astype(k_ref.dtype)  # [rows * heads, dim]
            scores = lax.dot_general(
                keys, queries, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32, precision=DEFAULT
            )  # [rows * block, rows * heads]
            position = (first + j) * block + lax.broadcasted_iota(jnp.int32, (block, heads), 0)
            for r in range(rows):
                own = scores[r * block : (r + 1) * block, r * heads : (r + 1) * heads]  # [block, heads]
                valid = (position >= scalar_ref[core * rows + r]) & (position < length_value)
                own = jnp.where(valid, own, NEG_INF)
                m_prev = m_ref[r]  # [1, heads]
                m_next = jnp.maximum(m_prev, jnp.max(own, axis=0, keepdims=True))
                alpha = jnp.exp(m_prev - m_next)
                p = jnp.where(valid, jnp.exp(own - m_next), 0.0)  # [block, heads]
                l_ref[r] = alpha * l_ref[r] + jnp.sum(p, axis=0, keepdims=True)
                m_ref[r] = m_next
                values = v_ref[r].astype(jnp.float32)  # [block, dim]
                p_rounded = p.astype(v_ref.dtype).astype(jnp.float32)
                for h in range(heads):
                    row = r * heads + h
                    contribution = jnp.sum(values * p_rounded[:, h : h + 1], axis=0, keepdims=True)  # [1, dim]
                    acc_ref[row : row + 1, :] = alpha[:, h : h + 1] * acc_ref[row : row + 1, :] + contribution

        @pl.when(j == pl.num_programs(1) - 1)
        def _finish() -> None:
            for r in range(rows):
                for h in range(heads):
                    row = r * heads + h
                    o_ref[row : row + 1, :] = acc_ref[row : row + 1, :] / l_ref[r][:, h : h + 1]

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(cores, blocks),
        in_specs=[
            pl.BlockSpec((rows * heads, dim), lambda c, j, s: (c, 0)),
            pl.BlockSpec((rows, block, dim), kv_index),
            pl.BlockSpec((rows, block, dim), kv_index),
        ],
        out_specs=pl.BlockSpec((rows * heads, dim), lambda c, j, s: (c, 0)),
        scratch_shapes=[
            pltpu.VMEM((rows, 1, heads), jnp.float32),
            pltpu.VMEM((rows, 1, heads), jnp.float32),
            pltpu.VMEM((rows * heads, dim), jnp.float32),
        ],
    )
    out = pl.pallas_call(
        kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((batch * heads, dim), jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        interpret=interpret,
        name="decode_attention",
        # static shapes only: the whole cache, an upper bound on the slots a call reads
        cost_estimate=estimate(
            flops=4 * batch * heads * capacity * dim,
            transcendentals=batch * heads * capacity,
            bytes_accessed=2 * batch * capacity * dim * k_cache.dtype.itemsize + 2 * batch * heads * dim * 4,
        ),
    )(scalars, flat_q, k_cache, v_cache)
    return cast(jax.Array, out.reshape(batch, heads, dim).astype(q.dtype))


# K and V blocks are double-buffered: 4 * rows * block * dim * itemsize bytes of the 16 MiB TC VMEM.
KV_VMEM_BUDGET = 8 * 2**20


def pick_block(capacity: int, *, rows: int = 4, dim: int = 512, itemsize: int = 2) -> int:
    """The largest of 256/128/64 slots per block that divides the capacity and keeps the K/V buffers in budget.

    Without such a block (small test caches) the whole cache is one block.
    """
    for block in (256, 128, 64):
        if capacity % block == 0 and 4 * rows * block * dim * itemsize <= KV_VMEM_BUDGET:
            return block
    return capacity


def supported(batch: int, heads: int, capacity: int, *, cores: int = 2) -> bool:
    """Whether ``decode_attention`` takes this shape (otherwise use the XLA formula).

    On TPU the KV block must also fill whole bf16 tiles (16 rows); interpret mode takes any block.
    """
    tpu = jax.devices()[0].platform == "tpu"
    rows = batch // cores if batch % cores == 0 else batch
    block = pick_block(capacity, rows=rows)
    return batch % cores == 0 and (batch // cores * heads) % 8 == 0 and (not tpu or block % 16 == 0)


def reference_attention(q: jax.Array, k_cache: jax.Array, v_cache: jax.Array, starts: jax.Array, length: jax.Array):
    """The XLA path: full-capacity scores, mask, float32 softmax, probabilities in the value dtype."""
    capacity = k_cache.shape[1]
    scores = jnp.einsum("brd,btd->brt", q, k_cache, preferred_element_type=jnp.float32, precision=DEFAULT)
    position = jnp.arange(capacity)[None, None, :]
    valid = (position >= starts[:, None, None]) & (position < length)
    scores = jnp.where(valid, scores, jnp.finfo(jnp.float32).min)
    probs = jax.nn.softmax(scores, axis=-1).astype(v_cache.dtype)
    return jnp.einsum("brt,btd->brd", probs, v_cache, preferred_element_type=jnp.float32).astype(q.dtype)
