"""The decode MLP of one TP layer with the weight reads done by the kernel itself (prototype).

``h [R, M]`` (the normalized MLP input, the same on every chip) times this chip's slices of the gate and up
projections ``[M, F]`` and the down projection ``[F, M]``, with the rounding points of ``tp_rollout._mlp``:
``gate = bf16(h @ Wg)``, ``up = bf16(h @ Wu)``, ``act = bf16(gelu(gate) * up)`` (tanh approximation, float32
inside as in XLA's fusion) and the float32 partial ``act @ Wd``.

Each TensorCore takes half of ``F`` and streams its gate and up columns and its down rows from HBM in tiles of
``tile`` with double buffering, so the reads use the DMA engines of both cores of the chip. The matmuls use
the same products as XLA's (bf16 operands, float32 accumulation), but the float32 sums run in a different
order, so results agree with XLA up to rounding. Returns the two cores' partials ``[cores, R, M]``; their sum
is the chip's partial of the layer.

``precision="high"`` drops the bf16 roundings of gate, up and the activation: gate and up stay float32, and the
float32 activation enters the down projection as two bf16 parts (``hi = bf16(a)``, ``lo = bf16(a - hi)``,
``hi @ Wd + lo @ Wd``), close to a float32 activation times the bf16 weights. The extra MXU pass is cheap while
the weight reads bound the kernel.
"""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT = lax.Precision.DEFAULT
SQRT_2_OVER_PI = 0.7978845608028654
PRECISIONS = ("model", "high")


def gelu_tanh(x: Any) -> Any:
    """``jax.nn.gelu(x, approximate=True)`` in float32, written out for Mosaic."""

    return 0.5 * x * (1.0 + jnp.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)))


def tile_partial(x: Any, gate: Any, up: Any, down: Any, precision: str) -> Any:
    """The float32 contribution of one weight tile, ``act @ down``, with the rounding points of ``precision``."""

    dtype = x.dtype
    g = jnp.dot(x, gate, preferred_element_type=jnp.float32, precision=DEFAULT)
    u = jnp.dot(x, up, preferred_element_type=jnp.float32, precision=DEFAULT)
    if precision == "model":
        g = g.astype(dtype).astype(jnp.float32)
        u = u.astype(dtype).astype(jnp.float32)
    a = gelu_tanh(g) * u
    hi = a.astype(dtype)
    out = jnp.dot(hi, down, preferred_element_type=jnp.float32, precision=DEFAULT)
    if precision == "high":
        lo = (a - hi.astype(jnp.float32)).astype(dtype)
        out = out + jnp.dot(lo, down, preferred_element_type=jnp.float32, precision=DEFAULT)
    return out


def mlp_stream(
    h: jax.Array,
    gate: jax.Array,
    up: jax.Array,
    down: jax.Array,
    *,
    tile: int = 256,
    cores: int = 2,
    interpret: bool | None = None,
    precision: str = "model",
) -> jax.Array:
    """``[cores, R, M]`` float32 partials of the MLP whose weights the kernel streams (see the module docstring)."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")

    rows, dim = h.shape
    width = gate.shape[1]
    if gate.shape != (dim, width) or up.shape != (dim, width) or down.shape != (width, dim):
        raise ValueError("expects gate and up [M, F] and down [F, M]")
    if width % (cores * tile) or rows % 8:
        raise ValueError("F must split into cores x tiles and the rows into sublane groups")
    per_core = width // cores
    steps = per_core // tile
    dtype = h.dtype
    if interpret is None:
        interpret = jax.devices()[0].platform != "tpu"
    tc_mesh = pltpu.TensorCoreMesh(axis_name="tc", num_cores=cores)

    @pl.kernel(
        out_type=pltpu.HBM((cores, rows, dim), jnp.float32),
        mesh=tc_mesh,
        scratch_types=(
            pltpu.VMEM((rows, dim), dtype),
            pltpu.VMEM((2, dim, tile), gate.dtype),
            pltpu.VMEM((2, dim, tile), up.dtype),
            pltpu.VMEM((2, tile, dim), down.dtype),
            pltpu.VMEM((rows, dim), jnp.float32),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
        ),
        name=f"mlp_stream_{cores}core",
        interpret=interpret,
    )
    def kernel(h_hbm, g_hbm, u_hbm, d_hbm, o_hbm, h_v, g_v, u_v, d_v, acc, g_sem, u_sem, d_sem, io_sem) -> None:
        core = lax.axis_index("tc") if cores > 1 else 0
        base = core * per_core

        def weights(j: Any, slot: Any) -> tuple[Any, Any, Any]:
            start = base + j * tile
            return (
                pltpu.make_async_copy(g_hbm.at[:, pl.ds(start, tile)], g_v.at[slot], g_sem.at[slot]),
                pltpu.make_async_copy(u_hbm.at[:, pl.ds(start, tile)], u_v.at[slot], u_sem.at[slot]),
                pltpu.make_async_copy(d_hbm.at[pl.ds(start, tile), :], d_v.at[slot], d_sem.at[slot]),
            )

        for copy in weights(0, 0):
            copy.start()
        load = pltpu.make_async_copy(h_hbm, h_v, io_sem.at[0])
        load.start()
        acc[...] = jnp.zeros((rows, dim), jnp.float32)
        load.wait()

        def body(j: Any, carry: Any) -> Any:
            slot = j % 2

            @pl.when(j + 1 < steps)
            def _prefetch() -> None:
                for copy in weights(j + 1, 1 - slot):
                    copy.start()

            for copy in weights(j, slot):
                copy.wait()
            acc[...] += tile_partial(h_v[...], g_v[slot], u_v[slot], d_v[slot], precision)
            return carry

        lax.fori_loop(0, steps, body, 0)
        store = pltpu.make_async_copy(acc, o_hbm.at[core], io_sem.at[1])
        store.start()
        store.wait()

    return cast(jax.Array, kernel(h, gate, up, down))


def reference(h: jax.Array, gate: jax.Array, up: jax.Array, down: jax.Array) -> jax.Array:
    """The same rounding points with XLA ops over the whole slice (one float32 partial ``[R, M]``)."""

    dtype = h.dtype
    g = jnp.dot(h, gate, preferred_element_type=jnp.float32, precision=DEFAULT).astype(dtype)
    u = jnp.dot(h, up, preferred_element_type=jnp.float32, precision=DEFAULT).astype(dtype)
    a = (gelu_tanh(g.astype(jnp.float32)) * u.astype(jnp.float32)).astype(dtype)
    return jnp.dot(a, down, preferred_element_type=jnp.float32, precision=DEFAULT)
