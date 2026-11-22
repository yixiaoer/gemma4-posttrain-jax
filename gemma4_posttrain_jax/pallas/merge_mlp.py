"""The first merge of a TP decoder layer and its MLP in one kernel, so the MLP weights stream during the exchange.

In the TP decode step every layer waits twice for the four chips to exchange partial sums (about 14 µs each at
32 rows, a quarter of the step), and the TensorCores do nothing useful meanwhile. The MLP weights do not depend
on the exchanged values, so this kernel starts reading them at its first instruction:

1. Both TensorCores of the chip start DMAs of their half of this chip's gate, up and down slices (``F/2``
   columns each) into ``buffers`` VMEM slots of ``tile`` columns.
2. TensorCore 0 does what ``one_shot_psum_norm`` does after the attention block: it exchanges the attention
   partial sums with the other chips, adds them in mesh order, and computes ``x = residual + rms_norm(sum)``
   and the MLP input ``h = rms_norm(x)``. It writes ``h`` straight into TensorCore 1's VMEM (an on-chip remote
   DMA, after TensorCore 1 has signalled that it is inside the kernel).
3. Each TensorCore multiplies ``h`` by the tiles as they arrive and refills each slot with a later tile.
   TensorCore 1 sends its float32 partial to TensorCore 0, which adds the two (core 0's first) and writes the
   chip's MLP partial ``[R, M]``. The second merge (``one_shot_psum_norm``) is unchanged.

The rounding points are those of ``one_shot_psum_norm`` followed by ``tp_rollout._mlp``; the MLP products are
the same bf16 products with float32 accumulation, summed in a different order (``mlp_stream``).
``precision="high"`` keeps gate, up and the activation in float32 (``mlp_stream``).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from gemma4_posttrain_jax.pallas.cost import estimate
from gemma4_posttrain_jax.pallas.ici_allreduce import (
    CHIPS,
    _barrier,
    _exchange_copies,
    _gather_rows,
    _scatter_gather_rounded,
    _sum_in_mesh_order,
    check_merge_shapes,
)
from gemma4_posttrain_jax.pallas.mlp_stream import PRECISIONS, tile_partial

PARAMS = dict(collective_id=8, disable_bounds_checks=True, disable_semaphore_checks=True)


def merge_mlp(
    partial: jax.Array,
    residual: jax.Array,
    post_weight: jax.Array,
    pre_weight: jax.Array,
    gate: jax.Array,
    up: jax.Array,
    down: jax.Array,
    *,
    eps: float,
    axis_name: str,
    other_axes: tuple[str, ...] = (),
    tile: int = 256,
    buffers: int = 4,
    early: int | None = None,
    precision: str = "model",
    algorithm: str = "one_shot",
    vmem_limit_bytes: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """``(x, mlp_partial)`` for one layer inside ``shard_map`` over four chips (see the module docstring).

    ``partial`` is the attention block's float32 partial ``[R, M]``, ``residual`` the layer input, the norm
    weights ``[M]`` or ``[1, M]``, and ``gate``/``up`` ``[M, F]``, ``down`` ``[F, M]`` this chip's slices.
    ``early`` is how many tiles TensorCore 0 requests before the exchange (default: all ``buffers``); it
    requests the others right after starting the exchange DMAs. ``algorithm="scatter_gather"`` merges in two
    rounds with the same result bits (``ici_allreduce._scatter_gather_rounded``); ``algorithm="gather_rows"``
    stacks the chips' row blocks for attention split by rows (``partial`` holds this chip's ``R/4`` rows).
    """
    rows, cols, local = check_merge_shapes(partial, residual, algorithm)
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    width = gate.shape[1]
    if gate.shape != (cols, width) or up.shape != (cols, width) or down.shape != (width, cols):
        raise ValueError("expects gate and up [M, F] and down [F, M]")
    if width % (2 * tile) or rows % 8:
        raise ValueError("F must split into two cores x tiles and the rows into sublane groups")
    two_rounds = algorithm == "scatter_gather"
    gather = algorithm == "gather_rows"
    per_core = width // 2
    steps = per_core // tile
    buffers = min(buffers, steps)
    early = buffers if early is None else min(early, buffers)
    dtype = residual.dtype
    norm_rows = [w.reshape(1, cols).astype(post_weight.dtype) for w in (post_weight, pre_weight)]
    exchanges = CHIPS - 1
    weight_bytes = 3 * cols * width * gate.dtype.itemsize
    params: dict[str, Any] = dict(PARAMS)
    if vmem_limit_bytes is not None:
        params["vmem_limit_bytes"] = vmem_limit_bytes

    def rounded(value: Any) -> Any:
        return value.astype(dtype).astype(jnp.float32)

    def norm(value: Any, weight: Any) -> Any:
        factor = lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + jnp.float32(eps))
        return rounded(value * factor * weight.astype(jnp.float32))

    @pl.kernel(
        out_type=(pltpu.HBM((rows, cols), dtype), pltpu.HBM((rows, cols), jnp.float32)),
        mesh=pltpu.TensorCoreMesh(axis_name="tc", num_cores=2),
        scratch_types=(
            pltpu.VMEM((local, cols), jnp.float32),  # own attention partial (own rows when gathering)
            # the peers' partials, with two rounds the four chips' column chunks of this chip, or the row blocks
            pltpu.VMEM(
                (CHIPS, rows, cols // CHIPS)
                if two_rounds
                else (CHIPS, local, cols)
                if gather
                else (exchanges, rows, cols),
                jnp.float32,
            ),
            pltpu.VMEM((CHIPS, rows, cols // CHIPS) if two_rounds else (16, 128), residual.dtype),  # rounded chunks
            pltpu.VMEM((rows, cols), dtype),  # residual
            pltpu.VMEM((2, 1, cols), post_weight.dtype),  # norm weights
            pltpu.VMEM((rows, cols), dtype),  # x
            pltpu.VMEM((rows, cols), dtype),  # h (core 0 writes it into core 1's copy)
            pltpu.VMEM((buffers, cols, tile), gate.dtype),
            pltpu.VMEM((buffers, cols, tile), up.dtype),
            pltpu.VMEM((buffers, tile, cols), down.dtype),
            pltpu.VMEM((rows, cols), jnp.float32),  # MLP partial of this core
            pltpu.VMEM((rows, cols), jnp.float32),  # core 1's MLP partial (on core 0)
            pltpu.SemaphoreType.DMA((4,)),  # loads
            pltpu.SemaphoreType.DMA((2,)),  # stores
            pltpu.SemaphoreType.DMA((exchanges,)),
            pltpu.SemaphoreType.DMA((exchanges,)),
            pltpu.SemaphoreType.DMA((buffers,)),
            pltpu.SemaphoreType.DMA((buffers,)),
            pltpu.SemaphoreType.DMA((buffers,)),
            pltpu.SemaphoreType.DMA((4,)),  # h and partial handoffs: send, receive
            pltpu.SemaphoreType.REGULAR,  # core 1 is inside the kernel
            pltpu.SemaphoreType.DMA((CHIPS,)),  # second round of the two-round merge
        ),
        name="merge_mlp",
        compiler_params=pltpu.CompilerParams(**params),
        cost_estimate=estimate(
            flops=16 * rows * cols + 6 * rows * cols * width,
            transcendentals=rows * width,
            bytes_accessed=weight_bytes + rows * cols * (4 + 2 + 2 + 4),
            remote_bytes=exchanges * rows * cols * 4,
        ),
    )
    def kernel(
        partial_hbm: Any,
        residual_hbm: Any,
        post_hbm: Any,
        pre_hbm: Any,
        g_hbm: Any,
        u_hbm: Any,
        d_hbm: Any,
        x_hbm: Any,
        o_hbm: Any,
        own: Any,
        inbox: Any,
        gathered: Any,
        res: Any,
        weight: Any,
        x_v: Any,
        h_v: Any,
        g_v: Any,
        u_v: Any,
        d_v: Any,
        acc: Any,
        other: Any,
        load_sems: Any,
        store_sems: Any,
        send_sems: Any,
        recv_sems: Any,
        g_sems: Any,
        u_sems: Any,
        d_sems: Any,
        hand_sems: Any,
        ready: Any,
        more_sems: Any,
    ) -> None:
        core = lax.axis_index("tc")
        me = lax.axis_index(axis_name)
        fixed = {name: lax.axis_index(name) for name in other_axes}
        base = core * per_core

        def weights(j: Any, slot: Any) -> tuple[Any, Any, Any]:
            start = base + j * tile
            return (
                pltpu.make_async_copy(g_hbm.at[:, pl.ds(start, tile)], g_v.at[slot], g_sems.at[slot]),
                pltpu.make_async_copy(u_hbm.at[:, pl.ds(start, tile)], u_v.at[slot], u_sems.at[slot]),
                pltpu.make_async_copy(d_hbm.at[pl.ds(start, tile), :], d_v.at[slot], d_sems.at[slot]),
            )

        def request(first: int, last: int) -> None:
            for j in range(first, last):
                for copy in weights(j, j):
                    copy.start()

        def on_chip(tc: int) -> Any:
            return {**fixed, axis_name: me, "tc": tc}

        hand_h = pltpu.make_async_remote_copy(
            h_v,
            h_v,
            hand_sems.at[0],
            hand_sems.at[1],
            device_id=on_chip(1),
            device_id_type=pl.DeviceIdType.MESH,
        )
        hand_partial = pltpu.make_async_remote_copy(
            acc,
            other,
            hand_sems.at[2],
            hand_sems.at[3],
            device_id=on_chip(0),
            device_id_type=pl.DeviceIdType.MESH,
        )
        store_x = pltpu.make_async_copy(x_v, x_hbm, store_sems.at[0])

        @pl.when(core == 0)
        def _merge() -> None:
            loads = [
                pltpu.make_async_copy(partial_hbm, own, load_sems.at[0]),
                pltpu.make_async_copy(residual_hbm, res, load_sems.at[1]),
                pltpu.make_async_copy(post_hbm, weight.at[0], load_sems.at[2]),
                pltpu.make_async_copy(pre_hbm, weight.at[1], load_sems.at[3]),
            ]
            for load in loads:
                load.start()
            request(0, early)
            _barrier(axis_name, fixed)
            loads[0].wait()
            if gather:
                request(early, buffers)
                total = _gather_rows(own, inbox, send_sems, recv_sems, more_sems, axis_name, fixed)
            elif two_rounds:
                request(early, buffers)
                total = _scatter_gather_rounded(own, inbox, gathered, send_sems, recv_sems, more_sems, axis_name, fixed)
            else:
                exchange = _exchange_copies(own, inbox, send_sems, recv_sems, axis_name, fixed)
                for copy in exchange:
                    copy.start()
                request(early, buffers)
                for copy in exchange:
                    copy.wait_send()
                    copy.wait_recv()
                total = _sum_in_mesh_order(own, inbox, axis_name)
            for load in loads[1:]:
                load.wait()
            x = rounded(res[...].astype(jnp.float32) + norm(rounded(total), weight[0]))
            x_v[...] = x.astype(dtype)
            store_x.start()
            h_v[...] = norm(x, weight[1]).astype(dtype)
            pl.semaphore_wait(ready, 1)
            hand_h.start()

        @pl.when(core == 1)
        def _wait_for_h() -> None:
            request(0, buffers)
            pl.semaphore_signal(ready, 1, device_id=on_chip(0), device_id_type=pl.DeviceIdType.MESH)
            hand_h.wait_recv()

        acc[...] = jnp.zeros((rows, cols), jnp.float32)

        def body(j: Any, carry: Any) -> Any:
            slot = j % buffers
            for copy in weights(j, slot):
                copy.wait()
            acc[...] += tile_partial(h_v[...], g_v[slot], u_v[slot], d_v[slot], precision)

            @pl.when(j + buffers < steps)
            def _refill() -> None:
                for copy in weights(j + buffers, slot):
                    copy.start()

            return carry

        lax.fori_loop(0, steps, body, 0)

        @pl.when(core == 1)
        def _send_partial() -> None:
            hand_partial.start()
            hand_partial.wait_send()

        @pl.when(core == 0)
        def _finish() -> None:
            hand_h.wait_send()
            hand_partial.wait_recv()
            acc[...] += other[...]
            store = pltpu.make_async_copy(acc, o_hbm, store_sems.at[1])
            store.start()
            store_x.wait()
            store.wait()

    x, mlp = kernel(partial, residual, *norm_rows, gate, up, down)
    return x, mlp
