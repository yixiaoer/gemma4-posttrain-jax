"""Small-message all-reduce over four TPU v4 chips by one direct exchange (prototype).

Each chip sends its whole array to the other three with remote DMA and adds what it
receives. For small arrays the fixed cost of each transfer dominates, so one round is
cheaper than XLA's ring algorithm (pallas-tpu-tutorial chapter02/07: 16 KiB per chip,
13063 cycles for ``psum`` against 3416 for this design). Transfers to the two physical
neighbours start first and the two-hop diagonal transfer last; a diagonal flow that
waits for its second hop would otherwise block a neighbour flow on its first hop.

The kernel runs on TensorCore 0 of each chip inside ``shard_map``. All four chips meet
at a barrier before writing into each other's buffers. Every chip adds the four arrays
in the same order (mesh position 0, 1, 2, 3), so all chips hold bitwise identical sums.

``one_shot_psum_norm(exchange_dtype=jnp.bfloat16)`` sends each chip's partial sum rounded to bf16, half
the bytes of the float32 exchange. A bf16 VMEM array is tiled in 16-row blocks, so an 8-row bf16 buffer
would take as many bytes as the float32 one; the kernel instead packs columns ``j`` and ``j + cols/2``
into one uint32 and exchanges ``[rows, cols/2]`` uint32 arrays with the usual (8, 128) tiling.

``one_shot_psum_norm(algorithm="scatter_gather")`` sums in two rounds with the same result bits
(``_scatter_gather_rounded``): a float32 reduce-scatter by column chunks, then an all-gather of the chunks
rounded to the residual dtype, which the norm does first anyway. The busiest link
then carries 3/8 of the array instead of twice the array,
for one more fixed transfer cost: faster at 32 rows, slower at 8.

``algorithm="gather_rows"`` is for attention split by rows: each chip holds the complete float32 output of its
own block of rows (``partial`` is ``[rows/4, cols]``) and the kernel stacks the four blocks in mesh order
(``_gather_rows``) instead of adding them. ``algorithm="scatter_rows"`` is the merge after the MLP when every chip
keeps only its own rows of the residual stream: ``partial`` holds all rows and each chip adds the four chips' copies
of its own block of rows only (``_scatter_rows``). ``norm_gather`` is the matching first merge: norm and residual on
this chip's rows, then the MLP input of all rows gathered. That input is already rounded to bf16, so it travels
packed two values per uint32 (``_pack_bf16``): half the bytes, the same bits.
"""

from __future__ import annotations

from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from gemma4_posttrain_jax.pallas.cost import estimate

CHIPS = 4
ALGORITHMS = ("one_shot", "scatter_gather", "gather_rows", "scatter_rows")
DEFAULT = lax.Precision.DEFAULT
PARAMS = pltpu.CompilerParams(collective_id=7, disable_bounds_checks=True, disable_semaphore_checks=True)


def peer_order(axis_size: int) -> tuple[int, int, int]:
    """Offsets along the mesh axis: neighbours first, diagonal last (for devices 0,1,2,3 on a 2x2 slice)."""
    if axis_size != CHIPS:
        raise ValueError("the one-shot all-reduce is written for four chips")
    # Mesh positions 0,1,2,3 hold devices at (0,0),(1,0),(0,1),(1,1); XOR 1 and XOR 2 are neighbours, XOR 3 diagonal.
    return (1, 2, 3)


def _barrier(axis_name: str, fixed: dict[str, Any]) -> None:
    """All four chips meet before writing into each other's buffers."""

    ready = pltpu.get_barrier_semaphore()
    for rank in range(CHIPS):
        pl.semaphore_signal(
            ready, 1, device_id={**fixed, axis_name: rank, "tc": 0}, device_id_type=pl.DeviceIdType.MESH
        )
    pl.semaphore_wait(ready, CHIPS)


def _exchange_copies(
    send: Any, inbox: Any, send_sems: Any, recv_sems: Any, axis_name: str, fixed: dict[str, Any]
) -> list[Any]:
    """The remote copies of ``_exchange`` (not started), neighbours first."""

    me = lax.axis_index(axis_name)
    return [
        pltpu.make_async_remote_copy(
            send,
            inbox.at[k],
            send_sems.at[k],
            recv_sems.at[k],
            device_id=cast(Any, {**fixed, axis_name: me ^ peer, "tc": 0}),
            device_id_type=pl.DeviceIdType.MESH,
        )
        for k, peer in enumerate(peer_order(CHIPS))
    ]


def _exchange(send: Any, inbox: Any, send_sems: Any, recv_sems: Any, axis_name: str, fixed: dict[str, Any]) -> None:
    """Send ``send`` to the three peers and wait for theirs; inbox slot k receives position me ^ (k + 1)."""

    copies = _exchange_copies(send, inbox, send_sems, recv_sems, axis_name, fixed)
    for copy in copies:
        copy.start()
    for copy in copies:
        copy.wait_send()
        copy.wait_recv()


def _mesh_order(own: Any, inbox: Any, axis_name: str) -> list[Any]:
    """The four chips' arrays by mesh position 0..3, this chip's own array at its position."""

    me = lax.axis_index(axis_name)
    return [jnp.where(me == rank, own, inbox[jnp.maximum((me ^ rank) - 1, 0)]) for rank in range(CHIPS)]


def _exchange_and_sum(
    acc: Any, inbox: Any, send_sems: Any, recv_sems: Any, axis_name: str, fixed: dict[str, Any]
) -> Any:
    """Send ``acc`` to the three peers, wait for theirs, and add positions 0..3 in order."""

    _exchange(acc, inbox, send_sems, recv_sems, axis_name, fixed)
    return _sum_in_mesh_order(acc, inbox, axis_name)


def _sum_in_mesh_order(acc: Any, inbox: Any, axis_name: str) -> Any:
    """The four chips' arrays added by mesh position 0..3, so every chip holds the same bits."""

    total = None
    for part in _mesh_order(acc[...], inbox, axis_name):
        total = part if total is None else total + part
    return total


def _scatter_gather_rounded(
    acc: Any,
    chunks: Any,
    gathered: Any,
    send_sems: Any,
    recv_sems: Any,
    more_sems: Any,
    axis_name: str,
    fixed: dict[str, Any],
) -> Any:
    """The four chips' sum of ``acc`` rounded to ``gathered.dtype`` (returned as float32), in two rounds.

    Round one is a reduce-scatter in float32: chip r receives column chunk r of the other chips' arrays and adds
    the four chunks in mesh order, the same float32 operations as ``_sum_in_mesh_order``. Round two sends every
    chip's chunk, rounded, to the others. The result is bitwise ``_sum_in_mesh_order`` rounded to the dtype.
    ``chunks`` is ``[4, rows, cols/4]`` float32 and ``gathered`` ``[4, rows, cols/4]``; ``send_sems`` and
    ``more_sems`` hold at least three and four DMA semaphores, ``recv_sems`` at least two.
    """

    me = lax.axis_index(axis_name)
    width = chunks.shape[-1]

    def device(peer: int) -> Any:
        return {**fixed, axis_name: me ^ peer, "tc": 0}

    def columns(rank: Any) -> Any:
        return acc.at[:, pl.ds(pl.multiple_of(rank * width, 128), width)]

    own = pltpu.make_async_copy(columns(me), chunks.at[me], more_sems.at[CHIPS - 1])
    own.start()
    scatter = [
        pltpu.make_async_remote_copy(
            columns(me ^ peer),
            chunks.at[me],
            send_sems.at[k],
            recv_sems.at[0],
            device_id=device(peer),
            device_id_type=pl.DeviceIdType.MESH,
        )
        for k, peer in enumerate(peer_order(CHIPS))
    ]
    for copy in scatter:
        copy.start()
    own.wait()
    # Each peer's chunk lands in chunks[peer] and adds one chunk to recv_sems[0]; wait for three.
    for copy in scatter:
        copy.wait_send()
        copy.wait_recv()
    total = chunks[0]
    for rank in range(1, CHIPS):
        total = total + chunks[rank]
    gathered[me] = total.astype(gathered.dtype)
    gather = [
        pltpu.make_async_remote_copy(
            gathered.at[me],
            gathered.at[me],
            more_sems.at[k],
            recv_sems.at[1],
            device_id=device(peer),
            device_id_type=pl.DeviceIdType.MESH,
        )
        for k, peer in enumerate(peer_order(CHIPS))
    ]
    for copy in gather:
        copy.start()
    for copy in gather:
        copy.wait_send()
        copy.wait_recv()
    return jnp.concatenate([gathered[rank] for rank in range(CHIPS)], axis=-1).astype(jnp.float32)


def _gather_rows(
    own: Any, blocks: Any, send_sems: Any, recv_sems: Any, more_sems: Any, axis_name: str, fixed: dict[str, Any]
) -> Any:
    """The four chips' row blocks ``own`` (``[b, cols]``) stacked by mesh position into ``[4 b, cols]``.

    Chip r's block lands in ``blocks[r]`` on every chip (``blocks`` is ``[4, b, cols]``); every chip ends with the
    same array.
    """

    me = lax.axis_index(axis_name)

    def device(peer: int) -> Any:
        return {**fixed, axis_name: me ^ peer, "tc": 0}

    local = pltpu.make_async_copy(own, blocks.at[me], more_sems.at[CHIPS - 1])
    local.start()
    copies = [
        pltpu.make_async_remote_copy(
            own,
            blocks.at[me],
            send_sems.at[k],
            recv_sems.at[0],
            device_id=device(peer),
            device_id_type=pl.DeviceIdType.MESH,
        )
        for k, peer in enumerate(peer_order(CHIPS))
    ]
    for copy in copies:
        copy.start()
    local.wait()
    # Each peer's block adds one block to recv_sems[0]; wait for three.
    for copy in copies:
        copy.wait_send()
        copy.wait_recv()
    return jnp.concatenate([blocks[rank] for rank in range(CHIPS)], axis=0)


def check_merge_shapes(partial: jax.Array, residual: jax.Array, algorithm: str) -> tuple[int, int, int]:
    """``(rows, cols, partial rows)`` of a merge: ``partial`` has the residual's shape, a quarter of its rows when
    gathering row blocks, or four times its rows when scattering them."""

    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}")
    expected = {"gather_rows": residual.shape[0] // CHIPS, "scatter_rows": residual.shape[0] * CHIPS}.get(
        algorithm, residual.shape[0]
    )
    if (
        partial.ndim != 2
        or partial.dtype != jnp.float32
        or residual.ndim != 2
        or partial.shape[1] != residual.shape[1]
        or partial.shape[0] != expected
        or (algorithm == "gather_rows" and partial.shape[0] * CHIPS != residual.shape[0])
    ):
        raise ValueError("expects a 2-D float32 partial with the rows the merge algorithm implies")
    rows, cols = residual.shape
    if algorithm in ("gather_rows", "scatter_rows") and min(rows, partial.shape[0]) % 8:
        raise ValueError("row blocks must be whole 8-row tiles")
    if algorithm == "scatter_gather":
        check_scatter_gather(rows, cols)
    return rows, cols, partial.shape[0]


def _scatter_rows(
    acc: Any, blocks: Any, send_sems: Any, recv_sems: Any, more_sems: Any, axis_name: str, fixed: dict[str, Any]
) -> Any:
    """This chip's block of rows of the four chips' ``acc`` (``[4 b, cols]``), added by mesh position.

    Chip r receives row block r of the other chips' arrays into ``blocks`` (``[4, b, cols]``) and adds the four
    blocks in mesh order: the same float32 operations as ``_sum_in_mesh_order`` on those rows.
    """

    me = lax.axis_index(axis_name)
    rows = blocks.shape[1]

    def device(peer: int) -> Any:
        return {**fixed, axis_name: me ^ peer, "tc": 0}

    def block(rank: Any) -> Any:
        return acc.at[pl.ds(pl.multiple_of(rank * rows, 8), rows), :]

    own = pltpu.make_async_copy(block(me), blocks.at[me], more_sems.at[CHIPS - 1])
    own.start()
    copies = [
        pltpu.make_async_remote_copy(
            block(me ^ peer),
            blocks.at[me],
            send_sems.at[k],
            recv_sems.at[0],
            device_id=device(peer),
            device_id_type=pl.DeviceIdType.MESH,
        )
        for k, peer in enumerate(peer_order(CHIPS))
    ]
    for copy in copies:
        copy.start()
    own.wait()
    # Each peer's block adds one block to recv_sems[0]; wait for three.
    for copy in copies:
        copy.wait_send()
        copy.wait_recv()
    total = blocks[0]
    for rank in range(1, CHIPS):
        total = total + blocks[rank]
    return total


def check_scatter_gather(rows: int, cols: int) -> None:
    """The shapes the two-round merge supports: whole bf16 tiles per chunk."""

    if rows % 16 or cols % (CHIPS * 128):
        raise ValueError("the scatter-gather merge needs rows divisible by 16 and cols by 512")


def _pack_bf16(value: Any) -> Any:
    """Round float32 ``[rows, cols]`` to bf16 and pack columns ``j`` and ``j + cols/2`` into one uint32."""

    half = value.shape[-1] // 2
    bits = lax.bitcast_convert_type(value.astype(jnp.bfloat16).astype(jnp.float32), jnp.uint32)
    return bits[:, :half] | lax.shift_right_logical(bits[:, half:], jnp.uint32(16))


def _unpack_bf16(packed: Any) -> tuple[Any, Any]:
    """Columns ``:cols/2`` and ``cols/2:`` of a ``_pack_bf16`` array as float32."""

    high = lax.bitcast_convert_type(packed & jnp.uint32(0xFFFF0000), jnp.float32)
    low = lax.bitcast_convert_type(lax.shift_left(packed, jnp.uint32(16)), jnp.float32)
    return high, low


def _exchange_and_sum_bf16(
    acc: Any, send: Any, inbox: Any, send_sems: Any, recv_sems: Any, axis_name: str, fixed: dict[str, Any]
) -> Any:
    """``_exchange_and_sum`` with each chip's array rounded to bf16 before it is sent (half the bytes).

    The four rounded arrays are added in float32 in mesh order, this chip's own one rounded as well, so all
    chips still hold bitwise identical sums.
    """

    send[...] = _pack_bf16(acc[...])
    _exchange(send, inbox, send_sems, recv_sems, axis_name, fixed)
    parts = [_unpack_bf16(part) for part in _mesh_order(send[...], inbox, axis_name)]
    high, low = parts[0]
    for part_high, part_low in parts[1:]:
        high = high + part_high
        low = low + part_low
    return jnp.concatenate([high, low], axis=-1)


def one_shot_psum(x: jax.Array, axis_name: str, other_axes: tuple[str, ...] = ()) -> jax.Array:
    """``lax.psum(x, axis_name)`` for a small 2-D float32 array inside ``shard_map`` over four chips.

    ``other_axes`` names the remaining mesh axes; the peers keep this chip's position on them.
    """
    if x.ndim != 2 or x.dtype != jnp.float32:
        raise ValueError("expects a 2-D float32 array")
    rows, cols = x.shape
    tc_mesh = pltpu.TensorCoreMesh(axis_name="tc", num_cores=1)
    exchanges = CHIPS - 1

    @pl.kernel(
        out_type=pltpu.HBM((rows, cols), x.dtype),
        mesh=tc_mesh,
        scratch_types=(
            pltpu.VMEM((rows, cols), x.dtype),
            pltpu.VMEM((exchanges, rows, cols), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((exchanges,)),
            pltpu.SemaphoreType.DMA((exchanges,)),
        ),
        name="one_shot_psum",
        compiler_params=PARAMS,
        cost_estimate=estimate(
            flops=CHIPS * rows * cols, bytes_accessed=2 * rows * cols * 4, remote_bytes=exchanges * rows * cols * 4
        ),
    )
    def kernel(x_hbm: Any, o_hbm: Any, acc: Any, inbox: Any, sem: Any, send_sems: Any, recv_sems: Any) -> None:
        fixed = {name: lax.axis_index(name) for name in other_axes}
        load = pltpu.make_async_copy(x_hbm, acc, sem)
        load.start()
        _barrier(axis_name, fixed)
        load.wait()
        acc[...] = _exchange_and_sum(acc, inbox, send_sems, recv_sems, axis_name, fixed)
        store = pltpu.make_async_copy(acc, o_hbm, sem)
        store.start()
        store.wait()

    return cast(jax.Array, kernel(x))


class PerLayerInput(NamedTuple):
    """Gemma 4's per-layer-input block after the MLP residual add, applied inside ``one_shot_psum_norm``.

    Adds ``rms_norm(gelu(x @ gate) * inputs[:, layer * width : (layer + 1) * width] @ projection, post_weight)``
    to the residual stream. ``inputs`` holds every layer's per-layer input as ``[rows, layers * width]``;
    the kernel reads only this layer's slice.
    """

    gate: jax.Array  # [cols, width]
    projection: jax.Array  # [width, cols]
    inputs: jax.Array  # [rows, layers * width]
    layer: int
    post_weight: jax.Array  # [cols] or [1, cols]


def one_shot_psum_norm(
    partial: jax.Array,
    residual: jax.Array,
    post_weight: jax.Array,
    next_weight: jax.Array | None,
    *,
    eps: float,
    axis_name: str,
    other_axes: tuple[str, ...] = (),
    per_layer_input: PerLayerInput | None = None,
    scale: jax.Array | None = None,
    exchange_dtype: Any = jnp.float32,
    algorithm: str = "one_shot",
) -> tuple[jax.Array, jax.Array | None]:
    """Sum ``partial`` over four chips and apply the decoder's norm and residual in the same kernel.

    Returns ``x = residual + rms_norm(sum, post_weight)`` and, when ``next_weight`` is given,
    ``rms_norm(x, next_weight)`` (the input of the next block). With ``per_layer_input`` the
    per-layer-input block is added to ``x`` and with ``scale`` (the layer scalar as a ``[cols]``
    row) ``x`` is multiplied by it, both before the next norm, as at the end of a Gemma 4 layer.
    The rounding points follow the separate XLA ops of the TP decoder: the float32 sum is rounded
    to ``residual.dtype``, each norm, matmul, activation, add and multiply runs in float32 and is
    rounded. ``partial`` is a 2-D float32 array; the weights have shape ``[cols]`` or ``[1, cols]``.
    The per-layer-input weights are read while the chips exchange their sums. With
    ``exchange_dtype=jnp.bfloat16`` each chip sends its partial sum rounded to bf16 and the four
    rounded arrays are added in float32 in mesh order (``cols`` must be a multiple of 256).
    ``algorithm="scatter_gather"`` (float32 exchange only) sums in two rounds with the same result bits;
    ``algorithm="gather_rows"`` stacks the chips' row blocks (``partial`` holds this chip's ``rows/4`` rows).
    """
    rows, cols, local = check_merge_shapes(partial, residual, algorithm)
    if jnp.dtype(exchange_dtype) not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16)):
        raise ValueError("exchange_dtype must be float32 or bfloat16")
    packed = jnp.dtype(exchange_dtype) == jnp.dtype(jnp.bfloat16)
    if packed and cols % 256:
        raise ValueError("the bf16 exchange packs column halves and needs cols divisible by 256")
    two_rounds = algorithm == "scatter_gather"
    gather = algorithm == "gather_rows"
    scatter = algorithm == "scatter_rows"
    if packed and algorithm != "one_shot":
        raise ValueError("the bf16 exchange works with the one-shot merge only")
    inbox_type = pltpu.VMEM((CHIPS - 1, rows, cols // 2), jnp.uint32) if packed else None
    if two_rounds:
        inbox_type = pltpu.VMEM((CHIPS, rows, cols // CHIPS), jnp.float32)
    if gather:
        inbox_type = pltpu.VMEM((CHIPS, local, cols), jnp.float32)
    if scatter:
        inbox_type = pltpu.VMEM((CHIPS, rows, cols), jnp.float32)
    dtype = residual.dtype
    rows_in = [post_weight] + ([next_weight] if next_weight is not None else [])
    ple_row = scale_row = None
    if per_layer_input is not None:
        ple_row = len(rows_in)
        rows_in.append(per_layer_input.post_weight)
    if scale is not None:
        scale_row = len(rows_in)
        rows_in.append(scale)
    rows_in = [row.reshape(1, cols).astype(post_weight.dtype) for row in rows_in]
    extra = (
        [] if per_layer_input is None else [per_layer_input.gate, per_layer_input.projection, per_layer_input.inputs]
    )
    width = 0 if per_layer_input is None else per_layer_input.gate.shape[1]
    tc_mesh = pltpu.TensorCoreMesh(axis_name="tc", num_cores=1)
    exchanges = CHIPS - 1
    outputs = 2 if next_weight is not None else 1
    ple_scratch = (
        ()
        if per_layer_input is None
        else (
            pltpu.VMEM((cols, width), per_layer_input.gate.dtype),
            pltpu.VMEM((width, cols), per_layer_input.projection.dtype),
            pltpu.VMEM((rows, width), per_layer_input.inputs.dtype),
        )
    )
    loads_count = 2 + len(rows_in) + len(extra)

    def rounded(value: Any) -> Any:
        return value.astype(dtype).astype(jnp.float32)

    def norm(value: Any, weight: Any) -> Any:
        factor = lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + jnp.float32(eps))
        return rounded(value * factor * weight.astype(jnp.float32))

    @pl.kernel(
        out_type=tuple(pltpu.HBM((rows, cols), dtype) for _ in range(outputs)),
        mesh=tc_mesh,
        scratch_types=(
            pltpu.VMEM((local, cols), jnp.float32),
            inbox_type or pltpu.VMEM((exchanges, rows, cols), jnp.float32),
            pltpu.VMEM((rows, cols), dtype),
            pltpu.VMEM((len(rows_in), 1, cols), post_weight.dtype),
            pltpu.VMEM((outputs, rows, cols), dtype),
            *((pltpu.VMEM((rows, cols // 2), jnp.uint32),) if packed else ()),
            *((pltpu.VMEM((CHIPS, rows, cols // CHIPS), dtype),) if two_rounds else ()),
            *ple_scratch,
            pltpu.SemaphoreType.DMA((loads_count,)),
            pltpu.SemaphoreType.DMA((outputs,)),
            pltpu.SemaphoreType.DMA((exchanges,)),
            pltpu.SemaphoreType.DMA((exchanges,)),
            *((pltpu.SemaphoreType.DMA((CHIPS,)),) if two_rounds or gather or scatter else ()),
        ),
        name="one_shot_psum_norm_bf16"
        if packed
        else "psum_norm_scatter_gather"
        if two_rounds
        else "gather_rows_norm"
        if gather
        else "scatter_rows_norm"
        if scatter
        else "one_shot_psum_norm",
        compiler_params=PARAMS,
        cost_estimate=estimate(
            flops=16 * rows * cols + (4 * rows * cols * width if per_layer_input is not None else 0),
            transcendentals=2 * rows,
            bytes_accessed=rows * cols * (4 + 2 + 2 * outputs)
            + sum(int(a.size) * a.dtype.itemsize for a in extra)
            + len(rows_in) * cols * 2,
            remote_bytes=exchanges * rows * cols * (2 if packed else 4),
        ),
    )
    def kernel(*raw_refs: Any) -> None:
        refs = list(raw_refs)
        partial_hbm, residual_hbm = refs.pop(0), refs.pop(0)
        row_hbm = [refs.pop(0) for _ in rows_in]
        extra_hbm = [refs.pop(0) for _ in extra]
        out_hbm = [refs.pop(0) for _ in range(outputs)]
        acc, inbox, res, weight, out = (refs.pop(0) for _ in range(5))
        send = refs.pop(0) if packed else None
        gathered = refs.pop(0) if two_rounds else None
        ple_vmem = [refs.pop(0) for _ in ple_scratch]
        load_sems, store_sems, send_sems, recv_sems, *more_sems = refs
        fixed = {name: lax.axis_index(name) for name in other_axes}
        sources = [residual_hbm, *row_hbm]
        targets = [res, *(weight.at[k] for k in range(len(rows_in)))]
        if per_layer_input is not None:
            gate_hbm, projection_hbm, inputs_hbm = extra_hbm
            start = per_layer_input.layer * width
            sources += [gate_hbm, projection_hbm, inputs_hbm.at[:, start : start + width]]
            targets += ple_vmem
        first = pltpu.make_async_copy(partial_hbm, acc, load_sems.at[0])
        others = [
            pltpu.make_async_copy(source, target, load_sems.at[1 + k])
            for k, (source, target) in enumerate(zip(sources, targets, strict=True))
        ]
        first.start()
        for load in others:
            load.start()
        _barrier(axis_name, fixed)
        first.wait()
        if packed:
            total = _exchange_and_sum_bf16(acc, send, inbox, send_sems, recv_sems, axis_name, fixed)
        elif two_rounds:
            total = _scatter_gather_rounded(acc, inbox, gathered, send_sems, recv_sems, more_sems[0], axis_name, fixed)
        elif gather:
            total = _gather_rows(acc, inbox, send_sems, recv_sems, more_sems[0], axis_name, fixed)
        elif scatter:
            total = _scatter_rows(acc, inbox, send_sems, recv_sems, more_sems[0], axis_name, fixed)
        else:
            total = _exchange_and_sum(acc, inbox, send_sems, recv_sems, axis_name, fixed)
        for load in others:
            load.wait()
        x = rounded(res[...].astype(jnp.float32) + norm(rounded(total), weight[0]))
        if per_layer_input is not None:
            gate, projection, inputs = ple_vmem
            # Use DEFAULT explicitly for single-pass BF16 accumulation; retain configured precision for FP32.
            dot_precision = DEFAULT if dtype == jnp.bfloat16 else None
            gated = rounded(
                jnp.dot(x.astype(dtype), gate[...], preferred_element_type=jnp.float32, precision=dot_precision)
            )
            gated = rounded(jax.nn.gelu(gated, approximate=True))
            mixed = rounded(gated * inputs[...].astype(jnp.float32))
            projected = rounded(
                jnp.dot(
                    mixed.astype(dtype), projection[...], preferred_element_type=jnp.float32, precision=dot_precision
                )
            )
            x = rounded(x + norm(projected, weight[ple_row]))
        if scale is not None:
            x = rounded(x * weight[scale_row].astype(jnp.float32))
        out[0] = x.astype(dtype)
        if next_weight is not None:
            out[1] = norm(x, weight[1]).astype(dtype)
        stores = [pltpu.make_async_copy(out.at[k], ref, store_sems.at[k]) for k, ref in enumerate(out_hbm)]
        for store in stores:
            store.start()
        for store in stores:
            store.wait()

    result = kernel(partial, residual, *rows_in, *extra)
    result = result if isinstance(result, (tuple, list)) else (result,)
    return (result[0], result[1]) if next_weight is not None else (result[0], None)


def norm_gather(
    partial: jax.Array,
    residual: jax.Array,
    post_weight: jax.Array,
    next_weight: jax.Array,
    *,
    eps: float,
    axis_name: str,
    other_axes: tuple[str, ...] = (),
) -> tuple[jax.Array, jax.Array]:
    """``x = residual + rms_norm(partial)`` on this chip's rows and ``rms_norm(x, next_weight)`` of all chips' rows.

    The first merge of a layer when attention is split by rows and every chip keeps only its own rows of the
    residual stream: ``partial`` (``[b, cols]`` float32) is the complete attention output of this chip's rows and
    ``residual`` their residual. Returns ``x`` (``[b, cols]``) and the next block's input for all rows, stacked by
    mesh position (``[4 b, cols]``). The rounding points are those of ``one_shot_psum_norm``, so the rows equal
    the ``gather_rows`` merge bit for bit. With a bf16 residual the gathered input, already rounded to bf16, is sent
    packed (``_pack_bf16``, columns ``j`` and ``j + cols/2`` in one uint32): half the bytes on every link.
    """
    if partial.ndim != 2 or partial.dtype != jnp.float32 or residual.shape != partial.shape or partial.shape[0] % 8:
        raise ValueError("expects a 2-D float32 partial of whole 8-row tiles and a residual of the same shape")
    rows, cols = partial.shape
    dtype = residual.dtype
    weights = [w.reshape(1, cols).astype(post_weight.dtype) for w in (post_weight, next_weight)]
    exchanges = CHIPS - 1
    # Packed columns stay whole 128-lane tiles.
    packed = dtype == jnp.bfloat16 and cols % 256 == 0
    sent = (rows, cols // 2) if packed else (rows, cols)
    sent_dtype = jnp.uint32 if packed else jnp.float32

    def rounded(value: Any) -> Any:
        return value.astype(dtype).astype(jnp.float32)

    def norm(value: Any, weight: Any) -> Any:
        factor = lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + jnp.float32(eps))
        return rounded(value * factor * weight.astype(jnp.float32))

    @pl.kernel(
        out_type=(pltpu.HBM((rows, cols), dtype), pltpu.HBM((CHIPS * rows, cols), dtype)),
        mesh=pltpu.TensorCoreMesh(axis_name="tc", num_cores=1),
        scratch_types=(
            pltpu.VMEM((rows, cols), jnp.float32),
            pltpu.VMEM((rows, cols), dtype),
            pltpu.VMEM((2, 1, cols), post_weight.dtype),
            pltpu.VMEM((rows, cols), dtype),
            pltpu.VMEM(sent, sent_dtype),
            pltpu.VMEM((CHIPS, *sent), sent_dtype),
            pltpu.VMEM((CHIPS * rows, cols), dtype),
            pltpu.SemaphoreType.DMA((4,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((exchanges,)),
            pltpu.SemaphoreType.DMA((exchanges,)),
            pltpu.SemaphoreType.DMA((CHIPS,)),
        ),
        name="norm_gather_rows",
        compiler_params=PARAMS,
    )
    def kernel(
        partial_hbm: Any,
        residual_hbm: Any,
        post_hbm: Any,
        next_hbm: Any,
        x_hbm: Any,
        h_hbm: Any,
        acc: Any,
        res: Any,
        weight: Any,
        x_v: Any,
        h_v: Any,
        blocks: Any,
        gathered: Any,
        load_sems: Any,
        store_sems: Any,
        send_sems: Any,
        recv_sems: Any,
        more_sems: Any,
    ) -> None:
        fixed = {name: lax.axis_index(name) for name in other_axes}
        loads = [
            pltpu.make_async_copy(source, target, load_sems.at[k])
            for k, (source, target) in enumerate(
                ((partial_hbm, acc), (residual_hbm, res), (post_hbm, weight.at[0]), (next_hbm, weight.at[1]))
            )
        ]
        for load in loads:
            load.start()
        _barrier(axis_name, fixed)
        for load in loads:
            load.wait()
        x = rounded(res[...].astype(jnp.float32) + norm(rounded(acc[...]), weight[0]))
        x_v[...] = x.astype(dtype)
        store_x = pltpu.make_async_copy(x_v, x_hbm, store_sems.at[0])
        store_x.start()
        h = norm(x, weight[1])
        h_v[...] = _pack_bf16(h) if packed else h
        stacked = _gather_rows(h_v, blocks, send_sems, recv_sems, more_sems, axis_name, fixed)
        if packed:
            stacked = jnp.concatenate(_unpack_bf16(stacked), axis=-1)
        gathered[...] = stacked.astype(dtype)
        store_h = pltpu.make_async_copy(gathered, h_hbm, store_sems.at[1])
        store_h.start()
        store_x.wait()
        store_h.wait()

    x, h = kernel(partial, residual, *weights)
    return x, h
