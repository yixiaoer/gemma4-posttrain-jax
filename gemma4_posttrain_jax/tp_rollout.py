"""Tensor-parallel rollout layout for decode (prototype).

Mesh ``(dp, tp)``: batch rows are split over ``dp``; within a ``tp`` group every chip
holds all of its rows and one slice of each large matrix:

* Q split by query head, O by input head, reduced with one ``psum`` per layer;
* MLP gate/up split by the intermediate dimension, down by the same dimension,
  reduced with one ``psum`` per layer;
* the single K/V head, its cache and all small weights replicated;
* the token embedding (also the tied LM head) and the PLE table split by vocabulary.
  A lookup takes the local row or zero and sums over ``tp``; adding zeros is exact.

Reductions run in float32 and are rounded to the activation dtype once, so the result
differs from the unsharded matmul only by float32 summation order. Everything runs
inside ``shard_map``, so XLA cannot reintroduce weight gathers.

This module covers prefill, cached decode, selected-token scoring on a fixed history and
full-vocabulary sampling. Each chip draws the Gumbel noise of its own vocabulary slice with
the same counters as ``jax.random``'s partitionable threefry, so sampling consumes the same
random stream as ``sampler.generate``; only top_k=0 and top_p=1 are supported.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax._src.random.threefry2x32 import threefry2x32_p
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from gemma4_posttrain_jax.chunked_prefill import chunked_prefill
from gemma4_posttrain_jax.model import (
    FULL,
    SLIDING,
    AttentionParams,
    DecoderLayerParams,
    Gemma4TextConfig,
    Gemma4TextParams,
    KVCache,
    LayerKVCache,
    MLPParams,
    PerLayerInputParams,
    RMSNormParams,
    _scale,
    _update_layer_cache,
    forward_per_layer_input,
    forward_rms_norm,
    forward_rotary_embedding,
    init_kv_cache,
    is_kv_shared,
    make_cached_qk_masks,
    make_chunked_qk_masks,
    make_qk_masks,
    make_rotary_values,
    update_cache_mask,
)
from gemma4_posttrain_jax.sampler import RolloutBatch, SamplerConfig, _is_eos

DP_AXIS = "dp"
TP_AXIS = "tp"
# "one_shot_fused" also applies the norm and residual that follow each block's reduction in the same kernel;
# "one_shot_fused_bf16" does the same with each chip's partial sum rounded to bf16 before the exchange.
# "one_shot_fused_mlp" is "one_shot_fused" with the attention block's merge and the MLP in one kernel, which reads
# the MLP weights while the chips exchange (kernels.pallas.merge_mlp; "_high" keeps gate, up and the activation
# in float32). A "_sg" suffix merges in two rounds with the same result bits (reduce-scatter in float32, then an
# all-gather of the rounded chunks; ici_allreduce._scatter_gather_rounded) when the rows split into 16-row tiles.
# A "_rows" suffix splits attention by rows instead of heads (``attention_rows``): each chip computes all heads for
# a quarter of the rows and reads only their K/V; the first merge of a layer then stacks the chips' rows.
# "_rows_local" also keeps only each chip's rows of the residual stream between the merges: the merge after the MLP
# adds each chip's own rows only (``scatter_rows``) and the first merge gathers the MLP input (``norm_gather``); the
# The merge kernels match their isolated references bit for bit; changed attention shapes still need model checks.
# "_high" without "_mlp" computes the decode MLP with the arithmetic of the kernel's high-precision mode in XLA ops
# (``_mlp(high=True)``).
ALLREDUCE = (
    "xla",
    "one_shot",
    "one_shot_fused",
    "one_shot_fused_bf16",
    "one_shot_fused_mlp",
    "one_shot_fused_mlp_high",
    "one_shot_fused_sg",
    "one_shot_fused_mlp_sg",
    "one_shot_fused_mlp_high_sg",
    "xla_rows",
    "one_shot_fused_rows",
    "one_shot_fused_sg_rows",
    "one_shot_fused_mlp_high_sg_rows",
    "one_shot_fused_high_sg",
    "one_shot_fused_high_sg_rows",
    "one_shot_fused_rows_local",
    "one_shot_fused_high_rows_local",
)
# Arrays up to this size (per chip, as float32) use the one-shot kernel; larger ones (prefill) keep XLA's ring.
ONE_SHOT_MAX_BYTES = 256 * 1024
# The merge + MLP kernel needs at least this many rows per merge. At 32 rows (B8/G4) complete generation is 11%
# faster; at 8 rows XLA's memory-space assignment then stopped prefetching into CMEM inside the decode loop
# (147 MB -> 0.1 MB per step) and generation was 17% slower, so fewer rows keep one_shot_psum_norm and XLA's MLP.
MLP_KERNEL_MIN_ROWS = 32
# Compiler options for programs that run ``tp_generate``. XLA's memory-space assignment estimates a
# while loop whose trip count it cannot compute (the generation loop stops when every row is done)
# as running 5 times. With the fused per-layer-input kernel, 5 left the loop without the weight
# prefetches into on-chip memory that the fixed-length history loop gets (2.42 ms per decode step);
# 1 restores them (2.16 ms), while 100 or more made it slower. Measured on JAX 0.11.1 / libtpu
# 0.0.46; recheck after upgrading either.
GENERATE_COMPILER_OPTIONS = {"xla_tpu_memory_space_assignment_while_execution_count": "1"}


def generate_compiler_options(allreduce: str) -> dict[str, str] | None:
    """Compiler options for a program running ``tp_generate`` (only measured with the fused kernel)."""

    return dict(GENERATE_COMPILER_OPTIONS) if allreduce.startswith("one_shot_fused") else None


# What "--tp-allreduce auto" (the default of the training and evaluation scripts) picks. When every generation call
# of the run gives each chip whole 8-row blocks (32 rows on four chips, e.g. B8/G4): the fastest mode that passed
# numerical and complete-run checks (it is not bit-identical to AUTO_HEADS).
# Otherwise the merge kernel without the row split.
AUTO_ROWS = "one_shot_fused_high_rows_local"
AUTO_HEADS = "one_shot_fused"


def resolve_allreduce(
    requested: str | None,
    *,
    layout: str,
    tp: int,
    rows: Iterable[int],
    saved: Mapping[str, Any] | None = None,
    decode_kernel: bool = False,
    prefill_widths: Sequence[int] = (),
) -> str:
    """The merge mode for ``--tp-allreduce requested``; ``None`` (option omitted) means "auto".

    "auto" keeps the mode of the checkpoint the run starts from (``saved`` is its ``run_config``): resuming must not
    change it, and TP checkpoints without the field used XLA's merges. Without a checkpoint it picks ``AUTO_ROWS``
    when every generation call has ``rows`` divisible by ``8 * tp`` and no option outside that mode's tests is on
    (decode attention kernel, narrow prefill), ``AUTO_HEADS`` otherwise, and XLA's merges where the one-shot kernels
    do not apply (a layout other than tp, tp != 4).
    """

    if requested not in (None, "auto"):
        return requested
    if layout != "tp":
        return "xla"
    if saved is not None:
        return str(saved.get("tp_allreduce", "xla"))
    if tp != 4:
        return "xla"
    if not decode_kernel and not prefill_widths and all(count % (8 * tp) == 0 for count in rows):
        return AUTO_ROWS
    return AUTO_HEADS


def make_tp_mesh(dp: int, tp: int, devices: Any | None = None) -> Mesh:
    """``(dp, tp)`` mesh; consecutive devices share a ``tp`` group (neighbours on a v4 2x2 slice)."""

    selected = np.asarray(jax.devices() if devices is None else devices)
    if selected.size != dp * tp:
        raise ValueError(f"mesh {dp}x{tp} needs {dp * tp} devices, got {selected.size}")
    return Mesh(selected.reshape(dp, tp), (DP_AXIS, TP_AXIS))


def check_tp_config(config: Gemma4TextConfig, tp: int) -> None:
    heads = config.num_attention_heads // config.num_key_value_heads
    if config.num_attention_heads % tp or heads % tp:
        raise ValueError(f"query heads per K/V head ({heads}) must be divisible by tp={tp}")
    if config.intermediate_size % tp or config.vocab_size % tp or config.vocab_size_per_layer_input % tp:
        raise ValueError("intermediate and vocabulary sizes must be divisible by tp")
    if config.attention_k_eq_v:
        raise ValueError("the TP prototype does not cover attention_k_eq_v")


def tp_param_specs(params: Gemma4TextParams, *, attention_rows: bool = False) -> Any:
    """PartitionSpec tree matching ``params``; ``None`` leaves stay ``None``.

    ``attention_rows`` replicates the query and output projections (attention split by rows).
    """

    # Reuse parameter containers for a sharding tree whose leaves are PartitionSpec objects.
    spec = cast(Any, P)

    def spec_or_none(value: Any, spec: P) -> Any:
        return None if value is None else spec

    norm = RMSNormParams(spec())

    def layer_spec(layer: DecoderLayerParams) -> DecoderLayerParams:
        attention = layer.attention
        return DecoderLayerParams(
            input_norm=norm,
            attention=AttentionParams(
                q_proj=spec() if attention_rows else spec(None, TP_AXIS, None, None),
                k_proj=spec_or_none(attention.k_proj, spec()),
                v_proj=spec_or_none(attention.v_proj, spec()),
                o_proj=spec() if attention_rows else spec(TP_AXIS, None, None, None),
                q_norm=norm,
                k_norm=None if attention.k_norm is None else norm,
            ),
            post_attention_norm=norm,
            pre_feedforward_norm=norm,
            mlp=MLPParams(spec(None, TP_AXIS), spec(None, TP_AXIS), spec(TP_AXIS, None)),
            post_feedforward_norm=norm,
            per_layer_input=None if layer.per_layer_input is None else PerLayerInputParams(spec(), spec(), norm),
            layer_scalar=spec(),
        )

    return Gemma4TextParams(
        embed_tokens=spec(TP_AXIS, None),
        embed_tokens_per_layer=spec_or_none(params.embed_tokens_per_layer, spec(TP_AXIS, None, None)),
        per_layer_model_projection=spec_or_none(params.per_layer_model_projection, spec()),
        per_layer_projection_norm=None if params.per_layer_projection_norm is None else norm,
        layers=tuple(layer_spec(layer) for layer in params.layers),
        final_norm=norm,
    )


def tp_cache_specs(cache: KVCache, *, attention_rows: bool = False) -> KVCache:
    """Cache rows split over ``dp`` and replicated over ``tp``; with ``attention_rows`` the K/V rows are split
    over ``(dp, tp)`` while the key mask stays split over ``dp`` only."""

    spec = cast(Any, P)
    rows = spec((DP_AXIS, TP_AXIS)) if attention_rows else spec(DP_AXIS)
    layers = tuple(None if layer is None else LayerKVCache(rows, rows) for layer in cache.layers)
    return KVCache(layers, spec(DP_AXIS), spec(DP_AXIS) if cache.length.ndim else spec())


def _init_cache(config: Gemma4TextConfig, rows: int, capacity: int, dtype: Any, row_split: int = 1) -> KVCache:
    """One chip's empty cache: K/V for ``rows // row_split`` rows (attention split by rows), the key mask for all."""

    cache = init_kv_cache(config, rows // row_split, capacity, dtype)
    return cache._replace(key_mask=jnp.zeros((rows, capacity), jnp.bool_))


def _named(mesh: Mesh, specs: Any) -> Any:
    return jax.tree.map(lambda spec: NamedSharding(mesh, spec), specs, is_leaf=lambda x: isinstance(x, P))


def prepare_tp_params(
    params_f32: Gemma4TextParams, mesh: Mesh, *, dtype: Any = jnp.bfloat16, attention_rows: bool = False
) -> tuple[Any, Any]:
    """Cast the FP32 master to the rollout dtype in the TP layout (same values as other layouts)."""

    shardings = _named(mesh, tp_param_specs(params_f32, attention_rows=attention_rows))
    cast = jax.jit(lambda tree: jax.tree.map(lambda value: value.astype(dtype), tree), out_shardings=shardings)
    return cast(params_f32), shardings


# --- per-device computation (inside shard_map) -------------------------------------------------------


def attention_rows(allreduce: str) -> bool:
    """Whether the mode splits attention by rows (a ``_rows`` suffix) instead of by heads.

    E2B has one K/V head, so with attention split by heads every chip reads the whole K/V cache; split by rows,
    each chip computes all heads for ``rows / tp`` rows, keeps only their cache and reads a quarter of the K/V.
    The query and output projections are then replicated (``tp_param_specs(attention_rows=True)``), the cache
    rows are split over ``(dp, tp)`` (``tp_cache_specs``), and the layer's first merge stacks row blocks.
    """

    return allreduce.endswith(("_rows", "_rows_local"))


def _tp_sum(x: Array, allreduce: str) -> Array:
    """Sum over ``tp``. Small arrays can use the one-shot kernel (four chips), summed in float32."""

    if allreduce.startswith("one_shot") and x.size * 4 <= ONE_SHOT_MAX_BYTES:
        from gemma4_posttrain_jax.pallas.ici_allreduce import one_shot_psum

        flat = x.astype(jnp.float32).reshape(-1, x.shape[-1])
        return one_shot_psum(flat, TP_AXIS, other_axes=(DP_AXIS,)).reshape(x.shape).astype(x.dtype)
    return cast(Array, lax.psum(x, TP_AXIS))


def _check_allreduce(allreduce: str, mesh: Mesh) -> None:
    if allreduce not in ALLREDUCE:
        raise ValueError(f"allreduce must be one of {ALLREDUCE}")
    if allreduce.startswith("one_shot") and mesh.shape[TP_AXIS] != 4:
        raise ValueError("the one-shot all-reduce needs tp=4")


def _vocab_lookup(table: Array, ids: Array, allreduce: str = "xla") -> Array:
    """Rows of a vocabulary-split table: local row or zero, summed over ``tp`` (exact)."""

    rows = table.shape[0]
    local = ids - lax.axis_index(TP_AXIS) * rows
    valid = (local >= 0) & (local < rows)
    found = jnp.take(table, jnp.clip(local, 0, rows - 1), axis=0)
    found = jnp.where(valid.reshape(valid.shape + (1,) * (found.ndim - valid.ndim)), found, jnp.zeros((), found.dtype))
    return _tp_sum(found, allreduce)


def _reduced(partial: Array, dtype: Any, allreduce: str = "xla") -> Array:
    return _tp_sum(partial, allreduce).astype(dtype)


def _row_norm_weights(params: Gemma4TextParams, allreduce: str) -> Gemma4TextParams:
    """Norm weights and layer scalars read by the fused kernel as ``[1, M]`` rows.

    The kernel takes 2-D rows; reshaping a 1-D bf16 weight changes its TPU layout and costs a
    copy. Doing it once per call, outside the decode loop, avoids one copy per kernel call; an
    optimization barrier keeps XLA from moving the broadcasts back into the loop. The other code
    paths broadcast the rows exactly like the original weights and scalars.
    """

    if not allreduce.startswith("one_shot_fused"):
        return params
    width = params.embed_tokens.shape[-1]

    def row(norm: RMSNormParams) -> RMSNormParams:
        return RMSNormParams(norm.weight.reshape(1, -1))

    def layer_rows(layer: DecoderLayerParams) -> DecoderLayerParams:
        ple = layer.per_layer_input
        return layer._replace(
            input_norm=row(layer.input_norm),
            post_attention_norm=row(layer.post_attention_norm),
            pre_feedforward_norm=row(layer.pre_feedforward_norm),
            post_feedforward_norm=row(layer.post_feedforward_norm),
            per_layer_input=None if ple is None else ple._replace(post_norm=row(ple.post_norm)),
            layer_scalar=jnp.broadcast_to(layer.layer_scalar.reshape(1, 1), (1, width)),
        )

    layers, final_norm = lax.optimization_barrier((tuple(map(layer_rows, params.layers)), row(params.final_norm)))
    return params._replace(layers=layers, final_norm=final_norm)


def _reduced_norm_residual(
    partial: Array,
    residual: Array,
    post: RMSNormParams,
    following: RMSNormParams | None,
    eps: float,
    *,
    layer_end: tuple[DecoderLayerParams, Array | None, int] | None = None,
    exchange_dtype: Any = jnp.float32,
    algorithm: str = "one_shot",
) -> tuple[Array, Array | None]:
    """``residual + rms_norm(psum(partial))`` and optionally the next block's norm, in one kernel.

    ``layer_end=(layer, per_layer_inputs, index)`` also applies the end of the layer after the MLP:
    the per-layer-input block (with ``per_layer_inputs`` as ``[rows, layers * width]``) and the
    layer scalar.
    """

    from gemma4_posttrain_jax.pallas.ici_allreduce import PerLayerInput, one_shot_psum_norm

    shape = residual.shape
    ple = scale = None
    if layer_end is not None:
        layer, inputs, index = layer_end
        scale = layer.layer_scalar
        if layer.per_layer_input is not None:
            assert inputs is not None
            block = layer.per_layer_input
            ple = PerLayerInput(block.gate, block.projection, inputs, index, block.post_norm.weight)
    x, h = one_shot_psum_norm(
        partial.reshape(-1, shape[-1]),
        residual.reshape(-1, shape[-1]),
        post.weight,
        None if following is None else following.weight,
        eps=eps,
        axis_name=TP_AXIS,
        other_axes=(DP_AXIS,),
        per_layer_input=ple,
        scale=scale,
        exchange_dtype=exchange_dtype,
        algorithm=algorithm,
    )
    return x.reshape(shape), None if h is None else h.reshape(shape)


def _norm_gather(
    partial: Array, residual: Array, post: RMSNormParams, pre: RMSNormParams, eps: float
) -> tuple[Array, Array]:
    """This chip's rows of ``x = residual + rms_norm(partial)`` and ``rms_norm(x)`` of all rows (``norm_gather``)."""

    from gemma4_posttrain_jax.pallas.ici_allreduce import norm_gather

    shape = residual.shape
    x, h = norm_gather(
        partial.reshape(-1, shape[-1]),
        residual.reshape(-1, shape[-1]),
        post.weight,
        pre.weight,
        eps=eps,
        axis_name=TP_AXIS,
        other_axes=(DP_AXIS,),
    )
    return x.reshape(shape), h.reshape((-1,) + shape[1:])


def _merge_mlp(
    partial: Array,
    residual: Array,
    post: RMSNormParams,
    pre: RMSNormParams,
    mlp: MLPParams,
    eps: float,
    precision: str,
    algorithm: str = "one_shot",
) -> tuple[Array, Array]:
    """``x = residual + rms_norm(psum(partial))`` and the float32 MLP partial of ``rms_norm(x)`` in one kernel."""

    from gemma4_posttrain_jax.pallas.merge_mlp import merge_mlp

    shape = residual.shape
    x, out = merge_mlp(
        partial.reshape(-1, shape[-1]),
        residual.reshape(-1, shape[-1]),
        post.weight,
        pre.weight,
        mlp.gate_proj,
        mlp.up_proj,
        mlp.down_proj,
        eps=eps,
        axis_name=TP_AXIS,
        other_axes=(DP_AXIS,),
        precision=precision,
        algorithm=algorithm,
    )
    return x.reshape(shape), out.reshape(shape)


def _attention(
    params: AttentionParams,
    seq: Array,
    qk_mask: Array,
    rotary_values: Any,
    *,
    layer_index: int,
    config: Gemma4TextConfig,
    shared_kv: dict[str, tuple[Array, Array]],
    kv_cache: LayerKVCache | None,
    cache_length: Array,
    cache_mode: str,
    allreduce: str = "xla",
    reduce: bool = True,
    decode_ranges: tuple[Array, Array] | None = None,
    decode_window: tuple[Array, tuple[tuple[int, int], ...]] | None = None,
    row_split: int = 1,
) -> tuple[Array, LayerKVCache | None]:
    """Same formulas as ``model.forward_attention`` with local query heads and one reduction.

    ``reduce=False`` returns this chip's float32 partial output instead of the sum.
    ``row_split > 1`` (attention split by rows) computes all heads for this chip's ``rows / row_split`` rows
    against its own cache rows; the output is then complete for those rows: ``reduce`` gathers the rounded rows
    of all chips, ``reduce=False`` returns this chip's float32 rows.
    ``decode_ranges=(starts, length)`` computes one-token attention on global layers with one KV head
    by ``kernels.pallas.decode_attention``, which reads only the cache slots ``[min(starts), length)``.
    It treats every slot in ``[starts[b], length)`` as real: true for rows still generating (the
    prompt is left-padded and the generated tokens follow it); rows that have finished attend to
    their trailing padding too, which only changes their own unused outputs.
    ``decode_window=(index, windows)`` computes one-token attention on global layers over the static
    cache slice ``windows[index]`` only (``lax.switch``); the caller picks a window that holds every
    unmasked slot, so the result equals the full-cache attention up to rounding.
    """

    layer_type = config.layer_types[layer_index]
    eps = config.rms_norm_eps
    dtype = seq.dtype
    if row_split > 1:
        if decode_ranges is not None or decode_window is not None:
            raise ValueError("attention split by rows covers the XLA attention only")
        local = seq.shape[0] // row_split
        start = lax.axis_index(TP_AXIS) * local

        def mine(value: Array) -> Array:
            return lax.dynamic_slice_in_dim(value, start, local, axis=0)

        seq, qk_mask, rotary_values = mine(seq), mine(qk_mask), jax.tree.map(mine, rotary_values)
        if cache_length.ndim:
            cache_length = mine(cache_length)
    q = jnp.einsum("bsm,mrhd->bsrhd", seq, params.q_proj)
    q = forward_rms_norm(params.q_norm, q, eps)
    q = forward_rotary_embedding(q, rotary_values)

    new_cache = None
    if is_kv_shared(config, layer_index):
        k, v = shared_kv[layer_type]
    else:
        assert params.k_proj is not None and params.v_proj is not None
        k = jnp.einsum("btm,mhd->bthd", seq, params.k_proj)
        v = jnp.einsum("btm,mhd->bthd", seq, params.v_proj)
        k = forward_rms_norm(params.k_norm, k, eps)
        k = forward_rotary_embedding(k, rotary_values)
        v = forward_rms_norm(None, v, eps)
        if kv_cache is not None:
            new_cache = _update_layer_cache(kv_cache, k, v, start=cache_length, circular=layer_type == SLIDING)
            if cache_mode == "append":
                k, v = new_cache
            elif cache_mode == "extend":
                if layer_type == SLIDING:
                    k, v = jnp.concatenate((kv_cache.key, k), axis=1), jnp.concatenate((kv_cache.value, v), axis=1)
                else:
                    k, v = new_cache
        shared_kv[layer_type] = (k, v)

    if decode_ranges is not None and layer_type == FULL and k.shape[2] == 1 and q.shape[1] == 1:
        from gemma4_posttrain_jax.pallas.decode_attention import decode_attention, supported

        batch, _, heads, _, _ = q.shape
        if supported(batch, heads, k.shape[1]):
            starts, length = decode_ranges
            out = decode_attention(q[:, 0, :, 0, :], k[:, :, 0, :], v[:, :, 0, :], starts, length)[:, None, :, None, :]
            partial = jnp.einsum("bsrhd,rhdm->bsm", out, params.o_proj, preferred_element_type=jnp.float32)
            return (_reduced(partial, seq.dtype, allreduce) if reduce else partial), new_cache

    def core(keys: Array, values: Array, mask: Array) -> Array:
        scores = jnp.einsum("bsrhd,bthd->brhst", q, keys)
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
        probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q.dtype)
        return jnp.einsum("brhst,bthd->bsrhd", probs, values)

    if decode_window is not None and layer_type == FULL and q.shape[1] == 1:
        index, windows = decode_window

        def window(bounds: tuple[int, int]) -> Callable[[None], Array]:
            start, end = bounds
            return lambda _: core(k[:, start:end], v[:, start:end], qk_mask[..., start:end])

        out = lax.switch(index, [window(bounds) for bounds in windows], None)
    else:
        out = core(k, v, qk_mask)
    partial = jnp.einsum("bsrhd,rhdm->bsm", out, params.o_proj, preferred_element_type=jnp.float32)
    if row_split > 1:
        # Rounding commutes with gathering the complete rows, so gather them in bf16.
        gathered = lax.all_gather(partial.astype(dtype), TP_AXIS, axis=0, tiled=True)
        return (gathered if reduce else partial), new_cache
    return (_reduced(partial, dtype, allreduce) if reduce else partial), new_cache


def _mlp(params: MLPParams, x: Array, allreduce: str = "xla", reduce: bool = True, high: bool = False) -> Array:
    """The TP MLP block (this chip's slice of the intermediate size).

    ``high`` keeps gate, up and the activation in float32 (no bf16 rounding of the matmul outputs) and multiplies
    the activation by the bf16 down projection as two bf16 parts, ``hi = bf16(a)`` and ``lo = bf16(a - hi)``,
    stacked along the rows so the weights are read once: the arithmetic of ``mlp_stream(precision="high")``,
    close to a float32 activation times the bf16 weights.
    """

    if high:
        f32 = jnp.float32
        gate = jnp.einsum("bsm,mf->bsf", x, params.gate_proj, preferred_element_type=f32)
        up = jnp.einsum("bsm,mf->bsf", x, params.up_proj, preferred_element_type=f32)
        act = jax.nn.gelu(gate, approximate=True) * up
        hi = act.astype(x.dtype)
        lo = (act - hi.astype(f32)).astype(x.dtype)
        both = jnp.einsum(
            "bsf,fm->bsm", jnp.concatenate([hi, lo], axis=0), params.down_proj, preferred_element_type=f32
        )
        partial = both[: x.shape[0]] + both[x.shape[0] :]
    else:
        h = jax.nn.gelu(x @ params.gate_proj, approximate=True) * (x @ params.up_proj)
        partial = jnp.einsum("bsf,fm->bsm", h, params.down_proj, preferred_element_type=jnp.float32)
    return _reduced(partial, x.dtype, allreduce) if reduce else partial


def tp_forward_text(
    params: Gemma4TextParams,
    config: Gemma4TextConfig,
    input_ids: Array,
    position_ids: Array,
    attention_mask: Array,
    kv_cache: KVCache,
    *,
    cache_mode: str,
    allreduce: str = "xla",
    decode_kernel: bool = False,
    decode_windows: tuple[tuple[int, int], ...] = (),
    row_split: int = 1,
) -> tuple[Array, KVCache]:
    """Per-device decoder pass mirroring ``model.forward_gemma4_text`` with a cache.

    ``row_split`` > 1 splits attention by rows (``attention_rows``; the cache holds this chip's rows).

    ``decode_kernel`` uses the decode attention kernel on global layers in append mode.
    ``decode_windows`` (static ``(start, end)`` cache slices, the last one the whole cache) makes
    global layers in append mode attend only to the first window that holds every row's real slots:
    it must start at or before the first real slot of every row and end at or after the slots written.
    """

    eps = config.rms_norm_eps
    batch, queries = input_ids.shape
    attention_mask = attention_mask.astype(jnp.bool_)
    x = _vocab_lookup(params.embed_tokens, input_ids, allreduce)
    x = x * jnp.asarray(math.sqrt(config.hidden_size), x.dtype)
    per_layer_inputs = None
    if config.hidden_size_per_layer_input:
        width = config.hidden_size_per_layer_input
        token_part = _vocab_lookup(cast(Array, params.embed_tokens_per_layer), input_ids, allreduce)
        token_part = token_part * jnp.asarray(math.sqrt(width), token_part.dtype)
        context = jnp.einsum("bsm,mlp->bslp", x, params.per_layer_model_projection)
        context = _scale(context, config.hidden_size**-0.5)
        context = forward_rms_norm(params.per_layer_projection_norm, context, eps)
        per_layer_inputs = _scale(context + token_part, 2.0**-0.5)

    key_mask = update_cache_mask(kv_cache.key_mask, attention_mask, kv_cache.length)
    if cache_mode == "extend":
        qk_masks = make_chunked_qk_masks(
            kv_cache.key_mask, attention_mask, config.sliding_window, query_start=kv_cache.length
        )
    elif cache_mode == "append":
        qk_masks = make_cached_qk_masks(key_mask, config.sliding_window, query_start=kv_cache.length, num_queries=1)
    else:
        qk_masks = make_qk_masks(attention_mask, config.sliding_window, query_start=0, num_queries=queries)
    rotary = make_rotary_values(config, position_ids)
    decode_window = None
    if decode_windows and cache_mode == "append":
        written = jnp.max(kv_cache.length) + queries
        first_real = jnp.min(jnp.argmax(key_mask, axis=1)).astype(jnp.int32)
        fits = [(first_real >= start) & (written <= end) for start, end in decode_windows[:-1]]
        # Index of the first window that fits, the whole cache (last) when none does.
        index = jnp.asarray(len(fits), jnp.int32)
        for position in range(len(fits) - 1, -1, -1):
            index = jnp.where(fits[position], position, index)
        decode_window = (index, tuple(decode_windows))
    decode_ranges = None
    if decode_kernel and cache_mode == "append":
        if kv_cache.length.ndim:
            raise ValueError("当前 decode attention kernel 只支持同步标量位置")
        # First real slot of each row (prompts are left-padded) and the slots written after this append.
        decode_ranges = (jnp.argmax(key_mask, axis=1).astype(jnp.int32), kv_cache.length + queries)
    # Fusing needs the one-shot kernel, i.e. a small hidden state (decode).
    rows = batch * queries
    fuse = allreduce.startswith("one_shot_fused") and x.size * 4 <= ONE_SHOT_MAX_BYTES
    fuse = fuse and (row_split == 1 or (rows // row_split) % 8 == 0)
    exchange = jnp.bfloat16 if allreduce == "one_shot_fused_bf16" else jnp.float32
    two_rounds = "_sg" in allreduce and rows % 16 == 0 and config.hidden_size % 512 == 0
    algorithm = "scatter_gather" if two_rounds else "one_shot"
    # With attention split by rows the first merge of a layer stacks the chips' rows.
    first_merge = "gather_rows" if row_split > 1 else algorithm
    # Each chip keeps only its rows of the residual stream (merges: norm_gather, then scatter_rows).
    row_local = fuse and row_split > 1 and allreduce.endswith("_rows_local")
    second_merge = "scatter_rows" if row_local else algorithm
    shared_kv: dict[str, tuple[Array, Array]] = {}
    new_layers: list[LayerKVCache | None] = []
    # With fusing, each layer's last kernel also returns the next layer's normalized input.
    normalized = None
    flat_inputs = None
    attention_cache_length = kv_cache.length
    if fuse and per_layer_inputs is not None:
        flat_inputs = per_layer_inputs.reshape(batch * queries, -1)
    if row_local:
        local = rows // row_split
        start = lax.axis_index(TP_AXIS) * local

        def mine(value: Array) -> Array:
            return lax.dynamic_slice_in_dim(value, start, local, axis=0)

        x = mine(x)
        qk_masks = {name: mine(mask) for name, mask in qk_masks.items()}
        rotary = {name: jax.tree.map(mine, values) for name, values in rotary.items()}
        flat_inputs = None if flat_inputs is None else mine(flat_inputs)
        # Slice write positions with the TP-sharded KV rows; keep global masks and returned lengths as full vectors.
        if attention_cache_length.ndim:
            attention_cache_length = mine(attention_cache_length)
    for i, layer in enumerate(params.layers):
        layer_type = config.layer_types[i]
        residual = x
        h = forward_rms_norm(layer.input_norm, x, eps) if normalized is None else normalized
        h, layer_cache = _attention(
            layer.attention,
            h,
            qk_masks[layer_type],
            rotary[layer_type],
            layer_index=i,
            config=config,
            shared_kv=shared_kv,
            kv_cache=kv_cache.layers[i],
            cache_length=attention_cache_length,
            cache_mode=cache_mode,
            allreduce=allreduce,
            reduce=not fuse,
            decode_ranges=decode_ranges,
            decode_window=decode_window,
            row_split=1 if row_local else row_split,
        )
        if fuse:
            if allreduce.startswith("one_shot_fused_mlp") and rows >= MLP_KERNEL_MIN_ROWS:
                precision = "high" if "_high" in allreduce else "model"
                x, h = _merge_mlp(
                    h,
                    residual,
                    layer.post_attention_norm,
                    layer.pre_feedforward_norm,
                    layer.mlp,
                    eps,
                    precision,
                    first_merge,
                )
            elif row_local:
                x, h = _norm_gather(h, residual, layer.post_attention_norm, layer.pre_feedforward_norm, eps)
                h = _mlp(layer.mlp, h, allreduce, reduce=False, high="_high" in allreduce)
            else:
                x, mlp_input = _reduced_norm_residual(
                    h,
                    residual,
                    layer.post_attention_norm,
                    layer.pre_feedforward_norm,
                    eps,
                    exchange_dtype=exchange,
                    algorithm=first_merge,
                )
                h = _mlp(layer.mlp, cast(Array, mlp_input), allreduce, reduce=False, high="_high" in allreduce)
            following = params.layers[i + 1].input_norm if i + 1 < len(params.layers) else params.final_norm
            x, normalized = _reduced_norm_residual(
                h,
                x,
                layer.post_feedforward_norm,
                following,
                eps,
                layer_end=(layer, flat_inputs, i),
                exchange_dtype=exchange,
                algorithm=second_merge,
            )
            new_layers.append(layer_cache)
            continue
        else:
            h = forward_rms_norm(layer.post_attention_norm, h, eps)
            x = residual + h
            residual = x
            h = forward_rms_norm(layer.pre_feedforward_norm, x, eps)
            h = _mlp(layer.mlp, h, allreduce)
            h = forward_rms_norm(layer.post_feedforward_norm, h, eps)
            x = residual + h
        if layer.per_layer_input is not None:
            assert per_layer_inputs is not None
            x = x + forward_per_layer_input(layer.per_layer_input, x, per_layer_inputs[:, :, i], eps)
        x = x * layer.layer_scalar.astype(x.dtype)
        new_layers.append(layer_cache)
    hidden = forward_rms_norm(params.final_norm, x, eps) if normalized is None else normalized
    if row_local:
        hidden = lax.all_gather(hidden, TP_AXIS, axis=0, tiled=True)
    return hidden, KVCache(tuple(new_layers), key_mask, kv_cache.length + queries)


def tp_local_logits(params: Gemma4TextParams, config: Gemma4TextConfig, hidden: Array) -> Array:
    """Logits for this chip's vocabulary slice, softcapped like ``model.forward_lm_head``."""

    logits = jnp.einsum("bsm,vm->bsv", hidden, params.embed_tokens)
    if config.final_logit_softcapping is not None:
        cap = config.final_logit_softcapping
        logits = jnp.tanh(logits / cap) * cap
    return logits


def tp_score_targets(local_logits: Array, targets: Array) -> Array:
    """Full-vocabulary log-probability of ``targets`` from vocabulary-split logits.

    Same formula as ``sampler._sample_token``: subtract the global max, log of the
    HIGHEST-accuracy exp-sum. Only the order of the exp-sum differs (local, then psum).
    """

    logits32 = local_logits.astype(jnp.float32)
    rows = logits32.shape[-1]
    shifted = logits32 - lax.pmax(jnp.max(logits32, axis=-1, keepdims=True), TP_AXIS)
    total = lax.psum(jnp.sum(lax.exp(shifted, accuracy=lax.AccuracyMode.HIGHEST), axis=-1, keepdims=True), TP_AXIS)
    normalizer = lax.log(total, accuracy=lax.AccuracyMode.HIGHEST)
    local = targets - lax.axis_index(TP_AXIS) * rows
    valid = (local >= 0) & (local < rows)
    picked = jnp.take_along_axis(shifted, jnp.clip(local, 0, rows - 1)[..., None], axis=-1)[..., 0]
    picked = lax.psum(jnp.where(valid, picked, 0.0), TP_AXIS)
    return cast(Array, picked - normalizer[..., 0])


# --- shard_map entry points -------------------------------------------------------------------------


def decode_window_list(width: int, capacity: int, step: int, lower: int) -> tuple[tuple[int, int], ...]:
    """Static global-layer windows ``[width - lower, width + k * step)`` and the whole cache last.

    ``lower`` is how many prompt columns before the decode start a window keeps (prompts longer than
    that use the whole cache). ``step=0`` disables the windows.
    """

    if step <= 0:
        return ()
    start = max(width - lower, 0)
    ends = list(range(width + step, capacity, step))
    return tuple((start, end) for end in ends) + ((start, capacity), (0, capacity))


def _position_ids(attention_mask: Array) -> Array:
    return jnp.maximum(jnp.cumsum(attention_mask.astype(jnp.int32), axis=-1) - 1, 0)


def _prefill_logits(
    current: Gemma4TextParams,
    config: Gemma4TextConfig,
    ids: Array,
    mask: Array,
    *,
    capacity: int,
    widths: tuple[int, ...],
    vocab_kernel: bool | str = False,
    row_split: int = 1,
    cache_prompt_width: int | None = None,
    prefill_chunk_size: int = 0,
) -> tuple[Array, KVCache]:
    """Prefill (inside ``shard_map``) and the float32 local logits of the last prompt position.

    ``row_split`` > 1: attention split by rows (``attention_rows``).

    With ``vocab_kernel`` it returns the last position's final hidden state ``[B, M]`` instead (the
    fused vocabulary kernel computes the logits while it samples).

    Prompts are left-padded. When the longest real prompt of these rows has at most ``w`` tokens
    for one of ``widths`` (narrower than the prompt width), only the last ``w`` columns are run:
    the cache starts at slot ``width - w``, so the real tokens land in the same slots and the cache
    length after prefill is the same as for a full-width prefill. The skipped slots stay empty and
    masked. The narrowest fitting width is chosen at run time (``lax.switch``); the decode loop that
    follows sees the same shapes either way.
    """

    width = ids.shape[1]
    physical_width = width if cache_prompt_width is None else cache_prompt_width
    if physical_width < width:
        raise ValueError("cache_prompt_width 不能小于实际 prefill 宽度")
    dtype = current.embed_tokens.dtype
    candidates = sorted({w for w in widths if 0 < w < width}) + [width]

    def run(w: int) -> Callable[[None], tuple[Array, KVCache]]:
        def branch(_: None) -> tuple[Array, KVCache]:
            cache = _init_cache(config, ids.shape[0], capacity, dtype, row_split)
            cache = cache._replace(length=jnp.asarray(physical_width - w, jnp.int32))
            part_ids, part_mask = ids[:, width - w :], mask[:, width - w :]
            if prefill_chunk_size and w > prefill_chunk_size:

                def run_chunk(ids, positions, mask, kv):
                    return tp_forward_text(
                        current, config, ids, positions, mask, kv, cache_mode="extend", row_split=row_split
                    )

                hidden, cache = chunked_prefill(
                    run_chunk, part_ids, _position_ids(part_mask), part_mask, cache, chunk_size=prefill_chunk_size
                )
            else:
                hidden, cache = tp_forward_text(
                    current,
                    config,
                    part_ids,
                    _position_ids(part_mask),
                    part_mask,
                    cache,
                    cache_mode="prefill",
                    row_split=row_split,
                )
            if vocab_kernel:
                return hidden[:, -1], cache
            return tp_local_logits(current, config, hidden[:, -1:])[:, -1].astype(jnp.float32), cache

        return branch

    if len(candidates) == 1:
        return run(width)(None)
    longest = jnp.max(jnp.sum(mask, axis=-1, dtype=jnp.int32))
    index = sum((longest > w).astype(jnp.int32) for w in candidates[:-1])
    return cast(tuple[Array, KVCache], lax.switch(index, [run(w) for w in candidates], None))


def tp_prefill(
    params: Gemma4TextParams,
    config: Gemma4TextConfig,
    prompt_ids: Array,
    prompt_mask: Array,
    *,
    mesh: Mesh,
    max_new_tokens: int,
    param_specs: Any,
    cache_prompt_width: int | None = None,
    prefill_chunk_size: int = 0,
    prefill_widths: tuple[int, ...] = (),
    attention_rows: bool = False,
) -> tuple[Array, KVCache]:
    """Prefill on the TP mesh; returns last-position local logits and the cache (rows over ``dp``).

    ``prefill_widths`` runs only the last prompt columns when every prompt fits (``_prefill_logits``);
    the logits are then float32. ``attention_rows`` splits attention (and the cache rows) by rows.
    """

    batch, width = prompt_ids.shape
    physical_width = width if cache_prompt_width is None else cache_prompt_width
    dp = mesh.shape[DP_AXIS]
    row_split = mesh.shape[TP_AXIS] if attention_rows else 1
    template = jax.eval_shape(
        lambda: _init_cache(config, batch // dp, physical_width + max_new_tokens, jnp.bfloat16, row_split)
    )
    cache_specs = tp_cache_specs(template, attention_rows=attention_rows)

    def body(current: Any, ids: Array, mask: Array) -> tuple[Array, KVCache]:
        if prefill_widths or cache_prompt_width is not None or prefill_chunk_size:
            logits, cache = _prefill_logits(
                current,
                config,
                ids,
                mask,
                capacity=physical_width + max_new_tokens,
                widths=tuple(prefill_widths),
                cache_prompt_width=physical_width,
                prefill_chunk_size=prefill_chunk_size,
                row_split=row_split,
            )
            return logits[:, None, :], cache
        dtype = current.embed_tokens.dtype
        cache = _init_cache(config, ids.shape[0], width + max_new_tokens, dtype, row_split)
        hidden, cache = tp_forward_text(
            current, config, ids, _position_ids(mask), mask, cache, cache_mode="prefill", row_split=row_split
        )
        return tp_local_logits(current, config, hidden[:, -1:]), cache

    return jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(param_specs, P(DP_AXIS), P(DP_AXIS)),
        out_specs=(P(DP_AXIS, None, TP_AXIS), cache_specs),
        check_vma=False,
    )(params, prompt_ids, prompt_mask)


def tp_fixed_history(
    params: Gemma4TextParams,
    config: Gemma4TextConfig,
    cache: KVCache,
    completion_ids: Array,
    completion_mask: Array,
    prompt_lengths: Array,
    *,
    mesh: Mesh,
    param_specs: Any,
    reference_length: int,
    allreduce: str = "xla",
    decode_kernel: bool = False,
    decode_window_step: int = 0,
    decode_window_lower: int = 256,
) -> tuple[Array, KVCache]:
    """TP version of ``reverse.rollout.fixed_history_decode`` (same append/score order)."""

    _check_allreduce(allreduce, mesh)
    rows_mode = attention_rows(allreduce)
    row_split = mesh.shape[TP_AXIS] if rows_mode else 1
    cache_specs = tp_cache_specs(cache, attention_rows=rows_mode)
    capacity = cache.key_mask.shape[1]
    windows = decode_window_list(capacity - completion_ids.shape[1], capacity, decode_window_step, decode_window_lower)

    def body(current: Any, kv: KVCache, ids: Array, mask: Array, lengths: Array) -> tuple[Array, KVCache]:
        current = _row_norm_weights(current, allreduce)

        def step(state: KVCache, index: Array) -> tuple[KVCache, Array]:
            tokens = lax.dynamic_slice_in_dim(ids, index, 1, axis=1)
            step_mask = lax.dynamic_slice_in_dim(mask, index, 1, axis=1)
            positions = (lengths + index)[:, None]
            hidden, state = tp_forward_text(
                current,
                config,
                tokens,
                positions,
                step_mask,
                state,
                cache_mode="append",
                allreduce=allreduce,
                decode_kernel=decode_kernel,
                decode_windows=windows,
                row_split=row_split,
            )
            logits = tp_local_logits(current, config, hidden)[:, -1]
            next_tokens = lax.dynamic_slice_in_dim(ids, index + 1, 1, axis=1)[:, 0]
            next_mask = lax.dynamic_slice_in_dim(mask, index + 1, 1, axis=1)[:, 0]
            scores = tp_score_targets(logits, next_tokens)
            return state, jnp.where(next_mask, scores, jnp.float32(0.0))

        final, scores = lax.scan(step, kv, jnp.arange(reference_length - 1, dtype=jnp.int32))
        return scores.T, final

    return jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(param_specs, cache_specs, P(DP_AXIS), P(DP_AXIS), P(DP_AXIS)),
        out_specs=(P(DP_AXIS), cache_specs),
        check_vma=False,
    )(params, cache, completion_ids, completion_mask, prompt_lengths)


def tp_shardings(mesh: Mesh, specs: Any) -> Any:
    return _named(mesh, specs)


# --- sampling ---------------------------------------------------------------------------------------


def gumbel_slice(
    key: Array, row_offset: Array, column_offset: Array, local_shape: tuple[int, int], width: int
) -> Array:
    """Gumbel noise equal to ``jax.random.gumbel(key, (rows, width))[row_offset:, column_offset:]`` for one slice.

    Uses the partitionable threefry counters (the flattened element index) and the same
    bits-to-float conversion as ``jax.random.uniform``; requires rows * width < 2**32.
    """

    rows = row_offset.astype(jnp.uint32) + lax.iota(jnp.uint32, local_shape[0])
    columns = column_offset.astype(jnp.uint32) + lax.iota(jnp.uint32, local_shape[1])
    low = rows[:, None] * jnp.uint32(width) + columns[None, :]
    high = jnp.zeros_like(low)
    data = jax.random.key_data(key) if jnp.issubdtype(key.dtype, jax.dtypes.prng_key) else key
    bits1, bits2 = threefry2x32_p.bind(data[0], data[1], high, low)
    bits = bits1 ^ bits2
    one = np.array(1.0, np.float32).view(np.uint32)
    float_bits = lax.bitwise_or(lax.shift_right_logical(bits, jnp.uint32(32 - 23)), jnp.uint32(one))
    floats = lax.bitcast_convert_type(float_bits, jnp.float32) - jnp.array(1.0, jnp.float32)
    minval = jnp.array(jnp.finfo(jnp.float32).tiny, jnp.float32)
    maxval = jnp.array(1.0, jnp.float32)
    uniform = lax.max(minval, floats * (maxval - minval) + minval)
    return -jnp.log(-jnp.log(uniform))


def _first_max_index(values: Array, best: Array) -> Array:
    """``jnp.argmax(values, axis=-1)`` given ``best = jnp.max(values, axis=-1)``: the first index holding it.

    A NaN makes ``best`` NaN and ``jnp.argmax`` return the first NaN, so NaNs count as hits too. With the
    Gumbel noise computed in the same fusion, XLA's ``argmax`` (a reduction over value and index pairs)
    takes 150 us for 32 rows x 65536 columns on a TPU v4 chip; a max and then a min over matching indices
    take 82 us (``reverse/rollout/benchmark_sampling.py``).
    """

    width = values.shape[-1]
    hit = (values == best[..., None]) | jnp.isnan(values)
    return jnp.min(jnp.where(hit, lax.iota(jnp.int32, width), jnp.int32(width)), axis=-1)


def _global_argmax(values: Array) -> Array:
    """First index of the maximum over vocabulary-split ``values`` (same tie rule as ``jnp.argmax``)."""

    width = values.shape[-1]
    best = jnp.max(values, axis=-1)
    local_index = _first_max_index(values, best) + lax.axis_index(TP_AXIS) * width
    best_values = lax.all_gather(best, TP_AXIS)
    best_indices = lax.all_gather(local_index, TP_AXIS)
    shard = jnp.argmax(best_values, axis=0)
    return jnp.take_along_axis(best_indices, shard[None], axis=0)[0]


def tp_sample_token(
    local_logits: Array, key: Array, *, temperature: float, row_offset: Array, vocab_size: int
) -> tuple[Array, Array]:
    """Distributed version of ``sampler._sample_token`` for top_k=0 and top_p=1."""

    logits32 = local_logits.astype(jnp.float32)
    width = logits32.shape[-1]
    shifted = logits32 - lax.pmax(jnp.max(logits32, axis=-1, keepdims=True), TP_AXIS)
    total = lax.psum(jnp.sum(lax.exp(shifted, accuracy=lax.AccuracyMode.HIGHEST), axis=-1, keepdims=True), TP_AXIS)
    full_logps = shifted - lax.log(total, accuracy=lax.AccuracyMode.HIGHEST)
    if temperature <= 0:
        token = _global_argmax(logits32)
    else:
        noise = gumbel_slice(
            key, row_offset, lax.axis_index(TP_AXIS) * width, cast(tuple[int, int], logits32.shape), vocab_size
        )
        token = _global_argmax(noise + logits32 / temperature)
    local = token - lax.axis_index(TP_AXIS) * width
    valid = (local >= 0) & (local < width)
    picked = jnp.take_along_axis(full_logps, jnp.clip(local, 0, width - 1)[:, None], axis=-1)[:, 0]
    return token, lax.psum(jnp.where(valid, picked, 0.0), TP_AXIS)


def tp_sample_from_hidden(
    hidden: Array,
    embed: Array,
    key: Array,
    *,
    temperature: float,
    row_offset: Array,
    vocab_size: int,
    softcap: float | None,
    rounding: str = "dot",
) -> tuple[Array, Array]:
    """``tp_sample_token`` on ``tp_local_logits`` of ``hidden [B, M]``, with the fused vocabulary kernel.

    ``rounding`` is the kernel's (``vocab_sample.ROUNDING``): ``"dot"`` rounds the logits to bf16 like the XLA
    einsum, ``"none"`` keeps them in float32.

    Same noise, tie rule and distributed argmax as ``tp_sample_token``; the logits stay in VMEM and the
    log-normalizer is accumulated over vocabulary tiles, so log-probabilities differ by rounding
    (``kernels.pallas.vocab_sample``).
    """
    from gemma4_posttrain_jax.pallas.vocab_sample import vocab_sample_partials

    width = embed.shape[0]
    column_offset = lax.axis_index(TP_AXIS) * width
    data = jax.random.key_data(key) if jnp.issubdtype(key.dtype, jax.dtypes.prng_key) else key
    partials = vocab_sample_partials(
        hidden,
        embed,
        data,
        row_offset,
        column_offset,
        vocab_size=vocab_size,
        temperature=max(temperature, 0.0),
        softcap=softcap,
        rounding=rounding,
    )
    best_values = lax.all_gather(partials.best, TP_AXIS)
    best_indices = lax.all_gather(partials.position + column_offset, TP_AXIS)
    shard = jnp.argmax(best_values, axis=0)
    token = jnp.take_along_axis(best_indices, shard[None], axis=0)[0]
    global_max = lax.pmax(partials.row_max, TP_AXIS)
    total = lax.psum(partials.row_sumexp * jnp.exp(partials.row_max - global_max), TP_AXIS)
    logit = lax.psum(jnp.where(lax.axis_index(TP_AXIS) == shard, partials.logit, 0.0), TP_AXIS)
    return token, logit - global_max - jnp.log(total)


def _decode_loop(
    current: Gemma4TextParams,
    config: Gemma4TextConfig,
    logits: Array,
    cache: KVCache,
    mask: Array,
    rng: Array,
    *,
    sampler_config: SamplerConfig,
    allreduce: str,
    decode_every_step: bool,
    decode_chunk: int,
    token_dtype: Any,
    decode_kernel: bool = False,
    decode_window_step: int = 0,
    decode_window_lower: int = 256,
    vocab_kernel: bool | str = False,
    row_split: int = 1,
) -> tuple[Array, Array, Array, Array]:
    """The sampling loop of ``tp_generate`` after the prefill (inside ``shard_map``).

    With ``vocab_kernel`` the loop carries the final hidden state instead of the logits and samples
    with ``tp_sample_from_hidden``.

    ``logits`` are the float32 local logits of the last prompt position, ``cache`` the prefilled cache
    and ``mask`` the prompt mask of these rows. Returns completion ids, mask, log-probabilities, lengths.
    """

    rows = mask.shape[0]
    max_new = sampler_config.max_new_tokens
    row_offset = lax.axis_index(DP_AXIS) * rows
    capacity = cache.key_mask.shape[1]
    windows = decode_window_list(capacity - max_new, capacity, decode_window_step, decode_window_lower)
    prompt_lengths = jnp.sum(mask, axis=-1, dtype=jnp.int32)

    def any_global(flags: Array) -> Array:
        return cast(Array, lax.psum(jnp.any(flags).astype(jnp.int32), DP_AXIS) > 0)

    state = (
        jnp.zeros((), jnp.int32),
        rng,
        jnp.ones((rows,), jnp.bool_),
        jnp.ones((), jnp.bool_),
        logits,
        cache,
        jnp.full((rows, max_new), config.pad_token_id, token_dtype),
        jnp.zeros((rows, max_new), jnp.bool_),
        jnp.zeros((rows, max_new), jnp.float32),
    )

    def cond_fn(current_state: Any) -> Array:
        step, _, _, any_active, *_ = current_state
        return cast(Array, (step < max_new) & any_active)

    def body_fn(current_state: Any) -> Any:
        step, key_state, active, _, step_logits, kv, completion, completion_mask, logps = current_state
        next_key, sample_key = jax.random.split(key_state)
        if vocab_kernel:
            sampled, logp = tp_sample_from_hidden(
                step_logits,
                current.embed_tokens,
                sample_key,
                temperature=sampler_config.temperature,
                row_offset=row_offset,
                vocab_size=config.vocab_size,
                softcap=config.final_logit_softcapping,
                rounding="dot" if vocab_kernel is True else vocab_kernel,
            )
        else:
            sampled, logp = tp_sample_token(
                step_logits,
                sample_key,
                temperature=sampler_config.temperature,
                row_offset=row_offset,
                vocab_size=config.vocab_size,
            )
        token = jnp.where(active, sampled, config.pad_token_id).astype(token_dtype)
        logp = jnp.where(active, logp, 0.0)
        completion = completion.at[:, step].set(token)
        completion_mask = completion_mask.at[:, step].set(active)
        logps = logps.at[:, step].set(logp)
        next_active = active & ~_is_eos(token, sampler_config.eos_ids)
        any_active = any_global(next_active)
        should_decode = (step + 1 < max_new) & any_active

        def run_decode(_: None) -> tuple[Array, KVCache]:
            decoded, next_kv = tp_forward_text(
                current,
                config,
                token[:, None],
                (prompt_lengths + step)[:, None],
                active[:, None],
                kv,
                cache_mode="append",
                allreduce=allreduce,
                decode_kernel=decode_kernel,
                decode_windows=windows,
                row_split=row_split,
            )
            if vocab_kernel:
                return decoded[:, -1], next_kv
            return tp_local_logits(current, config, decoded)[:, -1].astype(jnp.float32), next_kv

        if decode_every_step:
            step_logits, kv = run_decode(None)
        else:
            step_logits, kv = lax.cond(should_decode, run_decode, lambda _: (step_logits, kv), None)
        return (step + 1, next_key, next_active, any_active, step_logits, kv, completion, completion_mask, logps)

    def chunk_fn(current_state: Any) -> Any:
        return lax.fori_loop(0, decode_chunk, lambda _, inner: body_fn(inner), current_state)

    final = lax.while_loop(cond_fn, body_fn if decode_chunk == 1 else chunk_fn, state)
    completion, completion_mask, logps = final[6], final[7], final[8]
    return completion, completion_mask, logps, jnp.sum(completion_mask, axis=-1, dtype=jnp.int32)


def tp_generate(
    params: Gemma4TextParams,
    config: Gemma4TextConfig,
    prompt_ids: Array,
    prompt_mask: Array,
    *,
    sampler_config: SamplerConfig,
    key: Array,
    mesh: Mesh,
    param_specs: Any,
    allreduce: str = "xla",
    decode_every_step: bool = True,
    decode_chunk: int = 1,
    prefill_widths: tuple[int, ...] = (),
    decode_kernel: bool = False,
    decode_window_step: int = 0,
    decode_window_lower: int = 256,
    vocab_kernel: bool | str = False,
) -> RolloutBatch:
    """TP version of ``sampler.generate`` with the same loop, stopping rule and random stream.

    ``allreduce="one_shot"`` sums small decode reductions with the one-shot kernel (tp=4);
    prefill keeps XLA's all-reduce because its arrays are large.

    ``decode_every_step`` runs the decode unconditionally in every loop iteration instead of
    under ``lax.cond``. The last iteration then computes one decode whose logits are never used
    (its cache write still fits the capacity), so the returned rollout is the same. Without the
    conditional, XLA can prefetch weights into on-chip memory across iterations as it does for
    a plain scan; under the conditional the decode matmuls read every weight from HBM.

    ``decode_chunk > 1`` checks the stopping rule only every ``decode_chunk`` steps and runs the
    steps in between as a loop with a fixed trip count. Steps after every row has finished only
    write padding, so the rollout is the same; at most ``decode_chunk - 1`` extra decodes run.

    ``prefill_widths`` (e.g. ``(128, 256)``) runs the prefill only over the last ``w`` prompt columns
    when every real prompt fits in them (``_prefill_logits``); the rollout then differs from the
    full-width prefill only by rounding.

    ``decode_kernel`` computes the decode attention of global layers with the Pallas kernel that reads
    only the written cache slots (``_attention``).

    ``decode_window_step`` (e.g. 256) lets global layers attend only to the static cache window
    ``[width - decode_window_lower, width + k * step)`` that holds the slots written so far
    (``decode_window_list``); prompts longer than ``decode_window_lower`` use the whole cache.

    ``vocab_kernel`` (True, or the kernel's logit rounding, e.g. ``"none"``) samples with the fused vocabulary
    projection kernel (``tp_sample_from_hidden``):
    the same tokens whenever the logits agree, log-probabilities up to rounding.
    """
    _check_allreduce(allreduce, mesh)
    if any(w <= 0 for w in prefill_widths):
        raise ValueError("prefill widths must be positive")
    if decode_chunk < 1 or sampler_config.max_new_tokens % decode_chunk:
        raise ValueError("decode_chunk must divide max_new_tokens")
    if decode_chunk > 1 and not decode_every_step:
        raise ValueError("decode_chunk > 1 decodes in every step")
    row_split = mesh.shape[TP_AXIS] if attention_rows(allreduce) else 1
    if row_split > 1 and (decode_kernel or decode_window_step):
        raise ValueError("attention split by rows covers the XLA attention only")

    batch, width = prompt_ids.shape
    if width != sampler_config.max_prompt_len:
        raise ValueError("prompt width must equal max_prompt_len")
    if sampler_config.top_k or sampler_config.top_p < 1:
        raise ValueError("TP sampling supports only the full vocabulary (top_k=0, top_p=1)")
    if batch * config.vocab_size >= 2**32:
        raise ValueError("batch * vocabulary must stay below 2**32 for the 32-bit counters")
    max_new = sampler_config.max_new_tokens

    def body(current: Any, ids: Array, mask: Array, rng: Array) -> tuple[Array, Array, Array, Array]:
        current = _row_norm_weights(current, allreduce)
        mask = mask.astype(jnp.bool_)
        logits, cache = _prefill_logits(
            current,
            config,
            ids,
            mask,
            capacity=width + max_new,
            widths=tuple(prefill_widths),
            vocab_kernel=vocab_kernel,
            row_split=row_split,
        )
        return _decode_loop(
            current,
            config,
            logits,
            cache,
            mask,
            rng,
            sampler_config=sampler_config,
            allreduce=allreduce,
            decode_every_step=decode_every_step,
            decode_chunk=decode_chunk,
            token_dtype=ids.dtype,
            decode_kernel=decode_kernel,
            decode_window_step=decode_window_step,
            decode_window_lower=decode_window_lower,
            vocab_kernel=vocab_kernel,
            row_split=row_split,
        )

    rows = P(DP_AXIS)
    completion, completion_mask, logps, lengths = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(param_specs, rows, rows, P()),
        out_specs=(rows, rows, rows, rows),
        check_vma=False,
    )(params, prompt_ids, prompt_mask, key)
    return RolloutBatch(prompt_ids, completion, completion_mask, logps, lengths)


def tp_generate_prefill(
    params: Gemma4TextParams,
    config: Gemma4TextConfig,
    prompt_ids: Array,
    prompt_mask: Array,
    *,
    mesh: Mesh,
    param_specs: Any,
    max_new_tokens: int,
    prefill_widths: tuple[int, ...] = (),
    vocab_kernel: bool | str = False,
    attention_rows: bool = False,
) -> tuple[Array, KVCache]:
    """The prefill of ``tp_generate`` as a program of its own.

    ``attention_rows`` splits attention and the cache rows by rows (pass the same mode to ``tp_generate_decode``).

    With ``vocab_kernel`` the first output is the last position's hidden state ``[B, M]`` (the same on
    every ``tp`` chip) instead of the split logits.

    Returns the float32 local logits of the last prompt position (``[B, V]`` split over ``tp``) and
    the cache, to be passed to ``tp_generate_decode``. Compiled apart from the decode loop, a change
    in the prefill (for example ``prefill_widths``) cannot change how XLA schedules the loop.
    """

    batch, width = prompt_ids.shape
    dp = mesh.shape[DP_AXIS]
    row_split = mesh.shape[TP_AXIS] if attention_rows else 1
    template = jax.eval_shape(lambda: _init_cache(config, batch // dp, width + max_new_tokens, jnp.bfloat16, row_split))
    cache_specs = tp_cache_specs(template, attention_rows=attention_rows)

    def body(current: Any, ids: Array, mask: Array) -> tuple[Array, KVCache]:
        return _prefill_logits(
            current,
            config,
            ids,
            mask.astype(jnp.bool_),
            capacity=width + max_new_tokens,
            widths=tuple(prefill_widths),
            vocab_kernel=vocab_kernel,
            row_split=row_split,
        )

    rows = P(DP_AXIS)
    return jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(param_specs, rows, rows),
        out_specs=(P(DP_AXIS) if vocab_kernel else P(DP_AXIS, TP_AXIS), cache_specs),
        check_vma=False,
    )(params, prompt_ids, prompt_mask)


def tp_generate_decode(
    params: Gemma4TextParams,
    config: Gemma4TextConfig,
    logits: Array,
    cache: KVCache,
    prompt_ids: Array,
    prompt_mask: Array,
    *,
    sampler_config: SamplerConfig,
    key: Array,
    mesh: Mesh,
    param_specs: Any,
    allreduce: str = "xla",
    decode_every_step: bool = True,
    decode_chunk: int = 1,
    decode_kernel: bool = False,
    decode_window_step: int = 0,
    decode_window_lower: int = 256,
    vocab_kernel: bool | str = False,
) -> RolloutBatch:
    """The sampling loop of ``tp_generate`` as a program of its own, from ``tp_generate_prefill``'s outputs.

    Together the two programs compute the same rollout as ``tp_generate`` up to rounding (XLA fuses
    the two parts differently when they are one program).
    """

    _check_allreduce(allreduce, mesh)
    rows_mode = attention_rows(allreduce)
    row_split = mesh.shape[TP_AXIS] if rows_mode else 1
    if row_split > 1 and (decode_kernel or decode_window_step):
        raise ValueError("attention split by rows covers the XLA attention only")
    cache_specs = tp_cache_specs(cache, attention_rows=rows_mode)

    def body(current: Any, step_logits: Array, kv: KVCache, ids: Array, mask: Array, rng: Array) -> Any:
        current = _row_norm_weights(current, allreduce)
        return _decode_loop(
            current,
            config,
            step_logits,
            kv,
            mask.astype(jnp.bool_),
            rng,
            sampler_config=sampler_config,
            allreduce=allreduce,
            decode_every_step=decode_every_step,
            decode_chunk=decode_chunk,
            token_dtype=ids.dtype,
            decode_kernel=decode_kernel,
            decode_window_step=decode_window_step,
            decode_window_lower=decode_window_lower,
            vocab_kernel=vocab_kernel,
            row_split=row_split,
        )

    rows = P(DP_AXIS)
    head = P(DP_AXIS) if vocab_kernel else P(DP_AXIS, TP_AXIS)
    completion, completion_mask, logps, lengths = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(param_specs, head, cache_specs, rows, rows, P()),
        out_specs=(rows, rows, rows, rows),
        check_vma=False,
    )(params, logits, cache, prompt_ids, prompt_mask, key)
    return RolloutBatch(prompt_ids, completion, completion_mask, logps, lengths)
