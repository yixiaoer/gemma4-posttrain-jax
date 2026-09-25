"""四芯片 TPU v4 训练 logprob：原生前向/dH 与 Pallas 归一化项 dW。"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from gemma4_posttrain_jax.sharding import DATA_AXIS


def _dot(a: Array, b: Array) -> Array:
    # 当前 Mosaic 将 BF16 的 HIGHEST 映射为不接受 BF16 操作数的指令。
    # BF16 操作数使用原生 BF16 乘法、FP32 累加；FP32 操作数保留 HIGHEST。
    precision = lax.Precision.DEFAULT if a.dtype == jnp.bfloat16 else lax.Precision.HIGHEST
    return jnp.dot(a, b, precision=precision, preferred_element_type=jnp.float32)


def _shape(w: Array, h: Array, bm: int, bv: int) -> tuple[int, ...]:
    m, v, d = math.prod(h.shape[:-1]), w.shape[0], w.shape[1]
    return m, v, d, math.ceil(m / (2 * bm)) * 2 * bm, math.ceil(v / (2 * bv)) * 2 * bv, math.ceil(d / 128) * 128


def _prepare(w: Array, h: Array, t: Array, bm: int, bv: int) -> tuple[Array, ...]:
    m, v, d, mp, vp, dp = _shape(w, h, bm, bv)
    return (
        jnp.pad(h.reshape(m, d), ((0, mp - m), (0, dp - d))),
        jnp.pad(w, ((0, vp - v), (0, dp - d))),
        jnp.pad(t.reshape(m, 1), ((0, mp - m), (0, 0)), constant_values=-1),
    )


def per_token_logps(
    w: Array,
    h: Array,
    t: Array,
    *,
    native_logps: Callable[..., Array],
    native_logps_and_normalization: Callable[..., tuple[Array, Array]],
    softcap: float | None,
    mesh: Mesh | None,
    vocab_chunk: int = 8192,
    sequence_chunk: int = 256,
) -> Array:
    """保留原生浮点边界，按运行时上游梯度跳过全零块的一阶反向。"""
    if mesh is None or mesh.size != 4 or DATA_AXIS not in mesh.axis_names:
        raise ValueError("Pallas logprob requires a four-chip vocabulary-parallel mesh")
    if any("TPU v4" not in device.device_kind for device in mesh.devices.flat):
        raise ValueError("Pallas logprob currently supports TPU v4")
    if w.dtype != jnp.bfloat16 or h.dtype != jnp.bfloat16:
        raise ValueError("Pallas logprob requires BF16 weights and hidden states")
    if w.ndim != 2 or h.shape[:-1] != t.shape or h.shape[-1] != w.shape[1] or not t.size:
        raise ValueError("Pallas logprob received incompatible embedding/hidden/target shapes")
    if w.shape[0] % mesh.shape[DATA_AXIS]:
        raise ValueError("Vocabulary size must be divisible by its mesh axis")
    if (vocab_chunk, sequence_chunk) != (8192, 256):
        raise ValueError("Pallas logprob requires vocab_chunk=8192 and sequence_chunk=256")
    if softcap is not None and (not math.isfinite(softcap) or softcap <= 0):
        raise ValueError("softcap must be finite and positive, or None")

    def native(a: Array, b: Array) -> Array:
        return native_logps(a, b, t, softcap=softcap, mesh=mesh, vocab_chunk=vocab_chunk, sequence_chunk=sequence_chunk)

    @jax.custom_vjp
    def apply(a: Array, b: Array) -> Array:
        return native(a, b)

    def fwd(a: Array, b: Array) -> Any:
        values, pullback, normalization = _shared_hidden_and_normalization(
            a,
            b,
            t,
            softcap=softcap,
            mesh=mesh,
            vocab_chunk=vocab_chunk,
            native_logps_and_normalization=native_logps_and_normalization,
        )
        return values, (a, b, normalization, pullback)

    def bwd(saved: Any, g: Array) -> Any:
        a, b, normalization, pullback = saved
        dw = _weight_gradient_from_normalization(a, b, t, normalization, g, softcap=softcap, mesh=mesh)
        return dw, pullback(g)

    apply.defvjp(fwd, bwd)
    return cast(Array, apply(w, h))


def _weight_gradient_from_normalization(
    w: Any,
    h: Any,
    t: Any,
    normalization: Any,
    g: Any,
    *,
    softcap: float | None,
    mesh: Any,
) -> Any:
    """复用原生前向的归一化结果；只由 Pallas 计算权重梯度。"""

    def local(a: Any, b: Any, target: Any, values: Any, dy: Any) -> Any:
        offset = lax.axis_index(DATA_AXIS) * a.shape[0] if mesh is not None else 0
        # 保留原 sequence scan 的 target gather/scatter 和 BF16 累加边界。
        rows, width = math.prod(target.shape), b.shape[-1]
        padding = math.ceil(rows / 256) * 256 - rows
        hidden_blocks = jnp.pad(b.reshape(rows, width), ((0, padding), (0, 0))).reshape(-1, 256, width)
        target_blocks = jnp.pad(target.reshape(rows), ((0, padding),)).reshape(-1, 256)

        def target_score(weights: Any) -> Any:
            def body(_: None, inputs: Any) -> Any:
                hidden_block, target_block = inputs
                owned_block = (target_block >= offset) & (target_block < offset + weights.shape[0])
                local_ids = jnp.clip(target_block - offset, 0, weights.shape[0] - 1)
                selected = jnp.take(weights, local_ids, axis=0)
                score = jnp.einsum(
                    "nm,nm->n",
                    hidden_block,
                    selected,
                    precision=lax.Precision.HIGHEST,
                    preferred_element_type=jnp.float32,
                )
                if softcap is not None:
                    score = jnp.tanh(score / softcap) * softcap
                score = jnp.where(owned_block, score, 0.0)
                if mesh is not None:
                    score = lax.psum(score, DATA_AXIS)
                return None, score

            _, scores = lax.scan(body, None, (hidden_blocks, target_blocks))
            return scores.reshape(-1)[:rows].reshape(target.shape)

        _, target_pullback = jax.vjp(target_score, a)
        positive = target_pullback(dy)[0]
        return _normalization_weight_gradient(a, b, values, dy, softcap, positive)

    if mesh is None:
        return local(w, h, t, normalization, g)
    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P(DATA_AXIS, None), P(), P(), P(), P()),
        out_specs=P(DATA_AXIS, None),
        axis_names={DATA_AXIS},
    )(w, h, t, normalization, g)


def _shared_hidden_and_normalization(
    w: Any,
    h: Any,
    targets: Any,
    *,
    softcap: float | None,
    mesh: Any,
    vocab_chunk: int,
    native_logps_and_normalization: Callable[..., tuple[Array, Array]],
) -> tuple[Any, Any, Any]:
    """原生统计只作为辅助输出，反向仍按 256 行条件执行 hidden VJP。"""
    replicated = NamedSharding(mesh, P())
    h, targets = (lax.with_sharding_constraint(x, replicated) for x in (h, targets))
    rows, width = math.prod(targets.shape), h.shape[-1]
    padding = math.ceil(rows / 256) * 256 - rows
    tokens = jnp.pad(targets.reshape(rows), ((0, padding),)).reshape(-1, 256)

    def native(hidden: Any, token: Any) -> Any:
        return native_logps_and_normalization(
            w, hidden, token, softcap=softcap, mesh=mesh, vocab_chunk=vocab_chunk, sequence_chunk=256
        )

    @jax.custom_vjp
    def block(hidden: Any, token: Any) -> Any:
        return native(hidden, token)

    def block_fwd(hidden: Any, token: Any) -> Any:
        value, pullback, statistics = jax.vjp(lambda x: native(x, token), hidden, has_aux=True)
        return (value, statistics), pullback

    def block_bwd(pullback: Any, cotangent: Any) -> Any:
        dy, _ = cotangent
        dh = lax.cond(
            jnp.any(dy != 0),
            lambda grad: pullback(grad)[0],
            lambda _: jnp.zeros((256, width), h.dtype),
            dy,
        )
        return dh, None

    block.defvjp(block_fwd, block_bwd)

    def forward(hidden: Any) -> Any:
        blocks = jnp.pad(hidden.reshape(rows, width), ((0, padding), (0, 0))).reshape(-1, 256, width)

        def body(_: None, inputs: Any) -> Any:
            return None, block(*inputs)

        _, (values, statistics) = lax.scan(body, None, (blocks, tokens))
        normalized = jnp.moveaxis(statistics, 1, 0).reshape(2, -1)[:, :rows]
        return values.reshape(-1)[:rows].reshape(targets.shape), normalized.reshape((2, *targets.shape))

    values, pullback, statistics = jax.vjp(forward, h, has_aux=True)

    def backward(saved: Any, g: Any) -> Any:
        g = lax.with_sharding_constraint(g, replicated)
        return saved(g)[0]

    return values, jax.tree_util.Partial(backward, pullback), statistics


def _normalization_weight_gradient(
    w: Array, h: Array, normalization: Array, g: Array, cap: float | None, positive: Array
) -> Array:
    """按原生顺序舍入三分量 dW，最后写回时加上原生目标梯度。"""
    bm, bv = 256, 512
    m, v, d, _, vp, dp = _shape(w, h, bm, bv)
    # 此 kernel 按词表块分给两个 TensorCore，token 只需补齐到一个计算块。
    mp = math.ceil(m / bm) * bm

    def token_block(_i: Any, j: Any) -> Any:
        return mp // bm - 1 - j

    specs = (
        pl.BlockSpec((bm, dp), lambda i, j: (token_block(i, j), 0)),
        pl.BlockSpec((bv, dp), lambda i, j: (i, 0)),
        pl.BlockSpec((2, bm, 1), lambda i, j: (0, token_block(i, j), 0)),
        pl.BlockSpec((bm, 1), lambda i, j: (token_block(i, j), 0)),
        pl.BlockSpec((bv, dp), lambda i, j: (i, 0), pipeline_mode=pl.Buffered(1)),
    )

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((vp, dp), w.dtype, manual_axis_type=jax.typeof(w).manual_axis_type),
        mesh=pltpu.TensorCoreMesh(axis_name="core", num_cores=2),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
        name="gemma4_lm_head_training_normalization_dw",
    )
    def kernel(*refs: Any) -> None:
        def scan(acc: Any, rounded: Any) -> None:
            def body(hr: Any, wr: Any, nr: Any, gr: Any, positive_ref: Any, out: Any) -> None:
                i, j = pl.program_id(0), pl.program_id(1)

                @pl.when(j == 0)
                def initialize() -> None:
                    rounded[...] = jnp.zeros((bv, dp), w.dtype)

                # 全零上游块不读写累加器；首块的 rounded 初始化仍无条件执行。
                @pl.when(jnp.any(gr[...] != 0))
                def accumulate() -> None:
                    acc[...] = jnp.zeros((bv, dp), jnp.float32)
                    logits = _dot(hr[...], wr[...].T)
                    if cap is not None:
                        tanh = jnp.tanh(logits / cap)
                        logits = tanh * cap
                    valid = i * bv + jnp.arange(bv)[None, :] < v
                    dz = jnp.where(valid, (-gr[...] / nr[1, ...]) * jnp.exp(logits - nr[0, ...]), 0.0)
                    if cap is not None:
                        lower = (dz * cap) * (1.0 - tanh)
                        dz = (lower + lower * tanh) / cap
                    high = dz.astype(jnp.bfloat16)
                    remainder = dz - high.astype(jnp.float32)
                    low = remainder.astype(jnp.bfloat16)
                    third = (remainder - low.astype(jnp.float32)).astype(jnp.bfloat16)
                    acc[...] += _dot(third.T, hr[...])
                    acc[...] += _dot(low.T, hr[...])
                    acc[...] += _dot(high.T, hr[...])

                    rounded[...] = (rounded[...] + acc[...].astype(w.dtype)).astype(w.dtype)

                @pl.when(j == mp // bm - 1)
                def store() -> None:
                    out[...] = rounded[...] + positive_ref[...]

            pltpu.emit_pipeline(
                body,
                grid=(vp // bv, mp // bm),
                in_specs=specs,
                out_specs=pl.BlockSpec((bv, dp), lambda i, j: (i, 0)),
                core_axis_name="core",
                dimension_semantics=(pltpu.PARALLEL, pltpu.ARBITRARY),
            )(*refs)

        pl.run_scoped(scan, pltpu.VMEM((bv, dp), jnp.float32), pltpu.VMEM((bv, dp), w.dtype))

    # 复用通用输入布局，再去掉本 kernel 不需要的 token padding；目标项仍由原生 VJP 计算。
    hidden, weights, _ = _prepare(w, h, jnp.zeros(h.shape[:-1], jnp.int32), bm, bv)
    hidden = hidden[:mp]
    statistics = jnp.stack(
        tuple(jnp.pad(normalization[i].reshape(m, 1), ((0, mp - m), (0, 0)), constant_values=i) for i in range(2))
    )
    dy = jnp.pad(g.reshape(m, 1), ((0, mp - m), (0, 0)))
    positive = jnp.pad(positive, ((0, vp - v), (0, dp - d)))
    return cast(Array, kernel(hidden, weights, statistics, dy, positive)[:v, :d])
