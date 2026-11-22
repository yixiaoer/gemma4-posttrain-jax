"""Single-chip TPU v4 attention with Splash, two TensorCore groups, and a separate backward pass.

Validation covers execution within each chip, not communication between chips. This is not the model's default
backend.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu import splash_attention as splash


def make_block_sizes(
    *,
    block_q: int,
    block_kv: int,
    block_kv_compute: int | None = None,
    block_kv_dkv_compute: int | None = None,
    block_q_dkv: int | None = None,
    block_kv_dkv: int | None = None,
    block_q_dq: int | None = None,
    block_kv_dq: int | None = None,
    fused_backward: bool = False,
    q_layout: str = "head",
    k_layout: str = "head",
    v_layout: str = "head",
) -> splash.BlockSizes:
    """Specify each attention stage explicitly for execution and inspection, without automatic fallback."""
    if fused_backward and (block_q_dq is not None or block_kv_dq is not None):
        raise ValueError("融合反向不能指定独立dQ分块")
    q_dkv = block_q if block_q_dkv is None else block_q_dkv
    kv_dkv = block_kv if block_kv_dkv is None else block_kv_dkv
    q_dq = None if fused_backward else (block_q if block_q_dq is None else block_q_dq)
    kv_dq = None if fused_backward else (block_kv if block_kv_dq is None else block_kv_dq)
    if any(x < 128 or x % 128 for x in (block_q, block_kv, q_dkv, kv_dkv, q_dq, kv_dq) if x is not None):
        raise ValueError("Splash 各阶段分块必须是128的正倍数")
    dkv_compute = block_kv_compute if block_kv_dkv_compute is None else block_kv_dkv_compute
    for name, compute, memory in (("前向", block_kv_compute, block_kv), ("dKV", dkv_compute, kv_dkv)):
        if compute is not None and (compute < 128 or compute % 128 or memory % compute):
            raise ValueError(f"{name} KV计算块必须是128的正倍数且整除对应搬运块")
    layouts = {"head": splash.QKVLayout.HEAD_DIM_MINOR, "sequence": splash.QKVLayout.SEQ_MINOR}
    if any(x not in layouts for x in (q_layout, k_layout, v_layout)):
        raise ValueError("Q/K/V layout 必须为 head 或 sequence")
    return splash.BlockSizes(
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=block_kv_compute,
        block_q_dkv=q_dkv,
        block_kv_dkv=kv_dkv,
        block_kv_dkv_compute=dkv_compute,
        block_q_dq=q_dq,
        block_kv_dq=kv_dq,
        use_fused_bwd_kernel=fused_backward,
        q_layout=layouts[q_layout],
        k_layout=layouts[k_layout],
        v_layout=layouts[v_layout],
    )


def attention(
    q: Any,
    k: Any,
    v: Any,
    key_mask: Any,
    *,
    block_q: int = 128,
    block_kv: int = 128,
    block_kv_compute: int | None = None,
    block_kv_dkv_compute: int | None = None,
    block_q_dkv: int | None = None,
    block_kv_dkv: int | None = None,
    block_q_dq: int | None = None,
    block_kv_dq: int | None = None,
    query_start: int = 0,
    window: int = 0,
    fused_backward: bool = False,
    fp32_interface: bool = False,
    bf16_storage: bool = False,
    bf16_dout: bool = False,
    dkv_kv_buffers: int | None = None,
    dkv_accumulate_outputs: bool = False,
    dkv_vmem_scratch: bool = False,
    dkv_output_buffers: int | None = None,
    dq_vmem_scratch: bool = False,
    dq_output_buffers: int | None = None,
    dq_kv_compute: int | None = None,
    dq_compute_unroll: bool = False,
    head_splits: int = 1,
    q_layout: str = "head",
    k_layout: str = "head",
    v_layout: str = "head",
) -> Any:
    """Accept BF16 Q[B,S,R,H,D], K/V[B,T,H,D], and boolean key_mask[B,T].

    Preserve scaling=1, k<=q, and k>q-window. Support the training and contiguous-prefill core, not cached
    decoding, dropout, softcapping, or packed segments. Splash uses BF16 MXU precision independently of the
    surrounding model and high-precision reference.
    """
    if q.ndim != 5 or k.ndim != 4 or k.shape != v.shape:
        raise ValueError("需要 Q[B,S,R,H,D] 和同形状 K/V[B,T,H,D]")
    b, s, r, h, d = q.shape
    t = k.shape[1]
    if k.shape != (b, t, h, d) or min(b, s, r, h, d, t) <= 0:
        raise ValueError("Q/K/V 的 batch、KV heads、head dim 必须对应且非空")
    if any(x.dtype != jnp.bfloat16 for x in (q, k, v)):
        raise ValueError("当前 Splash 包装只支持 BF16 Q/K/V")
    if key_mask.shape != (b, t) or key_mask.dtype != jnp.bool_:
        raise ValueError("需要 bool key_mask[B,T]，不接受 dense mask")
    if query_start < 0 or query_start + s > t or window < 0:
        raise ValueError("需要 0<=query_start、query_start+S<=T、window>=0")
    if head_splits < 1 or r % head_splits:
        raise ValueError("head_splits 必须是每个 KV head 对应的 Q head 数 R 的正因数")

    if bf16_storage and not fp32_interface:
        raise ValueError("BF16存储实验要求FP32接口")
    if bf16_dout and not bf16_storage:
        raise ValueError("BF16 dO存储实验要求先启用BF16 Q/K/V存储")

    if dkv_kv_buffers is not None and (not bf16_storage or dkv_kv_buffers not in (1, 2)):
        raise ValueError("dKV K/V缓冲实验要求BF16 Q/K/V存储，缓冲数量为1或2")

    if dkv_accumulate_outputs and (
        not bf16_storage or fused_backward or dkv_vmem_scratch or dkv_output_buffers not in (1, 2)
    ):
        raise ValueError("dKV输出累加要求BF16存储、独立反向、1或2个输出缓冲，且不另设VMEM scratch")

    if (dkv_vmem_scratch or dkv_output_buffers is not None) and (not bf16_storage or fused_backward):
        raise ValueError("dKV VMEM/缓冲实验要求BF16 Q/K/V存储和独立反向")
    if dkv_output_buffers is not None and dkv_output_buffers not in (1, 2):
        raise ValueError("dKV输出缓冲数量必须是1或2")

    if (dq_vmem_scratch or dq_output_buffers is not None) and (not bf16_storage or fused_backward):
        raise ValueError("dQ VMEM/缓冲实验要求BF16 Q/K/V存储和独立反向")
    if dq_output_buffers is not None and dq_output_buffers not in (1, 2):
        raise ValueError("dQ输出缓冲数量必须是1或2")

    if dq_compute_unroll and dq_kv_compute is None:
        raise ValueError("展开dQ循环要求指定计算块")
    if dq_kv_compute is not None:
        memory = block_kv if block_kv_dq is None else block_kv_dq
        if not bf16_storage or fused_backward:
            raise ValueError("dQ计算子块实验要求BF16 Q/K/V存储和独立反向")
        if dq_kv_compute < 128 or dq_kv_compute % 128 or memory % dq_kv_compute:
            raise ValueError("dQ计算块必须是128的正倍数且整除搬运块")

    blocks = make_block_sizes(
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=block_kv_compute,
        block_kv_dkv_compute=block_kv_dkv_compute,
        block_q_dkv=block_q_dkv,
        block_kv_dkv=block_kv_dkv,
        block_q_dq=block_q_dq,
        block_kv_dq=block_kv_dq,
        fused_backward=fused_backward,
        q_layout=q_layout,
        k_layout=k_layout,
        v_layout=v_layout,
    )
    # Share inputs across forward and backward block sizes, padding to their least common multiple.
    q_multiple = math.lcm(*(x for x in (blocks.block_q, blocks.block_q_dkv, blocks.block_q_dq) if x is not None))
    kv_multiple = math.lcm(*(x for x in (blocks.block_kv, blocks.block_kv_dkv, blocks.block_kv_dq) if x is not None))
    sp = (s + q_multiple - 1) // q_multiple * q_multiple
    tp = (max(t, query_start + sp) + kv_multiple - 1) // kv_multiple * kv_multiple
    dp = (d + 127) // 128 * 128
    qp = jnp.pad(q, ((0, 0), (0, sp - s), (0, 0), (0, 0), (0, dp - d)))
    kp = jnp.pad(k, ((0, 0), (0, tp - t), (0, 0), (0, dp - d)))
    vp = jnp.pad(v, ((0, 0), (0, tp - t), (0, 0), (0, dp - d)))
    # Splash groups R consecutive query heads per KV head; the model uses logical axes R,H.
    qp = qp.transpose(0, 3, 2, 1, 4).reshape(b, h * r, sp, dp)
    kp, vp = (x.transpose(0, 2, 1, 3) for x in (kp, vp))
    # This setting changes library input, saved output, and returned cotangent dtypes together.
    # Its effect cannot be attributed to one residual; matrix precision remains BF16.
    if fp32_interface:
        qp, kp, vp = (x.astype(jnp.float32) for x in (qp, kp, vp))

    # Detect empty rows with prefix counts, storing O(T+S) data instead of an S*T mask.
    prefix = jnp.pad(jnp.cumsum(key_mask.astype(jnp.int32), axis=1), ((0, 0), (1, 0)))
    end = query_start + jnp.arange(s) + 1
    begin = jnp.maximum(0, end - window) if window else jnp.zeros_like(end)
    nonempty = prefix[:, end] > prefix[:, begin]
    q_ids = jnp.pad(nonempty.astype(jnp.int32), ((0, 0), (0, sp - s)))
    kv_ids = jnp.pad(key_mask.astype(jnp.int32), ((0, 0), (0, tp - t)))
    # Temporarily match an empty row to the invalid key that must exist at its absolute position.
    # Then select the model's uniform result so these Splash rows receive zero cotangent.
    static_mask = splash.LocalMask((sp, tp), (window - 1 if window else None, 0), offset=query_start)
    factory = splash.make_splash_mha_single_device
    if bf16_storage:
        from gemma4_posttrain_jax.pallas.splash_storage_one_chip_tpuv4 import module

        factory = module(
            bf16_dout=bf16_dout,
            dkv_kv_buffers=dkv_kv_buffers,
            dkv_accumulate_outputs=dkv_accumulate_outputs,
            dkv_vmem_scratch=dkv_vmem_scratch,
            dkv_output_buffers=dkv_output_buffers,
            dq_vmem_scratch=dq_vmem_scratch,
            dq_output_buffers=dq_output_buffers,
            dq_kv_compute=dq_kv_compute,
            dq_compute_unroll=dq_compute_unroll,
        ).make_splash_mha_single_device
    kernel = factory(splash.MultiHeadMask(tuple(static_mask for _ in range(h * r // head_splits))), block_sizes=blocks)

    def invoke(q, k, v, qi, ki):
        # The outer FP32 interface preserves grouped-gradient reduction precision; inner BF16 is for storage only.
        if bf16_storage:
            q, k, v = (x.astype(jnp.bfloat16) for x in (q, k, v))
        return kernel(q, k, v, segment_ids=splash.SegmentIds(qi, ki))

    @jax.custom_vjp
    def one(q, k, v, qi, ki):
        with jax.default_matmul_precision("bfloat16"):
            return invoke(q, k, v, qi, ki)

    def one_fwd(q, k, v, qi, ki):
        with jax.default_matmul_precision("bfloat16"):
            out, pullback = jax.vjp(lambda q, k, v: invoke(q, k, v, qi, ki), q, k, v)
        return out, pullback

    def one_bwd(pullback, do):
        # A context around forward execution alone does not constrain a custom VJP traced later.
        # Call the library's backward rule without rewriting or additionally accumulating gradients in this wrapper.
        with jax.default_matmul_precision("bfloat16"):
            return (*pullback(do), None, None)

    one.defvjp(one_fwd, one_bwd)
    if head_splits == 1:
        out = jax.vmap(one)(qp, kp, vp, q_ids, kv_ids)
    else:
        # Keep every KV head in each group, splitting only its corresponding query heads.
        # The outer vmap creates independent groups that can run in parallel.
        # JAX sums shared-KV cotangents once; slicing a flat H*R axis misaligns GQA when H>1.
        rp = r // head_splits
        grouped_q = qp.reshape(b, h, head_splits, rp, sp, dp).transpose(0, 2, 1, 3, 4, 5)
        grouped_q = grouped_q.reshape(b, head_splits, h * rp, sp, dp)
        grouped = jax.vmap(one, in_axes=(0, None, None, None, None))
        out = jax.vmap(grouped)(grouped_q, kp, vp, q_ids, kv_ids)
        out = out.reshape(b, head_splits, h, rp, sp, dp).transpose(0, 2, 1, 3, 4, 5)
        out = out.reshape(b, h * r, sp, dp)
    out = out.reshape(b, h, r, sp, dp).transpose(0, 3, 2, 1, 4)[:, :s, :, :, :d]
    out = out.astype(q.dtype)
    # Dense attention rounds 1/T to BF16 for fully masked rows; preserve that rounding and the V gradient.
    uniform = jnp.einsum("t,bthd->bhd", jnp.full((t,), 1.0 / t, q.dtype), v)
    return jnp.where(nonempty[:, :, None, None, None], out, uniform[:, None, None, :, :])
