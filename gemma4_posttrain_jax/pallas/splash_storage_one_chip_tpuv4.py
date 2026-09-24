"""TPU v4 单 chip Splash 存储实验：BF16 搬运 Q/K/V，片上恢复 FP32 运算接口。

只在独立模块中改写已核验的安装源码；不修改 JAX 模块或 wheel。
保存的输出与返回的 cotangent 保持 FP32。实际精度和收益须单独验证。
"""

from __future__ import annotations

import ast
import functools
import linecache
import sys
import types
from pathlib import Path

import jax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as installed

VARIANT = "bf16-storage-fp32-compute-v1"


def _split_dq_compute(source: str, compute: int, *, unroll: bool) -> str:
    """保持搬运窗口，在dQ kernel内部串行计算更小的KV子块。"""
    if compute < 128 or compute % 128:
        raise ValueError("dQ计算块必须是128的正倍数")
    start = source.index("def _flash_attention_dq_kernel(")
    end = source.index("def _splash_attention_bwd_dq(", start)
    kernel = source[start:end]
    changes = [
        (
            "  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR",
            "  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR\n"
            f"  bkv_compute = {compute}\n"
            "  if bkv % bkv_compute:\n"
            "    raise ValueError('dQ计算块必须整除搬运块')",
        ),
        (
            "  @pl.when(should_run)\n  def run():",
            "  def body(sub_index, _):\n    k_slice = pl.ds(sub_index * bkv_compute, bkv_compute)",
        ),
        (
            "    k = k_ref[...].astype(jnp.float32)\n    v = v_ref[...].astype(jnp.float32)",
            "    def load_compute(ref, layout):\n"
            "      part = ref[k_slice, :] if layout == HEAD_DIM_MINOR else ref[:, k_slice]\n"
            "      return part.astype(jnp.float32)\n"
            "    k = load_compute(k_ref, k_layout)\n"
            "    v = load_compute(v_ref, v_layout)",
        ),
        ("    k_slice = pl.ds(0, bkv)\n", ""),
        (
            "        k_offset=global_kv_index * bkv,",
            "        k_offset=global_kv_index * bkv + sub_index * bkv_compute,",
        ),
        (
            "  @pl.when(j == grid_width - 1)",
            "  @pl.when(should_run)\n"
            "  def run():\n"
            f"    lax.fori_loop(0, bkv // bkv_compute, body, None, unroll={unroll})\n\n"
            "  @pl.when(j == grid_width - 1)",
        ),
    ]
    for before, after in changes:
        if kernel.count(before) != 1:
            raise RuntimeError(f"Splash dQ计算子块源码定位失败：{before!r}")
        kernel = kernel.replace(before, after)
    source = source[:start] + kernel + source[end:]
    marker = "      block_kv_dq=bkv,"
    if source.count(marker) != 1:
        raise RuntimeError("Splash dQ计算元数据定位失败")
    source = source.replace(marker, marker + f"\n      block_kv_dq_compute={compute},")
    suffix = f"_dqc{compute}" + ("_unroll" if unroll else "")
    return source.replace("{residuals}_bf16_storage", "{residuals}_bf16_storage" + suffix)


def _dkv_scratch(source: str, *, vmem: bool, output_buffers: int | None) -> str:
    """dK/dV累加量跨Q块和Q heads保留，只改变分配及输出缓冲。"""
    if output_buffers is not None and output_buffers not in (1, 2):
        raise ValueError("dKV输出缓冲数量必须是1或2")
    start = source.index("def _flash_attention_dkv_kernel(")
    end = source.index("def _splash_attention_bwd(", start)
    part = source[start:end]
    changes = []
    if vmem:
        changes.extend(
            [
                (
                    "    # Outputs\n    dq_scratch_ref,\n    dk_scratch_ref,\n    dv_scratch_ref,\n"
                    "    dq_ref,\n    dk_ref,\n    dv_ref,",
                    "    # Outputs\n    dq_scratch_ref,\n    dq_ref,\n    dk_ref,\n    dv_ref,\n"
                    "    # VMEM临时量：跨Q块和Q heads保留。\n    dk_scratch_ref,\n    dv_scratch_ref,",
                ),
                ("      jax.ShapeDtypeStruct((bkv, head_dim_qk), jnp.float32),\n", ""),
                ("      jax.ShapeDtypeStruct((bkv, head_dim_v), jnp.float32),\n", ""),
                ("      pl.BlockSpec((bkv, head_dim_qk), lambda *_: (0, 0)),\n", ""),
                ("      pl.BlockSpec((bkv, head_dim_v), lambda *_: (0, 0)),\n", ""),
                (
                    "    _, _, _, dq_unreduced, dk, dv = pl.pallas_call(",
                    "    _, dq_unreduced, dk, dv = pl.pallas_call(",
                ),
                (
                    "            grid=grid,",
                    "            grid=grid,\n            scratch_shapes=(\n"
                    "                pltpu.VMEM((bkv, head_dim_qk), jnp.float32),\n"
                    "                pltpu.VMEM((bkv, head_dim_v), jnp.float32),\n            ),",
                ),
            ]
        )
    if output_buffers is not None:
        changes.append(
            (
                "  out_shapes = [",
                f"  dk_spec = dataclasses.replace(dk_spec, pipeline_mode=pl.Buffered({output_buffers}))\n"
                f"  dv_spec = dataclasses.replace(dv_spec, pipeline_mode=pl.Buffered({output_buffers}))\n\n"
                "  out_shapes = [",
            )
        )
    for before, after in changes:
        if part.count(before) != 1:
            raise RuntimeError(f"Splash dKV scratch源码定位失败：{before!r}")
        part = part.replace(before, after)
    source = source[:start] + part + source[end:]
    suffix = "_dkvvmem" if vmem else ""
    suffix += f"_dkvout{output_buffers}" if output_buffers is not None else ""
    return source.replace("{residuals}_bf16_storage", "{residuals}_bf16_storage" + suffix)


def _dkv_output_accumulator(source: str) -> str:
    """FP32输出窗口直接承载dK/dV累加，省去独立scratch及结束复制。"""
    start = source.index("def _flash_attention_dkv_kernel(")
    middle = source.index("def _splash_attention_bwd_dkv(", start)
    end = source.index("def _splash_attention_bwd(", middle)
    kernel, wrapper = source[start:middle], source[middle:end]
    arguments = "    dk_scratch_ref,\n    dv_scratch_ref,\n"
    if kernel.count(arguments) != 1:
        raise RuntimeError("Splash dKV输出累加参数定位失败")
    kernel = kernel.replace(arguments, "")
    finish = kernel.index("  should_write = q_index == grid_width - 1")
    # 独立反向没有dQ临时量；输出不再在结束时从scratch复制或被清零。
    kernel = kernel[:finish] + "\n\n"
    kernel = kernel.replace("dk_scratch_ref", "dk_ref").replace("dv_scratch_ref", "dv_ref")
    kernel = kernel.replace("gradient scratch buffers", "gradient output accumulators")
    changes = [
        ("      jax.ShapeDtypeStruct((bkv, head_dim_qk), jnp.float32),\n", ""),
        ("      jax.ShapeDtypeStruct((bkv, head_dim_v), jnp.float32),\n", ""),
        ("      pl.BlockSpec((bkv, head_dim_qk), lambda *_: (0, 0)),\n", ""),
        ("      pl.BlockSpec((bkv, head_dim_v), lambda *_: (0, 0)),\n", ""),
        ("    _, _, _, dq_unreduced, dk, dv = pl.pallas_call(", "    _, dq_unreduced, dk, dv = pl.pallas_call("),
        (
            "  num_q_heads, q_seq_len, head_dim_qk = q.shape",
            "  if use_fused_bwd_kernel:\n    raise ValueError('dKV输出累加只支持独立反向')\n"
            "  num_q_heads, q_seq_len, head_dim_qk = q.shape",
        ),
    ]
    for before, after in changes:
        if wrapper.count(before) != 1:
            raise RuntimeError(f"Splash dKV输出累加源码定位失败：{before!r}")
        wrapper = wrapper.replace(before, after)
    source = source[:start] + kernel + wrapper + source[end:]
    return source.replace("{residuals}_bf16_storage", "{residuals}_bf16_storage_dkvoutacc")


def transformed_source(
    *,
    bf16_dout: bool = False,
    dkv_kv_buffers: int | None = None,
    dkv_accumulate_outputs: bool = False,
    dkv_vmem_scratch: bool = False,
    dkv_output_buffers: int | None = None,
    dq_vmem_scratch: bool = False,
    dq_output_buffers: int | None = None,
    dq_kv_compute: int | None = None,
    dq_compute_unroll: bool = False,
) -> str:
    """JAX 版本不兼容或改写位置不唯一时，拒绝生成候选模块。"""
    assert installed.__file__ is not None
    data = Path(installed.__file__).read_bytes()
    if jax.__version__ != "0.11.1":
        raise RuntimeError(f"TPU v4 单 chip 存储实验要求 JAX 0.11.1，当前为 {jax.__version__}")
    source = data.decode()
    replacements = (
        (
            'return f"splash_{attention_type}_{phase}{segments}{residuals}"',
            'return f"splash_{attention_type}_{phase}{segments}{residuals}_bf16_storage"',
            1,
        ),
        (
            "jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim_v), q.dtype)",
            "jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim_v), jnp.float32)",
            1,
        ),
        ("jax.ShapeDtypeStruct.like(q)", "jax.ShapeDtypeStruct(q.shape, jnp.float32)", 1),
        ("jax.ShapeDtypeStruct.like(k)", "jax.ShapeDtypeStruct(k.shape, jnp.float32)", 1),
        ("jax.ShapeDtypeStruct.like(v)", "jax.ShapeDtypeStruct(v.shape, jnp.float32)", 1),
        (
            "jax.ShapeDtypeStruct((kv_seq_len // bkv, *q.shape), q.dtype)",
            "jax.ShapeDtypeStruct((kv_seq_len // bkv, *q.shape), jnp.float32)",
            1,
        ),
        (
            "q = q_ref[...] if q_layout == HEAD_DIM_MINOR else q_ref[...].T",
            "q = (q_ref[...] if q_layout == HEAD_DIM_MINOR else q_ref[...].T).astype(jnp.float32)",
            2,
        ),
        (
            "qk = lax.dot_general(q, k, qk_dims, preferred_element_type=float32)",
            "qk = lax.dot_general(q, k.astype(jnp.float32), qk_dims, preferred_element_type=float32)",
            1,
        ),
        (
            "    k = k_ref[...]\n    v = v_ref[...]",
            "    k = k_ref[...].astype(jnp.float32)\n    v = v_ref[...].astype(jnp.float32)",
            1,
        ),
        (
            "    q = q_ref[...]  # We keep q potentially transposed, since it's always RHS",
            "    q = q_ref[...].astype(jnp.float32)  # Preserve arithmetic, compress storage",
            1,
        ),
        (
            "    k = _load_kv(k_ref, k_layout)\n    v = _load_kv(v_ref, v_layout)",
            "    k = _load_kv(k_ref, k_layout).astype(jnp.float32)\n"
            "    v = _load_kv(v_ref, v_layout).astype(jnp.float32)",
            1,
        ),
    )
    for before, after, count in replacements:
        if source.count(before) != count:
            raise RuntimeError(f"Splash storage源码定位不唯一：{before!r}")
        source = source.replace(before, after)
    if bf16_dout:
        # DI仍使用原始FP32 dO；只压缩随后送入反向矩阵计算的dO。
        # 提前压缩DI输入的v1在完整模型中有两份梯度超差，原始失败保留。
        marker = '  di = jnp.einsum("hsd,hsd->hs", o.astype(jnp.float32), do.astype(jnp.float32))'
        if source.count(marker) != 1 or source.count("    do = do_ref[...]") != 2:
            raise RuntimeError("Splash dO存储源码定位失败")
        source = source.replace(marker, marker + "\n  do = do.astype(jnp.bfloat16)")
        source = source.replace("    do = do_ref[...]", "    do = do_ref[...].astype(jnp.float32)")
        source = source.replace("{residuals}_bf16_storage", "{residuals}_bf16_storage_dout_di_fp32")
    if dkv_kv_buffers is not None:
        if dkv_kv_buffers not in (1, 2):
            raise ValueError("dKV K/V缓冲数量必须是1或2")
        # K/V沿Q循环复用；只控制它们的搬运缓冲，Q/dO及输出策略保持原样。
        marker = "  do_spec = o_spec\n\n  def logsumexp_index_map(\n      kv_index,"
        if source.count(marker) != 1:
            raise RuntimeError("Splash dKV缓冲源码定位失败")
        source = source.replace(
            marker,
            f"  k_spec = dataclasses.replace(k_spec, pipeline_mode=pl.Buffered({dkv_kv_buffers}))\n"
            f"  v_spec = dataclasses.replace(v_spec, pipeline_mode=pl.Buffered({dkv_kv_buffers}))\n" + marker,
        )
        source = source.replace("{residuals}_bf16_storage", f"{{residuals}}_bf16_storage_kvbuf{dkv_kv_buffers}")
    if dkv_vmem_scratch or dkv_output_buffers is not None:
        source = _dkv_scratch(source, vmem=dkv_vmem_scratch, output_buffers=dkv_output_buffers)
    if dkv_accumulate_outputs:
        if dkv_vmem_scratch or dkv_output_buffers not in (1, 2):
            raise ValueError("dKV输出累加要求1或2个输出缓冲且不另设VMEM scratch")
        source = _dkv_output_accumulator(source)
    if dq_vmem_scratch or dq_output_buffers is not None:
        if dq_output_buffers is not None and dq_output_buffers not in (1, 2):
            raise ValueError("dQ输出缓冲数量必须是1或2")
        start = source.index("def _flash_attention_dq_kernel(")
        end = source.index("def _flash_attention_dkv_kernel(", start)
        dq_source = source[start:end]
        changes = []
        if dq_vmem_scratch:
            changes.extend(
                [
                    (
                        "    # Outputs\n    dq_scratch_ref,\n    dq_ref,",
                        "    # Outputs\n    dq_ref,\n    # VMEM临时量：跨KV循环保留。\n    dq_scratch_ref,",
                    ),
                    ("      jax.ShapeDtypeStruct((bq, head_dim_qk), jnp.float32),\n", ""),
                    ("      pl.BlockSpec((bq, head_dim_qk), lambda *_: (0, 0)),\n", ""),
                    ("    _, dq = pl.pallas_call(", "    (dq,) = pl.pallas_call("),
                    (
                        "            grid=grid,",
                        "            grid=grid,\n"
                        "            scratch_shapes=(pltpu.VMEM((bq, head_dim_qk), jnp.float32),),",
                    ),
                ]
            )
        if dq_output_buffers is not None:
            changes.append(
                (
                    "pl.BlockSpec((None, bq, head_dim_qk), lambda h, i, *_: (h, i, 0)),",
                    "pl.BlockSpec((None, bq, head_dim_qk), lambda h, i, *_: (h, i, 0), "
                    f"pipeline_mode=pl.Buffered({dq_output_buffers})),",
                )
            )
        for before, after in changes:
            if dq_source.count(before) != 1:
                raise RuntimeError(f"Splash dQ scratch源码定位失败：{before!r}")
            dq_source = dq_source.replace(before, after)
        source = source[:start] + dq_source + source[end:]
        suffix = "_dqvmem" if dq_vmem_scratch else ""
        suffix += f"_dqout{dq_output_buffers}" if dq_output_buffers is not None else ""
        source = source.replace("{residuals}_bf16_storage", "{residuals}_bf16_storage" + suffix)
    if dq_compute_unroll and dq_kv_compute is None:
        raise ValueError("展开dQ循环要求指定计算块")
    if dq_kv_compute is not None:
        source = _split_dq_compute(source, dq_kv_compute, unroll=dq_compute_unroll)
    return source


@functools.cache
def module(
    *,
    bf16_dout: bool = False,
    dkv_kv_buffers: int | None = None,
    dkv_accumulate_outputs: bool = False,
    dkv_vmem_scratch: bool = False,
    dkv_output_buffers: int | None = None,
    dq_vmem_scratch: bool = False,
    dq_output_buffers: int | None = None,
    dq_kv_compute: int | None = None,
    dq_compute_unroll: bool = False,
) -> types.ModuleType:
    source = transformed_source(
        bf16_dout=bf16_dout,
        dkv_kv_buffers=dkv_kv_buffers,
        dkv_accumulate_outputs=dkv_accumulate_outputs,
        dkv_vmem_scratch=dkv_vmem_scratch,
        dkv_output_buffers=dkv_output_buffers,
        dq_vmem_scratch=dq_vmem_scratch,
        dq_output_buffers=dq_output_buffers,
        dq_kv_compute=dq_kv_compute,
        dq_compute_unroll=dq_compute_unroll,
    )
    name = __name__ + ("._generated_dout_di_fp32" if bf16_dout else "._generated")
    if dkv_kv_buffers is not None:
        name += f"_kvbuf{dkv_kv_buffers}"
    if dkv_accumulate_outputs:
        name += "_dkvoutacc"
    if dkv_vmem_scratch:
        name += "_dkvvmem"
    if dkv_output_buffers is not None:
        name += f"_dkvout{dkv_output_buffers}"
    if dq_vmem_scratch:
        name += "_dqvmem"
    if dq_output_buffers is not None:
        name += f"_dqout{dq_output_buffers}"
    if dq_kv_compute is not None:
        name += f"_dqc{dq_kv_compute}"
    if dq_compute_unroll:
        name += "_unroll"
    filename = f"<{name}:{VARIANT}>"
    tree = ast.parse(source, filename=filename)
    # 使用安装库的接口类型，防止布局enum的identity比较或pytree结构分叉。
    shared_types = {"SegmentIds", "QKVLayout", "BlockSizes"}
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ClassDef) and node.name in shared_types)]
    variant = types.ModuleType(name)
    variant.__dict__.update({key: getattr(installed, key) for key in shared_types})
    variant.__file__ = filename
    variant.__package__ = __package__
    sys.modules[name] = variant
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    try:
        exec(compile(tree, filename, "exec"), variant.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        linecache.cache.pop(filename, None)
        raise
    return variant
