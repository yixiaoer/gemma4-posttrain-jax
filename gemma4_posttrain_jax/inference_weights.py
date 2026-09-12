"""将 JAX 文本模型的各个参数数组同步到已有的 Gemma4 NNX 推理模型。

只负责参数布局、BF16 转换和同步传输。调用方须先排空请求并释放旧 KV，随后统一
刷新 runner.state/state_leaves、重建 KV 和提交版本。此模块不导入 tpu-inference。
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from gemma4_posttrain_jax.model import Gemma4TextConfig, Gemma4TextParams

PREFIX = "model.language_model."
PRESERVED_PREFIXES = ("model.vision_tower.", "model.embed_vision.")


def bf16_finite(value: np.ndarray) -> bool:
    """全部BF16编码的指数域检查，不执行浮点运算。"""
    from ml_dtypes import bfloat16

    if value.dtype != np.dtype(bfloat16):
        raise ValueError("要求真实bfloat16数组，不能把其他dtype当作BF16位编码")
    return bool(((value.view(np.uint16) & 0x7F80) != 0x7F80).all())


def bf16_bits_equal(actual: np.ndarray, expected: np.ndarray) -> bool:
    """按逻辑索引比较位；调用方另核expected有限，正负零也须保持原位。"""
    from ml_dtypes import bfloat16

    if actual.dtype != np.dtype(bfloat16) or expected.dtype != np.dtype(bfloat16):
        raise ValueError("校验双方必须为bfloat16")
    return bool(np.array_equal(actual.view(np.uint16), expected.view(np.uint16)))


class InferenceWeightError(RuntimeError):
    """保留部分更新、失败阶段与原数组恢复结果，供生命周期层保存证据。"""

    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class _Mapping:
    name: str
    shape: tuple[int, ...]
    sources: tuple[tuple[str, Any], ...]
    layout: str = "identity"
    shards: int = 1


def _source_leaves(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if value is None:
        return
    if isinstance(value, tuple):
        names = getattr(value, "_fields", tuple(str(i) for i in range(len(value))))
        for name, child in zip(names, value, strict=True):
            yield from _source_leaves(child, f"{prefix}.{name}" if prefix else name)
    else:
        yield prefix, value


def _make_plan(
    params: Gemma4TextParams, config: Gemma4TextConfig, merged_shards: Callable[[int], int]
) -> tuple[list[_Mapping], set[str]]:
    from gemma4_posttrain_jax.model import (
        is_kv_shared,
        layer_has_v_proj,
        layer_head_dim,
        layer_intermediate_size,
        layer_kv_heads,
    )
    from gemma4_posttrain_jax.weights import check_gemma4_text_params

    check_gemma4_text_params(params, config)
    m, n, layers, p = (
        config.hidden_size,
        config.num_attention_heads,
        config.num_hidden_layers,
        config.hidden_size_per_layer_input,
    )
    plan: list[_Mapping] = []
    preserved: set[str] = set()

    def add(name: str, shape: tuple[int, ...], source: str, value: Any, layout: str = "identity") -> None:
        plan.append(_Mapping(PREFIX + name, shape, ((source, value),), layout))

    add("embed_tokens.weight", (config.vocab_size, m), "embed_tokens", params.embed_tokens)
    add("norm.weight", (m,), "final_norm.weight", params.final_norm.weight)
    if p:
        if params.per_layer_projection_norm is None:
            raise ValueError("PLE 缺少归一化参数")
        add(
            "embed_tokens_per_layer.weight",
            (config.vocab_size_per_layer_input, layers * p),
            "embed_tokens_per_layer",
            params.embed_tokens_per_layer,
            "reshape",
        )
        add(
            "per_layer_model_projection.weight",
            (m, layers * p),
            "per_layer_model_projection",
            params.per_layer_model_projection,
            "reshape",
        )
        add(
            "per_layer_projection_norm.weight",
            (p,),
            "per_layer_projection_norm.weight",
            params.per_layer_projection_norm.weight,
        )
    for index, layer in enumerate(params.layers):
        key, source = f"layers.{index}.", f"layers.{index}."
        dim, heads, width = (
            layer_head_dim(config, index),
            layer_kv_heads(config, index),
            layer_intermediate_size(config, index),
        )
        attention = layer.attention
        add(key + "self_attn.q_proj.weight", (m, n, dim), source + "attention.q_proj", attention.q_proj, "q_heads")
        add(key + "self_attn.o_proj.weight", (n, dim, m), source + "attention.o_proj", attention.o_proj, "o_heads")
        add(key + "self_attn.q_norm.weight", (dim,), source + "attention.q_norm.weight", attention.q_norm.weight)
        if is_kv_shared(config, index):
            preserved.update(PREFIX + key + "self_attn." + name for name in ("k_proj.weight", "k_norm.weight"))
            if layer_has_v_proj(config, index):
                preserved.add(PREFIX + key + "self_attn.v_proj.weight")
        else:
            if attention.k_norm is None:
                raise ValueError("非共享层缺少 K norm")
            add(key + "self_attn.k_proj.weight", (m, heads, dim), source + "attention.k_proj", attention.k_proj)
            add(key + "self_attn.k_norm.weight", (dim,), source + "attention.k_norm.weight", attention.k_norm.weight)
            if layer_has_v_proj(config, index):
                add(key + "self_attn.v_proj.weight", (m, heads, dim), source + "attention.v_proj", attention.v_proj)
        shards = merged_shards(index)
        if shards <= 0 or width % shards:
            raise ValueError(f"融合 MLP 分片数非法: layer={index}, width={width}, shards={shards}")
        plan.append(
            _Mapping(
                PREFIX + key + "mlp.gate_up_proj.weight",
                (m, 2 * width),
                ((source + "mlp.gate_proj", layer.mlp.gate_proj), (source + "mlp.up_proj", layer.mlp.up_proj)),
                "gate_up_interleaved",
                shards,
            )
        )
        add(key + "mlp.down_proj.weight", (width, m), source + "mlp.down_proj", layer.mlp.down_proj)
        for target, attr in (
            ("input_layernorm", "input_norm"),
            ("post_attention_layernorm", "post_attention_norm"),
            ("pre_feedforward_layernorm", "pre_feedforward_norm"),
            ("post_feedforward_layernorm", "post_feedforward_norm"),
        ):
            add(key + target + ".weight", (m,), source + attr + ".weight", getattr(layer, attr).weight)
        add(key + "layer_scalar", (1,), source + "layer_scalar", layer.layer_scalar, "reshape")
        if layer.per_layer_input is not None:
            ple = layer.per_layer_input
            add(key + "per_layer_input_gate.weight", (m, p), source + "per_layer_input.gate", ple.gate)
            add(key + "per_layer_projection.weight", (p, m), source + "per_layer_input.projection", ple.projection)
            add(
                key + "post_per_layer_input_norm.weight",
                (m,),
                source + "per_layer_input.post_norm.weight",
                ple.post_norm.weight,
            )
    expected_sources = dict(_source_leaves(params))
    consumed = [name for item in plan for name, _ in item.sources]
    if Counter(consumed) != Counter(expected_sources.keys()):
        raise ValueError("权重映射必须使用全部训练参数，每个参数恰好使用一次")
    for name, value in expected_sources.items():
        if str(value.dtype) != "float32":
            raise ValueError(f"训练 master 必须保持 FP32: {name}={value.dtype}")
    return plan, preserved


def _host_bf16(item: _Mapping) -> np.ndarray:
    import jax
    from ml_dtypes import bfloat16

    arrays = []
    for name, value in item.sources:
        host = np.asarray(jax.device_get(value))
        if host.dtype != np.float32 or not np.isfinite(host).all():
            raise ValueError(f"训练 master 包含非法 dtype 或非有限值: {name}")
        arrays.append(host.astype(bfloat16))
        del host
    if item.layout == "gate_up_interleaved":
        gate, up = arrays
        m, width = gate.shape
        result = np.concatenate(
            (gate.reshape(m, item.shards, width // item.shards), up.reshape(m, item.shards, width // item.shards)),
            axis=-1,
        ).reshape(item.shape)
    elif item.layout == "q_heads":
        result = arrays[0].transpose(0, 2, 1, 3).reshape(item.shape)
    elif item.layout == "o_heads":
        result = arrays[0].transpose(1, 0, 2, 3).reshape(item.shape)
    elif item.layout == "reshape":
        result = arrays[0].reshape(item.shape)
    else:
        result = arrays[0]
    if result.shape != item.shape or not bf16_finite(result):
        raise ValueError(f"BF16 转换后 shape 或有限性不符: {item.name}")
    return np.ascontiguousarray(result)


def apply_gemma4_params(runner: Any, params: Gemma4TextParams, config: Gemma4TextConfig) -> dict[str, Any]:
    """更新已有 runner.model 的完整有效文本参数；保留原分片、Param对象和FP32 master。

    成功后缓存 state_leaves 仍由调用方刷新。失败时恢复已经替换的 Param 原数组，
    并通过 InferenceWeightError.report 返回恢复证据；不擅自继续 serving。
    """
    import jax
    from flax import nnx
    from ml_dtypes import bfloat16

    report: dict[str, Any] = {
        "complete": False,
        "stage": "validate",
        "updated": [],
        "preserved": [],
        "dispatch_refresh_required": True,
        "scope": "完整有效文本 FP32 master 到 BF16 NNX 参数；KV共享死参数与视觉参数原样保留。",
    }
    originals: dict[str, Any] = {}
    written: list[str] = []
    named: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        model = runner.model
        if type(model).__name__ != "Gemma4ForConditionalGeneration":
            raise ValueError("仅支持已资格验证的 Gemma4ForConditionalGeneration")
        if hasattr(model, "lm_head"):
            raise ValueError("当前权重映射只支持共享输入输出权重，不能忽略独立的 lm_head")
        pairs = list(model.named_parameters())
        named = dict(pairs)
        if len(named) != len(pairs):
            raise ValueError("NNX named_parameters 含重复名字")
        modules = dict(model.named_modules())

        def merged_shards(index: int) -> int:
            module = modules[PREFIX + f"layers.{index}.mlp.gate_up_proj"]
            method = module.quant_method
            if type(method).__name__ != "UnquantizedMergedLinearMethod":
                raise ValueError("只支持原 unquantized gate/up 融合布局")
            from gemma4_posttrain_jax.model import layer_intermediate_size

            width = layer_intermediate_size(config, index)
            if list(method.linear_config.output_sizes) != [width, width] or list(module.output_sizes) != [width, width]:
                raise ValueError("融合 MLP 的 loader/forward 输出宽度不一致")
            return int(method.linear_config.n_shards)

        plan, preserved_dead = _make_plan(params, config, merged_shards)
        destinations = {item.name for item in plan}
        if len(destinations) != len(plan):
            raise ValueError("映射目的参数重复")
        preserved = preserved_dead | {name for name in named if name.startswith(PRESERVED_PREFIXES)}
        if set(named) != destinations | preserved:
            raise ValueError(
                f"模型参数覆盖不符: missing={sorted((destinations | preserved) - set(named))}, "
                f"unexpected={sorted(set(named) - destinations - preserved)}"
            )
        originals = {name: param.get_value() for name, param in named.items()}
        if len({id(param) for param in named.values()}) != len(named):
            raise ValueError("NNX Param 存在未登记的别名")
        if len({id(value) for value in originals.values()}) != len(originals):
            raise ValueError("NNX 参数数组存在未登记的别名")
        before_state = nnx.state(model)
        before_leaves, treedef = jax.tree_util.tree_flatten(before_state)
        signature = [(tuple(a.shape), str(a.dtype)) for a in before_leaves]
        if cast(Any, jax.tree_util.tree_structure(runner.state)) != treedef:
            raise ValueError("已有 runner.state 与模型的 treedef 不一致")
        if len(runner.state_leaves) != len(before_leaves) or any(
            a is not b for a, b in zip(runner.state_leaves, before_leaves, strict=True)
        ):
            raise ValueError("更新前 dispatch tuple 已与实际模型分离")
        devices = set(runner.mesh.devices.flat)
        for item in plan:
            original = originals[item.name]
            if tuple(original.shape) != item.shape or original.dtype != np.dtype(bfloat16):
                raise ValueError(
                    f"目标 shape/dtype 不符: {item.name}: {original.shape}/{original.dtype}, "
                    f"expected={item.shape}/bfloat16"
                )
            if original.sharding.device_set != devices:
                raise ValueError(f"目标没有位于完整现有推理 mesh: {item.name}")
        report["preserved"] = [
            {"name": name, "reason": "shared_kv_unused" if name in preserved_dead else "vision_outside_text_training"}
            for name in sorted(preserved)
        ]
        report["training_leaf_count"] = len(list(_source_leaves(params)))
        report["updated_parameter_count"] = len(plan)
        report["model_parameter_count"] = len(named)
        for item in plan:
            report["stage"] = item.name
            host = _host_bf16(item)
            incoming = jax.device_put(host, originals[item.name].sharding)
            jax.block_until_ready(incoming)
            if (
                incoming.dtype != originals[item.name].dtype
                or incoming.shape != originals[item.name].shape
                or incoming.sharding != originals[item.name].sharding
            ):
                raise ValueError(f"传输改变目标 dtype/shape/sharding: {item.name}")
            shard_checks = []
            for shard in incoming.addressable_shards:
                actual = np.asarray(jax.device_get(shard.data))
                expected = host[shard.index]
                if not bf16_bits_equal(actual, expected):
                    raise ValueError(f"实际设备分片与 BF16 输入不同: {item.name}, device={shard.device.id}")
                shard_checks.append({"device_id": int(shard.device.id), "shape": list(actual.shape), "exact": True})
                del actual, expected
            written.append(item.name)
            named[item.name].set_value(incoming)
            report["updated"].append(
                {
                    "name": item.name,
                    "sources": [name for name, _ in item.sources],
                    "layout": item.layout,
                    "merged_shards": item.shards,
                    "shape": list(host.shape),
                    "dtype": "bfloat16",
                    "host_sha256": hashlib.sha256(memoryview(host.view(np.uint8)).cast("B")).hexdigest(),
                    "addressable_shards": shard_checks,
                    "new_array": incoming is not originals[item.name],
                }
            )
            del host, incoming
        after_leaves, after_treedef = jax.tree_util.tree_flatten(nnx.state(model))
        if cast(Any, after_treedef) != treedef or [(tuple(a.shape), str(a.dtype)) for a in after_leaves] != signature:
            raise ValueError("完整 NNX treedef 或 leaf shape/dtype 改变")
        if any(named[name].get_value() is not originals[name] for name in preserved):
            raise ValueError("非训练参数被意外替换")
        if any(named[name].get_value() is originals[name] for name in destinations):
            raise ValueError("应同步参数没有更换实际数组")
        report.update(
            complete=True,
            stage="complete",
            treedef_shape_dtype_preserved=True,
            preserved_arrays_identical=True,
            elapsed_s=time.perf_counter() - started,
        )
        return report
    except BaseException as error:
        report["error_type"], report["error_message"] = type(error).__name__, str(error)
        recovery = []
        for name in reversed(written):
            try:
                named[name].set_value(originals[name])
                recovery.append({"name": name, "restored_exact_array": named[name].get_value() is originals[name]})
            except BaseException as restore_error:
                recovery.append(
                    {
                        "name": name,
                        "restored_exact_array": False,
                        "error": f"{type(restore_error).__name__}: {restore_error}",
                    }
                )
        report["rollback"] = recovery
        report["rollback_complete"] = all(item["restored_exact_array"] for item in recovery)
        report["elapsed_s"] = time.perf_counter() - started
        raise InferenceWeightError(str(error), report) from error
