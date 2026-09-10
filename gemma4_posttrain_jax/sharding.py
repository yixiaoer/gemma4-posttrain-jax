"""Single-host TPU FSDP parameter and batch sharding helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .model import (
    AttentionParams,
    DecoderLayerParams,
    Gemma4TextConfig,
    Gemma4TextParams,
    MLPParams,
    PerLayerInputParams,
    RMSNormParams,
    is_kv_shared,
    layer_has_v_proj,
    layer_head_dim,
    layer_intermediate_size,
    layer_kv_heads,
)

DATA_AXIS = "d"


def make_mesh(devices: Sequence[Any] | None = None) -> Mesh:
    """Create the one-dimensional data/FSDP mesh over all supplied devices."""

    selected = list(jax.devices() if devices is None else devices)
    if not selected:
        raise ValueError("make_mesh needs at least one device")
    return Mesh(np.asarray(selected), (DATA_AXIS,))


def _largest_divisible_axis(shape: tuple[int, ...], num_devices: int) -> int | None:
    candidates = [axis for axis, size in enumerate(shape) if size >= num_devices and size % num_devices == 0]
    return max(candidates, key=lambda axis: shape[axis]) if candidates else None


def _matrix_spec(shape: tuple[int, ...], num_devices: int, *, force_axis: int | None = None) -> Any:
    if force_axis is not None:
        axis = force_axis if shape[force_axis] % num_devices == 0 else None
    else:
        axis = _largest_divisible_axis(shape, num_devices)
    if axis is None:
        return P()
    partitions: list[str | None] = [None] * len(shape)
    partitions[axis] = DATA_AXIS
    return P(*partitions)


def _norm_spec() -> Any:
    return RMSNormParams(cast(Any, P()))


def param_specs_fsdp(config: Gemma4TextConfig, num_devices: int = 4) -> Any:
    """Specs matching ``Gemma4TextParams``; shard each matrix on its largest divisible dimension.

    Token and PLE embedding tables are vocabulary-sharded when evenly divisible by the mesh,
    otherwise safely replicated. Vectors and scalars are replicated. This is FSDP-style parameter
    storage: GSPMD inserts collectives required by each operation while gradients and optimizer
    state retain the matching parameter partition.
    """

    if num_devices <= 0:
        raise ValueError("num_devices must be positive")
    M, V, L, Pdim = (
        config.hidden_size,
        config.vocab_size,
        config.num_hidden_layers,
        config.hidden_size_per_layer_input,
    )
    layers: list[DecoderLayerParams] = []
    for i in range(L):
        D, H, F = layer_head_dim(config, i), layer_kv_heads(config, i), layer_intermediate_size(config, i)
        R = config.num_attention_heads // H
        attention = AttentionParams(
            q_proj=_matrix_spec((M, R, H, D), num_devices),
            k_proj=None if is_kv_shared(config, i) else _matrix_spec((M, H, D), num_devices),
            v_proj=(
                None
                if not layer_has_v_proj(config, i) or is_kv_shared(config, i)
                else _matrix_spec((M, H, D), num_devices)
            ),
            o_proj=_matrix_spec((R, H, D, M), num_devices),
            q_norm=_norm_spec(),
            k_norm=None if is_kv_shared(config, i) else _norm_spec(),
        )
        per_layer_input = None
        if Pdim:
            per_layer_input = PerLayerInputParams(
                gate=_matrix_spec((M, Pdim), num_devices),
                projection=_matrix_spec((Pdim, M), num_devices),
                post_norm=_norm_spec(),
            )
        layers.append(
            DecoderLayerParams(
                input_norm=_norm_spec(),
                attention=attention,
                post_attention_norm=_norm_spec(),
                pre_feedforward_norm=_norm_spec(),
                mlp=MLPParams(
                    gate_proj=_matrix_spec((M, F), num_devices),
                    up_proj=_matrix_spec((M, F), num_devices),
                    down_proj=_matrix_spec((F, M), num_devices),
                ),
                post_feedforward_norm=_norm_spec(),
                per_layer_input=per_layer_input,
                layer_scalar=cast(Any, P()),
            )
        )
    return Gemma4TextParams(
        embed_tokens=_matrix_spec((V, M), num_devices, force_axis=0),
        embed_tokens_per_layer=(
            _matrix_spec((config.vocab_size_per_layer_input, L, Pdim), num_devices, force_axis=0) if Pdim else None
        ),
        per_layer_model_projection=_matrix_spec((M, L, Pdim), num_devices) if Pdim else None,
        per_layer_projection_norm=_norm_spec() if Pdim else None,
        layers=tuple(layers),
        final_norm=_norm_spec(),
    )


def param_specs_replicated(config: Gemma4TextConfig) -> Any:
    """A parameter-spec tree of fully replicated leaves."""

    return jax.tree.map(lambda _: P(), param_specs_fsdp(config), is_leaf=lambda x: isinstance(x, P))


def param_specs_rollout(config: Gemma4TextConfig, *, layout: str = "replicated", num_devices: int = 4) -> Any:
    """训练、周期评估与独立评估共用的采样权重布局。"""
    if layout == "replicated":
        return param_specs_replicated(config)
    if layout == "fsdp":
        return param_specs_fsdp(config, num_devices)
    raise ValueError("rollout layout只能为replicated或fsdp")


def named_shardings(specs: Any, mesh: Mesh) -> Any:
    """Convert a pytree of ``PartitionSpec`` leaves to ``NamedSharding`` leaves."""

    return jax.tree.map(lambda spec: NamedSharding(mesh, spec), specs, is_leaf=lambda x: isinstance(x, P))


def reshard(tree: Any, specs: Any, mesh: Mesh) -> Any:
    """Place or reshard every array to the corresponding specification."""

    shardings = named_shardings(specs, mesh)
    return jax.tree.map(jax.device_put, tree, shardings)


def shard_gemma4_text_params(
    params: Gemma4TextParams,
    config: Gemma4TextConfig,
    mesh: Mesh,
    *,
    replicated: bool = False,
) -> Gemma4TextParams:
    """Place model parameters according to the FSDP or replicated layout."""

    specs = param_specs_replicated(config) if replicated else param_specs_fsdp(config, mesh.size)
    return cast(Gemma4TextParams, reshard(params, specs, mesh))


def batch_spec(ndim: int) -> P:
    """Partition the leading batch dimension and replicate all remaining dimensions."""

    if ndim < 1:
        raise ValueError("a batch array must have at least one dimension")
    return P(DATA_AXIS, *([None] * (ndim - 1)))


def shard_batch(tree: Any, mesh: Mesh) -> Any:
    """Device-put a batch pytree with every array's leading dimension sharded on ``d``."""

    return jax.tree.map(lambda x: jax.device_put(x, NamedSharding(mesh, batch_spec(x.ndim))), tree)


def constrain_batch(x: Array, mesh: Mesh) -> Array:
    """Apply the batch-axis sharding constraint inside a compiled function."""

    return cast(Array, jax.lax.with_sharding_constraint(x, NamedSharding(mesh, batch_spec(x.ndim))))


def constrain_batch_tree(tree: Any, mesh: Mesh) -> Any:
    """Constrain every non-scalar array on its leading batch axis; replicate scalar leaves."""

    def constrain(value: Array) -> Array:
        spec = P() if value.ndim == 0 else batch_spec(value.ndim)
        return cast(Array, jax.lax.with_sharding_constraint(value, NamedSharding(mesh, spec)))

    return jax.tree.map(constrain, tree)


def _cast_and_constrain_params(params: Gemma4TextParams, dtype: Any, shardings: Any) -> Gemma4TextParams:
    return cast(
        Gemma4TextParams,
        jax.tree.map(
            lambda value, sharding: jax.lax.with_sharding_constraint(value.astype(dtype), sharding),
            params,
            shardings,
        ),
    )


def reshard_for_rollout(
    params_f32: Gemma4TextParams,
    config: Gemma4TextConfig,
    mesh: Mesh,
    *,
    dtype: Any = jnp.bfloat16,
    layout: str = "replicated",
) -> Gemma4TextParams:
    """转换采样dtype，并将各叶保持在登记的复制或FSDP布局。

    这是可放入长生命周期jit的纯函数；不覆盖原FP32 master，也不创建内部jit。
    调用方须使用相同布局的out_shardings，分别记录编译、转换和稳态时间。
    """

    shardings = named_shardings(param_specs_rollout(config, layout=layout, num_devices=mesh.size), mesh)
    return _cast_and_constrain_params(params_f32, dtype, shardings)


def reshard_after_rollout(
    rollout_params: Gemma4TextParams,
    config: Gemma4TextConfig,
    mesh: Mesh,
    *,
    dtype: Any = jnp.float32,
) -> Gemma4TextParams:
    """Cast a replicated rollout tree into the model's FSDP layout for round-trip measurements."""

    shardings = named_shardings(param_specs_fsdp(config, mesh.size), mesh)
    return _cast_and_constrain_params(rollout_params, dtype, shardings)


def replicate_scalars(tree: Any, mesh: Mesh) -> Any:
    """Place scalar state leaves on the full mesh instead of the process's default single device."""

    replicated = NamedSharding(mesh, P())
    return jax.tree.map(lambda x: jax.device_put(x, replicated) if x.ndim == 0 else x, tree)


def tree_shardings(tree: Any) -> Any:
    """Extract the array-sharding pytree used for explicit ``jit`` input/output contracts."""

    return jax.tree.map(lambda x: x.sharding, tree)
