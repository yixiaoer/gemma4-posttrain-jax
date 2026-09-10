"""采样布局切换保持FP32 master、完整权重值及评估恢复身份。"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from gemma4_posttrain_jax.evaluation import resolve_eval_rollout_layout
from gemma4_posttrain_jax.sharding import (
    make_mesh,
    named_shardings,
    param_specs_rollout,
    reshard_for_rollout,
    shard_gemma4_text_params,
)


@pytest.mark.parametrize("layout", ["replicated", "fsdp"])
def test_rollout_layout_cast_preserves_master_and_global_weights(grad_tiny, layout):
    _, host, config = grad_tiny
    mesh = make_mesh()
    master = shard_gemma4_text_params(host, config, mesh)
    before = [np.asarray(x).copy() for x in jax.tree.leaves(master)]
    shardings = named_shardings(param_specs_rollout(config, layout=layout, num_devices=mesh.size), mesh)
    compiled = jax.jit(
        lambda current: reshard_for_rollout(current, config, mesh, layout=layout), out_shardings=shardings
    )
    actual = jax.block_until_ready(compiled(master))
    for value, original, expected_sharding in zip(
        jax.tree.leaves(actual), before, jax.tree.leaves(shardings), strict=True
    ):
        assert value.sharding == expected_sharding
        assert np.asarray(value).tobytes() == original.astype(value.dtype).tobytes()
    for value, original in zip(jax.tree.leaves(master), before, strict=True):
        assert value.dtype == original.dtype and np.asarray(value).tobytes() == original.tobytes()
    if mesh.size > 1:
        assert any(not x.is_fully_replicated for x in jax.tree.leaves(actual)) == (layout == "fsdp")


@pytest.mark.parametrize(
    "requested,saved,expected",
    [
        (None, None, "replicated"),
        (None, {}, "replicated"),
        (None, {"rollout_layout": "fsdp"}, "fsdp"),
        ("fsdp", {}, "fsdp"),
        ("replicated", {"rollout_layout": "fsdp"}, "replicated"),
    ],
)
def test_eval_layout_inherits_checkpoint_or_explicit_comparison(requested, saved, expected):
    assert resolve_eval_rollout_layout(requested, saved) == expected


def test_unknown_layout_is_rejected(grad_tiny):
    _, _, config = grad_tiny
    with pytest.raises(ValueError, match="layout"):
        param_specs_rollout(config, layout="unknown")
    with pytest.raises(ValueError, match="layout"):
        resolve_eval_rollout_layout(None, {"rollout_layout": "unknown"})
