"""The merge + MLP kernel against the current path (one_shot_psum_norm, then XLA's MLP) on four TPU chips."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

pytestmark = pytest.mark.tpu


@pytest.mark.parametrize("rows", [8, 32])
@pytest.mark.parametrize("width", [1536, 3072])
@pytest.mark.parametrize("buffers", [2, 4])
@pytest.mark.parametrize("algorithm", ["one_shot", "scatter_gather"])
def test_merge_mlp_follows_the_current_path(rows, width, buffers, algorithm) -> None:
    if jax.devices()[0].platform != "tpu" or jax.device_count() != 4:
        pytest.skip("needs a TPU v4-8")
    if algorithm == "scatter_gather" and rows % 16:
        pytest.skip("the two-round merge needs 16-row tiles")
    from gemma4_posttrain_jax.model import MLPParams
    from gemma4_posttrain_jax.pallas.ici_allreduce import one_shot_psum_norm
    from gemma4_posttrain_jax.pallas.merge_mlp import merge_mlp
    from gemma4_posttrain_jax.tp_rollout import DP_AXIS, TP_AXIS, _mlp, make_tp_mesh

    mesh = make_tp_mesh(1, 4)
    dim, eps = 1536, 1e-6
    rng = np.random.default_rng(rows + width + buffers)
    bf16 = jnp.bfloat16
    specs = (P(TP_AXIS), P(), P(), P(), P(None, TP_AXIS), P(None, TP_AXIS), P(TP_AXIS, None))
    values = (
        rng.standard_normal((4 * rows, dim)).astype(np.float32),
        rng.standard_normal((rows, dim)),
        1 + 0.1 * rng.standard_normal((1, dim)),
        1 + 0.1 * rng.standard_normal((1, dim)),
        0.03 * rng.standard_normal((dim, 4 * width)),
        0.03 * rng.standard_normal((dim, 4 * width)),
        0.03 * rng.standard_normal((4 * width, dim)),
    )
    inputs = [
        jax.device_put(jnp.asarray(v, jnp.float32 if k == 0 else bf16), NamedSharding(mesh, s))
        for k, (v, s) in enumerate(zip(values, specs, strict=True))
    ]

    def current(partial, residual, post, pre, gate, up, down):
        x, h = one_shot_psum_norm(partial, residual, post, pre, eps=eps, axis_name=TP_AXIS, other_axes=(DP_AXIS,))
        return x, _mlp(MLPParams(gate, up, down), h[:, None, :], reduce=False)[:, 0]

    def fused(*args):
        return merge_mlp(*args, eps=eps, axis_name=TP_AXIS, other_axes=(DP_AXIS,), buffers=buffers, algorithm=algorithm)

    def run(fn):
        mapped = jax.shard_map(fn, mesh=mesh, in_specs=specs, out_specs=(P(TP_AXIS), P(TP_AXIS)), check_vma=False)
        return [np.asarray(a, np.float32) for a in jax.jit(mapped)(*inputs)]

    want_x, want_mlp = run(current)
    got_x, got_mlp = run(fused)
    # the same summation order and rounding points as one_shot_psum_norm (either merge algorithm)
    np.testing.assert_array_equal(got_x, want_x)
    scale = np.maximum(np.abs(want_mlp), np.sqrt(np.mean(want_mlp * want_mlp, axis=-1, keepdims=True)))
    # a bf16 step of one intermediate value moves the float32 partial by a small fraction of the row scale
    assert np.max(np.abs(got_mlp - want_mlp) / scale) < 2**-6


@pytest.mark.parametrize("width", [1536, 3072])
def test_merge_mlp_gathers_row_blocks(width) -> None:
    """The first merge stacking the chips' row blocks equals the one-shot sum of zero-padded blocks, bit for bit."""

    if jax.devices()[0].platform != "tpu" or jax.device_count() != 4:
        pytest.skip("needs a TPU v4-8")
    from gemma4_posttrain_jax.pallas.merge_mlp import merge_mlp
    from gemma4_posttrain_jax.tp_rollout import DP_AXIS, TP_AXIS, make_tp_mesh

    mesh = make_tp_mesh(1, 4)
    rows, dim, eps, local = 32, 1536, 1e-6, 8
    rng = np.random.default_rng(width)
    bf16 = jnp.bfloat16
    blocks = rng.standard_normal((4, local, dim)).astype(np.float32)
    padded = np.zeros((4, rows, dim), np.float32)
    for rank in range(4):
        padded[rank, rank * local : (rank + 1) * local] = blocks[rank]
    shared = [
        jax.device_put(jnp.asarray(v, bf16), NamedSharding(mesh, s))
        for v, s in (
            (rng.standard_normal((rows, dim)), P()),
            (1 + 0.1 * rng.standard_normal((1, dim)), P()),
            (1 + 0.1 * rng.standard_normal((1, dim)), P()),
            (0.03 * rng.standard_normal((dim, 4 * width)), P(None, TP_AXIS)),
            (0.03 * rng.standard_normal((dim, 4 * width)), P(None, TP_AXIS)),
            (0.03 * rng.standard_normal((4 * width, dim)), P(TP_AXIS, None)),
        )
    ]
    specs = (P(TP_AXIS), P(), P(), P(), P(None, TP_AXIS), P(None, TP_AXIS), P(TP_AXIS, None))

    def run(partial, algorithm):
        def body(*args):
            return merge_mlp(*args, eps=eps, axis_name=TP_AXIS, other_axes=(DP_AXIS,), algorithm=algorithm)

        mapped = jax.shard_map(body, mesh=mesh, in_specs=specs, out_specs=(P(TP_AXIS), P(TP_AXIS)), check_vma=False)
        first = jax.device_put(jnp.asarray(partial.reshape(-1, dim)), NamedSharding(mesh, P(TP_AXIS)))
        return [np.asarray(a, np.float32) for a in jax.jit(mapped)(first, *shared)]

    for value, expected in zip(run(blocks, "gather_rows"), run(padded, "one_shot"), strict=True):
        np.testing.assert_array_equal(value, expected)
