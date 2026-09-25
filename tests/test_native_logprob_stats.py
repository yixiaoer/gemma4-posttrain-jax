"""检查额外导出归一化统计是否保持原生概率和两路 VJP。"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from gemma4_posttrain_jax.losses import _vocab_parallel_logps_and_normalization, per_token_logps
from gemma4_posttrain_jax.sharding import DATA_AXIS, make_mesh


@pytest.mark.parametrize("softcap", [None, 30.0])
@pytest.mark.parametrize("scale", [0.04, 1.5])
def test_native_normalization_and_vjp(softcap, scale):
    if len(jax.devices()) != 4:
        pytest.skip("requires four CPU or TPU devices")
    mesh = make_mesh()
    rng = np.random.default_rng(20260925)
    w = jax.device_put(
        jnp.asarray(rng.normal(size=(1036, 128)) * scale, jnp.bfloat16), NamedSharding(mesh, P(DATA_AXIS))
    )
    h = jax.device_put(jnp.asarray(rng.normal(size=(513, 128)), jnp.bfloat16), NamedSharding(mesh, P()))
    token = jax.device_put(
        np.resize(np.array([0, 258, 259, 517, 518, 776, 777, 1035], np.int32), 513), NamedSharding(mesh, P())
    )
    gradient = rng.normal(size=513).astype(np.float32)
    gradient[256:512] = 0
    dy = jax.device_put(gradient, NamedSharding(mesh, P()))

    @jax.jit
    def run_candidate(a, b, t, g):
        value, pullback, statistics = jax.vjp(
            lambda x, y: _vocab_parallel_logps_and_normalization(
                x, y, t, mesh=mesh, softcap=softcap, vocab_chunk=128, sequence_chunk=256
            ),
            a,
            b,
            has_aux=True,
        )
        return (value, pullback(g)), statistics

    @jax.jit
    def run_reference(a, b, t, g):
        original, native_pullback = jax.vjp(
            lambda x, y: per_token_logps(x, y, t, mesh=mesh, softcap=softcap, vocab_chunk=128, sequence_chunk=256),
            a,
            b,
        )
        dense = jnp.matmul(b, a.T, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
        if softcap is not None:
            dense = jnp.tanh(dense / softcap) * softcap
        maximum = jnp.max(dense, axis=-1)
        total = jnp.exp(dense - maximum[:, None]).sum(axis=-1)
        return (original, native_pullback(g)), jnp.stack((maximum, total))

    actual, statistics = jax.block_until_ready(run_candidate(w, h, token, dy))
    original, independent = jax.block_until_ready(run_reference(w, h, token, dy))
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(original), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(statistics, independent, atol=2e-5, rtol=2e-5)
    assert bool(np.isfinite(np.asarray(statistics)).all())
    assert bool((np.asarray(statistics)[1] > 0).all())
