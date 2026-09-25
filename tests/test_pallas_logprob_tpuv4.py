"""检查四芯片 Pallas logprob 的尾部、重复目标及 softcap 边界。"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from gemma4_posttrain_jax.losses import per_token_logps
from gemma4_posttrain_jax.sharding import DATA_AXIS, make_mesh


def _assert_close(actual, expected, *, logps):
    """保留 logprob 与 BF16 梯度各自的绝对、相对误差要求。"""
    a, b = np.asarray(actual).astype(np.float64), np.asarray(expected).astype(np.float64)
    assert a.shape == b.shape
    assert np.isfinite(a).all() and np.isfinite(b).all()
    delta = a - b
    reference = np.linalg.norm(b)
    relative = np.linalg.norm(delta) / reference if reference else np.linalg.norm(delta)
    absolute_limit = 1e-4 if logps else 0.01 * max(1.0, float(np.max(np.abs(b))))
    assert np.max(np.abs(delta)) <= absolute_limit
    assert relative <= (1e-5 if logps else 0.008)


@pytest.mark.tpu
@pytest.mark.parametrize("softcap", [None, 30.0])
@pytest.mark.parametrize("scale", [0.04, 1.5])
@pytest.mark.parametrize("zero_gradient", [False, True])
def test_product_tail_repeated_targets_and_softcap(softcap, scale, zero_gradient):
    if len(jax.devices()) != 4 or any("TPU v4" not in d.device_kind for d in jax.devices()):
        pytest.skip("requires four TPU v4 chips")
    mesh = make_mesh()
    rng = np.random.default_rng(20260925)
    # token、词表和 hidden 都有尾部；每个分片边界的目标重复出现。
    w = jax.device_put(
        jnp.asarray(rng.normal(size=(1036, 160)) * scale, jnp.bfloat16),
        NamedSharding(mesh, P(DATA_AXIS)),
    )
    h = jax.device_put(jnp.asarray(rng.normal(size=(3, 171, 160)), jnp.bfloat16), NamedSharding(mesh, P()))
    target = np.resize(np.array([0, 258, 259, 517, 518, 776, 777, 1035], np.int32), 513).reshape(3, 171)
    t = jax.device_put(target, NamedSharding(mesh, P()))
    gradient = rng.normal(size=513).astype(np.float32)
    gradient[256:512] = 0
    if zero_gradient:
        gradient[:] = 0
        gradient[0] = -0.0
    dy = jax.device_put(gradient.reshape(3, 171), NamedSharding(mesh, P()))

    def evaluate(a, b, targets, cotangent, *, backend):
        value, pullback = jax.vjp(
            lambda weights, hidden: per_token_logps(
                weights, hidden, targets, softcap=softcap, mesh=mesh, backend=backend
            ),
            a,
            b,
        )
        return value, pullback(cotangent)

    reference = jax.block_until_ready(jax.jit(lambda *args: evaluate(*args, backend="jax"))(w, h, t, dy))
    actual = jax.block_until_ready(jax.jit(lambda *args: evaluate(*args, backend="pallas"))(w, h, t, dy))
    for index, (left, right) in enumerate(zip(jax.tree.leaves(actual), jax.tree.leaves(reference), strict=True)):
        _assert_close(left, right, logps=index == 0)
    np.testing.assert_array_equal(actual[0], reference[0])
    np.testing.assert_array_equal(actual[1][1], reference[1][1])
    np.testing.assert_array_equal(np.asarray(actual[1][1]).reshape(513, 160)[256:512], 0)
