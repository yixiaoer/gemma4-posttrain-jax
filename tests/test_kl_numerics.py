"""直接验证K3的值、梯度、mask与裁剪，不依赖研究脚本。"""

from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gemma4_posttrain_jax.losses import _kl_estimate, grpo_loss


def reference(differences):
    with localcontext() as ctx:
        ctx.prec = 70
        numbers = [Decimal.from_float(float(x)) for x in np.asarray(differences).flat]
        values = np.asarray([float(x.exp() - x - 1) for x in numbers]).reshape(differences.shape)
        gradients = np.asarray([float(x.exp() - 1) for x in numbers]).reshape(differences.shape)
    return values, gradients


@pytest.mark.parametrize("compiled", [False, True])
def test_product_k3_matches_high_precision_value_and_gradient(compiled):
    boundary = np.float32(0.01)
    differences = np.asarray(
        [
            -20,
            -1,
            -boundary,
            np.nextafter(-boundary, np.float32(-1)),
            -0.03125,
            -(2**-24),
            0,
            2**-24,
            2**-16,
            np.nextafter(boundary, np.float32(0)),
            boundary,
            np.nextafter(boundary, np.float32(1)),
            0.03125,
            1,
            20,
        ],
        np.float32,
    )
    expected, expected_grad = reference(differences)

    def fn(x):
        return _kl_estimate(-x, jnp.zeros_like(x), "k3")

    run = jax.jit(fn) if compiled else fn
    grad = jax.grad(lambda x: fn(x).sum())
    grad = jax.jit(grad) if compiled else grad
    actual = np.asarray(run(jnp.asarray(differences)))
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-22)
    np.testing.assert_allclose(grad(jnp.asarray(differences)), expected_grad, rtol=3e-5, atol=1e-15)
    assert np.all(actual >= 0)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("clamp", [1e-6, 10000.0, float(np.finfo(np.float32).max)])
def test_clamped_product_k3_has_finite_gradients_at_large_differences(compiled, clamp):
    differences = np.asarray([-1e20, -100, -1, 0, 1, 20, 88, 88.72283, 88.72284, 100, 1e20], np.float32)

    def fn(x):
        return _kl_estimate(-x, jnp.zeros_like(x), "k3", clamp_value=clamp)

    run = jax.jit(fn) if compiled else fn
    grad = jax.grad(lambda x: fn(x).sum())
    grad = jax.jit(grad) if compiled else grad
    values = np.asarray(run(jnp.asarray(differences)))
    gradients = np.asarray(grad(jnp.asarray(differences)))
    assert np.isfinite(values).all()
    assert np.isfinite(gradients).all()
    assert values[-1] == np.float32(clamp)
    np.testing.assert_array_equal(gradients[-2:], 0)
    # 巨大负差值不需要计算指数，数学值约等于-d-1。
    assert values[0] == np.float32(min(1e20, clamp))
    assert gradients[0] == (0 if clamp < 1e20 else -1)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("clamp", [None, 0.1])
def test_kl_loss_mask_reduction_and_policy_gradient(compiled, clamp):
    difference = np.asarray([[2**-20, -0.02, 0.3], [1, -1, 2]], np.float32)
    mask = np.asarray([[True, True, True], [True, True, False]])
    values, gradients = reference(difference)
    if clamp is not None:
        gradients = np.where(values > clamp, 0, gradients)
        values = np.minimum(values, clamp)
    weights = mask / mask.sum(axis=1, keepdims=True) / len(mask)

    def fn(policy):
        return grpo_loss(
            policy,
            None,
            jnp.zeros_like(policy),
            jnp.zeros(len(mask)),
            jnp.asarray(mask),
            beta=0.04,
            kl_clamp_value=clamp,
        )[0]

    run = jax.value_and_grad(fn)
    run = jax.jit(run) if compiled else run
    value, gradient = run(-jnp.asarray(difference))
    np.testing.assert_allclose(value, 0.04 * np.sum(values * weights), rtol=3e-6)
    np.testing.assert_allclose(gradient, -0.04 * gradients * weights, rtol=3e-5, atol=1e-12)


@pytest.mark.parametrize("clamp", [0, -1, float("inf"), float("nan"), 1e100])
def test_kl_clamp_rejects_values_not_supported_by_float32(clamp):
    with pytest.raises(ValueError, match="kl_clamp_value"):
        grpo_loss(
            jnp.zeros((1, 1)), None, jnp.zeros((1, 1)), jnp.ones(1), jnp.ones((1, 1)), beta=0.04, kl_clamp_value=clamp
        )


@pytest.mark.parametrize("compiled", [False, True])
def test_unclamped_large_negative_difference_has_finite_gradient(compiled):
    def loss(x):
        return _kl_estimate(-x, jnp.zeros_like(x), "k3").sum()

    run = jax.value_and_grad(loss)
    if compiled:
        run = jax.jit(run)
    value, gradient = run(jnp.asarray([-1e20], jnp.float32))
    assert float(value) == float(np.float32(1e20))
    np.testing.assert_array_equal(gradient, [-1.0])
