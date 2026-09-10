"""PLE切线规则保持原查表的前向、负索引/越界和高阶可微接口。"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gemma4_posttrain_jax.model import _lookup_per_layer_tokens


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_ple_lookup_jvp_and_gradient_match_take(dtype) -> None:
    table = jnp.arange(72, dtype=jnp.float32).reshape(6, 3, 4).astype(dtype)
    direction = (table + 1).astype(dtype)
    ids = jnp.asarray([[0, 1, 1], [-1, 6, -7]])

    def candidate(value):
        return _lookup_per_layer_tokens(value, ids)

    def reference(value):
        return jnp.take(value, ids, axis=0)

    expected = jax.jvp(reference, (table,), (direction,))
    actual = jax.jvp(candidate, (table,), (direction,))
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_allclose(left.astype(jnp.float32), right.astype(jnp.float32), atol=0, rtol=0, equal_nan=True)
    for order in [1, 2]:

        def derivative(function, derivative_order):
            first = jax.grad(lambda value: jnp.nansum(jnp.square(function(value).astype(jnp.float32))))
            return first(table) if derivative_order == 1 else jax.jvp(first, (table,), (direction,))[1]

        np.testing.assert_allclose(
            derivative(candidate, order).astype(jnp.float32),
            derivative(reference, order).astype(jnp.float32),
            atol=0,
            rtol=0,
            equal_nan=True,
        )
