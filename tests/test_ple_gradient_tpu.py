"""覆盖三维PLE查表反向与微批scan的交互，防止累计梯度在分片上丢失。"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gemma4_posttrain_jax.model import forward_per_layer_inputs
from gemma4_posttrain_jax.sharding import constrain_batch, make_mesh, shard_batch, shard_gemma4_text_params


@pytest.mark.tpu
@pytest.mark.parametrize("rows,micro", [(8, 4), (16, 8)])
def test_tpu_ple_microbatch_gradient_retains_every_row(tpu_grad_tiny, rows: int, micro: int) -> None:
    assert jax.default_backend() == "tpu" and jax.device_count() == 4
    _, host, config = tpu_grad_tiny
    mesh = make_mesh()
    params = shard_gemma4_text_params(host, config, mesh)
    table = params.embed_tokens_per_layer
    assert table is not None
    ids = np.tile(np.asarray([[0, 2, 13, 9, 41, 8, 1, 0], [2, 5, 16, 12, 6, 44, 18, 1]]), (rows // 2, 1))
    weights = (
        np.arange(rows * 8 * config.num_hidden_layers * config.hidden_size_per_layer_input).reshape(
            rows, 8, config.num_hidden_layers, config.hidden_size_per_layer_input
        )
        % 17
        - 8
    ).astype(np.float32)
    expected = np.zeros(table.shape, np.float32)
    derivative = ((weights / rows) * np.float32(2.0**-0.5)) * np.float32(math.sqrt(config.hidden_size_per_layer_input))
    np.add.at(expected, ids, derivative)
    indices, cotangent = shard_batch((ids, weights), mesh)

    def objective(value, index, weight):
        hidden = jnp.zeros((*index.shape, config.hidden_size), jnp.float32)
        output = forward_per_layer_inputs(params._replace(embed_tokens_per_layer=value), config, index, hidden)
        return jnp.sum(output * weight) / rows

    def accumulate(value, index, weight):
        def body(total, step):
            current_ids = constrain_batch(jax.lax.dynamic_slice_in_dim(index, step * micro, micro, axis=0), mesh)
            current_weights = constrain_batch(jax.lax.dynamic_slice_in_dim(weight, step * micro, micro, axis=0), mesh)
            gradient = jax.grad(objective)(value, current_ids, current_weights)
            return total + gradient, None

        return jax.lax.scan(body, jnp.zeros_like(value), jnp.arange(rows // micro))[0]

    actual = jax.jit(accumulate)(table, indices, cotangent)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
