"""Dense references for the streaming vocabulary logsumexp."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gemma4_posttrain_jax.losses import per_token_logps


def _dense_logps(embed: jax.Array, hidden: jax.Array, targets: jax.Array, softcap: float | None) -> jax.Array:
    logits = jnp.matmul(hidden, embed.T, preferred_element_type=jnp.float32)
    if softcap is not None:
        logits = jnp.tanh(logits / softcap) * softcap
    return jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), targets[..., None], axis=-1)[..., 0]


@pytest.mark.parametrize("softcap", [None, 3.0])
@pytest.mark.parametrize("vocab_chunk,sequence_chunk", [(5, 4), (8, 7), (23, 10)])
def test_streaming_logps_and_gradients_match_dense(
    softcap: float | None, vocab_chunk: int, sequence_chunk: int
) -> None:
    key_embed, key_hidden = jax.random.split(jax.random.key(17))
    embed = jax.random.normal(key_embed, (23, 7), dtype=jnp.float32) * 0.4
    hidden = jax.random.normal(key_hidden, (2, 5, 7), dtype=jnp.float32) * 0.7
    targets = jnp.asarray([[0, 4, 22, 8, 2], [11, 3, 19, 1, 7]], jnp.int32)

    def streaming(e, h):
        values = per_token_logps(
            e,
            h,
            targets,
            softcap=softcap,
            vocab_chunk=vocab_chunk,
            sequence_chunk=sequence_chunk,
        )
        return values.sum(), values

    def dense(e, h):
        values = _dense_logps(e, h, targets, softcap)
        return values.sum(), values

    (stream_value, stream_logps), stream_grads = jax.value_and_grad(streaming, argnums=(0, 1), has_aux=True)(
        embed, hidden
    )
    (dense_value, dense_logps), dense_grads = jax.value_and_grad(dense, argnums=(0, 1), has_aux=True)(embed, hidden)

    np.testing.assert_allclose(stream_logps, dense_logps, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(stream_value, dense_value, atol=2e-6, rtol=2e-6)
    for actual, expected in zip(stream_grads, dense_grads, strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)


def test_streaming_logps_validates_shapes() -> None:
    with pytest.raises(ValueError, match="does not match"):
        per_token_logps(jnp.ones((7, 3)), jnp.ones((2, 4, 3)), jnp.ones((2, 3), dtype=jnp.int32), softcap=None)
