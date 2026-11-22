"""Decode attention kernel against the XLA formula (interpret mode on CPU; the hardware case under ``tpu``)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gemma4_posttrain_jax.pallas.decode_attention import decode_attention, reference_attention


def _inputs(batch, dim, capacity, dtype, seed=0):
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((batch, 2, dim)), dtype)
    k = jnp.asarray(rng.standard_normal((batch, capacity, dim)) * 0.1, dtype)
    v = jnp.asarray(rng.standard_normal((batch, capacity, dim)), dtype)
    return q, k, v


@pytest.mark.parametrize(("starts", "length"), [((3, 5, 4, 3, 6, 5, 3, 4), 12), ((0, 0, 1, 2, 0, 3, 1, 0), 24)])
def test_interpret_matches_reference_in_float32(starts, length) -> None:
    q, k, v = _inputs(8, 16, 24, jnp.float32)
    starts = jnp.asarray(starts, jnp.int32)
    out = decode_attention(q, k, v, starts, jnp.asarray(length, jnp.int32), block=8, interpret=True)
    expected = reference_attention(q, k, v, starts, jnp.asarray(length, jnp.int32))
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-5, atol=1e-6)


def _errors_against_float32(q, k, v, starts, length, kernel_out):
    f32 = reference_attention(q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32), starts, length)
    xla = jax.jit(reference_attention)(q, k, v, starts, length)
    f32 = np.asarray(f32)
    return np.abs(np.asarray(kernel_out, np.float32) - f32), np.abs(np.asarray(xla, np.float32) - f32)


def test_interpret_bf16_is_as_accurate_as_the_xla_formula() -> None:
    """Against float32, the kernel errs no more than the bf16 XLA formula (both round p and v to bf16)."""
    q, k, v = _inputs(8, 128, 64, jnp.bfloat16, seed=1)
    starts = jnp.asarray([10, 20, 15, 30, 12, 18, 25, 11], jnp.int32)
    length = jnp.asarray(50, jnp.int32)
    out = decode_attention(q, k, v, starts, length, block=16, interpret=True)
    kernel_error, xla_error = _errors_against_float32(q, k, v, starts, length, out)
    assert kernel_error.max() <= 1.25 * xla_error.max()
    assert kernel_error.mean() <= 1.25 * xla_error.mean()


@pytest.mark.tpu
@pytest.mark.parametrize("batch", [8, 32])
@pytest.mark.parametrize("length", [513, 900, 1536])
def test_tpu_global_layer_shapes(batch, length) -> None:
    if jax.devices()[0].platform != "tpu":
        pytest.skip("needs a TPU")
    q, k, v = _inputs(batch, 512, 1536, jnp.bfloat16, seed=batch + length)
    starts = jnp.asarray(np.random.default_rng(length).integers(300, 500, size=batch), jnp.int32)
    n = jnp.asarray(length, jnp.int32)
    out = jax.jit(decode_attention)(q, k, v, starts, n)
    kernel_error, xla_error = _errors_against_float32(q, k, v, starts, n, out)
    assert kernel_error.max() <= 1.25 * xla_error.max()
    assert kernel_error.mean() <= 1.25 * xla_error.mean()
