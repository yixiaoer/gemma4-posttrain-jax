"""The fused vocabulary projection and sampling kernel against XLA ops (interpret mode on CPU; TPU shapes too)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gemma4_posttrain_jax.pallas.vocab_sample import reference_partials, vocab_sample_partials
from gemma4_posttrain_jax.tp_rollout import gumbel_slice


def _inputs(batch: int, dim: int, vocab: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    hidden = jnp.asarray(rng.standard_normal((batch, dim)), jnp.bfloat16)
    embed = jnp.asarray(rng.standard_normal((vocab, dim)) * 0.2, jnp.bfloat16)
    key = jax.random.key_data(jax.random.PRNGKey(seed + 7))
    return hidden, embed, key


@pytest.mark.parametrize("temperature", [1.0, 0.7, 0.0])
@pytest.mark.parametrize("softcap", [30.0, None])
def test_partials_match_the_xla_ops(temperature, softcap) -> None:
    batch, dim, vocab, chips = 8, 256, 2048, 4
    hidden, embed, key = _inputs(batch, dim, vocab)
    options = dict(vocab_size=vocab * chips, temperature=temperature, softcap=softcap)
    args = (hidden, embed, key, jnp.int32(16), jnp.int32(vocab))
    got = jax.jit(lambda *a: vocab_sample_partials(*a, tile=256, interpret=True, **options))(*args)
    want = jax.jit(lambda *a: reference_partials(*a, **options))(*args)
    np.testing.assert_array_equal(np.asarray(got.position), np.asarray(want.position))
    np.testing.assert_array_equal(np.asarray(got.best), np.asarray(want.best))
    np.testing.assert_array_equal(np.asarray(got.logit), np.asarray(want.logit))
    np.testing.assert_array_equal(np.asarray(got.row_max), np.asarray(want.row_max))
    np.testing.assert_allclose(np.asarray(got.row_sumexp), np.asarray(want.row_sumexp), rtol=2e-6)


def test_noise_is_the_tp_sampler_noise() -> None:
    from gemma4_posttrain_jax.pallas.vocab_sample import _gumbel

    batch, vocab, chips = 8, 512, 4
    key = jax.random.key_data(jax.random.PRNGKey(3))
    row_offset, column_offset = 8, 2 * vocab
    rows = (row_offset + np.arange(batch, dtype=np.int64))[:, None]
    columns = (column_offset + np.arange(vocab, dtype=np.int64))[None, :]
    low = jnp.asarray((rows * vocab * chips + columns).astype(np.uint32))
    got = _gumbel(key[0], key[1], low)
    want = gumbel_slice(key, jnp.int32(row_offset), jnp.int32(column_offset), (batch, vocab), vocab * chips)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_ties_keep_the_first_index() -> None:
    batch, dim, vocab = 8, 128, 1024
    hidden = jnp.zeros((batch, dim), jnp.bfloat16)  # every logit 0: the first index wins
    embed = jnp.ones((vocab, dim), jnp.bfloat16)
    key = jax.random.key_data(jax.random.PRNGKey(0))
    got = vocab_sample_partials(
        hidden,
        embed,
        key,
        jnp.int32(0),
        jnp.int32(0),
        vocab_size=vocab,
        temperature=0.0,
        softcap=None,
        tile=128,
        interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(got.position), np.zeros(batch, np.int32))


@pytest.mark.tpu
@pytest.mark.parametrize("batch", [8, 32])
def test_tpu_shapes_follow_the_xla_logits(batch) -> None:
    """On TPU the kernel logits are those of the XLA einsum (bf16) and softcap up to one bf16 step."""
    if jax.devices()[0].platform != "tpu":
        pytest.skip("needs a TPU")
    hidden, embed, key = _inputs(batch, 1536, 65536)
    options = dict(vocab_size=4 * 65536, temperature=1.0, softcap=30.0)
    args = (hidden, embed, key, jnp.int32(0), jnp.int32(65536))
    got = jax.jit(lambda *a: vocab_sample_partials(*a, **options))(*args)

    def xla(hidden, embed):
        logits = jnp.einsum("bm,vm->bv", hidden, embed)
        return (jnp.tanh(logits / 30.0) * 30.0).astype(jnp.float32)

    logits = np.asarray(jax.jit(xla)(hidden, embed))
    # one bf16 step of a pre-softcap logit is at most 2**-7 of its size, and the softcap shrinks differences
    np.testing.assert_allclose(np.asarray(got.row_max), logits.max(axis=1), rtol=2**-7)
    rows = np.arange(batch)
    np.testing.assert_allclose(np.asarray(got.logit), logits[rows, np.asarray(got.position)], rtol=2**-7)
