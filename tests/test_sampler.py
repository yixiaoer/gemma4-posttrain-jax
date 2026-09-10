"""CPU correctness tests for the pure-JAX rollout core."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from gemma4_posttrain_jax.model import SLIDING, forward_gemma4_lm, init_kv_cache, is_kv_shared
from gemma4_posttrain_jax.sampler import SamplerConfig, _sample_token, decode_step, generate, prefill


def _ids(values: torch.Tensor) -> jax.Array:
    return jnp.asarray(values.numpy(), dtype=jnp.int32)


def _assert_logits_close(actual: jax.Array, expected: jax.Array) -> None:
    # Changing the LM-head leading dimension changes XLA's f32 dot reduction order slightly.
    scale = max(float(jnp.abs(expected).max()), 1.0)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-5 * scale, rtol=0.0)


def test_kv_cache_uses_bounded_buffer_only_for_nonshared_sliding_layers(tiny_a) -> None:
    _, _, config = tiny_a
    capacity = 11
    cache = init_kv_cache(config, batch_size=2, capacity=capacity, dtype=jnp.float32)
    assert cache.key_mask.shape == (2, capacity)
    for i, layer in enumerate(cache.layers):
        if is_kv_shared(config, i):
            assert layer is None
            continue
        assert layer is not None
        expected = config.sliding_window if config.layer_types[i] == SLIDING else capacity
        assert layer.key.shape[1] == expected
        assert layer.value.shape == layer.key.shape


def test_sliding_cache_prefill_and_decode_match_full_forward_across_wrap(tiny_a) -> None:
    _, params, config = tiny_a
    batch_size, sequence_length, prompt_length = 2, 10, 5
    torch.manual_seed(91)
    input_ids = torch.randint(3, config.vocab_size, (batch_size, sequence_length))
    ids = _ids(input_ids)
    positions = jnp.arange(sequence_length, dtype=jnp.int32)[None].repeat(batch_size, axis=0)
    out = prefill(
        params,
        config,
        ids[:, :prompt_length],
        jnp.ones((batch_size, prompt_length), dtype=jnp.bool_),
        max_new_tokens=sequence_length - prompt_length,
    )
    assert out.kv_cache is not None
    expected = forward_gemma4_lm(
        params,
        ids[:, :prompt_length],
        positions[:, :prompt_length],
        config=config,
        logits_to_keep=1,
    ).logits
    _assert_logits_close(out.logits[:, -1], expected[:, -1])

    cache = out.kv_cache
    for t in range(prompt_length, sequence_length):
        out = decode_step(params, config, ids[:, t], positions[:, t], cache)
        assert out.kv_cache is not None
        cache = out.kv_cache
        expected = forward_gemma4_lm(
            params,
            ids[:, : t + 1],
            positions[:, : t + 1],
            config=config,
            logits_to_keep=1,
        ).logits
        _assert_logits_close(out.logits[:, -1], expected[:, -1])

    assert int(cache.length) == sequence_length
    assert sequence_length > 2 * config.sliding_window  # exercised two physical-slot wraps


def test_left_padded_batch_matches_unpadded_prefill_and_decode(tiny_a) -> None:
    _, params, config = tiny_a
    torch.manual_seed(92)
    row0 = torch.randint(3, config.vocab_size, (6,))
    row1 = torch.randint(3, config.vocab_size, (4,))
    padded_ids = torch.stack([row0, torch.cat([torch.zeros(2, dtype=torch.long), row1])])
    padded_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]], dtype=torch.bool)

    batch = prefill(params, config, _ids(padded_ids), jnp.asarray(padded_mask.numpy()), max_new_tokens=2)
    single = prefill(
        params,
        config,
        _ids(row1[None]),
        jnp.ones((1, row1.shape[0]), dtype=jnp.bool_),
        max_new_tokens=2,
    )
    _assert_logits_close(batch.logits[1], single.logits[0])
    assert batch.kv_cache is not None and single.kv_cache is not None

    next_tokens = jnp.asarray([17, 19], dtype=jnp.int32)
    batch_step = decode_step(
        params,
        config,
        next_tokens,
        jnp.asarray([6, 4], dtype=jnp.int32),
        batch.kv_cache,
    )
    single_step = decode_step(
        params,
        config,
        next_tokens[1:],
        jnp.asarray([4], dtype=jnp.int32),
        single.kv_cache,
    )
    _assert_logits_close(batch_step.logits[1], single_step.logits[0])


@pytest.mark.parametrize("temperature,top_p", [(0.0, 1.0), (0.8, 1e-8)])
def test_generate_masks_every_slot_after_eos_and_scores_full_policy(tiny_a, temperature, top_p) -> None:
    _, params, config = tiny_a
    prompt_ids = jnp.asarray([[11, 12, 13, 14]], dtype=jnp.int32)
    prompt_mask = jnp.ones_like(prompt_ids, dtype=jnp.bool_)
    initial = prefill(params, config, prompt_ids, prompt_mask, max_new_tokens=4)
    first_token = int(jnp.argmax(initial.logits[0, -1]))
    expected_logp = torch.log_softmax(torch.tensor(np.asarray(initial.logits[0, -1]), dtype=torch.float32), dim=-1)[
        first_token
    ].item()
    sampler_config = SamplerConfig(
        max_prompt_len=4,
        max_new_tokens=4,
        temperature=temperature,
        top_k=0,
        top_p=top_p,
        eos_ids=(first_token,),
        seed=3,
    )
    rollout = generate(params, config, prompt_ids, prompt_mask, sampler_config=sampler_config)

    np.testing.assert_array_equal(np.asarray(rollout.completion_mask), [[True, False, False, False]])
    np.testing.assert_array_equal(np.asarray(rollout.completion_ids[0, 1:]), [config.pad_token_id] * 3)
    np.testing.assert_array_equal(np.asarray(rollout.rollout_logps[0, 1:]), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(rollout.rollout_logps[0, 0]), np.asarray(expected_logp), atol=1e-6)
    assert int(rollout.lengths[0]) == 1


@pytest.mark.parametrize(
    "temperature,top_k,top_p",
    [(0.0, 0, 1.0), (1.0, 0, 1.0), (0.7, 0, 1.0), (0.8, 7, 1.0), (0.8, 0, 0.9), (0.8, 7, 0.8)],
)
def test_generate_is_reproducible_for_fixed_key(tiny_a, temperature, top_k, top_p) -> None:
    _, params, config = tiny_a
    prompt_ids = jnp.asarray([[21, 22, 23], [0, 32, 33]], dtype=jnp.int32)
    prompt_mask = prompt_ids != 0
    sampler_config = SamplerConfig(3, 5, temperature=temperature, top_k=top_k, top_p=top_p, eos_ids=(), seed=17)
    key = jax.random.PRNGKey(123)
    compiled = jax.jit(
        lambda current_params, ids, mask, current_key: generate(
            current_params,
            config,
            ids,
            mask,
            sampler_config=sampler_config,
            key=current_key,
        )
    )
    first = compiled(params, prompt_ids, prompt_mask, key)
    second = compiled(params, prompt_ids, prompt_mask, key)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    np.testing.assert_array_equal(first.lengths, [5, 5])
    assert np.isfinite(np.asarray(first.rollout_logps)).all()


@pytest.mark.parametrize(
    "temperature,top_k,top_p",
    [(1.0, 0, 1.0), (0.7, 0, 1.0), (0.8, 3, 1.0), (0.8, 0, 0.85), (0.8, 4, 0.7), (0.8, 0, 1e-50)],
)
def test_sampling_distribution_matches_transformers_and_keeps_raw_logprob(temperature, top_k, top_p):
    """用独立Torch过滤器验证候选集合和频率，检查记录的仍是完整策略概率。"""
    from transformers.generation.logits_process import (
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    logits = torch.tensor([[1.1, -0.7, 0.3, 2.0, -2.0], [-1.0, 1.3, 0.6, -2.3, 0.1]], dtype=torch.float32)
    scores = TemperatureLogitsWarper(temperature)(None, logits)
    if top_k:
        scores = TopKLogitsWarper(top_k)(None, scores)
    if top_p < 1:
        scores = TopPLogitsWarper(top_p)(None, scores)
    expected = torch.softmax(scores, dim=-1).numpy()
    raw_logps = torch.log_softmax(logits, dim=-1).numpy()
    count = 8192
    inputs = jnp.repeat(jnp.asarray(logits.numpy()), count, axis=0)
    sample = jax.jit(lambda values, key: _sample_token(values, key, temperature=temperature, top_k=top_k, top_p=top_p))
    tokens, logps = sample(inputs, jax.random.PRNGKey(128))
    tokens = np.asarray(tokens).reshape(2, count)
    logps = np.asarray(logps).reshape(2, count)
    for row in range(2):
        frequencies = np.bincount(tokens[row], minlength=logits.shape[1]) / count
        assert np.all(expected[row, tokens[row]] > 0), "采到了过滤范围之外的token"
        np.testing.assert_allclose(frequencies, expected[row], atol=0.025, rtol=0)
        np.testing.assert_allclose(logps[row], raw_logps[row, tokens[row]], atol=2e-6, rtol=0)


def test_top_p_includes_threshold_crossing_token():
    # 0.5+0.3超过0.6时应保留前两项，不能只保留最高项或把第三项也加入。
    # 在CPU构造固定logits，避免把TPU默认log的误差混入采样器的输入。
    values = jnp.broadcast_to(jnp.asarray(np.log(np.asarray([0.5, 0.3, 0.2], np.float32))), (8192, 3))
    tokens, logps = jax.jit(lambda x, key: _sample_token(x, key, temperature=1.0, top_k=0, top_p=0.6))(
        values, jax.random.PRNGKey(29)
    )
    assert set(np.asarray(tokens).tolist()) == {0, 1}
    np.testing.assert_allclose(float(jnp.mean(tokens == 0)), 0.625, atol=0.025)
    np.testing.assert_allclose(logps, np.log(np.asarray([0.5, 0.3, 0.2]))[np.asarray(tokens)], atol=2e-6)


@pytest.mark.parametrize("top_p", [0.0, -0.5, 1.01, float("nan"), float("inf")])
def test_generate_rejects_invalid_top_p(tiny_a, top_p):
    _, params, config = tiny_a
    ids = jnp.asarray([[2, 5]], dtype=jnp.int32)
    with pytest.raises(ValueError, match="top_p"):
        generate(
            params, config, ids, jnp.ones_like(ids, dtype=jnp.bool_), sampler_config=SamplerConfig(2, 3, top_p=top_p)
        )
