"""TP rollout prototype on four CPU devices: same scores as the unsharded model, no weight gathers.

Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=4``; skipped otherwise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import TINY_A, build_hf_model
from tp_reference import fixed_history_decode

from gemma4_posttrain_jax.model import config_from_hf
from gemma4_posttrain_jax.sampler import decode_step, prefill
from gemma4_posttrain_jax.tp_rollout import (
    ALLREDUCE,
    AUTO_HEADS,
    AUTO_ROWS,
    check_tp_config,
    make_tp_mesh,
    prepare_tp_params,
    resolve_allreduce,
    tp_fixed_history,
    tp_param_specs,
    tp_prefill,
)
from gemma4_posttrain_jax.weights import convert_gemma4_text_params

pytestmark = pytest.mark.skipif(len(jax.devices()) != 4, reason="needs four (simulated) devices")

# E2B-like attention: one K/V head shared by eight query heads; vocabularies divisible by four.
TINY_MQA = {
    **TINY_A,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "vocab_size": 96,
    "vocab_size_per_layer_input": 96,
}


@pytest.fixture(scope="module")
def tiny_mqa():
    model = build_hf_model(TINY_MQA, seed=7, weight_std=0.03, norm_std=0.05, layer_scalar_min=0.8)
    config = config_from_hf(TINY_MQA)
    return convert_gemma4_text_params(model.state_dict().__getitem__, config), config


def _history():
    rng = np.random.default_rng(0)
    prompt_ids = rng.integers(3, 96, size=(4, 6)).astype(np.int32)
    prompt_mask = np.ones((4, 6), bool)
    prompt_mask[1, :2] = False  # left padding
    prompt_mask[3, :1] = False
    completion_ids = rng.integers(3, 96, size=(4, 8)).astype(np.int32)
    lengths = np.asarray([8, 5, 7, 3])
    completion_mask = np.arange(8)[None, :] < lengths[:, None]
    return prompt_ids, prompt_mask, completion_ids, completion_mask


def _reference(params, config, history):
    prompt_ids, prompt_mask, completion_ids, completion_mask = history
    out = prefill(params, config, jnp.asarray(prompt_ids), jnp.asarray(prompt_mask), max_new_tokens=8)

    def decode_fn(current, tokens, positions, kv, *, attention_mask):
        step = decode_step(current, config, tokens, positions, kv, attention_mask=attention_mask)
        return step.logits[:, -1], step.kv_cache

    scores, _ = fixed_history_decode(
        params,
        out.kv_cache,
        jnp.asarray(completion_ids),
        jnp.asarray(completion_mask),
        jnp.asarray(prompt_mask.sum(-1), jnp.int32),
        decode_fn=decode_fn,
        reference_length=8,
    )
    return np.asarray(out.logits[:, -1].astype(jnp.float32)), np.asarray(scores)


@pytest.mark.parametrize(("dp", "tp"), [(1, 4), (2, 2)])
@pytest.mark.parametrize(("dtype", "tolerance"), [(jnp.float32, 2e-5), (jnp.bfloat16, 3e-2)])
def test_tp_scores_match_unsharded_model(tiny_mqa, dp, tp, dtype, tolerance) -> None:
    params32, config = tiny_mqa
    check_tp_config(config, tp)
    history = _history()
    reference_params = jax.tree.map(lambda value: value.astype(dtype), params32)
    expected_logits, expected_scores = _reference(reference_params, config, history)

    mesh = make_tp_mesh(dp, tp)
    params, _ = prepare_tp_params(params32, mesh, dtype=dtype)
    specs = tp_param_specs(params32)
    prompt_ids, prompt_mask, completion_ids, completion_mask = history
    logits, cache = jax.jit(
        lambda current, ids, mask: tp_prefill(
            current, config, ids, mask, mesh=mesh, max_new_tokens=8, param_specs=specs
        )
    )(params, prompt_ids, prompt_mask)
    np.testing.assert_allclose(np.asarray(logits[:, -1], np.float32), expected_logits, atol=tolerance, rtol=tolerance)

    history_fn = jax.jit(
        lambda current, kv, ids, mask, lengths: tp_fixed_history(
            current, config, kv, ids, mask, lengths, mesh=mesh, param_specs=specs, reference_length=8
        )
    )
    lengths = prompt_mask.sum(-1).astype(np.int32)
    scores, _ = history_fn(params, cache, completion_ids, completion_mask, lengths)
    np.testing.assert_allclose(np.asarray(scores), expected_scores, atol=tolerance, rtol=tolerance)
    assert np.all(np.asarray(scores)[~completion_mask[:, 1:]] == 0)

    hlo = history_fn.lower(params, cache, completion_ids, completion_mask, lengths).compile().as_text()
    assert "all-gather" not in hlo, "the TP program must not gather weights"


@pytest.mark.parametrize(("dp", "tp"), [(1, 4), (2, 2)])
def test_attention_split_by_rows_scores_match_unsharded_model(tiny_mqa, dp, tp) -> None:
    import re

    params32, config = tiny_mqa
    history = _history()
    expected_logits, expected_scores = _reference(params32, config, history)
    mesh = make_tp_mesh(dp, tp)
    params, _ = prepare_tp_params(params32, mesh, dtype=jnp.float32, attention_rows=True)
    specs = tp_param_specs(params32, attention_rows=True)
    prompt_ids, prompt_mask, completion_ids, completion_mask = history
    logits, cache = jax.jit(
        lambda current, ids, mask: tp_prefill(
            current, config, ids, mask, mesh=mesh, max_new_tokens=8, param_specs=specs, attention_rows=True
        )
    )(params, prompt_ids, prompt_mask)
    np.testing.assert_allclose(np.asarray(logits[:, -1], np.float32), expected_logits, atol=2e-5, rtol=2e-5)
    first = next(layer for layer in cache.layers if layer is not None)
    assert first.key.addressable_shards[0].data.shape[0] == 4 // (dp * tp), "each chip keeps only its own rows"

    history_fn = jax.jit(
        lambda current, kv, ids, mask, lengths: tp_fixed_history(
            current,
            config,
            kv,
            ids,
            mask,
            lengths,
            mesh=mesh,
            param_specs=specs,
            reference_length=8,
            allreduce="xla_rows",
        )
    )
    lengths = prompt_mask.sum(-1).astype(np.int32)
    scores, _ = history_fn(params, cache, completion_ids, completion_mask, lengths)
    np.testing.assert_allclose(np.asarray(scores), expected_scores, atol=2e-5, rtol=2e-5)
    hlo = history_fn.lower(params, cache, completion_ids, completion_mask, lengths).compile().as_text()
    gathered = [
        re.search(r"= \w+\[(\d+)", line) for line in hlo.splitlines() if re.search(r"all-gather(-start)?\(", line)
    ]
    assert gathered, "the attention rows are gathered"
    # only the attention output rows (4 // dp per group) are gathered, never a weight
    assert {int(match.group(1)) for match in gathered if match} == {4 // dp}


@pytest.mark.parametrize(("dp", "tp"), [(1, 4), (2, 2)])
@pytest.mark.parametrize("temperature", [1.0, 0.0])
@pytest.mark.parametrize("split", [False, True], ids=["one-program", "split-programs"])
def test_attention_split_by_rows_follows_the_same_random_stream(tiny_mqa, dp, tp, temperature, split) -> None:
    from gemma4_posttrain_jax.sampler import SamplerConfig, generate
    from gemma4_posttrain_jax.tp_rollout import tp_generate, tp_generate_decode, tp_generate_prefill

    params32, config = tiny_mqa
    prompt_ids, prompt_mask, _, _ = _history()
    sampler_config = SamplerConfig(max_prompt_len=6, max_new_tokens=8, temperature=temperature, eos_ids=(1, 5))
    key = jax.random.fold_in(jax.random.PRNGKey(0), 0)
    expected = generate(
        params32, config, jnp.asarray(prompt_ids), jnp.asarray(prompt_mask), sampler_config=sampler_config, key=key
    )
    mesh = make_tp_mesh(dp, tp)
    params, _ = prepare_tp_params(params32, mesh, dtype=jnp.float32, attention_rows=True)
    specs = tp_param_specs(params32, attention_rows=True)
    ids, mask = jnp.asarray(prompt_ids), jnp.asarray(prompt_mask)
    options = dict(sampler_config=sampler_config, key=key, mesh=mesh, param_specs=specs, allreduce="xla_rows")
    if split:
        logits, cache = tp_generate_prefill(
            params, config, ids, mask, mesh=mesh, param_specs=specs, max_new_tokens=8, attention_rows=True
        )
        actual = tp_generate_decode(params, config, logits, cache, ids, mask, **options)
    else:
        actual = jax.jit(lambda current, i, m: tp_generate(current, config, i, m, **options))(params, ids, mask)
    np.testing.assert_array_equal(np.asarray(actual.completion_ids), np.asarray(expected.completion_ids))
    np.testing.assert_array_equal(np.asarray(actual.lengths), np.asarray(expected.lengths))
    np.testing.assert_allclose(np.asarray(actual.rollout_logps), np.asarray(expected.rollout_logps), atol=2e-5)


def test_tp_layout_rejects_indivisible_heads(tiny_mqa) -> None:
    _, config = tiny_mqa
    with pytest.raises(ValueError, match="divisible"):
        check_tp_config(config._replace(num_attention_heads=6, num_key_value_heads=1), 4)


def test_auto_allreduce_picks_by_rows_and_keeps_the_checkpoint_mode() -> None:
    assert {AUTO_ROWS, AUTO_HEADS} <= set(ALLREDUCE)
    tp = {"layout": "tp", "tp": 4}
    # B8/G4 rollouts and 32-question evaluation batches: whole 8-row blocks per chip.
    assert resolve_allreduce("auto", rows=[32, 32], **tp) == AUTO_ROWS
    assert resolve_allreduce(None, rows=[64], **tp) == AUTO_ROWS
    # B2/G4 rollouts or an 8-question evaluation batch, and options outside the row-split mode's tests.
    assert resolve_allreduce("auto", rows=[8, 32], **tp) == AUTO_HEADS
    assert resolve_allreduce("auto", rows=[32, 8], **tp) == AUTO_HEADS
    assert resolve_allreduce("auto", rows=[32], decode_kernel=True, **tp) == AUTO_HEADS
    assert resolve_allreduce("auto", rows=[32], prefill_widths=[128], **tp) == AUTO_HEADS
    # A checkpoint keeps its mode; TP checkpoints without the field used XLA's merges.
    assert resolve_allreduce("auto", rows=[32], saved={"tp_allreduce": "one_shot_fused"}, **tp) == "one_shot_fused"
    assert resolve_allreduce(None, rows=[32], saved={"rollout_layout": "tp"}, **tp) == "xla"
    # No one-shot kernel outside four-chip TP; explicit modes pass through.
    assert resolve_allreduce("auto", layout="fsdp", tp=4, rows=[32]) == "xla"
    assert resolve_allreduce("auto", layout="tp", tp=2, rows=[32]) == "xla"
    assert resolve_allreduce("one_shot_fused_sg", rows=[8], **tp) == "one_shot_fused_sg"


def test_first_max_index_is_jnp_argmax() -> None:
    from gemma4_posttrain_jax.tp_rollout import _first_max_index

    rng = np.random.default_rng(0)
    values = rng.integers(-3, 3, size=(64, 37)).astype(np.float32)  # many ties
    values[1] = -np.inf
    values[2, 5] = np.inf
    values[3, [4, 9]] = np.nan
    values[4] = np.nan
    values = jnp.asarray(values)
    got = _first_max_index(values, jnp.max(values, axis=-1))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(jnp.argmax(values, axis=-1)))


@pytest.mark.parametrize("typed", [False, True])
def test_gumbel_slice_reproduces_jax_random(typed) -> None:
    from gemma4_posttrain_jax.tp_rollout import gumbel_slice

    key = jax.random.key(3) if typed else jax.random.PRNGKey(3)
    expected = np.asarray(jax.random.gumbel(key, (4, 96), jnp.float32))
    for rows, columns in ((0, 0), (2, 24), (2, 72)):
        actual = gumbel_slice(key, jnp.int32(rows), jnp.int32(columns), (2, 24), 96)
        assert np.asarray(actual).tobytes() == expected[rows : rows + 2, columns : columns + 24].tobytes()


@pytest.mark.parametrize(("dp", "tp"), [(1, 4), (2, 2)])
@pytest.mark.parametrize("temperature", [1.0, 0.0])
@pytest.mark.parametrize(("decode_every_step", "decode_chunk"), [(True, 1), (False, 1), (True, 4)])
def test_tp_generate_follows_the_same_random_stream(
    tiny_mqa, dp, tp, temperature, decode_every_step, decode_chunk
) -> None:
    from gemma4_posttrain_jax.sampler import SamplerConfig, generate
    from gemma4_posttrain_jax.tp_rollout import tp_generate

    params32, config = tiny_mqa
    prompt_ids, prompt_mask, _, _ = _history()
    sampler_config = SamplerConfig(max_prompt_len=6, max_new_tokens=8, temperature=temperature, eos_ids=(1, 5))
    key = jax.random.fold_in(jax.random.PRNGKey(0), 0)
    expected = generate(
        params32, config, jnp.asarray(prompt_ids), jnp.asarray(prompt_mask), sampler_config=sampler_config, key=key
    )
    mesh = make_tp_mesh(dp, tp)
    params, _ = prepare_tp_params(params32, mesh, dtype=jnp.float32)
    actual = jax.jit(
        lambda current, ids, mask, rng: tp_generate(
            current,
            config,
            ids,
            mask,
            sampler_config=sampler_config,
            key=rng,
            mesh=mesh,
            param_specs=tp_param_specs(params32),
            decode_every_step=decode_every_step,
            decode_chunk=decode_chunk,
        )
    )(params, prompt_ids, prompt_mask, key)
    np.testing.assert_array_equal(np.asarray(actual.completion_ids), np.asarray(expected.completion_ids))
    np.testing.assert_array_equal(np.asarray(actual.completion_mask), np.asarray(expected.completion_mask))
    np.testing.assert_array_equal(np.asarray(actual.lengths), np.asarray(expected.lengths))
    np.testing.assert_allclose(np.asarray(actual.rollout_logps), np.asarray(expected.rollout_logps), atol=2e-5)


@pytest.mark.parametrize(("dp", "tp"), [(1, 4), (2, 2)])
@pytest.mark.parametrize("temperature", [1.0, 0.0])
@pytest.mark.parametrize("split", [False, True], ids=["one-program", "split-programs"])
def test_vocab_kernel_follows_the_same_random_stream(tiny_mqa, dp, tp, temperature, split) -> None:
    from gemma4_posttrain_jax.sampler import SamplerConfig, generate
    from gemma4_posttrain_jax.tp_rollout import tp_generate, tp_generate_decode, tp_generate_prefill

    params32, config = tiny_mqa
    prompt_ids, prompt_mask, _, _ = _history()
    sampler_config = SamplerConfig(max_prompt_len=6, max_new_tokens=8, temperature=temperature, eos_ids=(1, 5))
    key = jax.random.fold_in(jax.random.PRNGKey(0), 0)
    expected = generate(
        params32, config, jnp.asarray(prompt_ids), jnp.asarray(prompt_mask), sampler_config=sampler_config, key=key
    )
    mesh = make_tp_mesh(dp, tp)
    params, _ = prepare_tp_params(params32, mesh, dtype=jnp.float32)
    specs = tp_param_specs(params32)
    ids, mask = jnp.asarray(prompt_ids), jnp.asarray(prompt_mask)
    options = dict(sampler_config=sampler_config, key=key, mesh=mesh, param_specs=specs, vocab_kernel=True)
    if split:
        head, cache = tp_generate_prefill(
            params, config, ids, mask, mesh=mesh, param_specs=specs, max_new_tokens=8, vocab_kernel=True
        )
        actual = tp_generate_decode(params, config, head, cache, ids, mask, **options)
    else:
        actual = jax.jit(lambda current, i, m: tp_generate(current, config, i, m, **options))(params, ids, mask)
    np.testing.assert_array_equal(np.asarray(actual.completion_ids), np.asarray(expected.completion_ids))
    np.testing.assert_array_equal(np.asarray(actual.completion_mask), np.asarray(expected.completion_mask))
    np.testing.assert_allclose(np.asarray(actual.rollout_logps), np.asarray(expected.rollout_logps), atol=2e-5)


@pytest.mark.parametrize(("dp", "tp"), [(1, 4), (2, 2)])
@pytest.mark.parametrize(
    "real_lengths",
    [(2, 1, 2, 2), (3, 2, 4, 1), (6, 4, 6, 5), (2, 2, 6, 1)],
    ids=["narrow", "middle", "full", "mixed"],
)
def test_narrow_prefill_matches_the_full_width_rollout(tiny_mqa, dp, tp, real_lengths) -> None:
    """Prefill over the last 2 or 4 of 6 prompt columns when the real prompts fit, else the full width."""
    from gemma4_posttrain_jax.sampler import SamplerConfig, generate
    from gemma4_posttrain_jax.tp_rollout import tp_generate

    params32, config = tiny_mqa
    prompt_ids, _, _, _ = _history()
    prompt_mask = np.arange(6)[None, :] >= 6 - np.asarray(real_lengths)[:, None]
    sampler_config = SamplerConfig(max_prompt_len=6, max_new_tokens=8, temperature=1.0, eos_ids=(1, 5))
    key = jax.random.fold_in(jax.random.PRNGKey(0), 0)
    expected = generate(
        params32, config, jnp.asarray(prompt_ids), jnp.asarray(prompt_mask), sampler_config=sampler_config, key=key
    )
    mesh = make_tp_mesh(dp, tp)
    params, _ = prepare_tp_params(params32, mesh, dtype=jnp.float32)
    actual = jax.jit(
        lambda current, ids, mask, rng: tp_generate(
            current,
            config,
            ids,
            mask,
            sampler_config=sampler_config,
            key=rng,
            mesh=mesh,
            param_specs=tp_param_specs(params32),
            prefill_widths=(2, 4),
        )
    )(params, prompt_ids, prompt_mask, key)
    np.testing.assert_array_equal(np.asarray(actual.completion_ids), np.asarray(expected.completion_ids))
    np.testing.assert_array_equal(np.asarray(actual.lengths), np.asarray(expected.lengths))
    np.testing.assert_allclose(np.asarray(actual.rollout_logps), np.asarray(expected.rollout_logps), atol=2e-5)


@pytest.mark.parametrize(("dp", "tp"), [(1, 4), (2, 2)])
@pytest.mark.parametrize("prefill_widths", [(), (4,)], ids=["full-prefill", "narrow-prefill"])
def test_split_prefill_and_decode_programs_match_tp_generate(tiny_mqa, dp, tp, prefill_widths) -> None:
    """tp_generate_prefill and tp_generate_decode compiled apart give the rollout of tp_generate."""
    from gemma4_posttrain_jax.sampler import SamplerConfig
    from gemma4_posttrain_jax.tp_rollout import tp_generate, tp_generate_decode, tp_generate_prefill

    params32, config = tiny_mqa
    prompt_ids, _, _, _ = _history()
    prompt_mask = np.arange(6)[None, :] >= 6 - np.asarray([3, 2, 4, 1])[:, None]
    sampler_config = SamplerConfig(max_prompt_len=6, max_new_tokens=8, temperature=1.0, eos_ids=(1, 5))
    key = jax.random.fold_in(jax.random.PRNGKey(0), 0)
    mesh = make_tp_mesh(dp, tp)
    params, _ = prepare_tp_params(params32, mesh, dtype=jnp.float32)
    specs = tp_param_specs(params32)
    whole = jax.jit(
        lambda current, ids, mask, rng: tp_generate(
            current,
            config,
            ids,
            mask,
            sampler_config=sampler_config,
            key=rng,
            mesh=mesh,
            param_specs=specs,
            prefill_widths=prefill_widths,
        )
    )(params, prompt_ids, prompt_mask, key)
    logits, cache = jax.jit(
        lambda current, ids, mask: tp_generate_prefill(
            current, config, ids, mask, mesh=mesh, param_specs=specs, max_new_tokens=8, prefill_widths=prefill_widths
        )
    )(params, prompt_ids, prompt_mask)
    split = jax.jit(
        lambda current, step_logits, kv, ids, mask, rng: tp_generate_decode(
            current,
            config,
            step_logits,
            kv,
            ids,
            mask,
            sampler_config=sampler_config,
            key=rng,
            mesh=mesh,
            param_specs=specs,
        )
    )(params, logits, cache, prompt_ids, prompt_mask, key)
    np.testing.assert_array_equal(np.asarray(split.completion_ids), np.asarray(whole.completion_ids))
    np.testing.assert_array_equal(np.asarray(split.lengths), np.asarray(whole.lengths))
    np.testing.assert_allclose(np.asarray(split.rollout_logps), np.asarray(whole.rollout_logps), atol=1e-5)


@pytest.mark.parametrize(("dp", "tp"), [(1, 4), (2, 2)])
def test_decode_kernel_scores_match_the_xla_attention(tiny_mqa, dp, tp) -> None:
    """Fixed-history scores with the decode attention kernel on global layers (interpret mode on CPU)."""
    params32, config = tiny_mqa
    prompt_ids, prompt_mask, completion_ids, completion_mask = (np.concatenate([a, a]) for a in _history())
    mesh = make_tp_mesh(dp, tp)
    params, _ = prepare_tp_params(params32, mesh, dtype=jnp.float32)
    specs = tp_param_specs(params32)
    _, cache = tp_prefill(params, config, prompt_ids, prompt_mask, mesh=mesh, max_new_tokens=8, param_specs=specs)
    scores = {}
    for kernel in (False, True):
        scores[kernel], _ = tp_fixed_history(
            params,
            config,
            cache,
            jnp.asarray(completion_ids),
            jnp.asarray(completion_mask),
            jnp.asarray(prompt_mask.sum(-1), jnp.int32),
            mesh=mesh,
            param_specs=specs,
            reference_length=8,
            decode_kernel=kernel,
        )
    valid = np.asarray(completion_mask)[:, 1:]
    np.testing.assert_allclose(np.asarray(scores[True])[valid], np.asarray(scores[False])[valid], atol=2e-5)


def test_decode_window_list() -> None:
    from gemma4_posttrain_jax.tp_rollout import decode_window_list

    assert decode_window_list(512, 1536, 256, 256) == (
        (256, 768),
        (256, 1024),
        (256, 1280),
        (256, 1536),
        (0, 1536),
    )
    assert decode_window_list(512, 1536, 0, 256) == ()


@pytest.mark.parametrize(("dp", "tp"), [(1, 4), (2, 2)])
@pytest.mark.parametrize("real_lengths", [(2, 1, 2, 2), (6, 4, 6, 5)], ids=["windows", "long-prompt-fallback"])
def test_decode_windows_match_the_full_cache(tiny_mqa, dp, tp, real_lengths) -> None:
    """Global-layer attention over static windows [2, 8), [2, 10), [2, 12), [2, 14), else the whole cache."""
    from gemma4_posttrain_jax.sampler import SamplerConfig, generate
    from gemma4_posttrain_jax.tp_rollout import tp_generate

    params32, config = tiny_mqa
    prompt_ids, _, completion_ids, completion_mask = _history()
    prompt_mask = np.arange(6)[None, :] >= 6 - np.asarray(real_lengths)[:, None]
    mesh = make_tp_mesh(dp, tp)
    params, _ = prepare_tp_params(params32, mesh, dtype=jnp.float32)
    specs = tp_param_specs(params32)
    _, cache = tp_prefill(params, config, prompt_ids, prompt_mask, mesh=mesh, max_new_tokens=8, param_specs=specs)
    scores = {}
    for step in (0, 2):
        scores[step], _ = tp_fixed_history(
            params,
            config,
            cache,
            jnp.asarray(completion_ids),
            jnp.asarray(completion_mask),
            jnp.asarray(prompt_mask.sum(-1), jnp.int32),
            mesh=mesh,
            param_specs=specs,
            reference_length=8,
            decode_window_step=step,
            decode_window_lower=4,
        )
    valid = np.asarray(completion_mask)[:, 1:]
    np.testing.assert_allclose(np.asarray(scores[2])[valid], np.asarray(scores[0])[valid], atol=2e-5)

    sampler_config = SamplerConfig(max_prompt_len=6, max_new_tokens=8, temperature=1.0, eos_ids=(1, 5))
    key = jax.random.fold_in(jax.random.PRNGKey(0), 0)
    expected = generate(
        params32, config, jnp.asarray(prompt_ids), jnp.asarray(prompt_mask), sampler_config=sampler_config, key=key
    )
    actual = jax.jit(
        lambda current, ids, mask, rng: tp_generate(
            current,
            config,
            ids,
            mask,
            sampler_config=sampler_config,
            key=rng,
            mesh=mesh,
            param_specs=specs,
            decode_window_step=2,
            decode_window_lower=4,
        )
    )(params, prompt_ids, prompt_mask, key)
    np.testing.assert_array_equal(np.asarray(actual.completion_ids), np.asarray(expected.completion_ids))
    np.testing.assert_allclose(np.asarray(actual.rollout_logps), np.asarray(expected.rollout_logps), atol=2e-5)
