"""GRPO-family advantages, objectives, collation, and mathematical rewards."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gemma4_posttrain_jax.data import collate_grpo_prompts, grpo_prompt_indices
from gemma4_posttrain_jax.losses import (
    compute_advantages,
    compute_rloo_advantages,
    grpo_loss,
    grpo_train_step,
    init_train_state,
    make_optimizer,
    trainer_completion_logps,
)
from gemma4_posttrain_jax.model import forward_gemma4_lm
from gemma4_posttrain_jax.rewards import RewardConfig, score_completions


class FakeTokenizer:
    @staticmethod
    def apply_chat_template(messages, *, tokenize: bool, add_generation_prompt: bool = False):
        assert tokenize and add_generation_prompt and len(messages) == 1
        question_code = sum(messages[0]["content"].encode()) % 100
        return {"input_ids": [[2, question_code, 105]]}


def _fixed_completion_batch():
    prompt_ids = jnp.asarray([[0, 2, 13, 9], [2, 5, 16, 12]], jnp.int32)
    prompt_mask = jnp.asarray([[False, True, True, True], [True, True, True, True]])
    completion_ids = jnp.asarray([[41, 8, 1, 0], [6, 44, 18, 1]], jnp.int32)
    completion_mask = jnp.asarray([[True, True, True, False], [True, True, True, True]])
    return prompt_ids, prompt_mask, completion_ids, completion_mask


def _dense_completion_logps(params, config, prompt_ids, prompt_mask, completion_ids, completion_mask):
    input_ids = jnp.concatenate((prompt_ids, completion_ids), axis=-1)
    attention_mask = jnp.concatenate((prompt_mask, completion_mask), axis=-1)
    position_ids = jnp.maximum(jnp.cumsum(attention_mask, axis=-1) - 1, 0).astype(jnp.int32)
    output = forward_gemma4_lm(params, input_ids, position_ids, config=config, attention_mask=attention_mask)
    prompt_width, completion_width = prompt_ids.shape[1], completion_ids.shape[1]
    logits = output.logits[:, prompt_width - 1 : prompt_width + completion_width - 1].astype(jnp.float32)
    safe_targets = jnp.where(completion_mask, completion_ids, 0)
    selected = jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), safe_targets[..., None], axis=-1)[..., 0]
    return jnp.where(completion_mask, selected, 0.0)


def test_group_advantages_match_sample_std_and_rloo() -> None:
    rewards = jnp.asarray([1.0, 2.0, 4.0, -2.0, 0.0, 6.0], jnp.float32)
    grouped = np.asarray(rewards).reshape(2, 3)
    expected = (grouped - grouped.mean(axis=-1, keepdims=True)) / (grouped.std(axis=-1, ddof=1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(compute_advantages(rewards, 3), expected.reshape(-1), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        compute_advantages(rewards, 3, normalize_std=False),
        (grouped - grouped.mean(axis=-1, keepdims=True)).reshape(-1),
        rtol=1e-6,
        atol=1e-6,
    )
    loo = grouped - (grouped.sum(axis=-1, keepdims=True) - grouped) / 2
    np.testing.assert_allclose(compute_rloo_advantages(rewards, 3), loo.reshape(-1))


def test_grpo_mu_one_has_unit_ratio_but_nonzero_gradient() -> None:
    policy = jnp.asarray([[-2.0, -1.5, -0.7], [-1.2, -2.1, -3.0]], jnp.float32)
    advantages = jnp.asarray([-1.0, 2.0], jnp.float32)
    mask = jnp.asarray([[True, True, False], [True, True, True]])

    def objective(values):
        return grpo_loss(values, None, None, advantages, mask)[0]

    loss, metrics = grpo_loss(policy, None, None, advantages, mask)
    gradient = jax.grad(objective)(policy)
    np.testing.assert_allclose(metrics.ratio_mean, 1.0)
    np.testing.assert_allclose(metrics.ratio_min, 1.0)
    np.testing.assert_allclose(metrics.ratio_max, 1.0)
    np.testing.assert_allclose(loss, -0.5)
    np.testing.assert_allclose(gradient[0, :2], [0.25, 0.25])
    np.testing.assert_allclose(gradient[1], [-1 / 3, -1 / 3, -1 / 3])
    np.testing.assert_array_equal(gradient[0, 2], 0.0)


def test_grpo_clipping_kl_and_sequence_ratio_are_finite() -> None:
    policy = jnp.asarray([[-0.1, -1.2], [-2.0, -0.3]], jnp.float32)
    old = jnp.asarray([[-1.0, -1.0], [-1.0, -1.0]], jnp.float32)
    reference = jnp.asarray([[-0.3, -1.0], [-1.7, -0.5]], jnp.float32)
    advantages = jnp.asarray([1.0, -1.0], jnp.float32)
    mask = jnp.asarray([[True, True], [True, False]])
    loss, metrics = grpo_loss(
        policy,
        old,
        reference,
        advantages,
        mask,
        eps_low=0.2,
        eps_high=0.28,
        beta=0.04,
        kl_estimator="k3",
        agg_mode="token-mean",
        ratio_level="sequence",
        dual_clip_c=3.0,
    )
    assert bool(jnp.isfinite(loss))
    assert bool(
        jnp.isfinite(
            jax.grad(lambda values: grpo_loss(values, old, reference, advantages, mask, ratio_level="sequence")[0])(
                policy
            )
        ).all()
    )
    assert float(metrics.kl_loss) >= 0.0
    assert 0.0 <= float(metrics.pg_clipfrac) <= 1.0


def test_grpo_prompt_collation_repeats_groups_and_left_pads() -> None:
    examples = [
        {"question": "first", "answer": "reason\n#### 4"},
        {"question": "second", "answer": "reason\n#### 7"},
    ]
    batch = collate_grpo_prompts(examples, FakeTokenizer(), group_size=3, max_prompt_len=5, pad_token_id=0)
    assert batch.prompt_ids.shape == batch.prompt_mask.shape == (6, 5)
    np.testing.assert_array_equal(batch.prompt_ids[:, :2], 0)
    np.testing.assert_array_equal(batch.prompt_mask[:, :2], False)
    np.testing.assert_array_equal(batch.prompt_ids[0], batch.prompt_ids[1])
    assert batch.questions == ("first",) * 3 + ("second",) * 3
    assert batch.golds == ("4",) * 3 + ("7",) * 3


def test_math_rewards_keep_success_format_and_length_separate() -> None:
    output = score_completions(
        ["Work. The answer is \\boxed{14}.", "The answer is 12."],
        ["14", "14"],
        completion_lengths=[8, 10],
        config=RewardConfig(max_completion_length=10, overlong_buffer_length=4, overlong_penalty=1.0),
        workers=1,
    )
    np.testing.assert_array_equal(output.task_success, [1.0, 0.0])
    np.testing.assert_array_equal(output.base_reward, [1.0, 0.0])
    np.testing.assert_allclose(output.format_reward, [0.1, 0.0])
    np.testing.assert_allclose(output.length_penalty, [-0.5, -1.0])
    np.testing.assert_allclose(output.score, [0.6, -1.0])


def test_reward_workers_after_jax_initialization_match_serial() -> None:
    # 模拟真实 host loop 的顺序：先运行 JAX，再启动数学奖励 worker。
    jax.block_until_ready(jnp.arange(4) + 1)
    completions = ["The answer is \\boxed{14}.", "The answer is \\boxed{12}."]
    serial = score_completions(completions, ["14", "14"], workers=1)
    parallel = score_completions(completions, ["14", "14"], workers=2)
    for actual, expected in zip(parallel, serial, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_grpo_data_and_rng_resume_across_epoch_boundary() -> None:
    # 每个 epoch 两批，丢弃尾部一个样本；从全局步数可重建同一批和随机 key。
    continuous = [grpo_prompt_indices(9, 4, step, seed=7) for step in range(5)]
    resumed = [grpo_prompt_indices(9, 4, step, seed=7) for step in range(2, 5)]
    assert continuous[2:] == resumed
    assert len(set(continuous[0] + continuous[1])) == 8
    assert continuous[0] != continuous[2]
    base_key = jax.random.PRNGKey(7)
    keys = [jax.random.fold_in(base_key, step) for step in range(5)]
    for step in range(2, 5):
        np.testing.assert_array_equal(jax.random.fold_in(base_key, step), keys[step])
    assert not np.array_equal(keys[0], keys[1])


def test_trainer_completion_logps_match_dense_shift_and_gradient(grad_tiny) -> None:
    _, params, config = grad_tiny
    prompt_ids, prompt_mask, completion_ids, completion_mask = _fixed_completion_batch()

    def streaming(current):
        values = trainer_completion_logps(
            current,
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            config=config,
            compute_dtype=jnp.float32,
            vocab_chunk=17,
            sequence_chunk=5,
        )
        weights = jnp.asarray([[0.5, -0.3, 0.8, 9.0], [-0.2, 0.4, -0.7, 0.6]], jnp.float32)
        return (values * weights).sum(), values

    def dense(current):
        values = _dense_completion_logps(current, config, prompt_ids, prompt_mask, completion_ids, completion_mask)
        weights = jnp.asarray([[0.5, -0.3, 0.8, 9.0], [-0.2, 0.4, -0.7, 0.6]], jnp.float32)
        return (values * weights).sum(), values

    (streaming_value, streaming_logps), streaming_grads = jax.value_and_grad(streaming, has_aux=True)(params)
    (dense_value, dense_logps), dense_grads = jax.value_and_grad(dense, has_aux=True)(params)
    np.testing.assert_allclose(streaming_logps, dense_logps, atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(streaming_value, dense_value, atol=3e-6, rtol=3e-6)
    np.testing.assert_array_equal(np.asarray(streaming_logps)[~np.asarray(completion_mask)], 0.0)
    squared_delta = sum(
        float(jnp.vdot(actual - expected, actual - expected))
        for actual, expected in zip(jax.tree.leaves(streaming_grads), jax.tree.leaves(dense_grads), strict=True)
    )
    squared_reference = sum(float(jnp.vdot(expected, expected)) for expected in jax.tree.leaves(dense_grads))
    gradient_relative_l2 = np.sqrt(squared_delta / squared_reference)
    print(
        f"trainer completion parity: logp_max_abs={float(jnp.max(jnp.abs(streaming_logps - dense_logps))):.9g} "
        f"gradient_relative_l2={gradient_relative_l2:.9g}"
    )
    assert gradient_relative_l2 < 2e-5


def test_tiny_grpo_update_has_unit_ratio_respects_mask_and_freeze(grad_tiny) -> None:
    _, params, config = grad_tiny
    prompt_ids, prompt_mask, completion_ids, completion_mask = _fixed_completion_batch()
    advantages = jnp.asarray([1.0, -0.5], jnp.float32)
    optimizer, trainable = make_optimizer(params, learning_rate=1e-3, freeze_embeddings=True)
    state = init_train_state(params, optimizer)

    step = jax.jit(
        lambda current, tokens: grpo_train_step(
            current,
            prompt_ids,
            prompt_mask,
            tokens,
            completion_mask,
            advantages,
            None,
            None,
            config=config,
            optimizer=optimizer,
            trainable_mask=trainable,
            compute_dtype=jnp.float32,
            vocab_chunk=17,
            sequence_chunk=5,
        )
    )
    updated, metrics = step(state, completion_ids)
    masked_token_changed = completion_ids.at[0, -1].set(96)
    masked_updated, masked_metrics = step(state, masked_token_changed)

    assert int(updated.step) == 1
    assert float(metrics.grad_norm) > 0.0
    np.testing.assert_allclose(metrics.loss_metrics.ratio_mean, 1.0)
    np.testing.assert_allclose(metrics.loss_metrics.ratio_min, 1.0)
    np.testing.assert_allclose(metrics.loss_metrics.ratio_max, 1.0)
    np.testing.assert_array_equal(metrics.loss_metrics.completion_tokens, completion_mask.sum())
    np.testing.assert_array_equal(updated.params_f32.embed_tokens, params.embed_tokens)
    assert params.embed_tokens_per_layer is not None and updated.params_f32.embed_tokens_per_layer is not None
    np.testing.assert_array_equal(updated.params_f32.embed_tokens_per_layer, params.embed_tokens_per_layer)
    assert not np.array_equal(updated.params_f32.final_norm.weight, params.final_norm.weight)
    np.testing.assert_array_equal(masked_metrics.loss_metrics.loss, metrics.loss_metrics.loss)
    for actual, expected in zip(jax.tree.leaves(masked_updated), jax.tree.leaves(updated), strict=True):
        np.testing.assert_array_equal(actual, expected)
    final_norm_max_update = float(jnp.max(jnp.abs(updated.params_f32.final_norm.weight - params.final_norm.weight)))
    print(
        f"tiny GRPO update: loss={float(metrics.loss_metrics.loss):.9g} grad_norm={float(metrics.grad_norm):.9g} "
        f"final_norm_max_update={final_norm_max_update:.9g}"
    )


@pytest.mark.parametrize("aggregation", ["sequence-mean-token-mean", "token-mean", "sequence-mean-token-scale"])
def test_grpo_microbatch_matches_full_gradient_and_adam(grad_tiny, aggregation) -> None:
    # 不同微批的有效 token/行数不同，并含整行 padding，防止误用微批均值的平均。
    import optax

    _, params, config = grad_tiny
    tokens = tuple(jnp.tile(value, (2, 1)) for value in _fixed_completion_batch())
    tokens = (*tokens[:3], tokens[3].at[2].set(False).at[3, 1:].set(False))
    advantages = jnp.asarray([1.0, -0.5, 0.7, -1.2])
    reference = jnp.full(tokens[3].shape, -3.0)
    old = jnp.full(tokens[3].shape, -4.0)
    adam, trainable = make_optimizer(params, learning_rate=1e-3, freeze_embeddings=False)
    for optimizer in (optax.sgd(1.0), adam):
        initial = init_train_state(params, optimizer)

        def step(state, size, optimizer=optimizer):
            return grpo_train_step(
                state,
                *tokens,
                advantages,
                old,
                reference,
                config=config,
                optimizer=optimizer,
                trainable_mask=trainable,
                compute_dtype=jnp.float32,
                vocab_chunk=17,
                sequence_chunk=5,
                beta=0.04,
                agg_mode=aggregation,
                microbatch_size=size,
            )

        full, full_metrics = jax.jit(lambda state: step(state, None))(initial)
        split, split_metrics = jax.jit(lambda state: step(state, 2))(initial)
        for actual, expected in zip(jax.tree.leaves(split_metrics), jax.tree.leaves(full_metrics), strict=True):
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
        delta = sum(
            float(jnp.sum((a - b) ** 2))
            for a, b in zip(jax.tree.leaves(split.params_f32), jax.tree.leaves(full.params_f32), strict=True)
        )
        update = sum(
            float(jnp.sum((a - b) ** 2))
            for a, b in zip(jax.tree.leaves(full.params_f32), jax.tree.leaves(params), strict=True)
        )
        assert np.sqrt(delta / update) < 2e-4
        for actual, expected in zip(jax.tree.leaves(split.opt_state), jax.tree.leaves(full.opt_state), strict=True):
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-4)
        assert int(full.step) == int(split.step) == 1
    full_logps = trainer_completion_logps(params, *tokens, config=config, vocab_chunk=17, sequence_chunk=5)
    split_logps = jax.jit(
        lambda p: trainer_completion_logps(
            p, *tokens, config=config, vocab_chunk=17, sequence_chunk=5, microbatch_size=2
        )
    )(params)
    np.testing.assert_allclose(split_logps, full_logps, atol=3e-6, rtol=3e-6)


@pytest.mark.tpu
@pytest.mark.parametrize("beta,kl_clamp_value", [(0.0, None), (0.04, None), (0.04, 1e-6)])
def test_grpo_microbatch_tpu_matches_full_update(tpu_grad_tiny, beta, kl_clamp_value) -> None:
    from gemma4_posttrain_jax.sharding import make_mesh, replicate_scalars, shard_batch, shard_gemma4_text_params

    assert jax.default_backend() == "tpu" and jax.device_count() == 4
    _, host_params, config = tpu_grad_tiny
    mesh = make_mesh()
    params = shard_gemma4_text_params(host_params, config, mesh)
    tokens = shard_batch(tuple(jnp.tile(value, (4, 1)) for value in _fixed_completion_batch()), mesh)
    advantages = shard_batch(jnp.asarray([1.0, -0.5] * 4), mesh)
    optimizer, trainable = make_optimizer(params, learning_rate=1e-3, freeze_embeddings=True)
    initial = replicate_scalars(init_train_state(params, optimizer), mesh)
    reference = None
    if beta:
        reference_params = jax.tree.map(lambda x: x * 1.01, params)
        reference = jax.jit(
            lambda p: trainer_completion_logps(p, *tokens, config=config, vocab_chunk=16, sequence_chunk=8, mesh=mesh)
        )(reference_params)

    def step(state, size):
        return grpo_train_step(
            state,
            *tokens,
            advantages,
            None,
            reference,
            config=config,
            optimizer=optimizer,
            trainable_mask=trainable,
            compute_dtype=jnp.float32,
            vocab_chunk=16,
            sequence_chunk=8,
            mesh=mesh,
            microbatch_size=size,
            beta=beta,
            kl_clamp_value=kl_clamp_value,
        )

    full, full_metrics = jax.jit(lambda state: step(state, None))(initial)
    split, split_metrics = jax.jit(lambda state: step(state, 4))(initial)
    np.testing.assert_allclose(split_metrics.grad_norm, full_metrics.grad_norm, rtol=2e-5)
    if beta:
        assert float(full_metrics.loss_metrics.kl_loss) > 0
        if kl_clamp_value is not None:
            assert float(full_metrics.loss_metrics.kl_loss) <= kl_clamp_value * 1.00001
    for actual, expected in zip(jax.tree.leaves(split), jax.tree.leaves(full), strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
    assert int(split.step) == 1


@pytest.mark.parametrize("group_size", [2, 3, 4, 8])
def test_constant_reward_groups_have_exact_zero_advantages(group_size: int) -> None:
    rewards = jnp.repeat(jnp.asarray([0.0, 0.1, 1.0, 1.1, -0.3], jnp.float32), group_size)
    for normalize in (True, False):
        np.testing.assert_array_equal(compute_advantages(rewards, group_size, normalize_std=normalize), 0.0)
    np.testing.assert_array_equal(compute_rloo_advantages(rewards, group_size), 0.0)


@pytest.mark.tpu
def test_tpu_constant_group_does_not_create_policy_gradient() -> None:
    assert jax.default_backend() == "tpu"
    for reward in (0.1, 1.1, -0.3):
        rewards = jnp.full((256,), reward, jnp.float32)
        advantages = compute_advantages(rewards, 8)
        np.testing.assert_array_equal(advantages, 0.0)
        policy = jnp.full((256, 4), -0.5, jnp.float32)
        mask = jnp.ones(policy.shape, jnp.bool_)
        loss, gradient = jax.value_and_grad(lambda p, a, m: grpo_loss(p, None, None, a, m)[0])(policy, advantages, mask)
        np.testing.assert_array_equal(loss, 0.0)
        np.testing.assert_array_equal(gradient, 0.0)
