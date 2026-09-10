"""用论文公式、解析梯度与完整模型梯度验证算法组合，而非只检查配置字段。"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from test_grpo import _dense_completion_logps, _fixed_completion_batch

from gemma4_posttrain_jax.algorithms import (
    completion_loss_mask,
    mixed_success_groups,
    objective_advantages,
    objective_for_algorithm,
    truncated_importance_weights,
)
from gemma4_posttrain_jax.losses import grpo_loss, grpo_train_step, init_train_state, make_optimizer


def loss_options(config):
    return {key: value for key, value in config._asdict().items() if key != "advantage_estimator"}


def test_dynamic_sampling_uses_binary_success_and_keeps_whole_groups() -> None:
    success = np.asarray([0, 0, 1, 1, 0, 1, 1, 0], np.float32)
    np.testing.assert_array_equal(mixed_success_groups(success, 2), [False, False, True, True])
    with pytest.raises(ValueError, match="0/1"):
        mixed_success_groups(success + 0.1, 2)
    with pytest.raises(ValueError, match="连续G组"):
        mixed_success_groups(success[:-1], 2)


@pytest.mark.parametrize("algorithm", ["drgrpo", "rloo"])
def test_reinforce_length_weighting_matches_analytic_gradient(algorithm: str) -> None:
    config = objective_for_algorithm(algorithm, generation_budget=8)
    rewards = jnp.asarray([0.1, 1.1, 0.0, 1.0])
    advantage = objective_advantages(rewards, 2, config.advantage_estimator)
    policy = jnp.full((4, 4), -2.0)
    mask = jnp.asarray([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 0]], bool)
    actual = jax.grad(lambda p: grpo_loss(p, None, None, advantage, mask, **loss_options(config))[0])(policy)
    # RLOO G2为另一条回答的奖励；Dr.GRPO为组均值，整体除固定N=8。
    expected_advantage = jnp.asarray([-1.0, 1.0, -1.0, 1.0]) * (0.5 if algorithm == "drgrpo" else 1.0)
    denominator = 4 * (8 if algorithm == "drgrpo" else 1)
    np.testing.assert_allclose(actual, -expected_advantage[:, None] * mask / denominator, atol=1e-7, rtol=1e-6)


def test_gspo_token_matches_sequence_equation_with_constant_row_advantage() -> None:
    config = objective_for_algorithm("gspo-token", generation_budget=4)
    mask = jnp.asarray([[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 0, 0]], bool)
    old = jnp.full(mask.shape, -2.0)
    policy = old + jnp.asarray(
        [[0.02, -0.0198, 0, 0], [0.01, -0.01, 0.02, -0.0204], [0.2, 0.3, 0.1, 0], [-0.3, 0, 0, 0]]
    )
    advantage = jnp.asarray([0.7, -0.2, 0.3, -0.8])

    def reference(p):
        ratio = jnp.exp(((p - old) * mask).sum(-1) / mask.sum(-1))
        return -jnp.minimum(ratio * advantage, jnp.clip(ratio, 1 - 3e-4, 1 + 4e-4) * advantage).mean()

    expected_loss, expected_gradient = jax.value_and_grad(reference)(policy)
    actual_loss, actual_gradient = jax.value_and_grad(
        lambda p: grpo_loss(p, old, None, advantage, mask, **loss_options(config))[0]
    )(policy)
    np.testing.assert_allclose(actual_loss, expected_loss, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(actual_gradient, expected_gradient, atol=1e-7, rtol=1e-6)
    np.testing.assert_array_equal(actual_gradient[2:], 0.0)


def test_dapo_clip_higher_retains_positive_gradient_and_uses_total_tokens() -> None:
    policy = jnp.log(jnp.asarray([[1.24, 1.24], [0.75, 0.75]])) - 2
    old = jnp.full(policy.shape, -2.0)
    mask = jnp.asarray([[True, True], [True, False]])
    advantage = jnp.asarray([1.0, -1.0])
    config = objective_for_algorithm("dapo", generation_budget=2)
    value, gradient = jax.value_and_grad(lambda p: grpo_loss(p, old, None, advantage, mask, **loss_options(config))[0])(
        policy
    )
    np.testing.assert_allclose(value, (-2 * 1.24 + 0.8) / 3, atol=1e-6)
    np.testing.assert_allclose(gradient, [[-1.24 / 3, -1.24 / 3], [0, 0]], atol=1e-6)
    grpo_gradient = jax.grad(lambda p: grpo_loss(p, old, None, advantage, mask)[0])(policy)
    np.testing.assert_array_equal(grpo_gradient, 0.0)


def test_tis_is_frozen_capped_and_does_not_weight_kl() -> None:
    old = jnp.asarray([[-2.0, -3.0, -1.0], [-3.0, -2.0, -1.0]])
    behavior = old - jnp.log(jnp.asarray([[0.5, 4.0, 9.0], [1.0, 0.25, 0.75]]))
    mask = jnp.asarray([[1, 1, 0], [1, 1, 1]], bool)
    weights = truncated_importance_weights(old, behavior, mask, cap=2.0)
    for invalid_cap in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            truncated_importance_weights(old, behavior, mask, cap=invalid_cap)
    np.testing.assert_allclose(weights, [[0.5, 2, 0], [1, 0.25, 0.75]], atol=1e-6)
    for gradient in jax.grad(lambda a, b: truncated_importance_weights(a, b, mask, cap=2.0).sum(), argnums=(0, 1))(
        old, behavior
    ):
        np.testing.assert_array_equal(gradient, 0.0)
    reference = old - 0.2
    advantage = jnp.asarray([1.0, -1.0])

    def objective(policy, correction):
        return grpo_loss(policy, None, reference, advantage, mask, beta=0.1, sampler_is_weights=correction)[0]

    actual, weight_gradient = jax.grad(objective, argnums=(0, 1))(old, weights)
    expected = (
        (-advantage[:, None] * weights + 0.1 * (1 - jnp.exp(reference - old))) * mask / (2 * mask.sum(-1)[:, None])
    )
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
    np.testing.assert_array_equal(weight_gradient, 0.0)


def test_truncation_loss_filter_preserves_generated_context_mask() -> None:
    generated = jnp.asarray([[1, 1, 1], [1, 1, 0]], bool)
    truncated = jnp.asarray([True, False])
    np.testing.assert_array_equal(
        completion_loss_mask(generated, truncated, filter_truncated=True), [[0, 0, 0], [1, 1, 0]]
    )
    np.testing.assert_array_equal(completion_loss_mask(generated, truncated, filter_truncated=False), generated)
    np.testing.assert_array_equal(generated, [[1, 1, 1], [1, 1, 0]])


@pytest.mark.parametrize("microbatch_size", [None, 2])
def test_filtered_loss_and_tis_update_matches_dense_context_oracle(grad_tiny, microbatch_size) -> None:
    _, params, config = grad_tiny
    batch = jax.tree.map(lambda value: jnp.tile(value, (2, 1)), _fixed_completion_batch())
    loss_mask = batch[3].at[:2].set(False).at[2, 1].set(False)
    advantage = jnp.asarray([-1.0, 1.0, -0.5, 0.5])
    weights = jnp.asarray([[1, 2, 0.5, 0], [1, 1, 2, 1], [0.5, 1, 2, 0], [1, 0.5, 1, 2]], jnp.float32)
    old = _dense_completion_logps(params, config, *batch)
    reference = old - 0.2 * batch[3]
    optimizer, trainable = make_optimizer(params, learning_rate=1e-3)
    initial = init_train_state(params, optimizer)

    def dense_objective(current):
        effective = jax.tree.map(
            lambda value, train: value if train else jax.lax.stop_gradient(value), current, trainable
        )
        policy = _dense_completion_logps(effective, config, *batch)
        return grpo_loss(
            policy, old, reference, advantage, loss_mask, beta=0.04, agg_mode="token-mean", sampler_is_weights=weights
        )[0]

    expected_gradient = jax.jit(jax.grad(dense_objective))(params)

    def dense_update(state):
        gradient = jax.grad(dense_objective)(state.params_f32)
        updates, opt_state = optimizer.update(gradient, state.opt_state, state.params_f32)
        return optax.apply_updates(state.params_f32, updates), opt_state

    expected_params, expected_opt = jax.jit(dense_update)(initial)

    def run_step(state, transformation):
        return grpo_train_step(
            state,
            *batch,
            advantage,
            old,
            reference,
            config=config,
            optimizer=transformation,
            trainable_mask=trainable,
            compute_dtype=jnp.float32,
            vocab_chunk=17,
            sequence_chunk=5,
            beta=0.04,
            agg_mode="token-mean",
            microbatch_size=microbatch_size,
            loss_mask=loss_mask,
            sampler_is_weights=weights,
        )

    actual, metrics = jax.jit(lambda state: run_step(state, optimizer))(initial)
    # 通过测试用Optax状态保留实际trainer梯度，先检查数学，再检查Adam更新。
    collector = optax.GradientTransformation(
        lambda _: (),
        lambda gradient, _state, _params: (jax.tree.map(jnp.zeros_like, gradient), gradient),
    )
    captured, _ = jax.jit(lambda state: run_step(state, collector))(init_train_state(params, collector))
    difference = jax.tree.map(jnp.subtract, captured.opt_state, expected_gradient)
    relative_l2 = float(optax.tree.norm(difference) / optax.tree.norm(expected_gradient))
    print(f"filtered/TIS gradient relative L2: microbatch={microbatch_size} error={relative_l2:.9g}")
    assert relative_l2 < 1e-4
    for observed, expected in zip(jax.tree.leaves(captured.opt_state), jax.tree.leaves(expected_gradient), strict=True):
        np.testing.assert_allclose(observed, expected, atol=1e-6, rtol=1e-4)
    np.testing.assert_allclose(metrics.grad_norm, optax.tree.norm(expected_gradient), rtol=3e-5)
    for observed, expected in zip(
        jax.tree.leaves((actual.params_f32, actual.opt_state)),
        jax.tree.leaves((expected_params, expected_opt)),
        strict=True,
    ):
        np.testing.assert_allclose(observed, expected, atol=2e-5, rtol=3e-4)
    assert int(actual.step) == 1
