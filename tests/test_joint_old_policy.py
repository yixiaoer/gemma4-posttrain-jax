"""联合old捕获的梯度、独立冻结重放、TIS与动态标志编译边界。"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding

from gemma4_posttrain_jax.algorithms import objective_for_algorithm
from gemma4_posttrain_jax.losses import grpo_train_step, init_train_state, make_optimizer, trainer_completion_logps
from gemma4_posttrain_jax.sharding import (
    batch_spec,
    make_mesh,
    replicate_scalars,
    shard_batch,
    shard_gemma4_text_params,
    tree_shardings,
)


@pytest.mark.parametrize("algorithm", ["grpo", "gspo-token"])
@pytest.mark.parametrize("cap", [None, 1.5])
def test_joint_old_replay_and_frozen_is_match_explicit_reference(grad_tiny, algorithm, cap) -> None:
    assert jax.default_backend() == "cpu"
    check_joint_old_replay(grad_tiny, algorithm, cap)


@pytest.mark.tpu
@pytest.mark.parametrize("algorithm", ["grpo", "gspo-token"])
def test_joint_old_policy_on_four_tpu_shards(tpu_grad_tiny, algorithm) -> None:
    assert jax.default_backend() == "tpu" and jax.device_count() == 4
    check_joint_old_replay(tpu_grad_tiny, algorithm, 1.5, mesh=make_mesh())


def check_joint_old_replay(grad_tiny, algorithm, cap, *, mesh=None) -> None:
    _, params, config = grad_tiny
    if mesh is not None:
        params = shard_gemma4_text_params(params, config, mesh)
    ids = jnp.asarray([[0, 2, 7, 9], [2, 8, 3, 4], [0, 2, 7, 9], [2, 8, 3, 4]], jnp.int32)
    completions = jnp.asarray([[4, 5, 1, 0], [5, 1, 0, 0], [6, 7, 8, 1], [7, 2, 1, 0]], jnp.int32)
    advantage = jnp.asarray([1.0, -1.0, 1.0, -1.0])
    microbatch = 2
    if mesh is not None:
        ids, completions, advantage = shard_batch(
            (jnp.tile(ids, (2, 1)), jnp.tile(completions, (2, 1)), jnp.tile(advantage, 2)), mesh
        )
        microbatch = 4
    mask = completions != 0
    behavior = None
    if cap is not None:
        standalone = jax.jit(
            lambda p: trainer_completion_logps(
                p,
                ids,
                ids != 0,
                completions,
                mask,
                config=config,
                compute_dtype=jnp.float32,
                vocab_chunk=17,
                sequence_chunk=5,
                microbatch_size=microbatch,
                mesh=mesh,
            )
        )(params)
        # 选择明确跨过cap的两类采样概率；后面用NumPy检查实际捕获old与behavior的比值。
        behavior = standalone - jnp.log(jnp.tile(jnp.asarray([0.5, 3.0]), ids.shape[0] // 2))[:, None]
    optimizer, trainable = make_optimizer(params, learning_rate=1e-3)
    state = init_train_state(params, optimizer)
    if mesh is not None:
        state = replicate_scalars(state, mesh)
    objective = objective_for_algorithm(algorithm, generation_budget=4)
    options = {key: value for key, value in objective._asdict().items() if key != "advantage_estimator"}
    kwargs = dict(
        config=config,
        optimizer=optimizer,
        trainable_mask=trainable,
        compute_dtype=jnp.float32,
        vocab_chunk=17,
        sequence_chunk=5,
        microbatch_size=microbatch,
        mesh=mesh,
        **options,
    )

    def update(current, old, capture):
        return grpo_train_step(
            current,
            ids,
            ids != 0,
            completions,
            mask,
            advantage,
            old,
            None,
            use_current_policy_as_old=capture,
            behavior_logps=behavior,
            sampler_is_cap=cap,
            **kwargs,
        )

    placeholder = jnp.zeros(completions.shape)
    if mesh is None:
        step = jax.jit(update)
    else:
        placeholder = shard_batch(placeholder, mesh)
        step = jax.jit(
            update,
            in_shardings=(tree_shardings(state), NamedSharding(mesh, batch_spec(2)), None),
            out_shardings=(tree_shardings(state), None),
        )
    updated, first = step(state, placeholder, jnp.asarray(True))
    anchor = first.policy_logps
    assert anchor is not None
    if mesh is not None:
        anchor = shard_batch(anchor, mesh)
    anchor_host = np.asarray(anchor).copy()
    assert first.loss_metrics.ratio_min == first.loss_metrics.ratio_max == 1
    assert first.loss_metrics.pg_clipfrac == 0 and first.grad_norm > 0
    np.testing.assert_array_equal(anchor_host[~np.asarray(mask)], 0)
    replay, second = step(state, anchor, jnp.asarray(False))
    for actual, expected in zip(jax.tree.leaves((replay, second)), jax.tree.leaves((updated, first)), strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert step._cache_size() == 1

    weights = None
    if cap is not None:
        # 独立NumPy公式；不是再调用TIS函数作为oracle。
        expected_weights = np.where(np.asarray(mask), np.minimum(np.exp(anchor_host - np.asarray(behavior)), cap), 0)
        assert np.any(expected_weights[np.asarray(mask)] == cap)
        assert np.any((expected_weights > 0) & (expected_weights < cap))
        weights = jnp.asarray(expected_weights)
    explicit = jax.jit(
        lambda current: grpo_train_step(
            current, ids, ids != 0, completions, mask, advantage, anchor, None, sampler_is_weights=weights, **kwargs
        )
    )
    reference, reference_metrics = explicit(state)
    np.testing.assert_allclose(reference_metrics.grad_norm, first.grad_norm, atol=2e-5, rtol=1e-4)
    for actual, expected in zip(jax.tree.leaves(updated), jax.tree.leaves(reference), strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=3e-4)
    assert reference_metrics.policy_logps is None

    final, last = step(updated, anchor, jnp.asarray(False))
    assert int(final.step) == 2
    assert last.loss_metrics.ratio_min != 1 or last.loss_metrics.ratio_max != 1
    np.testing.assert_array_equal(anchor, anchor_host)
    assert step._cache_size() == 1
