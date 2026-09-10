"""在完整tiny模型上比较各目标的dense oracle与微批梯度/Adam，TPU另行执行。"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from test_grpo import _dense_completion_logps, _fixed_completion_batch

from gemma4_posttrain_jax.algorithms import objective_advantages, objective_for_algorithm
from gemma4_posttrain_jax.losses import grpo_loss, grpo_train_step, init_train_state, make_optimizer
from gemma4_posttrain_jax.sharding import make_mesh, replicate_scalars, shard_batch, shard_gemma4_text_params

ALGORITHMS = ("grpo", "drgrpo", "dapo", "gspo-token", "rloo")


def _check_adam_invariants(initial, actual, expected_gradient, grad_norm: float) -> tuple[float, int]:
    """从实际融合动量恢复梯度并独立核对首步Adam；不把近零梯度的符号当稳定oracle。"""

    adam = actual.opt_state[2].inner_state[0]
    assert int(adam.count) == 1

    def named(tree):
        return {
            jax.tree_util.keystr(path): np.asarray(value, np.float64)
            for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]
        }

    original, result, reference = map(named, (initial.params_f32, actual.params_f32, expected_gradient))
    first, second = named(adam.mu), named(adam.nu)
    error2 = reference2 = 0.0
    strict_mismatches = 0
    for path, value in original.items():
        if path not in first:
            np.testing.assert_array_equal(result[path], value)
            continue
        m, v = first[path], second[path]
        recovered_gradient = (m / 0.1) * max(grad_norm, 1.0)
        delta = recovered_gradient - reference[path]
        error2 += np.square(delta).sum()
        reference2 += np.square(reference[path]).sum()
        strict_mismatches += int(np.count_nonzero(np.abs(delta) > 1e-6 + 1e-4 * np.abs(reference[path])))
        # 第一轮零初态、无weight decay：NumPy f64公式独立检查两个动量和完整master更新。
        np.testing.assert_allclose(v, np.square(m / 0.1) * 0.001, atol=1e-20, rtol=1e-5)
        expected_parameter = value - 1e-3 * (m / 0.1) / (np.sqrt(v / 0.001) + 1e-8)
        np.testing.assert_allclose(result[path], expected_parameter, atol=1e-7, rtol=1e-6)
    relative_l2 = float(np.sqrt(error2 / reference2))
    assert relative_l2 < 1e-4
    return relative_l2, strict_mismatches


def _check_objective_update(fixture, algorithm: str, *, distributed: bool) -> None:
    _, host_params, config = fixture
    mesh = make_mesh() if distributed else None
    params = host_params if mesh is None else shard_gemma4_text_params(host_params, config, mesh)
    tokens = tuple(jnp.tile(value, (4, 1)) for value in _fixed_completion_batch())
    if mesh is not None:
        tokens = shard_batch(tokens, mesh)
    objective = objective_for_algorithm(algorithm, generation_budget=tokens[-1].shape[1])
    options = {name: value for name, value in objective._asdict().items() if name != "advantage_estimator"}
    advantage = objective_advantages(jnp.asarray([0.1, 1.1] * 4), 2, objective.advantage_estimator)
    initial_logps = jax.jit(lambda p: _dense_completion_logps(p, config, *tokens))(params)
    # 两个正负方向均覆盖裁剪区；离边界足够远，避免把舍入引起的分支变化当作数学误差。
    shift = jnp.asarray([0.5, 0.5, -0.5, -0.5] * 2)[:, None]
    old = None if algorithm == "rloo" else initial_logps - shift * tokens[-1]
    beta = 0.04 if algorithm == "grpo" else 0.0
    reference = initial_logps - 0.2 * tokens[-1] if beta else None
    if mesh is not None:
        advantage, old, reference = shard_batch((advantage, old, reference), mesh)
    optimizer, trainable = make_optimizer(params, learning_rate=1e-3)
    initial = init_train_state(params, optimizer)
    if mesh is not None:
        initial = replicate_scalars(initial, mesh)

    def dense_objective(current):
        effective = jax.tree.map(
            lambda value, train: value if train else jax.lax.stop_gradient(value), current, trainable
        )
        policy = _dense_completion_logps(effective, config, *tokens)
        return grpo_loss(policy, old, reference, advantage, tokens[-1], beta=beta, **options)[0]

    # 采用已有完整梯度gate的grad-only JIT参考边界，loss单独比较。
    expected_gradient = jax.jit(jax.grad(dense_objective))(params)
    expected_loss = jax.jit(dense_objective)(params)

    def dense_update(state):
        gradient = jax.grad(dense_objective)(state.params_f32)
        updates, opt_state = optimizer.update(gradient, state.opt_state, state.params_f32)
        return optax.apply_updates(state.params_f32, updates), opt_state

    def update(state, transformation, micro):
        return grpo_train_step(
            state,
            *tokens,
            advantage,
            old,
            reference,
            config=config,
            optimizer=transformation,
            trainable_mask=trainable,
            compute_dtype=jnp.float32,
            vocab_chunk=16 if distributed else 17,
            sequence_chunk=5,
            microbatch_size=micro,
            mesh=mesh,
            beta=beta,
            **options,
        )

    collector = optax.GradientTransformation(
        lambda _: (),
        lambda gradient, _state, _params: (jax.tree.map(jnp.zeros_like, gradient), gradient),
    )
    capture_state = init_train_state(params, collector)
    if mesh is not None:
        capture_state = replicate_scalars(capture_state, mesh)
    full, _ = jax.jit(lambda state: update(state, collector, None))(capture_state)
    gradient = full.opt_state
    difference = jax.tree.map(jnp.subtract, gradient, expected_gradient)
    relative_l2 = float(optax.tree.norm(difference) / optax.tree.norm(expected_gradient))
    assert relative_l2 < 1e-4
    dense_mismatches = sum(
        int(np.count_nonzero(np.abs(np.asarray(a) - np.asarray(b)) > 1e-6 + 1e-4 * np.abs(np.asarray(b))))
        for a, b in zip(jax.tree.leaves(gradient), jax.tree.leaves(expected_gradient), strict=True)
    )
    micro = 4 if distributed else 2
    captured, metrics = jax.jit(lambda state: update(state, collector, micro))(capture_state)
    micro_difference = jax.tree.map(jnp.subtract, captured.opt_state, gradient)
    micro_relative_l2 = float(optax.tree.norm(micro_difference) / optax.tree.norm(gradient))
    # 跨batch shape沿用微批gate的全局梯度/完整Adam口径；另记录严格逐元素差异，不掩盖舍入尾部。
    assert micro_relative_l2 < 1e-4
    strict_mismatches = sum(
        int(np.count_nonzero(np.abs(np.asarray(a) - np.asarray(b)) > 1e-6 + 1e-4 * np.abs(np.asarray(b))))
        for a, b in zip(jax.tree.leaves(captured.opt_state), jax.tree.leaves(gradient), strict=True)
    )
    np.testing.assert_allclose(metrics.loss_metrics.loss, expected_loss, atol=3e-6, rtol=3e-6)
    expected_state = jax.jit(dense_update)(initial)
    actual_state, actual_metrics = jax.jit(lambda state: update(state, optimizer, micro))(initial)
    # 保留原跨shape参数判据的失败计数；近零梯度经Adam放大时，它不是可靠的正确性gate。
    parameter_mismatches = sum(
        int(np.count_nonzero(np.abs(np.asarray(a) - np.asarray(b)) > 2e-5 + 3e-4 * np.abs(np.asarray(b))))
        for a, b in zip(jax.tree.leaves(actual_state.params_f32), jax.tree.leaves(expected_state[0]), strict=True)
    )
    for actual, expected in zip(
        jax.tree.leaves(actual_state.opt_state), jax.tree.leaves(expected_state[1]), strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=3e-4)
    np.testing.assert_allclose(actual_metrics.grad_norm, optax.tree.norm(expected_gradient), atol=1e-6, rtol=1e-4)
    fused_relative_l2, fused_mismatches = _check_adam_invariants(
        initial, actual_state, expected_gradient, float(actual_metrics.grad_norm)
    )
    assert int(actual_state.step) == 1
    assert float(actual_metrics.grad_norm) > 0
    print(
        f"{algorithm} backend={jax.default_backend()} dense/full relative_L2={relative_l2:.9g}, "
        f"full/micro relative_L2={micro_relative_l2:.9g}, "
        f"strict_element_mismatches(dense/full,full/micro)=({dense_mismatches},{strict_mismatches}); "
        f"fused_Adam_gradient relative_L2={fused_relative_l2:.9g}, strict_mismatches={fused_mismatches}, "
        f"original_dense_parameter_mismatches={parameter_mismatches}"
    )


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_cpu_algorithm_complete_update_matches_dense_oracle(grad_tiny, algorithm: str) -> None:
    if jax.default_backend() != "cpu":
        pytest.skip("CPU oracle只在显式CPU环境执行")
    _check_objective_update(grad_tiny, algorithm, distributed=False)


@pytest.mark.tpu
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_tpu_algorithm_complete_update_matches_dense_oracle(tpu_grad_tiny, algorithm: str) -> None:
    assert jax.default_backend() == "tpu" and jax.device_count() == 4
    _check_objective_update(tpu_grad_tiny, algorithm, distributed=True)
