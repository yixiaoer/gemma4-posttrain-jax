"""检查 GRPO 后端配置穿过 microbatch、联合 old 捕获和实际更新。"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gemma4_posttrain_jax import losses
from gemma4_posttrain_jax.logprob_protocol import logprob_run_metadata
from scripts.train_grpo import parse_args, validate_resume_config


def completion_batch():
    prompt = jnp.asarray([[0, 2, 13, 9], [2, 5, 16, 12]], jnp.int32)
    prompt_mask = prompt != 0
    completion = jnp.asarray([[41, 8, 1, 0], [6, 44, 18, 1]], jnp.int32)
    mask = completion != 0
    return prompt, prompt_mask, completion, mask


def install_recorded_head(monkeypatch):
    original = losses.per_token_logps
    seen = []

    def recorded(*args, backend="jax", **kwargs):
        seen.append(backend)
        # CPU 只核对路由；算术仍由原生实现执行，不能授予 Pallas 数值资格。
        return original(*args, backend="jax", **kwargs)

    monkeypatch.setattr(losses, "per_token_logps", recorded)
    return seen


@pytest.mark.parametrize("size", [None, 1])
def test_completion_scoring_preserves_backend_through_microbatches(monkeypatch, grad_tiny, size):
    _, params, config = grad_tiny
    batch = completion_batch()
    expected = losses.trainer_completion_logps(
        params,
        *batch,
        config=config,
        compute_dtype=jnp.float32,
        vocab_chunk=17,
        sequence_chunk=5,
        microbatch_size=size,
    )
    seen = install_recorded_head(monkeypatch)
    actual = losses.trainer_completion_logps(
        params,
        *batch,
        config=config,
        compute_dtype=jnp.float32,
        vocab_chunk=17,
        sequence_chunk=5,
        microbatch_size=size,
        logprob_backend="pallas",
    )
    assert seen and set(seen) == {"pallas"}
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.asarray(actual)[~np.asarray(batch[3])], 0.0)


@pytest.mark.parametrize("size", [None, 1])
def test_grpo_update_routes_backend_and_freezes_joint_old(monkeypatch, grad_tiny, size):
    _, params, config = grad_tiny
    batch = completion_batch()
    advantages = jnp.asarray([1.0, -0.5], jnp.float32)
    optimizer, trainable = losses.make_optimizer(params, learning_rate=1e-3)
    initial = losses.init_train_state(params, optimizer)
    reference = jnp.full(batch[3].shape, -3.0)
    old = jnp.zeros_like(reference)
    seen = install_recorded_head(monkeypatch)

    def update(state, old, capture, backend):
        return losses.grpo_train_step(
            state,
            *batch,
            advantages,
            old,
            reference,
            config=config,
            optimizer=optimizer,
            trainable_mask=trainable,
            compute_dtype=jnp.float32,
            vocab_chunk=17,
            sequence_chunk=5,
            microbatch_size=size,
            beta=0.04,
            logprob_backend=backend,
            use_current_policy_as_old=capture,
        )

    baseline = jax.jit(lambda state, old, capture: update(state, old, capture, "jax"))
    routed = jax.jit(lambda state, old, capture: update(state, old, capture, "pallas"))
    native, native_metrics = baseline(initial, old, jnp.asarray(True))
    seen.clear()
    actual, metrics = routed(initial, old, jnp.asarray(True))
    assert seen and set(seen) == {"pallas"}
    assert float(metrics.grad_norm) > 0
    assert float(metrics.loss_metrics.ratio_min) == float(metrics.loss_metrics.ratio_max) == 1.0
    for a, b in zip(jax.tree.leaves((actual, metrics)), jax.tree.leaves((native, native_metrics)), strict=True):
        np.testing.assert_array_equal(a, b)
    assert metrics.policy_logps is not None
    frozen_old = metrics.policy_logps
    native, native_metrics = baseline(native, frozen_old, jnp.asarray(False))
    actual, metrics = routed(actual, frozen_old, jnp.asarray(False))
    for a, b in zip(jax.tree.leaves((actual, metrics)), jax.tree.leaves((native, native_metrics)), strict=True):
        np.testing.assert_array_equal(a, b)
    assert int(actual.step) == 2
    assert (float(metrics.loss_metrics.ratio_min), float(metrics.loss_metrics.ratio_max)) != (1.0, 1.0)
    assert not np.array_equal(actual.params_f32.final_norm.weight, params.final_norm.weight)


@pytest.mark.parametrize(
    "extra",
    [
        ["--sequence-chunk", "128"],
        ["--vocab-chunk", "4096"],
        ["--rollout-backend", "inference"],
        ["--rollout-backend", "inference-process"],
        ["--lora-rank", "8"],
        ["--training-device-ids", "0", "1"],
    ],
)
def test_grpo_cli_rejects_unsupported_pallas_configuration(monkeypatch, extra):
    monkeypatch.setattr(
        sys, "argv", ["train_grpo.py", "--output-dir", "/tmp/unused", "--logprob-backend", "pallas", *extra]
    )
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize("backend", ["jax", "pallas"])
def test_grpo_cli_backend_is_explicit(monkeypatch, backend):
    monkeypatch.setattr(sys, "argv", ["train_grpo.py", "--output-dir", "/tmp/unused", "--logprob-backend", backend])
    assert parse_args().logprob_backend == backend


@pytest.mark.parametrize("backend", ["jax", "pallas"])
def test_grpo_resume_preserves_backend_and_protocol(backend):
    config = {"beta": 0.0, "kl_estimator": "none", "kl_clamp_value": None, **logprob_run_metadata(backend)}
    validate_resume_config(dict(config), config)


@pytest.mark.parametrize("saved_protocol", [None, "older-pallas-implementation"])
def test_grpo_resume_rejects_changed_pallas_protocol(saved_protocol):
    base = {"beta": 0.0, "kl_estimator": "none", "kl_clamp_value": None}
    saved = {**base, "logprob_backend": "pallas"}
    if saved_protocol is not None:
        saved["logprob_protocol"] = saved_protocol
    with pytest.raises(ValueError, match="logprob_protocol"):
        validate_resume_config(saved, {**base, **logprob_run_metadata("pallas")})


def test_grpo_resume_cannot_switch_from_native_to_pallas():
    saved = {"beta": 0.0, "kl_estimator": "none", "kl_clamp_value": None}
    with pytest.raises(ValueError, match="logprob_backend"):
        validate_resume_config(saved, {**saved, **logprob_run_metadata("pallas")})
