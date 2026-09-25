"""检查后端选择、旧检查点兼容与显式配置错误。"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gemma4_posttrain_jax import losses
from gemma4_posttrain_jax.logprob_protocol import PALLAS_LOGPROB_PROTOCOL, logprob_run_metadata
from gemma4_posttrain_jax.losses import per_token_logps
from scripts.train_sft import parse_args, validate_resume_config


def test_legacy_checkpoint_keeps_native_backend():
    saved = {"mode": "gsm8k", "seed": 7, "model_path": "fixture"}
    validate_resume_config(saved, {**saved, "logprob_backend": "jax"})
    with pytest.raises(ValueError, match="logprob_backend"):
        validate_resume_config(saved, {**saved, "logprob_backend": "pallas"})


def test_explicit_backend_cannot_change_on_resume():
    saved = {"mode": "overfit", "logprob_backend": "pallas"}
    validate_resume_config(saved, dict(saved))
    with pytest.raises(ValueError, match="logprob_backend"):
        validate_resume_config(saved, {**saved, "logprob_backend": "jax"})


@pytest.mark.parametrize(
    "extra",
    [("--compute-dtype", "f32"), ("--no-vocab-parallel",), ("--sequence-chunk", "128"), ("--vocab-chunk", "4096")],
)
def test_cli_rejects_unsupported_pallas_config(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["train_sft.py", "--logprob-backend", "pallas", *extra])
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2


def test_cli_backend_is_explicit_and_defaults_to_jax(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_sft.py"])
    assert parse_args().logprob_backend == "jax"
    monkeypatch.setattr(sys, "argv", ["train_sft.py", "--logprob-backend", "pallas"])
    assert parse_args().logprob_backend == "pallas"


@pytest.mark.parametrize("backend,message", [("unknown", "unknown logprob"), ("pallas", "four-chip")])
def test_invalid_backend_never_silently_falls_back(backend, message):
    w = jnp.ones((8, 4), jnp.bfloat16)
    h = jnp.ones((2, 4), jnp.bfloat16)
    targets = jnp.zeros((2,), jnp.int32)
    with pytest.raises(ValueError, match=message):
        per_token_logps(w, h, targets, softcap=30.0, backend=backend)


def test_versioned_pallas_checkpoint_resumes_same_implementation():
    saved = {"mode": "overfit-one-batch", **logprob_run_metadata("pallas")}
    validate_resume_config(saved, dict(saved))


@pytest.mark.parametrize("saved_protocol", [None, "older-pallas-implementation"])
def test_pallas_checkpoint_cannot_silently_change_implementation(saved_protocol):
    saved = {"mode": "overfit-one-batch", "logprob_backend": "pallas"}
    if saved_protocol is not None:
        saved["logprob_protocol"] = saved_protocol
    current = {"mode": "overfit-one-batch", **logprob_run_metadata("pallas")}
    with pytest.raises(ValueError, match="logprob_protocol"):
        validate_resume_config(saved, current)


def test_native_grpo_metadata_keeps_legacy_checkpoint_identity():
    saved = {"mode": "native", "compute_dtype": "bfloat16"}
    assert {**saved, **logprob_run_metadata("jax")} == saved
    candidate = {**saved, **logprob_run_metadata("pallas")}
    assert candidate["logprob_protocol"] == PALLAS_LOGPROB_PROTOCOL
    assert candidate != saved


def test_unknown_protocol_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown logprob"):
        logprob_run_metadata("unknown")


def test_sft_update_preserves_backend_selection(monkeypatch, grad_tiny):
    _, params, config = grad_tiny
    tokens = jnp.asarray([[2, 13, 9, 41, 8], [2, 5, 16, 12, 6]], jnp.int32)
    labels = tokens.at[:, :2].set(-100)
    mask = jnp.ones_like(tokens, jnp.bool_)
    optimizer, trainable = losses.make_optimizer(params, learning_rate=1e-3)
    initial = losses.init_train_state(params, optimizer)
    seen = []

    def recorded(*args, backend="jax", **kwargs):
        seen.append(backend)
        # CPU 验证完整 SFT 更新中的后端传递；不执行 TPU kernel。
        return per_token_logps(*args, backend="jax", **kwargs)

    monkeypatch.setattr(losses, "per_token_logps", recorded)

    def run(backend):
        return jax.jit(
            lambda state: losses.train_step(
                state,
                tokens,
                labels,
                mask,
                config=config,
                optimizer=optimizer,
                trainable_mask=trainable,
                compute_dtype=jnp.float32,
                vocab_chunk=17,
                sequence_chunk=5,
                logprob_backend=backend,
            )
        )(initial)

    reference = run("jax")
    seen.clear()
    actual = run("pallas")
    assert seen and set(seen) == {"pallas"}
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(reference), strict=True):
        np.testing.assert_array_equal(a, b)
    assert int(actual[0].step) == 1
