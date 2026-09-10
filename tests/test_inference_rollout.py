"""独立引擎边界的随机流、状态身份、mask和概率负控制。"""

import sys
from types import ModuleType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gemma4_posttrain_jax.inference_rollout import InferenceRollout, token_prompts
from gemma4_posttrain_jax.sampler import SamplerConfig


def test_prompt_mask_preserves_real_tokens_and_rejects_holes():
    ids = np.array([[0, 0, 5, 6], [0, 1, 2, 3]])
    assert token_prompts(ids, ids != 0) == [[5, 6], [1, 2, 3]]
    for mask in (np.zeros_like(ids), np.array([[0, 1, 0, 1], [0, 1, 1, 1]]), np.ones_like(ids) * 2):
        with pytest.raises(ValueError):
            token_prompts(ids, mask)


@pytest.fixture
def adapter(monkeypatch):
    module = ModuleType("gemma4_posttrain_jax.inference_runtime")
    module.EngineSampling = SimpleNamespace
    monkeypatch.setitem(sys.modules, module.__name__, module)
    runtime = SimpleNamespace(result_override=None)

    def generate(prompts, batch_rng, sampling, expected_version):
        rows = [
            SimpleNamespace(
                prompt_token_ids=prompts[0],
                token_ids=[4, 2],
                raw_logprobs=[-1.0, -2.0],
                proposal_logprobs=[-0.5, -1.5],
                finish_reason="stop",
                stop_reason=2,
            ),
            SimpleNamespace(
                prompt_token_ids=prompts[1],
                token_ids=[2],
                raw_logprobs=[-3.0],
                proposal_logprobs=[-2.5],
                finish_reason="stop",
                stop_reason=2,
            ),
        ]
        result = SimpleNamespace(
            rows=rows,
            policy_version=expected_version,
            metadata={"batch_rng_install": {"installed": batch_rng.to_dict()}},
        )
        if runtime.result_override:
            runtime.result_override(result)
        return result

    runtime.generate = generate
    value = InferenceRollout(
        runtime,
        SimpleNamespace(pad_token_id=0, vocab_size=16),
        sampler_config=SamplerConfig(3, 3, temperature=0.7, top_k=8, eos_ids=(2,)),
    )
    value.policy_version = 5
    return value


def call(adapter):
    return adapter.generate(
        jnp.array([[0, 3, 4], [0, 5, 6]], dtype=jnp.int32),
        jnp.array([[False, True, True], [False, True, True]]),
        key=jax.random.PRNGKey(7),
    )


def test_preserves_eos_policy_and_proposal_separately(adapter):
    batch = call(adapter)
    np.testing.assert_array_equal(batch.completion_mask, [[True, True, False], [True, False, False]])
    np.testing.assert_array_equal(batch.completion_ids, [[4, 2, 0], [2, 0, 0]])
    np.testing.assert_array_equal(batch.rollout_logps, [[-1, -2, 0], [-3, 0, 0]])
    np.testing.assert_array_equal(adapter.last_proposal_logps, [[-0.5, -1.5, 0], [-2.5, 0, 0]])
    assert adapter.last_generation["policy_version"] == 5
    rows = adapter.last_generation["rows"]
    assert rows[0]["prompt_token_ids"] == [3, 4]
    assert rows[0]["token_ids"] == [4, 2]
    assert rows[0]["raw_logprobs"] == [-1.0, -2.0]
    assert rows[0]["proposal_logprobs"] == [-0.5, -1.5]
    assert rows[1]["token_ids"] == [2] and rows[1]["stop_reason"] == 2


@pytest.mark.parametrize(
    "failure", ["version", "batch_rng", "post_eos", "nan", "positive_logp", "prompt", "early_stop", "bad_reason"]
)
def test_rejects_wrong_identity_and_invalid_output(adapter, failure):
    def change(result):
        row = result.rows[0]
        if failure == "version":
            result.policy_version = 4
        elif failure == "batch_rng":
            result.metadata["batch_rng_install"]["installed"]["words"][0] ^= 1
        elif failure == "post_eos":
            row.token_ids = [2, 4]
        elif failure == "nan":
            row.raw_logprobs[0] = float("nan")
        elif failure == "positive_logp":
            row.raw_logprobs[0] = 0.5
        elif failure == "early_stop":
            row.token_ids = [4, 5]
            row.finish_reason = "length"
            row.stop_reason = None
        elif failure == "bad_reason":
            row.stop_reason = 3
        else:
            row.prompt_token_ids = [8, 9]

    adapter.runtime.result_override = change
    with pytest.raises(RuntimeError):
        call(adapter)


def test_unsynchronized_weights_cannot_generate(adapter):
    adapter.policy_version = None
    with pytest.raises(RuntimeError, match="同步"):
        call(adapter)


def test_top_p_is_rejected_before_engine_can_ignore_it(adapter):
    adapter.sampler_config = adapter.sampler_config._replace(top_p=0.9)
    with pytest.raises(ValueError, match="Top-p"):
        adapter.generate(
            jnp.ones((2, 3), dtype=jnp.int32), jnp.ones((2, 3), dtype=jnp.bool_), key=jax.random.PRNGKey(0)
        )


def test_full_policy_sampling_cannot_use_a_different_proposal(adapter):
    adapter.sampler_config = adapter.sampler_config._replace(temperature=1.0, top_k=0)
    with pytest.raises(RuntimeError, match="proposal"):
        call(adapter)


def test_full_policy_proposal_matches_raw(adapter):
    adapter.sampler_config = adapter.sampler_config._replace(temperature=1.0, top_k=0)

    def equalize(result):
        for row in result.rows:
            row.proposal_logprobs = list(row.raw_logprobs)

    adapter.runtime.result_override = equalize
    batch = call(adapter)
    np.testing.assert_array_equal(batch.rollout_logps, adapter.last_proposal_logps)


def test_partial_weight_failure_invalidates_adapter_version(adapter, monkeypatch):
    module = ModuleType("gemma4_posttrain_jax.inference_weights")
    module.apply_gemma4_params = lambda *args: None
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def failed_update(params, config, version):
        raise RuntimeError("transport failed")

    adapter.runtime.sync_params = failed_update
    with pytest.raises(RuntimeError, match="transport"):
        adapter.update_params(None, version=6)
    assert adapter.policy_version is None
    with pytest.raises(RuntimeError, match="同步"):
        call(adapter)


@pytest.mark.parametrize(
    "report", [{"complete": False, "version": 6}, {"complete": True, "version": 5}, {"complete": True, "version": True}]
)
def test_adapter_requires_complete_parameter_commit(adapter, report):
    adapter.runtime.sync_params = lambda params, config, version: report
    with pytest.raises(RuntimeError, match="版本提交"):
        adapter.update_params(None, version=6)
    assert adapter.policy_version is None and adapter.last_update is None


def test_common_parameter_interface_commits_version_and_keeps_receipt(adapter):
    calls = []
    receipt = {"complete": True, "version": 6}

    def sync(params, config, version):
        calls.append((params, config, version))
        return receipt

    adapter.runtime.sync_params = sync
    params = object()
    adapter.update_params(params, version=6)
    assert calls == [(params, adapter.config, 6)]
    assert adapter.policy_version == 6 and adapter.last_update is receipt
