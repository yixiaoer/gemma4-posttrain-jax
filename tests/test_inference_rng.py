"""真实CPU typed key运输、请求契约与生成失败生命周期回归。"""

import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gemma4_posttrain_jax.inference_rng import BatchRngState, install_runner_rng, read_runner_rng  # noqa: E402
from gemma4_posttrain_jax.inference_runtime import EngineConfig, EngineRuntime, EngineSampling  # noqa: E402


def placed_runner():
    devices = jax.devices("cpu")
    if len(devices) != 2:
        pytest.skip("本资格需要XLA_FLAGS=--xla_force_host_platform_device_count=2；完整控制器必须无skip通过")
    sharding = NamedSharding(Mesh(np.asarray(devices), ("data",)), PartitionSpec())
    return SimpleNamespace(rng_params_for_sampling=jax.device_put(jax.random.key(0), sharding))


def test_schema_rejects_ambiguous_or_truncated_checkpoint():
    state = BatchRngState("threefry2x32", (0, 2**32 - 1))
    assert BatchRngState.from_dict(state.to_dict()) == state
    for words in ((True, 1), (-1, 1), (0, 2**32), (1,), [0, 1], (0.0, 1)):
        with pytest.raises(ValueError):
            BatchRngState("threefry2x32", words)
    for value in (
        {},
        state.to_dict() | {"request_seed": 0},
        state.to_dict() | {"words": [1]},
        state.to_dict() | {"implementation": "rbg"},
    ):
        with pytest.raises(ValueError):
            BatchRngState.from_dict(value)


def test_actual_typed_key_transport_and_layout_guards():
    runner = placed_runner()
    wanted = BatchRngState("threefry2x32", (123, 456))
    report = install_runner_rng(runner, wanted, (0, 1))
    assert read_runner_rng(runner, (0, 1))[0] == wanted
    assert report["placement"]["mesh_device_ids"] == [0, 1]
    assert report["placement"]["fully_replicated"]
    with pytest.raises(ValueError, match="mesh"):
        read_runner_rng(runner, (0,))
    with pytest.raises(TypeError):
        install_runner_rng(runner, [0, 1], (0, 1))
    runner.rng_params_for_sampling = jax.random.PRNGKey(0)
    with pytest.raises(ValueError, match="typed"):
        read_runner_rng(runner, (0, 1))


def runtime_fixture(*, fail=False, advance=True):
    runtime = EngineRuntime.__new__(EngineRuntime)
    runtime.config = EngineConfig(model_path="fixture", device_indexes=(0, 1))
    runtime._closed = runtime._broken = False
    runtime._pending = None
    runtime.policy_version = 7
    runtime.metadata = {}
    runtime.jax = jax
    runtime.runner = placed_runner()
    runtime.SamplingParams = SimpleNamespace
    runtime._ensure_quiescent = lambda: {"zero_token_cleanup_steps": 0}
    calls = []

    def generate(requests, params, *, use_tqdm):
        calls.append(params)
        if advance:
            runtime.runner.rng_params_for_sampling, _ = jax.random.split(runtime.runner.rng_params_for_sampling)
        if fail:
            raise RuntimeError("在采样状态推进后模拟生成失败")
        return [
            SimpleNamespace(
                prompt_token_ids=row["prompt_token_ids"],
                outputs=[
                    SimpleNamespace(
                        token_ids=[42],
                        logprobs=[{42: SimpleNamespace(logprob=-0.5, rank=1)}],
                        finish_reason="length",
                        stop_reason=None,
                    )
                ],
            )
            for row in requests
        ]

    runtime.llm = SimpleNamespace(generate=generate)
    return runtime, calls


def test_generate_passes_no_request_seed_and_records_real_progress():
    runtime, calls = runtime_fixture()
    state = BatchRngState("threefry2x32", (123, 456))
    result = runtime.generate([[2, 3]], state, EngineSampling(max_tokens=1), expected_version=7)
    assert all(params.seed is None for params in calls[0])
    assert result.rows[0].raw_logprobs == result.rows[0].proposal_logprobs == [-0.5]
    assert not hasattr(result.rows[0], "request_seed")
    assert result.metadata["batch_rng_install"]["installed"] == state.to_dict()
    assert result.metadata["batch_rng_after"] != state.to_dict()
    repeated = runtime.generate([[2, 3]], state, EngineSampling(max_tokens=1), expected_version=7)
    assert repeated.rows == result.rows
    assert repeated.metadata["batch_rng_after"] == result.metadata["batch_rng_after"]


def test_generation_failure_or_no_rng_progress_invalidates_backend():
    for fail, advance in ((True, True), (False, False)):
        runtime, _ = runtime_fixture(fail=fail, advance=advance)
        state = BatchRngState("threefry2x32", (123, 456))
        with pytest.raises(RuntimeError):
            runtime.generate([[2, 3]], state, EngineSampling(max_tokens=1))
        assert runtime._broken
        with pytest.raises(RuntimeError, match="关闭或此前失败"):
            runtime.generate([[2, 3]], state, EngineSampling(max_tokens=1))
