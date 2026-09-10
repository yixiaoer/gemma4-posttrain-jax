"""真实tiny模型的一次Adam滞后、行为概率、TIS与包含快照的原子恢复。"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import load_file
from test_grpo_host_loop import TinyDataset, TinyTokenizer

from gemma4_posttrain_jax.losses import TrainState
from gemma4_posttrain_jax.model import Gemma4TextParams
from gemma4_posttrain_jax.rewards import RewardOutput


def run_lagged_host_case(
    model_fixture,
    tmp_path: Path,
    monkeypatch,
    mu: int,
    is_cap: float | None,
    dynamic: bool,
    *,
    tpu: bool = False,
    prompt_style: str = "chat",
) -> None:
    assert jax.default_backend() == ("tpu" if tpu else "cpu")
    _, params, config = model_fixture
    path = Path(__file__).resolve().parents[1] / "scripts/train_grpo.py"
    spec = importlib.util.spec_from_file_location("lagged_grpo_host_test", path)
    assert spec is not None and spec.loader is not None
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    dataset = TinyDataset({"question": f"question {i}", "answer": "work\n#### 1"} for i in range(8))
    monkeypatch.setattr(trainer, "load_gsm8k", lambda split: dataset)
    monkeypatch.setattr(
        trainer,
        "load_hf_params",
        lambda *args, **kwargs: (jax.tree.map(lambda x: jnp.array(x, copy=True), params), config),
    )
    monkeypatch.setattr(trainer, "load_hf_eos_token_ids", lambda *args, **kwargs: (1,))
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: TinyTokenizer())

    def rewards(completions, golds, **kwargs):
        success = np.asarray([0, 0, 0, 1], np.float32) if dynamic else np.arange(len(completions), dtype=np.float32) % 2
        zeros = np.zeros_like(success)
        return RewardOutput(success, success, zeros, zeros, success)

    monkeypatch.setattr(trainer, "score_completions", rewards)
    original_call = trainer.timed_call
    update_params = {}
    generation_params = []
    phase = "continuous"

    def record_call(fn, *args):
        if args and isinstance(args[0], TrainState):
            step = int(args[0].step)
            update_params[phase, step] = jax.tree.map(lambda value: np.asarray(value).copy(), args[0].params_f32)
        if len(args) == 4 and isinstance(args[0], Gemma4TextParams):
            generation_params.append((phase, jax.tree.map(lambda value: np.asarray(value).copy(), args[0])))
        return original_call(fn, *args)

    monkeypatch.setattr(trainer, "timed_call", record_call)

    def run(name: str, resume: Path | None = None):
        nonlocal phase
        phase = name
        output, checkpoints = tmp_path / name, tmp_path / f"state_{name}"
        argv = [
            "train_grpo.py",
            "--model-path",
            str(tmp_path / "model"),
            "--output-dir",
            str(output),
            "--prompt-batch-size",
            "2",
            "--group-size",
            "4" if tpu else "2",
            "--max-prompt-len",
            "4",
            "--max-new-tokens",
            "4",
            "--microbatch-size",
            "4" if tpu else "2",
            "--max-steps",
            "4",
            "--updates-per-rollout",
            str(mu),
            "--learning-rate",
            "0.001",
            "--save-every",
            "2",
            "--checkpoint-dir",
            str(checkpoints),
            "--vocab-chunk",
            "17",
            "--sequence-chunk",
            "5",
            "--reward-workers",
            "1",
            "--rollout-lag-updates",
            "1",
        ]
        if prompt_style == "plain":
            argv += ["--prompt-style", "plain"]
        if dynamic:
            argv += [
                "--dynamic-sampling",
                "--max-sampling-attempts",
                "2",
                "--dev-size",
                "2",
                "--dev-seed",
                "19",
                "--eval-split",
                "train-dev",
                "--eval-size",
                "2",
                "--eval-every",
                "2",
                "--eval-batch-size",
                "4",
                "--eval-max-new-tokens",
                "4",
            ]
        if is_cap is not None:
            argv += ["--sampler-is-cap", str(is_cap)]
        if resume is not None:
            argv += ["--resume", str(resume)]
        monkeypatch.setattr(sys, "argv", argv)
        trainer.main()
        return output, checkpoints

    continuous, continuous_states = run("continuous")
    generated = [value for name, value in generation_params if name == "continuous"]
    attempts = 2 if dynamic else 1
    assert len(generated) == (4 // mu) * attempts
    for index, value in enumerate(generated):
        version = max(0, (index // attempts) * mu - 1)
        expected = update_params["continuous", version]
        for actual, before in zip(jax.tree.leaves(value), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(actual, before.astype(jnp.bfloat16))
    stale_version = max(0, mu - 1)
    assert any(
        not np.array_equal(old.astype(jnp.bfloat16), current.astype(jnp.bfloat16))
        for old, current in zip(
            jax.tree.leaves(update_params["continuous", stale_version]),
            jax.tree.leaves(update_params["continuous", mu]),
            strict=True,
        )
    )
    with (continuous / "metrics.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert [int(row["behavior_policy_step"]) for row in rows] == [max(0, (step // mu) * mu - 1) for step in range(4)]
    assert [int(row["target_old_policy_step"]) for row in rows] == [(step // mu) * mu for step in range(4)]
    assert [int(row["current_policy_step"]) for row in rows] == list(range(4))
    assert all(float(row["behavior_snapshot_s"]) > 0 for row in rows[mu - 1 :: mu])
    resumed, resumed_states = run("resumed", continuous_states / "step_00000002")
    expected = load_file(continuous_states / "step_00000004/state.safetensors")
    observed = load_file(resumed_states / "step_00000004/state.safetensors")
    assert expected.keys() == observed.keys()
    assert any("behavior" in key for key in expected)
    for key in expected:
        np.testing.assert_array_equal(observed[key], expected[key], err_msg=key)
    before = [
        json.loads(line)
        for line in (continuous / "rollouts.jsonl").read_text().splitlines()
        if json.loads(line)["step"] >= 3
    ]
    after = [json.loads(line) for line in (resumed / "rollouts.jsonl").read_text().splitlines()]
    assert before == after
    exported = np.load(resumed / "parity_batch.npz")
    assert int(exported["behavior_policy_step"]) == 1
    assert int(exported["target_old_policy_step"]) == 2
    assert (resumed / "startup_target_drift.json").exists()
    assert np.all(exported["prompt_ids"][:, : (2 if prompt_style == "plain" else 1)] == 0)
    if is_cap is not None:
        expected_is = np.minimum(np.exp(exported["old_logps"] - exported["rollout_logps"]), is_cap)
        expected_is = np.where(exported["completion_mask"], expected_is, 0)
        np.testing.assert_allclose(exported["sampler_is_weights"], expected_is, atol=1e-6, rtol=1e-6)
    # H2在同一行为版本比较；当前target的跨版本变化单独保存。
    drift = json.loads((resumed / "startup_drift.json").read_text())
    assert drift["delta_definition"] == "standalone_behavior_trainer_logp - behavior_rollout_full_policy_logp"

    if dynamic:
        # 独立评估入口读取带滞后快照的checkpoint，但用当前训练参数做greedy评估。
        from gemma4_posttrain_jax.evaluation import EvaluationData

        eval_path = path.parent / "evaluate_gsm8k.py"
        eval_spec = importlib.util.spec_from_file_location("lagged_grpo_eval_test", eval_path)
        assert eval_spec is not None and eval_spec.loader is not None
        monkeypatch.syspath_prepend(str(path.parent))
        evaluator = importlib.util.module_from_spec(eval_spec)
        eval_spec.loader.exec_module(evaluator)
        monkeypatch.setattr(evaluator, "load_hf_params", trainer.load_hf_params)
        monkeypatch.setattr(evaluator, "load_hf_eos_token_ids", trainer.load_hf_eos_token_ids)
        indices = json.loads((continuous / "data_split.json").read_text())["dev_indices"]
        monkeypatch.setattr(
            evaluator,
            "load_evaluation_data",
            lambda *args, **kwargs: EvaluationData(dataset, indices, {}, "train", "train-dev"),
        )
        evaluated = tmp_path / "independent_eval"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "evaluate_gsm8k.py",
                "--checkpoint",
                str(continuous_states / "step_00000004"),
                "--output-dir",
                str(evaluated),
                "--size",
                "2",
                "--batch-size",
                "4",
                "--max-prompt-len",
                "4",
                "--max-new-tokens",
                "4",
                "--subset",
                "train-dev",
                "--dev-size",
                "2",
                "--dev-seed",
                "19",
                "--reward-workers",
                "1",
            ],
        )
        evaluator.main()
        assert json.loads((evaluated / "meta.json").read_text())["arguments"]["prompt_style"] == prompt_style
        invalid_args = sys.argv.copy()
        invalid_args[invalid_args.index("--output-dir") + 1] = str(tmp_path / "invalid_eval_style")
        invalid_args += ["--prompt-style", "chat" if prompt_style == "plain" else "plain"]
        monkeypatch.setattr(sys, "argv", invalid_args)
        with pytest.raises(ValueError, match="prompt style"):
            evaluator.main()
        assert (evaluated / "evaluation/predictions.jsonl").read_bytes() == (
            continuous / "eval/step_00000004/predictions.jsonl"
        ).read_bytes()

    if is_cap is not None:
        metadata_reader = trainer.read_checkpoint_metadata

        def read_previous_tis_protocol(path):
            metadata = metadata_reader(path)
            del metadata["metadata"]["run_config"]["sampler_is_protocol"]
            return metadata

        monkeypatch.setattr(trainer, "read_checkpoint_metadata", read_previous_tis_protocol)
        with pytest.raises(ValueError, match="恢复配置"):
            run("legacy_tis_protocol", continuous_states / "step_00000002")


@pytest.mark.parametrize(
    "mu,is_cap,dynamic",
    [(1, None, False), (1, 1.5, False), (2, None, False), (2, 1.5, False), (2, 1.5, True)],
)
def test_lagged_rollout_uses_previous_update_and_restores_complete_state(
    grad_tiny, tmp_path: Path, monkeypatch, mu: int, is_cap: float | None, dynamic: bool
) -> None:
    run_lagged_host_case(grad_tiny, tmp_path, monkeypatch, mu, is_cap, dynamic)


@pytest.mark.tpu
@pytest.mark.parametrize("mu", [1, 2])
def test_tpu_lagged_rollout_restores_host_snapshot_and_four_shard_training(
    tpu_grad_tiny, tmp_path: Path, monkeypatch, mu: int
) -> None:
    assert len(jax.devices()) == 4
    run_lagged_host_case(tpu_grad_tiny, tmp_path, monkeypatch, mu, 1.5, False, tpu=True)


def test_plain_lagged_resume_and_independent_eval(grad_tiny, tmp_path: Path, monkeypatch) -> None:
    def forbidden_chat(*args, **kwargs):
        raise AssertionError("plain路径不应调用chat template")

    monkeypatch.setattr(TinyTokenizer, "apply_chat_template", forbidden_chat)
    run_lagged_host_case(grad_tiny, tmp_path, monkeypatch, 2, 1.5, True, prompt_style="plain")
