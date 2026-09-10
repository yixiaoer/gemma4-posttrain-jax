"""在真实tiny模型上验证host的μ更新、固定old、RNG/数据游标和完整恢复。"""

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

from gemma4_posttrain_jax.algorithms import validate_update_schedule
from gemma4_posttrain_jax.losses import TrainState
from gemma4_posttrain_jax.rewards import RewardOutput


@pytest.mark.parametrize(
    "field,value",
    [("updates_per_rollout", 0), ("max_steps", 3), ("eval_every", 1), ("save_every", 3), ("start_step", 1)],
)
def test_schedule_rejects_partial_rollout_boundaries(field: str, value: int) -> None:
    arguments = dict(updates_per_rollout=2, max_steps=4, eval_every=2, save_every=2, start_step=0)
    arguments[field] = value
    with pytest.raises(ValueError):
        validate_update_schedule(**arguments)


class TinyDataset(list):
    _fingerprint = "fixed-tiny-host-loop"


class TinyTokenizer:
    bos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
        assert tokenize and add_generation_prompt
        token = int(messages[0]["content"].splitlines()[0].split()[-1]) + 3
        return {"input_ids": [[2, token, 5]]}

    def __call__(self, text: str, *, add_special_tokens: bool):
        token = int(text.splitlines()[0].split()[-1]) + 3
        return {"input_ids": ([2] if add_special_tokens else []) + [token]}

    def batch_decode(self, rows, *, skip_special_tokens: bool):
        return [" ".join(map(str, np.asarray(row).tolist())) for row in rows]


@pytest.mark.parametrize(
    "dynamic,attempt_budget,is_cap,prompt_style,rollout_layout,lora",
    [
        (False, 2, None, "chat", "replicated", False),
        (False, 2, 1.5, "chat", "replicated", False),
        (True, 2, None, "chat", "replicated", False),
        (True, 2, 1.5, "chat", "replicated", False),
        (True, 1, None, "chat", "replicated", False),
        (True, 2, 1.5, "plain", "replicated", False),
        (True, 2, None, "chat", "fsdp", False),
        pytest.param(False, 2, None, "chat", "fsdp", True, id="lora-sync"),
        pytest.param(True, 2, 1.5, "chat", "fsdp", True, id="lora-dynamic-is"),
    ],
)
def test_mu_two_host_loop_resume_preserves_old_batch_and_full_state(
    grad_tiny,
    tpu_grad_tiny,
    tmp_path,
    monkeypatch,
    dynamic: bool,
    attempt_budget: int,
    is_cap: float | None,
    prompt_style: str,
    rollout_layout: str,
    lora: bool,
) -> None:
    assert jax.default_backend() == "cpu"
    hf, params, config = tpu_grad_tiny if len(jax.devices()) > 1 else grad_tiny
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_grpo.py"
    spec = importlib.util.spec_from_file_location("grpo_host_test", path)
    assert spec is not None and spec.loader is not None
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    dataset = TinyDataset({"question": f"question {i}", "answer": "work\n#### 1"} for i in range(8))
    monkeypatch.setattr(trainer, "load_gsm8k", lambda split: dataset)
    monkeypatch.setattr(
        trainer,
        "load_hf_params",
        lambda *args, **kwargs: (
            jax.tree.map(lambda x: jnp.array(x, dtype=kwargs.get("dtype", x.dtype), copy=True), params),
            config,
        ),
    )
    monkeypatch.setattr(trainer, "load_hf_eos_token_ids", lambda *args, **kwargs: (1,))
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: TinyTokenizer())

    def rewards(completions, golds, **kwargs):
        # 仅替换评分器，完整生成、logprob、梯度、Adam、保存/恢复使用产品实现。
        success = np.asarray([0, 0, 0, 1], np.float32) if dynamic else np.arange(len(completions), dtype=np.float32) % 2
        zeros = np.zeros_like(success)
        return RewardOutput(success, success, zeros, zeros, success)

    monkeypatch.setattr(trainer, "score_completions", rewards)
    original_timed_call = trainer.timed_call
    old_snapshots = []
    capture_flags = []

    def record_call(fn, *arguments):
        is_update = bool(arguments) and isinstance(arguments[0], TrainState)
        # donation会释放旧state；先记录step和传入的冻结数组。
        before_step = int(arguments[0].step) if is_update else None
        frozen = np.asarray(arguments[6]).copy() if is_update else None
        captured = bool(arguments[-2] if lora else arguments[-1]) if is_update else False
        result = original_timed_call(fn, *arguments)
        if is_update:
            actual_old = np.asarray(result[0][1].policy_logps).copy() if captured else frozen
            old_snapshots.append((before_step, actual_old))
            capture_flags.append(captured)
        return result

    monkeypatch.setattr(trainer, "timed_call", record_call)
    model_path = tmp_path / "model"
    if lora:
        hf.save_pretrained(model_path)

    def run(name, resume=None, style=None, layout=None):
        checkpoint_dir = tmp_path / f"checkpoints_{name}"
        output = tmp_path / name
        command = [
            "train_grpo.py",
            "--model-path",
            str(model_path),
            "--output-dir",
            str(output),
            "--prompt-batch-size",
            "2",
            "--group-size",
            "2",
            "--max-prompt-len",
            "4",
            "--max-new-tokens",
            "4",
            "--microbatch-size",
            str(max(2, len(jax.devices()))),
            "--max-steps",
            "4",
            "--updates-per-rollout",
            "2",
            "--learning-rate",
            "0.001",
            "--save-every",
            "2",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--vocab-chunk",
            "17",
            "--sequence-chunk",
            "5",
            "--reward-workers",
            "1",
        ]
        if (layout or rollout_layout) != "replicated":
            command += ["--rollout-layout", layout or rollout_layout]
        if (style or prompt_style) != "chat":
            command += ["--prompt-style", style or prompt_style]
        if is_cap is not None:
            command += ["--sampler-is-cap", str(is_cap)]
        if lora:
            command += ["--lora-rank", "2", "--lora-alpha", "3"]
        if dynamic:
            command += ["--dynamic-sampling", "--max-sampling-attempts", str(attempt_budget)]
            command += ["--loss-aggregation", "token-mean"]
            command += [
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
        if resume is not None:
            command += ["--resume", str(resume)]
        monkeypatch.setattr(sys, "argv", command)
        trainer.main()
        return output, checkpoint_dir

    if dynamic and attempt_budget == 1:
        with pytest.raises(RuntimeError, match="未更新参数"):
            run("exhausted")
        assert not old_snapshots
        assert not list((tmp_path / "checkpoints_exhausted").iterdir())
        raw = [json.loads(line) for line in (tmp_path / "exhausted" / "rollouts.jsonl").read_text().splitlines()]
        assert len(raw) == 4
        assert sum(row["selected_for_update"] for row in raw) == 2
        assert (tmp_path / "exhausted" / "metrics.csv").stat().st_size == 0
        return
    continuous, continuous_checkpoints = run("continuous")
    assert capture_flags == [True, False, True, False]
    exported = np.load(continuous / "parity_batch.npz")
    np.testing.assert_array_equal(exported["old_logps"], old_snapshots[0][1])
    assert (continuous / "startup_parity_batch.npz").exists() and (continuous / "joint_old_drift.json").exists()
    if is_cap is not None:
        expected_is = np.minimum(np.exp(exported["old_logps"] - exported["rollout_logps"]), is_cap)
        expected_is = np.where(exported["completion_mask"], expected_is, 0)
        np.testing.assert_allclose(exported["sampler_is_weights"], expected_is, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(old_snapshots[0][1], old_snapshots[1][1])
    np.testing.assert_array_equal(old_snapshots[2][1], old_snapshots[3][1])
    with (continuous / "metrics.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert [int(row["rollout_step"]) for row in rows] == [1, 1, 2, 2]
    generated_rows = 8 if dynamic else 4
    assert [int(row["generated_rows"]) for row in rows] == [generated_rows, 0, generated_rows, 0]
    assert [int(row["sampling_attempts"]) for row in rows] == [2 if dynamic else 1, 0, 2 if dynamic else 1, 0]
    assert float(rows[1]["ratio_min"]) != 1.0 or float(rows[1]["ratio_max"]) != 1.0
    assert float(rows[1]["rollout_s"]) == float(rows[1]["old_logps_s"]) == 0.0
    raw_rows = [json.loads(line) for line in (continuous / "rollouts.jsonl").read_text().splitlines()]
    assert sum(int(row["generated_tokens"]) for row in rows) == sum(row["length"] for row in raw_rows)
    assert sum(int(row["generated_rows"]) for row in rows) == len(raw_rows)
    split = json.loads((continuous / "data_split.json").read_text())
    assert not {row["dataset_index"] for row in raw_rows} & set(split["dev_indices"])
    if dynamic:
        evaluated = json.loads((continuous / "eval" / "step_00000004" / "summary.json").read_text())
        assert evaluated["dataset_indices"] == split["dev_indices"]
    resumed, resumed_checkpoints = run("resumed", continuous_checkpoints / "step_00000002")
    expected = load_file(continuous_checkpoints / "step_00000004" / "state.safetensors")
    observed = load_file(resumed_checkpoints / "step_00000004" / "state.safetensors")
    assert expected.keys() == observed.keys()
    for key in expected:
        np.testing.assert_array_equal(observed[key], expected[key], err_msg=key)
    before = [
        json.loads(line)
        for line in (continuous / "rollouts.jsonl").read_text().splitlines()
        if json.loads(line)["step"] == 3
    ]
    after = [json.loads(line) for line in (resumed / "rollouts.jsonl").read_text().splitlines()]
    assert before == after
    meta = json.loads((resumed_checkpoints / "step_00000004" / "meta.json").read_text())
    assert meta["metadata"]["data_cursor"] == (4 if dynamic else 2)

    saved_config = meta["metadata"]["run_config"]
    assert saved_config.get("rollout_layout", "replicated") == rollout_layout
    assert ("rollout_layout" in saved_config) == (rollout_layout == "fsdp")
    with pytest.raises(ValueError, match="恢复配置|首个LoRA"):
        run(
            "changed_rollout_layout",
            continuous_checkpoints / "step_00000002",
            layout="fsdp" if rollout_layout == "replicated" else "replicated",
        )
    assert saved_config.get("prompt_style", "chat") == prompt_style
    assert ("prompt_protocol" in saved_config) == (prompt_style == "plain")
    assert np.all(exported["prompt_ids"][:, : (2 if prompt_style == "plain" else 1)] == 0)
    with pytest.raises(ValueError, match="恢复配置"):
        run(
            "changed_prompt_style",
            continuous_checkpoints / "step_00000002",
            style="chat" if prompt_style == "plain" else "plain",
        )

    recorded_precision = meta["metadata"]["run_config"]["jax_default_matmul_precision"]
    assert recorded_precision == (jax.config.jax_default_matmul_precision or "default")
    alternate = "default" if recorded_precision == "highest" else "highest"
    with jax.default_matmul_precision(alternate), pytest.raises(ValueError, match="精度"):
        run("changed_precision", continuous_checkpoints / "step_00000002")

    original_metadata_reader = trainer.read_checkpoint_metadata

    def read_legacy_checkpoint(path):
        record = original_metadata_reader(path)
        del record["metadata"]["run_config"]["jax_default_matmul_precision"]
        return record

    monkeypatch.setattr(trainer, "read_checkpoint_metadata", read_legacy_checkpoint)
    with pytest.raises(ValueError, match="精度"):
        run("missing_precision", continuous_checkpoints / "step_00000002")
    if lora:
        assert saved_config["training_mode"] == "lora-unscaled-fp32-v1"
        assert saved_config["lora"] == dict(rank=2, alpha=3.0, targets=["q_proj", "k_proj", "v_proj", "o_proj"])
        assert saved_config["frozen_base_identity"] == trainer.lora_base_identity(model_path)
        assert all("embed_tokens" not in key for key in expected)
        monkeypatch.setattr(trainer, "read_checkpoint_metadata", original_metadata_reader)
        evaluator = None
        evaluation_command = None
        if dynamic:
            import gemma4_posttrain_jax.evaluation as evaluation

            monkeypatch.setattr(evaluation, "load_gsm8k", lambda split: dataset)
            monkeypatch.setitem(sys.modules, "train_grpo", trainer)
            eval_path = path.with_name("evaluate_gsm8k.py")
            eval_spec = importlib.util.spec_from_file_location("lora_checkpoint_evaluation", eval_path)
            assert eval_spec is not None and eval_spec.loader is not None
            evaluator = importlib.util.module_from_spec(eval_spec)
            eval_spec.loader.exec_module(evaluator)
            monkeypatch.setattr(evaluator, "load_hf_params", trainer.load_hf_params)
            monkeypatch.setattr(evaluator, "load_hf_eos_token_ids", lambda *args, **kwargs: (1,))
            evaluation_command = [
                "evaluate_gsm8k.py",
                "--checkpoint",
                str(continuous_checkpoints / "step_00000004"),
                "--subset",
                "train-dev",
                "--size",
                "2",
                "--dev-size",
                "2",
                "--dev-seed",
                "19",
                "--batch-size",
                "4",
                "--max-prompt-len",
                "4",
                "--max-new-tokens",
                "4",
                "--reward-workers",
                "1",
            ]
            independent = tmp_path / "independent_lora_eval"
            monkeypatch.setattr(sys, "argv", [*evaluation_command, "--output-dir", str(independent)])
            evaluator.main()
            periodic = continuous / "eval/step_00000004/predictions.jsonl"
            assert (independent / "evaluation/predictions.jsonl").read_bytes() == periodic.read_bytes()
            assert json.loads((independent / "meta.json").read_text())["step"] == 4
        weights = model_path / "model.safetensors"
        data = weights.read_bytes()
        weights.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        with pytest.raises(ValueError, match="恢复配置"):
            run("changed_base_bytes", continuous_checkpoints / "step_00000002")
        if evaluator is not None:
            assert evaluation_command is not None
            monkeypatch.setattr(sys, "argv", [*evaluation_command, "--output-dir", str(tmp_path / "changed_base_eval")])
            with pytest.raises(ValueError, match="字节改变"):
                evaluator.main()
