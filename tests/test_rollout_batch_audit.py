"""用小模型的实际生成和 Adam 更新检查样本记录，包括动态补采中被丢弃的样本。"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file
from test_grpo_host_loop import TinyDataset, TinyTokenizer

from gemma4_posttrain_jax.rewards import RewardOutput


def test_audit_preserves_state_and_all_candidate_arrays(tpu_grad_tiny, tmp_path, monkeypatch):
    _, params, config = tpu_grad_tiny
    path = Path(__file__).resolve().parents[1] / "scripts/train_grpo.py"
    spec = importlib.util.spec_from_file_location("rollout_audit_driver", path)
    assert spec and spec.loader
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    dataset = TinyDataset({"question": f"question {i}", "answer": "work\n#### 1"} for i in range(8))
    monkeypatch.setattr(driver, "load_gsm8k", lambda split: dataset)
    monkeypatch.setattr(
        driver, "load_hf_params", lambda *a, **kw: (jax.tree.map(lambda x: jnp.array(x, copy=True), params), config)
    )
    monkeypatch.setattr(driver, "load_hf_eos_token_ids", lambda *a, **kw: (1,))
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: TinyTokenizer())

    def rewards(completions, golds, **kwargs):
        # 每批仅第二组符合补采筛选；生成和训练仍使用真实实现。
        success = np.asarray([0, 0, 0, 1], np.float32)
        zero = np.zeros_like(success)
        return RewardOutput(success, success, zero, zero, success)

    monkeypatch.setattr(driver, "score_completions", rewards)
    for name in ("plain", "audited"):
        argv = [
            "train_grpo.py",
            "--model-path",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path / name),
            "--prompt-batch-size",
            "2",
            "--group-size",
            "2",
            "--max-prompt-len",
            "4",
            "--max-new-tokens",
            "4",
            "--microbatch-size",
            "2",
            "--max-steps",
            "2",
            "--learning-rate",
            "0.001",
            "--save-every",
            "2",
            "--checkpoint-dir",
            str(tmp_path / (name + "_state")),
            "--vocab-chunk",
            "17",
            "--sequence-chunk",
            "5",
            "--reward-workers",
            "1",
            "--dynamic-sampling",
            "--max-sampling-attempts",
            "2",
        ]
        if name == "audited":
            argv.append("--audit-rollout-batches")
        monkeypatch.setattr(sys, "argv", argv)
        driver.main()
    a = load_file(tmp_path / "plain_state/step_00000002/state.safetensors")
    b = load_file(tmp_path / "audited_state/step_00000002/state.safetensors")
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    assert (tmp_path / "plain/rollouts.jsonl").read_bytes() == (tmp_path / "audited/rollouts.jsonl").read_bytes()
    assert not (tmp_path / "plain/rollout_batches").exists()
    rows = [json.loads(line) for line in (tmp_path / "audited/rollouts.jsonl").read_text().splitlines()]
    assert len(rows) == 16 and sum(row["selected_for_update"] for row in rows) == 8
    batches = sorted((tmp_path / "audited/rollout_batches").glob("*.npz"))
    assert len(batches) == 4
    for cursor, file in enumerate(batches):
        with np.load(file, allow_pickle=False) as z:
            assert z["protocol"].item() == "rollout-candidate-arrays-v1"
            assert z["data_cursor"].item() == cursor and z["policy_step"].item() == cursor // 2
            assert z["backend"].item() == "jax" and z["proposal_known"].item() is True
            assert z["temperature"].item() == 1.0 and z["top_k"].item() == 0
            np.testing.assert_array_equal(
                z["key_data"], jax.random.key_data(jax.random.fold_in(jax.random.PRNGKey(0), cursor))
            )
            np.testing.assert_array_equal(z["lengths"], z["completion_mask"].sum(axis=1))
            assert z["prompt_ids"].shape == z["prompt_mask"].shape == (4, 4)
            for row in rows[cursor * 4 : (cursor + 1) * 4]:
                index = row["row"]
                assert row["dataset_index"] == z["dataset_indices"][index]
                tokens = z["completion_ids"][index, : row["length"]]
                assert " ".join(map(str, tokens.tolist())) == row["completion"]
            assert np.isfinite(z["raw_logprobs"][z["completion_mask"]]).all()
            np.testing.assert_array_equal(z["raw_logprobs"], z["proposal_logprobs"])
    with (tmp_path / "audited/metrics.csv").open() as handle:
        metrics = list(csv.DictReader(handle))
    assert all(float(row["rollout_audit_s"]) > 0 for row in metrics)
