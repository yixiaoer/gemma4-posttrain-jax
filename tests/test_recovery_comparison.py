"""恢复比较必须发现RNG、缺失字段和浮点字节差异，而非只比较解码文本。"""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("difference", [None, "rng", "missing", "missing-both", "signed-zero", "empty-both"])
def test_candidate_comparison_checks_all_saved_fields(tmp_path, monkeypatch, difference):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    compare = importlib.import_module("compare_recovery").compare_candidate_files
    original = {
        "protocol": np.asarray("rollout-candidate-arrays-v1"),
        "backend": np.asarray("jax"),
        "data_cursor": np.asarray(8, np.int64),
        "policy_step": np.asarray(2, np.int64),
        "target_old_policy_step": np.asarray(2, np.int64),
        "dataset_indices": np.asarray([5], np.int64),
        "key_data": np.asarray([0, 42], np.uint32),
        "prompt_ids": np.asarray([[2, 7]], np.int32),
        "prompt_mask": np.asarray([[True, True]]),
        "completion_ids": np.asarray([[4, 2]], np.int32),
        "completion_mask": np.asarray([[True, True]]),
        "lengths": np.asarray([2], np.int32),
        "raw_logprobs": np.asarray([[0.0, -1.0]], np.float32),
        "proposal_logprobs": np.asarray([[0.0, -1.0]], np.float32),
        "proposal_known": np.asarray(True),
        "temperature": np.asarray(1.0),
        "top_k": np.asarray(0),
        "top_p": np.asarray(1.0),
        "eos_ids": np.asarray([2], np.int32),
    }
    changed = {name: value.copy() for name, value in original.items()}
    if difference == "rng":
        changed["key_data"][1] += 1
    elif difference == "missing":
        del changed["key_data"]
    elif difference == "missing-both":
        del original["key_data"], changed["key_data"]
    elif difference == "signed-zero":
        changed["raw_logprobs"][0, 0] = -0.0
    elif difference == "empty-both":
        original["completion_ids"] = changed["completion_ids"] = np.empty((0, 2), np.int32)
    left, right = tmp_path / "left.npz", tmp_path / "right.npz"
    np.savez(left, **original)
    np.savez(right, **changed)
    if difference is None:
        assert compare(left, right)["rows"] == 1
    else:
        with pytest.raises(ValueError, match="候选"):
            compare(left, right)


@pytest.mark.parametrize("corruption", [None, "step", "count", "duplicate", "indices", "source", "sampler"])
def test_evaluation_comparison_checks_model_and_complete_question_set(tmp_path, monkeypatch, corruption):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("compare_recovery")
    train_meta = {
        "run_config": {"dataset_fingerprint": "dataset-v1"},
        "source_files_sha256": {"gemma4_posttrain_jax/sampler.py": "sampler-v1"},
    }
    rows = [dict(dataset_index=i, prompt_token_ids=[2, 3], completion_token_ids=[4, 1], length=2) for i in range(4)]
    if corruption == "duplicate":
        rows[1] = rows[0]
    indices = [row["dataset_index"] for row in rows]
    (tmp_path / "baseline").mkdir()
    (tmp_path / "baseline/data_split.json").write_text(json.dumps({"training_indices": [8, 9]}))
    for run in ("baseline", "resume"):
        folder = tmp_path / f"eval-{run}"
        (folder / "evaluation").mkdir(parents=True)
        (folder / "evaluation/predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        meta = dict(
            step=3 if corruption == "step" else 4,
            split="train",
            subset="train-dev",
            checkpoint_metadata=dict(run_config=train_meta["run_config"], data_cursor=17),
            dataset_fingerprint="dataset-v1",
            indices_sha256=module.indices_sha256(indices),
            prediction_record_schema="raw-token-ids-v1",
            sampling_math_protocol="fp32-exp-log-highest-v1",
            source_files_sha256=dict(train_meta["source_files_sha256"]),
            sampler=dict(temperature=0),
            jax_default_matmul_precision="highest",
        )
        if corruption == "indices":
            meta["indices_sha256"] = module.indices_sha256([4, 5, 6, 7])
        if corruption == "source":
            meta["source_files_sha256"]["gemma4_posttrain_jax/sampler.py"] = "another-sampler"
        if corruption == "sampler" and run == "resume":
            meta["sampler"]["temperature"] = 1
        (folder / "meta.json").write_text(json.dumps(meta))
        (folder / "summary.json").write_text(
            json.dumps(
                dict(
                    step=4,
                    count=1 if corruption == "count" else 4,
                    dataset_indices=indices,
                )
            )
        )
    if corruption is None:
        assert module.check_evaluation_records(tmp_path, train_meta, 17) == rows
    else:
        with pytest.raises(ValueError, match="评估"):
            module.check_evaluation_records(tmp_path, train_meta, 17)
