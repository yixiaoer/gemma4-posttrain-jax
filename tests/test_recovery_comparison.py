"""恢复比较必须发现RNG、缺失字段和浮点字节差异，而非只比较解码文本。"""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize(
    "backend,protocol,accepted",
    [
        ("jax", "fp32-exp-log-highest-v1", True),
        ("tpu-inference-separate-process-v1", "engine-v1", True),
        ("jax", "engine-v1", False),
        ("tpu-inference-separate-process-v1", "fp32-exp-log-highest-v1", False),
        ("tpu-inference-separate-process-v1", "unknown", False),
        ("tpu-inference-same-process-v1", "engine-v1", False),
        ("inference-process", "engine-v1", False),
        ("unknown", "engine-v1", False),
        (None, None, False),
    ],
)
def test_training_sampling_protocol_is_bound_to_supported_backend(monkeypatch, backend, protocol, accepted):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    check = importlib.import_module("compare_recovery").check_training_sampling_protocol
    config = {"rollout_backend": backend, "sampling_math_protocol": protocol}
    if accepted:
        assert check(config) == protocol
    else:
        with pytest.raises(ValueError, match="后端"):
            check(config)


def test_native_records_without_backend_do_not_require_engine_files(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("compare_recovery")
    assert module.check_backend_records(tmp_path, {"sampling_math_protocol": "fp32-exp-log-highest-v1"}) is None


def test_matching_candidate_arrays_cannot_claim_another_backend(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("compare_recovery")
    values = {field: np.asarray("fixture") for field in module.CANDIDATE_FIELDS}
    values.update(protocol=np.asarray("rollout-candidate-arrays-v1"), backend=np.asarray("jax"))
    path = tmp_path / "candidate.npz"
    np.savez(path, **values)
    with pytest.raises(ValueError, match="候选后端"):
        module.compare_candidate_files(path, path, expected_backend="inference-process")


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


@pytest.mark.parametrize(
    ("expected_count", "expected_length", "corruption"),
    [
        (count, length, corruption)
        for count, length in ((4, None), (32, 1024))
        for corruption in (
            None,
            "step",
            "count",
            "duplicate",
            "indices",
            "source",
            "sampler",
            "both-nongreedy",
            "empty",
        )
    ]
    + [(32, 1024, corruption) for corruption in ("length", "missing-length", "over-limit")],
)
def test_evaluation_comparison_checks_model_and_complete_question_set(
    tmp_path, monkeypatch, expected_count, expected_length, corruption
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("compare_recovery")
    train_meta = {
        "run_config": {"dataset_fingerprint": "dataset-v1"},
        "source_files_sha256": {"gemma4_posttrain_jax/sampler.py": "sampler-v1"},
    }
    rows = [
        dict(dataset_index=i, prompt_token_ids=[2, 3], completion_token_ids=[4, 1], length=2)
        for i in range(expected_count)
    ]
    if corruption == "duplicate":
        rows[1] = rows[0]
    if corruption in ("empty", "over-limit"):
        length = 0 if corruption == "empty" else expected_length + 1
        rows[0].update(completion_token_ids=[4] * length, length=length)
    indices = [row["dataset_index"] for row in rows]
    (tmp_path / "baseline").mkdir()
    (tmp_path / "baseline/data_split.json").write_text(json.dumps({"training_indices": [100, 101]}))
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
        if corruption == "both-nongreedy":
            meta["sampler"]["temperature"] = 1
        if expected_length is not None and corruption != "missing-length":
            meta["sampler"]["max_new_tokens"] = 256 if corruption == "length" else expected_length
        (folder / "meta.json").write_text(json.dumps(meta))
        (folder / "summary.json").write_text(
            json.dumps(
                dict(
                    step=4,
                    count=1 if corruption == "count" else expected_count,
                    dataset_indices=indices,
                )
            )
        )
    if corruption is None:
        assert (
            module.check_evaluation_records(
                tmp_path, train_meta, 17, expected_count=expected_count, expected_max_new_tokens=expected_length
            )
            == rows
        )
    else:
        with pytest.raises(ValueError, match="评估"):
            module.check_evaluation_records(
                tmp_path, train_meta, 17, expected_count=expected_count, expected_max_new_tokens=expected_length
            )
