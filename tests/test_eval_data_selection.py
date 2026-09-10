"""评估集合身份、MATH gold/source_id 与 checkpoint 留出声明的 CPU 回归。"""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_evaluation import EvalTokenizer

from gemma4_posttrain_jax import evaluation
from gemma4_posttrain_jax.data import eval_subset, split_holdout_indices
from gemma4_posttrain_jax.evaluation import evaluate_batches, indices_sha256, load_evaluation_data, prepare_eval_batches
from gemma4_posttrain_jax.math_data import MathData
from gemma4_posttrain_jax.sampler import RolloutBatch


class FakeDataset(list):
    _fingerprint = "fixed-source-fingerprint"


def checkpoint_config(dataset, *, name="gsm8k", dev_size=7, dev_seed=13, provenance=None):
    train, dev = split_holdout_indices(len(dataset), dev_size, seed=dev_seed)
    return {
        "dataset": name,
        "dataset_fingerprint": dataset._fingerprint,
        "dataset_size": len(dataset),
        "dev_size": dev_size,
        "dev_seed": dev_seed,
        "train_indices_sha256": indices_sha256(train),
        "dev_indices_sha256": indices_sha256(dev),
        "data_provenance": provenance,
    }


def test_gsm_default_and_unused_preserve_historical_selection(monkeypatch) -> None:
    rows = FakeDataset(range(700))
    monkeypatch.setattr(evaluation, "load_gsm8k", lambda split: rows)
    selected = load_evaluation_data(size=11, seed=8)
    assert (selected.subset, selected.split) == ("test-monitor", "test")
    assert selected.indices == eval_subset(list(range(700)), size=11, seed=8)
    remaining, old_monitor = split_holdout_indices(700, 500, seed=0)
    selected = load_evaluation_data(subset="test-unused", size=150, seed=8)
    assert selected.indices == eval_subset(remaining, size=150, seed=8)
    assert set(selected.indices).isdisjoint(old_monitor)
    with pytest.raises(ValueError, match="subset size"):
        load_evaluation_data(subset="test-unused", size=201)


def test_train_dev_preserves_exclusion_order_and_legacy_dataset_field(monkeypatch) -> None:
    rows = FakeDataset(range(37))
    monkeypatch.setattr(evaluation, "load_gsm8k", lambda split: rows)
    config = checkpoint_config(rows)
    del config["dataset"]
    selected = load_evaluation_data(
        subset="train-dev", size=4, seed=99, dev_size=7, dev_seed=13, checkpoint_run_config=config
    )
    train, dev = split_holdout_indices(37, 7, seed=13)
    assert selected.indices == dev[:4]
    assert set(selected.indices).isdisjoint(train)
    with pytest.raises(ValueError, match="超过"):
        load_evaluation_data(subset="train-dev", size=8, dev_size=7, dev_seed=13)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset", "math"),
        ("dataset_fingerprint", "changed"),
        ("dataset_size", 38),
        ("dev_size", 8),
        ("dev_seed", 14),
        ("train_indices_sha256", "changed"),
        ("dev_indices_sha256", "changed"),
        ("dev_indices_sha256", None),
    ],
)
def test_train_dev_rejects_changed_or_missing_checkpoint_identity(monkeypatch, field, value) -> None:
    rows = FakeDataset(range(37))
    monkeypatch.setattr(evaluation, "load_gsm8k", lambda split: rows)
    config = checkpoint_config(rows)
    if value is None:
        del config[field]
    else:
        config[field] = value
    with pytest.raises(ValueError, match="身份/排除集合"):
        load_evaluation_data(subset="train-dev", size=4, dev_size=7, dev_seed=13, checkpoint_run_config=config)


def test_math_train_dev_checks_complete_json_safe_gold_provenance(monkeypatch) -> None:
    rows = FakeDataset(range(37))
    provenance = {"revision": "fixed", "gold_protocol": "audited", "corrections": {"row": ["a", "b"]}}
    monkeypatch.setattr(evaluation, "load_math_training_data", lambda: MathData(rows, provenance))
    config = json.loads(json.dumps(checkpoint_config(rows, name="math", provenance=provenance)))
    selected = load_evaluation_data(
        "math", subset="train-dev", size=4, dev_size=7, dev_seed=13, checkpoint_run_config=config
    )
    assert selected.indices == split_holdout_indices(37, 7, seed=13)[1][:4]
    assert selected.provenance == provenance
    config["data_provenance"]["gold_protocol"] = "obsolete"
    with pytest.raises(ValueError, match="gold 协议"):
        load_evaluation_data("math", subset="train-dev", size=4, dev_size=7, dev_seed=13, checkpoint_run_config=config)


def test_math500_all_rows_and_transfer_evaluation_are_allowed(monkeypatch) -> None:
    rows = FakeDataset(range(500))
    monkeypatch.setattr(evaluation, "load_math500_data", lambda: MathData(rows, {"revision": "fixed"}))
    selected = load_evaluation_data("math", seed=8, checkpoint_run_config={"dataset": "gsm8k"})
    assert (selected.subset, selected.split) == ("math500", "test")
    assert selected.indices == eval_subset(list(range(500)), size=500, seed=8)
    assert sorted(selected.indices) == list(range(500))
    with pytest.raises(ValueError, match="subset size"):
        load_evaluation_data("math", size=501)


@pytest.mark.parametrize(
    ("dataset", "subset", "size"),
    [
        ("other", None, 1),
        ("gsm8k", "math500", 1),
        ("math", "test-unused", 1),
        ("math", "test-monitor", 1),
        ("math", None, 0),
    ],
)
def test_invalid_dataset_subset_fails_before_download(monkeypatch, dataset, subset, size) -> None:
    def unexpected_load(*args):
        pytest.fail("invalid selection must fail before loading data")

    monkeypatch.setattr(evaluation, "load_gsm8k", unexpected_load)
    monkeypatch.setattr(evaluation, "load_math500_data", unexpected_load)
    monkeypatch.setattr(evaluation, "load_math_training_data", unexpected_load)
    with pytest.raises(ValueError):
        load_evaluation_data(dataset, subset=subset, size=size)


def test_math_source_identity_and_alternative_gold_survive_padding_and_json(tmp_path) -> None:
    dataset = [
        {
            "source_id": f"test/algebra/{index}",
            "question": f"question {index}",
            "gold": r"\boxed{7}",
            "gold_alternatives": [r"\boxed{8}"],
        }
        for index in range(3)
    ]
    batches = prepare_eval_batches(dataset, EvalTokenizer(), size=3, indices=[2, 0, 1], batch_size=4, max_prompt_len=4)
    assert batches[0].source_ids == ("test/algebra/2", "test/algebra/0", "test/algebra/1")

    def generate(params, ids, mask, key):
        tokens = np.asarray([[7, 1], [8, 1], [9, 1], [7, 1]], np.int32)
        return RolloutBatch(ids, tokens, np.ones_like(tokens, bool), np.zeros((4, 2)), np.full(4, 2, np.int32))

    summary = evaluate_batches(generate, None, batches, EvalTokenizer(), output_dir=tmp_path / "eval", eos_ids=(1,))
    records = [json.loads(line) for line in (tmp_path / "eval/predictions.jsonl").read_text().splitlines()]
    assert (summary["count"], summary["correct"]) == (3, 2)
    assert [record["source_id"] for record in records] == ["test/algebra/2", "test/algebra/0", "test/algebra/1"]
    assert [record["dataset_index"] for record in records] == [2, 0, 1]
    assert all(record["gold"] == [r"\boxed{7}", r"\boxed{8}"] for record in records)


@pytest.mark.parametrize("source_ids", [["test/0", None], ["test/0", ""], ["test/0", "test/0"], ["test/0", 7]])
def test_source_identity_must_be_complete_and_unique(source_ids) -> None:
    rows = [{"question": "question", "gold": r"\boxed{7}", "source_id": value} for value in source_ids]
    with pytest.raises(ValueError, match="source_id"):
        prepare_eval_batches(rows, EvalTokenizer(), size=2, batch_size=4, max_prompt_len=4)
