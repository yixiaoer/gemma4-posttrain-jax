"""固定评估子集、末批补齐、EOS 与统计分母的 CPU 回归。"""

from __future__ import annotations

import json

import jax
import numpy as np
import pytest
from test_data import FakeTokenizer

from gemma4_posttrain_jax.data import eval_subset, grpo_prompt_indices, split_holdout_indices
from gemma4_posttrain_jax.evaluation import evaluate_batches, prepare_eval_batches
from gemma4_posttrain_jax.sampler import RolloutBatch


class EvalTokenizer(FakeTokenizer):
    def batch_decode(self, rows, *, skip_special_tokens):
        assert skip_special_tokens
        return [f"The answer is \\boxed{{{row[0]}}}." for row in rows]


def test_eval_padding_is_excluded_and_eos_at_limit_is_not_truncated(tmp_path) -> None:
    tokenizer = EvalTokenizer()
    dataset = [{"question": f"question {index}", "answer": "reasoning\n#### 7"} for index in range(5)]
    batches = prepare_eval_batches(dataset, tokenizer, size=5, seed=11, batch_size=4, max_prompt_len=4)
    assert [index for batch in batches for index in batch.indices] == eval_subset(list(range(5)), size=5, seed=11)
    assert len(batches[1].indices) == 1
    np.testing.assert_array_equal(batches[1].prompt_ids, np.repeat(batches[1].prompt_ids[:1], 4, axis=0))
    np.testing.assert_array_equal(batches[0].prompt_mask[:, 0], False)
    params = np.asarray([1.25, 2.5])
    keys = []

    def generate(current, ids, mask, key):
        np.testing.assert_array_equal(current, params)
        assert ids.shape == mask.shape == (4, 4)
        keys.append(np.asarray(key))
        tokens = np.asarray([[7, 1, 0], [8, 0, 1], [7, 7, 7], [7, 1, 0]], np.int32)
        lengths = np.asarray([2, 3, 3, 2], np.int32)
        return RolloutBatch(ids, tokens, np.arange(3)[None, :] < lengths[:, None], np.zeros((4, 3)), lengths)

    summary = evaluate_batches(generate, params, batches, tokenizer, output_dir=tmp_path / "eval", eos_ids=(1,), seed=9)
    assert (summary["count"], summary["correct"], summary["accuracy"]) == (5, 4, 0.8)
    assert summary["truncated_fraction"] == 0.2
    records = [json.loads(line) for line in (tmp_path / "eval/predictions.jsonl").read_text().splitlines()]
    assert len(records) == len(set(summary["dataset_indices"])) == 5
    assert records[1]["truncated"] is False
    assert records[2]["truncated"] is True
    np.testing.assert_array_equal(keys[1], jax.random.fold_in(jax.random.PRNGKey(9), 1))
    np.testing.assert_array_equal(params, [1.25, 2.5])
    with pytest.raises(FileExistsError):
        evaluate_batches(generate, params, batches, tokenizer, output_dir=tmp_path / "eval", eos_ids=(1,))


def test_eval_rejects_prompt_truncation_and_bad_subset() -> None:
    dataset = [{"question": "question", "answer": "reasoning\n#### 7"}]
    with pytest.raises(ValueError, match="prompt 长度"):
        prepare_eval_batches(dataset, EvalTokenizer(), size=1, batch_size=4, max_prompt_len=2)
    with pytest.raises(ValueError, match="eval subset"):
        prepare_eval_batches(dataset, EvalTokenizer(), size=2)


def test_eval_subset_preserves_question_gold_alignment() -> None:
    dataset = [{"question": f"question {index}", "answer": f"reasoning\n#### {index}"} for index in range(9)]
    batches = prepare_eval_batches(dataset, EvalTokenizer(), size=7, seed=13, batch_size=4, max_prompt_len=4)
    assert [index for batch in batches for index in batch.indices] == eval_subset(list(range(9)), size=7, seed=13)
    for batch in batches:
        assert batch.golds == tuple(str(index) for index in batch.indices)


def test_explicit_holdout_excludes_training_prompts_across_epochs_and_keeps_raw_ids() -> None:
    train, dev = split_holdout_indices(17, 5, seed=13)
    assert sorted(train + dev) == list(range(17))
    assert not set(train) & set(dev)
    assert dev == eval_subset(list(range(17)), size=5, seed=13)
    for cursor in range(20):
        prompts = [train[position] for position in grpo_prompt_indices(len(train), 3, cursor, seed=2)]
        assert not set(prompts) & set(dev)
    dataset = [{"question": f"question {index}", "answer": f"reasoning\n#### {index}"} for index in range(17)]
    batches = prepare_eval_batches(dataset, EvalTokenizer(), size=5, indices=dev, batch_size=4, max_prompt_len=4)
    assert [index for batch in batches for index in batch.indices] == dev
    assert [gold for batch in batches for gold in batch.golds] == [str(index) for index in dev]
    assert split_holdout_indices(17, 0) == (list(range(17)), [])
    remaining, monitor = split_holdout_indices(1319, 500)
    assert len(remaining) == 819 and not set(remaining) & set(monitor)


def test_explicit_evaluation_indices_reject_duplicates_or_invalid_raw_ids() -> None:
    dataset = [{"question": "question", "answer": "work\n#### 7"}] * 5
    for indices in ([0, 0], [0, 9], [0, 0.5], [0]):
        with pytest.raises(ValueError, match="显式评估题号"):
            prepare_eval_batches(dataset, EvalTokenizer(), size=2, indices=indices)
    for size in (-1, 5):
        with pytest.raises(ValueError, match="holdout"):
            split_holdout_indices(5, size)
