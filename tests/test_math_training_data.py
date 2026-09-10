"""MATH训练不得泄漏重复/测试题，符号gold必须完整进入训练与评估。"""

import pytest
from test_data import FakeTokenizer

from gemma4_posttrain_jax.data import collate_grpo_prompts, example_gold, split_holdout_indices
from gemma4_posttrain_jax.evaluation import prepare_eval_batches
from gemma4_posttrain_jax.math_data import prepare_math_training_rows


def test_math_exclusion_preserves_identity_and_disjoint_holdout() -> None:
    rows = [
        {"source_id": f"train/{index}", "problem": question, "solution": solution}
        for index, (question, solution) in enumerate(
            [
                ("same question", r"\boxed{1}"),
                (" same\nquestion ", r"\boxed{2}"),
                ("test overlap", r"\boxed{3}"),
                ("empty gold", r"\boxed{}"),
                ("valid interval", r"\boxed{(1,2)}"),
                ("valid fraction", r"\boxed{\frac{1}{2}}"),
                ("valid set", r"\boxed{\{1,2\}}"),
            ]
        )
    ]
    kept, excluded = prepare_math_training_rows(rows, [{"problem": "test\n overlap"}])
    assert [row["source_id"] for row in kept] == ["train/4", "train/5", "train/6"]
    assert [row["source_id"] for row in excluded] == ["train/0", "train/1", "train/2", "train/3"]
    assert excluded[0]["reasons"] == ["duplicate_train_question"]
    assert excluded[2]["reasons"] == ["exact_original_test_overlap"]
    assert excluded[3]["reasons"][0].startswith("invalid_gold:")
    train, dev = split_holdout_indices(len(kept), 1, seed=0)
    assert {kept[index]["question_sha256"] for index in train}.isdisjoint(
        kept[index]["question_sha256"] for index in dev
    )


def test_math_symbolic_gold_reaches_rollout_and_evaluation_unchanged() -> None:
    rows, excluded = prepare_math_training_rows(
        [
            {"source_id": "train/0", "problem": "an interval", "solution": r"\boxed{(1,2)}"},
            {"source_id": "train/1", "problem": "a matrix", "solution": r"\boxed{\begin{pmatrix}1\\2\end{pmatrix}}"},
        ],
        [],
    )
    assert not excluded
    tokenizer = FakeTokenizer()
    batch = collate_grpo_prompts(rows, tokenizer, group_size=2, max_prompt_len=8, pad_token_id=0)
    expected = (r"\boxed{(1,2)}", r"\boxed{\begin{pmatrix}1\\2\end{pmatrix}}")
    assert batch.golds == (expected[0], expected[0], expected[1], expected[1])
    evaluation = prepare_eval_batches(rows, tokenizer, size=2, batch_size=4, max_prompt_len=8, indices=[0, 1])
    assert evaluation[0].golds == expected
    assert len(evaluation[0].indices) == 2
    assert example_gold({"answer": "reason\n#### 1,234"}) == "1234"


@pytest.mark.parametrize("gold", [None, "", "  ", 12])
def test_explicit_invalid_gold_does_not_fall_back_to_gsm8k(gold) -> None:
    with pytest.raises(ValueError, match="gold"):
        example_gold({"gold": gold, "answer": "reason\n#### 1"})
