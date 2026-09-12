"""完整答案集合不能用单个成员得分；任选一题也不能提交两个答案。"""

import pytest
from test_data import FakeTokenizer

from gemma4_posttrain_jax.data import collate_grpo_prompts, example_gold
from gemma4_posttrain_jax.evaluation import prepare_eval_batches
from gemma4_posttrain_jax.math_data import prepare_math_training_rows
from gemma4_posttrain_jax.math_gold import MATH_GOLD_CORRECTIONS
from gemma4_posttrain_jax.rewards import score_completions


@pytest.mark.parametrize(
    "source_id,correct,partial",
    [
        ("train/algebra/row_334", r"\boxed{(-2,18)},\boxed{(8,38)}", r"\boxed{(8,38)}"),
        ("train/algebra/row_504", r"\boxed{(-2,23)},\boxed{(0,1)}", r"\boxed{(0,1)}"),
        ("train/intermediate_algebra/row_75", r"\boxed{2x+1},\boxed{-2x-1}", r"\boxed{-2x-1}"),
        ("train/intermediate_algebra/row_309", r"\boxed{1/3},\boxed{-5/2}", r"\boxed{-5/2}"),
        ("train/intermediate_algebra/row_612", r"\boxed{1},\boxed{9}", r"\boxed{9}"),
        ("train/precalculus/row_26", r"\boxed{12},\boxed{18}", r"\boxed{18}"),
        ("train/precalculus/row_62", r"\boxed{5},\boxed{-3}", r"\boxed{-3}"),
        ("train/precalculus/row_89", r"\boxed{30^\circ},\boxed{150^\circ}", r"\boxed{150^\circ}"),
        ("train/precalculus/row_660", r"\boxed{3/2},\boxed{-3}", r"\boxed{-3}"),
    ],
)
def test_required_answers_reject_partial_solution(source_id, correct, partial) -> None:
    gold = MATH_GOLD_CORRECTIONS[source_id].answers[0]
    result = score_completions([correct, partial, r"\boxed{12345}"], [gold] * 3, workers=1)
    assert result.task_success.tolist() == [1, 0, 0]


def test_either_focus_accepts_each_but_rejects_both_and_survives_worker_serialization() -> None:
    gold = MATH_GOLD_CORRECTIONS["train/intermediate_algebra/row_868"].answers
    completions = [r"\boxed{(-5,7)}", r"\boxed{(-5,1)}", r"\boxed{(-5,7)},\boxed{(-5,1)}", r"\boxed{(5,1)}"]
    result = score_completions(completions, [gold] * len(completions), workers=2)
    assert result.task_success.tolist() == [1, 1, 0, 0]


def test_gold_alternatives_survive_training_and_evaluation_collation() -> None:
    row = {"question": "either focus", "gold": r"\boxed{(-5,7)}", "gold_alternatives": [r"\boxed{(-5,1)}"]}
    gold = (r"\boxed{(-5,7)}", r"\boxed{(-5,1)}")
    assert example_gold(row) == gold
    batch = collate_grpo_prompts([row], FakeTokenizer(), group_size=2, max_prompt_len=8, pad_token_id=0)
    assert batch.golds == (gold, gold)
    evaluation = prepare_eval_batches([row], FakeTokenizer(), size=1, batch_size=2, max_prompt_len=8)
    assert evaluation[0].golds == (gold,)


@pytest.mark.parametrize("alternatives", [None, "1", [""], [1]])
def test_invalid_alternatives_are_not_silently_ignored(alternatives) -> None:
    with pytest.raises(ValueError, match="gold_alternatives"):
        example_gold({"gold": "1", "gold_alternatives": alternatives})


def test_pinned_correction_rejects_changed_source() -> None:
    with pytest.raises(ValueError, match="train/algebra/row_334"):
        prepare_math_training_rows(
            [{"source_id": "train/algebra/row_334", "problem": "changed", "solution": r"\boxed{1}"}], []
        )


@pytest.mark.parametrize(
    "source_id,correct,partial",
    [
        ("train/precalculus/row_168", r"\boxed{45^\circ,60^\circ,75^\circ}", r"\boxed{75^\circ}"),
        ("train/precalculus/row_471", r"\boxed{-4,4}", r"\boxed{-4}"),
    ],
)
def test_self_replay_pass_does_not_justify_a_missing_answer(source_id, correct, partial) -> None:
    gold = MATH_GOLD_CORRECTIONS[source_id].answers[0]
    result = score_completions([correct, partial], [gold] * 2, workers=1)
    assert result.task_success.tolist() == [1, 0]


@pytest.mark.parametrize(
    "source_id,first,second,wrong",
    [
        (
            "train/precalculus/row_663",
            r"\boxed{\begin{pmatrix}\frac23\\-\frac23\\-\frac13\end{pmatrix}}",
            r"\boxed{\begin{pmatrix}-\frac23\\\frac23\\\frac13\end{pmatrix}}",
            r"\boxed{\begin{pmatrix}2\\-2\\-1\end{pmatrix}}",
        ),
        ("train/precalculus/row_739", r"\boxed{\pi/3}", r"\boxed{-\pi/3}", r"\boxed{0}"),
    ],
)
def test_source_declared_equivalent_answers_are_both_accepted(source_id, first, second, wrong) -> None:
    gold = MATH_GOLD_CORRECTIONS[source_id].answers
    result = score_completions([first, second, wrong], [gold] * 3, workers=1)
    assert result.task_success.tolist() == [1, 1, 0]
