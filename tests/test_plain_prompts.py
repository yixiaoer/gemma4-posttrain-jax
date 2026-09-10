"""plain指令的真实token边界与训练/评估模板身份。"""

from __future__ import annotations

import numpy as np
import pytest
from test_data import FakeTokenizer

from gemma4_posttrain_jax.data import collate_grpo_prompts, format_prompt
from gemma4_posttrain_jax.evaluation import prepare_eval_batches, resolve_eval_prompt_style


@pytest.mark.parametrize("chat", [True, False])
def test_train_and_eval_use_same_tokens_with_grouping_and_left_padding(chat: bool) -> None:
    tokenizer = FakeTokenizer()
    dataset = [{"question": f"question {i}", "answer": f"work\n#### {i}"} for i in range(3)]
    train = collate_grpo_prompts(dataset, tokenizer, group_size=2, max_prompt_len=5, pad_token_id=0, chat=chat)
    batches = prepare_eval_batches(
        dataset, tokenizer, size=3, indices=[0, 1, 2], batch_size=2, max_prompt_len=5, chat=chat
    )
    for i, example in enumerate(dataset):
        tokens = format_prompt(example["question"], tokenizer, chat=chat)
        assert len(tokens) == (3 if chat else 2)
        np.testing.assert_array_equal(train.prompt_ids[2 * i, -len(tokens) :], tokens)
        np.testing.assert_array_equal(train.prompt_ids[2 * i], batches[i // 2].prompt_ids[i % 2])
        np.testing.assert_array_equal(train.prompt_mask[2 * i], batches[i // 2].prompt_mask[i % 2])
        np.testing.assert_array_equal(train.prompt_ids[2 * i], train.prompt_ids[2 * i + 1])
    assert train.golds == ("0", "0", "1", "1", "2", "2")


def test_eval_inherits_checkpoint_style_and_rejects_protocol_changes() -> None:
    plain = {"prompt_style": "plain", "prompt_protocol": "plain-bos-instruction-v2"}
    assert resolve_eval_prompt_style(None, None) == "chat"
    assert resolve_eval_prompt_style(None, {}) == "chat"
    assert resolve_eval_prompt_style("chat", {}) == "chat"
    assert resolve_eval_prompt_style("plain", None) == "plain"
    assert resolve_eval_prompt_style(None, plain) == "plain"
    assert resolve_eval_prompt_style("plain", plain) == "plain"
    for requested, saved in (
        ("plain", {}),
        ("chat", plain),
        (None, {"prompt_style": "unknown"}),
        (None, {"prompt_style": "plain"}),
        (None, {"prompt_style": "plain", "prompt_protocol": "plain-instruction-v1"}),
        ("unknown", None),
    ):
        with pytest.raises(ValueError):
            resolve_eval_prompt_style(requested, saved)


def test_plain_does_not_inherit_tokenizer_default_bos_policy() -> None:
    class NoAutomaticBOS(FakeTokenizer):
        def __call__(self, text: str, *, add_special_tokens: bool):
            return {"input_ids": [self._code(text)]}

    question = "question 1"
    assert format_prompt(question, FakeTokenizer(), chat=False) == format_prompt(question, NoAutomaticBOS(), chat=False)
    assert format_prompt(question, NoAutomaticBOS(), chat=False)[0] == 2
