"""GSM8K normalization, masking, fixed shapes, and resume order."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from gemma4_posttrain_jax.data import (
    GSM8KBatchStream,
    collate_sft,
    encode_sft_example,
    eval_subset,
    format_prompt,
    normalize_gsm8k_answer,
)


class FakeTokenizer:
    bos_token_id = 2
    eos_token_id = 1
    pad_token_id = 0

    @staticmethod
    def _code(text: str) -> int:
        return 100 + sum(text.encode()) % 1000

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool = False
    ) -> dict[str, list[list[int]]]:
        assert tokenize
        prompt = [2, self._code(messages[0]["content"]), 105]
        if len(messages) == 1:
            assert add_generation_prompt
            tokens = prompt
        else:
            tokens = prompt + [self._code(messages[1]["content"]), 106, 107]
        return {"input_ids": [tokens]}

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        return {"input_ids": ([2] if add_special_tokens else []) + [self._code(text)]}


def test_normalize_gsm8k_answer_removes_calculator_annotation() -> None:
    response, gold = normalize_gsm8k_answer("There are <<48/2=24>> 24 groups.\n#### 1,234")
    assert gold == "1234"
    assert response == "There are 24 groups.\nThe answer is \\boxed{1234}."


def test_prompt_mask_response_boundaries_and_truncation() -> None:
    tokenizer = FakeTokenizer()
    prompt = format_prompt("What is 2+2?", tokenizer)
    tokens, labels = encode_sft_example(tokenizer, "What is 2+2?", "2+2=4.\n#### 4", 16)

    assert all(label == -100 for label in labels[: len(prompt)])
    assert labels[len(prompt)] == tokens[len(prompt)]
    assert labels[-2] == tokens[-2] == 106  # Gemma-style assistant turn terminator remains a target.
    assert labels[-1] == tokens[-1] == 107  # The template's trailing newline is also a target.
    assert "\\boxed{}" in tokenizer_last_user_content(tokenizer, "What is 2+2?")

    one_response_token, one_response_label = encode_sft_example(
        tokenizer, "What is 2+2?", "2+2=4.\n#### 4", len(prompt) + 1
    )
    assert one_response_label[-1] == one_response_token[-1] != -100
    with pytest.raises(ValueError, match="leaves no assistant tokens"):
        encode_sft_example(tokenizer, "What is 2+2?", "2+2=4.\n#### 4", len(prompt))


def tokenizer_last_user_content(tokenizer: FakeTokenizer, question: str) -> str:
    """Mirror the fake token code to assert that prompt formatting includes the instruction."""

    candidate = question + "\nSolve the problem step by step and put the final answer in \\boxed{}."
    assert tokenizer._code(candidate) == format_prompt(question, tokenizer)[1]
    return candidate


def test_collate_has_fixed_shape_padding_and_mask() -> None:
    batch = collate_sft([([2, 7, 8], [-100, 7, 8])], batch_size=2, sequence_length=5, pad_token_id=0)
    ids, labels, mask = batch
    assert ids.shape == labels.shape == mask.shape == (2, 5)
    np.testing.assert_array_equal(ids[0], [2, 7, 8, 0, 0])
    np.testing.assert_array_equal(labels[0], [-100, 7, 8, -100, -100])
    np.testing.assert_array_equal(mask[0], [True, True, True, False, False])


def test_gsm8k_stream_resume_reproduces_next_batch_and_eval_subset() -> None:
    tokenizer = FakeTokenizer()
    dataset: list[dict[str, Any]] = [
        {"question": f"question {index}", "answer": f"reason {index}\n#### {index}"} for index in range(5)
    ]
    stream = GSM8KBatchStream(dataset, tokenizer, batch_size=2, sequence_length=8, pad_token_id=0, seed=7)
    next(stream)
    next(stream)
    state = json.loads(json.dumps(stream.state_dict()))
    expected = next(stream)

    restored = GSM8KBatchStream(dataset, tokenizer, batch_size=2, sequence_length=8, pad_token_id=0, seed=7)
    restored.load_state_dict(state)
    actual = next(restored)
    for expected_array, actual_array in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(expected_array, actual_array)

    assert eval_subset(dataset, size=3, seed=11) == eval_subset(dataset, size=3, seed=11)
    assert len(eval_subset(dataset, size=3, seed=11)) == 3
