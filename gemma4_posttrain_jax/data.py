"""Deterministic, fixed-shape SFT data processing shared with later RL phases."""

from __future__ import annotations

import random
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

from .rewards import Gold

IGNORE_INDEX = -100
_CALCULATOR_ANNOTATION = re.compile(r"<<.*?>>")


class PromptBatch(NamedTuple):
    prompt_ids: np.ndarray
    prompt_mask: np.ndarray
    questions: tuple[str, ...]
    golds: tuple[Gold, ...]


def load_gsm8k(split: str = "train") -> Any:
    """Load the pinned public GSM8K configuration through Hugging Face datasets."""

    from datasets import load_dataset

    return load_dataset("openai/gsm8k", "main", split=split)


def normalize_gsm8k_answer(text: str) -> tuple[str, str]:
    """Remove calculator annotations and return a stable SFT response plus gold."""

    if "####" not in text:
        raise ValueError("GSM8K answer has no `####` gold delimiter")
    reasoning, gold = text.rsplit("####", 1)
    reasoning = _CALCULATOR_ANNOTATION.sub("", reasoning)
    reasoning = "\n".join(" ".join(line.split()) for line in reasoning.splitlines() if line.strip())
    gold = gold.strip().replace(",", "")
    if not reasoning or not gold:
        raise ValueError("GSM8K answer has empty reasoning or gold")
    return f"{reasoning}\nThe answer is \\boxed{{{gold}}}.", gold


def example_gold(example: Mapping[str, Any]) -> Gold:
    """显式规范化gold保留符号；原GSM8K行继续走既有数值答案协议。"""
    if "gold" not in example:
        return normalize_gsm8k_answer(str(example["answer"]))[1]
    gold = example["gold"]
    if not isinstance(gold, str) or not gold.strip():
        raise ValueError("显式gold必须为非空字符串")
    alternatives = example.get("gold_alternatives", [])
    if not isinstance(alternatives, (list, tuple)) or any(
        not isinstance(value, str) or not value.strip() for value in alternatives
    ):
        raise ValueError("gold_alternatives必须是非空字符串组成的序列")
    return (gold.strip(), *(value.strip() for value in alternatives)) if alternatives else gold.strip()


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def _user_content(question: str) -> str:
    return question.strip() + "\nSolve the problem step by step and put the final answer in \\boxed{}."


def format_prompt(question: str, tokenizer: Any, *, chat: bool = True) -> list[int]:
    """Tokenize the common GSM8K instruction, optionally through a chat template."""

    content = _user_content(question)
    if chat:
        user = {"role": "user", "content": content}
        return _token_ids(tokenizer.apply_chat_template([user], tokenize=True, add_generation_prompt=True))
    # base/it tokenizer默认BOS处理不同；plain对照显式保留一个相同BOS。
    return [int(tokenizer.bos_token_id), *_token_ids(tokenizer(content, add_special_tokens=False))]


def encode_sft_example(
    tokenizer: Any,
    question: str,
    answer: str,
    max_len: int,
    *,
    chat: bool = True,
) -> tuple[list[int], list[int]]:
    """Encode one example while masking every prompt token from the SFT loss."""

    if max_len < 2:
        raise ValueError("max_len must be at least two")
    response, _ = normalize_gsm8k_answer(answer) if "####" in answer else (answer.strip(), "")
    if not response:
        raise ValueError("SFT response is empty")
    prompt = format_prompt(question, tokenizer, chat=chat)
    if chat:
        messages = [
            {"role": "user", "content": _user_content(question)},
            {"role": "assistant", "content": response},
        ]
        full = _token_ids(tokenizer.apply_chat_template(messages, tokenize=True))
    else:
        response_ids = _token_ids(tokenizer(response, add_special_tokens=False))
        eos = tokenizer.eos_token_id
        full = prompt + response_ids + ([] if eos is None else [int(eos)])
    if full[: len(prompt)] != prompt:
        response_ids = _token_ids(tokenizer(response, add_special_tokens=False))
        eos = tokenizer.eos_token_id
        full = prompt + response_ids + ([] if eos is None else [int(eos)])
    full = full[:max_len]
    prompt_length = min(len(prompt), len(full))
    labels = [IGNORE_INDEX] * prompt_length + full[prompt_length:]
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError(f"max_len={max_len} leaves no assistant tokens")
    return full, labels


def collate_sft(
    examples: Sequence[tuple[list[int], list[int]]],
    batch_size: int,
    sequence_length: int,
    pad_token_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-pad/repeat examples to one fixed ``[batch_size, sequence_length]`` signature."""

    if not examples or batch_size <= 0 or sequence_length < 2:
        raise ValueError("collate_sft needs examples, positive batch_size, and sequence_length >= 2")
    ids = np.full((batch_size, sequence_length), pad_token_id, np.int32)
    labels = np.full((batch_size, sequence_length), IGNORE_INDEX, np.int32)
    mask = np.zeros((batch_size, sequence_length), np.bool_)
    for row in range(batch_size):
        tokens, targets = examples[row % len(examples)]
        if len(tokens) != len(targets):
            raise ValueError("token and label lengths differ")
        length = min(len(tokens), sequence_length)
        ids[row, :length] = tokens[:length]
        labels[row, :length] = targets[:length]
        mask[row, :length] = True
    return ids, labels, mask


def collate_grpo_prompts(
    examples: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    group_size: int,
    max_prompt_len: int,
    pad_token_id: int,
    chat: bool = True,
) -> PromptBatch:
    """Left-pad prompts and repeat each source example contiguously for grouped rollout."""

    if not examples or group_size < 2 or max_prompt_len <= 0:
        raise ValueError("GRPO collation needs examples, group_size >= 2, and positive max_prompt_len")
    batch_size = len(examples) * group_size
    prompt_ids = np.full((batch_size, max_prompt_len), pad_token_id, np.int32)
    prompt_mask = np.zeros((batch_size, max_prompt_len), np.bool_)
    questions: list[str] = []
    golds: list[Gold] = []
    for prompt_index, example in enumerate(examples):
        question = str(example["question"])
        gold = example_gold(example)
        tokens = format_prompt(question, tokenizer, chat=chat)
        if len(tokens) > max_prompt_len:
            raise ValueError(f"prompt has {len(tokens)} tokens, above max_prompt_len={max_prompt_len}")
        for generation_index in range(group_size):
            row = prompt_index * group_size + generation_index
            prompt_ids[row, -len(tokens) :] = tokens
            prompt_mask[row, -len(tokens) :] = True
            questions.append(question)
            golds.append(gold)
    return PromptBatch(prompt_ids, prompt_mask, tuple(questions), tuple(golds))


def eval_subset(dataset: Sequence[Any], *, size: int = 500, seed: int = 0) -> list[Any]:
    """Choose a fixed-size deterministic evaluation subset without mutating the dataset."""

    if size <= 0 or size > len(dataset):
        raise ValueError(f"eval subset size must be in [1, {len(dataset)}]")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return [dataset[index] for index in indices[:size]]


def grpo_prompt_indices(dataset_size: int, prompt_batch_size: int, step: int, *, seed: int = 0) -> list[int]:
    """按候选问题批次的位置还原顺序；step 接收 data_cursor，包含补采后未用于训练的批次。"""

    if not 0 < prompt_batch_size <= dataset_size or step < 0:
        raise ValueError("need 0 < prompt_batch_size <= dataset_size and step >= 0")
    batches_per_epoch = dataset_size // prompt_batch_size
    epoch, batch_index = divmod(step, batches_per_epoch)
    order = list(range(dataset_size))
    random.Random(seed + epoch).shuffle(order)
    start = batch_index * prompt_batch_size
    return order[start : start + prompt_batch_size]


def split_holdout_indices(dataset_size: int, holdout_size: int, *, seed: int = 0) -> tuple[list[int], list[int]]:
    """返回原始题号的剩余集/固定留出集；不重新编号，零留出保持原顺序。"""
    if dataset_size <= 0 or not 0 <= holdout_size < dataset_size:
        raise ValueError("holdout必须在[0, dataset_size)内，剩余集不能为空")
    held_out = eval_subset(list(range(dataset_size)), size=holdout_size, seed=seed) if holdout_size else []
    excluded = set(held_out)
    return [index for index in range(dataset_size) if index not in excluded], held_out


@dataclass
class GSM8KBatchStream(Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    """Epoch-derived shuffle order with a compact, JSON-safe resume state."""

    dataset: Sequence[Mapping[str, Any]]
    tokenizer: Any
    batch_size: int
    sequence_length: int
    pad_token_id: int
    seed: int = 0
    split: str = "train"
    epoch: int = 0
    cursor: int = 0
    _order: list[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.batch_size > len(self.dataset):
            raise ValueError("batch_size must be positive and no larger than the dataset")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        self._order = self._epoch_order(self.epoch)

    def _epoch_order(self, epoch: int) -> list[int]:
        order = list(range(len(self.dataset)))
        random.Random(self.seed + epoch).shuffle(order)
        return order

    def __iter__(self) -> GSM8KBatchStream:
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.cursor + self.batch_size > len(self._order):
            self.epoch += 1
            self.cursor = 0
            self._order = self._epoch_order(self.epoch)
        indices = self._order[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        encoded = [
            encode_sft_example(
                self.tokenizer,
                str(self.dataset[index]["question"]),
                str(self.dataset[index]["answer"]),
                self.sequence_length,
            )
            for index in indices
        ]
        return collate_sft(encoded, self.batch_size, self.sequence_length, self.pad_token_id)

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "gsm8k",
            "version": 1,
            "split": self.split,
            "dataset_size": len(self.dataset),
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "seed": self.seed,
            "epoch": self.epoch,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = self.state_dict()
        for key in ("type", "version", "split", "dataset_size", "batch_size", "sequence_length", "seed"):
            if state.get(key) != expected[key]:
                raise ValueError(f"GSM8K data state {key}={state.get(key)!r}, expected {expected[key]!r}")
        epoch, cursor = int(state.get("epoch", -1)), int(state.get("cursor", -1))
        if epoch < 0 or cursor < 0 or cursor > len(self.dataset) or cursor % self.batch_size:
            raise ValueError(f"invalid GSM8K data position epoch={epoch} cursor={cursor}")
        self.epoch = epoch
        self.cursor = cursor
        self._order = self._epoch_order(epoch)
