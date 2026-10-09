#!/usr/bin/env python3
"""Check the GSM8K splits on CPU before training: overlapping questions, unparsable gold answers, prompt length.

Only the tokenizer is loaded; no model weights and no TPU. The splits are the ones the training and
evaluation scripts use: --dev-size/--dev-seed hold out dev questions from the training set, and the
500 test questions drawn with seed 0 are test-monitor while the rest are test-unused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gemma4_posttrain_jax.data import format_prompt, load_gsm8k, normalize_gsm8k_answer, split_holdout_indices

# Two questions count as duplicates only if they are identical after normalizing whitespace;
# a reworded copy of the same problem is not detected.
OVERLAP_PAIRS = (
    ("train", "dev"),
    ("train", "test_monitor"),
    ("train", "test_unused"),
    ("dev", "test_monitor"),
    ("dev", "test_unused"),
    ("test_monitor", "test_unused"),
)


def has_gold(answer: str) -> bool:
    try:
        return bool(normalize_gsm8k_answer(answer)[1])
    except ValueError:
        return False


def check_partitions(
    partitions: dict[str, tuple[Any, list[int]]], tokenizer: Any, *, max_prompt_len: int, chat: bool
) -> dict[str, Any]:
    """Prompt lengths, unparsable answers and duplicates per split, then identical questions across splits."""
    questions: dict[str, set[str]] = {}
    summary: dict[str, Any] = {}
    passed = True
    for name, (dataset, indices) in partitions.items():
        lengths = np.asarray([len(format_prompt(str(dataset[i]["question"]), tokenizer, chat=chat)) for i in indices])
        normalized = [" ".join(str(dataset[i]["question"]).split()) for i in indices]
        questions[name] = set(normalized)
        too_long = [i for i, length in zip(indices, lengths, strict=True) if length > max_prompt_len]
        no_gold = [i for i in indices if not has_gold(str(dataset[i]["answer"]))]
        summary[name] = {
            "count": len(indices),
            "max_prompt_tokens": int(lengths.max()),
            "mean_prompt_tokens": float(lengths.mean()),
            "p99_prompt_tokens": float(np.quantile(lengths, 0.99)),
            "over_length_ids": too_long,
            "missing_gold_ids": no_gold,
            "duplicate_questions": len(normalized) - len(questions[name]),
        }
        passed &= not too_long and not no_gold and bool((lengths > 0).all())
    overlaps = {f"{left}__{right}": len(questions[left] & questions[right]) for left, right in OVERLAP_PAIRS}
    passed = passed and not any(overlaps.values())
    return {"partitions": summary, "identical_question_overlaps": overlaps, "passed": passed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", type=Path, required=True, help="used only to load the tokenizer")
    parser.add_argument("--dev-size", type=int, default=500)
    parser.add_argument("--dev-seed", type=int, default=0)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--prompt-style", choices=("chat", "plain"), default="chat")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dev_size <= 0 or args.max_prompt_len <= 0:
        raise ValueError("--dev-size and --max-prompt-len must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train, test = load_gsm8k("train"), load_gsm8k("test")
    train_indices, dev_indices = split_holdout_indices(len(train), args.dev_size, seed=args.dev_seed)
    test_unused, test_monitor = split_holdout_indices(len(test), 500, seed=0)
    result = {
        "model_path": str(args.model_path.resolve()),
        "dev_size": args.dev_size,
        "dev_seed": args.dev_seed,
        "prompt_style": args.prompt_style,
        "max_prompt_len": args.max_prompt_len,
        "train_fingerprint": train._fingerprint,
        "test_fingerprint": test._fingerprint,
        **check_partitions(
            {
                "train": (train, train_indices),
                "dev": (train, dev_indices),
                "test_monitor": (test, test_monitor),
                "test_unused": (test, test_unused),
            },
            tokenizer,
            max_prompt_len=args.max_prompt_len,
            chat=args.prompt_style == "chat",
        ),
        "scope": "Only identical questions are detected; reworded copies and leakage into pretraining data are not.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("passed", "identical_question_overlaps")}, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(f"data check failed; see {args.output}")


if __name__ == "__main__":
    main()
