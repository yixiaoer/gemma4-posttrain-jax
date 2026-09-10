"""固定MATH原始划分、显式去重/去重叠与保持数学符号的gold规范化。"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from .math_gold import MATH_GOLD_CORRECTIONS

MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"

MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"

SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def last_boxed_answer(solution: str) -> str:
    """取最后一个boxed/fbox的平衡内容；转义的集合花括号不改变嵌套深度。"""
    starts = list(re.finditer(r"\\(?:boxed|fbox)\b\s*", solution))
    if not starts:
        raise ValueError("solution没有boxed/fbox答案")
    start = starts[-1].end()
    if start >= len(solution):
        raise ValueError("boxed命令后没有答案")
    if solution[start] != "{":
        # 固定MATH来源有两条合法的单数字TeX参数；不把无括号多token表达式猜作完整答案。
        if solution[start] in "0123456789" and (start + 1 == len(solution) or solution[start + 1] in " $.,;)]}"):
            return solution[start]
        raise ValueError("不支持无花括号的多token答案")
    start += 1
    depth = 1
    escaped = False
    for index in range(start, len(solution)):
        character = solution[index]
        if not escaped:
            depth += (character == "{") - (character == "}")
            if depth == 0:
                answer = solution[start:index].strip()
                if not answer:
                    raise ValueError("boxed答案为空")
                return answer
        escaped = character == "\\" and not escaped
    raise ValueError("最后一个boxed答案的花括号未闭合")


def question_key(problem: str) -> str:
    return hashlib.sha256(" ".join(problem.split()).encode()).hexdigest()


class MathData(NamedTuple):
    dataset: Any
    provenance: dict[str, Any]


def prepare_math_training_rows(
    training: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """排除空gold、所有重复题行和与原test相同的题；每个排除理由均返回供存档。"""
    counts = Counter(question_key(str(row["problem"])) for row in training)
    test_keys = {question_key(str(row["problem"])) for row in test}
    included, excluded = [], []
    for row in training:
        question = str(row["problem"]).strip()
        if not question:
            raise ValueError("MATH存在空问题，固定来源需要重新审计")
        key = question_key(question)
        reasons = []
        if counts[key] > 1:
            reasons.append("duplicate_train_question")
        if key in test_keys:
            reasons.append("exact_original_test_overlap")
        try:
            answer = last_boxed_answer(str(row["solution"]))
        except ValueError as error:
            answer = ""
            reasons.append("invalid_gold: " + str(error))
        correction = MATH_GOLD_CORRECTIONS.get(str(row["source_id"]))
        if correction is not None:
            solution_sha = hashlib.sha256(str(row["solution"]).encode()).hexdigest()
            if key != correction.question_sha256 or solution_sha != correction.solution_sha256:
                raise ValueError(f"MATH审计修正来源哈希失配: {row['source_id']}")
            if not correction.answers:
                reasons.append(correction.reason)
        if reasons:
            excluded.append({"source_id": row["source_id"], "question_sha256": key, "reasons": reasons})
            continue
        included.append(
            {
                "source_id": row["source_id"],
                "question": question,
                "answer": str(row["solution"]),
                "gold": correction.answers[0] if correction else r"\boxed{" + answer + "}",
                "gold_alternatives": list(correction.answers[1:]) if correction else [],
                "gold_audit": correction.reason if correction else "last-boxed-v1",
                "question_sha256": key,
                "level": str(row.get("level", "")),
                "subject": str(row.get("type", "")),
            }
        )
    return included, excluded


def _load_original_rows(split: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    from datasets import load_dataset

    rows: list[dict[str, Any]] = []
    fingerprints: dict[str, str] = {}
    for subject in SUBJECTS:
        dataset = load_dataset("EleutherAI/hendrycks_math", subject, split=split, revision=MATH_REVISION)
        fingerprints[f"{split}/{subject}"] = dataset._fingerprint
        rows.extend({**row, "source_id": f"{split}/{subject}/row_{index}"} for index, row in enumerate(dataset))
    return rows, fingerprints


def load_math_training_data() -> MathData:
    """仅使用原始train7500；test问题只用于完全相同题排除，不使用其答案生成训练标签。"""
    from datasets import Dataset

    training, train_fingerprints = _load_original_rows("train")
    test, test_fingerprints = _load_original_rows("test")
    if (len(training), len(test)) != (7500, 5000):
        raise ValueError("固定MATH原始train/test数量不符合7500/5000")
    rows, excluded = prepare_math_training_rows(training, test)
    if not rows:
        raise ValueError("MATH清洗后没有可训练问题")
    dataset = Dataset.from_list(rows)
    provenance = {
        "source": "EleutherAI/hendrycks_math",
        "revision": MATH_REVISION,
        "protocol": "original-train-clean-audited-gold-v3",
        "gold_protocol": "last-boxed-with-source-bound-semantic-corrections-v3",
        "gold_corrections": {
            key: {**value._asdict(), "answers": list(value.answers)} for key, value in MATH_GOLD_CORRECTIONS.items()
        },
        "raw_train_count": len(training),
        "usable_count": len(rows),
        "source_fingerprints": {**train_fingerprints, **test_fingerprints},
        "excluded": excluded,
        "ordered_source_ids_sha256": hashlib.sha256(
            ("\n".join(row["source_id"] for row in rows) + "\n").encode()
        ).hexdigest(),
        "scope": "只去除空白规范化后完全相同题，不证明语义或预训练去污染；后续dev须从清洗后的训练题中排除。",
    }
    return MathData(dataset, provenance)


def load_math500_data() -> MathData:
    """保留500条原始题号及带boxed的符号gold；加载本身不执行模型评估。"""
    from datasets import Dataset, load_dataset

    source = load_dataset("HuggingFaceH4/MATH-500", split="test", revision=MATH500_REVISION)
    rows = []
    for row in source:
        answer = row["answer"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("MATH500包含空或非字符串gold")
        rows.append(
            {
                "source_id": row["unique_id"],
                "question": row["problem"],
                "answer": row["solution"],
                "gold": r"\boxed{" + answer.strip() + "}",
                "question_sha256": question_key(row["problem"]),
                "level": str(row["level"]),
                "subject": row["subject"],
            }
        )
    identities = [row["source_id"] for row in rows]
    if len(rows) != 500 or len(set(identities)) != 500 or not all(value.startswith("test/") for value in identities):
        raise ValueError("MATH500数量或原始题号不符合固定协议")
    return MathData(
        Dataset.from_list(rows),
        {
            "source": "HuggingFaceH4/MATH-500",
            "revision": MATH500_REVISION,
            "fingerprint": source._fingerprint,
            "ordered_source_ids_sha256": hashlib.sha256(("\n".join(identities) + "\n").encode()).hexdigest(),
            "gold_protocol": "provided-answer-boxed-v1",
        },
    )
