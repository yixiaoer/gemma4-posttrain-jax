"""真实plain输出中的U+2028不能被拆成额外JSON记录。"""

import json
from pathlib import Path

import pytest

from gemma4_posttrain_jax.jsonl import read_jsonl


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_preserve_unicode_separators_in_json_strings(tmp_path: Path, separator: str, newline: str) -> None:
    rows = [
        {"dataset_index": 7, "completion": f"first{separator}second", "task_success": 1},
        {"dataset_index": 9, "completion": "a\\nb\nc", "task_success": 0},
    ]
    raw = newline.join(json.dumps(row, ensure_ascii=False) for row in rows).encode("utf-8")
    path = tmp_path / "predictions.jsonl"
    path.write_bytes(raw)
    assert read_jsonl(path) == rows
    assert path.read_bytes() == raw


def test_malformed_physical_record_is_still_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"completion":"unfinished\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)
