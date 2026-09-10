"""按物理行读取JSONL，保留生成文本内部的Unicode行分隔符。"""

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    # str.splitlines还会拆U+0085/U+2028/U+2029，而JSON允许它们出现在字符串内部。
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
