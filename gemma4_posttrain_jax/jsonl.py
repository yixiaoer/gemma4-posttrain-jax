"""Read JSONL by physical lines, preserving Unicode line separators inside generated text."""

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    # str.splitlines also splits U+0085/U+2028/U+2029, which JSON permits inside string values.
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
