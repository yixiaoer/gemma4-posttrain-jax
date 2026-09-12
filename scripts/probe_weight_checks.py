#!/usr/bin/env python3
"""比较 BF16 数值检查与位检查，验证特殊值并交错测量 C/F 布局。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from gemma4_posttrain_jax import inference_weights
from gemma4_posttrain_jax.inference_weights import bf16_bits_equal, bf16_finite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--columns", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=4, help="每种实现的测量次数，须为正偶数")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows, args.columns, args.repeats) < 1 or args.repeats % 2:
        raise ValueError("形状必须为正，repeats 须为正偶数")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    codes = np.arange(65536, dtype=np.uint16)
    values = codes.view(bfloat16)
    with np.errstate(invalid="ignore"):
        finite = np.isfinite(values)
    np.testing.assert_array_equal((codes & 0x7F80) != 0x7F80, finite)
    # 逐一调用产品函数，避免只测试重新写出的位公式。
    np.testing.assert_array_equal([bf16_finite(values[i : i + 1]) for i in range(len(values))], finite)
    positive_zero = np.array([0], dtype=np.uint16).view(bfloat16)
    negative_zero = np.array([0x8000], dtype=np.uint16).view(bfloat16)
    nan = np.array([0x7FC0], dtype=np.uint16).view(bfloat16)
    checks = dict(
        exhaustive_finite_codes=len(values),
        signed_zero_numeric_equal=bool(np.array_equal(positive_zero, negative_zero)),
        signed_zero_bits_equal=bf16_bits_equal(positive_zero, negative_zero),
        nan_numeric_equal=bool(np.array_equal(nan, nan)),
        nan_bits_equal=bf16_bits_equal(nan, nan),
    )
    if not checks["signed_zero_numeric_equal"] or checks["signed_zero_bits_equal"]:
        raise ValueError("正负零控制结果不符")
    if checks["nan_numeric_equal"] or not checks["nan_bits_equal"]:
        raise ValueError("NaN 控制结果不符")

    source = np.random.default_rng(20260910).normal(size=(args.rows, args.columns)).astype(bfloat16)
    expected = source.copy(order="C")
    results = []
    for layout in ("C", "F"):
        actual = source.copy(order=layout)
        # F/C 按同一逻辑索引比较，不能直接比较两块内存的物理字节顺序。
        if not bf16_bits_equal(actual, expected):
            raise ValueError("布局转换改变内容")
        changed = expected.copy()
        changed.view(np.uint16)[0, 0] ^= 1
        if bf16_bits_equal(actual, changed):
            raise ValueError("未检测到单 bit 改变")
        methods = {
            "finite": {
                "numeric": lambda value=actual: bool(np.isfinite(value).all()),
                "bits": lambda value=actual: bf16_finite(value),
            },
            "equal": {
                "numeric": lambda value=actual: bool(np.array_equal(value, expected)),
                "bits": lambda value=actual: bf16_bits_equal(value, expected),
            },
        }
        for operation, functions in methods.items():
            for function in functions.values():
                if not function():
                    raise ValueError("预热检查失败")
            samples: dict[str, list[float]] = {"numeric": [], "bits": []}
            for _ in range(args.repeats // 2):
                for name in ("numeric", "bits", "bits", "numeric"):
                    start = time.perf_counter_ns()
                    passed = functions[name]()
                    duration_ms = (time.perf_counter_ns() - start) / 1e6
                    if not passed:
                        raise ValueError("计时期间检查失败")
                    samples[name].append(duration_ms)
            results.append(
                dict(
                    actual_layout=layout,
                    expected_layout="C",
                    operation=operation,
                    iterations_ms=samples,
                    median_ms={name: statistics.median(data) for name, data in samples.items()},
                )
            )
    report = dict(
        complete=True,
        checks=checks,
        shape=list(source.shape),
        bytes_per_array=source.nbytes,
        platform=platform.platform(),
        machine=platform.machine(),
        versions={name: importlib.metadata.version(name) for name in ("numpy", "ml_dtypes")},
        source_sha256=hashlib.sha256(Path(inference_weights.__file__).read_bytes()).hexdigest(),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sequence="预热后重复 A/B/B/A，每种实现各 repeats 次",
        results=results,
        scope="主机 BF16 检查微实验，不包含权重传输、引擎更新或训练。",
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (args.output_dir / "summary.json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
