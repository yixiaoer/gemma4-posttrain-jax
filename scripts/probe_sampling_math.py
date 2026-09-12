#!/usr/bin/env python3
"""用固定 logits 比较旧归一化和当前采样器，按需生成编译图。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from gemma4_posttrain_jax import sampler
from gemma4_posttrain_jax.diagnostics import optional_package_version


def legacy_sample(logits: Array, key: Array, *, temperature: float, top_k: int) -> tuple[Array, Array]:
    """保留旧采样函数的计算顺序，作为独立对照；不支持 Top-p。"""
    with jax.named_scope("rollout_policy"):
        logits32 = logits.astype(jnp.float32)
        full_logps = jax.nn.log_softmax(logits32, axis=-1)
        vocab_size = logits.shape[-1]
        if temperature <= 0:
            token = jnp.argmax(logits32, axis=-1).astype(jnp.int32)
        elif 0 < top_k < vocab_size:
            top_values, top_indices = lax.top_k(logits32 / temperature, top_k)
            selected = jax.random.categorical(key, top_values, axis=-1)
            token = jnp.take_along_axis(top_indices, selected[:, None], axis=-1)[:, 0]
        else:
            token = jax.random.categorical(key, logits32 / temperature, axis=-1).astype(jnp.int32)
        logp = jnp.take_along_axis(full_logps, token[:, None], axis=-1)[:, 0]
        return token, logp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "tpu"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-ir", action="store_true", help="额外保存 StableHLO 和优化后的 HLO，仅供本次分析")
    args = parser.parse_args()
    if jax.default_backend() != args.backend:
        raise ValueError("实际 backend 不符；CPU 实验须在启动前设置 JAX_PLATFORMS=cpu")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    generator = np.random.default_rng(20260912)
    results = []
    for vocabulary, batch_size in ((97, 8), (262144, 2)):
        logits = generator.normal(0, 3, (batch_size, vocabulary)).astype(np.float32)
        shifted = logits.astype(np.float64) - np.max(logits.astype(np.float64), axis=-1, keepdims=True)
        reference = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        for temperature, top_k in ((0.0, 0), (1.0, 0), (0.8, 7)):
            name = f"v{vocabulary}_t{temperature}_k{top_k}"
            row: dict[str, Any] = dict(case=name, shape=list(logits.shape), temperature=temperature, top_k=top_k)
            row["input_sha256"] = hashlib.sha256(logits.astype("<f4").tobytes()).hexdigest()
            for label, function in (("legacy", legacy_sample), ("current", sampler._sample_token)):

                def run(values, key, fn=function, t=temperature, k=top_k):
                    return fn(values, key, temperature=t, top_k=k)

                values, key = jax.device_put(logits), jax.random.PRNGKey(123)
                jax.block_until_ready((values, key))
                lowered = jax.jit(run).lower(values, key)
                compiled = lowered.compile()
                hlo = compiled.as_text()
                if hlo is None:
                    raise ValueError("当前 backend 未返回优化 HLO")
                if args.save_ir:
                    (args.output_dir / f"{name}_{label}.stablehlo.txt").write_text(lowered.as_text())
                    (args.output_dir / f"{name}_{label}.hlo.txt").write_text(hlo)
                tokens, logps = jax.device_get(compiled(values, key))
                delta = logps.astype(np.float64) - reference[np.arange(batch_size), tokens]
                row[label] = dict(
                    tokens=tokens.tolist(),
                    logps=logps.tolist(),
                    max_abs_error=float(np.max(np.abs(delta))),
                    mean_abs_error=float(np.mean(np.abs(delta))),
                    finite=bool(np.all(np.isfinite(logps))),
                    highest_accuracy_attributes=hlo.count("result_accuracy={mode=highest}"),
                )
            row["tokens_identical"] = row["legacy"]["tokens"] == row["current"]["tokens"]
            row["raw_logps_identical"] = (
                np.asarray(row["legacy"]["logps"], np.float32).tobytes()
                == np.asarray(row["current"]["logps"], np.float32).tobytes()
            )
            results.append(row)
    report = dict(
        schema="sampling-math-final-v2",
        complete=True,
        backend=jax.default_backend(),
        devices=[dict(id=d.id, kind=d.device_kind) for d in jax.devices()],
        versions={
            **{name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "numpy")},
            "libtpu": optional_package_version("libtpu"),
        },
        input_recipe=dict(generator="numpy.default_rng/PCG64", seed=20260912, std=3, dtype="float32", key=123),
        matmul_precision=jax.config.jax_default_matmul_precision or "default",
        source_sha256=hashlib.sha256(Path(sampler.__file__).read_bytes()).hexdigest(),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="固定 logits 的数值对照；不测吞吐，不加载模型，不证明训练质量或恢复一致。",
        results=results,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (args.output_dir / "summary.json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
