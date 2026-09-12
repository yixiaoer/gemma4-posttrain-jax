#!/usr/bin/env python3
"""在小输入上比较自动分片和词表并行的 loss、梯度与实际 HLO。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from gemma4_posttrain_jax import losses
from gemma4_posttrain_jax.diagnostics import optional_package_version
from gemma4_posttrain_jax.losses import per_token_logps
from gemma4_posttrain_jax.sharding import DATA_AXIS, make_mesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "tpu"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-ir", action="store_true")
    args = parser.parse_args()
    if jax.default_backend() != args.backend or len(jax.devices()) != 4:
        raise ValueError("要求声明的 backend 和四设备；CPU 请在启动前设置四个虚拟设备")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    mesh = make_mesh()
    replicated = NamedSharding(mesh, P())
    sharded = NamedSharding(mesh, P(DATA_AXIS, None))
    rng = np.random.default_rng(20260903)
    embed = jax.device_put(rng.normal(0, 0.2, (1024, 64)).astype(np.float32), sharded)
    hidden = jax.device_put(rng.normal(0, 0.2, (4, 8, 64)).astype(np.float32), replicated)
    targets = jax.device_put(rng.integers(0, 1024, (4, 8), dtype=np.int32), replicated)
    results, outputs = {}, {}
    for mode in ("dense", "automatic", "vocab_parallel"):

        def loss(e, h, ids, implementation=mode):
            if implementation == "dense":
                logits = jnp.einsum("btm,vm->btv", h, e, precision=jax.lax.Precision.HIGHEST)
                logps = jnp.take_along_axis(jax.nn.log_softmax(logits), ids[..., None], axis=-1)[..., 0]
            else:
                logps = per_token_logps(
                    e,
                    h,
                    ids,
                    softcap=None,
                    vocab_chunk=128,
                    sequence_chunk=8,
                    mesh=mesh if implementation == "vocab_parallel" else None,
                )
            return -logps.mean()

        lowered = jax.jit(
            jax.value_and_grad(loss, argnums=(0, 1)),
            in_shardings=(sharded, replicated, replicated),
            out_shardings=(replicated, (sharded, replicated)),
        ).lower(embed, hidden, targets)
        compiled = lowered.compile()
        hlo = compiled.as_text()
        if hlo is None:
            raise ValueError("当前 backend 未返回优化 HLO")
        if args.save_ir:
            (args.output_dir / f"{mode}.stablehlo.txt").write_text(lowered.as_text())
            (args.output_dir / f"{mode}.hlo.txt").write_text(hlo)
        value, gradients = jax.device_get(compiled(embed, hidden, targets))
        outputs[mode] = (value, gradients)
        # 仅匹配指令定义；操作数引用和 metadata 中的名字不能重复计数。
        gather = [line.strip() for line in hlo.splitlines() if re.search(r"= .*? all-gather\(", line)]
        full_embed = []
        for line in gather:
            shape = re.search(r"= (?:bf16|f32)\[([0-9,]+)\]", line)
            if shape and int(np.prod([int(n) for n in shape[1].split(",")])) == embed.size:
                full_embed.append(line)
        results[mode] = dict(
            loss=float(value),
            all_gather_instructions=len(gather),
            embedding_sized_all_gather=len(full_embed),
            embedding_sized_all_gather_lines=full_embed,
        )
    reference_value, reference_grads = outputs["dense"]
    for mode in ("automatic", "vocab_parallel"):
        value, gradients = outputs[mode]
        comparisons = []
        for name, actual, expected in zip(("embedding", "hidden"), gradients, reference_grads, strict=True):
            actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
            relative_l2 = float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))
            if not np.isfinite(relative_l2) or relative_l2 > 1e-4:
                raise ValueError(f"{mode}/{name} 梯度误差超过 1e-4: {relative_l2}")
            comparisons.append(dict(name=name, relative_l2=relative_l2))
        np.testing.assert_allclose(value, reference_value, atol=1e-5, rtol=0)
        results[mode]["gradient_comparison"] = comparisons
    report = dict(
        complete=True,
        backend=jax.default_backend(),
        devices=[d.device_kind for d in jax.devices()],
        jax=jax.__version__,
        jaxlib=optional_package_version("jaxlib"),
        libtpu=optional_package_version("libtpu"),
        numpy=np.__version__,
        matmul_precision=jax.config.jax_default_matmul_precision or "default",
        config=dict(vocab=1024, hidden=64, batch=4, sequence=8, dtype="float32", softcap=None, seed=20260903),
        source_sha256=hashlib.sha256(Path(losses.__file__).read_bytes()).hexdigest(),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        results=results,
        scope="小输入、输出头 loss 和梯度；无完整模型或 Adam，不复现历史 22.676 倍训练收益。",
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (args.output_dir / "summary.json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
