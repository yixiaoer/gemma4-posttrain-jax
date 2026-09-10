"""Verify the TPU v4-8 JAX environment: versions, devices, HBM, sharded matmul, Pallas.

Run with the project venv while no other process holds the TPU:

    .venv/bin/python scripts/check_env.py
"""

from __future__ import annotations

import importlib.metadata as md

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def main() -> None:
    for name in ("jax", "jaxlib", "libtpu", "optax", "einops", "transformers", "torch"):
        try:
            print(f"{name:13s} {md.version(name)}")
        except md.PackageNotFoundError:
            print(f"{name:13s} (not installed)")

    devices = jax.devices()
    print("backend      ", jax.default_backend())
    print("device_count ", len(devices), devices[0].device_kind)
    stats = devices[0].memory_stats() or {}
    print("hbm_limit    ", round(stats.get("bytes_limit", 0) / 2**30, 1), "GiB per device")

    mesh = Mesh(devices, ("d",))
    x = jax.device_put(jnp.ones((8 * len(devices), 128), jnp.bfloat16), NamedSharding(mesh, P("d", None)))
    w = jax.device_put(jnp.ones((128, 256), jnp.bfloat16), NamedSharding(mesh, P(None, "d")))
    y = jax.jit(lambda a, b: (a @ b).astype(jnp.float32).sum())(x, w)
    assert float(y) == 128.0 * 256 * 8 * len(devices), float(y)
    print("sharded matmul OK")

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2

    z = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32))(jnp.ones((8, 128), jnp.float32))
    assert float(z.sum()) == 2048.0
    print("pallas_call  OK")


if __name__ == "__main__":
    main()
