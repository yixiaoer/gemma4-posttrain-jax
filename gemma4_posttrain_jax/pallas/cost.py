"""Cost estimates for the decode-path Pallas kernels.

XLA cannot see inside a Pallas kernel. Its scheduler and memory-space assignment (which decides what to
prefetch into on-chip memory while other ops run) get the kernel's cost only from ``pl.CostEstimate``.
The estimates are off unless ``GEMMA4_PALLAS_COST_ESTIMATES=1``: on the TP decode loop they moved the time per
step by -1.5% to +13% depending on the program, and they change the rounding of the sampling program.
"""

from __future__ import annotations

import os

from jax.experimental import pallas as pl


def estimate(
    *, flops: int, bytes_accessed: int, transcendentals: int = 0, remote_bytes: int = 0
) -> pl.CostEstimate | None:
    if os.environ.get("GEMMA4_PALLAS_COST_ESTIMATES", "0") != "1":
        return None
    return pl.CostEstimate(
        flops=int(flops),
        transcendentals=int(transcendentals),
        bytes_accessed=int(bytes_accessed),
        remote_bytes_transferred=int(remote_bytes),
    )
