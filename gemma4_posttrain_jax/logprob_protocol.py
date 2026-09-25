"""训练 logprob 后端的检查点身份。"""

from __future__ import annotations

PALLAS_LOGPROB_PROTOCOL = "v4-native-shared-bf16x3-v1"


def logprob_run_metadata(backend: str) -> dict[str, str]:
    """原生入口保持旧配置；Pallas 同时标明具体数值实现。"""
    if backend == "jax":
        return {}
    if backend == "pallas":
        return {"logprob_backend": backend, "logprob_protocol": PALLAS_LOGPROB_PROTOCOL}
    raise ValueError(f"unknown logprob backend: {backend}")
