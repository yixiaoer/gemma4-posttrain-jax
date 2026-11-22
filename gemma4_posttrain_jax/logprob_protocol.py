"""Record the training logprob backend's identity in checkpoints."""

from __future__ import annotations

PALLAS_LOGPROB_PROTOCOL = "v4-native-shared-bf16x3-v1"


def logprob_run_metadata(backend: str) -> dict[str, str]:
    """Preserve legacy JAX settings and identify the specific numerical implementation for Pallas."""
    if backend == "jax":
        return {}
    if backend == "pallas":
        return {"logprob_backend": backend, "logprob_protocol": PALLAS_LOGPROB_PROTOCOL}
    raise ValueError(f"unknown logprob backend: {backend}")
