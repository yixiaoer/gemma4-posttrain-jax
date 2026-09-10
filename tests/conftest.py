"""Tiny Gemma 4 configurations for CPU-only Hugging Face / JAX comparisons."""

from __future__ import annotations

import jax

jax.config.update("jax_default_matmul_precision", "highest")

import pytest  # noqa: E402
import torch  # noqa: E402
from transformers import Gemma4ForCausalLM, Gemma4TextConfig  # noqa: E402

from gemma4_posttrain_jax.model import config_from_hf  # noqa: E402
from gemma4_posttrain_jax.weights import convert_gemma4_text_params  # noqa: E402

ROPE = {
    "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
    "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0},
}

# Config A: E2B/E4B-like. 6 layers, the last 2 share KV (layer 4 sliding <- layer 3, layer 5 full <- layer 1),
# double-wide MLP on shared layers, per-layer embeddings, distinct head dims for sliding/full layers.
TINY_A: dict = dict(
    vocab_size=97,
    hidden_size=32,
    intermediate_size=48,
    num_hidden_layers=6,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    global_head_dim=16,
    layer_types=[
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
    sliding_window=3,
    num_kv_shared_layers=2,
    use_double_wide_mlp=True,
    hidden_size_per_layer_input=8,
    vocab_size_per_layer_input=97,
    rms_norm_eps=1e-6,
    rope_parameters=ROPE,
    final_logit_softcapping=30.0,
    attention_k_eq_v=False,
    max_position_embeddings=64,
    bos_token_id=2,
    eos_token_id=1,
    pad_token_id=0,
    tie_word_embeddings=True,
)

# Config B: 12B-like. No KV sharing, no PLE, K=V on full-attention layers with their own kv-head count.
TINY_B: dict = {
    **TINY_A,
    "num_kv_shared_layers": 0,
    "use_double_wide_mlp": False,
    "hidden_size_per_layer_input": 0,
    "attention_k_eq_v": True,
    "num_global_key_value_heads": 1,
}


def build_hf_model(
    kwargs: dict,
    seed: int = 0,
    *,
    weight_std: float = 0.2,
    norm_std: float = 0.3,
    layer_scalar_min: float = 0.1,
) -> Gemma4ForCausalLM:
    torch.manual_seed(seed)
    hf_config = Gemma4TextConfig(**kwargs)
    hf_config._attn_implementation = "eager"
    model = Gemma4ForCausalLM(hf_config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or "layernorm" in name:
                p.copy_(1.0 + norm_std * torch.randn_like(p))  # exercise the scale (init is all ones)
            else:
                p.normal_(0.0, weight_std)
        for layer in model.model.layers:
            layer.layer_scalar.copy_(torch.rand(1) * (1.0 - layer_scalar_min) + layer_scalar_min)
    return model


@pytest.fixture(autouse=True)
def deterministic_torch() -> None:
    torch.manual_seed(1234)
    torch.set_grad_enabled(False)


@pytest.fixture(scope="session")
def tiny_a():
    model = build_hf_model(TINY_A, seed=0)
    config = config_from_hf(TINY_A)
    params = convert_gemma4_text_params(model.state_dict().__getitem__, config)
    return model, params, config


@pytest.fixture(scope="session")
def tiny_b():
    model = build_hf_model(TINY_B, seed=1)
    config = config_from_hf(TINY_B)
    params = convert_gemma4_text_params(model.state_dict().__getitem__, config)
    return model, params, config


@pytest.fixture(scope="session")
def grad_tiny():
    """Well-conditioned full-structure tiny model for the stricter backward parity gate."""

    model = build_hf_model(
        TINY_A,
        seed=7,
        weight_std=0.03,
        norm_std=0.05,
        layer_scalar_min=0.8,
    )
    config = config_from_hf(TINY_A)
    params = convert_gemma4_text_params(model.state_dict().__getitem__, config)
    return model, params, config


@pytest.fixture(scope="session")
def tpu_tiny():
    """Tiny A with a four-way divisible vocabulary for the required vocab-axis FSDP layout."""

    kwargs = {**TINY_A, "vocab_size": 96, "vocab_size_per_layer_input": 96}
    model = build_hf_model(kwargs, seed=11)
    config = config_from_hf(kwargs)
    params = convert_gemma4_text_params(model.state_dict().__getitem__, config)
    return model, params, config


@pytest.fixture(scope="session")
def tpu_grad_tiny():
    """沿用梯度 oracle 的温和权重尺度，词表改为四芯片可整除。"""

    kwargs = {**TINY_A, "vocab_size": 96, "vocab_size_per_layer_input": 96}
    model = build_hf_model(kwargs, seed=7, weight_std=0.03, norm_std=0.05, layer_scalar_min=0.8)
    config = config_from_hf(kwargs)
    params = convert_gemma4_text_params(model.state_dict().__getitem__, config)
    return model, params, config
