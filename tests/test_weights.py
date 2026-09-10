"""Weight conversion: layouts, skipped dead weights, round trip."""

from __future__ import annotations

import numpy as np
import torch

from gemma4_posttrain_jax.weights import check_gemma4_text_params, convert_back_gemma4_text_params


def test_shapes_and_shared_layers(tiny_a, tiny_b) -> None:
    for _, params, config in (tiny_a, tiny_b):
        check_gemma4_text_params(params, config)
    _, params, _ = tiny_a
    assert params.layers[4].attention.k_proj is None and params.layers[5].attention.v_proj is None
    assert params.layers[0].attention.k_proj is not None
    _, params_b, _ = tiny_b
    assert params_b.layers[1].attention.v_proj is None and params_b.layers[1].attention.k_proj is not None
    assert params_b.embed_tokens_per_layer is None and params_b.layers[0].per_layer_input is None


def test_layer_scalar_and_norms_converted(tiny_a) -> None:
    model, params, _ = tiny_a
    for i, layer in enumerate(model.model.layers):
        assert np.isclose(float(params.layers[i].layer_scalar), float(layer.layer_scalar))
        np.testing.assert_allclose(np.asarray(params.layers[i].input_norm.weight), layer.input_layernorm.weight.numpy())


def test_round_trip(tiny_a, tiny_b) -> None:
    for model, params, config in (tiny_a, tiny_b):
        back = convert_back_gemma4_text_params(params, config)
        state = model.state_dict()
        skipped = {k for k in state if k.startswith("model.") and k not in back}
        for key in skipped:  # only the dead kv weights of shared layers may be missing
            assert ".self_attn." in key and any(s in key for s in ("k_proj", "v_proj", "k_norm")), key
        for key, value in back.items():
            np.testing.assert_allclose(value, state[key].to(torch.float32).numpy(), atol=1e-6, err_msg=key)
        assert "lm_head.weight" not in back  # tied to embed_tokens
