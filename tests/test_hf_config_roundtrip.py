"""HF保存配置的逐层attention覆盖必须在JAX重新加载后保持权重、shape和前向语义。"""

from __future__ import annotations

import copy
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gemma4_posttrain_jax.model import config_from_hf, forward_gemma4_lm, layer_head_dim, layer_kv_heads
from gemma4_posttrain_jax.weights import load_hf_params


@pytest.mark.parametrize("fixture_name", ["tiny_a", "tiny_b"])
def test_hf_save_reload_preserves_parameters_and_forward(request, fixture_name, tmp_path):
    model, original, config = request.getfixturevalue(fixture_name)
    model.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert "global_head_dim" not in saved and "per_layer_config" in saved
    reloaded, parsed = load_hf_params(str(tmp_path), dtype=jnp.float32)
    assert parsed == config
    assert jax.tree.structure(reloaded) == jax.tree.structure(original)
    for actual, expected in zip(jax.tree.leaves(reloaded), jax.tree.leaves(original), strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    ids = jnp.array([[2, 11, 12, 13], [2, 21, 22, 23]], dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(4), ids.shape)
    function = jax.jit(lambda params: forward_gemma4_lm(params, ids, positions, config=config).logits)
    np.testing.assert_array_equal(np.asarray(function(reloaded)), np.asarray(function(original)))


def test_explicit_layer_configuration_overrides_legacy_global_field(tiny_a):
    model, _, expected = tiny_a
    saved = model.config.to_dict()
    saved["global_head_dim"] = 999
    assert config_from_hf(saved) == expected
    nested = {"text_config": saved}
    assert config_from_hf(nested) == expected


def test_explicit_empty_layer_configuration_uses_global_head_dim_not_legacy_alias(tiny_a):
    model, _, _ = tiny_a
    saved = model.config.to_dict()
    saved["per_layer_config"] = {}
    saved["global_head_dim"] = 999
    parsed = config_from_hf(saved)
    assert parsed.global_head_dim == parsed.head_dim == saved["head_dim"]


@pytest.mark.parametrize(
    "case", ["different_full_width", "unsupported_norm", "index_out_of_range", "kv_override_without_k_eq_v"]
)
def test_unrepresentable_layer_overrides_fail_closed(tiny_a, case):
    model, _, _ = tiny_a
    saved = copy.deepcopy(model.config.to_dict())
    overrides = {str(key): value for key, value in saved["per_layer_config"].items()}
    saved["per_layer_config"] = overrides
    if case == "different_full_width":
        overrides["5"]["head_dim"] = 32
    elif case == "unsupported_norm":
        overrides["1"]["rms_norm_eps"] = 1e-3
    elif case == "index_out_of_range":
        overrides["6"] = {"head_dim": 16}
    else:
        for value in overrides.values():
            value["num_key_value_heads"] = 1
    with pytest.raises(ValueError):
        config_from_hf(saved)


def test_equal_explicit_global_kv_count_survives_hf_serialization(tiny_b):
    model, _, _ = tiny_b
    legacy = model.config.to_dict()
    legacy.pop("per_layer_config")
    legacy["global_head_dim"] = 16
    legacy["num_global_key_value_heads"] = legacy["num_key_value_heads"]
    serialized = type(model.config)(**legacy).to_dict()
    parsed, expected = config_from_hf(serialized), config_from_hf(legacy)
    # HF稀疏序列化会删除与全局值相同的冗余覆盖；比较实际逐层维度，而非已删除的字段身份。
    assert parsed == expected._replace(num_global_key_value_heads=None)
    for index in range(parsed.num_hidden_layers):
        assert layer_head_dim(parsed, index) == layer_head_dim(expected, index)
        assert layer_kv_heads(parsed, index) == layer_kv_heads(expected, index)
