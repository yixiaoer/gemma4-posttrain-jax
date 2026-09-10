"""Per-module CPU comparisons of the JAX Gemma 4 text model against transformers 5.16.1."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from gemma4_posttrain_jax.model import (
    FULL,
    SLIDING,
    forward_attention,
    forward_gemma4_lm,
    forward_gemma4_text,
    forward_mlp,
    forward_per_layer_inputs,
    forward_rms_norm,
    init_kv_cache,
    is_kv_shared,
    kv_source_layer,
    layer_head_dim,
    layer_intermediate_size,
    layer_kv_heads,
    make_qk_masks,
    make_rotary_values,
)


def as_jax(t: torch.Tensor) -> jnp.ndarray:
    return jnp.asarray(t.detach().cpu().numpy())


def assert_close(actual, expected: torch.Tensor, rel: float = 2e-5, mask=None) -> None:
    """Absolute tolerance scaled by the magnitude of ``expected`` (float32 rounding across frameworks)."""

    a = np.asarray(actual, dtype=np.float32)
    e = expected.detach().cpu().numpy().astype(np.float32)
    if mask is not None:
        a, e = a[mask], e[mask]
    np.testing.assert_allclose(a, e, atol=rel * max(1.0, float(np.abs(e).max())), rtol=0.0)


def additive_mask(bool_mask: jnp.ndarray) -> torch.Tensor:
    """Our ``[B, 1, 1, S, T]`` boolean mask -> HF eager ``[B, 1, S, T]`` float mask."""

    m = torch.from_numpy(np.array(bool_mask)[:, 0])
    return torch.where(m, 0.0, torch.finfo(torch.float32).min)


def test_config_derivations(tiny_a, tiny_b) -> None:
    for model, _, config in (tiny_a, tiny_b):
        for i, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            assert is_kv_shared(config, i) == attn.is_kv_shared_layer
            assert layer_head_dim(config, i) == attn.head_dim
            assert layer_kv_heads(config, i) == config.num_attention_heads // attn.num_key_value_groups
            assert layer_intermediate_size(config, i) == layer.mlp.intermediate_size
            has_v = getattr(attn, "v_proj", None) is not None
            assert has_v == (not attn.is_kv_shared_layer and not attn.use_alternative_attention)
    _, _, config = tiny_a
    assert kv_source_layer(config, 4) == 3 and kv_source_layer(config, 5) == 1


def test_rms_norm(tiny_a) -> None:
    model, params, config = tiny_a
    x = torch.randn(2, 5, config.hidden_size)
    assert_close(
        forward_rms_norm(params.layers[0].input_norm, as_jax(x), config.rms_norm_eps),
        model.model.layers[0].input_layernorm(x),
    )
    q = torch.randn(2, 5, 4, layer_head_dim(config, 1))
    attn = model.model.layers[1].self_attn
    assert_close(forward_rms_norm(params.layers[1].attention.q_norm, as_jax(q), config.rms_norm_eps), attn.q_norm(q))
    assert_close(forward_rms_norm(None, as_jax(q), config.rms_norm_eps), attn.v_norm(q))


@pytest.mark.parametrize("layer_type", [SLIDING, FULL])
def test_rotary_values(tiny_a, layer_type: str) -> None:
    model, _, config = tiny_a
    position_ids = torch.tensor([[0, 1, 2, 3, 7], [3, 4, 5, 6, 40]])
    dummy = torch.zeros(2, 5, config.hidden_size)
    cos, sin = model.model.rotary_emb(dummy, position_ids, layer_type)
    ours = make_rotary_values(config, as_jax(position_ids))[layer_type]
    assert_close(ours.cos, cos)
    assert_close(ours.sin, sin)


@pytest.mark.parametrize("layer_index", [0, 4])
def test_mlp(tiny_a, layer_index: int) -> None:
    model, params, config = tiny_a
    x = torch.randn(2, 3, config.hidden_size)
    assert_close(forward_mlp(params.layers[layer_index].mlp, as_jax(x)), model.model.layers[layer_index].mlp(x))


def test_per_layer_inputs(tiny_a) -> None:
    model, params, config = tiny_a
    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    embeds = model.model.embed_tokens(input_ids)
    expected = model.model.project_per_layer_inputs(embeds, model.model.get_per_layer_inputs(input_ids, None))
    ours = forward_per_layer_inputs(params, config, as_jax(input_ids), as_jax(embeds))
    assert_close(ours, expected)


def _run_hf_attention(
    model, layer_index: int, x: torch.Tensor, position_ids: torch.Tensor, mask: torch.Tensor, shared: dict
):
    attn = model.model.layers[layer_index].self_attn
    layer_type = model.config.layer_types[layer_index]
    pos_emb = model.model.rotary_emb(x, position_ids, layer_type)
    out, _ = attn(hidden_states=x, position_embeddings=pos_emb, attention_mask=mask, shared_kv_states=shared)
    return out


@pytest.mark.parametrize("fixture_name, layer_indices", [("tiny_a", [0, 1, 3, 4, 5]), ("tiny_b", [0, 1])])
def test_attention(request, fixture_name: str, layer_indices: list[int]) -> None:
    """Sliding, full, kv-shared (4 <- 3, 5 <- 1) and K=V (tiny_b layer 1) attention layers."""

    model, params, config = request.getfixturevalue(fixture_name)
    B, S = 2, 6
    x = torch.randn(B, S, config.hidden_size)
    position_ids = torch.arange(S).expand(B, S)
    key_mask = jnp.ones((B, S), jnp.bool_)
    masks = make_qk_masks(key_mask, config.sliding_window, query_start=0, num_queries=S)
    rotary = make_rotary_values(config, as_jax(position_ids))
    hf_shared: dict = {}
    our_shared: dict = {}
    for i in range(config.num_hidden_layers):  # run in order so shared layers see their source K/V
        layer_type = config.layer_types[i]
        expected = _run_hf_attention(model, i, x, position_ids, additive_mask(masks[layer_type]), hf_shared)
        ours, _ = forward_attention(
            params.layers[i].attention,
            as_jax(x),
            masks[layer_type],
            rotary[layer_type],
            layer_index=i,
            config=config,
            shared_kv=our_shared,
        )
        if i in layer_indices:
            assert_close(ours, expected)


def test_decoder_layers_and_final_hidden(tiny_a) -> None:
    model, params, config = tiny_a
    B, S = 2, 7
    input_ids = torch.randint(0, config.vocab_size, (B, S))
    attention_mask = torch.ones(B, S, dtype=torch.long)
    position_ids = torch.arange(S).expand(B, S)
    out = model.model(
        input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_hidden_states=True
    )
    hs = out.hidden_states
    assert len(hs) == config.num_hidden_layers + 1
    layer_outputs: list = []
    hidden, _ = forward_gemma4_text(
        params,
        as_jax(input_ids),
        as_jax(position_ids),
        config=config,
        attention_mask=as_jax(attention_mask).astype(bool),
        layer_outputs=layer_outputs,
    )
    for i, ours in enumerate(layer_outputs[:-1]):  # hs[-1] is the final-normed output, not layer L-1
        assert_close(ours, hs[i + 1], rel=1e-4)
    assert_close(hidden, out.last_hidden_state, rel=1e-4)
    assert_close(hidden, hs[-1], rel=1e-4)


@pytest.mark.parametrize("fixture_name", ["tiny_a", "tiny_b"])
def test_lm_logits_with_left_padding(request, fixture_name: str) -> None:
    model, params, config = request.getfixturevalue(fixture_name)
    B, S = 2, 8
    input_ids = torch.randint(3, config.vocab_size, (B, S))
    attention_mask = torch.ones(B, S, dtype=torch.long)
    attention_mask[1, :3] = 0
    input_ids[1, :3] = config.pad_token_id
    position_ids = torch.arange(S).expand(B, S)
    expected = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids).logits
    ours = forward_gemma4_lm(
        params,
        as_jax(input_ids),
        as_jax(position_ids),
        config=config,
        attention_mask=as_jax(attention_mask).astype(bool),
    )
    valid = attention_mask.bool().numpy()
    assert_close(ours.logits, expected, rel=2e-4, mask=valid)  # random tiny weights amplify f32 rounding
    assert float(jnp.abs(ours.logits).max()) <= config.final_logit_softcapping


def test_kv_cache_matches_full_forward_and_hf(tiny_a) -> None:
    model, params, config = tiny_a
    B, S, k = 2, 9, 5
    input_ids = torch.randint(3, config.vocab_size, (B, S))
    position_ids = torch.arange(S).expand(B, S)
    full = forward_gemma4_lm(params, as_jax(input_ids), as_jax(position_ids), config=config).logits

    cache = init_kv_cache(config, B, capacity=S + 2, dtype=jnp.float32)
    prefill = forward_gemma4_lm(
        params, as_jax(input_ids[:, :k]), as_jax(position_ids[:, :k]), config=config, kv_cache=cache
    )
    assert prefill.kv_cache is not None and int(prefill.kv_cache.length) == k
    assert_close(prefill.logits, torch.from_numpy(np.asarray(full[:, :k])))
    cache = prefill.kv_cache
    step_logits = []
    for t in range(k, S):  # one token at a time
        step = forward_gemma4_lm(
            params,
            as_jax(input_ids[:, t : t + 1]),
            as_jax(position_ids[:, t : t + 1]),
            config=config,
            kv_cache=cache,
            cache_mode="append",
        )
        cache = step.kv_cache
        step_logits.append(np.asarray(step.logits))
    ours_decode = np.concatenate(step_logits, axis=1)
    np.testing.assert_allclose(ours_decode, np.asarray(full[:, k:]), atol=2e-5 * float(np.abs(full).max()), rtol=0.0)

    hf_prefill = model(input_ids=input_ids[:, :k], position_ids=position_ids[:, :k], use_cache=True)
    hf_rest = model(
        input_ids=input_ids[:, k:],
        position_ids=position_ids[:, k:],
        past_key_values=hf_prefill.past_key_values,
        use_cache=True,
    )
    assert_close(ours_decode, hf_rest.logits, rel=1e-4)
