"""LoRA独立HF梯度、零初始化、K=V/共享KV、循环cache与HF合并导出。"""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from conftest import TINY_A, TINY_B, build_hf_model

from gemma4_posttrain_jax.lora import (
    AttentionLoRAParams,
    LoRAConfig,
    LowRankParams,
    check_lora_params,
    convert_back_lora_params,
    forward_lora_projection,
    init_lora_params,
    merge_lora_params,
    prepare_lora_params,
)
from gemma4_posttrain_jax.model import config_from_hf, forward_gemma4_lm, forward_gemma4_text, forward_lm_head
from gemma4_posttrain_jax.sampler import SamplerConfig, decode_step, generate, prefill
from gemma4_posttrain_jax.weights import convert_gemma4_text_params


@pytest.fixture(params=[TINY_A, TINY_B], ids=["ple-shared-kv", "12b-k-equals-v"])
def lora_fixture(request):
    # 复用已有HF小模型fixture构造器；小权重便于严格反向比较，不初始化基础模型。
    model = build_hf_model(request.param, seed=7, weight_std=0.03, norm_std=0.05, layer_scalar_min=0.95)
    config = config_from_hf(request.param)
    base = convert_gemma4_text_params(model.state_dict().__getitem__, config)
    settings = LoRAConfig(rank=4, alpha=6.0)
    initial = init_lora_params(base, settings, jax.random.PRNGKey(17))
    rng = np.random.default_rng(19)
    nonzero = jax.tree.map(lambda x: x + jnp.asarray(rng.normal(0, 0.002, x.shape), x.dtype), initial)
    ids = jnp.asarray([[0, 2, 13, 9, 41, 8, 3], [2, 5, 16, 12, 6, 44, 18]], dtype=jnp.int32)
    mask = ids != 0
    positions = jnp.maximum(jnp.cumsum(mask, axis=1) - 1, 0)
    return model, base, config, settings, initial, nonzero, ids, mask, positions


def test_zero_initialization_and_parameter_presence(lora_fixture):
    _, base, config, settings, initial, _, ids, mask, positions = lora_fixture
    check_lora_params(base, initial, settings)
    for source, layer in zip(base.layers, initial.layers, strict=True):
        for name in AttentionLoRAParams._fields:
            adapter = getattr(layer, name)
            assert (adapter is None) == (getattr(source.attention, name) is None)
            if adapter is not None:
                assert np.count_nonzero(adapter.a) > 0 and np.count_nonzero(adapter.b) == 0
    plain = forward_gemma4_lm(base, ids, positions, config=config, attention_mask=mask)
    adapted = forward_gemma4_lm(
        base, ids, positions, config=config, attention_mask=mask, lora=prepare_lora_params(initial, settings)
    )
    np.testing.assert_array_equal(plain.logits, adapted.logits)
    merged = merge_lora_params(base, initial, settings)
    for before, after in zip(jax.tree.leaves(base), jax.tree.leaves(merged), strict=True):
        np.testing.assert_array_equal(before, after)


def test_nonzero_adapter_forward_hf_export_and_cache_wrap(lora_fixture):
    hf, base, config, settings, _, adapters, ids, mask, positions = lora_fixture
    prepared = prepare_lora_params(adapters, settings)
    model = copy.deepcopy(hf)
    state = {
        name: torch.from_numpy(value.copy())
        for name, value in convert_back_lora_params(base, adapters, settings, config).items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected
    assert all(name == "lm_head.weight" or ".self_attn." in name for name in missing)
    with torch.no_grad():
        expected = model(
            input_ids=torch.tensor(np.asarray(ids), dtype=torch.long),
            attention_mask=torch.tensor(np.asarray(mask)),
            position_ids=torch.tensor(np.asarray(positions), dtype=torch.long),
        ).logits.numpy()
    actual = forward_gemma4_lm(base, ids, positions, config=config, attention_mask=mask, lora=prepared)
    np.testing.assert_allclose(actual.logits, expected, atol=2e-5, rtol=2e-5)
    prompt = 4
    out = prefill(base, config, ids[:, :prompt], mask[:, :prompt], max_new_tokens=3, lora=prepared)
    assert out.kv_cache is not None
    for t in range(prompt, ids.shape[1]):
        out = decode_step(
            base, config, ids[:, t], positions[:, t], out.kv_cache, attention_mask=mask[:, t], lora=prepared
        )
        assert out.kv_cache is not None
        full = forward_gemma4_lm(
            base, ids[:, : t + 1], positions[:, : t + 1], config=config, attention_mask=mask[:, : t + 1], lora=prepared
        )
        np.testing.assert_allclose(out.logits[:, -1], full.logits[:, -1], atol=2e-5, rtol=2e-5)
    sampler = SamplerConfig(max_prompt_len=4, max_new_tokens=4, temperature=0, eos_ids=())
    generated = jax.jit(
        lambda p: generate(
            base, config, ids[:, :4], mask[:, :4], sampler_config=sampler, lora=prepare_lora_params(p, settings)
        )
    )(adapters)
    merged_generated = generate(
        merge_lora_params(base, adapters, settings), config, ids[:, :4], mask[:, :4], sampler_config=sampler
    )
    np.testing.assert_array_equal(generated.completion_ids, merged_generated.completion_ids)
    np.testing.assert_allclose(generated.rollout_logps, merged_generated.rollout_logps, atol=2e-5, rtol=2e-5)


def test_all_adapter_gradients_match_independent_torch_hf(lora_fixture):
    hf, base, config, settings, _, adapters, ids, mask, positions = lora_fixture
    model = copy.deepcopy(hf).requires_grad_(False)
    parameters = dict(model.named_parameters())
    expected_leaves = []
    with torch.enable_grad():
        for index, (base_layer, layer) in enumerate(zip(base.layers, adapters.layers, strict=True)):
            for name in AttentionLoRAParams._fields:
                value = getattr(layer, name)
                if value is None:
                    continue
                a = torch.tensor(np.asarray(value.a), requires_grad=True)
                b = torch.tensor(np.asarray(value.b), requires_grad=True)
                expected_leaves.extend((a, b))
                weight = getattr(base_layer.attention, name)
                delta = (a @ b * (settings.alpha / settings.rank)).reshape(weight.shape)
                # 显式HF轴转换独立于JAX导出函数；q/o的H/R顺序不可互换。
                if name == "q_proj":
                    delta = delta.permute(0, 2, 1, 3).reshape(weight.shape[0], -1).T
                elif name == "o_proj":
                    delta = delta.permute(1, 0, 2, 3).reshape(-1, weight.shape[-1]).T
                else:
                    delta = delta.reshape(weight.shape[0], -1).T
                key = f"model.layers.{index}.self_attn.{name}.weight"
                parameters[key] = parameters[key] + delta
        labels = torch.tensor(np.asarray(ids), dtype=torch.long)
        labels[0, :2] = -100
        expected = torch.func.functional_call(
            model,
            parameters,
            (),
            dict(
                input_ids=torch.tensor(np.asarray(ids), dtype=torch.long),
                attention_mask=torch.tensor(np.asarray(mask)),
                position_ids=torch.tensor(np.asarray(positions), dtype=torch.long),
                labels=labels,
            ),
        ).loss
        expected.backward()

    jax_labels = jnp.asarray(labels.numpy())[:, 1:]
    selected = jax_labels != -100

    def loss(p):
        hidden, _ = forward_gemma4_text(
            base,
            ids,
            positions,
            config=config,
            attention_mask=mask,
            remat_layers=True,
            lora=prepare_lora_params(p, settings),
        )
        logits = forward_lm_head(base.embed_tokens, hidden, config.final_logit_softcapping)[:, :-1]
        logps = jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1), jnp.maximum(jax_labels, 0)[..., None], axis=-1
        )[..., 0]
        return -jnp.sum(jnp.where(selected, logps, 0)) / jnp.sum(selected)

    base_before = [np.asarray(x).copy() for x in jax.tree.leaves(base)]
    actual, grads = jax.jit(jax.value_and_grad(loss))(adapters)
    np.testing.assert_allclose(actual, expected.detach().numpy(), atol=2e-5, rtol=2e-5)
    relative = []
    for actual_grad, tensor in zip(jax.tree.leaves(grads), expected_leaves, strict=True):
        assert tensor.grad is not None
        expected_grad = tensor.grad.numpy()
        relative.append(
            float(np.linalg.norm(np.asarray(actual_grad) - expected_grad) / max(1e-12, np.linalg.norm(expected_grad)))
        )
    assert max(relative) < 1e-4, max(relative)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grads))
    assert any(np.count_nonzero(x) for x in jax.tree.leaves(grads))
    for before, after in zip(base_before, jax.tree.leaves(base), strict=True):
        np.testing.assert_array_equal(before, after)


def test_bf16_keeps_low_rank_signal_lost_by_weight_merge():
    # 正负基础权重抵消；逐权重0.001会被BF16吞掉，独立分支的总增量仍可表示。
    x = jnp.ones((1, 1, 32), dtype=jnp.bfloat16)
    weight = jnp.tile(jnp.asarray([1, -1], dtype=jnp.bfloat16), 16)[:, None].repeat(4, axis=1)
    params = LowRankParams(jnp.ones((32, 1), dtype=jnp.float32), jnp.full((1, 4), 0.001, dtype=jnp.float32))
    base = x @ weight
    actual = forward_lora_projection(params, x, base)
    expected = (torch.ones(1, 1, 32) @ torch.ones(32, 1) @ torch.full((1, 4), 0.001)).to(torch.bfloat16)
    np.testing.assert_array_equal(np.asarray(actual).astype(np.float32), expected.float().numpy())
    assert np.count_nonzero(actual) == 4
    merged = (weight.astype(jnp.float32) + params.a @ params.b).astype(jnp.bfloat16)
    np.testing.assert_array_equal(x @ merged, jnp.zeros_like(base))
    grads = jax.grad(lambda p: forward_lora_projection(p, x, base).astype(jnp.float32).sum())(params)
    np.testing.assert_array_equal(grads.b, jnp.full_like(params.b, 32))
