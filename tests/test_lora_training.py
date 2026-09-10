"""LoRA复用GRPO的四CPU微批/轮内old/完整Adam恢复，并保持基础权重冻结。"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import TINY_B, build_hf_model
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from gemma4_posttrain_jax.checkpoint import load_train_state, read_checkpoint_metadata, save_train_state
from gemma4_posttrain_jax.lora import LoRAConfig, init_lora_params, prepare_lora_params
from gemma4_posttrain_jax.lora_training import lora_base_identity, make_lora_optimizer
from gemma4_posttrain_jax.losses import compute_advantages, grpo_train_step, init_train_state, trainer_completion_logps
from gemma4_posttrain_jax.model import config_from_hf
from gemma4_posttrain_jax.sharding import make_mesh, shard_batch, shard_gemma4_text_params, tree_shardings
from gemma4_posttrain_jax.weights import convert_gemma4_text_params


@pytest.fixture(scope="module")
def training_fixture():
    raw = {**TINY_B, "num_hidden_layers": 2, "layer_types": ["sliding_attention", "full_attention"], "vocab_size": 32}
    hf = build_hf_model(raw, seed=7, weight_std=0.03, norm_std=0.05, layer_scalar_min=0.95)
    config = config_from_hf(raw)
    base = convert_gemma4_text_params(hf.state_dict().__getitem__, config)
    settings = LoRAConfig(rank=2, alpha=3.0)
    params = init_lora_params(base, settings, jax.random.PRNGKey(31))
    rng = np.random.default_rng(43)
    params = jax.tree.map(lambda x: x + jnp.asarray(rng.normal(0, 0.002, x.shape), x.dtype), params)
    ids = jnp.asarray([[0, 2, 11], [2, 8, 7]] * 4, jnp.int32)
    completion = jnp.asarray([[3, 9, 0, 0], [5, 4, 1, 0], [1, 0, 0, 0], [8, 3, 6, 1]] * 2, jnp.int32)
    mask = completion != 0
    # 重复token行的第二组不能恰好取相反优势，否则真实梯度为零，Adam会放大归约舍入噪声。
    advantages = compute_advantages(jnp.asarray([0, 1, 1, 1, 1, 0, 1, 0]), 4)
    return base, config, settings, params, (ids, ids != 0, completion, mask, advantages)


@pytest.mark.parametrize("aggregation", ["sequence-mean-token-mean", "token-mean"])
def test_four_cpu_microbatch_joint_old_and_full_state_restore(training_fixture, tmp_path, aggregation):
    if jax.device_count() != 4:
        pytest.skip("显式JAX_NUM_CPU_DEVICES=4运行四设备LoRA资格")
    base, config, settings, params, tokens = training_fixture
    base_before = [np.asarray(x).copy() for x in jax.tree.leaves(base)]
    mesh = make_mesh()
    base = shard_gemma4_text_params(base, config, mesh)
    params = jax.device_put(params, NamedSharding(mesh, P()))
    optimizer = make_lora_optimizer(0.003)
    initial = jax.device_put(init_train_state(params, optimizer), NamedSharding(mesh, P()))
    batch = shard_batch(tokens, mesh)
    old = shard_batch(jnp.zeros(tokens[2].shape, jnp.float32), mesh)

    def update(state, frozen, old_logps, capture, microbatch):
        return grpo_train_step(
            state,
            *batch,
            old_logps,
            None,
            config=config,
            optimizer=optimizer,
            trainable_mask=None,
            compute_dtype=jnp.float32,
            vocab_chunk=8,
            sequence_chunk=4,
            remat_layers=True,
            mesh=mesh,
            agg_mode=aggregation,
            microbatch_size=microbatch,
            use_current_policy_as_old=capture,
            frozen_base=frozen,
            lora_config=settings,
        )

    full = jax.jit(lambda state, frozen, old, capture: update(state, frozen, old, capture, None))
    micro = jax.jit(lambda state, frozen, old, capture: update(state, frozen, old, capture, 4))
    full_state, full_metrics = full(initial, base, old, jnp.asarray(True))
    state, metrics = micro(initial, base, old, jnp.asarray(True))
    assert float(metrics.loss_metrics.ratio_min) == float(metrics.loss_metrics.ratio_max) == 1.0
    assert float(metrics.grad_norm) > 1e-4
    for actual, expected in zip(
        jax.tree.leaves((state, metrics)), jax.tree.leaves((full_state, full_metrics)), strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
    captured = metrics.policy_logps
    assert captured is not None
    state, second = micro(state, base, captured, jnp.asarray(False))
    assert second.policy_logps is not None
    assert np.max(np.abs(np.asarray(second.policy_logps - captured))) > 1e-6
    assert float(second.loss_metrics.ratio_min) != 1.0 or float(second.loss_metrics.ratio_max) != 1.0
    metadata = {"kind": "lora-unscaled-fp32-v1", "lora": dict(rank=2, alpha=3.0), "data_cursor": 1, "seed": 31}
    checkpoint = save_train_state(state, tmp_path / "step2", metadata=metadata)
    assert read_checkpoint_metadata(checkpoint)["metadata"] == metadata
    template = jax.eval_shape(lambda: state)
    restored = load_train_state(checkpoint, template, shardings=tree_shardings(state))
    # 完整轮边界后重新捕获old；恢复后的下两个Adam与连续运行逐叶完全一致。
    for branch in range(2):
        target = state if branch == 0 else restored
        target, next_metrics = micro(target, base, old, jnp.asarray(True))
        target, _ = micro(target, base, next_metrics.policy_logps, jnp.asarray(False))
        if branch == 0:
            continuous = target
        else:
            for left, right in zip(jax.tree.leaves(continuous), jax.tree.leaves(target), strict=True):
                np.testing.assert_array_equal(left, right)
    assert int(continuous.step) == 4
    assert all(x.dtype == jnp.float32 for x in jax.tree.leaves(continuous.params_f32))
    manifest = read_checkpoint_metadata(checkpoint)
    assert all("frozen_base" not in x["name"] and "embed_tokens" not in x["name"] for x in manifest["leaves"])
    for before, after in zip(base_before, jax.tree.leaves(base), strict=True):
        np.testing.assert_array_equal(before, after)


def test_prepared_lora_logps_microbatch_matches_full(training_fixture):
    base, config, settings, params, tokens = training_fixture

    def fn(size):
        return trainer_completion_logps(
            base,
            *tokens[:4],
            config=config,
            compute_dtype=jnp.float32,
            vocab_chunk=8,
            sequence_chunk=4,
            lora=prepare_lora_params(params, settings),
            microbatch_size=size,
        )

    np.testing.assert_allclose(fn(None), fn(4), atol=3e-6, rtol=3e-5)


def test_base_identity_detects_same_path_content_changes(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"original")
    (tmp_path / "tokenizer.json").write_text("{}")
    before = lora_base_identity(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"modified")
    after = lora_base_identity(tmp_path)
    assert before != after
    assert before["model.safetensors"]["size"] == after["model.safetensors"]["size"]
    (tmp_path / "tokenizer.json").write_text('{"changed":true}')
    assert lora_base_identity(tmp_path) != after
