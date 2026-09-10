"""Optimizer masking and a deterministic one-batch SFT overfit smoke."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gemma4_posttrain_jax.checkpoint import load_train_state, read_checkpoint_metadata, save_train_state
from gemma4_posttrain_jax.losses import init_train_state, make_optimizer, train_step
from gemma4_posttrain_jax.sharding import tree_shardings


def test_checkpoint_preserves_logical_order_and_scalar_shapes(tmp_path) -> None:
    # TPU复制矩阵的host视图可能为列优先；checkpoint仍须保存逻辑行列与零维标量。
    state = {
        "matrix": np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4)),
        "floating_scalar": np.asarray(0.5, dtype=np.float32),
        "integer_scalar": np.asarray(7, dtype=np.int32),
    }
    path = save_train_state(state, tmp_path / "column_major", metadata={"kind": "logical-order-control"})
    template = jax.tree.map(lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype), state)
    restored = load_train_state(path, template)
    for key, expected in state.items():
        np.testing.assert_array_equal(restored[key], expected)
        assert restored[key].shape == expected.shape
        assert restored[key].dtype == expected.dtype


def test_tiny_sft_overfits_one_batch(grad_tiny) -> None:
    _, params, config = grad_tiny
    input_ids = jnp.asarray(
        [
            [2, 13, 9, 41, 8, 3, 27],
            [2, 5, 16, 12, 6, 44, 18],
        ],
        jnp.int32,
    )
    labels = input_ids.at[:, :2].set(-100)
    attention_mask = jnp.ones_like(input_ids, dtype=jnp.bool_)
    optimizer, trainable = make_optimizer(params, learning_rate=1e-2, max_grad_norm=1.0)
    state = init_train_state(params, optimizer)
    step = jax.jit(
        lambda current: train_step(
            current,
            input_ids,
            labels,
            attention_mask,
            config=config,
            optimizer=optimizer,
            trainable_mask=trainable,
            compute_dtype=jnp.float32,
            vocab_chunk=17,
            sequence_chunk=5,
        )
    )
    losses = []
    for _ in range(100):
        state, metrics = step(state)
        losses.append(float(metrics.loss))
    print(f"tiny overfit: initial_loss={losses[0]:.6f} final_loss={losses[-1]:.6f}")
    assert losses[-1] < 0.05, (losses[0], losses[-1])


def test_freeze_embeddings_mask_keeps_lookup_tables_fixed(grad_tiny) -> None:
    _, params, config = grad_tiny
    input_ids = jnp.asarray([[2, 13, 9, 41, 8], [2, 5, 16, 12, 6]], jnp.int32)
    labels = input_ids.at[:, :1].set(-100)
    attention_mask = jnp.ones_like(input_ids, dtype=jnp.bool_)
    optimizer, trainable = make_optimizer(params, learning_rate=1e-2, freeze_embeddings=True)
    state = init_train_state(params, optimizer)
    new_state, _ = train_step(
        state,
        input_ids,
        labels,
        attention_mask,
        config=config,
        optimizer=optimizer,
        trainable_mask=trainable,
        compute_dtype=jnp.float32,
        vocab_chunk=19,
        sequence_chunk=4,
    )
    np.testing.assert_array_equal(new_state.params_f32.embed_tokens, params.embed_tokens)
    assert params.embed_tokens_per_layer is not None and new_state.params_f32.embed_tokens_per_layer is not None
    np.testing.assert_array_equal(new_state.params_f32.embed_tokens_per_layer, params.embed_tokens_per_layer)
    assert not np.array_equal(new_state.params_f32.final_norm.weight, params.final_norm.weight)


def test_gradient_accumulation_updates_only_after_k_steps(grad_tiny) -> None:
    _, params, config = grad_tiny
    input_ids = jnp.asarray([[2, 13, 9, 41, 8], [2, 5, 16, 12, 6]], jnp.int32)
    labels = input_ids.at[:, :1].set(-100)
    attention_mask = jnp.ones_like(input_ids, dtype=jnp.bool_)
    optimizer, trainable = make_optimizer(params, learning_rate=1e-2, gradient_accumulation_steps=2)
    state = init_train_state(params, optimizer)
    state_one, _ = train_step(
        state,
        input_ids,
        labels,
        attention_mask,
        config=config,
        optimizer=optimizer,
        trainable_mask=trainable,
        compute_dtype=jnp.float32,
        vocab_chunk=19,
        sequence_chunk=4,
    )
    np.testing.assert_array_equal(state_one.params_f32.final_norm.weight, params.final_norm.weight)
    state_two, _ = train_step(
        state_one,
        input_ids,
        labels,
        attention_mask,
        config=config,
        optimizer=optimizer,
        trainable_mask=trainable,
        compute_dtype=jnp.float32,
        vocab_chunk=19,
        sequence_chunk=4,
    )
    assert not np.array_equal(state_two.params_f32.final_norm.weight, params.final_norm.weight)


def test_checkpoint_resume_matches_uninterrupted_training(grad_tiny, tmp_path) -> None:
    _, params, config = grad_tiny
    input_ids = jnp.asarray([[2, 13, 9, 41, 8], [2, 5, 16, 12, 6]], jnp.int32)
    labels = input_ids.at[:, :1].set(-100)
    attention_mask = jnp.ones_like(input_ids, dtype=jnp.bool_)
    optimizer, trainable = make_optimizer(params, learning_rate=1e-2, max_grad_norm=1.0)
    initial_state = init_train_state(params, optimizer)
    step = jax.jit(
        lambda current: train_step(
            current,
            input_ids,
            labels,
            attention_mask,
            config=config,
            optimizer=optimizer,
            trainable_mask=trainable,
            compute_dtype=jnp.float32,
            vocab_chunk=19,
            sequence_chunk=4,
        )
    )

    continuous_state = initial_state
    continuous_losses = []
    for _ in range(10):
        continuous_state, metrics = step(continuous_state)
        continuous_losses.append(float(metrics.loss))

    split_state = initial_state
    split_losses = []
    for _ in range(6):
        split_state, metrics = step(split_state)
        split_losses.append(float(metrics.loss))
    checkpoint = save_train_state(
        split_state,
        tmp_path / "step_00000006",
        metadata={"completed_step": 6, "run_config": {"mode": "fixed-test"}},
    )
    template = init_train_state(params, optimizer)
    restored_state = load_train_state(checkpoint, template, shardings=tree_shardings(template))
    for _ in range(4):
        restored_state, metrics = step(restored_state)
        split_losses.append(float(metrics.loss))

    np.testing.assert_allclose(split_losses, continuous_losses, rtol=0.0, atol=1e-6)
    for restored, continuous in zip(jax.tree.leaves(restored_state), jax.tree.leaves(continuous_state), strict=True):
        np.testing.assert_array_equal(restored, continuous)
    document = read_checkpoint_metadata(checkpoint)
    assert document["metadata"]["completed_step"] == 6
    assert int(restored_state.step) == 10
