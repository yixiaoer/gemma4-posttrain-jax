"""Tiny full-model loss and gradient parity against Hugging Face autograd."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import torch

from gemma4_posttrain_jax.losses import sft_loss
from gemma4_posttrain_jax.weights import convert_back_gemma4_text_params


def test_sft_loss_and_all_gradients_match_hf(grad_tiny) -> None:
    model, params, config = grad_tiny
    input_ids = torch.tensor(
        [
            [2, 13, 9, 41, 8, 3, 27],
            [2, 5, 16, 12, 6, 44, 18],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1]).expand_as(input_ids)
    labels = input_ids.clone()
    labels[0, :2] = -100

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        expected_loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
        ).loss
        expected_loss.backward()

    jax_ids = jnp.asarray(input_ids.numpy())
    jax_labels = jnp.asarray(labels.numpy())
    jax_mask = jnp.asarray(attention_mask.numpy(), dtype=jnp.bool_)
    jax_positions = jnp.asarray(position_ids.numpy())

    def loss_fn(p):
        return sft_loss(
            p,
            jax_ids,
            jax_labels,
            config=config,
            attention_mask=jax_mask,
            position_ids=jax_positions,
            vocab_chunk=17,
            sequence_chunk=5,
            remat_layers=True,
        )

    actual_loss, grads = jax.value_and_grad(loss_fn)(params)
    np.testing.assert_allclose(actual_loss, expected_loss.detach().numpy(), atol=2e-5, rtol=2e-5)

    actual_grads = convert_back_gemma4_text_params(grads, config)
    named_params = dict(model.named_parameters())
    max_relative_error = 0.0
    leaf_l2_errors: list[tuple[float, str]] = []
    squared_error = 0.0
    squared_expected = 0.0
    compared: set[str] = set()
    for key, actual in actual_grads.items():
        if key.endswith("layer_scalar"):
            continue  # HF registers this checkpoint value as a persistent, frozen buffer
        expected_tensor = named_params[key].grad
        assert expected_tensor is not None, key
        compared.add(key)
        expected = expected_tensor.detach().cpu().numpy().astype(np.float32)
        absolute_error = float(np.max(np.abs(actual - expected)))
        relative_error = absolute_error / max(1e-7, float(np.max(np.abs(expected))))
        max_relative_error = max(max_relative_error, relative_error)
        leaf_l2_error = float(np.linalg.norm(actual - expected) / max(1e-12, np.linalg.norm(expected)))
        leaf_l2_errors.append((leaf_l2_error, key))
        squared_error += float(np.square(actual - expected).sum())
        squared_expected += float(np.square(expected).sum())
    global_relative_error = (squared_error / squared_expected) ** 0.5
    largest_leaf_l2 = sorted(leaf_l2_errors, reverse=True)
    print(
        "gradient parity:",
        f"global_l2={global_relative_error:.3e}",
        f"max_elementwise={max_relative_error:.3e}",
        f"largest_leaf_l2={largest_leaf_l2[:8]}",
    )
    assert compared == set(named_params)
    assert global_relative_error <= 1e-4
    assert largest_leaf_l2[0][0] <= 1e-4
