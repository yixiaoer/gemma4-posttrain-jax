"""Four-chip TPU checks for FSDP placement, forward parity, and one SFT update."""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from gemma4_posttrain_jax.losses import init_train_state, make_optimizer, per_token_logps, sft_loss, train_step
from gemma4_posttrain_jax.model import forward_gemma4_text
from gemma4_posttrain_jax.sampler import RolloutBatch, SamplerConfig, decode_step, generate, prefill
from gemma4_posttrain_jax.sharding import (
    batch_spec,
    make_mesh,
    named_shardings,
    param_specs_replicated,
    replicate_scalars,
    reshard_after_rollout,
    reshard_for_rollout,
    shard_batch,
    shard_gemma4_text_params,
    tree_shardings,
)

pytestmark = pytest.mark.tpu


def _copy_params(params):
    """Give each donation test independent buffers from the session-scoped fixture."""

    return jax.tree.map(lambda value: jnp.array(value, copy=True), params)


def _relative_l2(actual, expected) -> float:
    actual_leaves = [np.asarray(value, np.float64).reshape(-1) for value in jax.tree.leaves(actual)]
    expected_leaves = [np.asarray(value, np.float64).reshape(-1) for value in jax.tree.leaves(expected)]
    numerator = np.sqrt(sum(np.vdot(a - b, a - b).real for a, b in zip(actual_leaves, expected_leaves, strict=True)))
    denominator = np.sqrt(sum(np.vdot(b, b).real for b in expected_leaves))
    return float(numerator / max(denominator, 1e-30))


def test_vocab_parallel_logps_extreme_max_and_gradients() -> None:
    if jax.default_backend() != "tpu" or jax.device_count() != 4:
        pytest.skip("requires the four-device TPU v4-8")
    mesh = make_mesh()
    rng = np.random.default_rng(7)
    embed = rng.normal(0.0, 0.2, (96, 8)).astype(np.float32)
    hidden = rng.normal(0.0, 0.3, (4, 5, 8)).astype(np.float32)
    for shard, row in enumerate((3, 27, 51, 75)):
        embed[row, shard] = 30.0 + 10.0 * shard
        hidden[shard, :, shard] = 1.0
    targets = np.asarray(
        [[0, 23, 24, 47, 48], [71, 72, 95, 3, 27], [51, 75, 12, 36, 60], [84, 8, 32, 56, 80]],
        np.int32,
    )
    weights = jnp.linspace(0.5, 1.5, targets.size, dtype=jnp.float32).reshape(targets.shape)

    def objective(e, h, t, *, distributed: bool):
        return (
            per_token_logps(
                e,
                h,
                t,
                softcap=30.0,
                vocab_chunk=17,
                sequence_chunk=6,
                mesh=mesh if distributed else None,
            )
            * weights
        ).sum()

    replicated = NamedSharding(mesh, P())
    embed_sharded = jax.device_put(embed, NamedSharding(mesh, P("d", None)))
    hidden_sharded = jax.device_put(hidden, NamedSharding(mesh, P("d", None, None)))
    targets_sharded = jax.device_put(targets, NamedSharding(mesh, P("d", None)))
    reference_fn = jax.jit(
        jax.value_and_grad(lambda e, h, t: objective(e, h, t, distributed=False), argnums=(0, 1)),
        in_shardings=(replicated, replicated, replicated),
    )
    distributed_fn = jax.jit(
        jax.value_and_grad(lambda e, h, t: objective(e, h, t, distributed=True), argnums=(0, 1)),
        in_shardings=(embed_sharded.sharding, hidden_sharded.sharding, targets_sharded.sharding),
    )
    reference_value, reference_grads = reference_fn(
        jax.device_put(embed, replicated), jax.device_put(hidden, replicated), jax.device_put(targets, replicated)
    )
    distributed_value, distributed_grads = distributed_fn(embed_sharded, hidden_sharded, targets_sharded)
    jax.block_until_ready((reference_value, reference_grads, distributed_value, distributed_grads))
    np.testing.assert_allclose(distributed_value, reference_value, atol=5e-6, rtol=5e-6)
    assert _relative_l2(distributed_grads, reference_grads) <= 2e-6

    hlo = distributed_fn.lower(embed_sharded, hidden_sharded, targets_sharded).compile().as_text() or ""
    full_embedding_gathers = [
        line for line in hlo.splitlines() if "all-gather(" in line and re.search(r"f32\[96,8\]", line)
    ]
    assert not full_embedding_gathers, "\n".join(full_embedding_gathers)


def test_vocab_parallel_tiny_sft_full_gradient_parity(tpu_tiny) -> None:
    if jax.default_backend() != "tpu" or jax.device_count() != 4:
        pytest.skip("requires the four-device TPU v4-8")
    _, host_params, config = tpu_tiny
    mesh = make_mesh()
    params = shard_gemma4_text_params(_copy_params(host_params), config, mesh)
    input_ids = np.asarray(
        [[2, 7, 11, 5, 19, 4, 8], [2, 3, 23, 9, 10, 6, 14], [2, 8, 17, 4, 12, 5, 29], [2, 6, 13, 7, 15, 9, 31]],
        np.int32,
    )
    labels = input_ids.copy()
    labels[:, :2] = -100
    attention_mask = np.ones_like(input_ids, dtype=np.bool_)
    batch = shard_batch((input_ids, labels, attention_mask), mesh)
    input_shardings = (tree_shardings(params),) + tuple(tree_shardings(value) for value in batch)

    def make_loss(distributed: bool):
        return jax.jit(
            jax.value_and_grad(
                lambda p, ids, targets, mask: sft_loss(
                    p,
                    ids,
                    targets,
                    config=config,
                    attention_mask=mask,
                    compute_dtype=jnp.float32,
                    vocab_chunk=17,
                    sequence_chunk=6,
                    mesh=mesh if distributed else None,
                )
            ),
            in_shardings=input_shardings,
        )

    reference_fn = make_loss(False)
    distributed_fn = make_loss(True)
    reference_loss, reference_grads = reference_fn(params, *batch)
    distributed_loss, distributed_grads = distributed_fn(params, *batch)
    jax.block_until_ready((reference_loss, reference_grads, distributed_loss, distributed_grads))
    gradient_relative_l2 = _relative_l2(distributed_grads, reference_grads)
    print(
        f"vocab_parallel_tiny_loss_reference={float(reference_loss):.10f} "
        f"distributed={float(distributed_loss):.10f} grad_relative_l2={gradient_relative_l2:.3e}"
    )
    np.testing.assert_allclose(distributed_loss, reference_loss, atol=5e-6, rtol=5e-6)
    assert gradient_relative_l2 <= 3e-6


def test_fsdp_forward_and_train_step(tpu_tiny) -> None:
    if jax.default_backend() != "tpu" or jax.device_count() != 4:
        pytest.skip("requires the four-device TPU v4-8")
    _, host_params, config = tpu_tiny
    mesh = make_mesh()
    replicated = shard_gemma4_text_params(_copy_params(host_params), config, mesh, replicated=True)
    fsdp = shard_gemma4_text_params(_copy_params(host_params), config, mesh)
    assert isinstance(fsdp.embed_tokens.sharding, NamedSharding)
    assert fsdp.embed_tokens.sharding.spec == P("d", None)

    input_ids = np.asarray(
        [
            [2, 7, 11, 5, 19, 4, 8],
            [2, 3, 23, 9, 10, 6, 14],
            [2, 8, 17, 4, 12, 5, 29],
            [2, 6, 13, 7, 15, 9, 31],
        ],
        np.int32,
    )
    attention_mask = np.ones_like(input_ids, dtype=np.bool_)
    position_ids = np.broadcast_to(np.arange(input_ids.shape[1], dtype=np.int32), input_ids.shape)
    ids, mask, positions = shard_batch(
        (jnp.asarray(input_ids), jnp.asarray(attention_mask), jnp.asarray(position_ids)), mesh
    )

    forward = jax.jit(
        lambda p, token_ids, token_mask, token_positions: forward_gemma4_text(
            p,
            token_ids,
            token_positions,
            config=config,
            attention_mask=token_mask,
            remat_layers=True,
        )[0]
    )
    expected = forward(replicated, ids, mask, positions)
    actual = forward(fsdp, ids, mask, positions)
    jax.block_until_ready((expected, actual))
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-4, rtol=1e-4)

    labels = ids
    optimizer, trainable = make_optimizer(fsdp, learning_rate=1e-3, max_grad_norm=1.0)
    state = replicate_scalars(init_train_state(fsdp, optimizer), mesh)
    old_final_norm = np.asarray(state.params_f32.final_norm.weight).copy()
    state_shardings = tree_shardings(state)
    batch_sharding = NamedSharding(mesh, batch_spec(2))
    step = jax.jit(
        lambda current, batch_ids, batch_labels, batch_mask: train_step(
            current,
            batch_ids,
            batch_labels,
            batch_mask,
            config=config,
            optimizer=optimizer,
            trainable_mask=trainable,
            compute_dtype=jnp.float32,
            vocab_chunk=17,
            sequence_chunk=6,
        ),
        donate_argnums=(0,),
        in_shardings=(state_shardings, batch_sharding, batch_sharding, batch_sharding),
        out_shardings=(state_shardings, None),
    )
    new_state, metrics = step(state, ids, labels, mask)
    jax.block_until_ready((new_state, metrics))
    assert int(new_state.step) == 1
    assert np.isfinite(float(metrics.loss)) and np.isfinite(float(metrics.grad_norm))
    assert new_state.params_f32.embed_tokens.sharding.spec == P("d", None)
    assert not np.array_equal(np.asarray(new_state.params_f32.final_norm.weight), old_final_norm)


def test_layer_remat_recomputes_forward(tpu_tiny) -> None:
    """The Python layer loop must retain checkpoint's CSE barrier in optimized TPU HLO."""

    if jax.default_backend() != "tpu" or jax.device_count() != 4:
        pytest.skip("requires the four-device TPU v4-8")
    _, host_params, config = tpu_tiny
    mesh = make_mesh()
    fsdp = shard_gemma4_text_params(_copy_params(host_params), config, mesh)
    optimizer, trainable = make_optimizer(fsdp, learning_rate=1e-3, max_grad_norm=1.0)
    state = replicate_scalars(init_train_state(fsdp, optimizer), mesh)
    batch_size, sequence_length = 4, 64
    input_ids = np.broadcast_to(
        np.arange(3, 3 + sequence_length, dtype=np.int32) % config.vocab_size,
        (batch_size, sequence_length),
    ).copy()
    input_ids[:, 0] = config.bos_token_id
    labels = input_ids.copy()
    labels[:, 0] = -100
    attention_mask = np.ones_like(input_ids, dtype=np.bool_)
    batch = shard_batch((input_ids, labels, attention_mask), mesh)
    state_shardings = tree_shardings(state)
    batch_sharding = NamedSharding(mesh, batch_spec(2))

    def optimized_hlo(remat_layers: bool) -> str:
        step = jax.jit(
            lambda current, batch_ids, batch_labels, batch_mask: train_step(
                current,
                batch_ids,
                batch_labels,
                batch_mask,
                config=config,
                optimizer=optimizer,
                trainable_mask=trainable,
                compute_dtype=jnp.float32,
                vocab_chunk=24,
                sequence_chunk=8,
                remat_layers=remat_layers,
            ),
            in_shardings=(state_shardings, batch_sharding, batch_sharding, batch_sharding),
            out_shardings=(state_shardings, None),
        )
        return step.lower(state, *batch).compile().as_text() or ""

    opcode = re.compile(r"^\s*(?:ROOT\s+)?%\S+ = \S+ (?:dot|convolution)\(", re.MULTILINE)
    dots_off = len(opcode.findall(optimized_hlo(False)))
    dots_on = len(opcode.findall(optimized_hlo(True)))
    minimum_recomputed_dots = config.num_hidden_layers * 8
    print(f"remat_hlo_dots_on={dots_on} remat_hlo_dots_off={dots_off} delta={dots_on - dots_off}")
    assert dots_on - dots_off >= minimum_recomputed_dots, (
        f"remat did not preserve layer recomputation: dots_on={dots_on}, dots_off={dots_off}, "
        f"required_delta={minimum_recomputed_dots}"
    )


def test_rollout_reshard_and_four_chip_generate_match_one_chip(tpu_tiny) -> None:
    """B2 gate: BF16 layout round-trip plus FP32 batch-sharding numerical semantics."""

    if jax.default_backend() != "tpu" or jax.device_count() != 4:
        pytest.skip("requires the four-device TPU v4-8")
    _, host_params, config = tpu_tiny
    mesh = make_mesh()
    fsdp = shard_gemma4_text_params(_copy_params(host_params), config, mesh)
    replicated_shardings = named_shardings(param_specs_replicated(config), mesh)
    to_rollout = jax.jit(
        lambda current: reshard_for_rollout(current, config, mesh),
        in_shardings=(tree_shardings(fsdp),),
        out_shardings=replicated_shardings,
    )
    rollout_params = to_rollout(fsdp)
    jax.block_until_ready(rollout_params)
    for leaf in jax.tree.leaves(rollout_params):
        assert leaf.dtype == jnp.bfloat16
        assert isinstance(leaf.sharding, NamedSharding) and leaf.sharding.spec == P()

    to_fsdp = jax.jit(
        lambda current: reshard_after_rollout(current, config, mesh),
        in_shardings=(replicated_shardings,),
        out_shardings=tree_shardings(fsdp),
    )
    roundtrip = to_fsdp(rollout_params)
    jax.block_until_ready(roundtrip)
    expected_roundtrip = jax.tree.map(lambda value: value.astype(jnp.bfloat16).astype(jnp.float32), fsdp)
    assert _relative_l2(roundtrip, expected_roundtrip) == 0.0

    # The existing high-variance tiny fixture amplifies BF16 partition-order drift into unstable
    # autoregressive choices. Keep it for layout/round-trip, but use FP32 to gate sharding semantics;
    # real-E2B BF16 is the actual B2 rollout gate in scripts/check_rollout_tpu.py.
    to_rollout_f32 = jax.jit(
        lambda current: reshard_for_rollout(current, config, mesh, dtype=jnp.float32),
        in_shardings=(tree_shardings(fsdp),),
        out_shardings=replicated_shardings,
    )
    rollout_params_f32 = to_rollout_f32(fsdp)
    jax.block_until_ready(rollout_params_f32)

    prompt_ids = np.asarray(
        [[0, 0, 2, 7, 11, 5], [0, 2, 3, 23, 9, 10], [2, 8, 17, 4, 12, 5], [2, 6, 13, 7, 15, 9]],
        np.int32,
    )
    prompt_mask = prompt_ids != config.pad_token_id
    sampler_config = SamplerConfig(
        max_prompt_len=prompt_ids.shape[1],
        max_new_tokens=4,
        temperature=0.0,
        top_k=1,
        eos_ids=(),
        seed=29,
    )
    key = jax.random.PRNGKey(29)
    batch_sharding_2d = NamedSharding(mesh, batch_spec(2))
    batch_sharding_1d = NamedSharding(mesh, batch_spec(1))
    replicated = NamedSharding(mesh, P())
    four_output_shardings = RolloutBatch(
        batch_sharding_2d,
        batch_sharding_2d,
        batch_sharding_2d,
        batch_sharding_2d,
        batch_sharding_1d,
    )
    four_ids, four_mask = shard_batch((prompt_ids, prompt_mask), mesh)
    four_generate = jax.jit(
        lambda current, ids, mask, current_key: generate(
            current,
            config,
            ids,
            mask,
            sampler_config=sampler_config,
            key=current_key,
            mesh=mesh,
        ),
        in_shardings=(replicated_shardings, batch_sharding_2d, batch_sharding_2d, replicated),
        out_shardings=four_output_shardings,
    )
    four = four_generate(rollout_params_f32, four_ids, four_mask, key)
    jax.block_until_ready(four)
    for leaf in jax.tree.leaves(four):
        assert isinstance(leaf.sharding, NamedSharding)
        assert leaf.sharding.spec[0] == "d"

    single_mesh = make_mesh(jax.devices()[:1])
    single_replicated = named_shardings(param_specs_replicated(config), single_mesh)
    single_params = jax.device_put(jax.device_get(rollout_params_f32), single_replicated)
    single_batch_2d = NamedSharding(single_mesh, batch_spec(2))
    single_batch_1d = NamedSharding(single_mesh, batch_spec(1))
    single_scalar = NamedSharding(single_mesh, P())
    single_output_shardings = RolloutBatch(
        single_batch_2d,
        single_batch_2d,
        single_batch_2d,
        single_batch_2d,
        single_batch_1d,
    )
    single_generate = jax.jit(
        lambda current, ids, mask, current_key: generate(
            current,
            config,
            ids,
            mask,
            sampler_config=sampler_config,
            key=current_key,
            mesh=single_mesh,
        ),
        in_shardings=(single_replicated, single_batch_2d, single_batch_2d, single_scalar),
        out_shardings=single_output_shardings,
    )
    single = single_generate(
        single_params,
        jax.device_put(prompt_ids, single_batch_2d),
        jax.device_put(prompt_mask, single_batch_2d),
        jax.device_put(key, single_scalar),
    )
    jax.block_until_ready(single)

    four_prefill = jax.jit(
        lambda current, ids, mask: prefill(
            current,
            config,
            ids,
            mask,
            max_new_tokens=sampler_config.max_new_tokens,
            mesh=mesh,
        ),
        in_shardings=(replicated_shardings, batch_sharding_2d, batch_sharding_2d),
    )(rollout_params_f32, four_ids, four_mask)
    single_prefill = jax.jit(
        lambda current, ids, mask: prefill(
            current,
            config,
            ids,
            mask,
            max_new_tokens=sampler_config.max_new_tokens,
            mesh=single_mesh,
        ),
        in_shardings=(single_replicated, single_batch_2d, single_batch_2d),
    )(
        single_params,
        jax.device_put(prompt_ids, single_batch_2d),
        jax.device_put(prompt_mask, single_batch_2d),
    )
    jax.block_until_ready((four_prefill, single_prefill))
    assert four_prefill.kv_cache is not None and single_prefill.kv_cache is not None
    four_cache, single_cache = four_prefill.kv_cache, single_prefill.kv_cache
    four_logits, single_logits = four_prefill.logits[:, -1], single_prefill.logits[:, -1]
    four_decode = jax.jit(
        lambda current, token, position, cache: decode_step(current, config, token, position, cache, mesh=mesh),
        in_shardings=(
            replicated_shardings,
            batch_sharding_1d,
            batch_sharding_1d,
            tree_shardings(four_cache),
        ),
    )
    single_decode = jax.jit(
        lambda current, token, position, cache: decode_step(current, config, token, position, cache, mesh=single_mesh),
        in_shardings=(
            single_replicated,
            single_batch_1d,
            single_batch_1d,
            tree_shardings(single_cache),
        ),
    )
    prompt_lengths = prompt_mask.sum(axis=1, dtype=np.int32)
    forced_tokens = np.asarray(single.completion_ids)
    for step in range(sampler_config.max_new_tokens):
        four_host = np.asarray(four_logits, dtype=np.float32)
        single_host = np.asarray(single_logits, dtype=np.float32)
        sorted_single = np.sort(single_host, axis=-1)
        margins = sorted_single[:, -1] - sorted_single[:, -2]
        max_abs = float(np.max(np.abs(four_host - single_host)))
        top1_equal = np.argmax(four_host, axis=-1) == np.argmax(single_host, axis=-1)
        print(
            f"forced_step={step} logits_max_abs={max_abs:.6e} "
            f"single_top1_margin={margins.tolist()} top1_equal={top1_equal.tolist()}"
        )
        np.testing.assert_allclose(four_host, single_host, atol=2e-4, rtol=2e-4)
        if step + 1 == sampler_config.max_new_tokens:
            break
        tokens = forced_tokens[:, step]
        positions = prompt_lengths + step
        four_step = four_decode(
            rollout_params_f32,
            jax.device_put(tokens, batch_sharding_1d),
            jax.device_put(positions, batch_sharding_1d),
            four_cache,
        )
        single_step = single_decode(
            single_params,
            jax.device_put(tokens, single_batch_1d),
            jax.device_put(positions, single_batch_1d),
            single_cache,
        )
        jax.block_until_ready((four_step, single_step))
        assert four_step.kv_cache is not None and single_step.kv_cache is not None
        four_cache, single_cache = four_step.kv_cache, single_step.kv_cache
        four_logits, single_logits = four_step.logits[:, -1], single_step.logits[:, -1]

    np.testing.assert_array_equal(np.asarray(four.completion_ids), np.asarray(single.completion_ids))
    np.testing.assert_array_equal(np.asarray(four.completion_mask), np.asarray(single.completion_mask))
    np.testing.assert_array_equal(np.asarray(four.lengths), np.asarray(single.lengths))
    np.testing.assert_allclose(np.asarray(four.rollout_logps), np.asarray(single.rollout_logps), atol=2e-4, rtol=2e-4)
