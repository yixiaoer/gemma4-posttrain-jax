"""Memory-bounded log-probability and supervised fine-tuning objectives."""

from __future__ import annotations

import math
from typing import Any, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import optax
from jax import Array, lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from .lora import Gemma4LoRAParams, LoRAConfig, prepare_lora_params
from .model import Gemma4TextConfig, Gemma4TextParams, forward_gemma4_text
from .sharding import DATA_AXIS, constrain_batch


class TrainState[Params](NamedTuple):
    params_f32: Params
    opt_state: optax.OptState
    step: Array


class TrainMetrics(NamedTuple):
    loss: Array
    grad_norm: Array


class GRPOMetrics(NamedTuple):
    loss: Array
    policy_loss: Array
    kl_loss: Array
    ppo_kl: Array
    pg_clipfrac: Array
    dual_clipfrac: Array
    ratio_mean: Array
    ratio_min: Array
    ratio_max: Array
    log_ratio_abs_mean: Array
    advantage_abs_mean: Array
    advantage_min: Array
    advantage_max: Array
    completion_tokens: Array
    sampled_entropy: Array


class GRPOTrainMetrics(NamedTuple):
    loss_metrics: GRPOMetrics
    grad_norm: Array
    policy_logps: Array | None = None


def _softcap(logits: Array, softcap: float | None) -> Array:
    if softcap is None:
        return logits
    return jnp.tanh(logits / softcap) * softcap


def _matmul_f32(left: Array, right_transposed: Array) -> Array:
    """Matrix multiply with f32 accumulation/output, including for bf16 operands."""

    return jnp.matmul(
        left,
        right_transposed,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


def _single_partition_logps(
    embed_tokens: Array,
    hidden: Array,
    targets: Array,
    *,
    softcap: float | None,
    vocab_chunk: int = 8192,
    sequence_chunk: int = 256,
) -> Array:
    """Return target-token log-probabilities without materialising ``[..., vocab]`` logits.

    The flattened token dimension is scanned in ``sequence_chunk`` blocks. Within each token
    block, a second scan visits ``vocab_chunk`` rows of the tied embedding and performs an online
    logsumexp update. The vocabulary scan body is checkpointed, so its temporary logits are
    recomputed during backward instead of being retained for every vocabulary block.
    """

    if hidden.shape[:-1] != targets.shape:
        raise ValueError(f"hidden prefix {hidden.shape[:-1]} does not match targets {targets.shape}")
    if embed_tokens.ndim != 2 or hidden.shape[-1] != embed_tokens.shape[-1]:
        raise ValueError(f"incompatible embedding {embed_tokens.shape} and hidden {hidden.shape}")
    if vocab_chunk <= 0 or sequence_chunk <= 0:
        raise ValueError("vocab_chunk and sequence_chunk must be positive")

    output_shape = targets.shape
    hidden_size = hidden.shape[-1]
    num_tokens = math.prod(output_shape)
    vocab_size = embed_tokens.shape[0]
    num_sequence_chunks = math.ceil(num_tokens / sequence_chunk)
    num_vocab_chunks = math.ceil(vocab_size / vocab_chunk)

    sequence_pad = num_sequence_chunks * sequence_chunk - num_tokens
    vocab_pad = num_vocab_chunks * vocab_chunk - vocab_size
    flat_hidden = jnp.pad(hidden.reshape(num_tokens, hidden_size), ((0, sequence_pad), (0, 0)))
    flat_targets = jnp.pad(targets.reshape(num_tokens), ((0, sequence_pad),))
    padded_embed = jnp.pad(embed_tokens, ((0, vocab_pad), (0, 0)))
    hidden_blocks = flat_hidden.reshape(num_sequence_chunks, sequence_chunk, hidden_size)
    target_blocks = flat_targets.reshape(num_sequence_chunks, sequence_chunk)
    embed_blocks = padded_embed.reshape(num_vocab_chunks, vocab_chunk, hidden_size)
    vocab_indices = jnp.arange(num_vocab_chunks, dtype=jnp.int32)

    def sequence_body(_: None, inputs: tuple[Array, Array]) -> tuple[None, Array]:
        hidden_block, target_block = inputs
        target_embed = jnp.take(embed_tokens, target_block, axis=0)
        target_logits = jnp.einsum(
            "nm,nm->n",
            hidden_block,
            target_embed,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        target_logits = _softcap(target_logits, softcap)

        def vocab_body(carry: tuple[Array, Array], inputs: tuple[Array, Array]) -> tuple[tuple[Array, Array], None]:
            running_max, running_sum = carry
            chunk_index, embed_block = inputs
            logits = _softcap(_matmul_f32(hidden_block, embed_block.T), softcap)
            valid = chunk_index * vocab_chunk + jnp.arange(vocab_chunk) < vocab_size
            logits = jnp.where(valid[None, :], logits, -jnp.inf)
            block_max = jnp.max(logits, axis=-1)
            new_max = jnp.maximum(running_max, block_max)
            new_sum = running_sum * jnp.exp(running_max - new_max)
            new_sum += jnp.sum(jnp.exp(logits - new_max[:, None]), axis=-1)
            return (new_max, new_sum), None

        initial = (
            jnp.full((sequence_chunk,), -jnp.inf, jnp.float32),
            jnp.zeros((sequence_chunk,), jnp.float32),
        )
        remat_vocab_body = jax.checkpoint(vocab_body, prevent_cse=False)
        (running_max, running_sum), _ = lax.scan(remat_vocab_body, initial, (vocab_indices, embed_blocks))
        log_normalizer = running_max + jnp.log(running_sum)
        return None, target_logits - log_normalizer

    _, logp_blocks = lax.scan(sequence_body, None, (hidden_blocks, target_blocks))
    return logp_blocks.reshape(-1)[:num_tokens].reshape(output_shape)


def _vocab_parallel_logps(
    embed_tokens: Array,
    hidden: Array,
    targets: Array,
    *,
    mesh: Mesh,
    softcap: float | None,
    vocab_chunk: int,
    sequence_chunk: int,
) -> Array:
    """Keep ``[V,H]`` sharded and combine local target/logsumexp statistics."""

    if DATA_AXIS not in mesh.axis_names:
        raise ValueError(f"vocab-parallel mesh needs axis {DATA_AXIS!r}: {mesh.axis_names}")
    if embed_tokens.shape[0] % mesh.shape[DATA_AXIS]:
        raise ValueError(f"vocab size {embed_tokens.shape[0]} must be divisible by mesh axis {mesh.shape[DATA_AXIS]}")

    def local_logps(local_embed: Array, replicated_hidden: Array, replicated_targets: Array) -> Array:
        output_shape = replicated_targets.shape
        hidden_size = replicated_hidden.shape[-1]
        token_count = math.prod(output_shape)
        local_vocab_size = local_embed.shape[0]
        sequence_blocks = math.ceil(token_count / sequence_chunk)
        vocabulary_blocks = math.ceil(local_vocab_size / vocab_chunk)
        sequence_pad = sequence_blocks * sequence_chunk - token_count
        vocabulary_pad = vocabulary_blocks * vocab_chunk - local_vocab_size
        flat_hidden = jnp.pad(replicated_hidden.reshape(token_count, hidden_size), ((0, sequence_pad), (0, 0)))
        flat_targets = jnp.pad(replicated_targets.reshape(token_count), ((0, sequence_pad),))
        padded_embed = jnp.pad(local_embed, ((0, vocabulary_pad), (0, 0)))
        hidden_blocks = flat_hidden.reshape(sequence_blocks, sequence_chunk, hidden_size)
        target_blocks = flat_targets.reshape(sequence_blocks, sequence_chunk)
        embed_blocks = padded_embed.reshape(vocabulary_blocks, vocab_chunk, hidden_size)
        block_indices = jnp.arange(vocabulary_blocks, dtype=jnp.int32)
        vocabulary_offset = lax.axis_index(DATA_AXIS) * local_vocab_size

        def sequence_body(_: None, inputs: tuple[Array, Array]) -> tuple[None, Array]:
            hidden_block, target_block = inputs
            owned = (target_block >= vocabulary_offset) & (target_block < vocabulary_offset + local_vocab_size)
            local_targets = jnp.clip(target_block - vocabulary_offset, 0, local_vocab_size - 1)
            target_embed = jnp.take(local_embed, local_targets, axis=0)
            local_target_logits = jnp.einsum(
                "nm,nm->n",
                hidden_block,
                target_embed,
                precision=lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
            target_logits = lax.psum(jnp.where(owned, _softcap(local_target_logits, softcap), 0.0), DATA_AXIS)

            def vocabulary_body(
                carry: tuple[Array, Array], inputs: tuple[Array, Array]
            ) -> tuple[tuple[Array, Array], None]:
                running_max, running_sum = carry
                block_index, embed_block = inputs
                logits = _softcap(_matmul_f32(hidden_block, embed_block.T), softcap)
                valid = block_index * vocab_chunk + jnp.arange(vocab_chunk) < local_vocab_size
                logits = jnp.where(valid[None, :], logits, -jnp.inf)
                block_max = jnp.max(logits, axis=-1)
                new_max = jnp.maximum(running_max, block_max)
                new_sum = running_sum * jnp.exp(running_max - new_max)
                new_sum += jnp.sum(jnp.exp(logits - new_max[:, None]), axis=-1)
                return (new_max, new_sum), None

            initial = (
                lax.pcast(jnp.full((sequence_chunk,), -jnp.inf, jnp.float32), DATA_AXIS, to="varying"),
                lax.pcast(jnp.zeros((sequence_chunk,), jnp.float32), DATA_AXIS, to="varying"),
            )
            remat_body = jax.checkpoint(vocabulary_body, prevent_cse=False)
            (local_max, local_sum), _ = lax.scan(remat_body, initial, (block_indices, embed_blocks))
            # The exact logsumexp derivative through this numerical shift cancels. Stop before pmax because
            # JAX 0.11.1 has no pmax transpose rule; stopping the pmax output is already too late for AD tracing.
            global_max = lax.pmax(lax.stop_gradient(local_max), DATA_AXIS)
            global_sum = lax.psum(local_sum * jnp.exp(local_max - global_max), DATA_AXIS)
            return None, target_logits - global_max - jnp.log(global_sum)

        _, blocks = lax.scan(sequence_body, None, (hidden_blocks, target_blocks))
        return blocks.reshape(-1)[:token_count].reshape(output_shape)

    mapped = jax.shard_map(
        local_logps,
        mesh=mesh,
        in_specs=(P(DATA_AXIS, None), P(), P()),
        out_specs=P(),
        axis_names={DATA_AXIS},
    )
    return mapped(embed_tokens, hidden, targets)


def per_token_logps(
    embed_tokens: Array,
    hidden: Array,
    targets: Array,
    *,
    softcap: float | None,
    vocab_chunk: int = 8192,
    sequence_chunk: int = 256,
    mesh: Mesh | None = None,
) -> Array:
    """Return target log-probabilities without materialising ``[..., vocab]`` logits.

    A multi-device ``mesh`` selects explicit vocabulary parallelism: hidden/targets
    are replicated at the shard-map boundary while the much larger embedding table
    remains vocabulary-sharded. CPU/single-device calls retain the generic streaming
    implementation.
    """

    if hidden.shape[:-1] != targets.shape:
        raise ValueError(f"hidden prefix {hidden.shape[:-1]} does not match targets {targets.shape}")
    if embed_tokens.ndim != 2 or hidden.shape[-1] != embed_tokens.shape[-1]:
        raise ValueError(f"incompatible embedding {embed_tokens.shape} and hidden {hidden.shape}")
    if vocab_chunk <= 0 or sequence_chunk <= 0:
        raise ValueError("vocab_chunk and sequence_chunk must be positive")
    if mesh is not None and mesh.size > 1:
        return _vocab_parallel_logps(
            embed_tokens,
            hidden,
            targets,
            mesh=mesh,
            softcap=softcap,
            vocab_chunk=vocab_chunk,
            sequence_chunk=sequence_chunk,
        )
    return _single_partition_logps(
        embed_tokens,
        hidden,
        targets,
        softcap=softcap,
        vocab_chunk=vocab_chunk,
        sequence_chunk=sequence_chunk,
    )


def cast_floating_tree(tree: Any, dtype: Any) -> Any:
    """Cast floating-point leaves while preserving the NamedTuple parameter structure."""

    return jax.tree.map(lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.inexact) else x, tree)


def compute_advantages(rewards: Array, group_size: int, *, normalize_std: bool = True) -> Array:
    """Compute contiguous group-relative advantages, matching Tunix 0.1.7.

    ``normalize_std=True`` uses the sample standard deviation (``ddof=1``), as Tunix GRPO does.
    Disabling it gives the Dr. GRPO reward-centering rule.
    """

    if group_size < 2:
        raise ValueError("group_size must be at least two")
    if rewards.size % group_size:
        raise ValueError(f"reward count {rewards.size} is not divisible by group_size={group_size}")
    output_shape = rewards.shape
    grouped = rewards.astype(jnp.float32).reshape(-1, group_size)
    # 先减组内锚点，再求均值/标准差，避免常量组的均值舍入残差被 1e-6 放大。
    shifted = grouped - grouped[:, :1]
    centered = shifted - shifted.mean(axis=-1, keepdims=True)
    if normalize_std:
        centered = centered / (shifted.std(axis=-1, ddof=1, keepdims=True) + 1e-6)
    return centered.reshape(output_shape)


def compute_rloo_advantages(rewards: Array, group_size: int) -> Array:
    """Subtract each completion's leave-one-out group baseline."""

    if group_size < 2:
        raise ValueError("RLOO group_size must be at least two")
    if rewards.size % group_size:
        raise ValueError(f"reward count {rewards.size} is not divisible by group_size={group_size}")
    output_shape = rewards.shape
    grouped = rewards.astype(jnp.float32).reshape(-1, group_size)
    shifted = grouped - grouped[:, :1]
    baseline = (shifted.sum(axis=-1, keepdims=True) - shifted) / (group_size - 1)
    return (shifted - baseline).reshape(output_shape)


def aggregate_token_loss(
    per_token_loss: Array,
    completion_mask: Array,
    mode: str,
    *,
    token_scale: float | None = None,
) -> Array:
    """Aggregate masked token values with the modes used by GRPO-family objectives."""

    if per_token_loss.shape != completion_mask.shape or per_token_loss.ndim != 2:
        raise ValueError(
            f"per-token loss and mask must share [B, N], got {per_token_loss.shape}/{completion_mask.shape}"
        )
    values = per_token_loss.astype(jnp.float32)
    mask = completion_mask.astype(jnp.float32)
    row_tokens = mask.sum(axis=-1)
    nonempty_rows = jnp.maximum((row_tokens > 0).sum(), 1)
    if mode == "token-mean":
        return (values * mask).sum() / jnp.maximum(mask.sum(), 1)
    if mode in ("seq-mean-token-mean", "sequence-mean-token-mean"):
        row_loss = (values * mask).sum(axis=-1) / jnp.maximum(row_tokens, 1)
        return row_loss.sum() / nonempty_rows
    if mode in ("seq-mean-token-sum", "sequence-mean-token-sum-norm"):
        return (values * mask).sum() / nonempty_rows
    if mode in ("seq-mean-token-scale", "sequence-mean-token-scale"):
        scale = float(per_token_loss.shape[-1]) if token_scale is None else token_scale
        if scale <= 0:
            raise ValueError("token_scale must be positive")
        row_loss = (values * mask).sum(axis=-1) / scale
        return row_loss.sum() / nonempty_rows
    raise ValueError(f"unsupported loss aggregation mode: {mode}")


def _masked_mean(values: Array, mask: Array) -> Array:
    cast_mask = mask.astype(values.dtype)
    return (values * cast_mask).sum() / jnp.maximum(cast_mask.sum(), 1)


def _stable_k3(difference: Array) -> Array:
    """避免接近零时的相减误差，以及未选中多项式分支的梯度溢出。"""

    small = jnp.abs(difference) <= 0.01
    local = jnp.where(small, difference, 0.0)
    polynomial = jnp.square(local) * (0.5 + local * (1.0 / 6.0 + local * (1.0 / 24.0 + local / 120.0)))
    outer = lax.expm1(difference, accuracy=lax.AccuracyMode.HIGHEST) - difference
    return jnp.where(small, polynomial, outer)


def _kl_estimate(
    policy_logps: Array,
    reference_logps: Array,
    estimator: str,
    *,
    clamp_value: float | None = None,
) -> Array:
    if estimator in ("k3", "low_var_kl"):
        difference = reference_logps - policy_logps
        if clamp_value is not None:
            # 比此值更大的下一个FP32数，其指数已经超出FP32范围。
            # 先隔离溢出分支，避免最终clip的零梯度乘上无穷大而产生NaN。
            max_log = jnp.nextafter(jnp.log(jnp.finfo(jnp.float32).max), -jnp.inf)
            overflow = difference > max_log
            safe_difference = jnp.where(overflow, 0.0, difference)
            estimate = jnp.clip(_stable_k3(safe_difference), -clamp_value, clamp_value)
            return jnp.where(overflow, clamp_value, estimate)
        estimate = _stable_k3(difference)
    elif estimator in ("k1", "kl"):
        estimate = policy_logps - reference_logps
    elif estimator in ("k2", "mse_kl"):
        estimate = 0.5 * jnp.square(policy_logps - reference_logps)
    else:
        raise ValueError(f"unsupported KL estimator: {estimator}")
    return estimate if clamp_value is None else jnp.clip(estimate, -clamp_value, clamp_value)


def grpo_loss(
    policy_logps: Array,
    old_logps: Array | None,
    reference_logps: Array | None,
    advantages: Array,
    completion_mask: Array,
    *,
    eps_low: float = 0.2,
    eps_high: float = 0.2,
    beta: float = 0.0,
    kl_estimator: Literal["k1", "k2", "k3", "kl", "mse_kl", "low_var_kl"] = "k3",
    kl_clamp_value: float | None = None,
    agg_mode: str = "sequence-mean-token-mean",
    ratio_level: Literal["token", "sequence"] = "token",
    dual_clip_c: float | None = None,
    token_scale: float | None = None,
    clip_policy: bool = True,
    sampler_is_weights: Array | None = None,
) -> tuple[Array, GRPOMetrics]:
    """Composable GRPO-family objective over already selected token log-probabilities.

    With ``old_logps=None``, the old policy is ``stop_gradient(policy_logps)``. This is the
    single-update GRPO path: the ratio is exactly one in value while retaining the policy gradient.
    ``ratio_level='sequence'`` applies Tunix 0.1.7's GSPO-token stop-gradient construction.
    """

    if policy_logps.shape != completion_mask.shape or policy_logps.ndim != 2:
        raise ValueError(
            f"policy logps and completion mask must share [B, N], got {policy_logps.shape}/{completion_mask.shape}"
        )
    if advantages.shape not in ((policy_logps.shape[0],), policy_logps.shape):
        raise ValueError(f"advantages must have shape [B] or [B, N], got {advantages.shape}")
    if old_logps is not None and old_logps.shape != policy_logps.shape:
        raise ValueError(f"old logps shape {old_logps.shape} differs from policy logps {policy_logps.shape}")
    if reference_logps is not None and reference_logps.shape != policy_logps.shape:
        raise ValueError(
            f"reference logps shape {reference_logps.shape} differs from policy logps {policy_logps.shape}"
        )
    if sampler_is_weights is not None and sampler_is_weights.shape != policy_logps.shape:
        raise ValueError("sampler IS权重必须与policy logps共享[B,N]")
    if eps_low < 0 or eps_high < 0 or beta < 0:
        raise ValueError("eps_low, eps_high, and beta must be non-negative")
    if kl_clamp_value is not None and (
        not math.isfinite(kl_clamp_value) or kl_clamp_value <= 0 or kl_clamp_value > float(jnp.finfo(jnp.float32).max)
    ):
        raise ValueError("kl_clamp_value must be a positive finite float32 value")
    if dual_clip_c is not None and dual_clip_c <= 0:
        raise ValueError("dual_clip_c must be positive")
    if ratio_level not in ("token", "sequence"):
        raise ValueError(f"unsupported ratio level: {ratio_level}")
    if beta and reference_logps is None:
        raise ValueError("non-zero beta requires reference_logps")

    policy = policy_logps.astype(jnp.float32)
    old = lax.stop_gradient(policy) if old_logps is None else old_logps.astype(jnp.float32)
    mask = completion_mask.astype(jnp.bool_)
    log_ratio = policy - old
    ppo_kl = _masked_mean(-log_ratio, mask)
    log_ratio = jnp.clip(log_ratio, -20.0, 20.0)
    if ratio_level == "sequence":
        sequence_ratio = (log_ratio * mask).sum(axis=-1) / jnp.maximum(mask.sum(axis=-1), 1)
        log_ratio = policy - lax.stop_gradient(policy) + lax.stop_gradient(sequence_ratio[:, None])
        log_ratio = jnp.minimum(log_ratio, 10.0)
    ratio = jnp.exp(log_ratio)
    advantage = advantages.astype(jnp.float32)
    if advantage.ndim == 1:
        advantage = advantage[:, None]
    unclipped = -advantage * ratio
    clipped = -advantage * jnp.clip(ratio, 1.0 - eps_low, 1.0 + eps_high)
    per_token_policy = jnp.maximum(unclipped, clipped) if clip_policy else unclipped
    pg_clipfrac = (
        _masked_mean((clipped > unclipped).astype(jnp.float32), mask) if clip_policy else jnp.zeros((), jnp.float32)
    )
    dual_bound = per_token_policy if dual_clip_c is None else -dual_clip_c * advantage
    dual_applies = (per_token_policy > dual_bound) & (advantage < 0.0)
    dual_clipfrac = _masked_mean(jnp.broadcast_to(dual_applies, policy.shape).astype(jnp.float32), mask)
    per_token_policy = jnp.where(advantage < 0.0, jnp.minimum(dual_bound, per_token_policy), per_token_policy)
    if sampler_is_weights is not None:
        # 冻结的采样修正只乘policy目标，不乘独立KL正则项。
        per_token_policy = per_token_policy * lax.stop_gradient(sampler_is_weights.astype(jnp.float32))
    policy_loss = aggregate_token_loss(per_token_policy, mask, agg_mode, token_scale=token_scale)

    if reference_logps is None:
        kl_loss = jnp.zeros((), jnp.float32)
    else:
        per_token_kl = _kl_estimate(
            policy, reference_logps.astype(jnp.float32), kl_estimator, clamp_value=kl_clamp_value
        )
        kl_loss = aggregate_token_loss(per_token_kl, mask, agg_mode, token_scale=token_scale)
    loss = policy_loss + beta * kl_loss
    broadcast_advantage = jnp.broadcast_to(advantage, policy.shape)
    ratio_min = jnp.min(jnp.where(mask, ratio, jnp.inf))
    ratio_max = jnp.max(jnp.where(mask, ratio, 0.0))
    advantage_min = jnp.min(jnp.where(mask, broadcast_advantage, jnp.inf))
    advantage_max = jnp.max(jnp.where(mask, broadcast_advantage, -jnp.inf))
    metrics = GRPOMetrics(
        loss=loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        ppo_kl=ppo_kl,
        pg_clipfrac=pg_clipfrac,
        dual_clipfrac=dual_clipfrac,
        ratio_mean=_masked_mean(ratio, mask),
        ratio_min=ratio_min,
        ratio_max=ratio_max,
        log_ratio_abs_mean=_masked_mean(jnp.abs(log_ratio), mask),
        advantage_abs_mean=_masked_mean(jnp.abs(broadcast_advantage), mask),
        advantage_min=advantage_min,
        advantage_max=advantage_max,
        completion_tokens=mask.sum(),
        sampled_entropy=-_masked_mean(policy, mask),
    )
    return loss, metrics


def _check_microbatch_size(rows: int, size: int, mesh: Mesh | None) -> None:
    if size <= 0 or rows % size or (mesh is not None and size % mesh.size):
        raise ValueError("microbatch_size 必须为行数的正约数，且能被设备数整除")


def _slice_microbatch(batch: Any, index: Array, size: int, mesh: Mesh | None) -> Any:
    def take(value: Array) -> Array:
        sliced = lax.dynamic_slice_in_dim(value, index * size, size, axis=0)
        return sliced if mesh is None else constrain_batch(sliced, mesh)

    return jax.tree.map(take, batch)


def _loss_denominator(mask: Array, mode: str) -> Array:
    count = mask.sum() if mode == "token-mean" else jnp.any(mask, axis=-1).sum()
    return jnp.maximum(count, 1).astype(jnp.float32)


def trainer_completion_logps(
    params: Gemma4TextParams,
    prompt_ids: Array,
    prompt_mask: Array,
    completion_ids: Array,
    completion_mask: Array,
    *,
    config: Gemma4TextConfig,
    compute_dtype: Any | None = None,
    vocab_chunk: int = 8192,
    sequence_chunk: int = 256,
    remat_layers: bool = False,
    mesh: Mesh | None = None,
    microbatch_size: int | None = None,
    lora: Gemma4LoRAParams | None = None,
) -> Array:
    """Recompute policy log-probabilities for fixed completion tokens.

    Prompt rows are left-padded and completions are right-padded. Completion token ``j`` is
    predicted by hidden position ``prompt_width + j - 1``. Invalid completion slots are returned
    as zero so this output has the same masked-storage convention as rollout log-probabilities.
    """

    if prompt_ids.ndim != 2 or prompt_mask.shape != prompt_ids.shape:
        raise ValueError(f"prompt ids/mask must share [B, P], got {prompt_ids.shape}/{prompt_mask.shape}")
    if completion_ids.ndim != 2 or completion_mask.shape != completion_ids.shape:
        raise ValueError(f"completion ids/mask must share [B, N], got {completion_ids.shape}/{completion_mask.shape}")
    if prompt_ids.shape[0] != completion_ids.shape[0]:
        raise ValueError(f"prompt/completion batch sizes differ: {prompt_ids.shape[0]} != {completion_ids.shape[0]}")
    if prompt_ids.shape[1] < 1 or completion_ids.shape[1] < 1:
        raise ValueError("prompt and completion widths must both be positive")

    rows = prompt_ids.shape[0]
    if microbatch_size is not None:
        _check_microbatch_size(rows, microbatch_size, mesh)
        if microbatch_size < rows:
            tokens = (prompt_ids, prompt_mask, completion_ids, completion_mask)

            def logps_slice(index: Array) -> Array:
                batch = _slice_microbatch(tokens, index, microbatch_size, mesh)
                return trainer_completion_logps(
                    params,
                    *batch,
                    config=config,
                    compute_dtype=compute_dtype,
                    vocab_chunk=vocab_chunk,
                    sequence_chunk=sequence_chunk,
                    remat_layers=remat_layers,
                    mesh=mesh,
                    lora=lora,
                )

            return cast(Array, lax.map(logps_slice, jnp.arange(rows // microbatch_size)).reshape(completion_ids.shape))

    prompt_mask = prompt_mask.astype(jnp.bool_)
    completion_mask = completion_mask.astype(jnp.bool_)
    input_ids = jnp.concatenate((prompt_ids, completion_ids), axis=-1)
    attention_mask = jnp.concatenate((prompt_mask, completion_mask), axis=-1)
    position_ids = jnp.maximum(jnp.cumsum(attention_mask, axis=-1) - 1, 0).astype(jnp.int32)
    compute_params = cast_floating_tree(params, compute_dtype) if compute_dtype is not None else params
    with jax.named_scope("grpo_model_forward"):
        hidden, _ = forward_gemma4_text(
            compute_params,
            input_ids,
            position_ids,
            config=config,
            attention_mask=attention_mask,
            remat_layers=remat_layers,
            lora=lora,
        )
    prompt_width, completion_width = prompt_ids.shape[1], completion_ids.shape[1]
    prediction_hidden = hidden[:, prompt_width - 1 : prompt_width + completion_width - 1]
    safe_targets = jnp.where(completion_mask, completion_ids, 0)
    logprob_scope = "grpo_vocab_parallel_logprob" if mesh is not None and mesh.size > 1 else "grpo_streaming_logprob"
    with jax.named_scope(logprob_scope):
        logps = per_token_logps(
            compute_params.embed_tokens,
            prediction_hidden,
            safe_targets,
            softcap=config.final_logit_softcapping,
            vocab_chunk=vocab_chunk,
            sequence_chunk=sequence_chunk,
            mesh=mesh,
        )
    return jnp.where(completion_mask, logps, 0.0)


def truncated_importance_weights(
    old_trainer_logps: Array,
    behavior_logps: Array,
    completion_mask: Array,
    *,
    cap: float,
) -> Array:
    """固定target-old / behavior的逐token截断权重；不是完整轨迹的无偏IS。"""
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("IS cap必须为正的有限数")
    if (
        old_trainer_logps.ndim != 2
        or old_trainer_logps.shape != behavior_logps.shape
        or old_trainer_logps.shape != completion_mask.shape
    ):
        raise ValueError("old/behavior logps与completion mask必须共享[B,N]")
    difference = old_trainer_logps.astype(jnp.float32) - behavior_logps.astype(jnp.float32)
    # 在exp之前施加上限；不额外引入未声明的下限裁剪。
    # TPU默认exp在真实TIS输入上偏差约3e-6；只固定此冻结权重的指数精度。
    weights = lax.exp(
        jnp.minimum(difference, jnp.log(jnp.asarray(cap, jnp.float32))), accuracy=lax.AccuracyMode.HIGHEST
    )
    return lax.stop_gradient(jnp.where(completion_mask, weights, 0.0))


def grpo_train_step[Params](
    state: TrainState[Params],
    prompt_ids: Array,
    prompt_mask: Array,
    completion_ids: Array,
    completion_mask: Array,
    advantages: Array,
    old_logps: Array | None,
    reference_logps: Array | None,
    *,
    config: Gemma4TextConfig,
    optimizer: optax.GradientTransformation,
    trainable_mask: Gemma4TextParams | None,
    compute_dtype: Any = jnp.bfloat16,
    vocab_chunk: int = 8192,
    sequence_chunk: int = 256,
    remat_layers: bool = False,
    mesh: Mesh | None = None,
    eps_low: float = 0.2,
    eps_high: float = 0.2,
    beta: float = 0.0,
    kl_estimator: Literal["k1", "k2", "k3", "kl", "mse_kl", "low_var_kl"] = "k3",
    kl_clamp_value: float | None = None,
    agg_mode: str = "sequence-mean-token-mean",
    ratio_level: Literal["token", "sequence"] = "token",
    dual_clip_c: float | None = None,
    token_scale: float | None = None,
    microbatch_size: int | None = None,
    clip_policy: bool = True,
    loss_mask: Array | None = None,
    sampler_is_weights: Array | None = None,
    use_current_policy_as_old: Array | None = None,
    behavior_logps: Array | None = None,
    sampler_is_cap: float | None = None,
    frozen_base: Gemma4TextParams | None = None,
    lora_config: LoRAConfig | None = None,
) -> tuple[TrainState[Params], GRPOTrainMetrics]:
    """固定token的一次FP32 master更新；可从同一联合图捕获轮内冻结old概率。"""

    if lora_config is None:
        if frozen_base is not None or trainable_mask is None or not isinstance(state.params_f32, Gemma4TextParams):
            raise ValueError("基础模型训练需要完整参数和trainable mask，不能附带LoRA冻结base")
    elif frozen_base is None or trainable_mask is not None or not isinstance(state.params_f32, Gemma4LoRAParams):
        raise ValueError("LoRA训练需要独立适配器state、显式冻结base且不能使用基础参数mask")

    if use_current_policy_as_old is not None and (
        old_logps is None or jnp.shape(use_current_policy_as_old) != () or use_current_policy_as_old.dtype != jnp.bool_
    ):
        raise ValueError("联合old捕获需要显式old数组和动态bool标量")
    if behavior_logps is not None:
        if sampler_is_cap is None or sampler_is_weights is not None:
            raise ValueError("behavior logps需要IS cap，不能同时提供预计算IS权重")
        if behavior_logps.shape != completion_mask.shape:
            raise ValueError("behavior logps必须与completion mask共享[B,N]")
    elif sampler_is_cap is not None:
        raise ValueError("IS cap需要behavior logps")

    if loss_mask is not None and loss_mask.shape != completion_mask.shape:
        raise ValueError("loss mask必须与completion mask共享[B,N]")
    effective_loss_mask = completion_mask if loss_mask is None else completion_mask & loss_mask
    tokens = (
        prompt_ids,
        prompt_mask,
        completion_ids,
        completion_mask,
        advantages,
        old_logps,
        reference_logps,
        effective_loss_mask,
        sampler_is_weights,
        behavior_logps,
    )

    def loss_from_logps(policy: Array, batch: Any) -> tuple[Array, GRPOMetrics]:
        old = batch[5]
        if use_current_policy_as_old is not None:
            # 只在首个update取当前联合前向；old分母与IS均不得反向传播。
            old = jnp.where(use_current_policy_as_old, lax.stop_gradient(policy), old)
        is_weights = batch[8]
        if sampler_is_cap is not None:
            target_old = lax.stop_gradient(policy) if old is None else old
            is_weights = truncated_importance_weights(target_old, batch[9], batch[3], cap=sampler_is_cap)
        return grpo_loss(
            policy,
            old,
            batch[6],
            batch[4],
            batch[7],
            eps_low=eps_low,
            eps_high=eps_high,
            beta=beta,
            kl_estimator=kl_estimator,
            kl_clamp_value=kl_clamp_value,
            agg_mode=agg_mode,
            ratio_level=ratio_level,
            dual_clip_c=dual_clip_c,
            token_scale=token_scale,
            clip_policy=clip_policy,
            sampler_is_weights=is_weights,
        )

    def objective(params: Params, batch: Any, weight: Array) -> tuple[Array, Array]:
        lora = None
        if lora_config is None:
            effective_params = jax.tree.map(
                lambda value, is_trainable: value if is_trainable else lax.stop_gradient(value),
                params,
                trainable_mask,
            )
        else:
            # AD只接受A/B为变量；base显式作为调用参数传入，不建立基础模型梯度或Adam槽。
            effective_params = jax.tree.map(lax.stop_gradient, frozen_base)
            lora = prepare_lora_params(cast(Gemma4LoRAParams, params), lora_config)
        policy_logps = trainer_completion_logps(
            effective_params,
            *batch[:4],
            config=config,
            compute_dtype=compute_dtype,
            vocab_chunk=vocab_chunk,
            sequence_chunk=sequence_chunk,
            remat_layers=remat_layers,
            mesh=mesh,
            lora=lora,
        )
        loss, _ = loss_from_logps(policy_logps, batch)
        return loss * weight, policy_logps

    with jax.named_scope("grpo_forward_backward"):
        rows = prompt_ids.shape[0]
        if microbatch_size is not None:
            _check_microbatch_size(rows, microbatch_size, mesh)
        if microbatch_size is None or microbatch_size == rows:
            (_, policy_logps), grads = jax.value_and_grad(objective, has_aux=True)(
                state.params_f32, tokens, jnp.ones((), jnp.float32)
            )
        else:
            # 优势在完整 G 组上先计算；这里只分行求梯度。按全局分母缩放，不能平均各微批均值。
            denominator = _loss_denominator(effective_loss_mask, agg_mode)

            def accumulate(total: Params, index: Array) -> tuple[Params, Array]:
                batch = _slice_microbatch(tokens, index, microbatch_size, mesh)
                weight = _loss_denominator(batch[7], agg_mode) / denominator
                (_, logps), gradient = jax.value_and_grad(objective, has_aux=True)(state.params_f32, batch, weight)
                return jax.tree.map(jnp.add, total, gradient), logps

            zero_grads = jax.tree.map(jnp.zeros_like, state.params_f32)
            grads, logps_blocks = lax.scan(accumulate, zero_grads, jnp.arange(rows // microbatch_size))
            policy_logps = logps_blocks.reshape(completion_ids.shape)
        # 仅保留 [B,N] 概率供全局指标归约，不保留各微批激活或 [B,N,V] logits。
        loss, loss_metrics = loss_from_logps(policy_logps, tokens)
    with jax.named_scope("optimizer_update"):
        grad_norm = optax.tree.norm(grads)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params_f32)
        params_f32 = optax.apply_updates(state.params_f32, updates)
    new_state = TrainState(params_f32, opt_state, state.step + 1)
    return new_state, GRPOTrainMetrics(
        loss_metrics._replace(loss=loss), grad_norm, policy_logps if use_current_policy_as_old is not None else None
    )


def sft_loss(
    params: Gemma4TextParams,
    input_ids: Array,
    labels: Array,
    *,
    config: Gemma4TextConfig,
    attention_mask: Array | None = None,
    position_ids: Array | None = None,
    compute_dtype: Any | None = None,
    vocab_chunk: int = 8192,
    sequence_chunk: int = 256,
    remat_layers: bool = False,
    mesh: Mesh | None = None,
) -> Array:
    """Mean next-token negative log-likelihood; ``labels == -100`` tokens are ignored."""

    if input_ids.shape != labels.shape or input_ids.ndim != 2:
        raise ValueError(f"input_ids and labels must have the same [B, S] shape: {input_ids.shape}, {labels.shape}")
    if input_ids.shape[1] < 2:
        raise ValueError("SFT sequences need at least two tokens")
    if attention_mask is None:
        attention_mask = jnp.ones_like(input_ids, dtype=jnp.bool_)
    else:
        attention_mask = attention_mask.astype(jnp.bool_)
    if position_ids is None:
        position_ids = jnp.maximum(jnp.cumsum(attention_mask, axis=-1) - 1, 0).astype(jnp.int32)

    compute_params = cast_floating_tree(params, compute_dtype) if compute_dtype is not None else params
    with jax.named_scope("model_forward"):
        hidden, _ = forward_gemma4_text(
            compute_params,
            input_ids,
            position_ids,
            config=config,
            attention_mask=attention_mask,
            remat_layers=remat_layers,
        )
    targets = labels[:, 1:]
    loss_mask = (targets != -100) & attention_mask[:, 1:]
    safe_targets = jnp.where(loss_mask, targets, 0)
    logprob_scope = "vocab_parallel_logprob" if mesh is not None and mesh.size > 1 else "streaming_logprob"
    with jax.named_scope(logprob_scope):
        logps = per_token_logps(
            compute_params.embed_tokens,
            hidden[:, :-1],
            safe_targets,
            softcap=config.final_logit_softcapping,
            vocab_chunk=vocab_chunk,
            sequence_chunk=sequence_chunk,
            mesh=mesh,
        )
    token_count = jnp.maximum(loss_mask.sum(), 1)
    return -jnp.where(loss_mask, logps, 0.0).sum() / token_count


def make_trainable_mask(params: Gemma4TextParams, *, freeze_embeddings: bool) -> Gemma4TextParams:
    """Boolean optimizer mask; checkpoint buffers are always frozen."""

    mask = cast(Gemma4TextParams, jax.tree.map(lambda _: True, params))
    layers = tuple(layer._replace(layer_scalar=cast(Any, False)) for layer in mask.layers)
    embed_tokens_per_layer = (
        False if freeze_embeddings and mask.embed_tokens_per_layer is not None else mask.embed_tokens_per_layer
    )
    return cast(
        Gemma4TextParams,
        mask._replace(
            embed_tokens=cast(Any, False if freeze_embeddings else mask.embed_tokens),
            embed_tokens_per_layer=cast(Any, embed_tokens_per_layer),
            layers=layers,
        ),
    )


def make_optimizer(
    params: Gemma4TextParams,
    *,
    learning_rate: float,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    freeze_embeddings: bool = False,
    gradient_accumulation_steps: int = 1,
) -> tuple[optax.GradientTransformation, Gemma4TextParams]:
    """AdamW with global clipping, optional frozen lookup tables, and ``MultiSteps`` accumulation."""

    if learning_rate <= 0 or max_grad_norm <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("learning_rate, max_grad_norm, and gradient_accumulation_steps must be positive")
    trainable = make_trainable_mask(params, freeze_embeddings=freeze_embeddings)
    frozen = jax.tree.map(lambda value: not value, trainable)
    base = optax.chain(
        optax.masked(optax.set_to_zero(), frozen),
        optax.clip_by_global_norm(max_grad_norm),
        optax.masked(optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay), trainable),
    )
    optimizer: optax.GradientTransformation = base
    if gradient_accumulation_steps > 1:
        optimizer = optax.MultiSteps(base, every_k_schedule=gradient_accumulation_steps)
    return optimizer, trainable


def init_train_state[Params](params_f32: Params, optimizer: optax.GradientTransformation) -> TrainState[Params]:
    return TrainState(params_f32, optimizer.init(params_f32), jnp.zeros((), jnp.int32))


def train_step(
    state: TrainState,
    input_ids: Array,
    labels: Array,
    attention_mask: Array,
    *,
    config: Gemma4TextConfig,
    optimizer: optax.GradientTransformation,
    trainable_mask: Gemma4TextParams,
    compute_dtype: Any = jnp.bfloat16,
    vocab_chunk: int = 8192,
    sequence_chunk: int = 256,
    remat_layers: bool = False,
    mesh: Mesh | None = None,
) -> tuple[TrainState, TrainMetrics]:
    """One SFT step over f32 master parameters, optionally rematerializing decoder layers."""

    def objective(params: Gemma4TextParams) -> Array:
        effective_params = jax.tree.map(
            lambda value, is_trainable: value if is_trainable else lax.stop_gradient(value),
            params,
            trainable_mask,
        )
        return sft_loss(
            effective_params,
            input_ids,
            labels,
            config=config,
            attention_mask=attention_mask,
            compute_dtype=compute_dtype,
            vocab_chunk=vocab_chunk,
            sequence_chunk=sequence_chunk,
            remat_layers=remat_layers,
            mesh=mesh,
        )

    with jax.named_scope("sft_forward_backward"):
        loss, grads = jax.value_and_grad(objective)(state.params_f32)
    with jax.named_scope("optimizer_update"):
        grad_norm = optax.tree.norm(grads)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params_f32)
        params_f32 = optax.apply_updates(state.params_f32, updates)
    return TrainState(params_f32, opt_state, state.step + 1), TrainMetrics(loss, grad_norm)
