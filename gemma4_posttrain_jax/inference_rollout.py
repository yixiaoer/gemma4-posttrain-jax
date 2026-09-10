"""将独立推理引擎适配到训练 Rollout 协议，显式保存版本和批次随机状态。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding

from .inference_rng import BatchRngState
from .model import Gemma4TextConfig, Gemma4TextParams
from .sampler import RolloutBatch, SamplerConfig
from .sharding import batch_spec

if TYPE_CHECKING:
    from .inference_runtime import EngineBatch, EngineSampling


class InferenceBackend(Protocol):
    """两种部署共享的参数提交与生成边界；实现负责实际运输与资源所有权。"""

    def sync_params(self, params: Gemma4TextParams, model_config: Gemma4TextConfig, version: int) -> dict[str, Any]: ...

    def generate(
        self,
        prompts: list[list[int]],
        batch_rng: BatchRngState,
        sampling: EngineSampling,
        *,
        expected_version: int | None = None,
    ) -> EngineBatch: ...

    def close(self) -> dict[str, Any]: ...


def token_prompts(prompt_ids: Any, prompt_mask: Any) -> list[list[int]]:
    """只接受本训练入口的左填充；拒绝内部空洞和全padding请求。"""
    ids, mask = np.asarray(prompt_ids), np.asarray(prompt_mask)
    if ids.ndim != 2 or mask.shape != ids.shape or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("prompt ids 必须为二维整数，mask 形状必须完全一致")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("prompt mask 只能包含0和1")
    active = mask.astype(bool)
    if not np.all(active.any(axis=1)) or np.any(active[:, :-1] & ~active[:, 1:]):
        raise ValueError("只支持非空且没有内部空洞的左填充prompt")
    if np.any(ids[active] < 0):
        raise ValueError("有效prompt token不能为负数")
    return [row[valid].astype(np.int64).tolist() for row, valid in zip(ids, active, strict=True)]


class InferenceRollout:
    """参数运输由独立映射器负责，生成结果放回指定训练mesh。"""

    def __init__(
        self,
        runtime: InferenceBackend,
        config: Gemma4TextConfig,
        *,
        sampler_config: SamplerConfig,
        mesh: Mesh | None = None,
    ) -> None:
        self.runtime = runtime
        self.config = config
        self.sampler_config = sampler_config
        self.mesh = mesh
        self.policy_version: int | None = None
        self.last_generation: dict[str, Any] | None = None
        self.last_update: dict[str, Any] | None = None
        self.last_proposal_logps: Array | None = None

    def update_params(self, params: Gemma4TextParams, *, version: int | None = None) -> None:
        version = 0 if version is None and self.policy_version is None else version
        if version is None:
            assert self.policy_version is not None
            version = self.policy_version + 1
        if (
            type(version) is not int
            or version < 0
            or (self.policy_version is not None and version <= self.policy_version)
        ):
            raise ValueError("每次真实权重更新须提交递增的非负policy版本")
        try:
            update = self.runtime.sync_params(params, self.config, version)
            if (
                update.get("complete") is not True
                or type(update.get("version")) is not int
                or update["version"] != version
            ):
                raise RuntimeError("完整参数接口没有确认本次设备版本提交")
        except BaseException:
            self.policy_version = None
            self.last_generation = None
            self.last_proposal_logps = None
            self.last_update = None
            raise
        self.policy_version = version
        self.last_update = update

    def generate(self, prompt_ids: Array, prompt_mask: Array, *, key: Array | None = None) -> RolloutBatch:
        from .inference_runtime import EngineSampling

        if self.policy_version is None:
            raise RuntimeError("先同步实际训练参数，不能默认引擎checkpoint等于当前训练策略")
        sampler = self.sampler_config
        if sampler.top_p != 1.0:
            raise ValueError("tpu-inference训练适配尚未验证Top-p及其proposal概率，请使用纯JAX生成入口")
        if prompt_ids.shape[1] != sampler.max_prompt_len:
            raise ValueError("prompt宽度与已声明采样配置不一致")
        prompts = token_prompts(jax.device_get(prompt_ids), jax.device_get(prompt_mask))
        if key is None:
            raise ValueError("训练引擎必须显式传入本批key，不能暗中回到默认seed")
        words = np.asarray(jax.device_get(jax.random.key_data(key)))
        if words.dtype != np.uint32 or words.shape != (2,):
            raise ValueError("训练随机key必须提供两个uint32字")
        batch_rng = BatchRngState(str(jax.random.key_impl(key)), (int(words[0]), int(words[1])))
        generated = self.runtime.generate(
            prompts,
            batch_rng,
            EngineSampling(
                max_tokens=sampler.max_new_tokens,
                temperature=sampler.temperature,
                top_k=sampler.top_k,
                top_p=sampler.top_p,
                eos_ids=sampler.eos_ids,
            ),
            expected_version=self.policy_version,
        )
        if generated.policy_version != self.policy_version or len(generated.rows) != len(prompts):
            raise RuntimeError("引擎返回的策略版本或行数不符")
        if generated.metadata.get("batch_rng_install", {}).get("installed") != batch_rng.to_dict():
            raise RuntimeError("引擎实际batch key与本次请求状态不同")
        shape = (len(prompts), sampler.max_new_tokens)
        completion_ids = np.full(shape, self.config.pad_token_id, dtype=np.int32)
        completion_mask = np.zeros(shape, dtype=bool)
        raw_logps = np.zeros(shape, dtype=np.float32)
        proposal_logps = np.zeros(shape, dtype=np.float32)
        lengths = np.zeros(len(prompts), dtype=np.int32)
        proposal_available = True
        stops = []
        for index, row in enumerate(generated.rows):
            if list(row.prompt_token_ids) != prompts[index]:
                raise RuntimeError("引擎请求顺序或原prompt发生变化")
            tokens = list(row.token_ids)
            count = len(tokens)
            if not 0 < count <= sampler.max_new_tokens or len(row.raw_logprobs) != count:
                raise RuntimeError("引擎返回无效长度或缺少逐token原始策略logprob")
            if any(token < 0 or token >= self.config.vocab_size for token in tokens):
                raise RuntimeError("引擎返回词表范围外的token")
            if any(token in sampler.eos_ids for token in tokens[:-1]):
                raise RuntimeError("引擎保留了首个EOS之后的token，与训练mask协议不符")
            ended_with_eos = tokens[-1] in sampler.eos_ids
            if row.finish_reason not in ("stop", "length"):
                raise RuntimeError("引擎请求没有正常完成，不能作为训练样本")
            if count < sampler.max_new_tokens and (not ended_with_eos or row.finish_reason != "stop"):
                raise RuntimeError("引擎在预算以内过早停止，且不是已声明EOS")
            if row.finish_reason == "stop" and (not ended_with_eos or row.stop_reason not in (None, tokens[-1])):
                raise RuntimeError("停止原因与已声明EOS不一致")
            if row.finish_reason == "length" and (count != sampler.max_new_tokens or row.stop_reason is not None):
                raise RuntimeError("长度停止与实际token数或stop_reason不一致")
            if (
                sampler.temperature == 1.0
                and sampler.top_k <= 0
                and (
                    row.proposal_logprobs is None
                    or not np.array_equal(np.asarray(row.proposal_logprobs), np.asarray(row.raw_logprobs))
                )
            ):
                raise RuntimeError("完整策略T1采样的proposal必须等于原始policy概率")
            if any(not math.isfinite(value) or value > 1e-5 for value in row.raw_logprobs):
                raise RuntimeError("原始策略logprob必须有限且不大于0")
            completion_ids[index, :count] = tokens
            completion_mask[index, :count] = True
            raw_logps[index, :count] = row.raw_logprobs
            lengths[index] = count
            if row.proposal_logprobs is None:
                proposal_available = False
            else:
                if len(row.proposal_logprobs) != count or any(
                    not math.isfinite(value) or value > 1e-5 for value in row.proposal_logprobs
                ):
                    raise RuntimeError("实际proposal logprob长度或概率范围不符")
                proposal_logps[index, :count] = row.proposal_logprobs
            stops.append({"finish_reason": row.finish_reason, "stop_reason": row.stop_reason})

        def put(values: np.ndarray) -> Array:
            if self.mesh is None:
                return jnp.asarray(values)
            return cast(Array, jax.device_put(values, NamedSharding(self.mesh, batch_spec(values.ndim))))

        self.last_proposal_logps = put(proposal_logps) if proposal_available else None
        self.last_generation = {
            "policy_version": self.policy_version,
            "rng_protocol": batch_rng.protocol,
            "batch_rng": batch_rng.to_dict(),
            "proposal_logprobs_available": proposal_available,
            "stops": stops,
            "rows": [
                {
                    "prompt_token_ids": list(row.prompt_token_ids),
                    "token_ids": list(row.token_ids),
                    "raw_logprobs": list(row.raw_logprobs),
                    "proposal_logprobs": None if row.proposal_logprobs is None else list(row.proposal_logprobs),
                    "finish_reason": row.finish_reason,
                    "stop_reason": row.stop_reason,
                }
                for row in generated.rows
            ],
            "engine": generated.metadata,
        }
        return RolloutBatch(prompt_ids, put(completion_ids), put(completion_mask), put(raw_logps), put(lengths))

    def close(self) -> dict[str, Any]:
        return self.runtime.close()
