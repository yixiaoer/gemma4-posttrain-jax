"""可选 tpu-inference 研究后端：同步生成、真实参数更新与版本化生命周期。

模块导入只依赖标准库；构造引擎才导入 JAX、Flax 与 vLLM。仅支持单进程指定芯片。
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import math
import multiprocessing
import os
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from .inference_rng import BatchRngState, install_runner_rng, read_runner_rng

PolicyVersion = int | str | None
EXPECTED_VERSIONS = {
    "jax": "0.11.0",
    "jaxlib": "0.11.0",
    "libtpu": "0.0.44",
    "tpu-inference": "0.28.0",
    "vllm-tpu": "0.28.0",
}
EXPECTED_LIBTPU_SHA256 = "7dcb7c90edfc9cdb790479db099da2dbecba647afdbd842a2e0e0fba20b192fe"
EXPECTED_SOURCE_SHA256 = {
    "tpu_inference/core/sched/dp_scheduler.py": ("a7e625074db0e3b7a97a080f637be641330a696ef29c7ab59c4366a86652fab4"),
    "tpu_inference/kernels/experimental/batched_rpa/__init__.py": (
        "e9b8af1551a068f26499e15c674fe09ca39fa39e96c7c93740c2f2afcf1a7588"
    ),
    "tpu_inference/kernels/experimental/batched_rpa/bref_override.py": (
        "f64ecdefc69b7ed486e0921135241501275ce4234ee8bac342173b1e669823ac"
    ),
    "tpu_inference/kernels/experimental/batched_rpa/configs.py": (
        "1df58118ea781f2d5ec5507f2a767d48a498589ebf2b2e559195b5b631f85ee6"
    ),
    "tpu_inference/kernels/experimental/batched_rpa/flash_attention.py": (
        "a9bb67f576bff5d59538b16509f7b52064844be53f1bf2d14358f0d1900ad658"
    ),
    "tpu_inference/kernels/experimental/batched_rpa/kernel.py": (
        "d082659d779655b77e2cac7b2d427954217f3740d7de72ffc6c1527555fa8345"
    ),
    "tpu_inference/kernels/experimental/batched_rpa/schedule.py": (
        "b770e6718f514081ad4778f9a71c4ebd504d31e26101b507136d264d3f6f1c88"
    ),
    "tpu_inference/kernels/experimental/batched_rpa/stitch_utils.py": (
        "fa3c17237707aa14be6a755dc31ac7e76252fe544730598bd27853a4fc6139ce"
    ),
    "tpu_inference/kernels/experimental/batched_rpa/tuned_params.py": (
        "8b1b3f78225448b73791ca9adc52b8f0e7eee8dd75f060da3ac4ceb8931f4c53"
    ),
    "tpu_inference/kernels/experimental/batched_rpa/utils.py": (
        "0ffb823cdf662d2907a37866fe33df74c8504d9227d588b750d221cbab717dda"
    ),
    "tpu_inference/kernels/experimental/batched_rpa/wrapper.py": (
        "3722a2b0e3d75b9ed5bc319a6252f0241370c0f48b8799034613ea212e03caf9"
    ),
    "tpu_inference/layers/common/attention_interface.py": (
        "4327f2ed6ec853275b840ba895435e66ea93aacebcac84b0d3a881df6336feac"
    ),
    "tpu_inference/models/common/model_loader.py": ("af7e7449e960fc6f346403aba1eae6f61962447a28829d45b87c6a0c9a18bb87"),
    "tpu_inference/models/jax/gemma4.py": ("bffaecd7f9832ae42a94caafdcb653604475720810b972cd0516075c2b746fb4"),
    "tpu_inference/models/jax/gemma4_mm.py": ("2ab723eac3d94f52c5a7ec444d00a54e160e172d6173c32307f1600907b34932"),
    "tpu_inference/runner/kv_cache_manager.py": ("1d43cb039db8df0a25be5c27b405af65cbf02f9f6080935bfd0aeee836c91bd5"),
    "tpu_inference/runner/tpu_runner.py": ("77585e7c3c1ae10b4465c163575fe8974100cbe845c9ef62959b6a644dcb9c76"),
    "tpu_inference/worker/tpu_worker.py": ("7e4c2d76f84d614e4f3799c623a0c13a679f5dc64bb4a4c099154514dbce5fe3"),
    "vllm/distributed/parallel_state.py": ("98d4fb7095d440ee5285a3742713c02865a494fe0e851f1b301f4861d1656cbb"),
    "vllm/distributed/utils.py": ("792146b357139e705064a68d3b66858de8a0baaca4dc7db1232fa217d45db231"),
    "vllm/entrypoints/generate/beam_search/offline.py": (
        "b90e29bc2a78a93ab38d87fd56202dfd351cb238c8d3949cd522ae43dfd4bd18"
    ),
    "vllm/entrypoints/llm.py": ("52de4ac99489e004ef6c61d0bedc84aa96020dd58b8bd1ae500814b548b2b83e"),
    "vllm/entrypoints/offline_utils.py": ("688fbad0af9c2180b83aa77dcd0dbda85ca076a6c72bffa61840896d950cf458"),
    "vllm/entrypoints/pooling/offline.py": ("52410b116ac5fec3510eb15073400fb0b6adb21e71fc397a49c7c8b8495daef3"),
    "vllm/v1/core/sched/scheduler.py": ("7dc71c574b6c7c53f5ffff4b1b39fb81b46b334e44116677896e9196169d0c76"),
    "vllm/v1/engine/core.py": ("0aa2b4a597efe3567c9aff5699aaf6018cc8d22431b6880a4c6f03352a8361b1"),
    "vllm/v1/engine/core_client.py": ("969832582303057621f5c3ced4561a6bcc2ecc73d4d79b4dd97da0e84cb43e6c"),
    "vllm/v1/engine/llm_engine.py": ("415b7d02460984e8678f78420be22dbc336d8bb73734d7a753b2badb62232744"),
    "vllm/v1/executor/abstract.py": ("7514edb4a569220851e43206dccfca92e5b36b4639fcfd27a2825f38990670f8"),
    "vllm/v1/executor/uniproc_executor.py": ("3cb039b80fe9b03dc1122f93c1f8a4098ef1355104b317ff6df25358f4349494"),
    "vllm/v1/structured_output/__init__.py": ("011e972604ed3a2a1e1db5421012060a0dc9133826ac0c58236eaca6a8756fc8"),
    "vllm/v1/worker/worker_base.py": ("b0684c84f2f2c4bff2d7f0574e13f48c01b4d76dbf555e7c52b32e9a702a9061"),
}


def resolve_weight_sync_transport(backend: str, transport: str = "auto") -> str:
    """同进程默认用 ICI；独立运行时保留主机协议，不静默回退。"""
    if backend not in ("jax", "inference", "inference-process", "inference-distributed") or transport not in (
        "auto",
        "host",
        "device",
    ):
        raise ValueError("未知 rollout 后端或权重同步方式")
    resolved = (
        ("device" if backend in ("inference", "inference-distributed") else "host")
        if transport == "auto"
        else transport
    )
    if resolved == "device" and backend not in ("inference", "inference-distributed"):
        raise ValueError("device 权重直传要求同进程 inference 或共同运行时 inference-distributed")
    if backend == "inference-distributed" and resolved != "device":
        raise ValueError("共同运行时只使用 device；独立运行时 host 对照使用 inference-process")
    return resolved


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    device_indexes: tuple[int, ...] = (2, 3)
    dp_size: int = 2
    tp_size: int = 1
    max_num_seqs_per_rank: int = 4
    token_budget_per_rank: int = 2048
    max_model_len: int = 528
    memory_fraction: float = 0.5
    block_size: int = 64
    kernel_route: str = "batched"
    max_logprobs: int = 128
    weight_sync_transport: str = "device"
    source_read_mode: str = "direct"
    trim_training_host_allocator_after_transfer: bool = False
    trim_host_allocator_after_update: bool = False

    def __post_init__(self) -> None:
        if self.weight_sync_transport not in ("host", "device"):
            raise ValueError("weight_sync_transport 必须为 host 或 device")
        if self.source_read_mode not in ("direct", "transient_copy"):
            raise ValueError("source_read_mode必须是direct或transient_copy")
        if type(self.trim_training_host_allocator_after_transfer) is not bool:
            raise ValueError("trim_training_host_allocator_after_transfer必须是bool")
        if type(self.trim_host_allocator_after_update) is not bool:
            raise ValueError("trim_host_allocator_after_update必须是bool")
        if self.trim_training_host_allocator_after_transfer and self.source_read_mode != "transient_copy":
            raise ValueError("训练侧分配器回收只允许与transient_copy一起启用")


@dataclass(frozen=True)
class EngineSampling:
    max_tokens: int
    temperature: float = 1.0
    top_k: int = 0
    eos_ids: tuple[int, ...] = ()
    top_p: float = 1.0


@dataclass(frozen=True)
class EngineRow:
    prompt_token_ids: list[int]
    token_ids: list[int]
    raw_logprobs: list[float]
    proposal_logprobs: list[float] | None
    finish_reason: str | None
    stop_reason: int | str | None


@dataclass(frozen=True)
class EngineBatch:
    rows: list[EngineRow]
    policy_version: PolicyVersion
    metadata: dict[str, Any]


def validate_sampling(sampling: EngineSampling) -> None:
    if isinstance(sampling.max_tokens, bool) or not isinstance(sampling.max_tokens, int) or sampling.max_tokens <= 0:
        raise ValueError("max_tokens 必须为正整数")
    if isinstance(sampling.top_k, bool) or not isinstance(sampling.top_k, int) or sampling.top_k < -1:
        raise ValueError("top_k 必须为整数；0/-1 表示完整词表，正数表示截断数量")
    if not math.isfinite(sampling.temperature) or sampling.temperature < 0 or sampling.top_p != 1:
        raise ValueError("温度必须有限且非负；此资格仅覆盖 top_p=1")


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def selected_proposal_logprob(
    selected: int, entries: dict[int, Any], sampling: EngineSampling
) -> tuple[float | None, str]:
    """有界 top-k 在 host FP64 重建分布；其误差口径不等同于 device sampler 的逐位概率。"""
    validate_sampling(sampling)
    raw = float(entries[selected].logprob)
    if not math.isfinite(raw):
        raise RuntimeError("所选 token 原始 logprob 非有限")
    if sampling.temperature == 0:
        return 0.0, "greedy_point_mass"
    if sampling.top_k <= 0:
        if sampling.temperature == 1 and sampling.top_p == 1:
            return raw, "raw_policy_temperature_1_full_vocab"
        return None, "unavailable_full_vocab_temperature_normalizer"
    selected_rank = entries[selected].rank
    if (
        not isinstance(selected_rank, int)
        or isinstance(selected_rank, bool)
        or not 1 <= selected_rank <= sampling.top_k
    ):
        raise RuntimeError("实际所选 token 不在要求的 top-k rank 范围内")
    ranked = [entry for entry in entries.values() if entry.rank is not None and entry.rank <= sampling.top_k]
    ranks = [entry.rank for entry in ranked]
    if any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks):
        raise RuntimeError("top-k ranks 不是整数")
    if sorted(ranks) != list(range(1, sampling.top_k + 1)):
        raise RuntimeError("top-k rank 集合不是完整且无重复的 1..k，不能重建 proposal")
    top = [float(entry.logprob) for entry in ranked]
    if not all(math.isfinite(value) for value in top):
        raise RuntimeError("top-k 原始 logprobs 含非有限值")
    scaled = [value / sampling.temperature for value in top]
    pivot = max(scaled)
    normalizer = pivot + math.log(math.fsum(math.exp(value - pivot) for value in scaled))
    return raw / sampling.temperature - normalizer, "host_fp64_top_k_reconstructed_from_raw_logprobs"


def shutdown_owned_scheduler(scheduler: Any, timeout_s: float = 5.0) -> dict[str, Any]:
    """只关闭本 scheduler；对上游无界 ack 使用限时等待与本对象进程/pipe 后备清理。"""
    report: dict[str, Any] = {"shutdown_returned": False, "errors": []}

    def shutdown() -> None:
        try:
            scheduler.shutdown()
            report["shutdown_returned"] = True
        except BaseException as error:
            report["errors"].append(f"{type(error).__name__}: {error}")

    thread = threading.Thread(target=shutdown, name="inference-scheduler-close", daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)
    processes = list(getattr(scheduler, "processes", []))
    pipes = list(getattr(scheduler, "input_conns", [])) + list(getattr(scheduler, "output_conns", []))
    report["bounded_fallback"] = thread.is_alive() or bool(report["errors"])
    if report["bounded_fallback"]:
        for process in processes:
            try:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)
            except BaseException as error:
                report["errors"].append(f"owned process {process.pid}: {type(error).__name__}: {error}")
        for connection in pipes:
            try:
                connection.close()
            except BaseException as error:
                report["errors"].append(f"owned pipe: {type(error).__name__}: {error}")
        thread.join(timeout=1.0)
    report["shutdown_thread_finished"] = not thread.is_alive()
    report["owned_processes"] = [
        {"pid": process.pid, "alive": process.is_alive(), "exitcode": process.exitcode} for process in processes
    ]
    report["all_owned_processes_stopped"] = all(not process.is_alive() for process in processes)
    report["all_owned_pipes_closed"] = all(connection.closed for connection in pipes)
    report["complete"] = (
        report["shutdown_returned"]
        and report["shutdown_thread_finished"]
        and report["all_owned_processes_stopped"]
        and report["all_owned_pipes_closed"]
        and not report["errors"]
    )
    return report


class EngineRuntime:
    """显式串行生命周期；同进程只应存在一个本类引擎，不能与另一 vLLM 引擎交错更新。"""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.policy_version: PolicyVersion = None
        self._pending: dict[str, Any] | None = None
        self._broken = False
        self._closed = False
        self.llm: Any = None
        self.core: Any = None
        self.worker: Any = None
        self.runner: Any = None
        self.metadata: dict[str, Any] = {"config": asdict(config)}
        if config.kernel_route != "batched" or config.tp_size != 1:
            raise ValueError("当前研究资格仅覆盖原始 batched 路由及 TP1")
        if len(config.device_indexes) != config.dp_size * config.tp_size or len(set(config.device_indexes)) != len(
            config.device_indexes
        ):
            raise ValueError("device_indexes 与 DP/TP 大小不符或存在重复芯片")
        if min(config.dp_size, config.max_num_seqs_per_rank, config.token_budget_per_rank, config.max_model_len) <= 0:
            raise ValueError("引擎容量必须为正")
        if not 0 < config.memory_fraction < 1 or config.block_size != 64:
            raise ValueError("研究资格需要 memory_fraction 在(0,1)且 block_size=64")
        for key, value in {
            "USE_BATCHED_RPA_KERNEL": "1",
            "USE_BATCHED_RPA_SEQ_ON_LANE": "0",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "TPU_MULTIPROCESS_DP": "0",
            "MODEL_IMPL_TYPE": "flax_nnx",
        }.items():
            if os.environ.setdefault(key, value) != value:
                raise ValueError(f"此后端要求 {key}={value}；请在导入引擎前配置")
        if os.environ.get("JAX_DEFAULT_MATMUL_PRECISION") != "default":
            raise ValueError("必须在进程启动时显式设置 JAX_DEFAULT_MATMUL_PRECISION=default")
        import jax
        from flax import nnx
        from vllm import LLM, SamplingParams

        self.jax = jax
        self.nnx = nnx
        self.SamplingParams = SamplingParams
        self.metadata = {
            "config": asdict(config),
            "transport": "callback_real_parameter_assignment_then_state_leaves_refresh",
            "external_dma_completion_claim": False,
            "upstream_monkeypatch": False,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("jax", "jaxlib", "libtpu", "tpu-inference", "vllm-tpu")
            },
        }
        if self.metadata["versions"] != EXPECTED_VERSIONS:
            raise RuntimeError("此生命周期入口需要已经核定的引擎版本组合；不自动升级或降级")
        distribution = importlib.metadata.distribution("tpu-inference")
        source_sha = {name: file_sha256(Path(str(distribution.locate_file(name)))) for name in EXPECTED_SOURCE_SHA256}
        if source_sha != EXPECTED_SOURCE_SHA256:
            raise RuntimeError("安装的生命周期源码已改变；需重新核定私有 API")
        self.metadata["source_sha256"] = source_sha
        libtpu_path = Path(str(importlib.metadata.distribution("libtpu").locate_file("libtpu/libtpu.so"))).resolve()
        if file_sha256(libtpu_path) != EXPECTED_LIBTPU_SHA256:
            raise RuntimeError("libtpu 实际库字节不匹配锁定 SHA")
        self.metadata["libtpu"] = {"path": str(libtpu_path), "sha256": EXPECTED_LIBTPU_SHA256}
        self._pickle_before = (
            multiprocessing.reduction.ForkingPickler.dumps,
            multiprocessing.reduction.ForkingPickler.loads,
        )
        try:
            with jax.default_matmul_precision("default"):
                if jax.default_backend() != "tpu":
                    raise RuntimeError("EngineRuntime 需要 TPU；CPU 控制只调用独立协议函数")
                mapped_libtpu = {
                    Path(line.split(maxsplit=5)[-1]).resolve()
                    for line in Path("/proc/self/maps").read_text().splitlines()
                    if "/libtpu.so" in line
                }
                if mapped_libtpu != {libtpu_path}:
                    raise RuntimeError("实际加载的 libtpu 映射不是锁定文件")
                self.metadata["libtpu"]["mapped_paths"] = sorted(str(path) for path in mapped_libtpu)
                self.llm = LLM(
                    model=config.model_path,
                    dtype="bfloat16",
                    tensor_parallel_size=config.tp_size,
                    data_parallel_size=config.dp_size,
                    additional_config={
                        "sharding": {"sharding_strategy": {"device_indexes": list(config.device_indexes)}}
                    },
                    max_num_seqs=config.max_num_seqs_per_rank,
                    max_num_batched_tokens=config.token_budget_per_rank,
                    max_model_len=config.max_model_len,
                    gpu_memory_utilization=config.memory_fraction,
                    block_size=config.block_size,
                    max_logprobs=config.max_logprobs,
                    async_scheduling=False,
                    enforce_eager=True,
                    enable_prefix_caching=False,
                    skip_tokenizer_init=True,
                    generation_config="vllm",
                    logprobs_mode="raw_logprobs",
                    limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
                    seed=0,
                )
                self.core = self.llm.llm_engine.engine_core.engine_core
                self.worker = self.core.model_executor.driver_worker.worker
                self.runner = self.worker.model_runner
                self._verify_destruction_chain()
                actual_ids = [int(device.id) for device in self.runner.mesh.devices.flat]
                if actual_ids != list(config.device_indexes):
                    raise RuntimeError("引擎实际 mesh 与指定芯片不同")
                attention = importlib.import_module("tpu_inference.layers.common.attention_interface")
                if attention.rpa.__name__ != "tpu_inference.kernels.experimental.batched_rpa.wrapper":
                    raise RuntimeError("引擎实际没有使用原始 batched wrapper")
                self.metadata["mesh_device_ids"] = actual_ids
                self.metadata["attention_module"] = attention.rpa.__name__
                self._ensure_quiescent()
        except BaseException as error:
            try:
                cleanup = self.close()
            except BaseException as cleanup_error:
                cleanup = {"unexpected_cleanup_error": f"{type(cleanup_error).__name__}: {cleanup_error}"}
            self.metadata["constructor_failure_cleanup"] = cleanup
            error.add_note(f"EngineRuntime 构造失败后的独立清理：{cleanup}")
            raise

    def _verify_destruction_chain(self) -> None:
        llm_engine = self.llm.llm_engine
        objects = (
            (self.llm, "vllm.entrypoints.llm", "LLM"),
            (llm_engine.engine_core, "vllm.v1.engine.core_client", "InprocClient"),
            (self.core, "vllm.v1.engine.core", "EngineCore"),
            (self.core.model_executor, "vllm.v1.executor.uniproc_executor", "UniProcExecutor"),
            (self.worker, "tpu_inference.worker.tpu_worker", "TPUWorker"),
        )
        for obj, module, name in objects:
            if type(obj).__module__ != module or type(obj).__name__ != name:
                raise RuntimeError("实际引擎销毁链不是核定的单进程类")
            if inspect.getattr_static(type(obj), "__del__", None) is not None:
                raise RuntimeError("核定链之外出现 __del__，不能保证只清理当前引擎")
        if type(llm_engine).__module__ != "vllm.v1.engine.llm_engine" or type(llm_engine).__name__ != "LLMEngine":
            raise RuntimeError("实际 LLMEngine 不是锁定实现")
        if llm_engine.external_launcher_dp:
            raise RuntimeError("此后端不能接管 external launcher 拥有的分布式组")
        scheduler_name = type(self.core.scheduler).__name__
        expected_scheduler_module = {
            "DPScheduler": "tpu_inference.core.sched.dp_scheduler",
            "Scheduler": "vllm.v1.core.sched.scheduler",
        }.get(scheduler_name)
        if expected_scheduler_module is None or type(self.core.scheduler).__module__ != expected_scheduler_module:
            raise RuntimeError("实际 scheduler 类型没有核定独立关闭路径")
        self.metadata["destruction_chain"] = {
            "client": "InprocClient",
            "core": "EngineCore",
            "scheduler": scheduler_name,
            "global_core_shutdown_used": False,
            "llm_engine_finalizer": "清理本模型 Torch hooks；关闭时显式 detach/处理",
            "dp_group_cleanup": "仅本 LLMEngine 的 stateless group",
        }

    def _check_open(self) -> None:
        if self._closed or self._broken:
            raise RuntimeError("引擎已关闭或此前失败；不能继续生成")

    def _ensure_quiescent(self) -> dict[str, Any]:
        core = self.core
        if core.async_scheduling or core.batch_queue is not None:
            raise RuntimeError("此后端要求实际关闭异步调度且 batch_queue=None")
        if self.llm.llm_engine.has_unfinished_requests() or core.scheduler.has_unfinished_requests():
            raise RuntimeError("引擎还有未完成请求")
        if core.scheduler.connector is not None or core.scheduler.ec_connector is not None:
            raise RuntimeError("此后端尚未覆盖外部 KV/EC connector 的后台生命周期")
        record: dict[str, Any] = {"zero_token_cleanup_steps": 0}
        if core.scheduler.has_requests():
            _, executed = core.step()
            core.post_step(executed)
            record["zero_token_cleanup_steps"] = 1
            if executed:
                raise RuntimeError("已完成请求的清理意外生成模型 token")
        if core.scheduler.has_requests() or self.llm.llm_engine.has_unfinished_requests():
            raise RuntimeError("清理后引擎仍非空闲")
        return record

    def generate(
        self,
        prompts: list[list[int]],
        batch_rng: BatchRngState,
        sampling: EngineSampling,
        expected_version: PolicyVersion = None,
    ) -> EngineBatch:
        self._check_open()
        if self._pending is not None:
            raise RuntimeError("权重更新尚未提交，不能生成")
        if expected_version is not None and expected_version != self.policy_version:
            raise RuntimeError("请求指定的策略版本与当前引擎不同")
        if not prompts or not isinstance(batch_rng, BatchRngState):
            raise ValueError("非空prompt批次需要显式BatchRngState")
        if any(not prompt or len(prompt) + sampling.max_tokens > self.config.max_model_len for prompt in prompts):
            raise ValueError("prompt 为空或总 token 超过引擎长度预算")
        validate_sampling(sampling)
        requested_logprobs = max(1, sampling.top_k) if sampling.temperature > 0 else 1
        if requested_logprobs > self.config.max_logprobs:
            raise ValueError("完整 top-k logprobs 数量超过配置预算")
        params = [
            self.SamplingParams(
                temperature=sampling.temperature,
                top_k=sampling.top_k if sampling.top_k > 0 else -1,
                top_p=1,
                max_tokens=sampling.max_tokens,
                ignore_eos=not bool(sampling.eos_ids),
                stop_token_ids=list(sampling.eos_ids),
                logprobs=requested_logprobs,
                detokenize=False,
                seed=None,
            )
            for _ in prompts
        ]
        started = time.perf_counter()
        try:
            with self.jax.default_matmul_precision("default"):
                self._ensure_quiescent()
                rng_install = install_runner_rng(self.runner, batch_rng, self.config.device_indexes)
                outputs = self.llm.generate(
                    [{"prompt_token_ids": prompt} for prompt in prompts], params, use_tqdm=False
                )
                rows = []
                semantics = set()
                for prompt, request in zip(prompts, outputs, strict=True):
                    item = request.outputs[0]
                    if request.prompt_token_ids != prompt or item.logprobs is None:
                        raise RuntimeError("实际 prompt 或 raw logprobs 返回不符合协议")
                    raw = []
                    proposal: list[float] | None = []
                    for token, entries in zip(item.token_ids, item.logprobs, strict=True):
                        value = float(entries[token].logprob)
                        if not math.isfinite(value):
                            raise RuntimeError("raw selected logprob 非有限")
                        raw.append(value)
                        proposed, meaning = selected_proposal_logprob(token, entries, sampling)
                        semantics.add(meaning)
                        if proposed is None:
                            proposal = None
                        elif proposal is not None:
                            if not math.isfinite(proposed):
                                raise RuntimeError("proposal selected logprob 非有限")
                            proposal.append(proposed)
                    rows.append(
                        EngineRow(
                            list(prompt), list(item.token_ids), raw, proposal, item.finish_reason, item.stop_reason
                        )
                    )
                self.jax.effects_barrier()  # type: ignore[no-untyped-call]
                cleanup = self._ensure_quiescent()
                rng_after, rng_placement = read_runner_rng(self.runner, self.config.device_indexes)
                if rng_placement != rng_install["placement"]:
                    raise RuntimeError("生成改变了采样key的mesh放置")
                if sampling.temperature > 0 and rng_after == batch_rng:
                    raise RuntimeError("随机生成后runner实际采样状态没有推进")
            return EngineBatch(
                rows,
                self.policy_version,
                {
                    "wall_s": time.perf_counter() - started,
                    "selected_logprob_semantics": "untempered_untruncated_raw_policy",
                    "proposal_semantics": sorted(semantics),
                    "requested_logprobs_per_token": requested_logprobs,
                    "sampling": asdict(sampling),
                    "cleanup": cleanup,
                    "batch_rng_install": rng_install,
                    "batch_rng_after": rng_after.to_dict(),
                    "batch_rng_scope": "固定顺序/调度/模型版本的批次状态，不是逐请求独立seed",
                },
            )
        except BaseException as error:
            self._broken = True
            self.metadata["last_generation_failure"] = {"error_type": type(error).__name__, "error": str(error)}
            raise

    def _refresh_dispatch(self, expected_treedef: Any, expected_shapes: list[Any]) -> None:
        state = self.nnx.state(self.runner.model)
        leaves, treedef = self.jax.tree_util.tree_flatten(state)
        if cast(Any, treedef) != expected_treedef:
            raise RuntimeError("模型 state treedef 改变，不能复用已编译 model_fn")
        if [(tuple(leaf.shape), str(leaf.dtype)) for leaf in leaves] != expected_shapes:
            raise RuntimeError("模型 state shape/dtype 改变")
        self.jax.block_until_ready(leaves)  # type: ignore[no-untyped-call]
        expected_devices = set(self.config.device_indexes)
        if any({int(device.id) for device in leaf.sharding.device_set} != expected_devices for leaf in leaves):
            raise RuntimeError("同步后的模型数组不在指定引擎芯片上")
        self.runner.state = state
        self.runner.state_leaves = tuple(leaves)

    def begin_update(self, version: int | str) -> None:
        self._check_open()
        if self._pending is not None:
            raise RuntimeError("不允许重入权重更新")
        with self.jax.default_matmul_precision("default"):
            cleanup = self._ensure_quiescent()
            if self.llm.reset_prefix_cache() is not True:
                raise RuntimeError("prefix cache reset 失败")
            leaves, treedef = self.jax.tree_util.tree_flatten(self.runner.state)
            snapshots = list(
                {id(param): (param, param.get_value()) for _, param in self.runner.model.named_parameters()}.values()
            )
            self._pending = {
                "version": version,
                "treedef": treedef,
                "shapes": [(tuple(leaf.shape), str(leaf.dtype)) for leaf in leaves],
                "snapshots": snapshots,
                "applied": False,
                "report": {"version": version, "cleanup": cleanup, "complete": False},
            }
            old_kv = self.jax.tree_util.tree_leaves(self.runner.kv_caches)
            if not old_kv:
                raise RuntimeError("更新前缺少真实 KV arrays")
            self.llm.start_weight_update()
            if not self.worker._weight_update_active or self.runner.kv_caches:
                raise RuntimeError("start_weight_update 未进入更新状态或没有清空 KV")
            if not all(array.is_deleted() for array in old_kv):
                raise RuntimeError("旧 KV arrays 未实际删除")
            self._pending["report"]["old_kv_arrays_deleted"] = True

    def apply_state(self, callback: Callable[[Any], dict[str, Any]]) -> dict[str, Any]:
        if self._pending is None or self._pending["applied"]:
            raise RuntimeError("需要未执行 apply 的活动权重事务")
        with self.jax.default_matmul_precision("default"):
            report = callback(self.runner)
            self._refresh_dispatch(self._pending["treedef"], self._pending["shapes"])
            self.llm.update_weights({"update_info": {"transport": "already-applied-jax-arrays"}})
            self._pending["report"]["mapping"] = report
            self._pending["report"]["dispatch_refreshed"] = True
            self._pending["report"]["external_dma_completion_claim"] = False
            self._pending["applied"] = True
            return report

    def _restore_kv(self) -> None:
        if self.worker._weight_update_active or self.worker._kv_cache_freed:
            self.llm.finish_weight_update(weight_version=None)
        elif not self.jax.tree_util.tree_leaves(self.runner.kv_caches):
            self.runner.reinitialize_kv_cache()
        arrays = self.jax.tree_util.tree_leaves(self.runner.kv_caches)
        if self.worker._weight_update_active or self.worker._kv_cache_freed or not arrays:
            raise RuntimeError("KV 恢复不完整")
        if any(array.is_deleted() for array in arrays):
            raise RuntimeError("恢复后的 KV 仍含已删除数组")
        self.jax.block_until_ready(arrays)  # type: ignore[no-untyped-call]

    def finish_update(self) -> dict[str, Any]:
        if self._pending is None or not self._pending["applied"]:
            raise RuntimeError("没有已完成 apply 的活动权重事务")
        pending = self._pending
        with self.jax.default_matmul_precision("default"):
            self.llm.finish_weight_update(weight_version=str(pending["version"]))
            self._restore_kv()
            if self.llm.get_weight_version() != str(pending["version"]):
                raise RuntimeError("引擎没有确认提交的新版本")
            self.policy_version = pending["version"]
            pending["report"]["complete"] = True
            self._pending = None
            return cast(dict[str, Any], pending["report"])

    def abort_update(self) -> dict[str, Any]:
        pending = self._pending
        if pending is None:
            return {"attempted": False}
        recovery: dict[str, Any] = {"attempted": True, "weights_restored": False, "kv_restored": False}
        with self.jax.default_matmul_precision("default"):
            parameter_errors = []
            for index, (param, original) in enumerate(pending["snapshots"]):
                try:
                    param.set_value(original)
                except BaseException as error:
                    parameter_errors.append({"snapshot_index": index, "error": f"{type(error).__name__}: {error}"})
            recovery["parameter_restore_errors"] = parameter_errors
            try:
                self._refresh_dispatch(pending["treedef"], pending["shapes"])
                recovery["dispatch_refreshed"] = True
                recovery["weights_restored"] = not parameter_errors
            except BaseException as error:
                recovery["weights_error"] = f"{type(error).__name__}: {error}"
            try:
                self._restore_kv()
                recovery["kv_restored"] = True
            except BaseException as error:
                recovery["kv_error"] = f"{type(error).__name__}: {error}"
        self._pending = None
        self._broken = True
        return recovery

    def sync_params(self, params: Any, model_config: Any, version: int) -> dict[str, Any]:
        """与独立进程后端共享完整参数接口，沿用原事务与映射实现。"""
        from .inference_weights import apply_gemma4_params

        return self.update_weights(
            lambda runner: apply_gemma4_params(
                runner, params, model_config, transport=self.config.weight_sync_transport
            ),
            version,
        )

    def update_weights(self, callback: Callable[[Any], dict[str, Any]], version: int | str) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            self.begin_update(version)
            self.apply_state(callback)
            report = self.finish_update()
            report["wall_s"] = time.perf_counter() - started
            return report
        except BaseException as error:
            recovery = self.abort_update()
            if hasattr(error, "add_note"):
                error.add_note(f"EngineRuntime 独立恢复结果：{recovery}")
            self.metadata["last_update_failure"] = {
                "error": str(error),
                "mapping_error_report": getattr(error, "report", None),
                "recovery": recovery,
            }
            raise

    def close(self) -> dict[str, Any]:
        if getattr(self, "_closed", False):
            return {"already_closed": True}
        self._closed = True
        report: dict[str, Any] = {"errors": [], "global_core_shutdown_used": False}
        llm = getattr(self, "llm", None)
        llm_engine = getattr(llm, "llm_engine", None)
        core = getattr(self, "core", None) or getattr(getattr(llm_engine, "engine_core", None), "engine_core", None)
        executor = getattr(core, "model_executor", None)
        wrapper = getattr(executor, "driver_worker", None)
        worker = getattr(self, "worker", None) or getattr(wrapper, "worker", None)
        runner = getattr(self, "runner", None) or getattr(worker, "model_runner", None)
        jax = getattr(self, "jax", None)
        model_arrays: dict[int, Any] = {}

        def attempt(name: str, action: Callable[[], Any]) -> Any:
            try:
                value = action()
                report[name] = value if value is not None else True
                return value
            except BaseException as error:
                report["errors"].append({"action": name, "error": f"{type(error).__name__}: {error}"})
                return None

        def collect_model_arrays() -> None:
            if runner is None or jax is None:
                return
            for state in (getattr(runner, "state", None), getattr(runner, "state_leaves", None)):
                model_arrays.update({id(array): array for array in jax.tree_util.tree_leaves(state)})
            model = getattr(runner, "model", None)
            if model is not None:
                for _, parameter in model.named_parameters():
                    array = parameter.get_value()
                    model_arrays[id(array)] = array
            pending = getattr(self, "_pending", None)
            if pending is not None:
                model_arrays.update({id(array): array for _, array in pending["snapshots"]})

        context = jax.default_matmul_precision("default") if jax is not None else nullcontext()
        with context:
            # 错误 traceback、NNX 对象或已编译闭包可能继续引用数组；显式 delete 解除设备内存所有权。
            attempt("collect_arrays_before_recovery", collect_model_arrays)
            if getattr(self, "_pending", None) is not None:
                attempt("recovery", self.abort_update)
            attempt("collect_arrays_after_recovery", collect_model_arrays)
            scheduler = getattr(core, "scheduler", None)
            if scheduler is not None:
                attempt("scheduler_shutdown", lambda: shutdown_owned_scheduler(scheduler))
            manager = getattr(core, "structured_output_manager", None)
            if manager is not None:
                attempt("structured_output_backend_cleared", manager.clear_backend)
                # 这些线程池均为本 manager 成员；跳过 Core.shutdown 的全局清理时仍显式处理。
                for name in ("executor", "executor_for_fillmask"):
                    pool = getattr(manager, name, None)
                    if pool is not None and callable(getattr(pool, "shutdown", None)):

                        def shutdown_pool(pool: Any = pool) -> None:
                            pool.shutdown(wait=True)

                        attempt("structured_output_" + name, shutdown_pool)
            if runner is not None:
                old_kv = jax.tree_util.tree_leaves(getattr(runner, "kv_caches", None)) if jax is not None else []
                attempt("kv_delete_called", runner.delete_kv_cache)
                report["old_kv_arrays_deleted"] = all(array.is_deleted() for array in old_kv)
            if executor is not None:
                attempt("executor_shutdown", executor.shutdown)
            finalizer = getattr(llm_engine, "_finalizer", None)
            if finalizer is not None:
                detached = attempt("model_finalizer_detached", finalizer.detach)
                # 不把 detach 返回的 model/args 放到可序列化报告里；否则会再次保活模型。
                report["model_finalizer_detached"] = detached is not None
                if detached is not None:
                    _, function, args, kwargs = detached
                    if args and hasattr(args[0], "modules"):
                        attempt("model_torch_hooks_cleared", lambda: function(*args, **kwargs))
                    else:
                        report["model_torch_hooks"] = "NNX 模型无 Torch modules；已移除不适用的 Torch finalizer。"
                    del detached
            group = getattr(llm_engine, "dp_group", None)
            if group is not None and not getattr(llm_engine, "external_launcher_dp", False):
                from vllm.distributed.utils import stateless_destroy_torch_distributed_process_group

                def destroy_own_group() -> None:
                    stateless_destroy_torch_distributed_process_group(group)
                    cast(Any, llm_engine).dp_group = None  # 防止 LLMEngine.__del__ 重复销毁本 group。

                attempt("owned_dp_group_destroyed", destroy_own_group)
            deleted = 0
            skipped = 0
            for array in model_arrays.values():
                try:
                    ids = {int(device.id) for device in array.sharding.device_set}
                    if ids != set(self.config.device_indexes):
                        skipped += 1
                        continue
                    if not array.is_deleted():
                        array.delete()
                    if not array.is_deleted():
                        raise RuntimeError("调用 delete 后数组仍有效")
                    deleted += 1
                except BaseException as error:
                    report["errors"].append({"action": "delete_owned_model_array", "error": str(error)})
            report["model_arrays_seen"] = len(model_arrays)
            report["owned_model_arrays_deleted"] = deleted
            report["foreign_device_arrays_not_deleted"] = skipped
            if scheduler is not None and type(scheduler).__module__ == "tpu_inference.core.sched.dp_scheduler":
                # 原 shutdown 在失败时可能没走到末尾；只恢复本 scheduler 安装的 pickle 函数。
                import atexit

                dp_module = importlib.import_module("tpu_inference.core.sched.dp_scheduler")
                attempt("owned_scheduler_atexit_removed", lambda: atexit.unregister(scheduler._atexit_cleanup))
                pickler = multiprocessing.reduction.ForkingPickler
                if pickler.dumps == dp_module._cloudpickle_dumps and pickler.loads == dp_module._cloudpickle_loads:
                    attempt("owned_cloudpickle_override_restored", dp_module._disable_cloudpickle)
            if hasattr(self, "_pickle_before"):
                report["pickle_functions_restored"] = self._pickle_before == (
                    multiprocessing.reduction.ForkingPickler.dumps,
                    multiprocessing.reduction.ForkingPickler.loads,
                )
        self._pending = None
        self.llm = None
        self.runner = None
        self.worker = None
        self.core = None
        report["engine_references_released"] = True
        scheduler_report = report.get("scheduler_shutdown", {})
        report["complete"] = (
            not report["errors"]
            and report.get("old_kv_arrays_deleted", True)
            and deleted + skipped == len(model_arrays)
            and skipped == 0
            and (scheduler is None or scheduler_report.get("complete", False))
            and report.get("pickle_functions_restored", True)
        )
        return report
