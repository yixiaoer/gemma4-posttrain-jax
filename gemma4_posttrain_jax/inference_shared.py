"""共同 JAX 运行时的双进程引擎：控制消息走本机 socket，完整权重走 ICI。"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .inference_remote import RemoteEngineRuntime
from .inference_runtime import EngineConfig
from .inference_wire import WireError, canonical, describe_tree, receive_control, send_control

SHARED_PROTOCOL = "gemma4-shared-runtime-ici-verified-v1"


def shared_devices(rank: int) -> list[Any]:
    import jax

    if not jax.distributed.is_initialized() or jax.process_count() != 2 or jax.process_index() != rank:
        raise ValueError("此入口需要先初始化两进程共同 JAX 运行时，并使用指定 rank")
    devices = [d for owner in (0, 1) for d in jax.devices("tpu") if d.process_index == owner]
    if len(devices) != 4 or len(jax.local_devices(backend="tpu")) != 2:
        raise ValueError("共同运行时固定训练两芯、推理两芯")
    return devices


def _merged_shards(model: Any, config: Any) -> list[int]:
    from .inference_weights import PREFIX
    from .model import layer_intermediate_size

    modules = dict(model.named_modules())
    counts = []
    for index in range(config.num_hidden_layers):
        module = modules[PREFIX + f"layers.{index}.mlp.gate_up_proj"]
        method = module.quant_method
        width = layer_intermediate_size(config, index)
        if (
            type(method).__name__ != "UnquantizedMergedLinearMethod"
            or list(method.linear_config.output_sizes) != [width, width]
            or list(module.output_sizes) != [width, width]
        ):
            raise WireError("只支持已核定的未量化 gate/up 融合布局")
        counts.append(int(method.linear_config.n_shards))
    return counts


def target_plan(runner: Any, descriptor: dict[str, Any], config: Any) -> tuple[list[Any], dict[str, Any]]:
    """根据完整训练参数规格及真实引擎核对覆盖，不分配模型大小的主机缓冲区。"""
    import jax
    import jax.numpy as jnp

    from .inference_weights import PRESERVED_PREFIXES, _make_plan
    from .inference_wire import rebuild_tree, validate_descriptor

    _, records = validate_descriptor(descriptor, config)
    abstract = rebuild_tree(descriptor, [jax.ShapeDtypeStruct(tuple(r["shape"]), jnp.float32) for r in records])
    model = runner.model
    if type(model).__name__ != "Gemma4ForConditionalGeneration" or hasattr(model, "lm_head"):
        raise WireError("仅支持已核定的 tied Gemma4 文本映射")
    shards = _merged_shards(model, config)
    plan, preserved = _make_plan(abstract, config, lambda i: shards[i])
    pairs = list(model.named_parameters())
    named = dict(pairs)
    preserved |= {n for n in named if n.startswith(PRESERVED_PREFIXES)}
    if len(named) != len(pairs) or set(named) != {item.name for item in plan} | preserved:
        raise WireError("推理参数没有被完整且唯一地覆盖")
    if len({id(p) for p in named.values()}) != len(named):
        raise WireError("推理参数存在未登记的别名")
    if len({id(p.get_value()) for p in named.values()}) != len(named):
        raise WireError("推理参数数组存在未登记的别名")
    for item in plan:
        value = named[item.name].get_value()
        if (
            value.shape != item.shape
            or value.dtype != jnp.bfloat16
            or not value.is_fully_addressable
            or not value.sharding.is_fully_replicated
            or value.sharding.device_set != set(runner.mesh.devices.flat)
        ):
            raise WireError(f"推理目标 shape/dtype/DP2 分片不符：{item.name}")
    return plan, named


class SharedEngineRuntime(RemoteEngineRuntime):
    """复用原串行控制协议和生成接口；进程由共同运行时启动器管理。"""

    def __init__(self, config: EngineConfig, *, path: Path, timeout_s: float = 900):
        from .distributed_ici import SharedRuntimeWeightTransfer

        devices = shared_devices(0)
        if config.weight_sync_transport != "device" or tuple(d.id for d in devices[2:]) != config.device_indexes:
            raise ValueError("共同运行时要求推理进程的全局设备ID及device通信")
        self.config = config
        self.policy_version = None
        self._closed = False
        self._broken = False
        self._lock = threading.Lock()
        self._request_id = 0
        self.transfer = SharedRuntimeWeightTransfer(devices, verify=True)
        self._connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._connection.settimeout(timeout_s)
        self.metadata: dict[str, Any] = {"protocol": SHARED_PROTOCOL, "config": asdict(config)}
        try:
            deadline = time.monotonic() + timeout_s
            while True:
                try:
                    self._connection.connect(str(path))
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("共同运行时的引擎服务没有启动") from None
                    time.sleep(0.1)
            nonce = os.environ["GEMMA4_SHARED_ICI_NONCE"]
            send_control(
                self._connection,
                {"operation": "hello", "protocol": SHARED_PROTOCOL, "nonce": nonce, "config": asdict(config)},
            )
            response = receive_control(self._connection)
            if response.get("status") != "ready" or response.get("protocol") != SHARED_PROTOCOL:
                raise WireError("共同运行时引擎初始化失败")
            self.metadata["remote"] = response["metadata"]
            if response["metadata"]["process_index"] != 1:
                raise WireError("接收引擎不在 rank1")
        except BaseException:
            self._broken = True
            self._connection.close()
            raise

    def sync_params(self, params: Any, model_config: Any, version: int) -> dict[str, Any]:
        import jax

        from .inference_weights import _make_plan

        if (
            type(version) is not int
            or version < 0
            or (self.policy_version is not None and version <= self.policy_version)
        ):
            raise ValueError("策略版本必须是递增的非负整数")
        remote = self.metadata["remote"]
        if hashlib.sha256(canonical(model_config._asdict())).hexdigest() != remote["model_config_sha256"]:
            raise WireError("训练与推理模型配置不一致")
        shards = remote["merged_shards"]
        plan, _ = _make_plan(params, model_config, lambda i: shards[i])
        descriptor, _ = describe_tree(params)
        started = time.perf_counter()
        with self._operation():
            request_id = self._request("update", version=version, descriptor=descriptor)
            self._response(request_id, "receiving")
            timings = []
            for index, item in enumerate(plan):
                mapped_start = time.perf_counter()
                bits, finite = self.transfer.map_source(item)
                jax.block_until_ready((bits, finite))
                if not bool(finite):
                    send_control(self._connection, {"status": "abort", "request_id": request_id})
                    raise WireError(f"参数包含非有限值或 BF16 溢出：{item.name}")
                send_control(self._connection, {"status": "parameter", "request_id": request_id, "index": index})
                mapping_s = time.perf_counter() - mapped_start
                _, record = self.transfer.transfer(item.shape, bits)
                timings.append({"name": item.name, "mapping_s": mapping_s, **record})
                del bits, finite
            response = self._response(request_id, "device_committed")
            report: dict[str, Any] = response["report"]
            if report.get("complete") is not True or report.get("version") != version:
                raise WireError("引擎尚未完整提交新版本")
            self.policy_version = version
            report["client_transport"] = {
                "protocol": SHARED_PROTOCOL,
                "wall_s": time.perf_counter() - started,
                "host_payload_bytes": 0,
                "all_replicas_bitwise_verified": True,
                "parameters": timings,
            }
            return report

    def close(self) -> dict[str, Any]:
        if self._closed:
            return {"complete": not self._broken, "already_closed": True}
        self._closed = True
        self.policy_version = None
        report: dict[str, Any] = {"complete": False, "errors": [], "process_exit_checked_by_launcher": True}
        try:
            self._connection.settimeout(30)
            if not self._broken:
                request_id = self._request("close")
                report["remote_close"] = self._response(request_id, "closed")["report"]
                report["complete"] = bool(report["remote_close"].get("complete"))
        except BaseException as error:
            report["errors"].append({"type": type(error).__name__, "message": str(error)})
        finally:
            self._connection.close()
        return report


def serve_shared_engine(path: Path, record_path: Path) -> None:
    """rank1入口；任何请求失败均关闭此会话，不复用未完整提交的状态。"""
    import jax

    from .distributed_ici import SharedRuntimeWeightTransfer
    from .inference_local_cpu import LocalCpuLoaders
    from .inference_runtime import BatchRngState, EngineRuntime, EngineSampling
    from .model import config_from_hf

    devices = shared_devices(1)
    transfer = SharedRuntimeWeightTransfer(devices, verify=True)
    report: dict[str, Any] = {"complete": False, "requests": [], "errors": []}
    runtime: Any = None

    def save() -> None:
        temporary = record_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(record_path)

    for key, value in {
        "USE_BATCHED_RPA_KERNEL": "1",
        "USE_BATCHED_RPA_SEQ_ON_LANE": "0",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "TPU_MULTIPROCESS_DP": "0",
        "MODEL_IMPL_TYPE": "flax_nnx",
    }.items():
        if os.environ.setdefault(key, value) != value:
            raise ValueError(f"共同运行时引擎要求 {key}={value}")
    nonce = os.environ["GEMMA4_SHARED_ICI_NONCE"]
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener, LocalCpuLoaders() as loaders:
        listener.settimeout(900)
        listener.bind(str(path))
        path.chmod(0o600)
        listener.listen(1)
        try:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(900)
                hello = receive_control(connection)
                if (
                    set(hello) != {"operation", "protocol", "nonce", "config"}
                    or hello["operation"] != "hello"
                    or hello["protocol"] != SHARED_PROTOCOL
                    or hello["nonce"] != nonce
                ):
                    raise WireError("共同运行时控制握手不符")
                config_args = dict(hello["config"])
                config_args["device_indexes"] = tuple(config_args["device_indexes"])
                engine_config = EngineConfig(**config_args)
                if (
                    engine_config.device_indexes != tuple(d.id for d in devices[2:])
                    or engine_config.weight_sync_transport != "device"
                ):
                    raise WireError("推理端的设备或通信方式不符")
                config = config_from_hf(json.loads((Path(engine_config.model_path) / "config.json").read_text()))
                runtime = EngineRuntime(engine_config)
                runtime.metadata.update(upstream_monkeypatch=True, local_cpu_loader=loaders.metadata)
                metadata = {
                    "engine": runtime.metadata,
                    "process_index": jax.process_index(),
                    "pid": os.getpid(),
                    "model_config_sha256": hashlib.sha256(canonical(config._asdict())).hexdigest(),
                    "merged_shards": _merged_shards(runtime.runner.model, config),
                }
                report["metadata"] = metadata
                save()
                send_control(connection, {"status": "ready", "protocol": SHARED_PROTOCOL, "metadata": metadata})
                expected_id = 0
                while True:
                    request = receive_control(connection)
                    request_id = request.get("request_id")
                    if type(request_id) is not int or request_id != expected_id:
                        raise WireError("请求序号不连续")
                    expected_id += 1
                    operation = request.get("operation")
                    entry: dict[str, Any] = {"operation": operation, "request_id": request_id, "complete": False}
                    report["requests"].append(entry)
                    save()
                    started = time.perf_counter()
                    if operation == "update":
                        if set(request) != {"operation", "request_id", "version", "descriptor"}:
                            raise WireError("权重请求字段不符")
                        version = request["version"]
                        if (
                            type(version) is not int
                            or version < 0
                            or (runtime.policy_version is not None and version <= runtime.policy_version)
                        ):
                            raise WireError("权重版本没有递增")
                        plan, named = target_plan(runtime.runner, request["descriptor"], config)

                        def assign(
                            runner: Any,
                            plan: list[Any] = plan,
                            named: dict[str, Any] = named,
                            request_id: int = request_id,
                        ) -> dict[str, Any]:
                            send_control(connection, {"status": "receiving", "request_id": request_id})
                            incoming = {}
                            timings = []
                            for index, item in enumerate(plan):
                                header = receive_control(connection)
                                if header != {"status": "parameter", "request_id": request_id, "index": index}:
                                    raise WireError("发送方中止或参数顺序不同")
                                local, timing = transfer.transfer(item.shape, None)
                                original = named[item.name].get_value()
                                incoming[item.name] = jax.make_array_from_single_device_arrays(
                                    original.shape,
                                    original.sharding,
                                    [local[devices.index(d)] for d in original.sharding._addressable_device_assignment],
                                )
                                timings.append({"name": item.name, **timing})
                            jax.block_until_ready(incoming)
                            for name, value in incoming.items():
                                named[name].set_value(value)
                            return {
                                "protocol": SHARED_PROTOCOL,
                                "assigned_parameters": len(incoming),
                                "all_replicas_bitwise_verified": True,
                                "host_payload_bytes": 0,
                                "parameters": timings,
                            }

                        updated = runtime.update_weights(assign, version)
                        entry.update(complete=True, version=version, report=updated)
                        send_control(
                            connection, {"status": "device_committed", "request_id": request_id, "report": updated}
                        )
                    elif operation == "generate":
                        if set(request) != {"operation", "request_id", "version", "prompts", "rng", "sampling"}:
                            raise WireError("生成请求字段不符")
                        key = request["rng"]
                        rng = BatchRngState(key["implementation"], tuple(key["words"]))
                        if rng.to_dict() != key or type(request["version"]) is not int:
                            raise WireError("生成随机键或版本不符")
                        sampling_args = dict(request["sampling"])
                        sampling_args["eos_ids"] = tuple(sampling_args["eos_ids"])
                        batch = runtime.generate(
                            request["prompts"],
                            rng,
                            EngineSampling(**sampling_args),
                            expected_version=request["version"],
                        )
                        entry.update(complete=True, version=batch.policy_version)
                        send_control(
                            connection, {"status": "generated", "request_id": request_id, "batch": asdict(batch)}
                        )
                    elif operation == "close":
                        if set(request) != {"operation", "request_id"}:
                            raise WireError("关闭请求字段不符")
                        closed = runtime.close()
                        entry["complete"] = report["complete"] = bool(closed.get("complete"))
                        report["close"] = closed
                        send_control(connection, {"status": "closed", "request_id": request_id, "report": closed})
                        break
                    else:
                        raise WireError("未知共同运行时操作")
                    entry["wall_s"] = time.perf_counter() - started
                    save()
        except BaseException as error:
            report["errors"].append({"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            if runtime is not None and not runtime._closed:
                try:
                    report["emergency_close"] = runtime.close()
                except BaseException as error:
                    report["errors"].append({"stage": "close", "type": type(error).__name__, "message": str(error)})
            save()
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
