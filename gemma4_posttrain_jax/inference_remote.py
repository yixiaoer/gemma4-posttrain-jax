"""训练侧同步RPC：完整FP32字节运输、版本提交、生成与本次进程专属清理。"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
import secrets
import signal
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .host_memory import trim_host_allocator
from .inference_rng import BatchRngState
from .inference_runtime import EngineBatch, EngineConfig, EngineRow, EngineSampling
from .inference_wire import (
    PROTOCOL,
    WireError,
    byte_sha,
    canonical,
    describe_tree,
    receive_control,
    send_control,
    validate_descriptor,
)

WORKER_EXIT_TIMEOUT_S = 30.0
DEFAULT_WORKER_MODULE = "gemma4_posttrain_jax.inference_process"


def validate_worker_module(value: str) -> str:
    """Accept an importable dotted module name without allowing command arguments."""
    if not value or any(not part.isidentifier() for part in value.split(".")):
        raise ValueError("引擎worker模块必须是合法的Python模块名")
    return value


@contextlib.contextmanager
def parameter_host_view(leaf: Any, jax: Any, np: Any, mode: str) -> Iterator[Any]:
    """按指定模式取得单个参数的连续FP32主机视图，并及时释放短期设备副本。"""
    temporary = None
    try:
        source = leaf
        if mode == "transient_copy":
            if not isinstance(leaf, jax.Array):
                raise TypeError("transient_copy只接受JAX数组")
            temporary = jax.device_put(leaf, leaf.sharding, may_alias=False)
            if temporary is leaf:
                raise RuntimeError("短期读回副本仍引用原参数数组")
            source = temporary
        elif mode != "direct":
            raise ValueError("未知参数读回模式")
        yield np.asarray(jax.device_get(source), dtype="<f4", order="C")
    finally:
        if temporary is not None:
            temporary.delete()


def isolated_library_environment(parent: Mapping[str, str]) -> dict[str, str]:
    """让子解释器的libtpu包选择自己的库；导入父包可能已修改父环境。"""
    environment = dict(parent)
    environment.pop("TPU_LIBRARY_PATH", None)
    return environment


def inspect_engine_environment(python_executable: Path) -> dict[str, Any]:
    """在指定解释器取得实际环境身份，供训练checkpoint稳定配置与实机握手比较。"""
    environment = isolated_library_environment(os.environ) | {
        "JAX_PLATFORMS": "cpu",
        "JAX_DEFAULT_MATMUL_PRECISION": "highest",
    }
    result = subprocess.run(
        [str(python_executable), "-m", "gemma4_posttrain_jax.inference_process", "--inspect-environment"],
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(f"目标推理环境CPU预检失败：{result.stderr[-4000:]}")
    identity: dict[str, Any] = json.loads(result.stdout)
    if not isinstance(identity, dict) or set(identity) != {
        "versions",
        "source_sha256",
        "libtpu",
        "selected_library",
        "python_executable",
        "python_prefix",
        "scope",
    }:
        raise WireError("目标环境预检没有返回完整身份")
    return identity


def live_owned_group_members(group_id: int) -> list[int]:
    """查询本次独立session的活进程；已退出但尚未被收养者回收的僵尸不再执行。"""
    if type(group_id) is not int or group_id <= 1 or group_id == os.getpgrp():
        raise ValueError("不能把训练进程组作为引擎清理对象")
    members = []
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == group_id:
            if int(fields[3]) != group_id:
                raise RuntimeError("待清理组不是本次独立session")
            if fields[0] not in ("Z", "X"):
                members.append(int(process.name))
    return sorted(members)


def stop_owned_group(group_id: int, *, timeout_s: float = 5) -> dict[str, Any]:
    """worker本身退出后仍检查同组子进程；仅处理启动时建立的独立进程组。"""
    remaining = live_owned_group_members(group_id)
    report: dict[str, Any] = {"initial_live_members": remaining, "signals": [], "complete": False}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not remaining:
            break
        with contextlib.suppress(ProcessLookupError):
            os.killpg(group_id, sig)
            report["signals"].append(sig.name)
        deadline = time.monotonic() + timeout_s
        while remaining and time.monotonic() < deadline:
            time.sleep(0.05)
            remaining = live_owned_group_members(group_id)
    report.update(remaining_live_members=remaining, complete=not remaining)
    return report


class RemoteEngineRuntime:
    """单个训练调用线程、单个本机推理子进程；不支持任意远程地址或并发请求。"""

    def __init__(
        self,
        config: EngineConfig,
        *,
        python_executable: Path,
        record_path: Path,
        timeout_s: float = 900,
        expected_environment: dict[str, Any] | None = None,
        worker_module: str = DEFAULT_WORKER_MODULE,
    ) -> None:
        worker_module = validate_worker_module(worker_module)
        if config.weight_sync_transport != "host":
            raise ValueError(
                "两套独立运行时要求 weight_sync_transport=host；跨进程 ICI 请使用 inference-distributed 共同运行时"
            )
        if os.environ.get("TPU_VISIBLE_CHIPS") != "0,1":
            raise ValueError("父训练进程必须在JAX初始化前绑定物理chips0,1")
        if tuple(config.device_indexes) != (0, 1) or not python_executable.is_file():
            raise ValueError("独立引擎本地设备必须是0,1且解释器存在")
        self.config = config
        self.policy_version: int | None = None
        self._broken = False
        self._closed = False
        self._lock = threading.Lock()
        self._request_id = 0
        self._directory = Path(tempfile.mkdtemp(prefix="g4rpc-"))
        self._socket_path = self._directory / "engine.sock"
        self._connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._connection.settimeout(timeout_s)
        self._process: subprocess.Popen[Any] | None = None
        self.metadata: dict[str, Any] = {"protocol": PROTOCOL, "worker_record": str(record_path)}
        nonce = secrets.token_hex(32)
        self.metadata["removed_parent_library_override"] = os.environ.get("TPU_LIBRARY_PATH")
        env = isolated_library_environment(os.environ) | {
            "TPU_VISIBLE_CHIPS": "2,3",
            "TPU_VISIBLE_DEVICES": "2,3",
            "TPU_CHIPS_PER_HOST_BOUNDS": "1,2,1",
            "TPU_HOST_BOUNDS": "1,1,1",
            "LIBTPU_INIT_ARGS": "deepsea_chips_per_host_bounds=1,2,1,deepsea_host_bounds=1,1,1",
            "JAX_PLATFORMS": "tpu,cpu",
            "JAX_DEFAULT_MATMUL_PRECISION": "default",
            "USE_BATCHED_RPA_KERNEL": "1",
            "USE_BATCHED_RPA_SEQ_ON_LANE": "0",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "TPU_MULTIPROCESS_DP": "0",
            "MODEL_IMPL_TYPE": "flax_nnx",
            "GEMMA4_ENGINE_NONCE": nonce,
        }
        command = [
            str(python_executable),
            "-u",
            "-m",
            worker_module,
            "--socket",
            str(self._socket_path),
            "--engine-config",
            canonical(asdict(config)).decode(),
            "--record",
            str(record_path),
        ]
        record_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = record_path.with_suffix(".log")
        self.metadata.update(
            command=command,
            worker_module=worker_module,
            log_path=str(log_path),
            environment={
                k: env[k]
                for k in (
                    "TPU_VISIBLE_CHIPS",
                    "TPU_VISIBLE_DEVICES",
                    "TPU_CHIPS_PER_HOST_BOUNDS",
                    "TPU_HOST_BOUNDS",
                    "LIBTPU_INIT_ARGS",
                    "JAX_DEFAULT_MATMUL_PRECISION",
                )
            },
        )
        try:
            with log_path.open("w") as log:
                self._process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                    close_fds=True,
                )
            self.metadata["worker_pid"] = self._process.pid
            deadline = time.monotonic() + timeout_s
            while True:
                if self._process.poll() is not None:
                    raise WireError(f"引擎子进程启动失败，exit={self._process.returncode}，见{log_path}")
                try:
                    self._connection.connect(str(self._socket_path))
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("等待引擎socket超时") from None
                    time.sleep(0.02)
            send_control(self._connection, {"protocol": PROTOCOL, "nonce": nonce, "operation": "hello"})
            response = receive_control(self._connection)
            if response.get("status") != "ready" or response.get("metadata", {}).get("protocol") != PROTOCOL:
                raise WireError("引擎初始化握手没有完成")
            self.metadata["remote"] = response["metadata"]
            if response["metadata"]["pid"] != self._process.pid or response["metadata"]["physical_chips"] != [2, 3]:
                raise WireError("握手返回了另一进程或错误芯片声明")
            if expected_environment is not None:
                remote = response["metadata"]
                actual = remote["engine"]
                if any(actual[name] != expected_environment[name] for name in ("versions", "source_sha256")):
                    raise WireError("实际引擎安装身份与训练恢复配置不同")
                if any(actual["libtpu"][name] != expected_environment["libtpu"][name] for name in ("path", "sha256")):
                    raise WireError("实际引擎库身份与CPU预检不同")
                if actual["libtpu"]["mapped_paths"] != [expected_environment["selected_library"]]:
                    raise WireError("实际引擎映射与CPU选择不同")
                if any(remote[name] != expected_environment[name] for name in ("python_executable", "python_prefix")):
                    raise WireError("实际worker解释器与CPU预检不同")
                self.metadata["environment_identity_verified"] = True
        except BaseException:
            self._broken = True
            self.metadata["constructor_failure_cleanup"] = self.close()
            raise

    @contextlib.contextmanager
    def _operation(self) -> Iterator[None]:
        if self._closed or self._broken:
            raise WireError("独立引擎已关闭或此前失败，不能继续复用")
        if not self._lock.acquire(blocking=False):
            raise WireError("独立引擎协议不允许并发或重入")
        try:
            if self._closed or self._broken:
                raise WireError("取得调用锁之前引擎已关闭或失效")
            yield
        except BaseException:
            self._broken = True
            self.policy_version = None
            raise
        finally:
            self._lock.release()

    def _request(self, operation: str, **fields: Any) -> int:
        request_id = self._request_id
        self._request_id += 1
        send_control(self._connection, {"operation": operation, "request_id": request_id, **fields})
        return request_id

    def _response(self, request_id: int, status: str) -> dict[str, Any]:
        response = receive_control(self._connection)
        if (
            response.get("status") != status
            or type(response.get("request_id")) is not int
            or response["request_id"] != request_id
        ):
            raise WireError("响应状态/次序不符，不能提交版本或接受生成")
        return response

    def sync_params(self, params: Any, model_config: Any, version: int) -> dict[str, Any]:
        import jax
        import numpy as np

        if (
            type(version) is not int
            or version < 0
            or (self.policy_version is not None and version <= self.policy_version)
        ):
            raise ValueError("策略版本必须递增")
        started = time.perf_counter()
        if (
            self.metadata["remote"]["model_config_sha256"]
            != hashlib.sha256(canonical(model_config._asdict())).hexdigest()
        ):
            raise WireError("训练与推理模型配置身份不同")
        descriptor, leaves = describe_tree(params)
        _, records = validate_descriptor(descriptor, model_config)
        source_read_mode = self.config.source_read_mode
        trim_training_allocator = self.config.trim_training_host_allocator_after_transfer
        times = {"source_get_contiguous_s": 0.0, "host_hash_s": 0.0, "socket_send_and_host_ack_s": 0.0}
        receipts = []
        with self._operation():
            request_id = self._request("update", version=version, descriptor=descriptor)
            ready = receive_control(self._connection)
            if canonical(ready) != canonical(
                {"status": "ready", "leaf_count": len(records), "total_bytes": sum(x["bytes"] for x in records)}
            ):
                raise WireError("接收方shape/容量准入不同")
            for leaf, record in zip(leaves, records, strict=True):
                start = time.perf_counter()
                with parameter_host_view(leaf, jax, np, source_read_mode) as host:
                    times["source_get_contiguous_s"] += time.perf_counter() - start
                    if list(host.shape) != record["shape"] or host.nbytes != record["bytes"]:
                        raise WireError("实际源host布局与descriptor不同")
                    raw = memoryview(host).cast("B")
                    start = time.perf_counter()
                    digest = byte_sha(raw)
                    times["host_hash_s"] += time.perf_counter() - start
                    start = time.perf_counter()
                    send_control(
                        self._connection,
                        {
                            "kind": "fp32_leaf",
                            "index": record["index"],
                            "path": record["path"],
                            "bytes": record["bytes"],
                            "sha256": digest,
                        },
                    )
                    self._connection.sendall(raw)
                    ack = receive_control(self._connection)
                    if canonical(ack) != canonical(
                        {"status": "host_received", "index": record["index"], "sha256": digest}
                    ):
                        raise WireError("完整主机接收ACK缺失或错误")
                    times["socket_send_and_host_ack_s"] += time.perf_counter() - start
                    receipts.append(
                        {"index": record["index"], "path": record["path"], "bytes": record["bytes"], "sha256": digest}
                    )
                    del raw
            host_cleanup: dict[str, Any] = {
                "after_all_host_acks": True,
                "before_device_commit_response": True,
                "gc_collected": None,
                "allocator_trim": {"enabled": False},
            }
            if trim_training_allocator:
                # Python会让循环变量继续引用最后一个主机数组；先释放它再请求glibc回收空闲页。
                del host
                host_cleanup["gc_collected"] = gc.collect()
                host_cleanup["allocator_trim"] = trim_host_allocator()
            # 所有host ACK仍不足以更新策略版本；必须收到实际设备提交响应。
            response = self._response(request_id, "device_committed")
            report: dict[str, Any] = response["report"]
            if (
                type(response["version"]) is not int
                or response["version"] != version
                or report.get("complete") is not True
                or type(report["version"]) is not int
                or report["version"] != version
            ):
                raise WireError("实际设备版本未提交")
            transport = report["host_transport"]
            if canonical(transport["leaves"]) != canonical(receipts) or transport["device_commit"] is not True:
                raise WireError("最终transport记录与发送的完整内容不同")
            self.policy_version = version
            report["client_transport"] = {
                "complete": True,
                "times_s": times,
                "total_bytes": sum(x["bytes"] for x in receipts),
                "wall_s": time.perf_counter() - started,
                "source_read_mode": source_read_mode,
                "host_cleanup": host_cleanup,
            }
            return report

    def generate(
        self,
        prompts: list[list[int]],
        batch_rng: BatchRngState,
        sampling: EngineSampling,
        *,
        expected_version: int | None = None,
    ) -> EngineBatch:
        if self.policy_version is None or type(expected_version) is not int or expected_version != self.policy_version:
            raise WireError("请求没有绑定已提交的策略版本")
        with self._operation():
            request_id = self._request(
                "generate",
                version=expected_version,
                prompts=prompts,
                rng=batch_rng.to_dict(),
                sampling=asdict(sampling),
            )
            response = self._response(request_id, "generated")
            value = response["batch"]
            if type(value["policy_version"]) is not int or value["policy_version"] != expected_version:
                raise WireError("返回了不同策略版本")
            return EngineBatch(
                rows=[EngineRow(**row) for row in value["rows"]],
                policy_version=value["policy_version"],
                metadata=value["metadata"],
            )

    def close(self) -> dict[str, Any]:
        if self._closed:
            return {"already_closed": True}
        self._closed = True
        self.policy_version = None
        report: dict[str, Any] = {"complete": False, "errors": [], "forced_group_termination": False}
        process = self._process
        try:
            if self._lock.locked():
                self._broken = True
                report["cancelled_active_request"] = True
                with contextlib.suppress(OSError):
                    self._connection.shutdown(socket.SHUT_RDWR)
            self._connection.settimeout(30)
            if not self._broken and process is not None and process.poll() is None:
                request_id = self._request("close")
                report["remote_close"] = self._response(request_id, "closed")["report"]
        except BaseException as error:
            report["errors"].append({"stage": "remote_close", "type": type(error).__name__, "message": str(error)})
        finally:
            self._connection.close()
            if process is not None:
                wait_started = time.monotonic()
                wait_report: dict[str, Any] = {
                    "timeout_s": WORKER_EXIT_TIMEOUT_S,
                    "timed_out": False,
                    "started_monotonic": wait_started,
                }
                report["worker_wait"] = wait_report
                try:
                    process.wait(timeout=WORKER_EXIT_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    wait_report["timed_out"] = True
                finally:
                    wait_finished = time.monotonic()
                    wait_report["finished_monotonic"] = wait_finished
                    wait_report["wall_s"] = wait_finished - wait_started
                try:
                    group = stop_owned_group(process.pid)
                    report["owned_group_cleanup"] = group
                    # 等待超时后进程可能自行退出；实际发信号与错过期限分别记录。
                    report["forced_group_termination"] = bool(group["signals"])
                    process.wait(timeout=5)
                except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                    report["errors"].append(
                        {"stage": "owned_group_cleanup", "type": type(error).__name__, "message": str(error)}
                    )
                report["worker_exit_code"] = process.returncode
                report["worker_stopped"] = process.poll() is not None
                report["worker_exit_observed_monotonic"] = time.monotonic()
            self._socket_path.unlink(missing_ok=True)
            self._directory.rmdir()
        report["complete"] = bool(
            report.get("remote_close", {}).get("complete")
            and report.get("worker_exit_code") == 0
            and not report["errors"]
            and not report.get("worker_wait", {}).get("timed_out", True)
            and not report["forced_group_termination"]
            and report.get("owned_group_cleanup", {}).get("complete")
        )
        return report
