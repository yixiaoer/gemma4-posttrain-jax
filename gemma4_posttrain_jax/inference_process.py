"""固定推理环境的本机Unix socket服务；完整参数收妥与设备版本提交分开确认。

导入时不初始化JAX；实际引擎由独立进程在指定物理芯片上构造。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import socket
import sys
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

from .host_memory import trim_host_allocator
from .inference_rng import BatchRngState
from .inference_runtime import EngineConfig, EngineRuntime, EngineSampling
from .inference_wire import (
    PROTOCOL,
    WireError,
    byte_sha,
    canonical,
    rebuild_tree,
    receive_control,
    receive_into,
    send_control,
    validate_descriptor,
)


def receive_parameters(
    connection: socket.socket, descriptor: dict[str, Any], config: Any
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    _, records = validate_descriptor(descriptor, config)
    send_control(
        connection, {"status": "ready", "leaf_count": len(records), "total_bytes": sum(x["bytes"] for x in records)}
    )
    leaves = []
    received = []
    read_s = 0.0
    hash_s = 0.0
    started = time.perf_counter()
    for record in records:
        packet = receive_control(connection)
        expected = {"kind": "fp32_leaf", "index": record["index"], "path": record["path"], "bytes": record["bytes"]}
        if set(packet) != set(expected) | {"sha256"} or canonical({k: packet[k] for k in expected}) != canonical(
            expected
        ):
            raise WireError("实际tensor帧顺序或长度与已核descriptor不同")
        digest = packet["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise WireError("tensor帧缺少合法SHA")
        leaf = np.empty(record["shape"], dtype="<f4", order="C")
        raw = memoryview(leaf).cast("B")
        start = time.perf_counter()
        receive_into(connection, raw)
        read_s += time.perf_counter() - start
        start = time.perf_counter()
        if byte_sha(raw) != digest:
            raise WireError("接收的完整FP32字节SHA不符")
        hash_s += time.perf_counter() - start
        leaves.append(leaf)
        received.append({"index": record["index"], "path": record["path"], "bytes": len(raw), "sha256": digest})
        # 只证明接收方host数组已持有完整副本；尚未提交推理设备参数。
        send_control(connection, {"status": "host_received", "index": record["index"], "sha256": digest})
    return rebuild_tree(descriptor, leaves), {
        "protocol": PROTOCOL,
        "complete_host_receive": True,
        "device_commit": False,
        "leaf_count": len(leaves),
        "total_bytes": sum(x["bytes"] for x in records),
        "leaves": received,
        "socket_receive_s": read_s,
        "host_hash_s": hash_s,
        "host_receive_wall_s": time.perf_counter() - started,
    }


def open_accel_devices() -> list[str]:
    """从本进程的文件描述符观察实际设备，环境声明单独记录。"""
    devices = set()
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        if target.startswith("/dev/accel"):
            devices.add(target)
    return sorted(devices)


def snapshot_device_memory(devices: list[Any], stage: str) -> dict[str, Any]:
    """记录当前进程的真实设备计数器；不把缺失峰值当成零或核函数独占用量。"""
    started = time.perf_counter()
    rows = []
    for device in devices:
        row: dict[str, Any] = {"local_device_id": int(device.id), "available": False}
        try:
            stats = device.memory_stats()
            if stats is not None:
                row["stats"] = dict(stats)
                required = [stats.get(name) for name in ("bytes_in_use", "peak_bytes_in_use")]
                row["available"] = all(type(value) is int and value >= 0 for value in required)
        except Exception as error:
            # 诊断失败独立记录，不撤销已经完成的参数事务或生成结果。
            row["error"] = {"type": type(error).__name__, "message": str(error)}
        rows.append(row)
    return {
        "stage": stage,
        "complete": bool(rows) and all(row["available"] for row in rows),
        "devices": rows,
        "observation_s": time.perf_counter() - started,
        "scope": "本进程各本地设备allocator当前值与累计峰值；不重置计数器，不代表单次请求的独占峰值。",
    }


def serve(path: Path, engine_config: EngineConfig, record_path: Path) -> None:
    nonce = os.environ.pop("GEMMA4_ENGINE_NONCE")
    physical = os.environ.get("TPU_VISIBLE_CHIPS", "")
    if physical != "2,3" or tuple(engine_config.device_indexes) != (0, 1):
        raise ValueError("P6资格固定物理chips2,3、进程本地devices0,1")
    if path.exists() or path.is_symlink():
        raise ValueError("不能覆盖已有socket路径")
    report: dict[str, Any] = {
        "complete": False,
        "pid": os.getpid(),
        "physical_chips": physical,
        "requests": [],
        "errors": [],
        "memory_snapshots": [],
    }

    def save() -> None:
        temporary = record_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(record_path)

    runtime: Any = None
    expected_request = 0
    save()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(path))
        os.chmod(path, 0o600)
        listener.listen(1)
        listener.settimeout(900)
        try:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(900)
                hello = receive_control(connection)
                if hello != {"protocol": PROTOCOL, "nonce": nonce, "operation": "hello"}:
                    raise WireError("本次进程握手不符")
                import jax

                from .inference_weights import apply_gemma4_params
                from .model import config_from_hf

                config = config_from_hf(json.loads((Path(engine_config.model_path) / "config.json").read_text()))
                devices = jax.devices("tpu")
                if len(devices) != 2 or sorted(int(d.id) for d in devices) != [0, 1]:
                    raise RuntimeError("独立引擎没有获得两个本地设备")
                report["tpu_bootstrap"] = {
                    "declared_versions": {
                        name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "libtpu")
                    },
                    "environment_library": os.environ.get("TPU_LIBRARY_PATH"),
                    "mapped_library_paths": sorted(
                        {
                            line.split(maxsplit=5)[-1]
                            for line in Path("/proc/self/maps").read_text().splitlines()
                            if "/libtpu.so" in line
                        }
                    ),
                    "open_accel_devices": open_accel_devices(),
                }
                save()
                runtime = EngineRuntime.__new__(EngineRuntime)
                runtime.__init__(engine_config)
                report["memory_snapshots"].append(snapshot_device_memory(devices, "initialized"))
                metadata = {
                    "protocol": PROTOCOL,
                    "pid": os.getpid(),
                    "python_executable": sys.executable,
                    "python_prefix": sys.prefix,
                    "physical_chips": [2, 3],
                    "devices": [
                        {
                            "id": int(d.id),
                            "kind": d.device_kind,
                            "coords": list(d.coords),
                            "local_hardware_id": int(d.local_hardware_id),
                        }
                        for d in devices
                    ],
                    "open_accel_devices": open_accel_devices(),
                    "engine": runtime.metadata,
                    "model_config_sha256": hashlib.sha256(canonical(config._asdict())).hexdigest(),
                }
                report["initialization"] = metadata
                save()
                send_control(connection, {"status": "ready", "metadata": metadata})
                while True:
                    request = receive_control(connection)
                    if type(request.get("request_id")) is not int or request["request_id"] != expected_request:
                        raise WireError("请求次序错误或重放")
                    expected_request += 1
                    operation = request.get("operation")
                    started = time.perf_counter()
                    entry: dict[str, Any] = {
                        "request_id": request["request_id"],
                        "operation": operation,
                        "complete": False,
                    }
                    report["requests"].append(entry)
                    save()
                    if operation == "update":
                        if set(request) != {"operation", "request_id", "version", "descriptor"}:
                            raise WireError("update字段不符")
                        version = request["version"]
                        if (
                            type(version) is not int
                            or version < 0
                            or (runtime.policy_version is not None and version <= runtime.policy_version)
                        ):
                            raise WireError("设备策略版本必须递增")
                        params, transport = receive_parameters(connection, request["descriptor"], config)
                        update = runtime.update_weights(
                            partial(apply_gemma4_params, params=params, config=config), version
                        )
                        if not update["complete"] or runtime.policy_version != version:
                            raise WireError("设备更新尚未提交")
                        transport["device_commit"] = True
                        update["host_transport"] = transport
                        del params
                        collected = gc.collect()
                        update["host_cleanup"] = {
                            "gc_collected": collected,
                            "allocator_trim": trim_host_allocator()
                            if engine_config.trim_host_allocator_after_update
                            else {"enabled": False},
                        }
                        report["memory_snapshots"].append(snapshot_device_memory(devices, f"update_{version}"))
                        entry.update(complete=True, version=version, wall_s=time.perf_counter() - started)
                        save()
                        send_control(
                            connection,
                            {
                                "status": "device_committed",
                                "request_id": request["request_id"],
                                "version": version,
                                "report": update,
                            },
                        )
                    elif operation == "generate":
                        if set(request) != {"operation", "request_id", "version", "prompts", "rng", "sampling"}:
                            raise WireError("generate字段不符")
                        if type(request["version"]) is not int:
                            raise WireError("生成必须绑定实际整数策略版本")
                        key = request["rng"]
                        if set(key) != {"implementation", "words", "protocol"}:
                            raise WireError("batch key字段不符")
                        rng = BatchRngState(key["implementation"], tuple(key["words"]))
                        if rng.to_dict() != key:
                            raise WireError("batch key协议不符")
                        sampling_args = dict(request["sampling"])
                        sampling_args["eos_ids"] = tuple(sampling_args["eos_ids"])
                        sampling = EngineSampling(**sampling_args)
                        batch = runtime.generate(request["prompts"], rng, sampling, expected_version=request["version"])
                        report["memory_snapshots"].append(
                            snapshot_device_memory(devices, f"generate_request_{request['request_id']}")
                        )
                        entry.update(complete=True, version=batch.policy_version, wall_s=time.perf_counter() - started)
                        save()
                        send_control(
                            connection,
                            {"status": "generated", "request_id": request["request_id"], "batch": asdict(batch)},
                        )
                    elif operation == "close":
                        if set(request) != {"operation", "request_id"}:
                            raise WireError("close字段不符")
                        report["memory_snapshots"].append(snapshot_device_memory(devices, "before_close"))
                        report["close"] = runtime.close()
                        report["memory_snapshots"].append(snapshot_device_memory(devices, "after_close"))
                        entry.update(complete=report["close"]["complete"], wall_s=time.perf_counter() - started)
                        report["complete"] = entry["complete"]
                        save()
                        send_control(
                            connection,
                            {"status": "closed", "request_id": request["request_id"], "report": report["close"]},
                        )
                        break
                    else:
                        raise WireError("未知操作")
        except BaseException as error:
            report["errors"].append({"type": type(error).__name__, "message": str(error)})
            save()
            raise
        finally:
            if runtime is not None and not runtime._closed:
                try:
                    report["emergency_close"] = runtime.close()
                except BaseException as error:
                    report["errors"].append(
                        {"type": type(error).__name__, "message": str(error), "stage": "emergency_close"}
                    )
            save()
            path.unlink(missing_ok=True)


def inspect_environment() -> dict[str, Any]:
    """CPU预检实际目标解释器/安装源/库选择；实际TPU映射仍由构造引擎时另核。"""
    from jax._src.cloud_tpu_init import get_tpu_library_path

    from .inference_runtime import EXPECTED_LIBTPU_SHA256, EXPECTED_SOURCE_SHA256, EXPECTED_VERSIONS, file_sha256

    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise ValueError("环境预检必须显式限制CPU")
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    distribution = importlib.metadata.distribution("tpu-inference")
    sources = {name: file_sha256(Path(str(distribution.locate_file(name)))) for name in EXPECTED_SOURCE_SHA256}
    library = Path(str(importlib.metadata.distribution("libtpu").locate_file("libtpu/libtpu.so"))).resolve()
    library_sha = file_sha256(library)
    selected = get_tpu_library_path()
    if versions != EXPECTED_VERSIONS or sources != EXPECTED_SOURCE_SHA256 or library_sha != EXPECTED_LIBTPU_SHA256:
        raise RuntimeError("目标推理环境身份不符合既有资格")
    if selected is None or Path(selected).resolve() != library:
        raise RuntimeError("目标解释器没有选择自身的已核定libtpu")
    return {
        "versions": versions,
        "source_sha256": sources,
        "libtpu": {"path": str(library), "sha256": library_sha},
        "selected_library": str(Path(selected).resolve()),
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "scope": "CPU预检；没有初始化TPU或读取实际设备maps",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-environment", action="store_true", help="只输出目标解释器的实际固定环境身份")
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--engine-config")
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    if args.inspect_environment:
        if args.socket is not None or args.engine_config is not None or args.record is not None:
            parser.error("环境预检不接收引擎构造参数")
        print(canonical(inspect_environment()).decode(), flush=True)
        return
    if args.socket is None or args.engine_config is None or args.record is None:
        parser.error("服务模式需要socket、engine-config和record")
    config = json.loads(args.engine_config)
    config["device_indexes"] = tuple(config["device_indexes"])
    serve(args.socket, EngineConfig(**config), args.record)


if __name__ == "__main__":
    main()
