"""真实socket协议与CPU进程组控制；设备提交响应在本测试中为显式模拟。"""

import contextlib
import hashlib
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

import gemma4_posttrain_jax.inference_remote as inference_remote
from gemma4_posttrain_jax.inference_process import receive_parameters
from gemma4_posttrain_jax.inference_remote import (
    DEFAULT_WORKER_MODULE,
    WORKER_EXIT_TIMEOUT_S,
    RemoteEngineRuntime,
    live_owned_group_members,
    validate_worker_module,
)
from gemma4_posttrain_jax.inference_wire import WireError, canonical, receive_control, send_control


def test_worker_module_is_a_single_importable_name() -> None:
    assert validate_worker_module(DEFAULT_WORKER_MODULE) == "gemma4_posttrain_jax.inference_process"
    assert validate_worker_module("reverse.rollout.q01_hidden_worker") == "reverse.rollout.q01_hidden_worker"
    for value in ("", "reverse/worker", "reverse.worker --flag", ".worker", "worker."):
        with pytest.raises(ValueError, match="Python模块名"):
            validate_worker_module(value)


def client_for_socket(connection, config):
    runtime = RemoteEngineRuntime.__new__(RemoteEngineRuntime)
    runtime._closed = False
    runtime._broken = False
    runtime._lock = threading.Lock()
    runtime._connection = connection
    runtime._request_id = 0
    runtime.policy_version = 0
    runtime.config = SimpleNamespace(source_read_mode="direct", trim_training_host_allocator_after_transfer=False)
    runtime.metadata = {"remote": {"model_config_sha256": hashlib.sha256(canonical(config._asdict())).hexdigest()}}
    return runtime


@pytest.mark.parametrize("finish", ["commit", "disconnect", "wrong_request", "boolean_version", "host_only"])
def test_host_receive_cannot_advance_version_without_valid_commit(tiny_a, finish):
    _, params, config = tiny_a
    host = jax.tree.map(lambda a: np.asarray(a, dtype=np.float32, order="C"), params)
    left, right = socket.socketpair()
    with left, right, ThreadPoolExecutor(max_workers=1) as pool:
        left.settimeout(5)
        right.settimeout(5)
        client = client_for_socket(left, config)
        future = pool.submit(client.sync_params, host, config, 1)
        request = receive_control(right)
        assert request["operation"] == "update" and request["version"] == 1
        received, transport = receive_parameters(right, request["descriptor"], config)
        for expected, actual in zip(jax.tree.leaves(host), jax.tree.leaves(received), strict=True):
            np.testing.assert_array_equal(expected.view(np.uint32), actual.view(np.uint32))
        # 全部主机字节和ACK都到达后，调用仍阻塞、旧版本仍保持。
        assert not future.done() and client.policy_version == 0 and transport["device_commit"] is False
        if finish == "disconnect":
            right.shutdown(socket.SHUT_RDWR)
        else:
            transport["device_commit"] = finish != "host_only"
            send_control(
                right,
                {
                    "status": "device_committed",
                    "request_id": 7 if finish == "wrong_request" else request["request_id"],
                    "version": True if finish == "boolean_version" else 1,
                    "report": {"complete": True, "version": 1, "host_transport": transport},
                },
            )
        if finish == "commit":
            client_transport = future.result(timeout=5)["client_transport"]
            assert client_transport["complete"]
            assert client_transport["host_cleanup"] == {
                "after_all_host_acks": True,
                "before_device_commit_response": True,
                "gc_collected": None,
                "allocator_trim": {"enabled": False},
            }
            assert client.policy_version == 1 and not client._broken
        else:
            with pytest.raises(WireError):
                future.result(timeout=5)
            assert client.policy_version is None and client._broken
            with pytest.raises(WireError, match="关闭或此前失败"), client._operation():
                pytest.fail("失效后端不应进入下一次调用")


def test_training_allocator_trim_runs_after_host_receive_and_before_device_response(tiny_a, monkeypatch):
    _, params, config = tiny_a
    trim_called = threading.Event()
    monkeypatch.setattr(inference_remote.gc, "collect", lambda: 17)

    def trim():
        trim_called.set()
        return {"enabled": True, "returned": 1, "rss_before_bytes": 20, "rss_after_bytes": 10, "wall_s": 0.1}

    monkeypatch.setattr(inference_remote, "trim_host_allocator", trim)
    left, right = socket.socketpair()
    with left, right, ThreadPoolExecutor(max_workers=1) as pool:
        left.settimeout(5)
        right.settimeout(5)
        client = client_for_socket(left, config)
        client.config = SimpleNamespace(
            source_read_mode="transient_copy", trim_training_host_allocator_after_transfer=True
        )
        future = pool.submit(client.sync_params, params, config, 1)
        request = receive_control(right)
        _, transport = receive_parameters(right, request["descriptor"], config)
        assert trim_called.wait(timeout=5)
        assert not future.done() and client.policy_version == 0
        transport["device_commit"] = True
        send_control(
            right,
            {
                "status": "device_committed",
                "request_id": request["request_id"],
                "version": 1,
                "report": {"complete": True, "version": 1, "host_transport": transport},
            },
        )
        cleanup = future.result(timeout=5)["client_transport"]["host_cleanup"]

    assert cleanup == {
        "after_all_host_acks": True,
        "before_device_commit_response": True,
        "gc_collected": 17,
        "allocator_trim": {
            "enabled": True,
            "returned": 1,
            "rss_before_bytes": 20,
            "rss_after_bytes": 10,
            "wall_s": 0.1,
        },
    }


def test_reentrant_request_does_not_cancel_active_request(tiny_a):
    _, _, config = tiny_a
    left, right = socket.socketpair()
    with left, right:
        client = client_for_socket(left, config)
        with client._operation(), pytest.raises(WireError, match="并发或重入"), client._operation():
            pytest.fail("不允许重入")
        assert not client._broken and client.policy_version == 0


@pytest.mark.skipif(sys.platform != "linux", reason="P6固定Linux TPU VM的/proc与session协议")
def test_close_cleans_orphan_in_own_session_and_keeps_unrelated_process():
    code = (
        "import subprocess,sys,os; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        "print(p.pid,flush=True); os._exit(0)"
    )
    worker = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        assert worker.stdout is not None
        orphan = int(worker.stdout.readline())
        worker.wait(timeout=5)
        assert live_owned_group_members(worker.pid) == [orphan]
        runtime = RemoteEngineRuntime.__new__(RemoteEngineRuntime)
        runtime._closed = False
        runtime._broken = True
        runtime._lock = threading.Lock()
        runtime._connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        runtime._process = worker
        runtime._directory = Path(tempfile.mkdtemp(prefix="g4-orphan-control-"))
        runtime._socket_path = runtime._directory / "engine.sock"
        report = runtime.close()
        assert report["complete"] is False
        assert report["worker_stopped"] and report["forced_group_termination"]
        assert report["owned_group_cleanup"]["complete"]
        assert report["owned_group_cleanup"]["initial_live_members"] == [orphan]
        assert not live_owned_group_members(worker.pid)
        assert unrelated.poll() is None
        assert not runtime._directory.exists()
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(worker.pid, signal.SIGKILL)
        if worker.stdout is not None:
            worker.stdout.close()
        worker.wait(timeout=5)
        unrelated.terminate()
        unrelated.wait(timeout=5)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux进程组边界")
def test_cannot_target_training_process_group():
    with pytest.raises(ValueError, match="训练进程组"):
        live_owned_group_members(os.getpgrp())


@pytest.mark.skipif(sys.platform != "linux", reason="Linux进程组边界")
@pytest.mark.parametrize(("exit_code", "timed_out"), [(0, False), (0, True), (1, True)])
def test_wait_timeout_is_not_a_signal_and_cannot_qualify_close(monkeypatch, exit_code, timed_out):
    """重放wait报告超时、组扫描时worker已自行退出的竞态。"""
    worker = subprocess.Popen(
        [sys.executable, "-c", f"raise SystemExit({exit_code})"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    original_wait = worker.wait
    original_poll = worker.poll
    try:
        worker.wait(timeout=5)
        calls = []
        polls = []

        def initially_live_poll():
            polls.append(True)
            return None if len(polls) == 1 else original_poll()

        def first_wait_times_out(timeout=None):
            calls.append(timeout)
            if len(calls) == 1 and timed_out:
                raise subprocess.TimeoutExpired(worker.args, timeout)
            return original_wait(timeout=timeout)

        monkeypatch.setattr(worker, "wait", first_wait_times_out)
        monkeypatch.setattr(worker, "poll", initially_live_poll)
        # 只重放超时竞态；若误发信号立即使本测试失败。
        monkeypatch.setattr(os, "killpg", lambda *args: pytest.fail("已退出的空组不应收到信号"))
        runtime = RemoteEngineRuntime.__new__(RemoteEngineRuntime)
        runtime._closed = False
        runtime._broken = False
        runtime._lock = threading.Lock()
        runtime._connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        runtime._process = worker
        runtime._directory = Path(tempfile.mkdtemp(prefix="g4-wait-timeout-control-"))
        runtime._socket_path = runtime._directory / "engine.sock"
        # close响应为显式模拟；这样退出0的成功控制只因wait是否超时而改变资格。
        monkeypatch.setattr(runtime, "_request", lambda operation: 0)
        monkeypatch.setattr(runtime, "_response", lambda request_id, status: {"report": {"complete": True}})
        report = runtime.close()
        assert calls == [WORKER_EXIT_TIMEOUT_S, 5]
        assert report["worker_wait"]["timed_out"] is timed_out
        assert report["worker_wait"]["wall_s"] >= 0
        assert report["worker_stopped"] and report["worker_exit_code"] == exit_code
        assert report["forced_group_termination"] is False
        assert report["owned_group_cleanup"]["initial_live_members"] == []
        assert report["owned_group_cleanup"]["signals"] == []
        assert report["owned_group_cleanup"]["complete"]
        assert report["remote_close"]["complete"]
        assert report["complete"] is (exit_code == 0 and not timed_out)
        assert not runtime._directory.exists()
    finally:
        original_wait(timeout=5)
