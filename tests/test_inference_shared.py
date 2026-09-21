"""共同运行时的实际设备内容检查、完整映射和版本确认；CPU控制不代替真实引擎。"""

import hashlib
import importlib.util
import socket
import sys
import threading
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from ml_dtypes import bfloat16

from gemma4_posttrain_jax import distributed_ici, inference_local_cpu, inference_shared, inference_weights
from gemma4_posttrain_jax.distributed_ici import SharedRuntimeWeightTransfer
from gemma4_posttrain_jax.inference_runtime import resolve_weight_sync_transport
from gemma4_posttrain_jax.inference_shared import SharedEngineRuntime, target_plan
from gemma4_posttrain_jax.inference_weights import PREFIX, _make_plan, _Mapping
from gemma4_posttrain_jax.inference_wire import WireError, canonical, describe_tree, receive_control, send_control
from gemma4_posttrain_jax.sharding import make_mesh


def test_shared_runtime_has_explicit_device_default():
    assert resolve_weight_sync_transport("inference-distributed") == "device"
    with pytest.raises(ValueError, match="共同运行时"):
        resolve_weight_sync_transport("inference-distributed", "host")
    with pytest.raises(ValueError, match="初始化"):
        inference_shared.shared_devices(0)


@pytest.mark.parametrize("bad_second_source", [False, True])
def test_local_cpu_loaders_restore_functions_and_cache(tmp_path, monkeypatch, bad_second_source):
    # 用实际可检查源码的两个模块，分别核正常退出和安装中途失败。
    records, modules, originals = [], [], []
    for index, name in enumerate(("cpu_mesh", "model_weights_single_file_generator")):
        module_name = f"shared_loader_control_{index}"
        path = tmp_path / f"{module_name}.py"
        function = f'def {name}():\n    return jax.devices("cpu")\n'
        path.write_text("import jax\n_cpu_mesh = object()\n" + function)
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setitem(sys.modules, module_name, module)
        modules.append(module)
        originals.append(getattr(module, name))
        digest = hashlib.sha256(function.encode()).hexdigest()
        records.append((module_name, name, "invalid" if index == 1 and bad_second_source else digest))
    monkeypatch.setattr(inference_local_cpu, "LOADERS", tuple(records))
    cache = modules[0]._cpu_mesh
    adapter = inference_local_cpu.LocalCpuLoaders()
    if bad_second_source:
        with pytest.raises(RuntimeError, match="固定版本"):
            adapter.__enter__()
    else:
        with adapter:
            assert modules[0]._cpu_mesh is None
            assert modules[0].cpu_mesh() == jax.local_devices(backend="cpu")
            assert modules[1].model_weights_single_file_generator() == jax.local_devices(backend="cpu")
            assert len(adapter.metadata) == 2
    assert modules[0]._cpu_mesh is cache
    for module, original, (_, name, _) in zip(modules, originals, records, strict=True):
        assert getattr(module, name) is original


def test_verified_transfer_all_finite_bf16_odd_rows_and_tail():
    bits = np.arange(65536, dtype=np.uint16)
    bits = bits[(bits & 0x7F80) != 0x7F80].reshape(255, 256)
    fp32 = (bits.astype(np.uint32) << 16).view(np.float32)
    source = jax.device_put(fp32, NamedSharding(make_mesh(jax.devices()[:2]), P(None, "d")))
    transfer = SharedRuntimeWeightTransfer(jax.devices(), chunk_bytes=7 * 256 * 2, verify=True)
    mapped, finite = transfer.map_source(_Mapping("test", fp32.shape, (("source", source),)))
    assert bool(finite)
    actual, report = transfer.transfer(fp32.shape, mapped)
    assert report["all_replicas_bitwise_verified"] and report["host_payload_bytes"] == 0
    assert report["chunks"] == 19
    for value in actual.values():
        np.testing.assert_array_equal(np.asarray(value).view(np.uint16), bits)
    np.testing.assert_array_equal(np.asarray(source).view(np.uint32), fp32.view(np.uint32))


def test_verified_transfer_detects_second_replica_write_error(monkeypatch):
    original_write = distributed_ici._write
    target_device = jax.devices()[3]

    def bad_write(rows, length):
        write = original_write(rows, length)

        def apply(value, bits, offset):
            result = write(value, bits, offset)
            if target_device in result.sharding.device_set:
                result = result.at[offset].set(result[offset] ^ jnp.uint16(1))
            return result

        return apply

    monkeypatch.setattr(distributed_ici, "_write", bad_write)
    transfer = SharedRuntimeWeightTransfer(jax.devices(), chunk_bytes=16, verify=True)
    source = jax.device_put(np.arange(16, dtype=np.uint16).reshape(4, 4), transfer.source_sharding)
    with pytest.raises(ValueError, match="实际内容"):
        transfer.transfer((4, 4), source)


def test_target_plan_checks_all_sources_and_targets(tiny_a):
    _, params, config = tiny_a
    plan, preserved = _make_plan(params, config, lambda _: 1)
    sharding = NamedSharding(make_mesh(jax.devices()[2:]), P())

    class Param:
        def __init__(self, shape):
            self.value = jax.device_put(np.zeros(shape, dtype=bfloat16), sharding)

        def get_value(self):
            return self.value

    parameters = {item.name: Param(item.shape) for item in plan}
    parameters.update({name: Param((1,)) for name in preserved})
    modules = {}
    for index in range(config.num_hidden_layers):
        width = next(
            item.shape[1] // 2 for item in plan if item.name == PREFIX + f"layers.{index}.mlp.gate_up_proj.weight"
        )
        method = type("UnquantizedMergedLinearMethod", (), {})()
        method.linear_config = SimpleNamespace(output_sizes=[width, width], n_shards=1)
        modules[PREFIX + f"layers.{index}.mlp.gate_up_proj"] = SimpleNamespace(
            quant_method=method, output_sizes=[width, width]
        )
    model = type("Gemma4ForConditionalGeneration", (), {})()
    model.named_parameters = lambda: list(parameters.items())
    model.named_modules = lambda: list(modules.items())
    runner = SimpleNamespace(model=model, mesh=sharding.mesh)
    descriptor, _ = describe_tree(params)
    actual, _ = target_plan(runner, descriptor, config)
    assert [(p.name, p.shape) for p in actual] == [(p.name, p.shape) for p in plan]
    parameters.pop(plan[-1].name)
    with pytest.raises(WireError, match="完整"):
        target_plan(runner, descriptor, config)


@pytest.mark.parametrize("reply_version", [2, 3])
def test_client_commits_version_only_after_device_ack(monkeypatch, reply_version):
    # 控制消息用真实socket；数据传输在前两个实际collective测试中检查。
    client, server = socket.socketpair()
    client.settimeout(5)
    server.settimeout(5)
    params = jnp.array([1.0], dtype=jnp.float32)
    item = _Mapping("test", (1,), (("source", params),))
    monkeypatch.setattr(inference_weights, "_make_plan", lambda *args: ([item], set()))
    monkeypatch.setattr(inference_shared, "describe_tree", lambda p: ({"test": True}, [p]))
    config = SimpleNamespace(_asdict=lambda: {"test": True})
    runtime = SharedEngineRuntime.__new__(SharedEngineRuntime)
    runtime._closed = runtime._broken = False
    runtime._lock = threading.Lock()
    runtime._request_id = 0
    runtime._connection = client
    runtime.policy_version = 1
    runtime.metadata = {
        "remote": {"merged_shards": [1], "model_config_sha256": hashlib.sha256(canonical(config._asdict())).hexdigest()}
    }
    runtime.transfer = SimpleNamespace(
        map_source=lambda item: (jnp.array([0x3F80], dtype=jnp.uint16), jnp.array(True)),
        transfer=lambda shape, bits: ({}, {"all_replicas_bitwise_verified": True}),
    )
    errors = []

    def respond():
        try:
            request = receive_control(server)
            assert request["version"] == 2
            send_control(server, {"status": "receiving", "request_id": 0})
            assert receive_control(server)["index"] == 0
            send_control(
                server,
                {"status": "device_committed", "request_id": 0, "report": {"complete": True, "version": reply_version}},
            )
        except BaseException as error:
            errors.append(error)
        finally:
            server.close()

    thread = threading.Thread(target=respond)
    thread.start()
    try:
        if reply_version == 2:
            assert runtime.sync_params(params, config, 2)["client_transport"]["host_payload_bytes"] == 0
            assert runtime.policy_version == 2 and not runtime._broken
        else:
            with pytest.raises(WireError, match="提交"):
                runtime.sync_params(params, config, 2)
            assert runtime.policy_version is None and runtime._broken
    finally:
        client.close()
        thread.join(timeout=5)
    assert not thread.is_alive() and not errors
