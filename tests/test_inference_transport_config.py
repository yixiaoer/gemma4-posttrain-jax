"""默认选择与进程边界，避免把尚未接通的独立运行时误标为 ICI。"""

import sys
from pathlib import Path

import pytest

from gemma4_posttrain_jax.inference_remote import RemoteEngineRuntime
from gemma4_posttrain_jax.inference_runtime import EngineConfig, resolve_weight_sync_transport


@pytest.mark.parametrize("backend,expected", [("jax", "host"), ("inference", "device"), ("inference-process", "host")])
def test_default_depends_on_execution_backend(backend, expected):
    assert resolve_weight_sync_transport(backend) == expected
    assert resolve_weight_sync_transport(backend, "host") == "host"


@pytest.mark.parametrize("backend", ["jax", "inference-process"])
def test_explicit_ici_is_not_silently_ignored(backend):
    with pytest.raises(ValueError, match="同进程"):
        resolve_weight_sync_transport(backend, "device")


def test_same_process_config_defaults_to_device_and_allows_host():
    assert EngineConfig(model_path="unused").weight_sync_transport == "device"
    assert EngineConfig(model_path="unused", weight_sync_transport="host").weight_sync_transport == "host"


def test_remote_runtime_rejects_ici_before_starting_worker(tmp_path):
    with pytest.raises(ValueError, match="inference-distributed 共同运行时"):
        RemoteEngineRuntime(
            EngineConfig(model_path="unused", device_indexes=(0, 1)),
            python_executable=Path(sys.executable),
            record_path=tmp_path / "worker.json",
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("backend,transport", [("unknown", "auto"), ("inference", "unknown")])
def test_invalid_transport_names_fail(backend, transport):
    with pytest.raises(ValueError, match="未知"):
        resolve_weight_sync_transport(backend, transport)


@pytest.mark.parametrize(
    "backend,training,inference,expected",
    [
        ("jax", [0, 1, 2, 3], None, (4, 0, 4)),
        ("inference", [0, 1], [2, 3], (2, 2, 4)),
        ("inference-distributed", [0, 2], [1, 3], (2, 2, 4)),
        ("inference-process", [0, 1], [0, 1], (2, 2, 4)),
    ],
)
def test_allocation_respects_runtime_device_namespaces(backend, training, inference, expected):
    from scripts.train_grpo import iteration_device_allocation

    assert tuple(iteration_device_allocation(backend, training, inference).values()) == expected


@pytest.mark.parametrize(
    "backend,training,inference",
    [
        ("inference", [0, 1], [0, 1]),
        ("inference-distributed", [0, 2], [2, 3]),
        ("inference", [0, 1], []),
        ("jax", [0, 0], None),
        ("jax", [], None),
    ],
)
def test_invalid_device_allocations_are_rejected(backend, training, inference):
    from scripts.train_grpo import iteration_device_allocation

    with pytest.raises(ValueError):
        iteration_device_allocation(backend, training, inference)
