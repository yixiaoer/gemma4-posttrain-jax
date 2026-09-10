"""观测数据缺失必须显式保留，不能用零填充造成虚假的HBM结论。"""

from types import SimpleNamespace

from gemma4_posttrain_jax.inference_process import snapshot_device_memory


def test_missing_device_memory_is_not_zero():
    devices = [
        SimpleNamespace(id=0, memory_stats=lambda: None),
        SimpleNamespace(id=1, memory_stats=lambda: {"bytes_in_use": 32}),
    ]
    report = snapshot_device_memory(devices, "control")
    assert not report["complete"]
    assert not any(row["available"] for row in report["devices"])
    assert "stats" not in report["devices"][0]
    assert "peak_bytes_in_use" not in report["devices"][1]["stats"]


def test_one_memory_error_preserves_the_other_device():
    def failed():
        raise RuntimeError("观测控制故障")

    report = snapshot_device_memory(
        [
            SimpleNamespace(id=0, memory_stats=failed),
            SimpleNamespace(id=1, memory_stats=lambda: {"bytes_in_use": 16, "peak_bytes_in_use": 64}),
        ],
        "control",
    )
    assert not report["complete"]
    assert report["devices"][0]["error"]["type"] == "RuntimeError"
    assert report["devices"][1]["available"]
    assert report["devices"][1]["stats"]["peak_bytes_in_use"] == 64


def test_real_zero_is_distinct_from_missing_and_empty_devices():
    report = snapshot_device_memory(
        [
            SimpleNamespace(id=0, memory_stats=lambda: {"bytes_in_use": 0, "peak_bytes_in_use": 0}),
        ],
        "control",
    )
    assert report["complete"] and report["devices"][0]["available"]
    assert not snapshot_device_memory([], "control")["complete"]
