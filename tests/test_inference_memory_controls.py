"""双进程权重同步的主机内存候选必须保持源数组和关闭语义。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from gemma4_posttrain_jax.host_memory import trim_host_allocator
from gemma4_posttrain_jax.inference_remote import parameter_host_view
from gemma4_posttrain_jax.inference_runtime import EngineConfig


def test_transient_parameter_read_does_not_read_or_delete_source() -> None:
    source = jax.device_put(np.arange(8, dtype=np.float32))
    copies = []

    def copy(value, sharding, *, may_alias):
        assert value is source and may_alias is False
        result = jax.device_put(value, sharding, may_alias=may_alias)
        copies.append(result)
        return result

    def read(value):
        assert value is copies[0] and value is not source
        return jax.device_get(value)

    observed_jax = SimpleNamespace(Array=jax.Array, device_put=copy, device_get=read)
    with parameter_host_view(source, observed_jax, np, "transient_copy") as host:
        np.testing.assert_array_equal(host, np.arange(8, dtype=np.float32))
        assert not copies[0].is_deleted()

    assert copies[0].is_deleted()
    assert not source.is_deleted()


def test_direct_parameter_read_uses_source() -> None:
    source = np.arange(4, dtype=np.float32)
    observed_jax = SimpleNamespace(device_get=lambda value: value)

    with parameter_host_view(source, observed_jax, np, "direct") as host:
        np.testing.assert_array_equal(host, source)


def test_transient_parameter_read_deletes_copy_after_body_error() -> None:
    source = jax.device_put(np.arange(4, dtype=np.float32))
    copies = []

    def copy(value, sharding, *, may_alias):
        result = jax.device_put(value, sharding, may_alias=may_alias)
        copies.append(result)
        return result

    observed_jax = SimpleNamespace(Array=jax.Array, device_put=copy, device_get=jax.device_get)
    with (
        pytest.raises(RuntimeError, match="发送失败"),
        parameter_host_view(source, observed_jax, np, "transient_copy"),
    ):
        raise RuntimeError("发送失败")

    assert copies[0].is_deleted()
    assert not source.is_deleted()


@pytest.mark.skipif(not Path("/proc/self/statm").is_file(), reason="glibc 回收控制需要 Linux /proc")
def test_malloc_trim_reports_rss_and_return_code() -> None:
    result = trim_host_allocator()

    assert result["enabled"] is True
    assert result["returned"] in (0, 1)
    assert result["rss_before_bytes"] > 0
    assert result["rss_after_bytes"] > 0
    assert result["wall_s"] >= 0


def test_training_allocator_trim_requires_transient_copy() -> None:
    with pytest.raises(ValueError, match="transient_copy"):
        EngineConfig(model_path="fixture", trim_training_host_allocator_after_transfer=True)

    config = EngineConfig(
        model_path="fixture",
        source_read_mode="transient_copy",
        trim_training_host_allocator_after_transfer=True,
    )
    assert config.trim_training_host_allocator_after_transfer is True
