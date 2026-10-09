"""The layout record reads shard attributes only, and keeps requested backends apart from actual shardings."""

from types import SimpleNamespace

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from gemma4_posttrain_jax.diagnostics import array_layout_manifest, execution_layout_record


def test_layout_record_never_reads_array_contents():
    device = SimpleNamespace(id=3, process_index=0)

    class UnreadableArray:
        shape = (8, 16)
        dtype = np.dtype("float32")
        sharding = "sentinel-sharding"
        addressable_shards = (SimpleNamespace(device=device, index=(slice(4, 8), slice(None))),)

        def __array__(self, *args, **kwargs):
            raise AssertionError("recording the layout must not read array contents")

    assert array_layout_manifest({"master": UnreadableArray()}) == [
        {
            "path": "['master']",
            "shape": [8, 16],
            "dtype": "float32",
            "sharding": "sentinel-sharding",
            "addressable_shards": [
                {"device_id": 3, "process_index": 0, "index": "(slice(4, 8, None), slice(None, None, None))"}
            ],
        }
    ]


def test_layout_record_reads_real_cpu_sharding():
    device = jax.devices("cpu")[0]
    mesh = Mesh(np.array([device]), ("fsdp",))
    master = jax.device_put(np.ones((4, 8), np.float32), NamedSharding(mesh, PartitionSpec("fsdp", None)))
    record = execution_layout_record(
        {"master": master, "adam": master},
        {"weights": master.astype("bfloat16")},
        mesh,
        rollout_backend="jax",
        rollout_layout="fsdp",
        logprob_backend="jax",
    )
    assert record["process_count"] == 1
    assert record["mesh"]["shape"] == {"fsdp": 1}
    assert record["mesh"]["devices"][0]["platform"] == "cpu"
    assert [row["dtype"] for row in record["training_state"]] == ["float32", "float32"]
    assert record["rollout_params"][0]["dtype"] == "bfloat16"
    assert record["rollout_params"][0]["addressable_shards"][0]["device_id"] == device.id
    assert record["selected_backends"] == {"rollout": "jax", "logprob": "jax"}
    assert record["requested_rollout_layout"] == "fsdp"


def test_layout_record_without_native_rollout_params():
    device = jax.devices("cpu")[0]
    mesh = Mesh(np.array([device]), ("fsdp",))
    record = execution_layout_record(
        {"master": np.zeros(2, np.float32)},
        None,
        mesh,
        rollout_backend="tpu-inference",
        rollout_layout="replicated",
        logprob_backend="jax",
    )
    assert record["rollout_params"] is None
    assert record["training_state"][0]["addressable_shards"] == []
