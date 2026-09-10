"""跨进程控制帧与完整参数结构：字节/shape/路径错误必须在运输前被拒绝。"""

from __future__ import annotations

import copy
import socket
import struct

import jax
import numpy as np
import pytest

from gemma4_posttrain_jax.inference_wire import (
    MAX_CONTROL_BYTES,
    PROTOCOL,
    WireError,
    canonical,
    describe_tree,
    rebuild_tree,
    receive_control,
    send_control,
    validate_descriptor,
)


def test_child_library_selection_does_not_inherit_training_override():
    from gemma4_posttrain_jax.inference_remote import isolated_library_environment

    parent = {"TPU_LIBRARY_PATH": "/training/libtpu.so", "PYTHONPATH": "/frozen/src", "OMP_NUM_THREADS": "2"}
    child = isolated_library_environment(parent)
    assert "TPU_LIBRARY_PATH" not in child
    assert parent["TPU_LIBRARY_PATH"] == "/training/libtpu.so"
    assert child == {"PYTHONPATH": "/frozen/src", "OMP_NUM_THREADS": "2"}


def test_complete_tiny_parameter_tree_and_all_leaf_bytes(tiny_a):
    _, params, config = tiny_a
    host = jax.tree.map(lambda value: np.asarray(value, dtype=np.float32), params)
    descriptor, leaves = describe_tree(host)
    shape_tree, records = validate_descriptor(descriptor, config)
    assert type(shape_tree) is type(params)
    assert len(records) == len(jax.tree.leaves(params))
    decoded = rebuild_tree(descriptor, [x.copy(order="C") for x in leaves])
    assert jax.tree.structure(decoded) == jax.tree.structure(params)
    for first, second in zip(jax.tree.leaves(host), jax.tree.leaves(decoded), strict=True):
        np.testing.assert_array_equal(first.view(np.uint32), second.view(np.uint32))


@pytest.mark.parametrize("mutation", ["path", "index", "dtype", "shape", "bytes", "type"])
def test_metadata_corruption_is_rejected_without_allocating_tensors(mutation):
    descriptor, _ = describe_tree((np.zeros((2, 3), np.float32),))
    bad = copy.deepcopy(descriptor)
    leaf = bad["tree"]["children"][0]
    if mutation == "path":
        leaf["path"] = "wrong"
    elif mutation == "index":
        leaf["index"] = True
    elif mutation == "dtype":
        leaf["dtype"] = ">f4"
    elif mutation == "shape":
        leaf["shape"] = [True, 3]
    elif mutation == "bytes":
        leaf["shape"] = [2**40]
    else:
        bad["tree"]["kind"] = ["forged"]
    with pytest.raises(WireError):
        validate_descriptor(bad)


def test_config_mismatch_and_unrecognized_containers(tiny_a):
    _, params, config = tiny_a
    host = jax.tree.map(lambda value: np.asarray(value, dtype=np.float32), params)
    descriptor, _ = describe_tree(host)
    descriptor["tree"]["children"][0]["shape"][0] += 1
    with pytest.raises(WireError, match="config"):
        validate_descriptor(descriptor, config)
    with pytest.raises(WireError):
        validate_descriptor({"protocol": PROTOCOL, "tree": {"kind": "ArbitraryClass", "children": []}})


def test_scalar_shape_and_wrong_received_dtype():
    descriptor, leaves = describe_tree((np.array(-0.0, np.float32),))
    _, records = validate_descriptor(descriptor)
    assert records[0]["shape"] == [] and records[0]["bytes"] == 4
    result = rebuild_tree(descriptor, leaves)
    assert np.signbit(result[0])
    with pytest.raises(WireError):
        rebuild_tree(descriptor, [np.array(0, np.float64)])
    with pytest.raises(WireError):
        rebuild_tree(descriptor, [])


def test_control_roundtrip_preserves_negative_zero_and_unicode():
    value = {"protocol": PROTOCOL, "version": 3, "说明": "仅表示主机已接收", "zero": -0.0}
    left, right = socket.socketpair()
    with left, right:
        send_control(left, value)
        assert canonical(receive_control(right)) == canonical(value)


@pytest.mark.parametrize("payload", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'{"x": 1}', b"[]", b"\xff"])
def test_noncanonical_or_invalid_json_is_rejected(payload):
    left, right = socket.socketpair()
    with left, right:
        left.sendall(struct.pack(">Q", len(payload)) + payload)
        with pytest.raises(WireError):
            receive_control(right)


def test_size_limit_and_truncated_transport():
    left, right = socket.socketpair()
    with left, right:
        left.sendall(struct.pack(">Q", MAX_CONTROL_BYTES + 1))
        with pytest.raises(WireError, match="超界"):
            receive_control(right)
    left, right = socket.socketpair()
    with left, right:
        left.sendall(struct.pack(">Q", 9) + b"{}")
        left.shutdown(socket.SHUT_WR)
        with pytest.raises(WireError, match="中断"):
            receive_control(right)


def test_received_full_host_tree_is_not_a_device_commit(tiny_a):
    from concurrent.futures import ThreadPoolExecutor

    from gemma4_posttrain_jax.inference_process import receive_parameters
    from gemma4_posttrain_jax.inference_wire import byte_sha

    _, params, config = tiny_a
    host = jax.tree.map(lambda value: np.asarray(value, dtype=np.float32, order="C"), params)
    descriptor, leaves = describe_tree(host)
    _, records = validate_descriptor(descriptor, config)
    left, right = socket.socketpair()
    with left, right, ThreadPoolExecutor(max_workers=1) as pool:
        left.settimeout(5)
        right.settimeout(5)
        future = pool.submit(receive_parameters, right, descriptor, config)
        ready = receive_control(left)
        assert ready == {"status": "ready", "leaf_count": len(records), "total_bytes": sum(x["bytes"] for x in records)}
        for leaf, record in zip(leaves, records, strict=True):
            raw = memoryview(leaf).cast("B")
            digest = byte_sha(raw)
            send_control(
                left,
                {
                    "kind": "fp32_leaf",
                    "index": record["index"],
                    "path": record["path"],
                    "bytes": record["bytes"],
                    "sha256": digest,
                },
            )
            left.sendall(raw)
            assert receive_control(left) == {"status": "host_received", "index": record["index"], "sha256": digest}
        actual, report = future.result(timeout=5)
    assert report["complete_host_receive"] is True and report["device_commit"] is False
    assert report["leaf_count"] == len(leaves)
    for x, y in zip(jax.tree.leaves(host), jax.tree.leaves(actual), strict=True):
        np.testing.assert_array_equal(x.view(np.uint32), y.view(np.uint32))


@pytest.mark.parametrize("failure", ["sha", "boolean_index"])
def test_corrupt_tensor_cannot_get_host_ack(tiny_a, failure):
    from concurrent.futures import ThreadPoolExecutor

    from gemma4_posttrain_jax.inference_process import receive_parameters

    _, params, config = tiny_a
    descriptor, leaves = describe_tree(jax.tree.map(lambda value: np.asarray(value, np.float32, order="C"), params))
    _, records = validate_descriptor(descriptor, config)
    record = records[0]
    left, right = socket.socketpair()
    with left, right, ThreadPoolExecutor(max_workers=1) as pool:
        left.settimeout(5)
        right.settimeout(5)
        future = pool.submit(receive_parameters, right, descriptor, config)
        assert receive_control(left)["status"] == "ready"
        packet = {
            "kind": "fp32_leaf",
            "index": 0 if failure == "sha" else False,
            "path": record["path"],
            "bytes": record["bytes"],
            "sha256": "0" * 64,
        }
        send_control(left, packet)
        if failure == "sha":
            left.sendall(memoryview(leaves[0]).cast("B"))
        with pytest.raises(WireError):
            future.result(timeout=5)
