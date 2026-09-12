"""本机独立进程的有限JSON控制帧、完整 FP32 参数结构描述与原始字节运输。

协议不反序列化pickle；只接收白名单参数结构，缓冲寿命以接收/提交ACK区分。
"""

from __future__ import annotations

import hashlib
import json
import math
import socket
import struct
from dataclasses import dataclass
from typing import Any

PROTOCOL = "gemma4-host-fp32-v1"
MAX_CONTROL_BYTES = 8 * 1024**2
MAX_LEAVES = 1024
MAX_TREE_BYTES = 24 * 1024**3
MAX_LEAF_BYTES = 12 * 1024**3
CHUNK_BYTES = 8 * 1024**2


class WireError(RuntimeError):
    """包含边界或传输失败，调用方不得继续复用半个请求。"""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def receive_exact(connection: socket.socket, length: int) -> bytes:
    if not 0 <= length <= MAX_CONTROL_BYTES:
        raise WireError("控制帧长度超界")
    parts = bytearray(length)
    receive_into(connection, memoryview(parts))
    return bytes(parts)


def receive_into(connection: socket.socket, destination: memoryview) -> None:
    offset = 0
    while offset < len(destination):
        count = connection.recv_into(destination[offset : offset + CHUNK_BYTES])
        if count == 0:
            raise WireError(f"传输中断：仅收到{offset}/{len(destination)}字节")
        offset += count


def send_control(connection: socket.socket, value: Any) -> None:
    raw = canonical(value)
    if len(raw) > MAX_CONTROL_BYTES:
        raise WireError("控制帧过大")
    connection.sendall(struct.pack(">Q", len(raw)))
    connection.sendall(raw)


def receive_control(connection: socket.socket) -> dict[str, Any]:
    length = struct.unpack(">Q", receive_exact(connection, 8))[0]
    raw = receive_exact(connection, length)

    def reject_constant(value: str) -> Any:
        raise WireError(f"JSON非有限常量: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise WireError(f"JSON重复字段: {key}")
            result[key] = value
        return result

    try:
        result = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)
        encoded = canonical(result)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise WireError("非法JSON控制帧") from error
    if not isinstance(result, dict) or encoded != raw:
        raise WireError("控制帧必须为规范JSON对象")
    return result


def parameter_types() -> dict[str, Any]:
    from .model import (
        AttentionParams,
        DecoderLayerParams,
        Gemma4TextParams,
        MLPParams,
        PerLayerInputParams,
        RMSNormParams,
    )

    return {
        kind.__name__: kind
        for kind in (
            AttentionParams,
            DecoderLayerParams,
            Gemma4TextParams,
            MLPParams,
            PerLayerInputParams,
            RMSNormParams,
        )
    }


def describe_tree(params: Any) -> tuple[dict[str, Any], list[Any]]:
    import numpy as np

    types = parameter_types()
    leaves: list[Any] = []

    def visit(value: Any, path: str, depth: int) -> dict[str, Any]:
        if depth > 64:
            raise WireError("参数结构的嵌套层数超出限制")
        if value is None:
            return {"kind": "none"}
        if isinstance(value, tuple):
            name = type(value).__name__
            if type(value) is tuple:
                names = [str(i) for i in range(len(value))]
                kind = "tuple"
            elif name in types and type(value) is types[name]:
                names = list(types[name]._fields)
                kind = name
            else:
                raise WireError("参数容器的类型不受支持")
            return {
                "kind": kind,
                "children": [
                    visit(child, f"{path}.{n}" if path else n, depth + 1) for n, child in zip(names, value, strict=True)
                ],
            }
        if np.dtype(value.dtype) != np.dtype("float32"):
            raise WireError("跨进程只运输完整FP32 master")
        index = len(leaves)
        leaves.append(value)
        return {"kind": "leaf", "index": index, "path": path, "shape": list(value.shape), "dtype": "<f4"}

    descriptor = {"protocol": PROTOCOL, "tree": visit(params, "", 0)}
    validate_descriptor(descriptor)
    return descriptor, leaves


@dataclass(frozen=True)
class ShapeOnly:
    shape: tuple[int, ...]


def validate_descriptor(descriptor: dict[str, Any], config: Any = None) -> tuple[Any, list[dict[str, Any]]]:
    types = parameter_types()
    records: list[dict[str, Any]] = []
    total = 0
    if set(descriptor) != {"protocol", "tree"} or descriptor["protocol"] != PROTOCOL:
        raise WireError("参数传输协议版本不符")

    def visit(node: Any, path: str, depth: int) -> Any:
        nonlocal total
        if not isinstance(node, dict) or depth > 64:
            raise WireError("参数结构不符合要求")
        kind = node.get("kind")
        if not isinstance(kind, str):
            raise WireError("参数类型名必须为字符串")
        if kind == "none" and set(node) == {"kind"}:
            return None
        if kind == "leaf":
            if set(node) != {"kind", "index", "path", "shape", "dtype"}:
                raise WireError("数组元数据的字段不符")
            shape = node["shape"]
            if not isinstance(shape, list) or len(shape) > 4 or any(type(x) is not int or x <= 0 for x in shape):
                raise WireError("参数数组的形状不符合要求")
            if (
                type(node["index"]) is not int
                or node["index"] != len(records)
                or node["path"] != path
                or node["dtype"] != "<f4"
            ):
                raise WireError("数组的顺序、名称或类型不符")
            size = math.prod(shape) * 4
            # 不从发送端的任意nbytes决定分配，按已经校验的shape计算。
            total += size
            if size > MAX_LEAF_BYTES or total > MAX_TREE_BYTES or len(records) >= MAX_LEAVES:
                raise WireError("参数总字节数或数组数量超出限制")
            records.append({**node, "bytes": size})
            return ShapeOnly(tuple(shape))
        if set(node) != {"kind", "children"} or not isinstance(node["children"], list):
            raise WireError("参数容器字段不符")
        children = node["children"]
        if kind == "tuple":
            if len(children) > MAX_LEAVES:
                raise WireError("tuple长度超界")
            names = [str(i) for i in range(len(children))]
        elif kind in types:
            names = list(types[kind]._fields)
            if len(names) != len(children):
                raise WireError("NamedTuple字段数量不符")
        else:
            raise WireError("未知参数容器")
        values = [
            visit(child, f"{path}.{n}" if path else n, depth + 1) for n, child in zip(names, children, strict=True)
        ]
        return tuple(values) if kind == "tuple" else types[kind](*values)

    tree = visit(descriptor["tree"], "", 0)
    if config is not None:
        from .model import Gemma4TextParams
        from .weights import check_gemma4_text_params

        if type(tree) is not Gemma4TextParams or not __debug__:
            raise WireError("需要完整的 Gemma4 模型参数，且不能关闭形状断言")
        try:
            check_gemma4_text_params(tree, config)
        except (AssertionError, AttributeError) as error:
            raise WireError("参数形状与固定模型config不同") from error
    return tree, records


def rebuild_tree(descriptor: dict[str, Any], leaves: list[Any]) -> Any:
    types = parameter_types()
    _, records = validate_descriptor(descriptor)
    if len(records) != len(leaves):
        raise WireError("接收的参数数组数量不同")

    def visit(node: dict[str, Any]) -> Any:
        kind = node["kind"]
        if kind == "none":
            return None
        if kind == "leaf":
            value = leaves[node["index"]]
            if list(value.shape) != node["shape"] or str(value.dtype) != "float32":
                raise WireError("实际接收数组shape/dtype不同")
            return value
        values = [visit(child) for child in node["children"]]
        return tuple(values) if kind == "tuple" else types[kind](*values)

    return visit(descriptor["tree"])


def byte_sha(value: memoryview) -> str:
    return hashlib.sha256(value).hexdigest()
