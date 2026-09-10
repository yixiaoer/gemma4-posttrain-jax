"""Portable, validated checkpoints for the complete SFT train state."""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO, cast

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file, save

FORMAT_NAME = "gemma4-rl-jax-train-state"
FORMAT_VERSION = 1


def _flatten_named(tree: Any) -> tuple[list[str], list[Any], Any]:
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    names = [jax.tree_util.keystr(path) for path, _ in path_leaves]
    if len(names) != len(set(names)):
        raise ValueError("train-state pytree produced duplicate checkpoint keys")
    return names, [leaf for _, leaf in path_leaves], treedef


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"checkpoint metadata value is not JSON serializable: {type(value).__name__}")


@lru_cache
def _safetensors_dtype(dtype: np.dtype) -> str:
    """让已安装的safetensors验证dtype，并从空数组取得格式中的dtype名称。"""

    encoded = save({"array": np.empty((0,), dtype=dtype)})
    header_size = int.from_bytes(encoded[:8], "little")
    return cast(str, json.loads(encoded[8 : 8 + header_size])["array"]["dtype"])


def _write_leaf(file: BinaryIO, leaf: Any) -> None:
    """只保留当前张量的主机视图，不给活跃训练状态积累device_get缓存。"""

    temporary = None
    try:
        if isinstance(leaf, jax.Array):
            if not leaf.is_fully_addressable:
                raise ValueError("checkpoint requires fully addressable arrays on a single host")
            # may_alias=False保证独立副本；释放它不会删除原训练状态。
            # 代价是当前张量的设备副本，避免为整个状态保留主机缓存。
            temporary = jax.device_put(leaf, leaf.sharding, may_alias=False)
            leaf = temporary
        array = np.asarray(jax.device_get(leaf), order="C")
        if not array.dtype.isnative:
            array = array.astype(array.dtype.newbyteorder("="))
        # safetensors要求小端、行优先；uint8视图同时兼容BF16和零维标量。
        if not np.little_endian:
            array = array.byteswap()
        data = memoryview(array.reshape(-1).view(np.uint8))
        if file.write(data) != len(data):
            raise OSError("incomplete checkpoint tensor write")
    finally:
        if temporary is not None:
            temporary.delete()


def _save_arrays(names: list[str], leaves: list[Any], path: Path) -> list[dict[str, Any]]:
    """逐张量写标准safetensors；头部仅依赖shape/dtype，不收集全量主机数组。"""

    header: dict[str, Any] = {"__metadata__": {"format": FORMAT_NAME}}
    offset = 0
    ordered = sorted(zip(names, leaves, strict=True), key=lambda item: (-np.dtype(item[1].dtype).itemsize, item[0]))
    for name, leaf in ordered:
        dtype = np.dtype(leaf.dtype)
        size = math.prod(leaf.shape) * dtype.itemsize
        header[name] = {
            "dtype": _safetensors_dtype(dtype),
            "shape": list(leaf.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as file:
        file.write(len(encoded).to_bytes(8, "little"))
        file.write(encoded)
        for _, leaf in ordered:
            _write_leaf(file, leaf)
        if file.tell() != 8 + len(encoded) + offset:
            raise OSError("checkpoint file size differs from tensor header")
    return [
        {
            "name": name,
            "shape": list(leaf.shape),
            "dtype": str(np.dtype(leaf.dtype).newbyteorder("=")),
            "sharding": None if getattr(leaf, "sharding", None) is None else str(leaf.sharding),
        }
        for name, leaf in zip(names, leaves, strict=True)
    ]


def save_train_state(state: Any, path: str | os.PathLike[str], *, metadata: Mapping[str, Any]) -> Path:
    """逐张量保存完整训练状态，最后原子地发布checkpoint目录。

    不覆盖已有目录，保持版本1的文件格式。主机传输缓冲随最大张量增长，
    每次需要该张量的独立设备副本；tmpfs中的输出文件仍占用主机物理内存。
    """

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"checkpoint already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        names, leaves, _ = _flatten_named(state)
        user_metadata = _jsonable(metadata)
        leaf_metadata = _save_arrays(names, leaves, temporary / "state.safetensors")
        document = {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "metadata": user_metadata,
            "leaves": leaf_metadata,
        }
        with open(temporary / "meta.json", "w", encoding="utf-8") as file:
            json.dump(document, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        os.rename(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def read_checkpoint_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate the non-array checkpoint manifest."""

    checkpoint = Path(path)
    with open(checkpoint / "meta.json", encoding="utf-8") as file:
        document = json.load(file)
    if document.get("format") != FORMAT_NAME or document.get("version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format/version: {document.get('format')!r}/{document.get('version')!r}"
        )
    if not isinstance(document.get("metadata"), dict) or not isinstance(document.get("leaves"), list):
        raise ValueError("checkpoint manifest is missing metadata or leaves")
    return cast(dict[str, Any], document)


def load_train_state[T](
    path: str | os.PathLike[str],
    template: T,
    *,
    shardings: Any | None = None,
) -> T:
    """Restore a train state after exact key/shape/dtype validation.

    ``template`` provides the Python pytree node types and expected array
    signatures. ``shardings`` may mirror it to place restored global arrays
    directly onto a device mesh.
    """

    checkpoint = Path(path)
    document = read_checkpoint_metadata(checkpoint)
    names, template_leaves, treedef = _flatten_named(template)
    stored = load_file(checkpoint / "state.safetensors")
    if set(stored) != set(names):
        missing = sorted(set(names) - set(stored))
        extra = sorted(set(stored) - set(names))
        raise ValueError(f"checkpoint pytree keys differ: missing={missing} extra={extra}")

    manifest = {entry["name"]: entry for entry in document["leaves"]}
    if set(manifest) != set(names):
        raise ValueError("checkpoint manifest leaf keys differ from the template")
    for name, template_leaf in zip(names, template_leaves, strict=True):
        array = stored[name]
        expected_shape = tuple(template_leaf.shape)
        expected_dtype = np.dtype(template_leaf.dtype)
        if array.shape != expected_shape or np.dtype(array.dtype) != expected_dtype:
            raise ValueError(
                f"checkpoint leaf {name} has {array.shape}/{array.dtype}, expected {expected_shape}/{expected_dtype}"
            )
        entry = manifest[name]
        if tuple(entry["shape"]) != array.shape or np.dtype(entry["dtype"]) != np.dtype(array.dtype):
            raise ValueError(f"checkpoint manifest disagrees with state.safetensors for {name}")

    if shardings is None:
        restored_leaves = [jnp.asarray(stored[name]) for name in names]
    else:
        sharding_leaves, sharding_treedef = jax.tree.flatten(shardings)
        if sharding_treedef != treedef:
            raise ValueError("checkpoint sharding tree does not match the train-state template")
        restored_leaves = [
            jax.device_put(stored[name], sharding) for name, sharding in zip(names, sharding_leaves, strict=True)
        ]
    return cast(T, jax.tree.unflatten(treedef, restored_leaves))
