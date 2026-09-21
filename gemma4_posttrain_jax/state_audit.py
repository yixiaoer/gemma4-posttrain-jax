"""记录各个状态数组的校验值，用于比较连续训练和恢复后的结果。"""

from __future__ import annotations

import hashlib
import json
from contextlib import closing
from typing import Any, cast


def summarize_train_state(state: Any) -> dict[str, Any]:
    """分块计算全部数组的内容摘要，避免大参数读回和源状态缓存累积。"""
    import jax
    import numpy as np

    from gemma4_posttrain_jax import checkpoint

    records = []
    paths, treedef = jax.tree_util.tree_flatten_with_path(state)
    for path, leaf in paths:
        signature = leaf if isinstance(leaf, jax.Array) else np.asarray(leaf)
        finite, nonzero, byte_count = True, 0, 0
        digest = hashlib.sha256()
        with closing(checkpoint.iter_host_array_blocks(leaf, max_bytes=checkpoint._CHECKPOINT_CHUNK_BYTES)) as blocks:
            for value in blocks:
                flat = value.reshape(-1)
                for start in range(0, flat.size, 2**20):
                    part = flat[start : start + 2**20]
                    finite = bool(np.isfinite(part).all()) and finite
                    nonzero += int(np.count_nonzero(part))
                    del part
                digest.update(memoryview(flat.view(np.uint8)))
                byte_count += value.nbytes
                del flat, value
        if byte_count != signature.size * signature.dtype.itemsize:
            raise ValueError("完整状态检查的读回字节数与数组大小不同")
        records.append(
            {
                "name": jax.tree_util.keystr(path),
                "shape": list(signature.shape),
                "dtype": str(signature.dtype),
                "bytes": byte_count,
                "sha256": digest.hexdigest(),
                "finite": finite,
                "nonzero_elements": nonzero,
            }
        )
    names = [row["name"] for row in records]
    if len(set(names)) != len(names):
        raise ValueError("训练状态中存在重名数组")
    identity = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "complete": True,
        "all_finite": all(row["finite"] for row in records),
        "leaf_count": len(records),
        "logical_bytes": sum(row["bytes"] for row in records),
        "content_sha256": hashlib.sha256(identity).hexdigest(),
        "treedef": str(treedef),
        "leaves": records,
    }


def snapshot_parameter_witness(
    params: Any, trainable: Any, *, max_elements: int = 8192, max_leaves: int = 8
) -> dict[str, Any]:
    """复制少量可训练参数到主机，用于比较更新前后的数值变化。"""
    import jax
    import numpy as np

    if max_elements <= 0 or max_leaves <= 0:
        raise ValueError("待检查的元素数和数组数量上限必须为正数")
    paths, structure = jax.tree_util.tree_flatten_with_path(params)
    if trainable is None:
        enabled = [True] * len(paths)
    else:
        enabled, mask_structure = jax.tree.flatten(trainable)
        if cast(Any, mask_structure) != structure or any(not isinstance(flag, bool) for flag in enabled):
            raise ValueError("trainable 必须与参数结构一致，每个数组对应一个布尔值")
    result = {}
    for (path, leaf), active in zip(paths, enabled, strict=True):
        if not active or not 0 < leaf.size <= max_elements or not np.issubdtype(leaf.dtype, np.floating):
            continue
        # 独立host副本不保留即将donate的设备buffer。
        result[jax.tree_util.keystr(path)] = np.array(jax.device_get(leaf), copy=True, order="C")
        if len(result) == max_leaves:
            break
    if not result:
        raise ValueError("没有符合大小限制的可训练浮点数组，无法检查参数变化")
    return result


def compare_parameter_witness(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """比较对应数组中的各个元素，记录数值变化和非有限值。"""
    import numpy as np

    if not before or before.keys() != after.keys():
        raise ValueError("更新前后的参数记录不能为空，且数组名称必须一致")
    rows = []
    for name, first in before.items():
        second = after[name]
        if first.shape != second.shape or first.dtype != second.dtype:
            raise ValueError("更新前后数组的形状或类型发生了变化")
        finite = bool(np.isfinite(first).all() and np.isfinite(second).all())
        rows.append(
            {
                "name": name,
                "shape": list(first.shape),
                "dtype": str(first.dtype),
                "before_sha256": hashlib.sha256(memoryview(first).cast("B")).hexdigest(),
                "after_sha256": hashlib.sha256(memoryview(second).cast("B")).hexdigest(),
                "changed_elements": int(np.count_nonzero(first != second)),
                "finite": finite,
                "max_abs_delta": float(np.max(np.abs(second.astype(np.float64) - first))) if finite else None,
            }
        )
    return {
        "complete": True,
        "all_finite": all(row["finite"] for row in rows),
        "changed_elements": sum(row["changed_elements"] for row in rows),
        "leaf_count": len(rows),
        "leaves": rows,
        "scope": "此处只检查选中的小型参数数组；全部状态的数值和恢复结果另行检查。",
    }
