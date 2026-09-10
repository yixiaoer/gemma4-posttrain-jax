"""归档前的CPU检查必须拒绝损坏状态和错位的Adam计数。"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_checkpoint import check_checkpoint  # noqa: E402


@pytest.mark.parametrize("corruption", [None, "nonfinite", "adam_count", "shape"])
def test_checkpoint_inspection_checks_serialized_state(tmp_path: Path, corruption: str | None) -> None:
    arrays = {
        ".step": np.asarray(3, np.int32),
        ".opt_state.adam.count": np.asarray(3, np.int32),
        ".params.weight": np.arange(15, dtype=np.float32).reshape(5, 3),
    }
    if corruption == "nonfinite":
        arrays[".params.weight"][4, 2] = np.nan
    if corruption == "adam_count":
        arrays[".opt_state.adam.count"] = np.asarray(2, np.int32)
    leaves = [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()]
    if corruption == "shape":
        leaves[-1]["shape"] = [3, 5]
    (tmp_path / "meta.json").write_text(
        json.dumps({"format": "gemma4-rl-jax-train-state", "version": 1, "leaves": leaves, "metadata": {}})
    )
    save_file(arrays, tmp_path / "state.safetensors")
    if corruption:
        with pytest.raises(ValueError):
            check_checkpoint(tmp_path, 3)
    else:
        result = check_checkpoint(tmp_path, 3)
        assert result["step"] == 3 and result["all_finite"]
        assert result["leaf_count"] == 3 and result["array_bytes"] == 68


@pytest.mark.parametrize("corruption", [None, "version", "dtype", "missing"])
def test_inspection_validates_lagged_behavior_snapshot(tmp_path: Path, corruption: str | None) -> None:
    import ml_dtypes

    arrays = {
        ".train.step": np.asarray(4, np.int32),
        ".train.opt_state.adam.count": np.asarray(4, np.int32),
        ".train.params_f32.weight": np.arange(12, dtype=np.float32).reshape(4, 3),
        ".behavior.step": np.asarray(3, np.int32),
        ".behavior.params_bf16.weight": np.arange(12, dtype=np.float32).astype(ml_dtypes.bfloat16).reshape(4, 3),
    }
    if corruption == "version":
        arrays[".behavior.step"] = np.asarray(4, np.int32)
    if corruption == "dtype":
        arrays[".behavior.params_bf16.weight"] = np.arange(12, dtype=np.float32).reshape(4, 3)
    if corruption == "missing":
        del arrays[".behavior.params_bf16.weight"]
    document = {
        "format": "gemma4-rl-jax-train-state",
        "version": 1,
        "metadata": {"run_config": {"rollout_lag_updates": 1}},
        "leaves": [
            {"name": key, "shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()
        ],
    }
    (tmp_path / "meta.json").write_text(json.dumps(document))
    save_file(arrays, tmp_path / "state.safetensors")
    if corruption is None:
        assert check_checkpoint(tmp_path, 4)["behavior_policy_step"] == 3
    else:
        with pytest.raises(ValueError):
            check_checkpoint(tmp_path, 4)


@pytest.mark.parametrize("zero_moment", [None, "mu", "nu", "both"])
def test_inspection_requires_both_actual_nonzero_adam_moments(tmp_path: Path, zero_moment: str | None) -> None:
    arrays = {
        ".step": np.asarray(4, np.int32),
        ".opt_state.inner.adam.count": np.asarray(4, np.int32),
        ".opt_state.inner.adam.mu.weight": np.asarray([0, 1, 0], np.float32),
        ".opt_state.inner.adam.nu.weight": np.asarray([0, 0, 2], np.float32),
        ".params_f32.weight": np.asarray([1, 2, 3], np.float32),
    }
    for key in ("mu", "nu"):
        if zero_moment in (key, "both"):
            arrays[f".opt_state.inner.adam.{key}.weight"][:] = 0
    document = {
        "format": "gemma4-rl-jax-train-state",
        "version": 1,
        "metadata": {},
        "leaves": [
            {"name": key, "shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()
        ],
    }
    (tmp_path / "meta.json").write_text(json.dumps(document))
    save_file(arrays, tmp_path / "state.safetensors")
    if zero_moment is None:
        result = check_checkpoint(tmp_path, 4, require_nonzero_adam=True)
        assert result["adam_nonzero_elements"] == {"mu": 1, "nu": 1}
    else:
        with pytest.raises(ValueError, match="非零Adam"):
            check_checkpoint(tmp_path, 4, require_nonzero_adam=True)
        assert check_checkpoint(tmp_path, 4)["all_finite"]
