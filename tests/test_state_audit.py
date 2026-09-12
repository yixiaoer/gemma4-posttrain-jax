"""恢复检查必须覆盖全部模型参数和 Adam 状态，能够发现单个数组的变化与非有限值。"""

import numpy as np

from gemma4_posttrain_jax.state_audit import summarize_train_state


def test_complete_state_content_identity_and_nonfinite_detection():
    first = {"params": np.array([[1, 2]], np.float32), "adam": (np.array([0.5], np.float32), np.array(2, np.int32))}
    reference = summarize_train_state(first)
    assert reference["all_finite"] and reference["leaf_count"] == 3
    assert reference["logical_bytes"] == 16
    copied = {"params": first["params"].copy(), "adam": tuple(x.copy() for x in first["adam"])}
    assert summarize_train_state(copied)["content_sha256"] == reference["content_sha256"]
    copied["adam"][0][0] += 0.25
    assert summarize_train_state(copied)["content_sha256"] != reference["content_sha256"]
    copied["adam"][0][0] = np.nan
    assert not summarize_train_state(copied)["all_finite"]


def test_update_witness_uses_actual_trainable_values_and_survives_mutation():
    from gemma4_posttrain_jax.state_audit import compare_parameter_witness, snapshot_parameter_witness

    params = {"frozen": np.array([5.0], np.float32), "norm": np.array([1.0, 2.0], np.float32)}
    mask = {"frozen": False, "norm": True}
    first = snapshot_parameter_witness(params, mask)
    unchanged = compare_parameter_witness(first, snapshot_parameter_witness(params, mask))
    assert unchanged["all_finite"] and unchanged["changed_elements"] == 0
    params["norm"][1] += 0.25
    params["frozen"][0] += 4
    changed = compare_parameter_witness(first, snapshot_parameter_witness(params, mask))
    assert changed["leaf_count"] == 1 and changed["changed_elements"] == 1
    assert changed["leaves"][0]["max_abs_delta"] == 0.25
    params["norm"][0] = np.nan
    bad = compare_parameter_witness(first, snapshot_parameter_witness(params, mask))
    assert not bad["all_finite"] and bad["leaves"][0]["max_abs_delta"] is None


def test_update_witness_rejects_incompatible_or_absent_observations():
    import pytest

    from gemma4_posttrain_jax.state_audit import compare_parameter_witness, snapshot_parameter_witness

    params = {"x": np.ones(4, np.float32)}
    with pytest.raises(ValueError, match="没有"):
        snapshot_parameter_witness(params, {"x": False})
    with pytest.raises(ValueError, match="trainable"):
        snapshot_parameter_witness(params, {"y": True})
    with pytest.raises(ValueError, match="没有"):
        snapshot_parameter_witness(params, None, max_elements=2)
    with pytest.raises(ValueError, match="形状|类型"):
        compare_parameter_witness({"x": np.ones(2, np.float32)}, {"x": np.ones(2, np.float64)})
