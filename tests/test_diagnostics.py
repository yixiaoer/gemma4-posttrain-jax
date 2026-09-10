"""Tests for reusable rollout/trainer numerical diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from gemma4_posttrain_jax.diagnostics import array_drift_metrics, logprob_drift_metrics


def test_array_and_logprob_drift_only_use_selected_rows() -> None:
    expected = np.asarray([[1.0, 2.0], [100.0, 200.0], [3.0, 4.0]])
    actual = np.asarray([[1.0, 2.5], [-100.0, -200.0], [2.5, 4.0]])
    mask = np.asarray([True, False, True])

    array_metrics = array_drift_metrics(actual, expected, mask)
    assert array_metrics["rows"] == 2
    assert array_metrics["elements"] == 4
    assert array_metrics["max_abs_delta"] == 0.5

    expected_logps = np.asarray([[-2.0, -3.0], [-999.0, -999.0], [-4.0, -5.0]])
    actual_logps = expected_logps.copy()
    actual_logps[[0, 2]] += np.log(2.0)
    logprob_metrics = logprob_drift_metrics(
        actual_logps,
        expected_logps,
        mask,
        delta_definition="actual-expected",
    )
    assert logprob_metrics["count"] == 4
    assert logprob_metrics["delta_definition"] == "actual-expected"
    np.testing.assert_allclose(logprob_metrics["ratio_mean"], 2.0)
    np.testing.assert_allclose(logprob_metrics["ratio_min"], 2.0)
    np.testing.assert_allclose(logprob_metrics["ratio_max"], 2.0)
    assert logprob_metrics["ratio_outside_10pct_fraction"] == 1.0


def test_drift_metrics_reject_shape_and_empty_mask() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        array_drift_metrics(np.ones((2, 3)), np.ones((2, 4)), np.ones((2,), dtype=np.bool_))
    with pytest.raises(ValueError, match="not a prefix"):
        array_drift_metrics(np.ones((2, 3)), np.ones((2, 3)), np.ones((3,), dtype=np.bool_))
    with pytest.raises(ValueError, match="at least one"):
        logprob_drift_metrics(
            np.ones((2,)),
            np.ones((2,)),
            np.zeros((2,), dtype=np.bool_),
            delta_definition="actual-expected",
        )
