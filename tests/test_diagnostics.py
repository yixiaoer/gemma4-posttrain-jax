"""Tests for reusable rollout/trainer numerical diagnostics."""

from __future__ import annotations

import importlib.metadata
import subprocess

import numpy as np
import pytest

from gemma4_posttrain_jax.diagnostics import (
    array_drift_metrics,
    logprob_drift_metrics,
    optional_package_version,
    source_git_state,
)


def test_source_identity_uses_source_root_and_rejects_unrelated_parent_repo(tmp_path, monkeypatch):
    repo = tmp_path / "source"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    source = repo / "model.py"
    source.write_text("value = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "model.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    monkeypatch.chdir(tmp_path)
    revision, diff = source_git_state(repo)
    assert revision == subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert diff == ""
    source.write_text("value = 2\n")
    assert "+value = 2" in source_git_state(repo)[1]
    nested = repo / "independent-snapshot"
    nested.mkdir()
    assert source_git_state(nested) == (None, None)
    assert source_git_state(tmp_path) == (None, None)


def test_missing_optional_runtime_is_recorded_as_null(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert optional_package_version("libtpu") is None


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
