"""Host-only tests for optional experiment tracking."""

from __future__ import annotations

import json
from typing import Any

import pytest

from gemma4_posttrain_jax.tracking import NullTracker, grpo_tracking_metrics, init_tracker, load_grpo_history


class FakeRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, int | float | None], int]] = []
        self.finished = False

    def log(self, metrics: dict[str, int | float | None], *, step: int) -> None:
        self.logged.append((metrics, step))

    def finish(self) -> None:
        self.finished = True


class FakeWandb:
    def __init__(self, run: FakeRun | None = None, error: Exception | None = None) -> None:
        self.run = run
        self.error = error
        self.init_kwargs: dict[str, Any] | None = None

    def init(self, **kwargs: Any) -> FakeRun | None:
        self.init_kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.run


def test_disabled_tracker_does_not_import_or_log() -> None:
    tracker = init_tracker(enabled=False, project="unused", run_name=None, tags=(), config={})
    assert isinstance(tracker, NullTracker)
    tracker.log({"loss": 1.0}, step=1)
    tracker.finish()


def test_wandb_adapter_forwards_config_metrics_and_finish() -> None:
    fake_run = FakeRun()
    fake_wandb = FakeWandb(fake_run)
    tracker = init_tracker(
        enabled=True,
        project="gemma4-rl-jax",
        run_name="offline-smoke",
        tags=("phase2.5", "tpu-v4"),
        config={"batch_size": 8},
        wandb_module=fake_wandb,
    )

    tracker.log({"loss": 0.5, "jit_cache_size": 1}, step=3)
    tracker.finish()

    assert fake_wandb.init_kwargs == {
        "project": "gemma4-rl-jax",
        "name": "offline-smoke",
        "tags": ["phase2.5", "tpu-v4"],
        "config": {"batch_size": 8},
    }
    assert fake_run.logged == [({"loss": 0.5, "jit_cache_size": 1}, 3)]
    assert fake_run.finished


def test_wandb_initialisation_failure_is_explicit() -> None:
    with pytest.raises(RuntimeError, match="W&B initialisation failed"):
        init_tracker(
            enabled=True,
            project="test",
            run_name=None,
            tags=(),
            config={},
            wandb_module=FakeWandb(error=ConnectionError("offline")),
        )


def test_grpo_metrics_separate_time_and_eval_and_reject_nan() -> None:
    metrics = grpo_tracking_metrics(
        {"step": 34, "loss": 0.25, "update_s": 1.5, "generated_tokens_per_s": 20},
        {"accuracy": 0.8, "dataset_indices": [3, 1], "peak_bytes": [1024]},
    )
    assert metrics == {
        "train_loss": 0.25,
        "time_update_s": 1.5,
        "train_generated_tokens_per_s": 20,
        "eval_accuracy": 0.8,
    }
    with pytest.raises(ValueError, match="非有限"):
        grpo_tracking_metrics({"loss": float("nan")})


def test_historical_training_merges_eval_before_logging_and_preserves_source(tmp_path) -> None:
    (tmp_path / "meta.json").write_text(json.dumps({"run_config": {"seed": 0}, "git_commit": "original"}))
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("step,loss,update_s,eval_s\n34,0.25,1.5,2.0\n35,0.125,1.0,2.5\n")
    original = csv_path.read_bytes()
    for step in (33, 34, 35):
        directory = tmp_path / "eval" / f"step_{step:08d}"
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(json.dumps({"accuracy": step / 100, "count": 500}))
    config, history = load_grpo_history(tmp_path)
    assert [step for step, _ in history] == [33, 34, 35]
    assert history[1][1] == {
        "train_loss": 0.25,
        "time_update_s": 1.5,
        "time_eval_s": 2.0,
        "eval_accuracy": 0.34,
        "eval_count": 500,
    }
    assert config["logging_mode"] == "historical_import"
    assert config["source_metadata"]["git_commit"] == "original"
    assert "metrics.csv" in config["source_sha256"]
    assert csv_path.read_bytes() == original
    csv_path.write_text("step,loss\n34,1\n34,2\n")
    with pytest.raises(ValueError, match="重复"):
        load_grpo_history(tmp_path)


def test_historical_standalone_eval_and_settings(tmp_path) -> None:
    (tmp_path / "meta.json").write_text(json.dumps({"step": 33}))
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation/summary.json").write_text("{}")
    (tmp_path / "summary.json").write_text(json.dumps({"step": 33, "correct": 443, "accuracy": 0.886}))
    _, history = load_grpo_history(tmp_path)
    assert history == [(33, {"eval_correct": 443, "eval_accuracy": 0.886})]
    fake = FakeWandb(FakeRun())
    tracker = init_tracker(
        enabled=True,
        project="test",
        run_name=None,
        tags=(),
        config={},
        wandb_module=fake,
        settings={"x_disable_stats": True},
    )
    tracker.finish()
    assert fake.init_kwargs["settings"] == {"x_disable_stats": True}
