"""Host-only tests for optional experiment tracking."""

from __future__ import annotations

from typing import Any

import pytest

from gemma4_posttrain_jax.tracking import NullTracker, grpo_tracking_metrics, init_tracker


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
        project="gemma4_posttrain_jax",
        run_name="offline-smoke",
        tags=("phase2.5", "tpu-v4"),
        config={"batch_size": 8},
        wandb_module=fake_wandb,
    )

    tracker.log({"loss": 0.5, "jit_cache_size": 1}, step=3)
    tracker.finish()

    assert fake_wandb.init_kwargs == {
        "project": "gemma4_posttrain_jax",
        "name": "offline-smoke",
        "tags": ["phase2.5", "tpu-v4"],
        "config": {"batch_size": 8},
    }
    assert fake_run.logged == [({"loss": 0.5, "jit_cache_size": 1}, 3)]
    assert fake_run.finished


def test_wandb_group_and_job_type_keep_stages_together() -> None:
    fake_wandb = FakeWandb(FakeRun())
    for job_type in ("train", "resume", "evaluate"):
        init_tracker(
            enabled=True,
            project="gemma4_posttrain_jax",
            run_name=f"seed0-{job_type}",
            tags=(),
            config={},
            wandb_module=fake_wandb,
            group="grpo-gsm8k-seed0",
            job_type=job_type,
        )
        assert fake_wandb.init_kwargs is not None
        assert fake_wandb.init_kwargs["group"] == "grpo-gsm8k-seed0"
        assert fake_wandb.init_kwargs["job_type"] == job_type


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
    with pytest.raises(ValueError, match="non-finite"):
        grpo_tracking_metrics({"loss": float("nan")})


def test_wandb_settings_are_forwarded() -> None:
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
    assert fake.init_kwargs is not None
    assert fake.init_kwargs["settings"] == {"x_disable_stats": True}
