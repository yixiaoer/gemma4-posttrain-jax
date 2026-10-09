"""Tools for long runs: resume comparison, run summary and the pre-training data check. All read small files only."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest

from gemma4_posttrain_jax.state_audit import summarize_train_state

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
RUN_CONFIG = {"prompt_batch_size": 1, "group_size": 2, "dev_size": 2, "dev_seed": 0, "train_eligible_size": 3}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def predictions(step: int) -> list[dict[str, Any]]:
    # Both questions are wrong at step 0; the first one is right afterwards.
    return [
        {"dataset_index": 7, "task_success": float(step > 0), "truncated": False, "length": 5},
        {"dataset_index": 9, "task_success": 0.0, "truncated": step == 0, "length": 8},
    ]


def write_run(root: Path, *, start: int, final: int, state: dict[str, Any], update_s: float = 1.0) -> Path:
    write_json(
        root / "meta.json",
        {"run_config": RUN_CONFIG, "git_commit": "c0ffee", "uncommitted_changes": False},
    )
    (root / "source.diff").write_text("")
    write_json(root / "data_split.json", {"training_indices": [1, 2, 3], "dev_indices": [7, 9]})
    write_json(
        root / "summary.json",
        {
            "start_step": start,
            "final_step": final,
            "final_data_cursor": final,
            "training_session_s": 100.0,
            "initial_eval_s": 4.0,
            "cleanup_s": 2.0,
        },
    )
    write_json(root / "final_state_audit.json", state)
    fields = {
        "loss": 0.5,
        "task_success": 0.5,
        "mixed_success_group_fraction": 1.0,
        "generated_rows": 2,
        "generated_tokens": 10,
        "truncated_fraction": 0.5,
        "rollout_s": 3.0,
        "update_s": update_s,
        "checkpoint_s": 0.0,
        "eval_s": 0.0,
        "unattributed_s": 0.5,
        "peak_hbm_gib": 1.0,
    }
    metric_rows = []
    timing_rows = []
    for step in range(start + 1, final + 1):
        row = {"step": step, "first_call": int(step == start + 1), **fields, "loss": 0.5 / step}
        if step % 2 == 0:
            row["eval_s"] = 2.0
        metric_rows.append(row)
        before_log = row["rollout_s"] + row["update_s"] + row["eval_s"] + row["unattributed_s"]
        timing_rows.append({"step": step, "metrics_logging_s": 0.1, "iteration_after_metrics_log_s": before_log + 0.1})
    with (root / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    write_rows(root / "iteration_timing.jsonl", timing_rows)
    write_rows(
        root / "rollouts.jsonl",
        [{"step": step, "row": row, "reward": 1.0} for step in range(start + 1, final + 1) for row in range(2)],
    )
    for step in range(start, final + 1, 2):
        rows = predictions(step)
        write_rows(root / "eval" / f"step_{step:08d}" / "predictions.jsonl", rows)
        correct = sum(row["task_success"] for row in rows)
        write_json(
            root / "eval" / f"step_{step:08d}" / "summary.json",
            {"count": 2, "accuracy": correct / 2, "mean_length": 6.5},
        )
    return root


@pytest.fixture
def tools(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import compare_resumed_run
    import summarize_grpo_run

    return compare_resumed_run, summarize_grpo_run


@pytest.fixture
def state() -> dict[str, Any]:
    return summarize_train_state({"weight": jnp.arange(4, dtype=jnp.float32), "adam": jnp.ones(2, jnp.float32)})


def test_resumed_run_matches_and_ignores_timing(tmp_path, tools, state):
    compare, _ = tools
    continuous = write_run(tmp_path / "continuous", start=0, final=4, state=state)
    resumed = write_run(tmp_path / "resume", start=2, final=4, state=state, update_s=9.0)
    result = compare.compare_runs(continuous, resumed)
    assert (result["resume_step"], result["final_step"]) == (2, 4)
    assert result["compared_metric_steps"] == 2 and result["compared_training_rows"] == 4
    assert [row["step"] for row in result["compared_evaluations"]] == [2, 4]


def test_resumed_run_rejects_changed_answer_or_state(tmp_path, tools, state):
    compare, _ = tools
    continuous = write_run(tmp_path / "continuous", start=0, final=4, state=state)
    resumed = write_run(tmp_path / "resume", start=2, final=4, state=state)
    rows = (resumed / "rollouts.jsonl").read_text().replace('"reward": 1.0', '"reward": 0.0', 1)
    (resumed / "rollouts.jsonl").write_text(rows)
    with pytest.raises(ValueError, match="training answers"):
        compare.compare_runs(continuous, resumed)

    other = summarize_train_state({"weight": jnp.arange(4, dtype=jnp.float32), "adam": jnp.zeros(2, jnp.float32)})
    resumed = write_run(tmp_path / "resume-other-state", start=2, final=4, state=other)
    with pytest.raises(ValueError, match="parameters or Adam"):
        compare.compare_runs(continuous, resumed)


def test_reload_eval_must_match_final_state_and_outputs(tmp_path, tools, state):
    compare, _ = tools
    continuous = write_run(tmp_path / "continuous", start=0, final=4, state=state)
    resumed = write_run(tmp_path / "resume", start=2, final=4, state=state)
    reload_dir = tmp_path / "reload"
    write_json(
        reload_dir / "meta.json",
        {"step": 4, "checkpoint_metadata": {"run_config": RUN_CONFIG}, "git_commit": "c0ffee"},
    )
    (reload_dir / "source.diff").write_text("")
    write_json(reload_dir / "loaded_state_audit.json", state)
    (reload_dir / "evaluation").mkdir()
    shutil.copy(continuous / "eval/step_00000004/predictions.jsonl", reload_dir / "evaluation/predictions.jsonl")
    assert compare.check_reload(reload_dir, continuous, resumed, 4)["correct"] == 1

    (reload_dir / "evaluation/predictions.jsonl").write_text(json.dumps(predictions(0)[0]) + "\n")
    with pytest.raises(ValueError, match="evaluation outputs"):
        compare.check_reload(reload_dir, continuous, resumed, 4)


def test_run_summary_curve_and_cost(tmp_path, tools, state):
    _, summarize = tools
    run = write_run(tmp_path / "continuous", start=0, final=4, state=state)
    result = summarize.summarize_run(run, process_wall_s=200.0)
    assert [point["correct"] for point in result["evaluation_curve"]] == [0, 1, 1]
    change = result["evaluation_curve"][-1]["change_from_first_eval"]
    assert (change["wrong_to_right"], change["right_to_wrong"], change["unchanged"]) == (1, 0, 1)
    assert result["evaluation_curve"][0]["truncated"] == 1
    assert result["training"]["truncated_training_rows"] == 4
    cost = result["cost"]
    assert cost["steps_with_save_or_eval"] == [2, 4]
    assert cost["stage_sum_residual_s"] == pytest.approx(0.0)
    # Steps 2-4 take 4.6 s each without evaluation, 3 s of which is rollout.
    assert cost["iterations"]["after_first_step_minus_save_and_eval"]["mean_s"] == pytest.approx(4.6)
    assert cost["after_first_step_shares_percent"]["rollout"] == pytest.approx(300 / 4.6)
    assert cost["process"]["outside_training_session_s"] == pytest.approx(100.0)


def test_data_check_finds_overlap_and_long_prompts(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import check_grpo_data

    class WordTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return list(range(len(messages[0]["content"].split())))

    data = [
        {"question": "one two", "answer": "work #### 3"},
        {"question": "three  four", "answer": "work #### 7"},
        {"question": "three four", "answer": "no delimiter"},
        {"question": "a much longer question with many words", "answer": "work #### 1"},
        {"question": "five six", "answer": "work #### 11"},
    ]
    passed = check_grpo_data.check_partitions(
        {"train": (data, [0]), "dev": (data, [1]), "test_monitor": (data, [3]), "test_unused": (data, [4])},
        WordTokenizer(),
        max_prompt_len=64,
        chat=True,
    )
    assert passed["passed"] and not any(passed["identical_question_overlaps"].values())
    # Every prompt appends a fixed 13-word instruction, so the long question has 20 words.
    result = check_grpo_data.check_partitions(
        {"train": (data, [0, 3]), "dev": (data, [1]), "test_monitor": (data, [2]), "test_unused": (data, [1])},
        WordTokenizer(),
        max_prompt_len=19,
        chat=True,
    )
    assert result["partitions"]["train"]["over_length_ids"] == [3]
    assert result["partitions"]["test_monitor"]["missing_gold_ids"] == [2]
    assert result["identical_question_overlaps"]["dev__test_unused"] == 1
    assert result["identical_question_overlaps"]["test_monitor__test_unused"] == 1
    assert not result["passed"]


def test_data_check_must_match_training_split(tmp_path, tools):
    compare, _ = tools
    report = {
        "passed": True,
        "dev_size": 2,
        "dev_seed": 0,
        "partitions": {"train": {"count": 3}, "dev": {"count": 2}},
    }
    write_json(tmp_path / "data_check.json", report)
    assert compare.check_data_report(tmp_path / "data_check.json", RUN_CONFIG) == {"train": 3, "dev": 2}
    write_json(tmp_path / "other_seed.json", {**report, "dev_seed": 1})
    with pytest.raises(ValueError, match="dev split"):
        compare.check_data_report(tmp_path / "other_seed.json", RUN_CONFIG)
    write_json(tmp_path / "failed.json", {**report, "passed": False})
    with pytest.raises(ValueError, match="data check failed"):
        compare.check_data_report(tmp_path / "failed.json", RUN_CONFIG)
