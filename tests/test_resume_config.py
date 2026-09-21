"""恢复检查只补齐不改变行为的旧默认值，其他差异必须拒绝。"""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "resume_config_trainer", Path(__file__).parents[1] / "scripts/train_grpo.py"
)
assert SPEC is not None and SPEC.loader is not None
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)


def test_legacy_beta_zero_is_compatible_without_mutating_saved_config():
    old = {"beta": 0.0, "seed": 4}
    trainer.validate_resume_config(old, {**old, "kl_estimator": "none", "kl_clamp_value": None})
    assert old == {"beta": 0.0, "seed": 4}


def test_legacy_nonzero_kl_requires_original_implementation():
    with pytest.raises(ValueError, match="不能按新 K3 实现精确续训"):
        trainer.validate_resume_config({"beta": 0.04}, {"beta": 0.04, "kl_estimator": "k3-stable-series-v2"})


@pytest.mark.parametrize(
    "field,value",
    [
        ("kl_estimator", "k3-stable-series-v1"),
        ("kl_clamp_value", 100.0),
        ("beta", 0.0),
        ("seed", 2),
        ("learning_rate", 1e-5),
    ],
)
def test_changed_behavior_is_rejected(field, value):
    current = {
        "beta": 0.04,
        "kl_estimator": "k3-stable-series-v2",
        "kl_clamp_value": None,
        "seed": 1,
        "learning_rate": 1e-6,
    }
    trainer.validate_resume_config(current.copy(), current)
    with pytest.raises(ValueError, match=field):
        trainer.validate_resume_config({**current, field: value}, current)


def test_missing_field_is_not_equal_to_explicit_none():
    current = {"beta": 0.0, "kl_estimator": "none", "kl_clamp_value": None, "important_setting": None}
    with pytest.raises(ValueError, match="important_setting"):
        trainer.validate_resume_config({"beta": 0.0}, current)


@pytest.mark.parametrize("saved", [None, [], "wrong"])
def test_invalid_metadata_is_rejected(saved):
    with pytest.raises(ValueError, match="训练配置"):
        trainer.validate_resume_config(saved, {})
