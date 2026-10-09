"""Evaluation records the state it loaded; it must equal the saved state, and non-finite values stop it."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import pytest

from gemma4_posttrain_jax.checkpoint import load_train_state, save_train_state
from gemma4_posttrain_jax.losses import TrainState, init_train_state
from gemma4_posttrain_jax.state_audit import summarize_train_state


@pytest.mark.parametrize("corrupt", [False, True])
def test_evaluation_records_complete_loaded_state(tmp_path, monkeypatch, corrupt):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_gsm8k import record_loaded_state

    params = {"weight": jnp.asarray([1.0, -2.0, 0.0], dtype=jnp.float32)}
    optimizer = optax.adam(1e-3)
    state = init_train_state(params, optimizer)
    updates, moments = optimizer.update({"weight": jnp.asarray([0.3, -0.1, 0.5])}, state.opt_state, params)
    state = TrainState(optax.apply_updates(params, updates), moments, jnp.asarray(1, jnp.int32))
    if corrupt:
        state = state._replace(params_f32={"weight": state.params_f32["weight"].at[1].set(jnp.nan)})
    expected = summarize_train_state(state)
    checkpoint = save_train_state(state, tmp_path / "checkpoint", metadata={"source": "loaded-state-test"})
    restored = load_train_state(checkpoint, jax.eval_shape(lambda: state))
    output = tmp_path / "loaded.json"
    if corrupt:
        with pytest.raises(ValueError, match="non-finite"):
            record_loaded_state(restored, output)
    else:
        record_loaded_state(restored, output)
    assert json.loads(output.read_text()) == expected
    assert int(restored.step) == 1
