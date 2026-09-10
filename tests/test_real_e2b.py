"""Real ``google/gemma-4-E2B-it`` checkpoint against Hugging Face on CPU in float32.

Run explicitly (multi-GB, several minutes):

    JAX_PLATFORMS=cpu .venv/bin/pytest -q -s -m full_model tests/test_real_e2b.py
"""

from __future__ import annotations

import glob
import os
import time

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from gemma4_posttrain_jax.model import forward_gemma4_lm, init_kv_cache
from gemma4_posttrain_jax.weights import check_gemma4_text_params, load_hf_eos_token_ids, load_hf_params

pytestmark = pytest.mark.full_model

SNAPSHOT_GLOB = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/*/")
PROMPT = "The three primary colors are"
MAX_NEW_TOKENS = 16


def _snapshot() -> str:
    paths = sorted(glob.glob(SNAPSHOT_GLOB))
    if not paths:
        pytest.skip("gemma-4-E2B-it is not in the Hugging Face cache")
    return paths[0]


def test_e2b_logits_and_greedy_match_hf() -> None:
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    snapshot = _snapshot()
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids  # adds <bos>
    S = input_ids.shape[1]

    t0 = time.time()
    hf = Gemma4ForConditionalGeneration.from_pretrained(snapshot, dtype=torch.float32).eval()
    print(f"\nHF load {time.time() - t0:.1f}s; prompt tokens {S}")
    t0 = time.time()
    expected = hf(input_ids=input_ids).logits[:, -1].numpy()
    hf_ids = hf.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)[0, S:].tolist()
    print(f"HF forward+greedy {time.time() - t0:.1f}s: {hf_ids} -> {tokenizer.decode(hf_ids)!r}")
    del hf

    t0 = time.time()
    params, config = load_hf_params(snapshot, dtype=jnp.float32)
    check_gemma4_text_params(params, config)
    eos_ids = load_hf_eos_token_ids(snapshot, config)
    print(f"JAX load {time.time() - t0:.1f}s")

    ids = jnp.asarray(input_ids.numpy())
    positions = jnp.arange(S)[None]
    t0 = time.time()
    out = forward_gemma4_lm(params, ids, positions, config=config, logits_to_keep=1)
    logits = np.asarray(out.logits[:, -1])
    print(f"JAX prefill {time.time() - t0:.1f}s")
    err = np.abs(logits - expected)
    print(f"last-position logits: max|err| {err.max():.3e}, max|hf| {np.abs(expected).max():.3f}")
    print(f"argmax ours/hf {logits.argmax()}/{expected.argmax()}")
    assert logits.argmax() == expected.argmax()
    np.testing.assert_allclose(logits, expected, atol=1e-3 * np.abs(expected).max())

    cache = init_kv_cache(config, 1, capacity=S + MAX_NEW_TOKENS, dtype=jnp.float32)
    out = forward_gemma4_lm(params, ids, positions, config=config, kv_cache=cache, logits_to_keep=1)
    ours: list[int] = []
    t0 = time.time()
    for step in range(MAX_NEW_TOKENS):
        token = int(out.logits[0, -1].argmax())
        ours.append(token)
        if token in eos_ids:
            break
        out = forward_gemma4_lm(
            params,
            jnp.asarray([[token]]),
            jnp.asarray([[S + step]]),
            config=config,
            kv_cache=out.kv_cache,
            cache_mode="append",
            logits_to_keep=1,
        )
    print(f"JAX greedy {time.time() - t0:.1f}s: {ours} -> {tokenizer.decode(ours)!r}")
    assert ours == hf_ids, (ours, hf_ids)
