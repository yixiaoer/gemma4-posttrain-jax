"""生成入口的参数、批量输入和真实小模型生成测试。"""

from __future__ import annotations

import ast
import re

import numpy as np
import pytest
import transformers

from gemma4_posttrain_jax import weights
from scripts import generate as cli


class Tokenizer:
    pad_token_id = 0

    def __call__(self, prompt):
        return {"input_ids": [2, 7] if prompt == "short" else [2, 8, 9]}

    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": [2, 11, *self(messages[0]["content"])["input_ids"], 12]}

    def decode(self, tokens):
        return " ".join(map(str, tokens))


@pytest.mark.parametrize("chat", [False, True])
def test_batch_padding_preserves_each_prompt(chat):
    tokenizer = Tokenizer()
    ids, mask = cli.encode_prompts(tokenizer, ["short", "long"], chat=chat)
    expected = [[2, 7], [2, 8, 9]]
    if chat:
        expected = [[2, 11, *row, 12] for row in expected]
    for row in range(2):
        assert ids[row, mask[row]].tolist() == expected[row]
    assert not mask[0, 0] and ids[0, 0] == 0
    assert np.all(mask[:, 1:])


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--top-p", "0"),
        ("--top-p", "1.1"),
        ("--top-p", "nan"),
        ("--temperature", "nan"),
        ("--top-k", "-2"),
        ("--max-new-tokens", "0"),
    ],
)
def test_invalid_sampling_flags_fail_before_loading_model(flag, value):
    with pytest.raises(SystemExit) as error:
        cli.parse_args([flag, value])
    assert error.value.code == 2


def test_transformers_full_vocab_sampling_does_not_inherit_default_top_k():
    args = cli.parse_args(["--temperature", "0.7"])
    config = cli.hf_generation_config(args, (1, 4), 0)
    assert config.do_sample and config.temperature == 0.7
    assert config.top_k == 0 and config.top_p == 1
    assert config.eos_token_id == [1, 4]


@pytest.mark.parametrize("temperature,top_p", [(0.0, 1.0), (0.8, 1e-8)])
def test_cli_batch_generation_matches_transformers_with_one_candidate(tiny_a, monkeypatch, capsys, temperature, top_p):
    """不下载权重；两边实际执行同一小模型，greedy和极小Top-p均只有一个候选。"""
    hf, params, config = tiny_a
    tokenizer = Tokenizer()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(transformers.Gemma4ForConditionalGeneration, "from_pretrained", lambda *a, **kw: hf)
    monkeypatch.setattr(weights, "load_hf_params", lambda *a, **kw: (params, config))
    monkeypatch.setattr(weights, "load_hf_eos_token_ids", lambda *a, **kw: ())
    cli.main(
        [
            "--model",
            "/unused/tiny",
            "--prompt",
            "short",
            "--prompt",
            "long",
            "--dtype",
            "float32",
            "--temperature",
            str(temperature),
            "--top-p",
            str(top_p),
            "--max-new-tokens",
            "4",
            "--compare-hf",
        ]
    )
    output = capsys.readouterr().out
    jax_rows = [ast.literal_eval(x) for x in re.findall(r"JAX回答\d+ token：(\[[^\n]*\])", output)]
    hf_rows = [ast.literal_eval(x) for x in re.findall(r"Transformers回答\d+ token：(\[[^\n]*\])", output)]
    assert len(jax_rows) == len(hf_rows) == 2
    assert all(len(row) == 4 for row in jax_rows)
    assert jax_rows == hf_rows
    if temperature == 0:
        assert output.count("greedy token完全一致：True") == 2
    else:
        assert "相同seed不保证相同回答" in output
