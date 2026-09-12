"""使用纯JAX生成回答，支持greedy、temperature、TopK和Top-p采样。

可重复传入--prompt，在一个设备上batch生成。--compare-hf会同时用Hugging Face
Transformers在CPU上以FP32加载同一模型，并按相同采样设置生成回答供比较。
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import time
from collections.abc import Sequence
from itertools import zip_longest
from typing import Any

DEFAULT_SNAPSHOT_GLOB = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/*/")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="已下载的Hugging Face模型目录，默认查找本地E2B缓存")
    parser.add_argument("--prompt", action="append", help="输入文本；重复传入可在同一个batch中生成多个回答")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="每条回答最多生成的token数，默认32")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="JAX模型计算精度")
    parser.add_argument("--temperature", type=float, default=0.0, help="0为greedy；大于0为随机采样，默认0")
    parser.add_argument("--top-k", type=int, default=0, help="随机采样的候选数；0使用完整词表，默认0")
    parser.add_argument("--top-p", type=float, default=1.0, help="按累计概率保留采样候选，范围(0,1]，默认1关闭")
    parser.add_argument("--seed", type=int, default=0, help="随机种子，默认0")
    parser.add_argument(
        "--compare-hf",
        action="store_true",
        help="同时用Hugging Face Transformers在CPU上以FP32加载同一模型，按相同采样设置生成并显示输出",
    )
    parser.add_argument("--chat", action="store_true", help="使用模型tokenizer的聊天模板包装每条输入")
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens必须大于0")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature必须是大于等于0的有限数值")
    if args.top_k < 0:
        parser.error("--top-k必须大于等于0")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        parser.error("--top-p必须是(0,1]内的有限数值")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed必须在0到2**32-1之间")
    if args.prompt is None:
        args.prompt = ["The three primary colors are"]
    return args


def resolve_snapshot(path: str | None) -> str:
    if path:
        return os.path.expanduser(path)
    paths = sorted(glob.glob(DEFAULT_SNAPSHOT_GLOB))
    if not paths:
        raise FileNotFoundError("未找到E2B模型缓存，请先下载模型并通过--model指定本地目录")
    return paths[0]


def encode_prompts(tokenizer: Any, prompts: Sequence[str], *, chat: bool) -> tuple[Any, Any]:
    """将不同长度的输入左填充到同一宽度，显式记录有效token。"""
    import numpy as np

    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer需要提供pad_token_id以构造batch输入")
    rows = []
    for prompt in prompts:
        if chat:
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
            )
        else:
            encoded = tokenizer(prompt)
        tokens = np.asarray(encoded["input_ids"], dtype=np.int32).reshape(-1)
        if not tokens.size:
            raise ValueError("输入经过tokenizer处理后没有token")
        rows.append(tokens)
    width = max(len(row) for row in rows)
    ids = np.full((len(rows), width), tokenizer.pad_token_id, dtype=np.int32)
    mask = np.zeros_like(ids, dtype=np.bool_)
    for index, tokens in enumerate(rows):
        ids[index, -len(tokens) :] = tokens
        mask[index, -len(tokens) :] = True
    return ids, mask


def hf_generation_config(args: argparse.Namespace, eos_ids: tuple[int, ...], pad_token_id: int) -> Any:
    """显式构造生成设置，避免模型默认的TopK或Top-p改变对照的采样分布。"""
    from transformers import GenerationConfig

    sampling = args.temperature > 0
    options = {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p} if sampling else {}
    return GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=sampling,
        num_beams=1,
        eos_token_id=list(eos_ids) if eos_ids else None,
        pad_token_id=pad_token_id,
        **options,
    )


def compare_with_hf(
    args: argparse.Namespace,
    snapshot: str,
    tokenizer: Any,
    ids: Any,
    mask: Any,
    eos_ids: tuple[int, ...],
    generated: list[list[int]],
) -> None:
    import torch
    from transformers import Gemma4ForConditionalGeneration

    start = time.perf_counter()
    hf = Gemma4ForConditionalGeneration.from_pretrained(snapshot, dtype=torch.float32).cpu().eval()
    torch.manual_seed(args.seed)
    with torch.no_grad():
        output = hf.generate(
            torch.as_tensor(ids, dtype=torch.long),
            attention_mask=torch.as_tensor(mask, dtype=torch.long),
            generation_config=hf_generation_config(args, eos_ids, int(tokenizer.pad_token_id)),
        )[:, ids.shape[1] :].tolist()
    print(f"Transformers CPU FP32生成耗时（含模型加载）：{time.perf_counter() - start:.3f}秒")
    if args.temperature > 0:
        print("两个框架使用不同的随机数实现，相同的随机种子也不保证相同的回答。下面分别显示生成结果。")
    for index, (jax_tokens, hf_tokens) in enumerate(zip(generated, output, strict=True), start=1):
        eos_index = next((i for i, token in enumerate(hf_tokens) if token in eos_ids), None)
        if eos_index is not None:
            hf_tokens = hf_tokens[: eos_index + 1]
        print(f"Transformers回答{index} token：{hf_tokens}")
        print(f"Transformers回答{index}文本：{tokenizer.decode(hf_tokens)!r}")
        if args.temperature == 0:
            first_diff = next((i for i, (a, b) in enumerate(zip_longest(jax_tokens, hf_tokens)) if a != b), None)
            print(f"回答{index} greedy token完全一致：{jax_tokens == hf_tokens}；首个差异位置：{first_diff}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    import jax
    import jax.numpy as jnp
    from transformers import AutoTokenizer

    from gemma4_posttrain_jax.sampler import SamplerConfig, generate
    from gemma4_posttrain_jax.weights import load_hf_eos_token_ids, load_hf_params

    snapshot = resolve_snapshot(args.model)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    ids, mask = encode_prompts(tokenizer, args.prompt, chat=args.chat)
    device = jax.devices()[0]
    print(
        f"设备：{device.device_kind}，使用1个设备；batch={len(args.prompt)}；prompt有效长度={mask.sum(axis=1).tolist()}"
    )
    print(f"采样设置：temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, seed={args.seed}")

    start = time.perf_counter()
    params, config = load_hf_params(snapshot, dtype=jnp.dtype(args.dtype))
    if args.top_k > config.vocab_size:
        raise ValueError(f"--top-k={args.top_k}超过模型词表大小{config.vocab_size}")
    if config.pad_token_id != tokenizer.pad_token_id:
        raise ValueError("模型与tokenizer的pad_token_id不一致")
    eos_ids = load_hf_eos_token_ids(snapshot, config)
    params = jax.device_put(params, device)
    jax.block_until_ready(params)
    print(f"JAX权重加载与设备传输：{time.perf_counter() - start:.3f}秒")

    sampler = SamplerConfig(
        max_prompt_len=ids.shape[1],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_ids=eos_ids,
        seed=args.seed,
    )

    @jax.jit
    def run(current, prompt_ids, prompt_mask, key):
        return generate(current, config, prompt_ids, prompt_mask, sampler_config=sampler, key=key)

    device_ids = jax.device_put(ids, device)
    device_mask = jax.device_put(mask, device)
    key = jax.device_put(jax.random.PRNGKey(args.seed), device)
    jax.block_until_ready((device_ids, device_mask, key))
    start = time.perf_counter()
    result = run(params, device_ids, device_mask, key)
    jax.block_until_ready(result)
    print(f"JAX生成耗时（含首次编译、prefill与decode）：{time.perf_counter() - start:.3f}秒")
    completions, lengths = jax.device_get((result.completion_ids, result.lengths))
    generated = [row[: int(length)].tolist() for row, length in zip(completions, lengths, strict=True)]
    for index, tokens in enumerate(generated, start=1):
        print(f"JAX回答{index} token：{tokens}")
        print(f"JAX回答{index}文本：{tokenizer.decode(tokens)!r}")
    peak = (device.memory_stats() or {}).get("peak_bytes_in_use")
    if peak is not None:
        print(f"设备内存峰值：{peak / 2**30:.3f} GiB")
    if args.compare_hf:
        compare_with_hf(args, snapshot, tokenizer, ids, mask, eos_ids, generated)


if __name__ == "__main__":
    main()
