# TPU performance optimization: start with an efficient framework and make full use of the hardware

**English** | [简体中文](tpu-performance-optimization.zh-CN.md)

Getting the most performance from a TPU starts with a good framework and continues down to the hardware. The framework must organize the complete workload efficiently. At the hardware level, we need to examine the actual data paths and instruction execution to find capabilities that the implementation leaves unused. Both determine end-to-end performance.

In synchronous GRPO experiments with Gemma 4 E2B-it, I compared complete iteration times for this project and a tuned Tunix baseline, each using its native rollout and the same tpu-inference engine. Across B8/N1024 and B16/N2048, this project with native rollout had the lowest iteration time. With native rollout on both sides, this project's throughput was **1.69–1.79×** that of Tunix. With the same inference engine on both sides, it was **1.03–1.14×**.

The table reports **seconds per complete iteration; lower is faster**. Each cell contains results from two independent processes. Each process performed 12 parameter updates, and its reported value is the median iteration time over iterations 7–12. The slash separates the two runs, not training and inference times.

| Configuration | This project + native rollout | Tunix + native rollout | This project + tpu-inference | Tunix + tpu-inference |
|---|---:|---:|---:|---:|
| B8 / N1024 | **4.095 / 4.098** | 7.177 / 7.350 | 9.754 / 9.623 | 11.084 / 10.825 |
| B16 / N2048 | **15.188 / 15.177** | 25.703 / 25.843 | 25.280 / 25.018 | 26.064 / 25.937 |

Both configurations use the same four TPU v4 chips, with training and generation taking turns on all four. B is the number of prompts per batch, with four responses per prompt. Prompt capacity is 512 tokens; N is the response-length limit, 1024 or 2048. The microbatch size is 4. Each rollout batch receives one GRPO update, with β=0 and layer rematerialization enabled. Model computation, inference weights, and KV caches use BF16; master parameters and Adam state use FP32. The two implementations have matching element counts for the common trainable parameters and Adam state.

Fixed complete response sequences keep the workloads equal. The models still execute generation, probability computation, gradient computation, and Adam updates. Each iteration contains 11,627 valid response tokens for B8 and 28,683 for B16, including EOS and excluding padding. For each generation backend, the run order is “this project → Tunix → Tunix → this project” to check for order effects.

Timing includes weight preparation and waits, generation, text decoding, reward computation, parameter updates, and logging inside the loop. It excludes initial loading and compilation, periodic checkpointing and evaluation, and the full-state audit after training. The table comes from completed original experiments. The reorganized public code still needs an E2B TPU rerun under the same configurations.

These results show that framework and generation choices can substantially affect complete iteration costs under the tested synchronous, fixed-workload conditions. The gap is smaller when both use the same inference engine. This supports using an efficient framework as the starting point for hardware optimization. It does not establish learning quality under natural sampling or a ranking across all batch sizes and generation lengths.

## 1. The objective is a better execution plan for TPU hardware

Operators, compilers, and computation graphs are ways to organize hardware instructions. Training and inference ultimately become instructions for moving data, computing, synchronizing, and communicating. Working at the instruction level lets us reason directly about TPU capabilities: where data resides, which execution units handle it, when DMA starts, and which dependencies require waits.

We can formalize this objective. Let:

- $`D`$ describe fixed hardware and runtime conditions: TPU generation, chip count, interconnect topology, memory hierarchy, and host environment.
- $`S`$ describe the required computational semantics, including precision, allowed numerical error, and output requirements.
- $`X`$ describe input conditions: ranges or distributions, shapes, batches, initial state, and request-arrival patterns for dynamic workloads.
- $`\mathcal{P}(D,S,X)`$ be the set of feasible execution plans satisfying these conditions, and $`T(p)`$ the end-to-end cost of plan $`p`$ under fixed timing boundaries and statistics.

When the objective is minimum execution time and a minimum exists, write an optimal plan as $`H`$:

```math
H \in \arg\min_{p\in\mathcal{P}(D,S,X)} T(p),
\qquad T^{\star}=T(H).
```

$`H`$ includes instruction sequences on each chip, data layouts, scheduling, communication, synchronization, and any necessary host-device coordination.

This defines a direction for optimization: reduce the gap between the current cost and $`T^{\star}`$. It does not imply that the optimum is unique, known, or reachable in finitely many experiments. Experiments can nevertheless bring us closer. If the minimum is not attained, $`\inf_{p\in\mathcal{P}}T(p)`$ expresses the theoretical lower bound. If the goal is serving cost or throughput under a latency constraint, the objective and constraints should change accordingly.

Higher-level abstractions affect which plans can be expressed. For a fixed framework $`F`$ and its permitted extensions, the valid plans it can produce form $`\mathcal{P}_F\subseteq\mathcal{P}`$. Therefore:

```math
\inf_{p\in\mathcal{P}}T(p)
\;\leq\;
\inf_{p\in\mathcal{P}_F}T(p).
```

Inappropriate operator boundaries, layouts, or communication choices can exclude faster plans. These constraints deserve examination at every abstraction layer. The inequality can still be an equality: a restriction on expressiveness does not mean that every layer necessarily adds runtime overhead.

We often treat compiler output as one feasible implementation and write custom kernels to improve it. A custom kernel is itself another feasible implementation within what its interface can express. Looking further down at instructions can reveal opportunities that remain hidden at the kernel level.

For example, I used tpuasm to optimize sampling in the separate `sampling-tpuasm-dev` project. The BF16 experiment sampled one token for each of 64 requests from a vocabulary of 131072 on four TPU v4 chips, with inputs already in HBM. Inspection of the native JAX MegaCore compilation showed that both TensorCores on each chip read the complete 4 MiB input and duplicated computation. I first used Pallas to assign 32 requests to each TensorCore and invoke the hardware PRNG, then improved data access through prefetching and tpuasm IAR rewrites.

The naive Pallas version merged results across chips outside the kernel, while the optimized version fused that work, so individual kernel times were not directly comparable. After aligning the complete sampling interval, TensorCore work assignment, and hardware random streams, median LCC fell from 78029 to 42095.5, about a 1.85× device speedup. The host-synchronized call fell from 328.11 to 300.39 microseconds, an 8.45% time reduction. Instruction-level reasoning made the optimization target more precise.

Making full use of the hardware means keeping the investigation open beyond the current framework, Pallas, or compiler interfaces. Lowering, tpuasm machine instructions, and execution timing remain useful places to look.

## 2. Why do we still need training and inference frameworks?

If hardware execution is the final target, why start with a framework? A complete system needs an efficient implementation that can be improved incrementally.

For a simple, fixed computation on one chip, it may be practical to start directly from TPU instructions and use experiments to approach the optimal plan $`H`$.

A distributed LLM workload involves much more: computation and dataflow, memory layouts, model sharding, cross-chip communication, and both compile-time and runtime scheduling. These decisions interact strongly. Expanding the search to instructions offers more possibilities, but the combined search space is enormous. Across hosts and chips, a layout change affects communication and memory use; fusion changes compiler prefetching and scheduling; dynamic requests alter batches, KV caches, and execution order. Even with direct instruction control, the computation and dataflow still need to be organized before useful optimization can begin.

Training and inference also contain reusable structures: matrix multiplication, attention, normalization, and distributed communication. Training adds gradients, optimizer state, and recovery; inference adds KV caches and generation scheduling. Established implementations provide a starting point for refinement by model, shape, precision, and hardware. Rebuilding the entire workflow from machine instructions every time would repeat substantial work.

An efficient framework provides a complete, correct, competitive baseline, a measurable cost, and concrete components to replace. The roughly 1.8× fixed-workload performance difference above shows that, under the measured configurations, starting from my framework already reduces execution cost. That advantage is useful, but it reflects the combined implementation and configuration. It cannot be attributed entirely to one scheduling policy or kernel.

Scheduling itself belongs in the optimization objective. In an earlier standard GRPO workflow, I found that generation and update programs were repeatedly loaded because there was insufficient space for both to remain resident. The profile showed about 111 ms of loading per update and another 43 ms for generation. Changing compilation settings reduced update-program code from about 0.73 GB to 0.18 GB, allowing both programs to remain resident. Device update computation became about 4 ms slower, but the complete workflow improved. Together with the preceding change to batched collection, mean training-step time fell from 2.981 to 2.748 seconds, a reduction of about 7.8%. The 1600 responses from 50 training steps, non-timing metrics, corresponding evaluations, and recovery and reload checks remained identical.

A framework provides a strong starting point for exploring better execution plans. I think a good TPU framework needs two complementary capabilities:

1. **Organize the complete workload.** It should support clear computation and dataflow, distributed execution, and scheduling. Scheduling includes overlap between computation and DMA within a chip, as well as multi-chip communication, pipelines, and dynamic batching at runtime. This gives the training or inference system the structure needed to run correctly, predictably, and efficiently.
2. **Expose the hardware's capabilities.** Developers should be able to control critical layouts, memory transfers, pipelines, and low-level execution. Defaults should be replaceable, and useful hardware capabilities should have a reasonable extension path. Performance observability should expose HLO, tpuasm instructions, data movement, and actual hardware cycles as well as elapsed time. This gives developers room to reason from the instructions themselves instead of stopping at a framework's existing computation or communication abstractions.

## 3. Agents accelerate exploration; TPU knowledge guides the questions and interpretation

Agents help me read code, implement candidates, organize experiments, and compare compiled output. They let me test more ideas and reach the compilation and instruction layers below Pallas faster. As experimental capacity grows, choosing worthwhile questions and interpreting results becomes even more important.

Generic requests such as “profile it, tune parameters, and make it faster” do not ensure that the investigation reaches the most valuable low-level opportunities. I need to turn knowledge of TPU memory, instruction dependencies, DMA, and ICI into specific hypotheses that an agent can implement and test.

I also need to judge whether the proposed and completed work follows the right direction. During early tpu-inference integration with two training chips and two inference chips, host-mediated weight synchronization took 41.529 seconds. I directed the investigation toward direct ICI transfer within a shared runtime and checked the actual communication path. The first attempt was slower. Further analysis showed that chunked writes repeatedly triggered full-parameter layout conversions. Preserving the original row layout while transferring chunks reduced complete synchronization of the same fixed weights to 3.345 seconds.

Interpreting results matters just as much. After I added the PLE block to a fused kernel, the local kernel became faster, but integrating it into the full generation loop slowed each decode step from 2.36 to 2.43 milliseconds. The agent initially attributed this to variation. That explanation did not fit my understanding: TPU execution is statically scheduled, so the pure-compute cycle count for the same program is repeatable. DMA and communication can vary, but within limits. The difference reproduced consistently in independent processes and in the same direction, which warranted checking the programs and execution conditions.

The difference turned out to involve the loop. The earlier kernel experiment used `scan`, whose iteration count was known at compilation. Complete generation used a `while` loop that stopped only when every response finished, so its trip count was unknown. XLA's memory planner estimated such a loop as executing five times. I had the agent sweep that setting, and the investigation traced the issue to this estimate in XLA's Memory Space Assignment (MSA). Under that estimate, an internal OOM-cache bug caused MSA to abandon prefetching, leaving matrix multiplication waiting for HBM DMA. Fixing the XLA bug restored better prefetching and reduced the time to 2.16 milliseconds.

Agent skills can establish systematic checks, but interpreting the evidence that supports a conclusion still requires my direct judgment and guidance.

## References

- [Pallas TPU Tutorial](https://github.com/ayaka14732/pallas-tpu-tutorial)
- [sampling-tpuasm-dev](https://github.com/yixiaoer/sampling-tpuasm-dev/tree/main)
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/index)
