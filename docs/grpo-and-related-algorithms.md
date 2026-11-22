# GRPO and related algorithms: from response rewards to parameter updates

**English** | [简体中文](grpo-and-related-algorithms.zh-CN.md)

Supervised fine-tuning (SFT) trains on questions paired with target answers. The model runs a forward pass on fixed text and learns to predict each next token. This is straightforward when a domain already has high-quality examples and ground truth.

For some tasks, however, checking an answer is much easier than writing a complete solution. In mathematics, checking whether the final answer is 42 can cost almost nothing, while writing all the reasoning steps takes human effort. Regardless of how many examples are written, SFT receives its direct supervision from those existing solutions. The model can generalize to new solutions, but the training objective itself does not evaluate the correctness of newly generated reasoning.

Reinforcement learning (RL) takes a different approach: the model generates several responses, an evaluator assigns rewards, and those rewards guide parameter updates. The training samples are the model's own outputs, with rewards providing the learning signal. This project's mathematical experiments still score responses against reference answers; they do not require a complete worked solution for every training response.

Group Relative Policy Optimization (GRPO) generates several responses to the same question and compares rewards within that group to decide which responses should become more probable. This document follows the complete calculation: rewards become advantages; probability-ratio clipping, KL regularization, masks, and length normalization define the loss; backpropagation and the optimizer update the parameters. It also explains the specific changes made by Dr. GRPO, DAPO, GSPO, and RLOO.

## 1. From imitating answers to learning from rewards

Let $`x`$ be a question, $`y`$ a response of length $`L`$, and $`\theta`$ the model parameters. At position $`t`$, the model predicts the next token from the question and preceding response tokens. The probability of the full response is therefore the product of these conditional probabilities:

```math
\pi_\theta(y\mid x)=\prod_{t=1}^{L}\pi_\theta(y_t\mid x,y_{\lt t})
```

Taking log-probabilities, or logprobs, turns the product into a sum:

```math
\log\pi_\theta(y\mid x)=\sum_{t=1}^{L}\log\pi_\theta(y_t\mid x,y_{\lt t})
```

**SFT** minimizes the negative logprob of a fixed target: run the forward pass, compute gradients, and update parameters.

**RL** aims to maximize expected response reward:

```math
J(\theta)=\mathbb E_x\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[R(x,y)]
```

Two parts are not differentiable here. Sampling a token is a discrete operation, and a reward function $`R(x,y)`$, such as a rule-based final-answer checker, is also non-differentiable.

The policy-gradient identity $`\nabla\pi = \pi\nabla\log\pi`$ gives:

```math
\nabla_\theta J=\mathbb E_x\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[R(x,y)\nabla_\theta\log\pi_\theta(y\mid x)]
```

The expectation $`E_{y\sim\pi_\theta(\cdot\mid x)}`$ requires sampling from the current model. After generation, the response $`y`$ is fixed, and $`R(x,y)`$ is a scalar. To compute the gradient, feed those generated tokens back through the model, just as in SFT, and calculate their logprobs. That forward pass is differentiable with respect to the parameters. Reward $`R`$ acts as a scalar multiplier on the gradient; neither the sampler nor the verifier participates in backpropagation.

This direct estimator can have high variance. Subtracting a suitable baseline from rewards is a common way to reduce it. GRPO builds its baseline from all responses to the same question, including the response being scored, and adds standardization, probability-ratio clipping, and length normalization. These choices change the actual objective from the simple expression above. The following sections examine each one.

## 2. Compute advantages within a group

In classical RL, the advantage $`A`$ measures how much the expected cumulative return of an action exceeds the baseline for its state, usually written as $`A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)`$. It concerns cumulative return, not just the immediate reward:

- $`A \gt  0`$: the action is better than the baseline, so its probability should increase.
- $`A \lt  0`$: the action is worse than the baseline, so its probability should decrease.

The baseline supplies a reference level. It might be a learned value function or a moving average of past rewards.

For GRPO here, an action is a complete response. The model generates $`G`$ responses to each question, forming a group. GRPO does not train a value network; it centers and scales rewards using the group's mean and standard deviation:

```math
A_i=\frac{R_i-\bar R}{s_R+\varepsilon}
```

The group mean is $`\bar R=\frac1G\sum_{i=1}^G R_i`$, the standard deviation is $`s_R=\sqrt{\frac1{G-1}\sum_{i=1}^G(R_i-\bar R)^2}`$, and $`\varepsilon=10^{-6}`$.

For example, four responses with rewards `[1, 1, 0, 0]` have mean $`\bar{R}=(1+1+0+0)/4=0.5`$ and sample standard deviation $`s_R\approx 0.57735`$. Their advantages are approximately $`A\approx[0.866, 0.866, -0.866, -0.866]`$. These values weight the gradients; the eventual probability changes also depend on the learning rate, optimizer state, and other factors.

Reward is a single score for the whole response, such as 1 for a correct answer, so every token in a response receives the same $`A_i`$. If reasoning goes wrong at step 50, the preceding 49 correct steps receive the same weight. One response-level score cannot identify where the error occurred.

The core of `compute_advantages` in [losses.py](../gemma4_posttrain_jax/losses.py) is:

```python
grouped = rewards.astype(jnp.float32).reshape(-1, group_size)
shifted = grouped - grouped[:, :1]  # Subtract the first reward from every reward in each group.
centered = shifted - shifted.mean(axis=-1, keepdims=True)
advantages = centered / (shifted.std(axis=-1, ddof=1, keepdims=True) + 1e-6)
```

`reshape` groups every $`G`$ rewards together, requiring responses to the same question to occupy consecutive array positions. `ddof=1` uses $`G-1`$ in the standard-deviation denominator rather than $`G`$.

The intermediate `shifted = grouped - grouped[:, :1]` subtracts the first reward from every reward in its group. This extra step improves floating-point stability; it is not required by the mathematical formula.

Consider a group in which all rewards equal `0.1`. Every advantage should be zero because no response is better than another. But 0.1 has no exact binary floating-point representation. Depending on the backend and reduction order, averaging 64 FP32 copies of 0.1 may not reproduce the stored FP32 value exactly. Subtraction can leave a small residual, which division by a denominator near $`10^{-6}`$ amplifies into an apparently meaningful advantage.

Subtracting the first value first makes each entry 0.1 − 0.1, exactly zero at the bit level. The mean and standard deviation of these zeros are also exactly zero, giving exactly zero advantages. The [tests](../tests/test_grpo.py) check constant rewards across different group sizes.

This only addresses the advantage contribution to the gradient. Zero advantages do not guarantee unchanged parameters: KL can still contribute gradients, and Adam's accumulated momentum can still produce updates, as discussed below.

## 3. Old policy and probability-ratio clipping

### Why keep an old policy?

Rollout, which samples complete responses from a generation policy, is often a major RL training cost. It generates tokens sequentially, and reasoning responses can be long. Its cost relative to an update depends on sequence length, batch size, and implementation. In practice, a generated batch is therefore often reused for $`\mu`$ parameter updates.

After the first update, however, the parameters have changed while the responses still come from the earlier model. Later updates need a reference for how far the current token probabilities have moved. The old policy records probabilities at the start of training on this batch and remains fixed throughout its updates.

For token $`t`$ of response $`i`$, define the probability ratio $`r`$ as:

```math
r_{i,t}(\theta)=\frac{\pi_\theta(y_{i,t}\mid x_i,y_{i,\lt t})}{\pi_{old}(y_{i,t}\mid x_i,y_{i,\lt t})}=\exp(\ell_{i,t}-\ell^{old}_{i,t})
```

Here $`\ell`$ is the current logprob and $`\ell^{old}`$ the old logprob. $`r=1.1`$ means that the token's conditional probability is 10% higher than the baseline. This implementation uses a ratio at each token position $`t`$, not a product over the entire response.

### PPO clipping limits the objective's incentive

The ratio $`r`$ measures movement relative to the old policy. Without a constraint, an optimizer can move some token probabilities substantially in one update. Proximal Policy Optimization (PPO) clips the ratio within an interval such as $`[0.8,1.2]`$. Once the probability has crossed the relevant boundary in the direction favored by the advantage, the objective stops rewarding further movement in that direction. This does not remove gradients at every position outside the interval. The policy loss to minimize is:

```math
\mathcal L^{policy}_{i,t}=-\min\left(r_{i,t}A_i,\;\mathrm{clip}(r_{i,t},1-\epsilon_l,1+\epsilon_h)A_i\right)
```

For positive advantage $`A=1`$, increasing $`r`$ from $`1.0`$ to $`1.1`$ improves the objective. Once $`r`$ exceeds $`1.2`$, or $`1+\epsilon_h`$, further increases no longer reduce the loss. Negative advantages produce the corresponding lower-bound protection against excessive probability decreases. The sign of $`A`$ determines which direction is clipped.

### The two clips in the code

`grpo_loss` expresses the maximization objective as a loss for the optimizer to minimize. For one advantage per response:

```python
old = lax.stop_gradient(policy) if old_logps is None else old_logps.astype(jnp.float32)
log_ratio = jnp.clip(policy - old, -20.0, 20.0)
ratio = jnp.exp(log_ratio)
advantage = advantages.astype(jnp.float32)[:, None]
unclipped = -advantage * ratio
clipped = -advantage * jnp.clip(ratio, 1.0 - eps_low, 1.0 + eps_high)
per_token_policy = jnp.maximum(unclipped, clipped)
```

The two `clip` calls have different purposes:

1. `log_ratio` is clipped to `[-20, 20]` to bound the exponential's input and prevent numerical overflow.
2. `ratio` is clipped to `[1-eps_low, 1+eps_high]` for the algorithm's PPO objective.

Their ranges, purposes, and effects on gradients are distinct.

### Zero loss does not imply zero gradient

`stop_gradient` preserves a value while blocking differentiation through that path. On the first update, a stopped-gradient copy of the current logprob serves as Old. The ratio $`r`$ is exactly 1, but its derivative with respect to the current logprob is still 1. Positive and negative advantages can cancel in the mean loss while leaving nonzero gradients at individual positions.

In the four-response example above, with two tokens per response, each logprob in a positive-advantage response has a gradient of about $`-0.108253`$; negative-advantage responses have the opposite sign. The [GRPO tests](../tests/test_grpo.py) check nonzero gradients at a ratio of 1 and updates in a small model.

Clipping is also not a hard bound on probabilities after an update. Gradients from other positions through shared parameters, together with Adam's historical momentum, can move the resulting $`r`$ outside the interval. Objective values, gradients, and actual parameter changes therefore need separate inspection.

## 4. Policy distributions and probability sources

Several logprob arrays in the training code have similar names. Their origins determine how to interpret the objective.

Consider the default synchronous workflow, starting RL from an SFT model $`\theta_0`$.

The **Reference Policy** is a frozen copy of $`\theta_0`$. Its parameters never change during training. It provides the fixed anchor used by KL regularization.

After 100 updates, the parameters are $`\theta_{100}`$. A new batch begins with the **Behavior Policy**, $`\theta_{100}`$, generating responses. Those responses remain fixed throughout training on the batch.

The first training forward pass uses the same $`\theta_{100}`$ to recompute logprobs for the fixed responses. These become the **Old Policy** probabilities, held constant as the denominator of $`r`$ throughout the $`\mu`$ updates.

Behavior and Old use the same parameters here, but their logprobs can differ. Generation decodes with a KV cache one token at a time; training processes the full sequence in parallel. Different floating-point operation orders explain why this code captures Old from the first training forward pass rather than reusing generation logprobs.

No additional complete Old model is needed. For fixed responses, storing their token-level `old_logps` is sufficient. This implementation captures them during the first actual gradient forward pass and keeps only the FP32 `[batch, completion_length]` array. If parameters, context, and probability computation match, Old and Behavior can share one logprob array; the names describe their sources, not an algorithmic prohibition on sharing. Keeping both records here lets us measure changes during training separately from differences introduced by generation, including native JAX rollout.

The $`\mu`$ updates then produce $`\theta_{101}`$, $`\theta_{102}`$, and so on. Logprobs recomputed with the parameters for each update are the **Current Policy** probabilities. These are the differentiable values and change with each update. Old remains the record from $`\theta_{100}`$; the ratio is Current divided by Old.

After all $`\mu`$ updates, the latest model generates the next batch. Behavior changes to that model, and Old is captured again. Reference stays fixed throughout. With sequential stale rollout, Behavior can come from an earlier parameter version while Old is still captured from the batch's first training pass. In that case their parameter versions, as well as their computation paths, differ.

| Name | Source and role |
|---|---|
| **Reference Policy** | Frozen at the start of training and never updated |
| **Behavior Policy** | The model that actually generated this batch; the current model in synchronous training, or an explicitly retained older policy |
| **Old Policy** | Captured during the first training forward pass for each batch and fixed across its $`\mu`$ updates |
| **Current Policy** | Changes after each gradient update and supplies the differentiable probabilities |

### Raw policy and actual sampling distributions

The raw policy distribution can differ from the proposal distribution used to sample. Temperature, top-k, and top-p change the distribution over selected tokens. The native generator's `rollout_logps` records raw full-vocabulary probabilities; there is no separate proposal-logprob field. The current RL entry point uses temperature 1 and full-vocabulary sampling. Truncated-sampling options from the generation CLI cannot simply be copied into training without addressing the sampling correction.

When Truncated Importance Sampling (TIS) is enabled, its weights act as fixed coefficients on the policy objective, not on the separate KL term. The behavior policy's parameter version, actual sampling method, and probability fields must agree. An array named `logprob` alone does not establish its meaning.

## 5. KL regularization

### Why use KL divergence?

Reward alone can encourage undesirable behavior: always using one response format, losing linguistic diversity, or exploiting a verifier to earn high scores for meaningless answers. A KL-divergence penalty measures departure from a fixed reference model and can reduce this drift. It does not guarantee that reward exploitation or mode collapse disappears. The project also permits $`\beta=0`$, which disables this term. The reference model is the frozen Reference Policy from Section 4.

Computing exact KL requires summing over the full vocabulary. This implementation uses the $`K_3`$ estimator, which needs only two logprobs at the sampled token. Let $`d = \log\pi_{ref} - \log\pi_\theta`$, the reference logprob minus the current logprob:

```math
k_3(d)=e^d-1-d
```

When $`d = 0`$, the two models assign the same token probability and $`k_3=e^0-1-0=0`$. The penalty grows as $`d`$ moves away from zero. For this uncapped formula at a fixed context, if tokens are sampled from the current policy $`\pi_\theta`$, both policies have the same support, and the relevant expectations are finite, the expectation of $`k_3`$ equals $`\mathrm{KL}(\pi_\theta\|\pi_{ref})`$. When reusing old responses, this project computes a regularizer on those samples; their batch mean is not the exact KL under the current policy.

The coefficient $`\beta`$ controls how strongly this term contributes to each token's loss:

```math
\mathcal L=\mathcal L^{policy}+\beta\mathcal L^{KL}
```

### Numerical stability

Even the short expression $`k_3(d)=e^d-1-d`$ requires care. Near zero, subtraction can lose significant digits. For large positive $`d`$, $`e^d`$ can dominate the update. The implementation uses a numerically stable piecewise calculation and treats the optional cap as a separate configuration. Improving numerical accuracy and capping the objective address different issues. The [KL numerical tests](../tests/test_kl_numerics.py) check values and gradients near zero, and finite results for larger differences.

## 6. Masks, length normalization, and microbatches

### Which positions contribute to the loss?

The forward pass processes the prompt and generated response as one sequence. Training uses the response logprobs weighted by the advantage. A loss mask excludes prompt positions and padding after EOS while retaining valid response tokens. To exclude a truncated response, zero its entire loss-mask row. The real prompt must remain available because the forward computation still needs the correct context.

### How reduction changes the objective

GRPO first averages over valid tokens within each response, then averages over nonempty responses. Let $`M_{i,t}`$ be the mask, $`L_i`$ the valid token count, and $`B_{valid}`$ the number of nonempty responses:

```math
\mathcal L=\frac1{B_{valid}}\sum_{i:L_i\gt 0}\frac1{L_i}\sum_t M_{i,t}\left(\mathcal L^{policy}_{i,t}+\beta k_{3,i,t}\right)
```

The corresponding code in `aggregate_token_loss` is:

```python
row_tokens = mask.sum(axis=-1)
nonempty_rows = jnp.maximum((row_tokens > 0).sum(), 1)
row_loss = (values * mask).sum(axis=-1) / jnp.maximum(row_tokens, 1)
return row_loss.sum() / nonempty_rows
```

These reduction choices determine each token's effective weight in the gradient. Suppose two responses have token losses `[1, 1]` and `[3]`. Averaging within each response and then across responses gives $`(1+3)/2=2`$. Averaging all three tokens gives $`(1+1+3)/3=5/3`$. The first gives each response equal total weight; the second gives each valid token equal weight.

### Gradient accumulation across microbatches

A microbatch is a smaller portion of the full batch used to reduce device memory use. Splitting must preserve the global objective: compute advantages over complete question groups first, split response rows for gradient computation, accumulate with the correct denominator weights, and perform one optimizer update at the end.

For the example above, consider the objective that averages over all valid tokens. This is distinct from default GRPO, which weights by nonempty response counts. Put the two responses in separate microbatches: their token means are 1 and 3. Equally averaging those means incorrectly gives 2. Weighting them by valid token counts, $`2/3`$ and $`1/3`$, recovers $`5/3`$. `grpo_train_step` accumulates gradients inside `lax.scan` and calls the optimizer only after the loop.

## 7. From logprob gradients to parameter changes

### Backpropagation

The training function concatenates prompts and fixed responses, computes the logprobs of the actual response tokens, and constructs a scalar loss. `jax.value_and_grad` differentiates through the model to obtain parameter gradients. The call structure in `grpo_train_step` is shown below; ellipses stand for configuration arguments:

```python
def objective(params, batch, weight):
    policy_logps = trainer_completion_logps(params, *batch[:4], ...)
    loss, _ = loss_from_logps(policy_logps, batch)
    return loss * weight, policy_logps

(_, policy_logps), grads = jax.value_and_grad(objective, has_aux=True)(
    state.params_f32, tokens, jnp.ones((), jnp.float32)
)
updates, opt_state = optimizer.update(grads, state.opt_state, state.params_f32)
params_f32 = optax.apply_updates(state.params_f32, updates)
```

### Adam updates

Adam (Adaptive Moment Estimation) uses both the current gradient and historical statistics. It maintains a first moment $`m`$ and a second raw moment $`v`$, the exponential moving average of squared gradients rather than their variance. After bias correction, the basic update is $`\Delta\theta = -\eta\hat{m}/(\sqrt{\hat{v}}+\varepsilon)`$. Consequently:

- The same gradient can produce different parameter changes under different Adam states.
- Even a zero current gradient can leave an update driven by existing momentum $`m`$.

### Mixed precision and master parameters

The project normally keeps master parameters and Adam state in FP32 while allowing BF16 (bfloat16) model computation. Small changes below BF16 resolution can then accumulate in the FP32 parameters. Casting a BF16 working copy back to FP32 every step does not recover the original master parameters; it discards the small updates being accumulated.

### Levels of numerical comparison

Correctness checks should follow the full chain. Similar losses do not establish that all gradients are similar, and similar gradients do not establish similar optimizer states or parameter changes. Recovery must also restore Adam state, the data cursor, and sampling RNG behavior. In the current native implementation, RNG behavior is reconstructed from the seed, cursor, and key-use rules. See the [design overview](design-overview.md) for checkpoint implementation and the [quickstart](quickstart.md) for commands.

## 8. Related algorithms

The basic path from forward logprobs to optimizer updates remains the same, but related algorithms change advantage computation, clipping, and reduction. The following combinations are the defaults in [algorithms.py](../gemma4_posttrain_jax/algorithms.py) and the training CLI. Command-line overrides determine the final configuration.

### Dr. GRPO

Dr. GRPO removes standard-deviation normalization, subtracting only the group mean, and uses a fixed generation-length budget as the denominator. For rewards `[1, 1, 0, 0]`, the advantages become positive and negative $`0.5`$ rather than $`0.866`$. Response length no longer changes the denominator. The two changes affect reward scaling and token weighting respectively.

### DAPO

DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) uses a larger upper clipping margin, such as $`0.28`$, and averages over all valid tokens in the batch. The full `dapo` entry point also includes dynamic resampling, truncation filtering, and a soft length penalty. `dapo-loss` selects only the corresponding objective. Dynamic resampling retains question groups containing both successes and failures, using binary success flags; a formatting reward cannot substitute for those flags. See the [DAPO algorithm description](https://dapo-sia.github.io/).

### GSPO-token

GSPO (Group Sequence Policy Optimization) defines probability ratios at the sequence level. This project implements GSPO-token: it first computes the response's mean log-ratio, then uses `policy − stop_gradient(policy) + stop_gradient(sequence_log_ratio)` to retain the current token's derivative. Both the forward statistic and backward path matter; describing it only as “sequence-level GRPO” would miss that distinction. The [GSPO paper](https://arxiv.org/abs/2507.18071) explains the motivation for sequence probability ratios; `grpo_loss` defines this implementation's exact computation.

### RLOO

RLOO (REINFORCE Leave-One-Out) subtracts the mean reward of the other $`G-1`$ responses in the group from the current response's reward. In the example above, advantages become positive and negative $`2/3`$. This implementation does not apply PPO probability-ratio clipping. It sums over tokens within each response, then averages over responses. The CLI restricts supported combinations, including $`\mu=1`$ and $`\beta=0`$.

### Comparison

| Algorithm | Advantage | Probability ratio and clipping interval | Loss reduction |
|---|---|---|---|
| **GRPO** | Center and scale by the group mean and standard deviation | Per token; $`[0.8, 1.2]`$ | Mean within each response, then mean over nonempty responses |
| **Dr. GRPO** | Subtract the group mean only | Per token; $`[0.8, 1.2]`$ | Divide by the fixed generation-length budget, then average over responses |
| **DAPO / DAPO loss** | Center and scale by the group mean and standard deviation | Per token; asymmetric interval $`[0.8, 1.28]`$ | Mean over all valid tokens in the batch |
| **GSPO-token** | Center and scale by the group mean and standard deviation | Ratio from the sequence's mean log-ratio; $`[0.9997, 1.0004]`$ | Mean within each response, then mean over nonempty responses |
| **RLOO** | Leave-one-out baseline | Per token; no PPO clipping | Sum within each response, then mean over responses |

These choices change gradient scales and sample weights. A smaller loss does not establish better learning. Comparisons also need aligned data, generated-token counts, update counts, and total cost. Equal Adam-step counts do not imply equal generation budgets, and training-time monitoring is separate from final independent evaluation.

---

## 9. Orchestrating the training loop

[train_grpo.py](../scripts/train_grpo.py) organizes data and generation on the host and executes model and gradient computations on the devices. The sequence is:

```text
Select questions by data_cursor; derive the random key from seed and cursor
       |
       v
Place G responses per question consecutively; generate tokens, masks, raw logprobs
       |
       v
Decode on the host and score with the verifier
       |  Optional DAPO resampling repeats until complete groups are available
       v
Compute advantages from final rewards and construct the loss mask
       |
       v
Compute frozen Reference logprobs if KL is enabled
       |
       v
Perform μ training updates on the same responses:
  1. Run the forward pass with the current parameters for Current logprobs.
  2. On the first update, capture Old from that pass; reuse it thereafter.
  3. Construct the loss: Policy Loss + β × KL Loss.
  4. Backpropagate; weight microbatch accumulation by the objective's denominator.
  5. Apply Adam to the FP32 master parameters.
       |
       v
Evaluate, save a checkpoint, or start the next rollout
```

`data_cursor` records the candidate question batches consumed; `step` records Adam updates. Responses discarded by dynamic resampling still advance the data position and the batch index used for random keys. These counters cannot substitute for one another, which connects the algorithm's control flow directly to checkpoint-recovery design.

This loop lets a language model generate its own attempts and adjust parameters using scalar response-level rewards, without a target answer for every reasoning step. Whether it improves complex reasoning depends on the rewards, data, and training settings and must be established through independent evaluation.
