# GRPO 及相关算法全解析：从回答的奖励到模型参数更新


监督微调（Supervised Fine-Tuning，SFT）依赖于"问题＋标准答案"的训练数据。训练时，模型对固定的目标文本做前向计算，逐 token 预测下一个未知的内容。如果该领域已有高质量数据以及对应 groundtruth，这种做法非常直接。

然而在有些任务中，判断一个回答是否正确远比写出完整的解答过程容易，比如在数学推导中，一道题检查最终答案是否等于 42 几乎没有成本，但要写出完整的推导步骤则需要实际的人力，而且无论写多少条，SFT 的直接监督信号仍来自这些已有解答；模型可以泛化出新解法，但训练目标本身不评价它新生成的推导是否正确。

强化学习（Reinforcement Learning，RL）给出了另一种解决思路：让模型自行生成多条回答，由评估器给出对回答的评价（奖励），再根据奖励反向调整模型参数。训练样本不再是事先写好的文本和对应标签，而是模型自己的生成结果；训练信号来自奖励。本仓库的数学实验仍使用题目对应的标准答案评分，只是不要求预先提供每条训练回答的完整推导。

GRPO（Group Relative Policy Optimization，组相对策略优化）的核心思路是：为同一道题生成多条回答，在组内比较它们的奖励，据此判断哪些回答值得提高概率。本文会沿着训练的完整计算顺序展开：从奖励怎样变成 advantage，到 loss 的各个部分（概率比裁剪、KL 正则、mask 与长度归一化），再到反向传播和优化器更新模型参数，拆解GRPO的完整计算过程，并横向对比 Dr. GRPO、DAPO、GSPO 与 RLOO 等相关算法的具体改进点。

## 1. 从模仿答案到利用奖励


设 $`x`$ 为输入问题，$`y`$ 为长度为 $`L`$ 的回答，$`\theta`$ 为模型参数。模型在每个位置 $`t`$ 根据问题和前面已有的回答内容预测下一个 token，因此整条回答的生成概率是各位置条件概率的连乘：

```math
\pi_\theta(y\mid x)=\prod_{t=1}^{L}\pi_\theta(y_t\mid x,y_{\lt t})
```

实际计算时取对数概率（log-probability，简称 logprob），把乘积变成求和：

```math
\log\pi_\theta(y\mid x)=\sum_{t=1}^{L}\log\pi_\theta(y_t\mid x,y_{\lt t})
```

**SFT 的做法**是直接最小化固定目标的负 logprob，模型对这条文本做前向计算，求梯度，更新参数。

**强化学习的做法**则是最大化回答的期望奖励：

```math
J(\theta)=\mathbb E_x\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[R(x,y)]
```




但此处有两个东西没法求导：从概率分布中抽取 token 是离散操作，而奖励函数 $`R(x,y)`$ （比如比对最终答案是否正确的规则验证器），也是不可导的。

这里利用策略梯度恒等式 $`\nabla\pi = \pi\nabla\log\pi`$，可以把期望奖励的梯度推导为：

```math
\nabla_\theta J=\mathbb E_x\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[R(x,y)\nabla_\theta\log\pi_\theta(y\mid x)]
```


根据这个恒等式，期望 $`E_{y\sim\pi_\theta(\cdot\mid x)}`$ 要求从当前模型采样，即用当前模型 generate 得到若干条具体的文本。采样完成后，生成的文本 $`y`$ 被确定，则 $`R(x,y)`$ 变成确定的标量。然后计算梯度：把已经生成好的文本喂回模型做一次前向计算，这和 SFT 完全一样，得到这些 token 的 logprob，而这个前向计算对模型参数可导。这样一来，奖励 $`R`$ 只是乘在梯度前面的标量系数，采样过程和验证器都不参与反向传播，绕开了前面说的没法求导的问题。

但直接使用这个公式方差很大，为了降低梯度估计的方差，通常需要让奖励减去一个合理的基准（Baseline）。GRPO 把同题内的全部回答（包括当前回答）放在一起构建基准，并进一步引入了标准化、概率比裁剪和按长度归一化等设计。加完之后实际优化的目标已经不是上面那个简单公式了，后面逐一介绍。



## 2. 组内相对优势计算（Advantage）

在经典强化学习中，优势估计（Advantage，记为 $`A`$）衡量的是在某个状态下选择一个动作的预期累计回报，比该状态的基准（Baseline）高多少，通常写作 $`A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)`$；它不只是比较当前一步的即时奖励：

- $`A \gt  0`$：该动作优于基准，希望提高其概率
- $`A \lt  0`$：该动作劣于基准，希望降低其概率

这里基准提供了平均水平的参照，具体可以是学习出来的价值函数（value）、历史奖励的移动平均等。

GRPO 中，一个"动作"就是对一道题生成的一条完整回答。GRPO 为同一道题生成 $`G`$ 条回答，这 $`G`$ 条回答构成一个组（Group，也是算法名称的由来）。GRPO 不训练价值网络，而是用组内所有回答的均值和标准差来构建基准。

```math
A_i=\frac{R_i-\bar R}{s_R+\varepsilon}
```

其中组内均值 $`\bar R=\frac1G\sum_{i=1}^G R_i`$，标准差$`s_R=\sqrt{\frac1{G-1}\sum_{i=1}^G(R_i-\bar R)^2}`$，$`\varepsilon=10^{-6}`$。


举一个具体例子：假设同一道题生成了 4 条回答，奖励为 `[1, 1, 0, 0]`。均值 $`\bar{R}=(1+1+0+0)/4=0.5`$，样本标准差 $`s_R\approx 0.57735`$，代入公式得到 advantage $`A\approx[0.866, 0.866, -0.866, -0.866]`$。这些 Advantage 数值是乘在梯度前面的权重，最终概率变化还取决于学习率、优化器状态等因素。另外，奖励是针对整条回答的一个分数（比如"答对了"得 1 分），所以同一条回答里的每个 token 都乘以同一个 $`A_i`$。如果一条回答在第 50 步推导出错，前面正确的 49 步也会被同样对待，整条回答一个分数，无法区分到哪一步出了问题。


[losses.py](../gemma4_posttrain_jax/losses.py) 中 `compute_advantages` 的核心实现：

```python
grouped = rewards.astype(jnp.float32).reshape(-1, group_size)
shifted = grouped - grouped[:, :1]  # 把每组的每个奖励都减去该组第一个奖励
centered = shifted - shifted.mean(axis=-1, keepdims=True)
advantages = centered / (shifted.std(axis=-1, ddof=1, keepdims=True) + 1e-6)
```


* reshape 把奖励按每 $`G`$ 条一组排列，要求同题回答在数组中连续存放。

* `shifted = grouped - grouped[:, :1]` 把每组的所有奖励减去该组第一项
    * 这不是公式要求的步骤，而是为了浮点数值稳定性额外加的操作
    * 当一组回答的奖励完全相同时，比如全部为 `0.1`
        * advantage 应该全部为零，因为没有任何一条比其他的好。
        * 但 0.1 在二进制浮点数里无法精确表示，在某些后端和归约顺序下，64 个 FP32 的 0.1 求和再取均值，可能不等于原来存储的那个 FP32 数值。两者相减就留下极小的残差，再除以接近 $`10^{-6}`$ 的分母（标准差加上 $`\varepsilon`$），会被放大成看起来有意义的数，但其实是浮点误差。
        * 在这种情况下，先把每个值减去组内第一项：0.1 − 0.1。因为是同一个浮点数减自己，结果在位级别上精确为 0。之后对一组全零的数求均值和标准差，结果都精确为零，advantage 也精确为零。
        * 上面讨论的只是 advantage 这一项梯度。即使 advantage 成功归零，模型参数也不一定就不动，因为 KL 正则项仍可能提供梯度，而 Adam 优化器里之前积累的动量也可能继续影响参数更新，后面会有专门的章节解释。
    * [对应测试](../tests/test_grpo.py)检查不同组大小下的常量奖励是否得到精确为零的 advantage。
* `ddof=1` 对应公式中除以 $`G-1`$，而非 $`G`$。



## 3. Old Policy 与概率比裁剪（Probability Ratio Clipping）

### 为什么需要 Old Policy

Rollout（用生成策略采样完整回答）通常是 RL 训练的主要开销之一：需要逐 token 生成，而推理任务的回答往往又比较长。它与一次训练更新的耗时关系取决于长度、批大小和实现。因此对于采样生成的本batch回答，在实践中常常复用 $`\mu`$ 次来更新模型。

但这就带来一个问题：第一次更新之后，模型参数已经变了，而回答仍然是更新前的模型生成的。后续每次更新时，我们需要知道当前模型给这些 token 的概率相比训练开始时偏移了多少，偏移过大则需要限制更新幅度。旧策略（Old Policy）就是在本batch训练开始时记录下来的概率，后续更新都拿它作参照。

Old Policy 在本batch开始训练时固定下来。对第 $`i`$ 条回答的第 $`t`$ 个 token，概率比（Probability Ratio，记为 $`r`$）定义为：

```math
r_{i,t}(\theta)=\frac{\pi_\theta(y_{i,t}\mid x_i,y_{i,\lt t})}{\pi_{old}(y_{i,t}\mid x_i,y_{i,\lt t})}=\exp(\ell_{i,t}-\ell^{old}_{i,t})
```

其中 $`\ell`$ 为当前 logprob，$`\ell^{old}`$ 为旧策略 logprob。$`r=1.1`$ 意味着这个 token 的条件概率比基准高了 10%。当前实现使用的是逐 token 的概率比，在每个位置 $`t`$，分别算当前模型和旧模型在该位置的条件概率，然后相除；并非整条回答概率的乘积。

### 怎样限制目标对概率变化的激励：PPO clipping
概率比 $`r`$ 衡量的是当前模型相对旧模型的偏移，如果不加限制，优化器可能在一次更新中把某些 token 的概率推得很远。近端策略优化（Proximal Policy Optimization，PPO）的 clipping（裁剪）方法通过截断这个比值 $`r`$，设定目标中的裁剪区间（比如 $`[0.8,1.2]`$）。当概率已沿 advantage 指示的方向越过相应边界时，目标不再鼓励继续朝该方向变化；它不是所有区间外位置都停止提供梯度。最小化的策略损失为：

```math
\mathcal L^{policy}_{i,t}=-\min\left(r_{i,t}A_i,\;\mathrm{clip}(r_{i,t},1-\epsilon_l,1+\epsilon_h)A_i\right)
```

代入具体数值帮助理解 `min` 的作用：以正优势 $`A=1`$ 为例，当 $`r`$ 从 $`1.0`$ 增长到 $`1.1`$ 时，目标如实奖励这种变化；但当 $`r`$ 已经超过 $`1.2`$（即 $`1+\epsilon_h`$），继续增加也不会进一步减小 loss 了。对于负优势，则形成对概率过度下降的下界保护。clipping 不是把所有超出区间的梯度都设为零，梯度的方向取决于 $`A`$ 的符号。

### 代码中的两次 clip

`grpo_loss` 将最大化目标改写成供优化器最小化的 loss。以下片段展示了每条回答一个 advantage 的情形：

```python
old = lax.stop_gradient(policy) if old_logps is None else old_logps.astype(jnp.float32)
log_ratio = jnp.clip(policy - old, -20.0, 20.0)
ratio = jnp.exp(log_ratio)
advantage = advantages.astype(jnp.float32)[:, None]
unclipped = -advantage * ratio
clipped = -advantage * jnp.clip(ratio, 1.0 - eps_low, 1.0 + eps_high)
per_token_policy = jnp.maximum(unclipped, clipped)
```

代码中有两次 `clip` 操作，作用完全不同：

1. **第 1 次**作用于 `log_ratio`，区间为 `[-20, 20]`，目的是限制指数计算的输入范围、防止数值溢出；
2. **第 2 次**作用于 `ratio`，区间为 `[1-eps_low, 1+eps_high]`，才是算法层面的 PPO 概率比裁剪。

两者的区间、目的和对梯度的影响各不相同。

### Loss 为零不代表梯度为零

`stop_gradient` 保留数值但阻止沿该路径求导。首次更新时使用当前 logprob 的停止求导副本作为 old，这样 $`r`$ 的数值恰好为 1，但对当前 logprob 的导数仍然为 1。即使正负 advantage 恰好抵消使得平均 loss 为零，各个位置的梯度也不为零。

在上面四条回答、每条两个 token 的例子中，正 advantage 回答每个 logprob 的梯度约为 $`-0.108253`$，负 advantage 回答则符号相反。[GRPO 测试](../tests/test_grpo.py)进一步检查了 ratio 为 1 时的非零梯度与小模型更新。

另外需要注意：概率比裁剪不是对最终概率的硬约束。其他位置通过共享参数产生的梯度，以及 Adam 的历史动量，都可能让更新后的 $`r`$ 超出裁剪区间。因此需要分别观察目标值、梯度和实际参数变化，不能只看其中一个。


## 4. 区分策略分布与概率来源

训练代码中有几组名字相似的 logprob，搞清楚它们各自的来源才能正确理解训练目标。

举例说明，下面先考虑默认的同步采样流程，假设 RL 训练开始之前有一个 SFT 过的模型 $`\theta_0`$。

**Reference Policy（参考策略）** 就是 $`\theta_0`$ 的一份冻结拷贝。从训练开始到结束，其参数永远不动，唯一作用是提供一个"别偏离太远"的锚点，后面 KL 正则项那节会用到它。

训练进行了一段时间，模型参数变成了 $`\theta_{100}`$（已经更新了 100 步）。现在要开始新一个 batch 的训练：

**Behavior Policy（行为策略）** 就是 $`\theta_{100}`$，用它跑 rollout 生成一个 batch 的回答，生成完成后这批回答就固定了，后面不管模型怎么变，回答不会重新生成。

接下来进入训练循环。第一次前向计算时，用同一个 $`\theta_{100}`$ 对这 batch 固定回答重新算一遍 logprob 并记录下来，这就是 **Old Policy（旧策略）**。后续 $`\mu`$ 次更新中，Old Policy 不再变化，始终作为概率比 $`r`$ 的分母。

Behavior Policy 和 Old Policy 来自同一份参数 $`\theta_{100}`$，但 logprob 的数值可能不同：生成时用 KV Cache 逐 token 解码，训练时用整条序列并行前向，浮点计算顺序不同。因此代码使用首次训练前向中计算的 logprob 作为 Old Policy，而不是直接复用 generate 时记录的 logprob。

这并不要求额外保存一整份 Old 模型。对已经生成的固定回答，保存对应 token 的 `old_logps` 就足够了；本实现从第一次实际梯度前向中捕获它，之后只保留这个 `[batch, completion_length]` 的 FP32 数组。若参数、上下文与概率计算一致，Old 和 Behavior 可以复用同一份 logprob；区分两个名字是为了表达来源，不是算法禁止共享。当前保留两份概率记录，是为了分别衡量训练中的策略变化和生成路径的概率差异，也适用于原生 JAX rollout。



然后开始 $`\mu`$ 次更新。第一次更新后模型变成 $`\theta_{101}`$，第二次变成 $`\theta_{102}`$……每次更新时用当前参数重新算出来的 logprob 就是 **Current Policy（当前策略）**。它是唯一参与求导的量，每次更新都在变。而 Old Policy 始终是 $`\theta_{100}`$ 时记录的那份，不变。概率比 $`r`$ 就是 Current 除以 Old。

整个 $`\mu`$ 次更新结束后，再用最新的模型跑下一个 batch 的 rollout，Behavior Policy 变成新的模型，Old Policy 也重新捕获，如此循环。Reference Policy 自始至终不动。若启用顺序滞后的 rollout 配置，Behavior Policy 可以来自较早的参数版本，而 Old Policy 仍从本批第一次训练计算中捕获；两者此时不仅计算方式不同，参数版本也不同。

总结：

| 名称 | 来源与作用 |
|---|---|
| **Reference Policy** | 全局固定，训练开始时冻结，永不更新 |
| **Behavior Policy** | 实际生成本 batch 回答的模型；同步流程中是当前模型，也可以是显式保存的较旧策略 |
| **Old Policy** | 每个 batch 第一次训练前向时捕获，$`\mu`$ 次更新中保持不变 |
| **Current Policy** | 每次梯度更新后都在变，是求导的对象 |

### 原始策略分布与实际采样分布

还需要区分 raw policy distribution（原始策略分布）与 proposal distribution（实际采样分布）。温度、TopK、Top-p 都会改变选择 token 的分布。原生生成器的 `rollout_logps` 记录的是原始完整词表的概率，没有单独的 proposal logprob 字段。当前 RL 入口固定使用温度 1 和完整词表采样；生成 CLI 开放的截断采样选项不能直接照搬进训练而忽略采样修正。

若启用截断重要性采样（Truncated Importance Sampling，TIS），权重作为固定系数作用于策略目标，不乘独立的 KL 项。行为策略的参数版本、实际采样方式和概率字段必须一起对应，单靠一个名叫 `logprob` 的数组不足以确定它的含义。



## 5. KL 正则项约束

### 为什么需要 KL 散度

RL 训练只看奖励信号。如果不加限制，模型可能为了拿高分而走极端，比如只输出一种固定格式的回答、丢掉语言的多样性，甚至在验证器上找到漏洞拿高分但回答实际上没意义。为了减轻这种偏离，可以在训练目标中加入惩罚项：用 KL 散度（Kullback–Leibler Divergence）衡量当前模型和固定参考模型的分布差异。它不能保证消除奖励漏洞或模式退化，而且本仓库允许 $`\beta=0`$，不启用这一项。这里的参考模型就是第 4 节中的 Reference Policy，全程不动。

直接算两个分布的 KL 散度需要对整个词表求和，计算代价很高。本仓库默认采用名为 $`K_3`$ 的估计式，只需要当前 token 位置上的两个 logprob 就能给出一个估计值。令对数差值 $`d = \log\pi_{ref} - \log\pi_\theta`$（参考模型的 logprob 减去当前模型的 logprob）：

```math
k_3(d)=e^d-1-d
```

当 $`d = 0`$，即两个模型在该 token 上概率相同时 $`k_3=e^0-1-0=0`$ ，没有惩罚。 $`d`$ 越偏离零，$`k_3`$越大惩罚越重。对于上面未加上限的公式，在固定上下文下，若 token 来自当前策略 $`\pi_\theta`$，两者具有相同的支持集且相关期望有限，则 $`k_3`$ 的期望等于 $`\mathrm{KL}(\pi_\theta\|\pi_{ref})`$。本仓库复用旧回答时，使用的是这些样本上的正则估计值，不能将其 batch 均值直接视作当前策略下的精确 KL。

这个值加到每个 token 的 loss 上，由系数$`\beta`$ 控制惩罚力度：

```math
\mathcal L=\mathcal L^{policy}+\beta\mathcal L^{KL}
```


### 数值稳定性

$`k_3(d)=e^d-1-d`$这个看似简短的公式也影响实现上的选择。$`d`$ 接近零时，直接相减可能丢失有效数字；$`d`$ 很大时，指数项 $`e^d`$ 可能主导整个更新。仓库采用了分段稳定的计算方式，并将可选上限视为独立的配置项，提高计算精度与改变目标上限解决的是不同层面的问题。[KL 数值测试](../tests/test_kl_numerics.py)分别检查接近零时的函数值、梯度以及较大差值下的有限性。


## 6. Mask、长度归一化与 Microbatch

### 哪些位置计入 Loss

训练时 prompt 和模型生成的回答拼成一条序列做前向计算，模型会对每个位置都输出预测，但我们只需要回答部分的 logprob 乘以 advantage 来产生梯度，prompt 部分不应该计入。Loss mask（掩码）就是用来标记哪些位置的 logprob 参与 loss 计算的：prompt 位置和 EOS 之后的 padding 位置标记为零，只保留回答区间内的有效 token。如果某条回答超长需要过滤，将其整行 mask 置零即可，不需要替换 prompt 内容，因为模型前向计算仍然需要读取真实的上下文。

### 聚合方式的影响

GRPO 的损失聚合方式是先在每条回答内按有效 token 数求平均，再对所有非空回答求平均。设 $`M_{i,t}`$ 为掩码标识，$`L_i`$ 为有效 token 长度，$`B_{valid}`$ 为非空回答总数：

```math
\mathcal L=\frac1{B_{valid}}\sum_{i:L_i\gt 0}\frac1{L_i}\sum_t M_{i,t}\left(\mathcal L^{policy}_{i,t}+\beta k_{3,i,t}\right)
```

`aggregate_token_loss` 中对应的代码：

```python
row_tokens = mask.sum(axis=-1)
nonempty_rows = jnp.maximum((row_tokens > 0).sum(), 1)
row_loss = (values * mask).sum(axis=-1) / jnp.maximum(row_tokens, 1)
return row_loss.sum() / nonempty_rows
```

求和与求平均的选择属于归约（Reduction）方式，它们决定每个 token 对最终梯度的实际权重。用一个具体例子说明差异：假设两条回答的 token loss 分别为 `[1, 1]` 和 `[3]`。先按回答平均得到 $`(1+3)/2=2`$；直接对三个 token 取平均则得到 $`(1+1+3)/3=5/3`$。前者让两条回答的总权重相同，后者让每个有效 token 的权重相同。

### Microbatch 的梯度累积规则

Microbatch（微批次）是为了控制显存占用而从完整 batch 中拆出的较小批次。拆分必须保持全局数学等价性：先在完整的同题组上算好 advantage，再拆分回答行求梯度，按正确的分母加权累积，最后执行一次优化器更新。

继续上面的例子，这里专门考察全部有效 token 平均的目标，而不是 GRPO 默认的按回答平均；后者应按非空回答数加权。如果把两条回答分进两个 microbatch，它们的 token 均值分别是 1 和 3。直接对这两个均值取平均会错误地得到 2，而按有效 token 数以 $`2/3`$ 和 $`1/3`$ 加权才能得到正确的 $`5/3`$。实际 `grpo_train_step` 在 `lax.scan` 中累积梯度，循环结束后才调用优化器。


## 7. 从 Logprob 梯度到参数变化

### 反向传播流程

训练函数将问题与固定回答拼接，计算回答位置上实际 token 的 logprob，然后构造标量 loss。`jax.value_and_grad` 沿模型的前向和反向传播路径，得到模型参数的梯度。以下是 `grpo_train_step` 的调用关系（省略号代表配置参数）：

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

### Adam 优化器的行为

Adam（Adaptive Moment Estimation，自适应矩估计）同时使用当前梯度与历史统计量。它维护梯度的一阶矩 $`m`$ 和二阶原点矩 $`v`$（梯度平方的指数移动平均，不是方差），经偏差修正后，基本更新形式是 $`\Delta\theta = -\eta\hat{m}/(\sqrt{\hat{v}}+\varepsilon)`$。因此：

- 相同的梯度在不同的 Adam 状态下可能得到不同的参数变化；
- 即使当前梯度为零，也可能因为已有的动量 $`m`$ 继续移动参数。

### 混合精度与主参数

仓库通常保留 FP32 精度的主参数（Master Weights）和 Adam 状态，允许模型计算使用 BF16（bfloat16，16 位浮点格式）。这样，小于 BF16 表示精度的微小参数变化仍有机会在 FP32 主参数中逐步累积。不能每步把 BF16 计算副本转回 FP32，就当作原来的主参数——这会截断累积中的微小更新。

### 数值对照的层级

验证实现正确性时，应当沿这条路径逐级检查：loss 接近不一定说明全部梯度都接近，梯度接近也不一定说明优化器状态和实际写回的参数变化接近。保存与续训还必须完整恢复 Adam 状态、数据游标（data cursor）和采样随机状态（Random Number Generator，RNG；当前原生路径由 seed、data cursor 和 key 使用规则共同重建），保存与恢复的实现见[整体设计](design-overview.md)，运行方法见[快速开始](quickstart.md)。

## 8. 相关衍生算法对比

在保持"计算前向 logprob → 优化器更新"基本路径不变的前提下，不同算法在 advantage 计算、裁剪机制和聚合方式上做出了针对性调整。以下是 [algorithms.py](../gemma4_posttrain_jax/algorithms.py) 及训练 CLI 的默认组合，命令行覆盖后以最终配置为准。

### Dr. GRPO

去掉了 advantage 的标准差归一化，改为只减组内均值，并用固定生成长度预算作为分母。对同一组 `[1, 1, 0, 0]` 的奖励，advantage 变为正负 $`0.5`$（而非正负 $`0.866`$）。回答长度的变化不再改变分母。这两项调整分别影响奖励尺度与 token 权重。

### DAPO

DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization，可理解为分离裁剪边界、结合动态采样的策略优化）使用较大的上侧裁剪范围（如 $`0.28`$），并对全 batch 的有效 token 取平均。完整的 `dapo` 入口还包括动态补采、截断过滤和软长度惩罚，而 `dapo-loss` 入口只选择相应的目标函数。动态补采按二值成功标记保留同题组内既有成功又有失败的组——格式奖励不能代替成功标记。这些组成部分对应 [DAPO 的算法说明](https://dapo-sia.github.io/)。

### GSPO-token

GSPO（Group Sequence Policy Optimization，组序列策略优化）将概率比的统计对象扩展到序列层面。本项目实现的是 GSPO-token：先计算回答内的平均 log-ratio，再用 `policy − stop_gradient(policy) + stop_gradient(sequence_log_ratio)` 保留当前 token 的导数。前向统计与反向路径需要同时说明，不能把它笼统概括为任意的"序列级 GRPO"。[GSPO 论文](https://arxiv.org/abs/2507.18071)介绍了序列概率比的动机；当前代码的具体计算以 `grpo_loss` 为准。

### RLOO

RLOO（REINFORCE Leave-One-Out，使用留一基准的策略梯度）先以同组其他 $`G-1`$ 条回答的平均奖励作为基准，再让当前回答的奖励减去这个基准。在上面的例子中，advantage 变为正负 $`2/3`$。当前实现不做 PPO 概率比裁剪，按回答内 token 求和后再对回答取平均；CLI 限制 $`\mu=1`$、$`\beta=0`$ 等组合。

### 对比总览

| 算法 | Advantage 计算方式 | 概率比与裁剪范围 | Loss 聚合方式 |
|---|---|---|---|
| **GRPO** | 组内均值及标准差标准化 | Token 级；$`[0.8, 1.2]`$ | 行内求均值，再按非空回答行求均值 |
| **Dr. GRPO** | 仅减去组内均值 | Token 级；$`[0.8, 1.2]`$ | 除以固定生成长度预算，再对回答求均值 |
| **DAPO / DAPO loss** | 组内均值及标准差标准化 | Token 级；非对称区间 $`[0.8, 1.28]`$ | 全 Batch 所有有效 Token 取均值 |
| **GSPO-token** | 组内均值及标准差标准化 | 序列平均 Log-Ratio 构造的 ratio；$`[0.9997, 1.0004]`$ | 行内求均值，再按非空回答行求均值 |
| **RLOO** | 留一法基准 | Token 级；无 PPO 裁剪 | 行内求和，再按回答行求均值 |

这些选择会改变梯度的尺度和样本的权重，不能因为某种算法的 loss 数值更小就断定效果更好。做学习实验时还需要对齐数据、生成 token 数、更新次数和总成本。相同 Adam 步数不代表相同生成预算，训练过程中的监测也需要与最终独立评估分开。

---

## 9. 训练流程的总控逻辑

[train_grpo.py](../scripts/train_grpo.py) 在主机端组织数据与生成，在设备端执行模型及梯度计算。整个流程按以下时序运行：

```
[按 data_cursor 选择问题，从 seed 与 cursor 派生随机 key]
       │
       ▼
[为每题建立连续 G 条回答输入，生成器返回 token、mask、raw logprob]
       │
       ▼
[主机解码文本 & 验证器评分] ──(可选: DAPO 动态补采，重复直到得到完整组)
       │
       ▼
[根据最终奖励计算 Advantage，建立 loss mask]
       │
       ▼
[计算固定 Reference 的 logprob（若启用 KL）]
       │
       ▼
┌─► [同一批回答执行 μ 次训练更新] ─────────────────────────────────┐
│   1. 用本次参数前向计算 current policy logprob          │
│   2. 首次更新：从同一次前向捕获 old；后续更新复用 old      │
│   3. 组装 Loss（Policy Loss + β × KL Loss）             │
│   4. 反向传播；若分 microbatch，按目标分母累积梯度         │
│   5. Adam 优化器更新 FP32 主参数                        │
└────────────────────────────────────────────────────────┘
       │
       ▼
[评估、保存 Checkpoint，或开始下一轮生成]
```

`data_cursor` 是指向训练数据的位置指针，记录的是已经处理的候选问题批次，`step` 记录的是 Adam 更新次数。被动态补采筛掉的回答虽然没有参与参数更新，但已经推进了数据位置和随机 key 的批次编号——因此这两个计数器不能互相替代。这恰恰是算法流程与断点续训设计衔接的地方。

通过这一套闭环流程，模型实现了从"自行尝试生成"到"利用标量奖励反馈调整参数"的完整循环，使语言模型可以利用回答级奖励训练，而不要求每个推理步骤都有标准答案；是否改善复杂推理能力，仍取决于奖励、数据和训练设置，需要由独立评估确认。
