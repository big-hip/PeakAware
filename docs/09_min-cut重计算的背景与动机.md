# Min-Cut 重计算的背景与动机

## 1. 背景：激活重计算解决的不是全部显存问题

大模型训练中的 GPU 显存峰值通常由参数、梯度、优化器状态、前向激活、反向临时张量和
kernel workspace 共同构成。激活重计算的基本思想是：前向阶段不保存部分中间激活，在反向
阶段按需重新执行相应前向算子，从而用额外计算换取较低的激活保存量。

在 PyTorch 2 编译栈中，AOTAutograd 会先构建包含 forward 与 backward 的 joint graph，再由
partitioner 将其拆分为前向图和反向图。`min_cut_rematerialization_partition` 使用
min-cut/max-flow 思想，在 joint graph 上选择哪些前向值需要保存给 backward，哪些值可以在
backward 中重算。它的核心目标不是最小化完整训练步的物理 CUDA 峰值，而是在一个编译区域内
减少 saved-for-backward activation，并在内存节省与重计算开销之间取得折中。

PyTorch 的 memory-budget 语义也有类似限定：这里的 memory 指固定区域中保存给 backward 的
内存，而不是训练 step 的 end-to-end peak memory。因此，`activation_memory_budget` 降低后，
前向结束时的 retained activation 可能明显减少，但整体峰值不一定同步降低。

## 2. 现象：saved activation 降低不保证 peak 降低

草稿实验在 PyTorch 2.13.0+cu130 和 RTX A6000 上复现了一个重要现象：min-cut 更激进地删除
前向保存值后，前向结束内存确实下降，但 backward 阶段的重算临时量可能抵消甚至超过该收益。

小规模 GPT-2 AOT 实验显示，min-cut 相比 default partition 只带来极小的 peak 降幅，却增加
了 step time：

| 策略 | overall peak | after-FW | step time |
|---|---:|---:|---:|
| AOT default | 144.359 MiB | 68.946 MiB | 9.036 ms |
| AOT min-cut | 143.843 MiB | 68.431 MiB | 10.095 ms |

在 activation-dominated 模型中，现象更清楚。降低 budget 会减少前向保留值，但整体峰值和
执行时间反而升高：

| 策略 | FW retained | overall peak | step time |
|---|---:|---:|---:|
| AOT min-cut, budget=1.0 | 16.0 MiB | 39.27 MiB | 8.10 ms |
| AOT min-cut, budget=0.75 | 4.0 MiB | 43.76 MiB | 8.97 ms |
| AOT min-cut, budget=0.0 | 0.0 MiB | 43.76 MiB | 10.27 ms |

草稿中更大规模的逐节点实验进一步显示：在相同 PyTorch 2.13.0+cu130 环境下，
`aot_mincut_b0_75` 的 after-FW 为 240.315 MiB，但 BW peak 达到 1216.332 MiB；而块级
Classic Activation Checkpointing 的 after-FW 接近，为 240.315 MiB，BW peak 仅 327.363 MiB。
这说明 saved activation 总量不能单独解释最终峰值，backward 中重算张量的生命周期同样关键。

## 3. 根因：优化目标与物理峰值错位

完整训练峰值可近似理解为时间轴上的最大活跃集合：

```text
Peak = max_t(
    parameters
  + saved activations live at t
  + recomputed tensors live at t
  + gradients materialized by t
  + optimizer state and fixed frontiers
  + kernel workspace and compiler materialization
)
```

min-cut 的直接决策对象主要是 saved values。这个目标缺少完整时间维度，也不直接建模
backward 中重算子图的实际执行顺序、张量 last-use、Inductor fusion 和 buffer reuse。因此，
它可能选择一个在 saved activation 上更优、但在物理执行时间线上更差的方案。

典型失败模式出现在 residual chain 中。代价模型倾向保存昂贵的 Linear/mm 输出，重算便宜且
可融合的 pointwise/add。单个 add 或 tanh 的重算成本很低，但当多个 block 通过残差状态串联时，
为了得到深层 backward 所需状态，反向图可能先连续重建一整段前向链：

```text
gelu -> sigmoid -> mul -> sin -> mul -> add -> tanh -> residual add
```

如果这些重算结果在第一个关键 backward op 前同时存活，就会形成 rematerialization wave。
此时，前向少保存的内存被 backward 临时量重新占用，甚至造成更高峰值。块级 Classic AC 往往
表现更稳，是因为它的 schedule 更接近：

```text
recompute block N -> backward block N -> release
recompute block N-1 -> backward block N-1 -> release
```

局部窗口较小，重算结果不会长期跨层存活。

## 4. PyTorch 已有策略及其边界

PyTorch 并非完全忽视该问题。当前实现中已有若干缓解机制：

| 策略 | 作用 | 边界 |
|---|---|---|
| conservative recomputation bans | 默认禁止重算 reduction、非白名单 op、materialized backward 相关节点等 | 只能降低坏决策概率，不能保证整步 peak 单调下降 |
| `reordering_to_mimic_autograd_engine()` | 重排反向图，避免重算前向子图过早产生并长期存活 | 不能违反数据依赖；错误 cut 仍可能强制形成长重算波 |
| Inductor fusion / buffer reuse | 将部分逻辑 ATen 中间值融合或复用，减少真实物化 | fusion 受 layout、mutation、consumer 和 kernel 边界限制 |
| memory budget | 在 0 到 1 间调节 saved activation 与重计算的折中 | budget 约束 saved-for-backward activation，不是 end-to-end peak 上限 |
| Classic AC / SAC | 用户显式指定 checkpoint 区域或保存策略 | 需要人工边界或 operator policy，缺少端到端 peak-aware 搜索 |

因此，更准确的结论不是“min-cut 没有意义”，而是：PyTorch min-cut 的目标函数与用户通常关心的
整步显存峰值之间存在缺口。当固定 frontier、重算波、编译物化或 backward transient 接管峰值时，
继续降低 saved activation 可能只增加计算时间，而不降低 peak。

## 5. Budget 是否可调

PyTorch 中的 `activation_memory_budget` 可以通过内部配置调节：

```python
import torch._functorch.config

torch._functorch.config.activation_memory_budget = 0.5
```

在本地 PyTorch 2.13.0+cu130 环境中，其默认值为 `1.0`，源码要求范围为 `[0, 1]`。通常可理解为：

| budget | 含义 |
|---:|---|
| 1.0 | 偏 runtime，保存更多值，减少重算 |
| 0.5 | 中间折中点，但可选 tensor 集合是离散的，不保证连续变化 |
| 0.0 | 尽可能少保存 activation，接近最激进重算 |

需要强调的是，该配置仍属于 PyTorch 编译栈的实验性内部接口。它适合研究和系统原型使用，但不应
被表述为稳定的公开峰值预算 API。

## 6. 对 PeakAware 的论文动机

上述现象为 PeakAware 提供了直接研究动机：现有 min-cut 主要优化 saved activation，而训练者
真正关心的是端到端物理峰值和吞吐。一个更实用的重计算系统需要把 SAVE 选择、重算闭包、
backward 执行顺序、compiler materialization、固定显存底座和 runtime profile 纳入同一个
peak-aware 反馈回路。

PeakAware 的目标不是简单替代 PyTorch min-cut，而是在其基础上补齐缺失的峰值语义：

1. 用统一 storage/liveness 模型重新解释 saved activation 的真实释放量。
2. 评估候选 SAVE 集合在 backward 中引入的 recomputation wave 和 transient peak。
3. 区分 after-FW gain、overall peak gain 和 step-time cost，避免把无效重算当成收益。
4. 对 budget=0、SAC、Classic AC 和 min-cut 等策略给出可比较的端到端诊断。
5. 当重算使 peak 不降反升或时间明显升高时，自动回退、调整边界或停止进一步 DROP。

因此，论文中的问题定义可以表述为：

```text
给定 PyTorch 编译训练图和目标硬件，如何选择跨 FW/BW 边界保存的前向值，使训练 step 的真实
峰值显存满足预算，并在该约束下最小化额外重计算时间？
```

这个定义把“减少 saved activation”从最终目标降为中间手段，把“端到端 peak + step time”作为
评价重计算策略是否有意义的核心标准。
