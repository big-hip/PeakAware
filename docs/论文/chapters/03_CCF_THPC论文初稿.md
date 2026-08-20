# PeakAware: 面向 PyTorch AtenIR 的端到端峰值感知选择性重计算仿真与执行

状态：CCF-THPC 投稿初稿。本文按终稿叙事组织，实验数字优先使用当前冻结证据 EV-20；后续补实验时，应以新的 frozen evidence 更新表格和结论。

## 摘要

激活重计算是缓解深度神经网络训练显存压力的常用技术。现有 PyTorch 编译训练路径已经将重计算从用户手写 checkpoint 扩展到 AOTAutograd joint graph、min-cut partitioner、Selective Activation Checkpointing 以及 memory budget 等自动化机制。然而，这些机制通常以 saved-for-backward tensor 或局部图切分代价为优化代理，而真实训练是否 OOM 取决于 forward、backward 和 optimizer 共同形成的完整训练步物理峰值。我们观察到，在 budget 约束下，min-cut 式选择性重计算即使减少了前向保存值，也可能无法降低反向阶段峰值，原因在于反向重计算瞬时量、storage alias、梯度/优化器固定边界以及编译器 fusion/workspace 会共同改变活跃显存曲线。

本文提出 PeakAware，一种面向 PyTorch AtenIR 的端到端峰值感知选择性重计算仿真与执行框架。PeakAware 使用 FakeTensor 与 AOTAutograd 捕获低开销联合训练图，构建 storage-aware AtenIR 表征；结合 Costmodel 估计算子时间，并使用张量生命周期、saved tensor 边界、重计算临时值和固定训练状态生成 L2/L2.5 级显存占用时间线；在此基础上通过 D0--D5 反事实诊断解释 saved-memory 目标与 physical peak 的偏差，并以 Top-K 实测闭环选择满足显存预算的可执行 SAVE/RECOMPUTE 计划。

在 RTX A6000、PyTorch 2.13.0+cu130 环境下，我们在 ResNet-50、ViT-B/16、BERT-like-2L-64H 和 GPT2-like-2L-64H 四个 workload 上评估 PeakAware。冻结证据 EV-20 覆盖 AOT eager 与 Inductor、diagnostic hints on/off、5 个相对预算比例和 5 次矩阵重复，共 400 条记录。结果显示，PeakAware 在所有成功记录中没有出现预算违约；相对 all-save 平均降低实测峰值 37.3 MB，平均 samples/s 提升 8.99%；相对 all-save、block checkpoint proxy、greedy 和 torch min-cut proxy，PeakAware 在同等或更低实测峰值下获得更好的吞吐权衡。进一步分析表明，原始静态峰值预测平均相对误差为 84.5%，经 all-save residual/phase 校准后降至 6.54%，说明纯 saved-memory 代理不足以解释真实峰值，而生命周期和 phase-aware 校准是必要的。

## 1 引言

大模型训练的显存瓶颈限制了 batch size、sequence length 和模型规模。Activation checkpointing 通过丢弃部分前向中间值并在反向阶段重算它们，以额外计算换取显存节省。早期 checkpointing 方法通常依赖用户按 layer 或 block 手工划分；现代 PyTorch 编译栈中，AOTAutograd 可以捕获 joint forward/backward graph，min-cut partitioner 可以在图上自动选择保存或重算的值，Selective Activation Checkpointing 与 Memory Budget API 进一步将策略控制暴露给用户。

然而，用户真正关心的问题通常不是“少保存了多少 tensor”，而是“这个训练 step 是否低于显存预算”。这两个目标并不等价。完整训练峰值由以下部分共同决定：

```text
Peak = max_t(
    parameter/buffer storage
  + saved activations live at t
  + recomputed tensors live at t
  + gradients materialized by t
  + optimizer state and temporaries
  + compiler/kernel workspace
  + allocator/runtime residual
)
```

在 residual network 或 Transformer block 中，min-cut 可能选择保存昂贵 matmul 输出、重算便宜 pointwise 或 residual add。单个 add 的重算代价很小，但沿着长残差链连续重建会在反向开始处形成 rematerialization wave，使大量临时值同时存活。此时 saved activation 下降并不必然转化为 backward peak 下降，甚至可能只是把峰值从 after-FW 迁移到 backward 或 optimizer 阶段。

本文的核心观点是：面向 PyTorch AtenIR 的重计算研究，应从 saved-memory 代理目标升级为 complete-step physical peak 目标。为此，需要同时回答三个问题：

1. 如何低开销捕获 PyTorch 真实编译训练图，避免对每个模型手工建模？
2. 如何在 AtenIR 粒度上同时估计算子执行时间和张量生命周期显存？
3. 如何把仿真结果下发为真实可执行计划，并用实测闭环校正静态模型误差？

我们提出 PeakAware。与只报告 profiler 数字不同，PeakAware 从 AtenIR 图中恢复依赖、storage、saved tensor 和重计算边界，生成可解释的内存占用折线图；与只做静态优化不同，PeakAware 将候选 SAVE/RECOMPUTE 集合 lowering 为 AOT FW/BW executable，并以 Top-K 实测选择满足预算的计划。

本文贡献如下：

1. 提出面向 PyTorch AtenIR 的低开销重计算表征框架，利用 FakeTensor/AOTAutograd 捕获 shape、dtype、layout、alias 和 joint graph 依赖，避免逐模型手工建模。
2. 提出 Costmodel + 生命周期联合仿真方法，以 Costmodel 对齐算子时间，以 L2/L2.5 生命周期模型生成 forward/backward/optimizer 显存曲线。
3. 揭示 budget-constrained min-cut 在反向峰值下降上的局限，并通过 D0--D5 反事实诊断分解 storage alias、recompute transient、fixed frontier、workspace 和 runtime residual 等误差来源。
4. 构建从图捕获、候选搜索、计划下发、正确性验证到完整训练步测量的 PeakAware 闭环，在冻结 EV-20 实验中验证预算安全性、吞吐权衡和预测校准效果。

## 2 背景与动机

### 2.1 PyTorch 编译训练与 AtenIR

PyTorch 2.x 编译训练通常经过动态图捕获、AOTAutograd 前后向联合图生成、Inductor lowering 和后端 kernel 调度。AtenIR 保留 PyTorch 算子语义、张量元信息和依赖关系，是连接用户模型、重计算策略和后端编译行为的合适层级。相比逐模型手写解析，AtenIR 捕获可以让 ResNet、ViT、BERT-like、GPT-like 等模型进入统一分析流程。

FakeTensor 为该流程提供了低开销图捕获能力。FakeTensor 不持有真实数据 storage，而携带 shape、stride、dtype、device 等元信息，适合在不真实分配大规模激活的情况下构造图、推导 tensor bytes 和生成 Costmodel signature。需要强调的是，FakeTensor 不能替代真实 CUDA 执行：它无法暴露缓存分配器碎片、cuDNN/Inductor workspace 或 runtime kernel 行为。因此本文将 FakeTensor 用于 L2/L2.5 静态表征，将实测用于校准和最终验收。

### 2.2 Saved-memory 目标与 Physical Peak 的错位

Activation checkpointing 和 min-cut 的基本目标是决定哪些 forward value 保存到 backward，哪些 value 在 backward 中重算。该目标适合描述图切分，但不足以描述真实训练峰值。完整训练 step 中，after-FW retained memory 只是峰值组成之一。反向阶段的 recompute tensor、梯度逐步物化、optimizer state 和 compiler workspace 都可能成为新的峰值来源。

本地草稿中的 24 层 activation-dominated residual chain 复现了该问题：在 PyTorch 2.13.0+cu130 下，min-cut budget 0.75 与 block AC 的 after-FW 接近，但 backward peak 分别约为 1216.3 MiB 与 327.4 MiB。逐节点分析显示，min-cut 在第一个真正 backward op 前连续重建多层 residual/pointwise 链，形成约 960 MiB 的重计算瞬时活跃集合。这表明 saved tensor 减少不等于 backward peak 下降。

### 2.3 现实解决场景

PeakAware 面向的现实场景是显存预算受限的 PyTorch 训练调优。用户给定模型、输入规模、优化器和显存预算，希望在真实运行前回答：

1. 当前 batch size 或 sequence length 是否可能在目标 GPU 上训练？
2. 哪种重计算策略能满足 physical peak 预算？
3. 如果 min-cut 或 SAC 没有降低峰值，峰值到底出现在 forward、backward 还是 optimizer？
4. 重计算带来的额外时间能否被接受？

传统做法需要对每个候选策略完成编译、预热和训练步测量。PeakAware 的目标是将大部分不可行候选提前排除，只把少量 Pareto Top-K 候选送入真实测量，从而把调参过程从反复 OOM 试错变成可诊断的时间-内存联合仿真。

## 3 问题定义

给定静态 shape 单设备训练任务 `W`、模型参数 `P`、输入 `X`、优化器 `O`、候选 SAVE 集合 `S` 和物理显存预算 `B`，系统需要选择一个可执行计划 `e`，满足：

```text
measured_peak(e, W, X, O) <= B
```

并在满足预算的候选中优化稳态 step time 或 throughput。本文关注 allocated physical peak，即 `torch.cuda.max_memory_allocated()` 对应的分配峰值；reserved memory 仅作为 allocator 行为讨论，不作为主指标。

本文默认边界：

- 静态 shape、单设备训练；
- optimizer 位于 joint graph 外，但纳入完整训练 step 测量；
- 不覆盖 FSDP/ZeRO、offload、activation compression、dynamic shape、多机通信联合优化和任意 backward schedule 搜索；
- Chakra ET/ASTRA-sim 仅作为后续互操作路径，不作为当前 AtenIR 级内存仿真的主工具。

## 4 PeakAware 系统设计

PeakAware 包含六个模块。

### 4.1 低开销 AtenIR 图捕获

PeakAware 对用户模型执行一次 joint graph capture，获得 forward/backward 依赖和 tensor metadata。FakeTensor 用于生成元信息，避免真实激活分配；AOTAutograd 提供前后向边界；Inductor 捕获路径用于获得 fusion 和 scheduler 侧线索。

捕获阶段输出：

- Aten/FX node 列表；
- tensor shape、dtype、device、stride；
- view/storage alias 信息；
- forward output 与 backward input 边界；
- saved tensor/residual 输出；
- graph key 与 guard 信息。

该设计避免了逐模型手工建模。模型结构变化时，只要 PyTorch 编译栈可以捕获相应 AtenIR，PeakAware 即可复用同一套生命周期和 Costmodel 流程。

### 4.2 Storage-aware IR 与生命周期分析

PeakAware 将 AtenIR 转为 storage-aware training IR。每个逻辑 value 被映射到 storage identity 和 byte size；view、getitem、tuple output 等不一定产生新分配的节点通过 alias 规则处理。生命周期由 def-use 关系和 backward last-use 决定。对于 unknown use，系统采用保守策略，使 tensor 保持 live，避免低估峰值。

L2 内存模型使用事件驱动 live-range 仿真：

```text
memory_L2(t) =
    fixed_parameter_bytes
  + buffer_bytes
  + gradient_frontier(t)
  + optimizer_frontier(t)
  + live_activation_storage(t)
  + saved_tensor_storage(t)
  + recompute_transient_storage(t)
```

L2.5 在 L2 上加入 fusion-aware 修正：融合组内部不会物化到 global memory 的中间值可以被消除；安全 buffer reuse 可以减少重复分配；unknown/materializing op 作为 barrier 处理。该层继承毕业论文中的 L2/L2.5 经验，但在 PeakAware 中进一步服务于预算选择和根因诊断。

### 4.3 Costmodel 时间建模

Costmodel 用于建立算子时间轴。给定 Aten op、shape、dtype、layout 和 hardware profile，Costmodel 输出 operator latency 或候选 cost ranking。本文不把 Costmodel memory traffic 直接等同为 live memory；只有当某个 op model 的 memory 字段具有 workspace 语义时，才作为 `workspace_bytes(t)` 叠加到生命周期曲线。

时间轴定义为：

```text
time(t) = cumulative_costmodel_latency(op_0 ... op_t)
```

实际使用中，PeakAware 按 phase 进行校准：

```text
FW raw span -> measured FW time
BW raw span -> measured BW time
OPT raw span -> measured optimizer time
```

这种做法承认纯 analytical Costmodel 难以完全模拟 PyTorch runtime、kernel launch、fusion 和调度噪声，同时保留了候选间相对时间排序和内存折线图定位能力。

### 4.4 D0--D5 反事实诊断

为解释 saved-memory 与 physical peak 的偏差，PeakAware 设计 D0--D5 分层诊断：

- D0：logical saved bytes；
- D1：storage-normalized saved bytes；
- D2：加入 after-FW retained 与基本 liveness；
- D3：加入 backward recompute transient 和 fixed frontier；
- D4：加入 compiler/runtime residual correction；
- D5：加入 measured Top-K validation。

每个候选都会输出 expectation gap、realization gap、dominant phase 和 root cause label，例如 `REMATERIALIZATION_WAVE`、`FIXED_BACKWARD_FRONTIER`、`WORKSPACE_GROWTH`、`COST_MODEL_MISRANK`。这些诊断结果既用于论文解释，也可生成 repair hint。

### 4.5 候选搜索与 Top-K 实测闭环

PeakAware 使用 all-save、block checkpoint、min-cut、greedy drop 等策略作为 seed，生成多个 SAVE/RECOMPUTE 候选。静态仿真阶段根据 estimated peak、estimated step time 和 budget feasibility 筛选 Pareto Top-K。随后系统将候选 lowering 为 AOT FW/BW executable，执行 correctness check 和完整 step measurement。最终选择满足预算且吞吐最优的候选。

这使 PeakAware 区别于只做离线估算的工具：候选不是图上的标签，而是被实际下发、执行和测量的训练计划。

### 4.6 Chakra ET 与 ASTRA-sim 边界

Chakra ET 是 ML workload execution trace 格式，ASTRA-sim 可消费 Chakra trace 进行系统级仿真，尤其适合分布式训练中的 compute/communication/network 分析。本文不将 ASTRA-sim 作为 AtenIR 算子级分析器。若后续需要接入，正确路径是：

```text
PeakAware AtenIR / JointTrainingIR
  -> Chakra ET compute/memory/communication nodes
  -> ASTRA-sim workload layer
  -> system-level simulation
```

AtenIR op 语义、算子时间、生命周期内存和重计算策略仍由 PeakAware 建模。

## 5 实现

当前实现包含两个层次。

第一层是原始 toolkit，用于建立 ATenIR 显存表征基础。它支持 AOTAutograd/Inductor 捕获、runtime phased profiling、L1/ShapeSum/L2/L2.5/L3 多层估算，以及多模型实验图表生成。毕业论文实验证明，L2/L2.5/L3 能显著优于简单 shape 累加。

第二层是 PeakAware 系统。其核心目录包括：

- `peakaware/capture/`：joint graph capture、fake input、graph key；
- `peakaware/ir/`：storage-aware IR、alias 与 legality；
- `peakaware/memory/`：liveness、timeline、fixed frontier 和 simulator；
- `peakaware/cost/`：Costmodel adapter、ProfileDB 和 calibration；
- `peakaware/search/`：candidate generation、repair、Pareto 和 exact search；
- `peakaware/partition/`：AOT partition lowering 与 verifier；
- `peakaware/runtime/`：完整训练步 executor、measurement 和 isolation；
- `peakaware/publication/` 与 `scripts/`：论文 artifact、baseline、figures、tables 和 evidence gate。

Costmodel 位于 `PeakAware/Costmodel`，包含 analytical op model、hardware config 和部分 fused/custom op cost model。当前论文中它主要作为算子时间和候选 ranking 的来源；内存 live bytes 仍以生命周期仿真为主。

## 6 实验设计

### 6.1 Workload 与环境

EV-20 使用四个 workload：

| Workload | 结构 | 参数量 | 输入 | 优化器 |
|---|---:|---:|---|---|
| BERT-like-2L-64H | 2 层、hidden 64、4 heads | 2.09M | `1x32` token | AdamW |
| GPT2-like-2L-64H | 2 层、hidden 64、4 heads | 6.54M | `1x32` token | AdamW |
| ResNet-50 | 3/4/6/3 blocks | 23.53M | `1x3x224x224` | SGD |
| ViT-B/16 | 12 层、hidden 768、12 heads | 85.81M | `1x3x224x224` | AdamW |

实验环境为 PyTorch 2.13.0+cu130、CUDA 13.0、NVIDIA RTX A6000。每个 workload 覆盖 AOT eager 与 Inductor paired full，预算比例为 all-save measured peak 的 50%、65%、80%、95% 和 100%，并进行 5 次矩阵重复。当前 EV-20 的 measurement repeats 和 warmup 口径应在 camera-ready 前与最终协议统一；若后续补跑采用更高 repeats/warmup，应以新 frozen artifact 替换本文数字。

### 6.2 Baseline

本文比较以下 baseline：

- All-save：不进行 activation recomputation；
- Block checkpoint：block 级 checkpoint baseline；
- Torch min-cut proxy：PyTorch min-cut/memory-budget 风格 baseline，在当前 EV-20 artifact 中仍按 proxy provenance 标注；
- Greedy：按估计收益/代价生成的贪心候选；
- SAC：Selective Activation Checkpointing，用于外部策略对比；EV-20 中 external SAC matched rows 为 0，正式对比需要补齐或单独报告；
- PeakAware：本文方法，从候选中选择满足 physical budget 且吞吐更优的 executable。

所有 baseline 都必须报告 runtime identity、plan id、correctness marker、失败原因和 budget violation。proxy baseline 不得在论文中冒充真实 PyTorch 官方实现。

### 6.3 指标

主要指标包括：

- `allocated physical peak`：完整 step 的 `torch.cuda.max_memory_allocated()`；
- `phase peak`：FW/BW/optimizer 各阶段峰值；
- `budget violation rate`：实测峰值超过预算比例；
- `samples/s` 与 `step time`；
- `prediction MRE`：估计峰值相对误差；
- `phase classification accuracy`：预测峰值阶段是否正确；
- `candidate ranking`：Costmodel 估计时间与实测时间的 Spearman/Kendall；
- `root cause counts`：诊断出的峰值误差来源。

### 6.4 消融

消融实验围绕四个问题：

1. 去掉生命周期，只用 ShapeSum 或 logical saved bytes 是否足够？
2. 去掉 recompute transient，是否会高估重计算收益？
3. 去掉 fixed frontier/optimizer frontier，是否会误判小 batch Adam 场景？
4. 去掉 Top-K 实测校准，Costmodel ranking 是否足以选择最优候选？

对应模块为 L2、L2.5、D0--D5、diagnostic hints 和 Costmodel/ProfileDB。

## 7 实验结果

### 7.1 预算满足与吞吐

EV-20 共生成 400 条记录，其中 200 条成功执行、200 条失败记录保留在分母。成功记录中 budget violation 为 0，说明 PeakAware 选中的 executable 在当前 workload 和预算口径下满足 physical peak 约束。

相对 all-save，PeakAware 平均降低实测峰值 37.3 MB，平均 samples/s 提升 8.99%。baseline comparison 显示，相对 all-save、block checkpoint proxy、greedy 和 torch_min_cut proxy，PeakAware 分别获得 1.09x、1.14x、1.15x 和 1.30x 的平均 samples/s speedup。该结果说明，在满足预算的前提下，PeakAware 可以选择更优的时间-内存折中点。由于当前 min-cut/block 口径仍带 proxy 标记，正式论文应将结论限定为“相对当前实现的可执行 proxy baseline”，或在补齐真实 baseline 后更新表述。

需要注意的是，峰值降低并非所有点严格成立。此前主矩阵中也观察到 fixed/optimizer frontier 会遮蔽 activation saving。因此本文报告 weak Pareto、strict Pareto、预算满足率和失败记录，而不是只报告平均收益。

### 7.2 Min-cut Budget 下反向峰值不降

在 budget-constrained min-cut 场景中，保存 tensor 数量减少并不保证 backward peak 下降。草稿中的 residual chain 证据显示，min-cut 可形成长 rematerialization wave，使 backward 初期堆积大量重计算临时值。EV-20 的 root cause 统计中，`REMATERIALIZATION_WAVE` 出现 115 次，是最主要的诊断标签；`WORKSPACE_GROWTH` 出现 36 次，`FIXED_BACKWARD_FRONTIER` 出现 15 次。

这证明 PeakAware 的研究重点不是替代所有 min-cut，而是指出：当优化目标仅约束 saved-memory 代理时，可能无法约束用户关心的 complete-step physical peak。Peak-aware lifetime timeline 能把该问题定位到具体 phase 和 live set。

### 7.3 峰值预测精度

原始静态 simulation 的平均相对误差为 84.5%，P50 为 49.7%，within-10% 比例仅 3.5%。这说明只靠未校准静态模型不足以直接作为最终决策依据。

加入 all-save residual/phase 校准后，平均相对误差降至 6.54%，P50 为 0.11%，within-10% 比例达到 84.25%。该结果支持本文的系统设计：静态生命周期模型用于解释和筛选，Top-K 实测闭环用于最终验收。

D0--D5 分层结果进一步表明，D0/D1 的 logical/storage saved bytes 误差很高；加入 fixed frontier 和 runtime correction 后，误差明显下降。D4/D5 的精度不能被解释为纯静态模型能力，而应表述为 compiler/runtime-aware 校准后的预测能力。

### 7.4 Costmodel 排序与 Top-K 必要性

EV-20 中候选时间排序统计包含 1180 个可比较 pair，mean Spearman 为 0.459，Kendall 为 0.402，mean best-rank error 为 0.865。Costmodel 提供了有用但不完美的时间排序，因此 PeakAware 不直接信任单一静态最优候选，而是保留 Pareto Top-K 并进行实测校正。

### 7.5 Diagnostic Hints 消融

Hints on/off 的 100 个配对中，30 个 pair 改善，28 个 pair 退化，42 个 inconclusive。diagnostic hints 改变了候选匹配情况，但没有稳定改变搜索顺序，repair success count 仍为 0。这一结果说明当前 hints 更适合作为诊断解释与候选过滤信息，而不能在正文中夸大为稳定提升搜索质量的关键算法。

## 8 讨论

### 8.1 为什么 THPC 需要这类工作

CCF-THPC 关注高性能计算系统、性能建模、资源管理和可复现实验。PeakAware 的贡献点不在于提出新的深度学习模型，而在于面向真实 PyTorch 编译训练系统建立资源建模闭环：自动捕获图、模拟时间-内存曲线、诊断目标错位、下发可执行计划并以 artifact gate 验证。这比单纯算法优化更贴近系统论文。

### 8.2 与毕业论文工作的关系

毕业论文已经建立了面向 PyTorch AtenIR 的重计算抽象表征技术，包含 L1/ShapeSum/L2/L2.5/L3 多层静态显存估计。本文在此基础上将目标从“峰值估计”推进到“预算约束下的计划选择和执行闭环”：Costmodel 对齐执行时间，生命周期负责显存曲线，D0--D5 解释误差来源，Top-K 实测保证最终计划满足 physical peak。

### 8.3 局限

本文仍有局限：

- 当前只覆盖静态 shape、单设备训练；
- Inductor 路径依赖 PyTorch 内部 ABI，跨版本稳定性需要持续验证；
- Costmodel 目前对候选时间排序仍有误差，必须依赖 Top-K 实测校准；
- diagnostic hints 在当前 EV-20 中没有稳定改善搜索顺序；
- EV-20 的部分 baseline identity 为 proxy/unknown，真实 PyTorch min-cut、block AC 和 SAC matched 对比仍需补强；
- 当前 EV-20 measurement repeats/warmup 与更严格的最终协议文本存在差异，camera-ready 前应统一；
- Chakra/ASTRA-sim 尚未进入主实验，只能作为未来分布式互操作方向；
- BERT/GPT workload 是 small-like 配置，不能外推为标准 BERT-Base/GPT-2 的完整训练结论。

## 9 相关工作

Activation checkpointing 最早系统化讨论了用额外计算换取训练内存的技术路线。Chen 等提出 sublinear memory cost 的训练方法，奠定了深度网络 rematerialization 的基本问题。Checkmate、Rotor、DTR、Capuchin 等工作从图优化、动态 runtime 或 tensor 管理角度研究重计算、交换和内存规划。与这些工作不同，PeakAware 不把重计算作为抽象计算图上的独立优化问题，而是聚焦 PyTorch AtenIR 和编译训练实际执行中的 physical peak。

PyTorch 官方文档与博客说明了 activation checkpointing、Selective Activation Checkpointing 和 Memory Budget API 的使用方式，也说明 memory budget 主要约束指定区域内 saved-for-backward activation。本文进一步研究该代理目标与完整训练 step allocated peak 的差异。

Chakra ET 与 ASTRA-sim 面向 ML workload trace 和系统级仿真，尤其适合多机通信和网络架构评估。本文可将 AtenIR 转换为 Chakra ET，但不依赖 ASTRA-sim 解释 Aten 算子语义或单卡生命周期显存。

## 10 结论

本文提出 PeakAware，一种面向 PyTorch AtenIR 的端到端峰值感知选择性重计算仿真与执行框架。PeakAware 使用 FakeTensor/AOTAutograd 实现低开销图捕获，以 Costmodel 对齐算子时间，以 L2/L2.5 生命周期模型生成显存曲线，并通过 D0--D5 诊断解释 saved-memory 与 physical peak 的目标错位。实验表明，在 EV-20 覆盖的四个 workload 和五档预算下，PeakAware 能在无预算违约的条件下获得更好的吞吐权衡，并揭示 rematerialization wave、fixed frontier 和 workspace growth 是 budget-constrained min-cut 峰值收益不稳定的重要原因。

未来工作将扩展到 dynamic shape、多卡训练、distributed trace export、真实 allocator replay 和更强 Costmodel/ProfileDB 校准，以进一步提升仿真和实测对齐精度。

## 参考文献与资料

[R1] PyTorch Foundation. Current and New Activation Checkpointing Techniques in PyTorch. https://pytorch.org/blog/activation-checkpointing-techniques/

[R2] PyTorch Documentation. Fake tensor. https://docs.pytorch.org/docs/2.13/torch.compiler_fake_tensor.html

[R3] PyTorch Documentation. torch.export. https://docs.pytorch.org/docs/stable/export.html

[R4] T. Chen et al. Training Deep Nets with Sublinear Memory Cost. arXiv:1604.06174. https://arxiv.org/abs/1604.06174

[R5] P. Jain et al. Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization. https://arxiv.org/abs/1910.02653

[R6] A. Kirisame et al. Dynamic Tensor Rematerialization. https://arxiv.org/abs/2006.09616

[R7] L. Beaumont et al. Rotor: A Fast and Efficient Algorithm for Checkpointing in Deep Learning. https://hal.inria.fr/hal-02352969

[R8] MLCommons Chakra. Execution Trace format. https://github.com/mlcommons/chakra

[R9] ASTRA-sim documentation. Workload layer. https://astra-sim.github.io/astra-sim-docs/workload-layer/overview.html

[R10] ASTRA-sim GitHub repository. https://github.com/astra-sim/astra-sim

[R11] 本地证据：`PeakAware/docs/论文/evidence/00_论点证据账本.md`

[R12] 本地证据：`PeakAware/artifacts/paper_full_matrix_combined_paired_5budget_5pass_r1/`

[R13] 本地证据：`唐成祥毕业论文终稿.pdf`
