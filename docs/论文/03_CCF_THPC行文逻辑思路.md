# CCF-THPC 行文逻辑思路与监督 Review 清单

状态：写作控制文档。目标是把“面向 PyTorch AtenIR 的重计算内存表征”升级为一篇适合 CCF-THPC 的系统论文，同时保留后续补实验入口。

## 1. 一句话主线

不要把论文写成“又一个重计算算法”。主线应是：

> 现有 saved-memory/min-cut 代理目标不能可靠约束完整训练步 physical peak；本文基于 PyTorch AtenIR 自动捕获、Costmodel 时间建模和生命周期显存仿真，构建可诊断、可下发、可实测校准的 PeakAware 选择性重计算闭环。

更短版本：

> 从 saved activation budget 到 complete-step physical peak budget。

## 2. 论文故事线

### 第一幕：现实痛点

用户希望在固定 GPU 显存下训练更大 batch 或更长 sequence。传统做法是反复尝试 checkpoint/min-cut/SAC，然后编译、预热、实测、OOM，再改策略。这个流程慢，而且 profiler 只能告诉用户峰值发生了，不能解释为什么某个 budget 降低后 backward peak 没降。

### 第二幕：目标错位

PyTorch min-cut 和 memory budget 主要围绕 saved-for-backward values 建模。真实训练峰值却是时间轴最大 live set，包含 activation、recompute transient、gradient、optimizer、workspace 和 allocator residual。因此 saved bytes 下降可能只迁移峰值。

### 第三幕：AtenIR 是合适层级

逐模型手工建模不可扩展。FakeTensor + AOTAutograd + Inductor 捕获可以自动获得 AtenIR/FX 图、shape、dtype、alias 和 forward/backward 边界。这样模型从 CNN 换到 Transformer，不需要重写分析器。

### 第四幕：时间和内存分别建模

时间由 Costmodel/ProfileDB 主导，按 phase 对齐实测；内存由生命周期仿真主导，Costmodel 只在 workspace 语义明确时补充。不要把 Costmodel 的 memory traffic 误写成 live allocated memory。

### 第五幕：解释后再选择

D0--D5 把误差从 logical saved bytes 逐步加到 storage、liveness、recompute transient、fixed frontier、compiler/runtime correction。诊断不仅服务于解释，也服务于候选 repair 和 Top-K 筛选。

### 第六幕：可执行闭环

最终计划必须是被 lowering、correctness check 和完整 step measurement 验证过的 executable。论文要反复强调：PeakAware 不是只输出图上标签，而是输出可执行计划。

## 3. CCF-THPC 适配重点

THPC 更容易接受的关键词：

- performance modeling；
- memory simulation；
- resource-constrained training；
- compiler/runtime co-design；
- profiling calibration；
- reproducible artifact；
- system evaluation；
- budget feasibility。

不宜作为主线的关键词：

- 单纯 activation checkpointing；
- 单纯 min-cut 算法；
- 单纯 PyTorch bug report；
- 单纯大模型训练技巧。

标题建议：

1. 面向 PyTorch AtenIR 的端到端峰值感知选择性重计算仿真与执行
2. PeakAware: 基于 AtenIR 生命周期分析的 PyTorch 编译训练峰值显存预算闭环
3. 从 Saved-Memory 到 Physical-Peak: PyTorch AtenIR 级选择性重计算建模与执行

## 4. 章节安排

推荐投稿结构：

```text
摘要
1 引言
2 背景与动机
3 问题定义
4 PeakAware 系统设计
5 实现
6 实验设计
7 实验结果
8 讨论与局限
9 相关工作
10 结论
```

引言必须回答：

- 用户为什么关心 physical peak，而不是 saved bytes？
- 为什么 min-cut budget 下 backward peak 可能不降？
- 为什么 AtenIR 捕获能避免逐模型建模？
- 为什么必须 Costmodel + liveness + measurement，而不是单一 profiling 或单一静态模型？

## 5. 现实解决场景

必须写进引言或动机：

1. 显存预算搜索：给定 GPU 和 batch/sequence，提前判断是否可训练。
2. 策略选择：在 all-save、block AC、SAC、min-cut 和 greedy 之间选择满足 budget 的计划。
3. 峰值诊断：当 budget 降了但 backward peak 没降时，定位 rematerialization wave、fixed frontier 或 workspace。
4. 低成本迁移：同一 pipeline 自动处理 ResNet、ViT、BERT-like、GPT-like，避免逐模型建模。

## 6. Baseline 设计

论文 baseline 分两层。

系统策略 baseline：

- All-save；
- Block checkpoint；
- PyTorch min-cut / memory-budget；
- SAC；
- Greedy；
- PeakAware。

仿真模型 baseline：

- L1 config formula；
- ShapeSum；
- L2 live-range；
- L2.5 fusion-aware；
- D0--D5 diagnosis；
- calibrated Top-K measurement。

监督要求：

- 如果 baseline 是 proxy，全文必须写 proxy。
- 不能把 BERT-like/GPT2-like 写成标准 BERT-Base/GPT-2。
- 不能只展示成功记录，失败、OOM、timeout、correctness failure 必须保留在分母。

## 7. 消融设计

必须围绕贡献拆模块：

1. 无生命周期：ShapeSum 或 logical saved bytes。
2. 无 storage alias：看 view/shared storage 是否造成重复计数。
3. 无 recompute transient：检验是否高估重计算收益。
4. 无 fixed frontier：检验 optimizer/gradient 是否遮蔽 activation saving。
5. 无 fusion-aware：比较 L2 与 L2.5。
6. 无 runtime calibration：比较 raw simulation 与 calibrated prediction。
7. 无 Top-K：检验 Costmodel 排序是否足够。
8. 无 diagnostic hints：检验 hints 是否改善成功率、吞吐或候选排序。

当前 EV-20 中 hints 结果不够漂亮，应写成“诊断解释价值强，搜索改善尚不稳定”，不要硬吹。

## 8. 当前材料如何合并

### 毕业论文

可继承：

- PyTorch AtenIR 重计算抽象表征；
- FakeTensor/AOTAutograd/Inductor 捕获；
- L1/ShapeSum/L2/L2.5/L3 多层显存估计；
- GPT-2/LLaMA/Mistral 风格模型实验；
- phase peak 与 L2.5 消融图。

升级方式：

- 毕业论文重在“估算准确性”；
- THPC 论文重在“预算约束下的系统闭环、诊断和执行”。

### toolkit

定位为方法基础与早期验证平台。可用于说明：

- ShapeSum 平均 MRE 远高于 L2/L2.5；
- L2/L2.5 证明 live-range 与 fusion-aware 建模必要；
- phase profiler 证明 peak 可从 FW 迁移到 BW/OPT。

### PeakAware

定位为投稿主系统。可用于说明：

- capture -> IR -> memory/cost -> search -> AOT lowering -> runtime measurement；
- D0--D5 根因诊断；
- Top-K 实测校准；
- evidence gate 与 artifact manifest。

### Costmodel

定位为时间轴与候选排序模块。注意：

- Costmodel 可估算算子时间；
- memory traffic 不直接等价于 live allocation；
- Costmodel 排序当前不是完美的，因此 Top-K calibration 是必要机制。

## 9. 关键实验图表

建议最终图表：

- F1 系统架构闭环；
- F2 saved gain vs physical peak gain；
- F3 不同 budget 下 min-cut backward peak 不降示例；
- F4 no recompute/min-cut/block AC/PeakAware 的显存折线图；
- F5 raw vs calibrated prediction parity；
- F6 D0--D5 waterfall；
- F7 Pareto peak-throughput；
- F8 Costmodel ranking Top-K regret；
- F9 hints on/off 消融；
- T1 workload manifest；
- T2 baseline identity/provenance；
- T3 budget satisfaction；
- T4 prediction error；
- T5 root cause frequency；
- T6 runtime/cost overhead。

## 10. 监督 Review 打回清单

以下任一项出现，应打回修改：

- 把 saved-memory budget 写成 complete-step physical memory budget。
- 声称 min-cut 错误或所有 min-cut 都会导致峰值反升。
- 声称首次发现 residual chain 重计算问题。
- 把 proxy baseline 写成真实 PyTorch 官方 baseline。
- 把 calibrated 误差写成纯静态模型误差。
- 只报平均值，不报失败、负结果、budget violation 和 phase migration。
- 用旧 invalid 矩阵支撑效果 claim。
- 把 ASTRA-sim 写成能直接分析 AtenIR 算子的工具。
- 把 Costmodel memory traffic 当作 allocated live memory。
- 把 small BERT/GPT workload 外推到标准 BERT-Base/GPT-2。
- 结论中写“全面优于 SAC/min-cut/AC”。
- 不忽略 EV-20 中 `proxy: 400, real: 0, unknown: 400` 的 baseline identity 风险。
- 不忽略 EV-20 的 `measurement_repeats=5`、warmup 口径与部分最终协议文档不一致的问题。
- 不给 artifact、manifest、checksum、环境和版本。

## 11. 可以保留的强表述

可以写：

- PeakAware 将 saved-memory 代理目标转化为 complete-step physical peak 预算下的可执行选择问题。
- FakeTensor/AOTAutograd 捕获避免了逐模型手工建模。
- Costmodel 负责时间轴，生命周期负责显存曲线，Top-K 实测负责最终校准。
- 在 budget-constrained min-cut 中，saved activation 降低不必然带来 backward peak 下降。
- EV-20 显示 PeakAware 在成功记录中没有预算违约，并在同等或更低峰值下改善吞吐权衡。

必须带限定：

- “在 EV-20 覆盖的 workload/backend/budget 口径下”；
- “当前为静态 shape、单设备训练”；
- “对 BERT-like/GPT2-like small workload 成立”；
- “校准后误差不是纯静态精度”。

## 12. 参考资料路径

本地资料：

- `唐成祥毕业论文终稿.pdf`
- `docs/18-thesis-writing-guide.md`
- `toolkit_examples/outputs/`
- `PeakAware/docs/论文/00_论文总控.md`
- `PeakAware/docs/论文/01_论点与故事线.md`
- `PeakAware/docs/论文/evidence/00_论点证据账本.md`
- `PeakAware/artifacts/paper_full_matrix_combined_paired_5budget_5pass_r1/`
- `草稿/mincut_backward_peak_verified/`
- `草稿/mincut_favorable_ablation/`

外部资料：

- PyTorch activation checkpointing blog: https://pytorch.org/blog/activation-checkpointing-techniques/
- PyTorch FakeTensor docs: https://docs.pytorch.org/docs/2.13/torch.compiler_fake_tensor.html
- Training Deep Nets with Sublinear Memory Cost: https://arxiv.org/abs/1604.06174
- Checkmate: https://arxiv.org/abs/1910.02653
- Dynamic Tensor Rematerialization: https://arxiv.org/abs/2006.09616
- Chakra: https://github.com/mlcommons/chakra
- ASTRA-sim: https://github.com/astra-sim/astra-sim

## 13. 后续补实验优先级

1. 补真实 PyTorch min-cut / memory-budget baseline 身份验证。
2. 补 SAC matched matrix，使 external SAC 不再是 0 matched。
3. 扩大 BERT/GPT 到更标准规格，或全文坚持 small-like 命名。
4. 统一最终测量协议，建议补 repeats>=20、warmup>=5 的主矩阵或明确 EV-20 的轻量口径。
5. 补 Costmodel 单算子和子图时间误差表。
6. 输出 per-op/per-phase memory timeline 曲线，展示 backward peak 不降的具体 live set。
7. 对 Chakra ET export 做可选附录，不进入主贡献。
