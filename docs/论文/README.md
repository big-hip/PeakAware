# PeakAware 论文工程索引

本目录是论文的控制层，不替代 `docs/00_*.md` 到 `docs/10_*.md` 的工程规范。工程文档回答
“系统应如何实现”，本目录回答“论文允许声称什么、用什么证据、按什么顺序完成”。

## 1. 唯一主线

> PyTorch saved-memory 优化目标与完整训练 step 的物理峰值存在错位；PeakAware 用
> storage/phase/fixed/compiler-aware 模型解释收益缺口，再以诊断反馈修复 SAVE 集合，并把计划
> 下发为可验证的 AOT FW/BW executable，在物理预算下选择峰值与吞吐更优的方案。

任何章节、图表或答辩表述偏离该主线，都必须在 `00_论文总控.md` 中说明原因。

## 2. 推荐阅读顺序

1. [论文总控](00_论文总控.md)：范围、冻结规则、P0 gate 和当前状态。
2. [论点与故事线](01_论点与故事线.md)：核心论点、创新点、摘要与引言叙事。
3. [术语与口径](02_术语与口径.md)：术语、baseline 名称、指标和统计规则。
4. [章节写作蓝图](chapters/00_章节写作蓝图.md)：逐章目标、段落和验收条件。
   - [论文初稿（无结果数字版）](chapters/01_论文初稿_无结果数字.md)：可先写的背景、方法、实现和实验设计骨架。
   - [论文初稿（EV-20 结果版）](chapters/02_论文初稿_EV20结果版.md)：基于冻结主矩阵的中文正文初稿，可作为后续排版和导师修改的起点。
5. [实验补强与测量协议](experiments/00_实验补强与测量协议.md)：正式实验矩阵和执行 gate。
6. [图表与输出规范](figures/00_图表与输出规范.md)：F/T 清单、数据字段和出版格式。
   - [F-1 系统闭环草图](figures/drafts/F1_system_loop.svg)：不依赖冻结结果的架构图草稿。
   - [初稿作图索引](figures/01_初稿作图索引.md)：当前可用于初稿排版的草稿图表路径和引用口径。
7. [论点证据账本](evidence/00_论点证据账本.md)：论点到 artifact 的唯一事实映射。
8. [复现与 Artifact 规范](reproduction/00_复现与Artifact规范.md)：环境、目录、manifest 和校验。
9. [执行排期与验收](schedule/00_执行排期与验收.md)：依赖顺序、里程碑和 Definition of Done。
10. [答辩提纲与风险问答](defense/00_答辩提纲与风险问答.md)：演示结构与高风险问题。
11. [外部参考与写作方法](references/00_外部参考与写作方法.md)：原始来源和可迁移写法。

## 3. 文档所有权

| 内容 | 唯一维护位置 | 其他文件如何使用 |
|---|---|---|
| 题目、范围、状态和 gate | `00_论文总控.md` | 只引用，不复制状态表 |
| 核心论点与声明边界 | `01_论点与故事线.md` | 用 `C-*` 引用 |
| 术语、单位和统计口径 | `02_术语与口径.md` | 不自行创造同义名称 |
| 章节段落结构 | `chapters/` | 不保存实验命令 |
| 实验设计 | `experiments/` | 不写最终结果数字 |
| 图表设计 | `figures/` | 不负责解释结论 |
| 结果数字与 artifact | `evidence/` | 所有正文数字带 `EV-*` |
| 复现命令和目录 | `reproduction/` | 不评价结果优劣 |
| 时间与任务状态 | `schedule/` | 不改变 claim 边界 |

## 4. 当前可运行的论文图表 smoke

正式论文图表仍必须等待 `EV-*` 冻结；当前可用以下命令验证“实验 records -> 图表目录”的生成链路：

```bash
conda run -n torch2.13-gpu env PYTHONPATH=. python scripts/run_experiments.py \
  --tasks tiny_residual_w8 --budget-mib 256 --microbatches 1 --top-k 3 \
  --device cpu --capture-backend fx --diagnostic-hints on \
  --measurement-warmup-steps 0 --measurement-repeats 1 \
  --output-json artifacts/paper_smoke_figure_chain/records.json \
  --output-summary-json artifacts/paper_smoke_figure_chain/summary.json

conda run -n torch2.13-gpu env PYTHONPATH=. python scripts/validate_publication_records.py \
  artifacts/paper_smoke_figure_chain/records.json

conda run -n torch2.13-gpu env PYTHONPATH=. python scripts/build_publication_summaries.py \
  --records-json artifacts/paper_smoke_figure_chain/records.json \
  --output-root artifacts/paper_smoke_figure_chain/derived

conda run -n torch2.13-gpu env PYTHONPATH=. python scripts/generate_publication_figures.py \
  --records-json artifacts/paper_smoke_figure_chain/records.json \
  --output-root artifacts/paper_smoke_figure_chain/figures \
  --status draft

conda run -n torch2.13-gpu env PYTHONPATH=. python scripts/validate_publication_figures.py \
  artifacts/paper_smoke_figure_chain/figures

conda run -n torch2.13-gpu env PYTHONPATH=. python scripts/verify_publication_artifact.py \
  artifacts/paper_smoke_figure_chain
```

该 smoke 会生成 `F2_pareto`、`F3_budget_feasibility`、`F4_baseline_comparison`、`F5_prediction_parity`、
`F6_diagnostic_waterfall`、`F7_phase_peaks`、`F8_topk_ranking`、`F9_optimization_cost` 和
`F10_ablation` 的 `figure.svg`、`source.csv`、`source.schema.json`、`plot_config.json`、
`caption.md` 和 `provenance.json`。这些产物只证明图表链路可用，不进入效果结论。

## 5. 当前草稿图件

可以开始写论文初稿和插入草稿图。当前最完整的草稿图件位于：

```text
artifacts/paper_draft_gpu_combined/figures/
```

该目录包含 4 个工作负载、2 个 diagnostic-hints 变体的 8 条成功 GPU 记录，并已生成 F2/F3/F4/F5/F6/F7/F8/F9/F10。
其中 ResNet50、ViT-B/16 和 GPT-2 来自 FX capture 草稿矩阵，BERT 来自 AOT capture 草稿补测。它适合用于
论文初稿排版、图注打磨和结果段落占位，但还不是最终 `EV-*` 冻结证据；正式结论数字仍需等待统一 backend
策略、baseline 矩阵和 artifact manifest 冻结。BERT-like 默认 workload 后续已改为 deterministic
dropout=0 以满足 compiler qualification，因此旧 BERT 草稿只保留作排版/链路参考，正式 BERT 数字必须重跑。

已验证的相关草稿 artifact：

```text
artifacts/paper_draft_gpu_matrix/
artifacts/paper_draft_gpu_bert_aot/
artifacts/paper_draft_gpu_combined/
artifacts/paper_draft_budget_matrix/
artifacts/paper_draft_budget_plus_high/
```

`artifacts/paper_draft_gpu_combined/` 还包含草稿级 `manifest.json`、`checksums.sha256`、
`configs/workloads.json`、`configs/budgets.json` 和 `tables/T1_workloads.md`。其中 `T1_workloads.md`
是当前初稿可直接引用的 workload 规格表；它会明确 BERT/GPT2 为 like 小规格，不能写成标准
BERT-Base/GPT-2。`configs/budgets.json` 从当前草稿 records 的 all-save 实测峰值派生 50/65/80/95/100%
相对预算，只用于规划后续正式矩阵，不是冻结预算证据。

从预算表驱动实验矩阵的入口已经可用；当前通过 CPU tiny canary 验证了 `budget manifest -> records ->
derived -> figures` 链路：

```bash
conda run -n torch2.13-gpu env PYTHONPATH=. python scripts/run_publication_matrix.py \
  --manifest artifacts/paper_matrix_runner_canary_seed/budgets.json \
  --output-root artifacts/paper_matrix_runner_canary \
  --diagnostic-hints on --top-k 2 \
  --measurement-warmup-steps 0 --measurement-repeats 1 \
  --limit-cells 1 --limit-budgets 1 --device cpu
```

当前预算驱动 GPU 草稿矩阵位于 `artifacts/paper_draft_budget_matrix/`。它使用
`configs/budgets.json` 的 50/65/80/95/100% 相对预算，生成 40 条记录和 F2--F10 图件；其中 15 条
`ok`、25 条 `failed`。失败记录已保留在 records 和 `derived/coverage_summary.json`，主要用于定位紧预算
不可行、Top-K dry-run/measurement 不通过和 BERT AOT lowered gradient mismatch，不得从论文可靠性分母中删除。

当前最适合初稿观察预算趋势和高预算可行锚点的合成草稿位于
`artifacts/paper_draft_budget_plus_high/`。它合并 `paper_draft_budget_matrix` 的 40 条相对预算记录和
`paper_draft_gpu_combined` 的 8 条 4GiB 高预算锚点，生成 48 条 records、F2--F10 图件、`source_map.json`
和 `derived/coverage_summary.json`，并自动导出 T2--T8 草稿表到 `tables/`。该目录只服务草稿分析，不得
替代后续 full/frozen matrix。

可用以下命令审计该草稿距离正式论文 gate 的缺口：

```bash
conda run -n torch2.13-gpu env PYTHONPATH=. python scripts/evaluate_publication_evidence_gates.py \
  artifacts/paper_draft_budget_plus_high \
  --summary-json artifacts/paper_draft_budget_plus_high/derived/summary.json \
  --workload-manifest artifacts/paper_draft_gpu_combined/configs/workloads.json \
  --budget-manifest artifacts/paper_draft_gpu_combined/configs/budgets.json \
  --output-json artifacts/paper_draft_budget_plus_high/evidence_gate_report.json
```

当前预期结果是草稿 `G-2` workload manifest 和 `G-8` 图表生成通过，其余冻结 gate 不通过；这说明
该目录可以支撑初稿排版和结果占位，但还不能登记为 `EV-*` 冻结证据。
后续正式 qualification 运行完成后，可额外传入
`--qualification-summary <records.jsonl.summary.json>`，让 G-1 从真实 runtime qualification artifact
读取 min-cut、block AC 和 SAC 的覆盖状态。

2026-07-20 更新：传入
`artifacts/paper_qualification_aot_baselines_4w_with_min_cut_ratio1_r1/records.jsonl.summary.json` 后，gate 审计已识别
四个 workload 上的 `all_save/pytorch_min_cut/block_ac/sac` AOT eager qualification，`G-1` baseline
identity 已 provisional 通过。该 canary 使用显式 `activation_memory_budget=1.0` 证明真实 PyTorch
min-cut runtime marker；物理预算到 activation ratio 的正式预算选择仍不得由该 canary 推出。

2026-08-02 更新：已补跑严格配对的官方 PyTorch min-cut 三档 canary：

```text
artifacts/paper_qualification_pytorch_min_cut_paired_ratio0_r1/
artifacts/paper_qualification_pytorch_min_cut_paired_ratio0p5_r1/
artifacts/paper_qualification_pytorch_min_cut_paired_ratio1_r1/
artifacts/official_pytorch_min_cut_pareto_4w_3ratio_paired_r1_20260802/
```

三档固定同一 seed、GPU UUID、execution fingerprint 和模型初始化摘要，4/4 workload 通过配对协议审计，
12/12 记录通过真实 min-cut API 身份、正确性和 publication 测量资格。保存 residual 数在 8/8 个相邻
ratio 转换中非减，但完整物理峰值仅 7/8 单调、full-step 时间仅 5/8 单调，说明
`activation_memory_budget` 不能替代完整训练峰值约束。每点目前仍只有 1 个独立 replicate，且没有同一
AOT-eager publication 协议的 PeakAware 曲线，因此该产物属于 paired canary，不能直接支持
“PeakAware 优于官方 min-cut”的结论。

当前已有 GPT2-like 单 workload qualification smoke：

```text
artifacts/paper_qualification_canary_gpt2/
artifacts/paper_qualification_canary_gpt2_scope_v2/
artifacts/paper_qualification_gpt2_aot_baselines_median_timing_r1/
```

前者证明 `all_save/block_ac/sac` 曾在 GPT2-like + AOT eager 上通过 20-repeat qualification，但 summary
生成于 matrix scope 字段加入前；第二个带 matrix scope 字段，但记录为 `timing_unqualified`。当前测量协议
已把 publication timing gate 对齐到正文使用的 median overall-step 口径，并增加 2ms 绝对 event/wall gap
容差以覆盖小模型固定 Python/event 记录开销；raw samples 仍保留逐 repeat outlier。基于该口径，
`paper_qualification_gpt2_aot_baselines_median_timing_r1` 在 GPT2-like + AOT eager 上通过
`all_save/block_ac/sac` 三方法 qualification。它仍只是单 workload、单 replicate、20 repeats 的
provisional canary，不满足四 workload/真实 min-cut 的 G-1 正式覆盖。

当前 BERT-like qualification blocker 有实质进展：registry 默认 BERT-like workload 已显式使用
`hidden_dropout_prob=0.0`、`attention_probs_dropout_prob=0.0` 和 `classifier_dropout=0.0`，避免
PyTorch 2.13 AOT eager 与 eager reference 在 dropout RNG 上产生不可比的随机掩码。以下 artifact
均为单 workload、单 replicate、20 repeats 的 provisional canary，不是冻结证据：

```text
artifacts/paper_qualification_bert_all_save_deterministic_r5/
artifacts/paper_qualification_bert_block_ac_deterministic_r1/
artifacts/paper_qualification_bert_aot_baselines_deterministic_r1/
```

其中 `paper_qualification_bert_all_save_deterministic_r5` 和
`paper_qualification_bert_block_ac_deterministic_r1` 分别通过 BERT-like AOT eager `all_save` 与
`block_ac` 单方法 qualification；`paper_qualification_bert_aot_baselines_deterministic_r1` 中
`all_save` 与 `sac` 为 `ok`，`block_ac` 在该合并运行中因 overall timing gap 失败，随后单方法复跑通过。
这些结果说明 BERT-like 的 stable digest、Transformers config 序列化、torch builtin activation
measurement snapshot 和 matched compiler runtime marker 已基本打通。

ViT-B/16 的 `norm_layer` measurement snapshot blocker 也已解锁。measurement 现在只对
`functools.partial` 包住 `torch.*` 函数或类的确定性 callable factory 做受限 snapshot，因此
`partial(torch.nn.LayerNorm, eps=1e-06)` 可恢复，用户自定义 lambda/partial 仍 fail-closed。新的
single-workload provisional canary：

```text
artifacts/paper_qualification_vit_all_save_norm_layer_fix_r1/
artifacts/paper_qualification_vit_all_save_sac_norm_layer_fix_r1/
artifacts/paper_qualification_vit_block_ac_norm_layer_fix_r1/
artifacts/paper_qualification_vit_block_ac_cpu_rng_restore_r1/
artifacts/paper_qualification_vit_sac_cpu_rng_restore_r1/
artifacts/paper_qualification_vit_aot_baselines_cpu_rng_restore_r1/
```

其中 `paper_qualification_vit_all_save_sac_norm_layer_fix_r1` 在 ViT-B/16 + AOT eager 上通过
`all_save` 和 `sac`。随后定位发现 `block_ac` 的数值、梯度、buffer 和 CUDA RNG 均匹配，失败来自
PyTorch checkpoint under compile 的 CPU RNG bookkeeping；matched compiler executable 现会在
`block_ac` 调用外层恢复 CPU RNG，并在 provenance 中记录 `restore_cpu_rng_after_call=true`。
`paper_qualification_vit_block_ac_cpu_rng_restore_r1` 和 `paper_qualification_vit_sac_cpu_rng_restore_r1`
分别通过 ViT-B/16 单方法 `block_ac` 与 `sac`。`paper_qualification_vit_aot_baselines_cpu_rng_restore_r1`
中 `all_save` 与 `block_ac` 为 `ok`，`sac` 在三方法同 GPU 合并运行中 OOM；单方法 SAC 复跑通过。
这说明 ViT 的测量状态恢复与 block AC CPU RNG drift 已解决。

ResNet50 的 matched compiler blocker 已定位为训练态 BatchNorm 在 cuDNN eager 与 AOT eager
decomposition 之间的数值后端差异；eval 模式完全匹配。publication compiler 现在会对 ResNet50
显式使用 `cudnn_enabled_override=false`，并在 runtime provenance 中记录该口径。新的 provisional
canary：

```text
artifacts/paper_qualification_resnet50_aot_baselines_cudnn_override_r1/
```

该 artifact 在 ResNet50 + AOT eager 上通过 `all_save/block_ac/sac` 三方法 qualification。

当前最完整的 baseline qualification canary：

```text
artifacts/paper_qualification_aot_baselines_4w_current_r1/
```

它覆盖 `resnet50`、`vit_b_16`、`bert_base` 和 `gpt2`，在 AOT eager 上对
`all_save/block_ac/sac` 产生 12 条 `ok` 记录，20 repeats、单 replicate、8GiB budget，并通过
artifact manifest/checksum 校验。

当前最完整的 baseline identity canary：

```text
artifacts/paper_qualification_aot_baselines_4w_with_min_cut_ratio1_r1/
```

它在同一 summary 中覆盖 `all_save/pytorch_min_cut/block_ac/sac`，产生 16 条 `ok` 记录并通过
artifact manifest/checksum 校验。传入该 summary 后，当前草稿 evidence gate 为 `G-1/G-2/G-8`
通过，`G-3/G-4/G-5/G-6/G-7` 仍未通过；正式论文仍需补主矩阵 selected lowered-AOT runtime identity、
Inductor 矩阵、相对预算 all-save 覆盖、5 次独立重复和 frozen manifest。

G-3 主矩阵 lowered-AOT runtime identity 的 canary 已开始：

```text
artifacts/paper_aot_g3_canary/
artifacts/paper_aot_g3_canary_50/
artifacts/paper_aot_g3_canary_80/
artifacts/paper_aot_g3_canary_95/
```

`paper_aot_g3_canary` 使用四 workload 的 100% 高预算 AOT eager 主链路，4 条记录均为 `ok` 并生成
F2--F10；其中 BERT-like、GPT2-like 和 ViT-B/16 的 selected plan 带
`selected_aot_partition_runtime=true`，ResNet50 因高预算选择 `all_save`，仍为
`selected_aot_partition_runtime=false`。50/80/95% canary 显示 BERT/GPT2 可在紧预算下选择 AOT
重计算候选；ResNet50 的 AOT dry-run 仍卡在 mutated buffer output verifier，ViT-B/16 在较紧预算下
存在 fixed lower bound 或 Top-K 无可测候选。G-3 仍未通过，但 blocker 已收窄到 ResNet AOT verifier、
ViT 候选可行性和 all-save selected marker 口径。

2026-07-20 追加进展：主矩阵 AOT policy partition 现在会在 PyTorch 2.13 default partitioner
触发 `Node ... was invalid, but is output` 时 fail-closed fallback 到 min-cut partitioner，并且
AOT candidate validation 对训练态 BatchNorm 模型使用 `cudnn.enabled=false`，与 ResNet50 matched
compiler qualification 口径一致。新的 ResNet50 100% 单 workload canary：

```text
artifacts/paper_aot_g3_resnet100_after_partition_fix/
```

该 artifact 产生 1 条 `ok` 记录，`selected_plan_id=greedy_drop_2`、
`selected_aot_partition_runtime=true`、`dry_run_replay_mode=lowered_aot`，并通过
`verify_publication_artifact.py` 的 records/summary/figures/records-to-figures 校验。80% canary
中的 `block_checkpoint` 已可通过 lowered-AOT dry-run，但实测峰值约 370MiB，高于 308MiB 预算，
因此紧预算 ResNet50 仍需继续调候选/预算口径；G-3 尚未整体通过。

随后复跑四 workload 100% AOT eager 主链路，并将 ResNet50 放在第一个 cell 以避免同进程
CUDA/compile 状态污染：

```text
artifacts/paper_aot_g3_canary_after_partition_fix_resnet_first/
```

该 artifact 覆盖 `resnet50/bert_base/gpt2/vit_b_16`，4 条记录均为 `ok`，且全部满足
`selected_aot_partition_runtime=true` 和 `dry_run_replay_mode=lowered_aot`。补齐 T2--T8 表格、
manifest/checksum 后通过 artifact 校验；传入 qualification summary 后 gate 审计为
`G-1/G-2/G-3/G-8` 通过，`G-4/G-5/G-6/G-7` 仍未通过。95% 预算复跑 artifact：

```text
artifacts/paper_aot_g3_canary_95_after_partition_fix_resnet_first/
```

其中 BERT-like 与 GPT2-like 为 `ok` 且带 lowered-AOT marker；ResNet50 与 ViT-B/16 仍失败于
所有可测候选超过预算线。当前 G-3 的 blocker 已从 runtime identity 转为正式矩阵的预算覆盖、
重复次数和进程隔离口径。

G-4 Inductor matrix 的 100% canary 也已打通：

```text
artifacts/paper_inductor_g4_canary_100_resnet_first/
```

该 artifact 使用同一四 workload、100% 相对预算和 ResNet-first ordering，将 `compile_backend`
设为 `inductor`。4 条记录均为 `ok`，并且均带 `selected_aot_partition_runtime=true` 与
`dry_run_replay_mode=lowered_aot`；补齐 F2--F10、T2--T8、manifest/checksum 后通过 artifact
校验。传入 qualification summary 后 gate 审计为 `G-1/G-2/G-3/G-4/G-8` 通过，`G-5/G-6/G-7`
仍未通过。该结果只证明 Inductor 路径的四 workload canary 可运行，不代表正式 full matrix 或冻结结论。

G-5/G-6 的独立 canary 也已补齐：

```text
artifacts/paper_budget_refs_aot_5pass_r1/
artifacts/paper_measurement_g6_aot_5pass_5repeat_r1/
```

`paper_budget_refs_aot_5pass_r1` 使用 AOT eager、4GiB 高预算、四 workload、5 个 matrix pass
和 1 次 measurement repeat，生成 20 条 `ok` 记录；由此导出的
`artifacts/paper_budget_refs_aot_5pass_r1/budgets.json` 覆盖 50/65/80/95/100% 相对预算，
四个 cell 的 `reference_count=5` 且 `complete=true`，预算 manifest 校验通过。传入该 budget
manifest 后，G-5 已 provisional 通过。

`paper_measurement_g6_aot_5pass_5repeat_r1` 使用同样 workload 与高预算，但将
`measurement_repeats=5`，产生 20 条 `ok` 记录；每个 workload 覆盖 pass 0--4，且所有 ok row
的 `measurement_repeats=5`。传入同一个 budget manifest 后，gate 审计为
`G-1/G-2/G-3/G-5/G-6/G-8` 通过，`G-4/G-7` 仍未通过。当前仍缺把 Inductor G-4 与
5-pass/5-repeat 协议合并到同一正式/full artifact，以及 frozen manifest。

最新 Inductor 5-pass/5-repeat canary 已完成：

```text
artifacts/paper_measurement_g6_inductor_5pass_5repeat_r1/
```

该 artifact 使用四 workload、4GiB 高预算、5 个 matrix pass、`measurement_repeats=5` 和
`compile_backend=inductor`，生成 20 条 `ok` 记录；每个 workload 覆盖 pass 0--4，所有 ok row
均带 `selected_aot_partition_runtime=true` 和 `dry_run_replay_mode=lowered_aot`。补齐
F2--F10、T2--T8、`configs/workloads.json`、`configs/budgets.json`、manifest/checksum 后通过
artifact 校验。传入 baseline qualification summary 后，gate 审计为
`G-1/G-2/G-3/G-4/G-5/G-6/G-8` 通过，仅 `G-7` 因 `evidence_status=provisional` 与 dirty git
state 未通过。当前下一步是把该 7/8 gate canary 扩展为正式 50/65/80/95/100% full matrix，
再冻结 manifest；在此之前仍不得写最终效果数字。

Inductor 50/65/80/95/100% full matrix 已完成 provisional 运行：

```text
artifacts/paper_full_matrix_inductor_5budget_5pass_r1/
```

该 artifact 从 `paper_budget_refs_aot_5pass_r1/budgets.json` 派生 5 个相对预算比例，并将
运行 backend 改为 Inductor；覆盖四 workload、5 个预算比例、5 个 matrix pass、
`measurement_repeats=5`，共 100 条 records，其中 48 条 `ok`、52 条 `failed`。失败记录保留为
紧预算不可行/违约证据，不从分母删除。该目录已生成 F2--F10、T2--T8、manifest/checksum 和
`evidence_gate_report.json`，并通过 artifact 校验。gate 审计为
`G-1/G-2/G-3/G-4/G-5/G-6/G-8` 通过，仅 `G-7` 因未冻结和 dirty git state 未通过。它是当前
最接近正式论文结果的 full artifact，但仍是 `provisional`；正式效果数字需等 frozen manifest。

AOT eager 与 Inductor paired combined full matrix 已生成当前主图优先数据源：

```text
artifacts/paper_full_matrix_combined_paired_5budget_5pass_r1/
```

该 artifact 合并 `paper_full_matrix_combined_5budget_5pass_r1` 与
`paper_full_matrix_combined_hintoff_5budget_5pass_r1`，覆盖四 workload、两个 compile backend、
diagnostic hints on/off、50/65/80/95/100% 相对预算、5 个 matrix pass 和 `measurement_repeats=5`，
共 400 条 records；其中 200 条 `ok`、200 条 `failed`，失败记录完整保留。该目录已生成
F2--F10、T2--T8、manifest/checksum 和 `evidence_gate_report.json`，并通过 frozen artifact 校验。
gate 审计为 `G-1`--`G-8` 全部通过。当前初稿图表应优先从该 paired artifact 取数；正文效果数字
必须绑定 `EV-20`，并保留未测 workload/硬件的边界。该 artifact 已补齐 SVG/PDF/PNG、
source/provenance、manifest/checksum 和环境 lock checksum。

## 6. 稳定编号

- `C-*`：论文论点或贡献；
- `RQ-*`：研究问题；
- `E-*`：实验；
- `EV-*`：冻结证据；
- `F-*`：图；
- `T-*`：表；
- `RISK-*`：风险；
- `G-*`：论文执行 gate。

编号一旦进入正文或答辩，不因排序变化而复用。

## 7. 冲突处理

1. 代码与当前命令输出优先于旧 artifact；
2. `docs/00_可行性与论文边界.md` 的范围优先于论文草稿；
3. `docs/01/02/05` 的算法和契约优先于章节描述；
4. `evidence/00_论点证据账本.md` 的冻结状态优先于正文中的手写数字；
5. 无法追溯到 artifact 字段的数字不得进入正文。

外层仓库 `docs/18-thesis-writing-guide.md` 和 `toolkit_examples/outputs/F*.png` 属于另一套
PyTorch 2.6 仿真工具口径，不能直接作为 PeakAware 论文证据或图表来源。
