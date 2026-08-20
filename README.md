# PeakAware

PeakAware 是面向 PyTorch 编译训练的端到端峰值显存约束激活重计算项目。系统计划只捕获一次
AOTAutograd FW/BW joint graph，以 storage-aware 方式选择 activation 的 SAVE/RECOMPUTE，并以
完整训练 step 的物理峰值和执行时间评价方案。

当前仓库已实现 M0--M2 主线：联合图捕获、storage-aware IR 与峰值仿真、SAVE 集合搜索、AOT
partition 下发、Top-K dry-run/实测校正、compiled FW/BW + eager optimizer executor、根因诊断、
三层 cache、ProfileDB、插件与可复现实验报告。M3 的 dynamic shape、MoE 和分布式联合优化仍是
显式扩展域，不混入当前静态训练 ABI。

## 目录

- `peakaware/`：捕获、IR、内存模型、搜索、partition、runtime、诊断与实验核心包；
- `scripts/`：实验矩阵、离线汇总、baseline、根因和 artifact 验证入口；
- `tests/`：unit、integration 和 correctness 验证；
- `docs/`：项目范围、算法、执行流程、实验口径和最终验收矩阵；
- `Costmodel/`、`Optim_IR/`：研究适配来源，不是核心 runtime 的强依赖。

实现前先阅读：

1. [项目结构与函数拆分](docs/05_项目结构与函数拆分.md)
2. [联合重计算实现计划](docs/01_联合重计算实现计划.md)
3. [端到端执行流程](docs/02_端到端执行流程.md)

## 验证

在项目约定环境中运行（依赖见 `requirements.txt`，论文基线为 NVIDIA A6000 / PyTorch 2.13.0+cu130）：

```bash
pip install -r requirements.txt
python -m pytest tests/unit tests/integration tests/correctness -q
```

最小端到端入口为 `scripts/run_mvp.py`，批量实验入口为 `scripts/run_experiments.py`。当前效果、
仿真误差、根因准确率、cache 复用和计划下发证据统一记录在
[最终验收矩阵](docs/10_最终验收矩阵.md)，该文档同时给出可复现实验命令和结果解释边界。

## 论文复现

- **论文四模型**定义在 `peakaware/models/registry.py`
  （`build_bert_base_task` / `build_gpt2_task` / `build_resnet50_task` / `build_vit_b16_task`）。
- **仿真预测**入口：`scripts/run_publication_matrix.py`；表格/图片生成：
  `scripts/generate_publication_tables.py`、`scripts/generate_publication_figures.py`。
- **冻结证据**（论文表格对应的 artifact）在 `artifacts/`。
- **跨硬件（昇腾 910B）通用性实验**：`scripts/cross_hardware/`，见其
  `README.md`——用论文自身 4 模型、同一套冻结规则预测昇腾峰值显存（3/4 模型 APE ≤ 6.4%）。
