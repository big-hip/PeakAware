# PeakAware

PeakAware 是面向 PyTorch 编译训练的端到端峰值显存约束激活重计算项目。系统计划只捕获一次
AOTAutograd FW/BW joint graph，以 storage-aware 方式选择 activation 的 SAVE/RECOMPUTE，并以
完整训练 step 的物理峰值和执行时间评价方案。

当前仓库保存项目设计规范、实验参考代码以及已有 Costmodel/Optimizer IR 适配来源。核心
`peakaware/` 包尚未开始实现，后续严格按照 M0 -> M1 -> M2 -> M3 推进，避免提前引入插件、完整
诊断、持久化缓存和其他非必要模块。

## 目录

- `docs/`：项目范围、算法、执行流程、分期实现契约和实验计划；
- `Costmodel/`：现有 Costmodel 参考与后续 adapter 来源；
- `Optim_IR/`：optimizer IR 研究参考，不是 M0/M1 运行时依赖。

实现前先阅读：

1. [项目结构与函数拆分](docs/05_项目结构与函数拆分.md)
2. [联合重计算实现计划](docs/01_联合重计算实现计划.md)
3. [端到端执行流程](docs/02_端到端执行流程.md)

## 当前版本

当前快照为设计基线版本，已明确：

- M0 最小可运行闭环和支持域；
- 跨阶段不可变数据契约；
- 模块依赖 DAG 与职责所有权；
- compiled FW/BW 与 eager optimizer 的运行期边界；
- M1/M2/M3 扩展准入规则。
