# PeakAware 文档索引

PeakAware 面向 PyTorch 编译训练中的端到端峰值显存约束。系统只捕获一次
AOTAutograd FW/BW 联合图，自动选择前向值的 `SAVE/RECOMPUTE`，并在满足真实训练峰值
预算的前提下尽量提高有效吞吐。

## 唯一主线

1. [00_可行性与论文边界.md](00_可行性与论文边界.md)
   定义论文主题、相关工作边界、可写贡献和不可过度声称的内容。
2. [01_联合重计算实现计划.md](01_联合重计算实现计划.md)
   定义正式优化问题、核心 IR、峰值模型和 peak-aware SAVE 搜索算法。
3. [02_端到端执行流程.md](02_端到端执行流程.md)
   描述从请求、一次捕获、分析、搜索、Top-K 编译到三层缓存的完整数据流。
4. [03_复杂度与开销分析.md](03_复杂度与开销分析.md)
   分析搜索复杂度、工程风险、降级路线和阶段性 Go/No-Go 条件。
5. [04_插件与补丁架构.md](04_插件与补丁架构.md)
   规定插件优先、`patch_method` 兜底的扩展机制和 PyTorch 内部接口隔离方式。
6. [05_项目结构与函数拆分.md](05_项目结构与函数拆分.md)
   给出 M0--M3 分期目录、核心数据契约、依赖 DAG、文件职责与扩展准入规则。
7. [06_Costmodel与Profiling.md](06_Costmodel与Profiling.md)
   规定现有 Costmodel 的渐进迁移、算子 Profiling 数据库和多数据源融合方式。
8. [07_实验与论文产出计划.md](07_实验与论文产出计划.md)
   定义模型、基线、指标、消融、论文结构和最终验收标准。
9. [08_min-cut缺口诊断与根因分析.md](08_min-cut缺口诊断与根因分析.md)
   定义期望收益差距、反事实分层诊断、峰值根因分类和自动修复建议。
10. [09_min-cut重计算的背景与动机.md](09_min-cut重计算的背景与动机.md)
   将 PyTorch min-cut/budget 的语义边界、反向峰值反升现象和 PeakAware 研究动机整理为论文背景。

实现者推荐阅读顺序：`00 -> 09 -> 05 -> 01 -> 02 -> 03 -> 04 -> 06 -> 08 -> 07`。

## 核心口径

```text
输入：模型、训练样例、绑定模型参数的 optimizer、目标硬件、物理峰值预算 B
捕获：一次 FakeTensor + AOTAutograd Joint Graph
决策：按物理 storage effect 选择哪些 activation 跨越 FW/BW 边界 SAVE
派生：RECOMPUTE 闭包、实际 partition 和 backward 执行顺序
约束：端到端实测或校准后的训练峰值 <= B
目标：最小 step time，或在固定任务下最大化 samples/s、tokens/s
输出：compiled FW/BW + eager optimizer 的可执行、可验证、可回退训练 step
```

搜索内部以 storage/alias/liveness 为计量单位；`saved_value_ids` 仅作为 AOT partition 下发 ABI。

无论最终使用哪种策略，系统都必须先比较其“预计 saved activation 降幅”和“端到端峰值实际
降幅”，解释差距来自何处，并为搜索器或用户生成可执行的修复建议。

第一版不把任意 backward 拓扑排序作为独立决策变量。SAVE 集合决定 partition 边界，当前
PyTorch partitioner 和依赖关系产生实际重计算顺序；PeakAware 评估并修复 SAVE 集合对该顺序
和张量生命期造成的影响。

## 范围边界

- FakeTensor 表示零真实张量数据内存的元数据捕获，不代表零 CPU 内存或零捕获时间。
- Optimizer 位于 joint graph 外，以 eager phase 进入固定显存与完整 step 峰值分析，不作为 SAVE
  决策变量。
- 参数、梯度和 optimizer state 无法容纳时，仅靠 activation recomputation 不能使训练可行。
- “充分利用算力”以有效吞吐为主指标，不能把额外重计算 FLOPs 当成收益。
- M0/M1 使用直接依赖和构造函数注入；M2 只把已证明需要替换的边界提升为插件。
- 只在 PyTorch 没有公开扩展点时使用作用域化、可恢复的补丁。

本目录中的所有文档均为当前规范。发生冲突时，以 `00` 的范围边界、`05` 的实现契约和依赖规则、
`01/02` 的算法与流程为准；`04/06` 只约束 M2 扩展接口，`03/07` 只规定风险、实验和验收。
