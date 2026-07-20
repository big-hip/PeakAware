# F-1 PeakAware 系统闭环（草图）

状态：`draft`，不依赖实验数字，不能替代 `frozen` 出版图。

建议 caption：

> PeakAware first constructs a storage-aware view of the AOTAutograd joint graph, diagnoses where saved-activation reductions diverge from complete-step physical peak, repairs candidate SAVE sets, and measures only a diverse Pareto Top-K after lowering. The optimizer is outside the joint graph but remains inside the measured training step, so a plan is accepted only when its executable identity, numerical correctness, and complete-step allocated peak satisfy the registered protocol.

进入正文前的冻结要求：

- 绑定 `EV-13`，证明主工作负载候选计划确实有 lowered-AOT runtime marker、plan identity 和 correctness 记录；
- 导出 PDF、SVG、PNG，并记录生成命令、checksum 和字体检查；
- 若正式 runtime 路径发生变化，同步更新图中的 ABI/fallback 边界。
