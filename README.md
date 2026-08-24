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
- `peakaware/memory/predictive_trajectory.py`：基于独立参考轨迹的 held-out 显存轨迹预测原语；
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

## Held-out 轨迹预测

PeakAware 还提供了一个不读取目标运行时轨迹的参考校准接口，用于预测未测
shape 的 FW/BW/OPT 显存曲线。实现位于
`peakaware/memory/predictive_trajectory.py`，并从 `peakaware.memory` 导出：

```python
from peakaware.memory import (
    apply_reference_trajectory_prediction,
    evaluate_reference_trajectory_prediction,
    fit_reference_trajectory_calibration,
)
```

### 预测协议

参考数据和目标数据必须分离。典型流程如下：

```python
calibration = fit_reference_trajectory_calibration(
    [
        {"batch": 2, "sequence": 512, "events": reference_b2},
        {"batch": 4, "sequence": 512, "events": reference_b4},
    ],
    target_shape={"batch": 8, "sequence": 512},
)

predicted = apply_reference_trajectory_prediction(
    target_scheduler_events,
    calibration,
)

metrics = evaluate_reference_trajectory_prediction(
    predicted,
    target_measurement_events,
)
```

其中，`reference_b2` 和 `reference_b4` 是独立参考运行的事件列表，
`target_scheduler_events` 只能包含目标编译器/Scheduler 产生的事件结构，不能
包含目标运行时显存结果。预测函数会：

1. 按 FW、BW、OPT 分别将参考轨迹归一化到阶段进度 `[0, 1]`；
2. 根据 `batch`、`sequence`、`image_size` 和 `memory_budget` 选择最近参考；
3. 保留目标事件的 `event_id` 和时间戳，把参考 allocator envelope 映射到目标事件；
4. 保留参数、优化器状态、激活、梯度、临时空间和 workspace 等结构化组件；
5. 将预测总量与物理组件之差写入 `trajectory_model_residual_bytes`。

预测事件满足以下审计不变量：

```text
physical_component_sum_bytes + trajectory_model_residual_bytes
    = predicted_total_bytes
```

这里的 signed residual 是参考模型对 allocator/Scheduler 未建模部分的修正项，
不是一个实际分配的 Tensor，不能在物理显存组成中重复计数。应用函数不会修改
目标事件身份和时间戳；评估函数只在预测文件生成之后读取目标实测轨迹。

### 输入事件字段

最少需要 `phase`、`time_us` 和以下任一总量字段：`current_bytes`、
`allocated_bytes` 或 `bytes`。若需要组件守恒审计，事件可以提供：

```text
parameter_bytes
buffer_bytes
optimizer_state_bytes
saved_activation_bytes
recomputed_activation_bytes
gradient_bytes
optimizer_temp_bytes
temporary_bytes
workspace_component_bytes
allocator_residual_bytes
```

### 结果边界

该 API 是 reference-calibrated prediction primitive，不等同于 zero-shot
预测，也不保证所有模型、memory budget 或重计算策略都具有相同精度。正式实验
应至少使用两个与目标候选不同的参考 shape，记录 workload/recomputation
fingerprint，并在预测产物冻结后再打开目标测量文件。Figure 7/9 的完整
Inductor Scheduler 组件管线和 held-out manifest 位于上层
`runtime_simulation/` 工程中；本模块负责可复用的轨迹模板、组件守恒和误差评估逻辑。
