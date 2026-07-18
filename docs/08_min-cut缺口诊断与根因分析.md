# Min-Cut 缺口诊断与峰值根因分析

M1 只实现 alias pinning、fixed frontier、peak migration 和 rematerialization wave 四类直接证据。
本章完整 D0--D5 counterfactual、自动 root-cause ranking 和报告系统属于 M2，不能阻塞 M0 可执行
partition 或 M1 基础搜索。

## 1. 目标

PeakAware 不仅输出更好的 SAVE 计划，还必须回答：

```text
原策略预计减少了多少 saved activation？
这些值是否真的没有跨 FW/BW 边界保存？
为什么 after-FW、BW peak 或 overall peak 没有按预期下降？
峰值发生在哪个 phase、哪个 op、哪些 storage 同时存活？
应该修改 SAVE 边界、cost、workspace 模型，还是停止重计算？
```

诊断对所有策略生效，包括 default、PyTorch min-cut、block AC、greedy、用户插件策略和
PeakAware 自身。它不是失败后的日志，而是搜索前后的必经阶段。

## 2. 期望收益差距

对 baseline `b` 和候选计划 `p`，先区分策略自己报告的预期与 PeakAware 统一 storage 模型
重算的预期。策略预期必须携带 provenance；没有公开预期模型的策略将相关字段标为
`unavailable`，仍可完成 D1--D5 诊断。

```text
strategy_expected_saved_gain
    = strategy_reported_saved_bytes(b) - strategy_reported_saved_bytes(p)

normalized_saved_gain
    = normalized_saved_storage(b) - normalized_saved_storage(p)

after_fw_gain
    = measured_or_simulated_after_fw(b) - after_fw(p)

overall_peak_gain
    = physical_peak(b) - physical_peak(p)

strategy_estimation_gap
    = strategy_expected_saved_gain - normalized_saved_gain

realization_gap
    = normalized_saved_gain - overall_peak_gain

total_expectation_gap
    = strategy_expected_saved_gain - overall_peak_gain
```

`strategy_estimation_gap` 用于判断策略的 logical-value/cut 估计是否高估了真实 storage 释放；
`realization_gap` 用于判断 storage 确实减少后，收益是否被 BW transient、fixed frontier 或编译
物化抵消。策略未报告预期时，只计算 normalized/realization 相关字段。差值不能直接分摊到
各组件，因为两个计划的峰值可能发生在不同 phase 和 op。

## 3. 反事实分层诊断

系统对同一 SAVE 计划构建一组逐层增加现实因素的反事实模型：

| 层级 | 加入因素 | 回答的问题 |
|---|---|---|
| D0 | strategy estimate + normalized storage | 策略与统一模型各预计节省多少？ |
| D1 | storage alias + after-FW liveness | 预计删除的值是否真的释放 storage？ |
| D2 | recompute closure + actual BW order | 重计算是否形成长生命期 transient？ |
| D3 | gradients + optimizer + fixed buffers | 新峰值是否被固定 frontier 接管？ |
| D4 | compiler materialization + workspace | Top-K 编译后是否出现新物化？ |
| D5 | allocator + runtime measurement | 真实执行是否推翻预测？ |

D0--D3 是所有可模拟计划的静态诊断。D4 只对已编译候选可用，D5 只对已完成 runtime 验证的
候选可用。不可用层必须记录 `unavailable` 及原因，不能用零变化代替；部分静态信息只能支持
低置信度 D4 时，应显式标记 `estimated`。

相邻层的变化用于归因：

```text
D0 -> D1: saved estimate / alias / pinning gap
D1 -> D2: rematerialization liveness gap
D2 -> D3: fixed frontier / phase migration gap
D3 -> D4: compiler and workspace gap
D4 -> D5: allocator and model residual
```

每个层级都分别计算该反事实模型下 baseline 和 candidate 的完整时间线最大值，并保存双方的
peak phase/op/snapshot。Waterfall 展示的是相邻层“两个最大值之差”的变化，不是在某一个峰值
时刻把组件相加。报告必须注明分层顺序，且不宣称这是唯一的数学因果分解。

## 4. 峰值快照

每个层级和真实测量保存：

```python
@dataclass(frozen=True)
class PeakSnapshot:
    phase: str
    op_id: int | None
    op_target: str | None
    total_bytes: int
    parameter_bytes: int
    gradient_bytes: int
    optimizer_bytes: int
    saved_activation_bytes: int
    recomputed_bytes: int
    workspace_bytes: int
    allocator_residual_bytes: int
    live_storage_ids: tuple[int, ...]

@dataclass(frozen=True)
class CounterfactualResult:
    level: str
    status: str  # available | estimated | unavailable
    unavailable_reason: str | None
    baseline_peak: PeakSnapshot | None
    candidate_peak: PeakSnapshot | None
    peak_gain_bytes: int | None
    confidence: float
```

进入搜索或 Top-K 的 `EvaluatedPlan` 还必须携带 `max_recompute_live_bytes`、
`recompute_span_ops`、`recompute_before_first_bw_op_bytes`、`risk_score` 和 `confidence`，使
`REMATERIALIZATION_WAVE` 可以参与约束和排序，而不只是事后文字诊断。

如果 baseline 和 plan 的峰值位置不同，报告同时展示两个快照，不能只在 baseline 峰值时刻
比较计划内存。

## 5. 根因分类

```python
class RootCause(Enum):
    UNKNOWN = auto()
    SAVED_ESTIMATE_MISMATCH = auto()
    ALIAS_OR_VIEW_PINNING = auto()
    MANDATORY_SAVE_DOMINANCE = auto()
    REMATERIALIZATION_WAVE = auto()
    SHARED_RECOMPUTE_EXPANSION = auto()
    FIXED_BACKWARD_FRONTIER = auto()
    PEAK_PHASE_MIGRATION = auto()
    COMPILER_MATERIALIZATION = auto()
    WORKSPACE_GROWTH = auto()
    ALLOCATOR_FRAGMENTATION = auto()
    COST_MODEL_MISRANK = auto()
    DYNAMIC_SHAPE_DRIFT = auto()
    MEASUREMENT_NOISE = auto()
```

一个计划可以有多个根因，但必须区分 primary cause 和 secondary causes。

## 6. 根因证据要求

### Saved Estimate Mismatch

证据：策略报告删除的 logical bytes 与 PeakAware 规范化后的 FW outputs/storage bytes 不一致。
没有策略预期 provenance 时不判定该根因。

常见原因：tuple output、view、alignment、mandatory output 或错误 value-to-storage mapping。

### Alias or View Pinning

证据：逻辑 value 被删除，但其 base storage 仍被其他 alias 或 graph output 持有。

### Rematerialization Wave

证据：第一个或某个关键 backward op 前连续执行大量 forward-like 节点，多个重计算 storage
同时存活；`recomputed_bytes` 显著抵消 saved reduction。报告至少给出上述三项 wave metrics，
并标明触发的是 hard limit、soft penalty 还是仅低置信度预警。

### Shared Recompute Expansion

证据：删除多个值导致共享 ancestor 被多次或过早重建，实际闭包大于逐值 cost 相加的预期。

### Fixed Backward Frontier

证据：峰值快照主要由参数、梯度、optimizer、LM-head/loss 或 mandatory workspace 构成；进一步
DROP activation 只增加重计算时间，overall peak 基本不变。

### Peak Phase Migration

证据：FW peak 下降，但 overall peak 移动到 BW 或 optimizer；新 phase 的峰值高于原收益。

### Compiler Materialization / Workspace

证据：D3 预测可行，但 D4/实测出现新的 materialized buffer、layout conversion、fusion boundary
或 kernel workspace。

### Cost Model Misrank

证据：预测为低 cost 的重计算 region 在真实 profile/Top-K 中成本较高，导致错误 SAVE/DROP
排序。该根因主要解释时间损失，也可能通过不同 compiler plan 间接影响峰值。

## 7. 诊断报告

```python
@dataclass(frozen=True)
class PlanDiagnosticReport:
    baseline_plan_id: str
    candidate_plan_id: str
    strategy_expectation_source: str | None
    strategy_expected_saved_gain_bytes: int | None
    normalized_saved_gain_bytes: int
    after_fw_gain_bytes: int
    overall_peak_gain_bytes: int
    strategy_estimation_gap_bytes: int | None
    realization_gap_bytes: int
    total_expectation_gap_bytes: int | None
    baseline_peak: PeakSnapshot
    candidate_peak: PeakSnapshot
    counterfactual_levels: tuple[CounterfactualResult, ...]
    primary_cause: RootCause
    secondary_causes: tuple[RootCause, ...]
    evidence: tuple[DiagnosticEvidence, ...]
    repair_hints: tuple[RepairHint, ...]
    confidence: float
```

报告同时提供面向人的文字解释和面向搜索器的结构化 hints。

## 8. 根因到解决方案的映射

| 根因 | 候选修复动作 | 用户建议 |
|---|---|---|
| Saved estimate mismatch | 改用 storage bytes 重新建 cut 权重 | 检查 IR/alias pass |
| Alias/view pinning | 合法时增加 base storage 候选 | 评估 compact materialization |
| Mandatory save dominance | 停止删除无关值 | 检查 graph break/副作用 |
| Rematerialization wave | SAVE block boundary；提高 wave penalty | 比较 block AC |
| Shared recompute expansion | 按闭包而非逐值 cost 重排候选 | 增加 shared-ancestor profile |
| Fixed backward frontier | 终止无效 DROP 搜索 | 改 optimizer、offload/FSDP 或 batch |
| Peak phase migration | 对新峰值 phase 重新 repair | 报告阶段变化 |
| Compiler materialization | 淘汰候选或增加物化 penalty | 检查 fusion/layout |
| Workspace growth | 淘汰候选或增加安全裕量 | profile/替换该 kernel |
| Allocator fragmentation | 增加 margin、隔离进程复测 | 考虑 allocator-aware 扩展 |
| Cost model misrank | 触发热点 profile 并重新搜索 | 更新 Costmodel calibration |
| Dynamic shape drift | 创建新 graph/plan key | 限制 guard domain |

表中动作默认只是 candidate hint。只有满足所需证据、通过 legality/closure/partition 预检，且
处于 PeakAware 授权范围内，才能标记为 automatic。FSDP、offload、optimizer、kernel 实现切换
等默认仅作为建议，不擅自改变用户训练语义。

## 9. 搜索器接入

```python
@dataclass(frozen=True)
class RepairHint:
    action: str
    mode: str  # suggested | automatic
    value_ids: tuple[int, ...]
    storage_ids: tuple[int, ...]
    region_ids: tuple[int, ...]
    expected_peak_delta: int | None
    expected_time_delta_us: float | None
    reason: RootCause
    confidence: float
    preconditions: tuple[str, ...]
    required_evidence_ids: tuple[str, ...]
    authority_scope: str
    legality_status: str
```

搜索流程：

1. Baseline 阶段至少诊断 default、min-cut 和 block AC；
2. 根据 root cause 调整候选和 seed；
3. 每轮 repair 后更新诊断；
4. 若 primary cause 转为 fixed frontier，提前终止；
5. 对进入 repair 或 Top-K 的 greedy/plugin/PeakAware `EvaluatedPlan` 执行诊断；
6. Top-K 实测后用 D5 residual 更新最终报告。

## 10. 插件接口

```python
class PlanDiagnostic(Protocol):
    def analyze(
        self,
        baseline: EvaluatedPlan,
        candidate: EvaluatedPlan,
        context: DiagnosticContext,
    ) -> PlanDiagnosticReport: ...
```

内置 `PlanDiagnosticPlugin` 注册：

- counterfactual ladder；
- root-cause rules；
- peak snapshot comparator；
- repair-hint mapper；
- human-readable report renderer。

插件事件：`after_baseline_analysis`、`after_plan_diagnostic`、`after_runtime_diagnostic`。

## 11. 输出示例

```text
Strategy: min_cut_budget_0.50
Strategy expected saved gain:   472 MiB (PyTorch min-cut estimator)
Normalized storage gain:        456 MiB
Observed after-FW gain:         456 MiB
Observed overall peak gain:       8 MiB
Strategy estimation gap:         16 MiB
Realization gap:                448 MiB
Total expectation gap:         464 MiB

Candidate peak: BW / aten.mm_backward / 12.31 GiB
Primary cause: FIXED_BACKWARD_FRONTIER
Secondary cause: REMATERIALIZATION_WAVE

Evidence:
- recompute transient increased by 281 MiB
- gradient + fixed frontier occupied 10.9 GiB at candidate peak
- further DROP simulation changed peak by < 16 MiB but added 7.4 ms

Recommended action:
- stop aggressive DROP search
- SAVE residual boundaries [v128, v244]
- profile recompute region r17
- report activation-only budget as low-headroom
```

## 12. 测试

构造可控图分别触发每个主根因：

- view base 被 alias pin；
- residual chain rematerialization wave；
- shared ancestor expansion；
- gradient/optimizer fixed frontier；
- FW -> BW peak migration；
- mock compiler workspace growth；
- profile cost misrank；
- measurement noise threshold。

还必须覆盖：

- default/block/greedy/plugin strategy 没有自带 expected bytes 时的降级行为；
- D4 未编译、D5 未实测时的 `unavailable` 状态；
- 低 confidence 或 `UNKNOWN` 时不生成 automatic hint；
- hint 未通过 legality/partition 预检时降级为 suggested；
- 不同峰值 phase 下不进行单时刻组件伪加和。

测试不仅断言 root-cause label，还要断言证据字段、峰值位置和 repair hint。无法获得充分证据时
返回低 confidence 或 `UNKNOWN`，不能强行给出确定原因。

## 13. 论文实验

需要报告：

- 根因分类准确率：在 synthetic ground-truth 图上的 precision/recall；
- 真实模型中各根因出现频率；
- 使用诊断 hints 前后的搜索迭代数、预算满足率和吞吐；
- predicted saved gain 与 actual peak gain 的散点；
- counterfactual D0--D5 waterfall；
- 错误归因案例和 confidence calibration。

这使论文不仅回答“PeakAware 是否更好”，还回答“PyTorch min-cut 为什么在该 workload 上没有
兑现预期收益，以及系统如何据此采取修复动作”。
