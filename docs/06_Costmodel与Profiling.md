# Costmodel 与算子 Profiling 设计

本章主体属于 M2。M0/M1 只要求统一 `CostProvider` 协议、Legacy adapter 和静态 fallback；
ProfileDB、插值、collector、calibration 与 provider plugin 不阻塞可执行闭环和 PeakAware 搜索。

## 1. 定位

Costmodel 与 Profiling 数据库互补：

- Costmodel 提供低成本、可外推、可解释的冷启动估计；
- ProfileDB 提供目标硬件上高置信度的真实算子时间和 workspace；
- Top-K 端到端测量负责校正 fusion、并发、launch 和候选级非加性误差。

现有 `PeakAware/Costmodel` 不需要先整体重写。第一阶段通过 adapter 接入，后续只重写覆盖率高
且误差大的算子。

## 2. 统一数据结构

```python
@dataclass(frozen=True)
class OpSignature:
    target: str
    overload: str | None
    input_shapes: tuple[tuple[int | str, ...], ...]
    input_dtypes: tuple[str, ...]
    input_strides: tuple[tuple[int | str, ...], ...]
    output_shapes: tuple[tuple[int | str, ...], ...]
    layout_flags: tuple[str, ...]
    attributes: tuple[tuple[str, str], ...]
    device_type: str
    hardware_id: str
    torch_version: str

@dataclass(frozen=True)
class OpCost:
    latency_us: float
    workspace_bytes: int
    source: str
    confidence: float
    sample_count: int = 0
    p50_us: float | None = None
    p90_us: float | None = None
```

同一 op name 在不同 shape、dtype、stride、layout 和属性下不能共享一个记录。Stride 用于
描述实际存储步幅，`layout_flags` 用于记录 contiguous、channels-last、transposed 或 backend
特定布局语义，两者不能互相替代。

## 3. Provider 接口

```python
class CostProvider(Protocol):
    name: str

    def supports(self, signature: OpSignature) -> bool: ...
    def estimate(self, signature: OpSignature) -> OpCost | None: ...
```

内置 provider：

- `ExactProfileProvider`；
- `InterpolatedProfileProvider`；
- `LegacyCostmodelAdapter`；
- `RooflineFallbackProvider`；
- `CompositeCostProvider`。

## 4. 查询与融合

默认优先级：

```text
exact target-hardware profile
-> same op/dtype/layout nearby-shape interpolation
-> analytical Costmodel
-> roofline + launch overhead fallback
```

每次查询返回 provenance。未知算子不能默认为零成本或零 workspace。

对已有 profile 和解析模型，可使用残差校准：

```text
calibrated_cost(signature)
    = analytical_cost(signature)
    + learned_residual(op_family, shape_features, hardware)
```

第一版不需要复杂神经模型；分段线性、log-linear 或简单回归足够。

## 5. Legacy Costmodel 适配

```python
class LegacyCostmodelAdapter(CostProvider):
    def __init__(self, legacy_factory, hardware):
        self.legacy_factory = legacy_factory
        self.hardware = hardware

    def supports(self, signature):
        return can_convert(signature, self.hardware)

    def estimate(self, signature):
        legacy_op = to_legacy_op_record(signature)
        result = self.legacy_factory(legacy_op, self.hardware)()
        return from_legacy_result(result)
```

迁移边界：

- adapter 位于新包，不修改旧 Costmodel 的公共语义；
- A2/A3 等硬件模型只声明自己支持的设备；
- GPU 不强行复用 NPU 参数，缺失时走 GPU profile/roofline provider；
- 解析模型异常返回 `None` 或显式 error，不吞掉错误；
- adapter 的转换规则有独立 golden tests。

## 6. 渐进重写顺序

1. 统计主模型中的 op family 覆盖率和时间占比。
2. 包装已有 matmul、elementwise、attention、fused op。
3. 在目标 GPU/NPU 上采集真实数据并计算误差。
4. 优先重写高时间占比、高误差算子。
5. 低频算子继续使用 profile 或 fallback。
6. 每次重写必须在同一 benchmark corpus 上比较误差和查询开销。

不要以“重写全部 Costmodel”为 PeakAware 的前置任务。

## 7. ProfileDB Schema

建议 SQLite 作为第一版：

```text
hardware(
    id, vendor, name, compute_capability,
    peak_flops, memory_bandwidth, driver_version
)

op_signature(
    signature_hash, target, overload,
    input_shapes_json, output_shapes_json,
    dtypes_json, strides_json, layout_flags_json,
    attributes_json, device_type,
    hardware_id, torch_version
)

op_profile(
    signature_hash, collector_version,
    sample_count, p50_us, p90_us, mean_us,
    workspace_bytes, measured_at, raw_artifact
)

calibration(
    model_key, plan_key, predicted_us, measured_us,
    residual_us, compiler_version
)
```

大规模离线分析可导出 Parquet，但在线查询仍优先 SQLite 或内存索引。

## 8. Profiling 采集

### 模型内采集

使用 `torch.profiler`/Kineto 获取 ATen op、CUDA kernel、shape、调用关系和内存事件。优点是
接近真实上下文，缺点是 fusion 后 ATen 与 kernel 不是一一对应。

### 微基准

对热点签名生成 runnable：

1. 分配真实输入；
2. warmup；
3. CUDA Event 计时；
4. 正确同步；
5. 重复采样；
6. 保存 P50/P90 和异常值统计。

### 深度分析

Nsight Systems/Compute 只用于少量代表方案，分析 kernel overlap、Tensor Core、SM Active 和
DRAM throughput，不进入在线搜索。

## 9. Workspace

workspace 是完整峰值的重要误差源。数据来源按优先级：

1. profile 期间的 peak delta；
2. backend/algorithm 可查询 workspace；
3. op-family 经验上界；
4. 未知时使用保守 margin，并降低 confidence。

不能简单把输出 tensor bytes 当作 workspace。

## 10. Fusion 与非加性成本

单算子 latency 相加无法精确预测 Inductor fusion 后时间。采用两层 cost：

```text
search cost: op/profile/analytical additive approximation
validation cost: compiled Top-K end-to-end measurement
```

可选 `FusionGroupCostProvider` 接收 Inductor 或静态 fusion group，但它是增强插件，不阻塞 MVP。

## 11. 置信度

示例：

```text
1.00 exact profile, same hardware/software/signature
0.80 nearby shape interpolation
0.60 calibrated analytical model
0.35 uncalibrated analytical model
0.20 roofline fallback
```

方案 cost confidence 不能简单取平均，建议记录最低置信度、低置信度重算时间占比和关键路径
置信度。搜索器据此计算 `risk_score`，并与 estimated peak/time、workspace/compiler residual 和
重算波风险一起进入 Top-K 排序。Top-K 应保留少量高不确定性但可能更优的探索候选。

诊断器若判定 `COST_MODEL_MISRANK`，应把导致错误排序的 op family、signature 和预测/实测
残差写入 profile queue。补充采集后生成新的 cost database version，并重新运行受影响的候选，
不能在原计划上静默替换 cost。

## 12. 插件接入

```python
class LegacyCostmodelPlugin:
    def register(self, registry):
        registry.register_service(
            "cost_provider",
            "legacy_costmodel",
            LegacyCostmodelAdapter(...),
            priority=50,
        )
```

Cost provider 不需要也不允许 patch 搜索器。`CompositeCostProvider` 由 registry snapshot 组装，
provider 集合与版本进入 executable cache key，不应使 capture/analysis cache 失效。

## 13. 数据质量测试

- 相同 signature 重复采集的方差；
- cold/warm cache 差异；
- profile 与端到端 kernel 时间的对应关系；
- 不同 shape 插值误差；
- 解析模型 MRE/P50/P90；
- workspace 误差；
- cost ranking 与候选真实 ranking 的 Kendall/Spearman 相关性；
- profile 数据跨 PyTorch/CUDA 版本失效。

论文最重要的 cost 实验不是绝对误差最低，而是 profile/calibration 是否改善最终 plan 排序和
相同峰值下的吞吐。
