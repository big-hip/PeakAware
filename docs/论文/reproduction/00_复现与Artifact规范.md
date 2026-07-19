# 复现与 Artifact 规范

## 1. 目标与边界

本文档规定论文实验如何运行、存档、校验和重建图表。它不规定实验矩阵，不保存最终结果数字，
也不判断某个论点是否成立。实验设计以 `experiments/00_实验补强与测量协议.md` 为准，冻结数字仅由
`evidence/00_论点证据账本.md` 维护，图表设计以 `figures/00_图表与输出规范.md` 为准。

这套规范服务于 G-3、G-4、G-6、G-7 和 G-8。旧 64 条效果矩阵因早于 plan-aware runtime，只能以
`invalid` 历史输入保留，不得装入论文发布包的冻结证据集。

## 2. 可复现级别

| 级别 | 用途 | 允许的证据状态 | 必须产物 |
|---|---|---|---|
| `smoke` | 验证环境、capture、lowering、测量和绘图链路 | `smoke` | manifest、环境指纹、raw、log、校验和示例图 |
| `full` | 按冻结协议生成论文候选证据 | `provisional` | 完整目录、所有失败、processed、plans、checksums |
| `release` | 从已审核 full run 构建论文 Artifact | `frozen` | 锁定代码/环境/数据，一键重建，预期输出，外部说明 |

`smoke` 成功不表示效果论点成立。`full` 运行完成后仍需通过证据账本审核，才能升级为
`release`。release 是发布状态，不是新的 GPU 运行类型；它引用一个或多个已审核 full `run_id`。任何 raw、
环境或源码变化都必须创建新 `run_id`，不允许在原 release 上就地修补。

## 3. `run_id` 与实验身份

### 3.1 命名格式

```text
<UTC timestamp>_<git-short-sha>_<protocol>_<backend>_<run-scope>
```

示例：

```text
20260719T142530Z_20a1bee4cafe_v3_inductor_full
```

- UTC 时间使用 `YYYYMMDDTHHMMSSZ`；
- `git-short-sha` 取实验启动时 `git rev-parse --short=12 HEAD`；
- `protocol` 是实验协议版本，不是自由备注；
- `backend` 使用冻结枚举，如 `aot_eager` 或 `inductor`；
- `run-scope` 只能是 `smoke` 或 `full`；release 包另用 `release_id` 并列出 `source_run_ids`。

同一 `run_id` 内的每条记录还必须有稳定 `case_id` 和 `attempt_id`。`case_id` 由模型实际规格、输入、
dtype、optimizer、backend、预算和策略规范化后哈希得到；`attempt_id` 用于区分独立进程重复。模型名称必须使用
`BERT-like-2L-64H` 和 `GPT2-like-2L-64H`，除非 G-2 已经通过且更新冻结规格。

### 3.2 运行前必须是真的不变量

1. `git status --porcelain` 保存到 manifest；正式 full/release 应使用 clean commit。
2. baseline 身份必须显式。当前 mandatory-only 实现写为 `min_cut_proxy`，排序前半保存实现写为
   `block_checkpoint_proxy`；不得写成 `pytorch_min_cut` 或真实 block AC。
3. 完整记录 batch、shape、dtype、optimizer、precision、backend、warmup、repeats、Top-K、预算定义和种子。
4. 预算必须同时保存 bytes 与相对 all-save measured peak 的比例；禁止只保存“4 GiB”这类宽松标签。
5. full/release 禁止在同一已污染进程内依次跑完所有策略；候选顺序应随机化，每个候选使用隔离 worker
   或等价独立状态。

## 4. Artifact 发布包目录

```text
peakaware-artifact/
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- manifest.json
|-- checksums.sha256
|-- environment/
|   |-- conda-explicit.txt
|   |-- conda-environment.yml
|   |-- pip-freeze.txt
|   |-- system.json
|   |-- torch-config.txt
|   `-- gpu-query.csv
|-- source/
|   |-- git-bundle/
|   `-- patches/
|-- configs/
|   |-- protocol.yaml
|   |-- workloads.yaml
|   `-- matrix.yaml
|-- raw/
|   `-- <run_id>/
|       |-- records.jsonl
|       |-- failures.jsonl
|       |-- events/
|       `-- workers/
|-- plans/
|   `-- <run_id>/<case_id>/
|-- logs/
|   `-- <run_id>/
|-- processed/
|   `-- <run_id>/
|-- figures/
|   |-- source/
|   |-- pdf/
|   |-- svg/
|   `-- png/
|-- scripts/
|   |-- bootstrap.sh
|   |-- run_smoke.sh
|   |-- run_full.sh
|   |-- validate_artifact.sh
|   `-- build_figures.sh
|-- expected/
|   |-- smoke-summary.json
|   `-- README.md
`-- docs/
    |-- artifact-evaluation.md
    |-- schema.md
    `-- troubleshooting.md
```

`raw/` 是只追加、不可变的观测层；`processed/` 是可由 raw 重建的导出层；`figures/` 不允许存在无法追溯到
processed/raw 的手工数字。`plans/` 保存 SAVE 集合、plan hash、partition provenance、lowering/runtime markers
和正确性状态。因为当前仓库 `.gitignore` 忽略 `artifacts/`，release 必须通过独立发布包、release asset
或归档存储提供，不能假设 clean clone 拥有本机历史 artifact。

## 5. Manifest 和 Schema

### 5.1 `manifest.json` 最小字段

```json
{
  "artifact_schema_version": "1.0.0",
  "run_id": "20260719T142530Z_20a1bee4cafe_v3_inductor_full",
  "scope": "full",
  "evidence_status": "provisional",
  "created_at_utc": "2026-07-19T14:25:30Z",
  "git": {
    "commit": "<40-hex-sha>",
    "dirty": false,
    "submodules": {},
    "patches": []
  },
  "environment": {
    "conda_env": "torch2.13-gpu",
    "python": "<exact-version>",
    "torch": "2.13.0+cu130",
    "cuda_runtime": "<version>",
    "driver": "<version>",
    "gpu": ["<exact-device-name>"],
    "os": "<distribution-and-kernel>"
  },
  "protocol": {
    "version": "v3",
    "config_sha256": "<hex>",
    "backend": "inductor",
    "measurement_clock": ["cuda_event", "synchronized_perf_counter"]
  },
  "inputs": [],
  "outputs": [],
  "known_limitations": []
}
```

`inputs`/`outputs` 中每项至少包含 relative path、media type、byte size、SHA-256 和生成者脚本。schema 修改遵循
semantic versioning：破坏兼容的字段语义变化升 major，新增可选字段升 minor，说明修正升 patch。

### 5.2 raw record 必须字段组

- 身份：`run_id`、`case_id`、`attempt_id`、`record_schema_version`、UTC 时间；
- 源码与环境：commit、dirty flag、GPU UUID/model、torch/CUDA/driver；
- workload：论文名称、registry key、完整 config、input shape、batch、dtype、optimizer、seed；
- 策略：strategy 枚举、baseline provenance、plan hash、SAVE storage/value IDs、Top-K rank；
- runtime：backend、capture/lowering 身份、`selected_aot_partition_runtime`、`selected_activation_checkpoint`、
  `dry_run_replay_mode`、fallback reason；
- 测量：warmup/repeats、每次原始时间和峰值样本、FW/BW/optimizer allocated/reserved peak、overall peak；
- 预算：absolute bytes、all-save baseline run/case ID、相对比例；
- 正确性：loss/output/gradient tolerance、实测差值、pass/fail；
- 结果：`ok`、`budget_violation`、`oom`、`timeout`、`correctness_failure`、`compile_failure`、
  `runtime_failure`、`unsupported` 或 `infra_failure`，以及结构化 stage/reason。`unsupported` 进入
  baseline coverage，`infra_failure` 续跑后不计方法胜负，其余预注册方法失败进入相应可靠性分母。

每次 repeat 的原始样本必须保留；median、P10/P90、bootstrap CI 等统计只能在 `processed/` 中派生。raw
中的缺失值使用 JSON `null` 并提供 reason，不得用 `0` 代表“未测量”。

## 6. 环境锁定

正式运行使用现有 `torch2.13-gpu` conda 环境，但环境名不是可复现证据。发布包必须同时记录：

```bash
conda list -n torch2.13-gpu --explicit > environment/conda-explicit.txt
conda env export -n torch2.13-gpu --no-builds > environment/conda-environment.yml
conda run -n torch2.13-gpu python -m pip freeze > environment/pip-freeze.txt
conda run -n torch2.13-gpu python -c 'import torch; print(torch.__version__); print(torch.__config__.show())' \
  > environment/torch-config.txt
nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,memory.total,pstate \
  --format=csv > environment/gpu-query.csv
```

`system.json` 还要记录 Python、glibc、kernel、CPU、RAM、CUDA runtime、cuDNN、相关环境变量、allocator 设置、
Inductor/Triton 版本和 cache 策略。私有 AOT API 对 PyTorch commit/版本敏感，因此不接受仅写“PyTorch 2.x”。

## 7. 执行命令模板

下列是 release 包必须暴露的稳定入口。具体 CLI 参数由当前代码实现落实，脚本不得依赖作者主目录的绝对路径。

```bash
# 1. 验证环境、schema 和小型端到端链路，目标约 10--30 分钟完成
conda run -n torch2.13-gpu bash scripts/run_smoke.sh \
  --config configs/protocol.yaml \
  --output raw/<run_id>

# 2. 运行冻结矩阵；可以从 manifest 识别未完成 case 续跑，但不覆盖旧 attempt
conda run -n torch2.13-gpu bash scripts/run_full.sh \
  --protocol configs/protocol.yaml \
  --matrix configs/matrix.yaml \
  --run-id <run_id> \
  --output raw/<run_id>

# 3. 从 raw 全量重建 processed 和出版图，不重跑 GPU 实验
conda run -n torch2.13-gpu bash scripts/build_figures.sh \
  --manifest manifest.json \
  --raw raw/<run_id> \
  --processed processed/<run_id> \
  --figures figures

# 4. 在任何对外发布前执行完整性校验
conda run -n torch2.13-gpu bash scripts/validate_artifact.sh \
  --manifest manifest.json \
  --checksums checksums.sha256
```

`run_smoke.sh` 至少要验证 capture、候选 plan identity、lowered-AOT runtime marker、数值/梯度正确性、分 phase
测量、raw schema 和一张示例图。它应将实际摘要与 `expected/smoke-summary.json` 的结构约束比较，但不要把
跨 GPU 易波动的时间/峰值写成完全相等断言。

## 8. Raw 不可变与失败保留

1. worker 只向临时文件追加，成功 `fsync` 后再原子 rename；不直接改写已存在的 `records.jsonl`。
2. 每条 raw record 一经纳入 checksum 即只读。解析/单位/schema 错误通过新版 processor 修正，不编辑原记录。
3. OOM、timeout、compile failure、runtime failure 和 correctness failure 全部保留。`failures.jsonl` 记录结构化类型、
   stage、exception、exit code、timeout、最后日志指针和可选 core dump 指针。
4. 因失败而没有的数值为 `null`；OOM 不可被记为“超预算的一个巨大峰值”，两者统计口径分开。
5. 续跑时创建新 `attempt_id`；只能由已版本化的 aggregation rule 决定采用哪个 attempt。
6. 运行中间状态也要保存。这包括编译中断前的 plan、已完成 warmup 数、GPU 状态和 stderr，用于区分系统错误与策略不可行。

## 9. Processed 和一键图表

`processed/` 的每个文件必须在 metadata 中记录：输入 raw SHA-256 集合、processor commit、命令行、schema 版本、
随机种子和生成时间。主抽样单位是模型规格、输入、microbatch、dtype、optimizer 和 loss 定义的唯一
workload configuration；backend/预算是实验 cell，独立进程/seed 是重复，策略、hints 和 matrix pass
是配对条件。跨 cell 汇总必须先在 workload 内聚合或按 workload cluster 重采样，不得扩充 `N`。

一键绘图脚本必须：

- 从空 `processed/` 和 `figures/` 开始可重建；
- 自动读取 evidence ledger 允许的 `EV-*` 对应 run，拒绝 `invalid`、`smoke` 或未冻结输入进入论文图；
- 对输入字段、单位、baseline provenance、模型实际规格和 runtime marker 做 fail-fast 校验；
- 同时产生 PDF、SVG 和 300--600 DPI PNG，并导出图中数据 CSV/JSON；
- 保存绘图代码、版本和所有参数，禁止图形编辑器中手工移动数据元素；
- 在 CI 或干净环境中至少验证文件集、尺寸、非空像素和导出数据 checksum。

## 10. Checksum 和发布冻结

在 Artifact 根目录执行：

```bash
LC_ALL=C find . -type f ! -path './checksums.sha256' -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum > checksums.sha256
sha256sum --check checksums.sha256
```

发布压缩包本身再发布一个独立 SHA-256，并在证据账本中记录永久地址、大小和哈希。`checksums.sha256`
的排序、locale 和路径分隔符必须稳定。如需分卷，manifest 记录每卷的顺序、字节数和哈希。

release 冻结顺序是：

1. 冻结 commit、protocol、matrix 和环境；
2. 生成 full run，保留成功与失败；
3. 从 raw 重建 processed 和 figures；
4. 审核 schema、runtime markers、baseline 身份、正确性和统计单位；
5. 在 evidence ledger 中为通过项分配/更新 `EV-*` 并标记 `frozen`；
6. 在新鲜环境运行 smoke、figure rebuild 和 checksum validation；
7. 生成只读发布包，记录归档 DOI/URL 和包哈希。

## 11. Artifact 评审对齐

发布包按 ACM Artifact Review and Badging 的目标自查，并参考 Systems Artifact Evaluation 的操作经验：

| 目标 | PeakAware 必须提供的可检查内容 |
|---|---|
| Available | 稳定公开地址、开放许可证、CITATION、完整 checksum；不依赖开发机 `artifacts/` |
| Functional | 清晰安装、可完成 smoke、预期输出、错误排查、schema/plan/runtime marker 校验 |
| Reusable | 结构化 config、版本化 schema、模块化脚本、新 workload/baseline 扩展点和开发文档 |
| Results Reproduced | 独立团队使用作者提供的 Artifact 得到论文主要结果，明确硬件、容差和非确定性 |
| Results Replicated | 独立团队不依赖作者 Artifact、使用自行开发的实现/产物获得主要结果；这不是本发布包可自行宣称的状态 |

评审指南必须区分“目标 10--30 分钟的功能 smoke”与“需要 GPU 时数的 full reproduction”，给出预计时间、显存、磁盘、
网络与硬件要求。无 RTX A6000 时可允许跑功能验证，但不应把不同 GPU 的绝对峰值直接判为论文数字复现成功。

## 12. 发布前验收清单

- [ ] `run_id`、commit、protocol 和环境一致，full/release 工作树 clean；
- [ ] G-1 前的 baseline 明确写为 proxy，G-2 前的模型使用 like 规格名；
- [ ] 所有候选含 plan hash、lowering/runtime markers 和正确性结果；
- [ ] raw 样本不可变，OOM/timeout/失败均保留，未测值不用 `0`；
- [ ] 每个文件都被 manifest 或 checksum 覆盖，`sha256sum --check` 通过；
- [ ] `processed/` 和全部图表可从 raw 一键重建，不需手工修数字；
- [ ] smoke 在干净环境通过，full 的计算/磁盘成本和预期输出已记录；
- [ ] evidence ledger 仅将审核通过的 release 产物标记为 `frozen`；
- [ ] 压缩包已在新目录解压后重新验证，公开地址与包哈希可访问。
