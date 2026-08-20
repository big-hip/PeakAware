# Costmodel Calibration Notes

Status: active A6000 implementation note (2026-08-02).

## Scope

This note records the current PeakAware costmodel calibration path.  It is not a
replacement for the frozen EV-20 artifact; it explains how the next run can
improve raw simulation accuracy.

## Implemented Changes

1. `peakaware.memory.simulator.simulate_plan` now estimates recompute transient
   memory with a live-range replay over recomputed forward ops.  The old M0
   model treated all dropped activation storage as live at the same time.
2. `simulate_plan` accepts a `CostProvider` and uses op-level `estimated_us`,
   `memory_bytes`, and `confidence` to calibrate recompute latency, workspace,
   and plan confidence.
3. `search.engine.evaluate_plan`, greedy search, and repair all pass the same
   `CostProvider`, so search ranking and simulation use one costmodel source.
4. `LegacyCostmodelAdapter` now attempts to call the bundled
   `Costmodel/zhanlu` analytical model through a conservative TensorRecord
   projection.  Unsupported ops still fall back to the static provider and keep
   explicit provenance.
5. `scripts/calibrate_costmodel_from_records.py` fits median peak residual rules
   from existing records and reports raw versus calibrated APE.
6. The default A6000 path loads explicit `RTX_A6000` chip/topology files
   (38.7 TFLOP/s FP32, 768 GB/s HBM, 48 GiB, 6 MiB L2, and PCIe latency/bandwidth)
   instead of silently using the old A3 hardware.
7. `ProfileDB` schema v2 removes unique FX node names from reusable signatures,
   stores complete input/output shapes and dtypes, and isolates every row by
   hardware and Torch/CUDA software fingerprints. Legacy unscoped rows are
   retained for audit but cannot satisfy a current-environment lookup.
8. Profile interpolation is no longer a universal total-byte multiplier.
   Elementwise operations use bytes, matmul uses shape-derived FLOPs, attention
   uses attention work, and unsafe families such as convolution require exact
   profiles or an operator-specific analytical model.
9. FX/AOT graph-interface nodes use an explicit `structural_zero` provider.
   Alias-only `view`, `t`, `transpose`, `permute`, `slice`, `select`, `expand`,
   `detach`, and tuple `getitem` nodes use `metadata_view_zero`; their device
   storage lifetime remains modeled by IR aliases, but they no longer receive
   a fictitious GPU-kernel launch cost.
10. Fused efficient/flash/cuDNN SDPA has a fusion-aware A6000 analytical model.
    Exact offline A6000 profiles override it through ProfileDB.

## A6000 Full-Model Coverage Audit

The profiled audit command is:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n torch2.13-gpu env PYTHONPATH=. \
python scripts/audit_costmodel_coverage.py \
  --tasks bert_base_full_s128,gpt2_small_full_s128,resnet50,vit_b_16 \
  --device cuda \
  --microbatch-size 1 \
  --profile-db artifacts/costmodel_profiles_a6000_v2_20260802/profiles.sqlite \
  --output-root artifacts/costmodel_coverage_a6000_full_profiled_v3_20260802
```

Authoritative result:

| Task | IR ops | Reusable signatures | Structural | Metadata/view | Analytical | Profile exact | Static fallback | Unresolved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BERT-base-S128 | 1898 | 276 | 206 | 1188 | 480 | 24 | 0 | 0 |
| GPT-2-small-S128 | 1776 | 254 | 165 | 1060 | 551 | 0 | 0 | 0 |
| ResNet-50 | 1364 | 489 | 323 | 638 | 403 | 0 | 0 | 0 |
| ViT-B/16 | 1797 | 251 | 155 | 1174 | 444 | 24 | 0 | 0 |

Aggregate coverage is 6835/6835 IR nodes with zero static fallback and zero
unresolved nodes. This is a coverage statement, not yet an end-to-end accuracy
statement.

Treating metadata/view nodes correctly reduced the summed raw op-time prior
from 327.9 ms to 166.5 ms across the four tasks. The previous value was inflated
because the legacy model charged about 35 us to each view-like node.

## Offline SDPA Profiling

The exact fused-attention profiles were collected without executing any search
candidate:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n torch2.13-gpu env PYTHONPATH=. \
python scripts/profile_sdpa_costs.py \
  --tasks bert_base_full_s128,vit_b_16 \
  --device cuda --microbatch-size 1 --warmup 10 --repeats 50 \
  --profile-db artifacts/costmodel_profiles_a6000_v2_20260802/profiles.sqlite \
  --output-json artifacts/costmodel_profiles_a6000_v2_20260802/sdpa_profiles.json
```

| Signature | Analytical | A6000 p50 | Relative error | Allocator workspace |
|---|---:|---:|---:|---:|
| BERT S128 SDPA forward | 14.26 us | 40.96 us | 65.18% | 0 B |
| BERT S128 SDPA backward | 34.02 us | 205.31 us | 83.43% | 405,504 B |
| ViT S197 SDPA forward | 32.41 us | 58.37 us | 44.47% | 0 B |
| ViT S197 SDPA backward | 79.22 us | 223.74 us | 64.59% | 797,184 B |

The large raw analytical error is why exact profile rows are required for these
paper configurations. In the profiled coverage audit all 48 repeated SDPA
nodes resolve through four exact environment-scoped records.

## High-Batch Canonical Transformer Configuration

The full-transformer main experiment now uses `microbatch_size=128` for both
`bert_base_full_s128` and `gpt2_small_full_s128`.  Batch 32 remains a historical
development point and must not be mixed into the B128 headline comparison.
Every compared method (all-save, real block AC, SAC, native AOT min-cut, and
PeakAware) must use the same model, sequence length, FP32 dtype, AdamW
configuration, input seed, and B128 microbatch within a workload.

The B64/B128 all-save capacity scan used the publication measurement protocol
(5 warmup steps and 20 CUDA-Event repeats) and measured only the single final
all-save plan for each configuration:

| Task | Microbatch | Allocated peak | Peak phase | Median step time |
|---|---:|---:|---|---:|
| BERT-base-S128 | 64 | 3.813 GiB | backward | 292.73 ms |
| BERT-base-S128 | 128 | 6.380 GiB | backward | 562.64 ms |
| GPT-2-small-S128 | 64 | 11.523 GiB | backward | 446.19 ms |
| GPT-2-small-S128 | 128 | 21.213 GiB | backward | 867.52 ms |

The authoritative scan artifact is
`artifacts/high_batch_capacity_probe_b64_b128_v2_20260802/`.  It contains four
measurements used only to choose the canonical workload size and all-save
reference; it is not a candidate-policy benchmark set.  Formal policy search
must still report zero candidate measurements and may execute only its selected
final plan for validation.

The exact B128 BERT SDPA profiles are stored in the same environment-scoped
ProfileDB and in
`artifacts/costmodel_profiles_a6000_v2_20260802/sdpa_profiles_bert_b128.json`:

| Signature | Analytical | A6000 p50 | Relative error | Allocator workspace |
|---|---:|---:|---:|---:|
| BERT B128 SDPA forward | 1698.31 us | 957.44 us | 77.38% | 0 B |
| BERT B128 SDPA backward | 4227.99 us | 2081.79 us | 103.09% | 51,904,512 B |

An initial zero-candidate-run B128 all-save simulation correctly classified
both peaks as backward, but its raw peak estimates were 12.746 GiB for BERT and
14.226 GiB for GPT-2, versus 6.380 GiB and 21.213 GiB measured.  Therefore B128
is frozen as the workload size, while the raw memory model is not yet qualified
for B128 search.  The next calibration step must correct activation-liveness
over-counting in BERT and the missing GPT-2 backward transient/workspace; a
single unsigned safety margin cannot correct errors with opposite signs.

## EV-20 Offline Calibration Check

The following command fits task-level residuals on matrix passes 0--3 and
evaluates pass 4:

```bash
conda run -n torch2.13-gpu env PYTHONPATH=. python scripts/calibrate_costmodel_from_records.py \
  --records-json artifacts/paper_full_matrix_combined_paired_5budget_5pass_r1/records.json \
  --output-json artifacts/costmodel_calibration_ev20_task_holdout_pass4/report.json \
  --key-fields task_name \
  --holdout-field matrix_pass_index \
  --holdout-value 4
```

Held-out result:

| Calibration key | Rows | Mean raw APE | Mean calibrated APE | P90 raw APE | P90 calibrated APE | Within 10% raw | Within 10% calibrated |
|---|---:|---:|---:|---:|---:|---:|---:|
| `task_name` | 160 | 0.4932 | 0.1540 | 0.8992 | 0.3321 | 0.0250 | 0.4625 |
| `task_name,budget_bytes` | 160 | 0.4932 | 0.1332 | 0.8992 | 0.4349 | 0.0250 | 0.5875 |

The task+budget key has lower mean APE and better within-10% rate, but it has
more rules and a worse P90 than task-only calibration.  The safer default for a
new publication run is task-level residual calibration unless additional
held-out passes show the finer key is stable.

## Caveats

- These reports are derived from existing records and do not mutate EV-20.
- Full paper claims still require a new frozen matrix generated with the
  calibrated simulator.
- Zero fallback proves routing completeness only. Most remaining non-SDPA
  kernels still use the A6000 analytical prior and require representative
  offline profiles plus held-out full-step validation.
- `metadata_view_zero` models CUDA-event time. If the publication metric changes
  to Python wall time, host dispatch/metadata overhead must be added separately.
- Full actual-versus-simulated memory trajectories and phase-wise time fit for
  the full BERT/GPT models remain the next acceptance gate.
