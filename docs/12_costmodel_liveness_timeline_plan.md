# Costmodel + Liveness Timeline Simulation Plan

Status: implementation plan after the CUDA full-model v15 audit.

## Goal

PeakAware should produce a simulated training-step timeline that is close to
real execution in both time and memory:

1. Time comes primarily from the Costmodel, calibrated to the measured hardware.
2. Memory comes from graph liveness, autograd saved tensors, fixed model state,
   optimizer state/temporaries, and Costmodel workspace when its semantics are
   workspace-like.
3. The plotted curve should expose model components instead of hiding residual
   error with a cosmetic offset.

## Current Finding

The project already has the right activation source of truth:

- IR liveness records when activation values are produced and last consumed.
- AOTAutograd residual outputs mark tensors that cross the FW/BW boundary.
- Runtime execution saves the same residual outputs with `ctx.save_for_backward`.

The missing piece is not "saved tensor discovery"; it is the mapping from that
logical graph lifetime to allocator-visible CUDA memory over time.

## Why Costmodel Alone Is Not Enough

The bundled atencost Costmodel estimates operator time and memory traffic.  In the
current adapter, `OpCost.memory_bytes` is interpreted as transient workspace,
not as live allocation.  Therefore:

- Costmodel can place ops on the time axis.
- Costmodel can contribute workspace only when the op model really returns
  allocator-like workspace.
- Costmodel cannot decide which activation storages are live across FW/BW.
- Costmodel cannot directly replay PyTorch's CUDA caching allocator, Inductor
  buffer reuse, cuDNN algorithm selection, or optimizer internal temporaries.

The correct composition is:

```text
time(t)   = cumulative Costmodel op_time, phase-aligned to measured FW/BW/OPT
memory(t) = fixed_state + IR_live_payload(t) + costmodel_workspace(t)
            + optimizer_frontier(t) + calibrated_runtime_gap(t)
```

## FakeTensor Capture Path

Using PyTorch FakeTensor for zero-allocation graph capture is feasible and
should be the default direction for large models, but it must be scoped:

- Good fit: shape/dtype/alias/liveness extraction, Costmodel signatures, L2
  memory simulation, and L2.5 fusion-aware graph analysis.
- Risk: FakeTensor does not execute real kernels and does not expose actual
  CUDA allocator reuse, runtime workspaces, or fused-kernel temporary buffers.
- Required guard: after capture, validate that all graph tensor metadata has
  usable `meta["val"]`, shape, dtype, and storage identity or a safe fallback.

PeakAware already has a lightweight fake-input safety layer, while the top-level
`toolkit` tree has more mature FakeTensor/L2/L2.5/L3 estimators.  The next
engineering step is to port or bridge the `toolkit.simulation.graph_estimator`
timeline output into PeakAware's `selected_simulated_memory_event_trace`.

## L2 / L2.5 Memory Timeline Path

L2 should be used for AOT/eager graphs:

- count storage-aware tensor allocation size;
- pin graph outputs and saved tensors;
- release activations at last consumer;
- keep parameters, buffers, gradients, and optimizer state as fixed components.

L2.5 should be used when the graph has Inductor-like fusion behavior:

- identify fusion groups;
- eliminate internal tensors that do not materialize in global memory;
- model conservative in-place/safe reuse;
- keep unknown/materializing ops as barriers to avoid underestimation.

This is a better memory source than Costmodel memory traffic.  Costmodel memory
fields should only be layered in as workspace after the op mapping has proven
the field means workspace for that op family.

## Chakra ET / ASTRA-sim Path

Chakra ET is a graph-based execution trace format for ML workloads.  ASTRA-sim
accepts MLCommons Chakra Execution Traces as workload-layer inputs:

- Chakra: https://github.com/mlcommons/chakra
- ASTRA-sim: https://github.com/astra-sim/astra-sim
- ASTRA-sim workload layer docs:
  https://astra-sim.github.io/astra-sim-docs/workload-layer/overview.html

This makes a Chakra export useful for interoperability, especially if later
work needs distributed compute/communication simulation.

However, ASTRA-sim does not directly analyze PeakAware ATenIR.  The required
path is:

```text
PeakAware JointTrainingIR / ATen FX
  -> Chakra ET nodes: compute, memory, communication
  -> Chakra ET dependencies
  -> ASTRA-sim workload layer
  -> ASTRA-sim simulated execution result
```

For the current single-GPU memory timeline objective, Chakra/ASTRA-sim is a
secondary path.  It adds a converter and simulator dependency before solving the
main allocator/liveness alignment problem.

## Implementation Order

1. Keep Costmodel hardware explicit: `PEAKAWARE_COSTMODEL_HARDWARE=RTX_A6000,RTX_A6000`.
2. Extend event trace rows with component fields:
   `fixed_bytes`, `payload_bytes`, `workspace_bytes`, `optimizer_bytes`,
   `gradient_bytes`, `parameter_bytes`, `buffer_bytes`, `live_storage_count`.
3. Align Costmodel event time per phase, not with one global scale:
   FW raw span -> measured FW time, BW raw span -> measured BW time, optimizer
   raw span -> measured optimizer time.
4. Make BW saved-tensor release use confirmed IR backward last-use only.
   Unknown BW use must be kept live to avoid underestimating.
5. Add a PeakAware bridge for `toolkit` L2/L2.5 timeline data or port the
   relevant estimator code into `peakaware.memory`.
6. Only after the single-GPU path is stable, add a Chakra ET exporter for
   downstream ASTRA-sim experiments.

## Acceptance Evidence

A complete run should provide:

- successful full-model records for `resnet50`, `vit_b_16`, `bert_base`, `gpt2`;
- `actual_vs_simulated_event_figure.svg` for each model;
- per-model `event_fit_summary.json`;
- source CSVs exposing the component fields above;
- record validation with `validate_publication_records.py`;
- unit tests that lock phase-wise time alignment and component export.
