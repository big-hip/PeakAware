#!/usr/bin/env python3
"""Probe: reconstruct the 15-candidate artifact workloads from registry defaults.

For each task, build default registry task -> capture -> IR -> default plans ->
full-model simulate_plan. Compare the all_save estimated peak against the
recorded artifact values. This verifies the reconstruction matches the GPU
anchor before the component ablation is run.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.api import _hardware_spec, _request_key
from peakaware.capture import capture_joint_graph
from peakaware.config import PeakAwareConfig
from peakaware.contracts import TrainingRequest
from peakaware.ir import build_joint_ir
from peakaware.memory.fixed_frontier import analyze_coarse_feasibility, build_optimizer_spec
from peakaware.memory.simulator import simulate_plan
from peakaware.models.registry import TrainingTaskRegistry
from peakaware.search.engine import _manual_default_plans

MIB = 1024 * 1024

# Task budgets from the 15-candidate artifact run_config.json.
TASK_BUDGETS = {
    "bert_base": 181403648,
    "gpt2": 354418688,
    "resnet50": 600834048,
    "vit_b_16": 2106589184,
}

# Recorded all-save estimated peaks (MiB) from exhaustive_records.json.
TARGETS = {
    "bert_base": 63.8,
    "gpt2": 199.4,
    "resnet50": 373.6,
    "vit_b_16": 1964.0,
}


def run_task(registry: TrainingTaskRegistry, name: str, *, batch: int = 1) -> None:
    torch.manual_seed(20260802)
    random.seed(20260802)
    task = registry.get(name)
    model = task.build_model()
    model.train()
    example_args, example_kwargs = task.build_batch(batch)
    loss_fn = task.loss_fn
    optimizer = task.build_optimizer(model)

    config = PeakAwareConfig(enable_compile=ENABLE_COMPILE, enable_inductor=False)
    memory_budget_bytes = TASK_BUDGETS[name]
    optimizer_spec = build_optimizer_spec(optimizer, model)
    request = TrainingRequest(
        model=model,
        example_args=example_args,
        example_kwargs=example_kwargs,
        loss_fn=loss_fn,
        optimizer=optimizer,
        memory_budget_bytes=memory_budget_bytes,
        config=config,
        optimizer_spec=optimizer_spec,
        hardware=_hardware_spec(example_args, example_kwargs),
        request_key=_request_key(model, example_args, example_kwargs, memory_budget_bytes, config),
    )

    fixed_timeline, coarse = analyze_coarse_feasibility(model, optimizer, memory_budget_bytes)
    capture = capture_joint_graph(request)
    ir, ir_report = build_joint_ir(capture)
    if not ir_report.valid:
        raise ValueError(f"invalid IR: {ir_report.errors}")

    plans = _manual_default_plans(ir, memory_budget_bytes, safety_margin_bytes=0)
    all_save = next(p for p in plans if p.plan_id == "all_save")
    sim = simulate_plan(ir, all_save, fixed_timeline, cost_provider=None)

    target = TARGETS.get(name)
    est_mib = sim.estimated_peak_bytes / MIB
    ck = getattr(capture, "capture_key", None)
    print(f"[{name}] batch={batch} capture_key={ck} "
          f"est_all_save={est_mib:.1f}MiB (target {target}MiB, diff {est_mib-target if target else float('nan'):+.1f}) "
          f"peak_phase={sim.peak_snapshot.phase} fw={sim.fw_peak_bytes/MIB:.1f} bw={sim.bw_peak_bytes/MIB:.1f} "
          f"opt={sim.optimizer_peak_bytes/MIB:.1f} ops={len(ir.ops)} values={len(ir.values)} storages={len(ir.storages)} "
          f"fixed_resident={fixed_timeline.resident_bytes/MIB:.1f}MiB")


def main() -> None:
    registry = TrainingTaskRegistry.with_defaults()
    for name in ["bert_base", "gpt2", "resnet50", "vit_b_16"]:
        run_task(registry, name, batch=BATCH)


ENABLE_COMPILE = True
BATCH = 1


if __name__ == "__main__":
    main()
