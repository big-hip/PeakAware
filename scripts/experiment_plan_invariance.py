#!/usr/bin/env python3
"""P0-2: Compiler plan-invariance validation (real activation-dominant workloads).

Question: does one shared dry compilation provide fusion/materialization facts
that remain valid across recomputation candidates (different saved boundaries)?

Design: capture one workload's joint graph once; build three candidate plans
(all-save, torch_min_cut mandatory-save, block_checkpoint); lower each
candidate's FW/BW graphs independently and run the L2 liveness trace on each
candidate's *own* lowered partition. The all-save lowered partition acts as the
"shared compilation". We report, per candidate:
  - L2 predicted peak (own compile) vs shared-compile peak
  - lowered FW graph structure (op counts), replay/BW op growth
  - L2 step time difference
This is the direct experiment the reviewer requested (fusion/materialization
agreement, predicted peak/time difference). Uses real BERT-like / ResNet-like
tasks at activation-dominant scale so the peak actually moves with the plan.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from peakaware.api import _hardware_spec, _request_key
from peakaware.capture import capture_joint_graph
from peakaware.config import PeakAwareConfig
from peakaware.contracts import TrainingRequest
from peakaware.ir import build_joint_ir
from peakaware.memory.fixed_frontier import analyze_coarse_feasibility, build_optimizer_spec
from peakaware.memory.fx_timeline import (
    simulate_lowered_fx_l2_event_trace,
    summarize_lowered_fx_l2_event_trace,
)
from peakaware.memory.simulator import simulate_plan
from peakaware.models.registry import build_bert_base_task, build_resnet50_task
from peakaware.partition.aot import partition_joint_graph
from peakaware.search.engine import _manual_default_plans

MIB = 1024 * 1024


def build_task(name: str, *, hidden: int = 256, layers: int = 4, seq: int = 128, vocab: int = 1000):
    if name == "bert":
        return build_bert_base_task(
            sequence_length=seq,
            vocab_size=vocab,
            hidden_size=hidden,
            num_hidden_layers=layers,
            num_attention_heads=max(2, hidden // 64),
            intermediate_size=4 * hidden,
        )
    if name == "resnet":
        return build_resnet50_task(image_size=64, num_classes=10)
    raise ValueError(name)


def run_task(task, *, out: str, batch: int = 2) -> None:
    torch.manual_seed(20260802)
    random.seed(20260802)
    model = task.build_model()
    model.train()
    example_args, example_kwargs = task.build_batch(batch)
    loss_fn = task.loss_fn
    optimizer = task.build_optimizer(model)

    config = PeakAwareConfig(enable_compile=False, enable_inductor=False)
    memory_budget_bytes = 4 * 1024 * MIB
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
    print(f"[{task.name}] coarse={coarse.status}")
    capture = capture_joint_graph(request)
    ir, ir_report = build_joint_ir(capture)
    if not ir_report.valid:
        raise ValueError(f"invalid IR: {ir_report.errors}")
    print(f"[{task.name}] backend={capture.backend} ops={len(ir.ops)} values={len(ir.values)} storages={len(ir.storages)}")

    plans = _manual_default_plans(ir, memory_budget_bytes, safety_margin_bytes=0)
    plan_rows = []
    shared = None
    for plan in plans:
        lowered = partition_joint_graph(capture.joint_module, plan, ir)
        trace = simulate_lowered_fx_l2_event_trace(lowered, fixed_timeline)
        summary = summarize_lowered_fx_l2_event_trace(trace)
        peak_bytes = max((row.get("bytes") or 0) for row in trace)
        peak_time_us = max((row.get("time_us") or 0) for row in trace)
        fw_ops = sum(1 for _ in lowered.fw_graph.graph.nodes)
        bw_ops = sum(1 for _ in lowered.bw_graph.graph.nodes)
        mat_bytes = sum(
            (row.get("payload_bytes") or 0)
            for row in trace
            if "materialize" in str(row.get("event", "")) or row.get("event") in ("allocate",)
        )
        # Main search simulator (shared compiler facts in IR, per-candidate replay):
        sim = simulate_plan(ir, plan, fixed_timeline, cost_provider=None)
        row = {
            "plan": plan.plan_id,
            "saved": len(plan.saved_value_ids),
            "dropped": len(ir.values) - len(plan.saved_value_ids | plan.mandatory_value_ids),
            "sim_peak_mib": round(sim.estimated_peak_bytes / MIB, 3),
            "sim_step_us": round(sim.estimated_step_us, 2),
            "l2_peak_mib": round(peak_bytes / MIB, 3),
            "l2_step_us": round(peak_time_us, 2),
            "fw_ops": fw_ops,
            "bw_ops": bw_ops,
            "trace_rows": len(trace),
            "materialized_payload_mib": round(mat_bytes / MIB, 3),
            "summary_kind": summary.get("kind") if isinstance(summary, dict) else None,
        }
        plan_rows.append(row)
        if plan.plan_id == "all_save":
            shared = row
        print(f"[{task.name} {plan.plan_id}] sim_peak={row['sim_peak_mib']} MiB l2_peak={row['l2_peak_mib']} MiB "
              f"sim_step={row['sim_step_us']}us fw={fw_ops} bw={bw_ops} trace={len(trace)} "
              f"saved={row['saved']} dropped={row['dropped']}")

    comparisons = []
    for row in plan_rows:
        l2_delta = row["l2_peak_mib"] - shared["l2_peak_mib"]
        l2_ratio = l2_delta / max(shared["l2_peak_mib"], 1e-9)
        sim_delta = row["sim_peak_mib"] - shared["sim_peak_mib"]
        sim_ratio = sim_delta / max(shared["sim_peak_mib"], 1e-9)
        step_ratio = (row["sim_step_us"] - shared["sim_step_us"]) / max(shared["sim_step_us"], 1e-9)
        comparisons.append(
            {
                "plan": row["plan"],
                "l2_peak_delta_vs_shared_pct": round(l2_ratio * 100, 3),
                "sim_peak_delta_vs_shared_pct": round(sim_ratio * 100, 3),
                "sim_step_delta_vs_shared_pct": round(step_ratio * 100, 2),
                "fw_op_delta": row["fw_ops"] - shared["fw_ops"],
                "bw_op_delta": row["bw_ops"] - shared["bw_ops"],
            }
        )
        print(f"[{task.name} cmp {row['plan']}] l2_peak_delta={l2_ratio*100:.2f}% "
              f"sim_peak_delta={sim_ratio*100:.2f}% sim_step={step_ratio*100:.1f}% "
              f"fw_delta={row['fw_ops']-shared['fw_ops']} bw_delta={row['bw_ops']-shared['bw_ops']}")

    payload = {
        "schema": "plan_invariance_v1",
        "task": task.name,
        "backend": capture.backend,
        "ir": {"ops": len(ir.ops), "values": len(ir.values), "storages": len(ir.storages)},
        "plans": plan_rows,
        "comparisons_vs_shared_all_save": comparisons,
    }
    Path(out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["bert", "resnet"], default="bert")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=1000)
    ap.add_argument("--out", type=str, default="plan_invariance_result.json")
    args = ap.parse_args()
    task = build_task(args.task, hidden=args.hidden, layers=args.layers, seq=args.seq, vocab=args.vocab)
    run_task(task, out=args.out, batch=args.batch)


if __name__ == "__main__":
    main()
