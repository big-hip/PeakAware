#!/usr/bin/env python3
"""P0-3: Component ablation of the PeakAware simulator against the GPU anchor.

The GPU anchor is the frozen 15-candidate artifact
(search_efficiency_15candidate_diverse_5pass_a6000_20260802): 4 tasks x 15
candidate plans x 5 passes of allocator-visible measured peaks.

Design
------
For each of the 4 tasks we re-run the capture -> IR -> default-plan pipeline,
rebuild the simulator's fixed timeline from the artifact's recorded
cost_breakdown.memory_components (so the reconstructed simulator matches the
frozen artifact's code version), then simulate every candidate both in the full
model and with one mechanism disabled at a time.  The full-model estimate is
checked against the artifact's recorded estimated_peak_bytes per candidate;
only candidates that reproduce within ``TOL`` are reported (reconstruction
guard).  The ablated estimates are compared against the artifact's measured
peak (median of the 5 passes).

Mechanisms
----------
  opt             optimizer-state + optimizer-temporary materialization
  workspace       per-op workspace bytes from the cost provider
  materialization compiler mandatory-workspace + runtime-replica residency
  lifetime        last-use release in forward liveness and replay liveness
  replay          all-simultaneous M0 recompute model (no live-range)
  alias           view/alias storage sharing (off = count logical bytes each)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.api import _hardware_spec, _request_key
from peakaware.capture import capture_joint_graph
from peakaware.config import PeakAwareConfig
from peakaware.contracts import FixedTimeline, TrainingRequest
from peakaware.cost.composite import build_composite_provider
from peakaware.ir import build_joint_ir
from peakaware.memory.fixed_frontier import analyze_coarse_feasibility, build_optimizer_spec
from peakaware.memory.simulator import simulate_plan
from peakaware.models.registry import TrainingTaskRegistry
from peakaware.plugins import ServiceKind, build_default_registry
from peakaware.search.engine import _greedy_seed_plans, _manual_default_plans

MIB = 1024 * 1024

ARTIFACT = (
    ROOT
    / "artifacts/search_efficiency_15candidate_diverse_5pass_a6000_20260802"
    / "exhaustive_records.json"
)

# Task budget (bytes) from the artifact run_config.json; key = registry name.
TASK_BUDGETS = {
    "bert_base": 181403648,
    "gpt2": 354418688,
    "resnet50": 600834048,
    "vit_b_16": 2106589184,
}
TASK_BATCH = {"bert_base": 1, "gpt2": 1, "resnet50": 1, "vit_b_16": 4}

# Reconstruction tolerance: only report (task, candidate) pairs whose full-model
# estimate reproduces the artifact's estimate within this relative fraction.
TOL = 0.08

ABLATIONS = ["opt", "workspace", "materialization", "lifetime", "replay", "alias"]


def load_anchor() -> dict[str, dict]:
    rows = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    by_budget: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_budget[row["budget_bytes"]].append(row)
    name_for_budget = {v: k for k, v in TASK_BUDGETS.items()}
    anchor: dict[str, dict] = {}
    for budget, records in by_budget.items():
        name = name_for_budget[budget]
        # per-candidate GPU truth: median measured peak across the 5 passes
        per_candidate: dict[str, dict] = {}
        for rec in records:
            for cand in rec["measured_plan_results"]:
                pid = cand["plan_id"]
                per_candidate.setdefault(
                    pid,
                    {"measured": [], "estimated": [], "phase": []},
                )
                per_candidate[pid]["measured"].append(cand["measured_peak_bytes"])
                per_candidate[pid]["estimated"].append(cand["estimated_peak_bytes"])
                per_candidate[pid]["phase"].append(cand.get("measured_peak_phase"))
        # fixed timeline components from the all_save record's memory_components
        all_save = next(
            cand
            for rec in records
            for cand in rec["measured_plan_results"]
            if cand["plan_id"] == "all_save"
        )
        mc = all_save.get("cost_breakdown", {}).get("memory_components", {})
        anchor[name] = {
            "budget_bytes": budget,
            "per_candidate": {
                pid: {
                    "measured_median": float(sorted(v["measured"])[len(v["measured"]) // 2]),
                    "measured": v["measured"],
                    "estimated_artifact": float(sorted(v["estimated"])[len(v["estimated"]) // 2]),
                }
                for pid, v in per_candidate.items()
            },
            "fixed": {
                "parameter_bytes": mc.get("parameter_bytes", 0),
                "buffer_bytes": mc.get("buffer_bytes", 0),
                "gradient_bytes": mc.get("gradient_bytes", 0),
                "optimizer_state_bytes": mc.get("optimizer_state_bytes", 0),
                "optimizer_temporary_bytes": mc.get("optimizer_temporary_bytes", 0),
                "mandatory_workspace_bytes": mc.get("mandatory_workspace_bytes", 0),
                "runtime_replica_bytes": mc.get("runtime_replica_bytes", 0),
            },
        }
    return anchor


def build_pipeline(name: str, budget_bytes: int, batch: int, registry: TrainingTaskRegistry):
    torch.manual_seed(20260802)
    random.seed(20260802)
    task = registry.get(name)
    model = task.build_model().to("cuda")
    model.train()
    args, kwargs = task.build_batch(batch)
    args = tuple(a.to("cuda") if torch.is_tensor(a) else a for a in args)
    kwargs = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in kwargs.items()}
    optimizer = task.build_optimizer(model)
    config = PeakAwareConfig(enable_compile=False, enable_inductor=False)
    optimizer_spec = build_optimizer_spec(optimizer, model)
    request = TrainingRequest(
        model=model,
        example_args=args,
        example_kwargs=kwargs,
        loss_fn=task.loss_fn,
        optimizer=optimizer,
        memory_budget_bytes=budget_bytes,
        config=config,
        optimizer_spec=optimizer_spec,
        hardware=_hardware_spec(args, kwargs),
        request_key=_request_key(model, args, kwargs, budget_bytes, config),
    )
    capture = capture_joint_graph(request)
    ir, ir_report = build_joint_ir(capture)
    if not ir_report.valid:
        raise ValueError(f"invalid IR: {ir_report.errors}")
    # composite cost provider (atencost analytical + profile_db if any)
    registry_svc = build_default_registry(profile_db_path=config.profile_db_path)
    provider = build_composite_provider(
        tuple(record.service for record in registry_svc.services_for(ServiceKind.COST_PROVIDER))
    )
    plans = (
        _manual_default_plans(ir, budget_bytes, safety_margin_bytes=0)
        + _greedy_seed_plans(
            ir,
            budget_bytes,
            safety_margin_bytes=0,
            cost_provider=provider,
            max_candidates=12,
        )
    )
    return model, optimizer, ir, plans, provider, capture.capture_key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="bert_base,gpt2,resnet50,vit_b_16")
    ap.add_argument("--out", default="ablation_component_result.json")
    ap.add_argument("--max-candidates", type=int, default=15)
    args = ap.parse_args()

    anchor = load_anchor()
    registry = TrainingTaskRegistry.with_defaults()
    rows: list[dict] = []
    for name in args.tasks.split(","):
        budget = TASK_BUDGETS[name]
        batch = TASK_BATCH[name]
        info = anchor[name]
        model, optimizer, ir, plans, provider, cap_key = build_pipeline(name, budget, batch, registry)

        # Rebuild the fixed timeline to match the frozen artifact exactly.
        fixed = FixedTimeline(**info["fixed"])
        print(
            f"[{name}] budget={budget/2**20:.0f}MiB batch={batch} ops={len(ir.ops)} "
            f"storages={len(ir.storages)} fixed_resident={fixed.resident_bytes/2**20:.1f}MiB "
            f"opt_state={fixed.optimizer_state_bytes/2**20:.1f}MiB opt_temp={fixed.optimizer_temporary_bytes/2**20:.1f}MiB"
        )

        for plan in plans[: args.max_candidates]:
            pid = plan.plan_id
            gcand = info["per_candidate"].get(pid)
            if gcand is None:
                print(f"  skip {pid}: not in artifact")
                continue
            # Full model (reconstruction verification + reference).
            full = simulate_plan(ir, plan, fixed, cost_provider=provider, materialize_event_trace=False)
            est_artifact = gcand["estimated_artifact"]
            rel = abs(full.estimated_peak_bytes - est_artifact) / max(est_artifact, 1e-9)
            measured = gcand["measured_median"]
            reproduce = rel <= TOL
            row = {
                "task": name,
                "plan": pid,
                "reproduced": reproduce,
                "full_est": full.estimated_peak_bytes,
                "artifact_est": est_artifact,
                "reproduction_rel_error": rel,
                "measured_peak": measured,
                "full_rel_error": abs(full.estimated_peak_bytes - measured) / max(measured, 1e-9),
                "ablated": {},
            }
            if not reproduce:
                print(
                    f"  [{pid}] NOT reproduced: full={full.estimated_peak_bytes/2**20:.1f}MiB "
                    f"artifact={est_artifact/2**20:.1f}MiB rel={rel*100:.1f}% (>{TOL*100:.0f}%)"
                )
                rows.append(row)
                continue
            for comp in ABLATIONS:
                ablated = simulate_plan(
                    ir,
                    plan,
                    fixed,
                    cost_provider=provider,
                    materialize_event_trace=False,
                    components={comp: False},
                )
                rel_err = abs(ablated.estimated_peak_bytes - measured) / max(measured, 1e-9)
                row["ablated"][comp] = {
                    "est": ablated.estimated_peak_bytes,
                    "rel_error": rel_err,
                    "phase": ablated.peak_snapshot.phase,
                }
            print(
                f"  [{pid}] reproduced ok full={full.estimated_peak_bytes/2**20:.1f}MiB "
                f"meas={measured/2**20:.1f}MiB full_rel={row['full_rel_error']*100:.1f}%"
            )
            rows.append(row)
        del model, optimizer, ir
        torch.cuda.empty_cache()

    # ---- Aggregation ----
    agg = {"total": len(rows), "reproduced": sum(1 for r in rows if r["reproduced"])}
    repr_rows = [r for r in rows if r["reproduced"]]
    if repr_rows:
        agg["full_mean_abs_rel_error"] = sum(r["full_rel_error"] for r in repr_rows) / len(repr_rows)
        for comp in ABLATIONS:
            errs = [r["ablated"][comp]["rel_error"] for r in repr_rows]
            agg[f"{comp}_mean_abs_rel_error"] = sum(errs) / len(errs)
            # error delta: ablated - full
            deltas = [
                r["ablated"][comp]["rel_error"] - r["full_rel_error"] for r in repr_rows
            ]
            agg[f"{comp}_delta_full"] = sum(deltas) / len(deltas)
            agg[f"{comp}_worsened_count"] = sum(
                1 for r in repr_rows if r["ablated"][comp]["rel_error"] > r["full_rel_error"]
            )
        # per-task
        by_task: dict[str, list] = defaultdict(list)
        for r in repr_rows:
            by_task[r["task"]].append(r)
        agg["by_task"] = {
            task: {
                "count": len(rs),
                "full_mean_abs_rel_error": sum(r["full_rel_error"] for r in rs) / len(rs),
                **{
                    f"{comp}_mean_abs_rel_error": sum(r["ablated"][comp]["rel_error"] for r in rs) / len(rs)
                    for comp in ABLATIONS
                },
            }
            for task, rs in by_task.items()
        }
    payload = {
        "schema": "ablation_component_v1",
        "tolerance": TOL,
        "aggregate": agg,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print("\n=== AGGREGATE (reproduced rows only) ===")
    print(f"reproduced {agg['reproduced']}/{agg['total']}")
    if repr_rows:
        print(f"full MAPE: {agg['full_mean_abs_rel_error']*100:.2f}%")
        for comp in ABLATIONS:
            print(
                f"{comp:16s}: MAPE {agg[f'{comp}_mean_abs_rel_error']*100:6.2f}%  "
                f"delta {agg[f'{comp}_delta_full']*100:+6.2f}pp  worsened "
                f"{agg[f'{comp}_worsened_count']}/{len(repr_rows)}"
            )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
