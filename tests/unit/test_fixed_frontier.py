import torch
from torch import nn

from peakaware.memory.fixed_frontier import analyze_coarse_feasibility, build_optimizer_spec


def test_adamw_fixed_memory_estimate_before_state_materialization():
    model = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    spec = build_optimizer_spec(optimizer, model)
    fixed, report = analyze_coarse_feasibility(model, optimizer, 1 << 30)

    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    assert spec.name == "AdamW"
    assert spec.state_bytes == 2 * parameter_bytes
    assert fixed.optimizer_temporary_bytes == parameter_bytes
    assert report.status == "FEASIBLE"
