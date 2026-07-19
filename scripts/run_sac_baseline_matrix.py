from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.sac_baseline import run_sac_baseline_matrix, write_sac_baseline_json


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _split_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _split_csv(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure a PyTorch SAC baseline matrix for PeakAware tasks.")
    parser.add_argument("--tasks", default="tiny_mlp_w8_d3")
    parser.add_argument("--microbatches", default="1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--measurement-warmup-steps", type=int, default=0)
    parser.add_argument("--measurement-repeats", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    payload = run_sac_baseline_matrix(
        task_names=_split_csv(args.tasks),
        microbatch_sizes=_split_int_csv(args.microbatches),
        device=args.device,
        warmup_steps=args.measurement_warmup_steps,
        repeats=args.measurement_repeats,
    )
    if args.output_json is not None:
        write_sac_baseline_json(payload, args.output_json)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
