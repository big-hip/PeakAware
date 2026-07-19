from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peakaware.cost.base import OpSignature
from peakaware.cost.collector import collect_microbenchmark
from peakaware.cost.profile_db import ProfileDB


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _profile_row(
    *,
    name: str,
    target: str,
    fn: Callable[..., torch.Tensor],
    args: tuple[torch.Tensor, ...],
    warmup: int,
    repeats: int,
    db: ProfileDB,
) -> dict[str, object]:
    output = fn(*args)
    signature = OpSignature(
        op_name=name,
        target=target,
        input_bytes=sum(_tensor_nbytes(arg) for arg in args),
        output_bytes=_tensor_nbytes(output),
        dtype=str(output.dtype).removeprefix("torch."),
    )
    result = collect_microbenchmark(signature, fn, args, warmup=warmup, repeats=repeats, db=db)
    return {
        "op_name": signature.op_name,
        "target": signature.target,
        "dtype": signature.dtype,
        "input_bytes": signature.input_bytes,
        "output_bytes": signature.output_bytes,
        "sample_count": result.record.sample_count,
        "p50_us": result.record.p50_us,
        "p90_us": result.record.p90_us,
        "mean_us": result.record.mean_us,
        "workspace_bytes": result.record.workspace_bytes,
    }


def _db_row_count(path: Path) -> int:
    with sqlite3.connect(path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM op_profile").fetchone()[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a small ProfileDB smoke artifact.")
    parser.add_argument("--profile-db", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--size", type=int, default=32)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)
    db = ProfileDB(args.profile_db)
    x = torch.randn(args.size, args.size, device=device)
    y = torch.randn(args.size, args.size, device=device)
    rows = [
        _profile_row(
            name="add",
            target="aten.add",
            fn=torch.add,
            args=(x, y),
            warmup=args.warmup,
            repeats=args.repeats,
            db=db,
        ),
        _profile_row(
            name="matmul",
            target="aten.matmul",
            fn=torch.matmul,
            args=(x, y),
            warmup=args.warmup,
            repeats=args.repeats,
            db=db,
        ),
    ]
    payload = {
        "device": args.device,
        "profile_db": str(args.profile_db),
        "profile_db_row_count": _db_row_count(args.profile_db),
        "rows": rows,
        "torch_version": torch.__version__,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
