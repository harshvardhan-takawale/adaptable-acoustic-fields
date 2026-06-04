"""Pick a 10-room maximin subset of the 45 LHS training rooms.

Used by Chunk P2-2.5 diagnostic runs A and C, which share the same 10-room
subset (their only difference is batch / coverage / n_pts_per_ray, so the
room set must be identical).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aaf.data.sample_rooms_3d import (
    DEFAULT_RANGES,
    Room3D,
    read_rooms_yaml,
    select_diag_subset_maximin,
    write_rooms_yaml,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-yaml", default=str(REPO_ROOT / "configs/sweeps_3d/train_rooms.yaml"),
        help="The full 45-room LHS training rooms YAML.",
    )
    ap.add_argument(
        "--output-yaml", default=str(REPO_ROOT / "configs/sweeps_3d/diag_10rooms.yaml"),
    )
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    payload = read_rooms_yaml(args.input_yaml)
    full = [Room3D(L=float(r["L"]), W=float(r["W"]), H=float(r["H"])) for r in payload["rooms"]]
    print(f"# input rooms: {len(full)} from {args.input_yaml}")

    subset = select_diag_subset_maximin(
        full, n=args.n, ranges=DEFAULT_RANGES, box_center_first=True,
    )
    print(f"# picked {len(subset)} maximin-spread rooms:")
    for i, r in enumerate(subset):
        print(f"  [{i}] L={r.L:.2f} W={r.W:.2f} H={r.H:.2f}")

    p = write_rooms_yaml(
        args.output_yaml,
        subset,
        set_name="diag_subset",
        extra_meta={
            "source_yaml": str(Path(args.input_yaml).name),
            "selection": "greedy maximin in normalized [0,1]^3, seeded at box center",
            "n_input": len(full),
        },
    )
    print(f"# wrote {p}")


if __name__ == "__main__":
    main()
