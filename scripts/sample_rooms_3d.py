"""Generate configs/sweeps_3d/{derisk,train,test}_rooms.yaml.

Driver script for ``aaf.data.sample_rooms_3d``. Idempotent — re-runs produce
the same YAMLs because the LHS seed is fixed (DECISIONS.md D14).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aaf.data.sample_rooms_3d import (
    DEFAULT_LHS_SEED,
    DEFAULT_RANGES,
    DEFAULT_REJECT_CUBIC_TOL,
    derisk_rooms,
    sample_test_rooms,
    sample_train_rooms_lhs,
    write_rooms_yaml,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir", default=str(REPO_ROOT / "configs/sweeps_3d"),
        help="Where to write the 3 YAMLs.",
    )
    ap.add_argument("--n-train", type=int, default=45)
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--lhs-seed", type=int, default=DEFAULT_LHS_SEED)
    ap.add_argument("--test-seed", type=int, default=1729)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. De-risk rooms (5, spec-prescribed)
    derisk = derisk_rooms()
    p = write_rooms_yaml(
        out_dir / "derisk_rooms.yaml",
        derisk,
        set_name="derisk",
        extra_meta={"comment": "5 spec-prescribed rooms for the P2-1 single-room overfit experiment."},
    )
    print(f"# wrote {p}  ({len(derisk)} rooms)")

    # 2. Training rooms (LHS, 45 by default)
    train = sample_train_rooms_lhs(
        n=args.n_train,
        ranges=DEFAULT_RANGES,
        seed=args.lhs_seed,
        reject_cubic_tol=DEFAULT_REJECT_CUBIC_TOL,
    )
    p = write_rooms_yaml(
        out_dir / "train_rooms.yaml",
        train,
        set_name="train",
        extra_meta={
            "lhs_seed": args.lhs_seed,
            "ranges_L": list(DEFAULT_RANGES[0]),
            "ranges_W": list(DEFAULT_RANGES[1]),
            "ranges_H": list(DEFAULT_RANGES[2]),
            "reject_cubic_tol": DEFAULT_REJECT_CUBIC_TOL,
        },
    )
    print(f"# wrote {p}  ({len(train)} rooms)")

    # 3. Test rooms (structured interpolative, 8)
    test = sample_test_rooms(
        n=args.n_test,
        lhs_rooms=train,
        ranges=DEFAULT_RANGES,
        seed=args.test_seed,
        candidate_pool=4096,
        box_center_first=True,
    )
    p = write_rooms_yaml(
        out_dir / "test_rooms.yaml",
        test,
        set_name="test",
        extra_meta={
            "test_seed": args.test_seed,
            "comment": (
                "First room = box center. Remainder selected by greedy maximin "
                "vs the LHS training rooms in normalized [0,1]^3 space — "
                "interpolative interior, not on the LHS grid."
            ),
        },
    )
    print(f"# wrote {p}  ({len(test)} rooms)")


if __name__ == "__main__":
    main()
