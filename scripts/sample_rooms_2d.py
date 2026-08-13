"""Write the P3-2 geometry + config YAMLs.

    python scripts/sample_rooms_2d.py [--out-dir configs/sweeps_2d_mat]

Emits, for each split, a YAML carrying the geometries plus everything the simulator and
loader need. The config LIST is not stored -- it is derived from the schema by
``aaf.data.mat_configs.enumerate_configs`` so the counts can never drift from the code.

The test geometry file is FROZEN once written: it is reused by every P3-2 evaluation, so
regenerating it would silently invalidate cross-run comparisons. The script refuses to
overwrite it unless --force is given.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from aaf.data.mat_configs import (
    HELDOUT_COMBOS,
    UNSEEN_ALPHA,
    coverage_report,
    enumerate_configs,
)
from aaf.data.sample_rooms_2d import (
    DEFAULT_LHS_SEED,
    DEFAULT_RANGES_2D,
    DEFAULT_TEST_SEED,
    nn_distance_report,
    sample_test_geometries,
    sample_train_geometries,
)
from aaf.walls import ALPHA_BASELINE, NON_BASELINE_MATERIALS, WALLS_2D

FS, N_TIME, MAX_ORDER = 4096, 8192, 60
SOURCE_POS = [0.5, 0.5]
N_RX_PER_SIDE, RX_MARGIN = 8, 0.3


def _payload(set_name, geometries, extra):
    d = {
        "set_name": set_name,
        "walls": list(WALLS_2D),
        "alpha_baseline": ALPHA_BASELINE,
        "materials": list(NON_BASELINE_MATERIALS),
        "fs": FS,
        "n_time_samples": N_TIME,
        "max_order": MAX_ORDER,          # P3-2 fixes this (D45); NOT the auto rule
        "source_pos": SOURCE_POS,
        "n_rx_per_side": N_RX_PER_SIDE,
        "rx_margin": RX_MARGIN,
        "ranges_L": list(DEFAULT_RANGES_2D[0]),
        "ranges_W": list(DEFAULT_RANGES_2D[1]),
        "geometries": [{"L": float(L), "W": float(W)} for L, W in geometries],
    }
    d.update(extra)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="configs/sweeps_2d_mat")
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--force", action="store_true", help="allow overwriting the FROZEN test set")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_geoms = sample_train_geometries(n=a.n_train, seed=DEFAULT_LHS_SEED)
    test_geoms = sample_test_geometries(train_geoms, n=a.n_test, seed=DEFAULT_TEST_SEED)

    train_path = out / "p3_2_train.yaml"
    test_path = out / "p3_2_test_frozen.yaml"
    if test_path.exists() and not a.force:
        raise SystemExit(f"{test_path} exists and is FROZEN; pass --force to overwrite")

    train_path.write_text(yaml.safe_dump(_payload(
        "p3_2_train", train_geoms,
        {"holdout_combos": [{"wall": w, "material": m} for w, m in HELDOUT_COMBOS],
         "lhs_seed": DEFAULT_LHS_SEED,
         "note": "training set: baseline + 12 single-wall edits - 2 held-out combos"},
    ), sort_keys=False))

    test_path.write_text(yaml.safe_dump(_payload(
        "p3_2_test_frozen", test_geoms,
        {"holdout_combos": [],
         "unseen_alpha": UNSEEN_ALPHA,
         "test_seed": DEFAULT_TEST_SEED,
         "frozen_note": "FROZEN — reused by every P3-2 eval; do NOT modify",
         "nn_to_train": nn_distance_report(test_geoms, train_geoms)},
    ), sort_keys=False))

    train_cfgs = enumerate_configs(train_geoms, exclude_combos=HELDOUT_COMBOS)
    test_cfgs = enumerate_configs(test_geoms)
    split_ii = enumerate_configs(train_geoms, only_combos=HELDOUT_COMBOS, include_baseline=False)
    split_iv = enumerate_configs(test_geoms, unseen_alpha=UNSEEN_ALPHA)

    summary = {
        "n_train_geometries": len(train_geoms),
        "n_test_geometries": len(test_geoms),
        "counts": {"train": len(train_cfgs), "test_i_and_iii": len(test_cfgs),
                   "split_ii": len(split_ii), "split_iv": len(split_iv),
                   "total_sims": len(train_cfgs) + len(test_cfgs) + len(split_ii) + len(split_iv)},
        "coverage_train": coverage_report(train_cfgs),
        "nn_to_train": nn_distance_report(test_geoms, train_geoms),
        "heldout_combos": [list(c) for c in HELDOUT_COMBOS],
        "unseen_alpha": UNSEEN_ALPHA,
    }
    (out / "p3_2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {train_path}\nwrote {test_path}")


if __name__ == "__main__":
    main()
