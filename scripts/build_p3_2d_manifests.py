"""Write the five P3-2d grid manifests (one per sampling interval).

Each manifest carries 960 grid-sampled training rows plus a test set that is the FROZEN
P3-2b 210 rows (so S1/S4/S5 stay exactly the P3-2b populations and their simulations are
reused byte-for-byte) *plus* this grid's midpoint rows, which are the actual hold-outs.

The midpoint rows are what make the sweep a measurement rather than an interpolation: they
sit maximally far from every training value, so the score at interval Delta is the score at
the worst case that interval admits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from aaf.data.mat_configs_cont import manifest_rows, rows_sha256
from aaf.data.mat_configs_grid import (
    GRID_ORDER,
    GRID_SPECS,
    assert_grid_invariants,
    build_grid,
    enumerate_midpoint_test_configs,
    grid_alphas,
    midpoints,
    near_preset_grid_values,
    realized_delta,
    sample_train_configs_grid,
)

TRAIN_YAML = "configs/sweeps_2d_mat/p3_2_train.yaml"
TEST_YAML = "configs/sweeps_2d_mat/p3_2_test_frozen.yaml"
FROZEN_MANIFEST = "configs/sweeps_2d_mat/p3_2b_manifest.json"
OUT_FMT = "configs/sweeps_2d_mat/p3_2d_{run}_manifest.json"


def _geoms(path: str):
    return [(round(float(g["L"]), 2), round(float(g["W"]), 2))
            for g in yaml.safe_load(Path(path).read_text())["geometries"]]


def build(run: str) -> dict:
    n = GRID_SPECS[run]
    train_geoms, test_geoms = _geoms(TRAIN_YAML), _geoms(TEST_YAML)

    train = sample_train_configs_grid(train_geoms, n)
    report = assert_grid_invariants(train, n)
    mids = enumerate_midpoint_test_configs(test_geoms, n)

    # The frozen P3-2b test rows, carried verbatim so S1/S4/S5 remain the P3-2b populations
    # and their .h5 files are reused rather than rebuilt.
    frozen = [dict(r) for r in json.loads(Path(FROZEN_MANIFEST).read_text())["configs"]
              if r["split"] == "test"]

    rows = manifest_rows(train, mids)
    # manifest_rows re-indexes from 0 over (train + mids); splice the frozen rows in after,
    # then renumber so `i` is dense and unique across the whole manifest.
    train_rows = [r for r in rows if r["split"] == "train"]
    mid_rows = [r for r in rows if r["split"] == "test"]
    for r in mid_rows:
        r["kind"] = "midpoint"
    allrows = train_rows + frozen + mid_rows
    for k, r in enumerate(allrows):
        r["i"] = k

    return {
        "schema": "p3_2d.manifest/1",
        "run": run,
        "n_grid_points": n,
        "nominal_delta_m": float(run[1:]) / 100.0,
        "realized_delta_m": realized_delta(n),
        "delta_axis_note": (
            "realized_delta_m is the reported x. The run name is a LABEL: anchoring n points "
            "inclusively on M_RANGE is what reproduces the intended counts, and it makes the "
            "realized interval differ from the nominal one (D53(c))."),
        "grid_m": build_grid(n),
        "grid_alpha": grid_alphas(n),
        "midpoints": midpoints(n),
        "near_preset_grid_values": near_preset_grid_values(n),
        "n_train": len(train_rows),
        "n_test_frozen_preset": len(frozen),
        "n_test_midpoint": len(mid_rows),
        "train_report": report,
        "frozen_test_source": FROZEN_MANIFEST,
        "rows_sha256": rows_sha256(allrows),
        "configs": allrows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=list(GRID_ORDER))
    args = ap.parse_args()
    for run in args.runs:
        man = build(run)
        out = Path(OUT_FMT.format(run=run))
        out.write_text(json.dumps(man, indent=1))
        nh = sum(1 for mp in man["midpoints"] if mp["headline"])
        print("{:6s} n={:2d} realD={:.4f} | train {} + frozen {} + midpoint {} | "
              "headline midpoints {}/{} | sha {}".format(
                  run, man["n_grid_points"], man["realized_delta_m"], man["n_train"],
                  man["n_test_frozen_preset"], man["n_test_midpoint"], nh,
                  len(man["midpoints"]), man["rows_sha256"][:12]))
        print("       -> {}".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
