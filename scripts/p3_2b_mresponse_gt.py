"""Generate the P3-2b ground-truth m-response: 3 geometries x 4 walls x 20 absorptions.

CPU only (pyroomacoustics ISM, ~4-5 s per config) and deliberately shardable, so it can run
as a small array next to the GPU training arms without competing for a GPU.

    # one shard of a 4-way array (3 cells = 60 simulations, ~6 min)
    python scripts/p3_2b_mresponse_gt.py --shard 0 --n-shards 4
    # merge the shards into the single artifact everything downstream reads
    python scripts/p3_2b_mresponse_gt.py --merge
    # GT-vs-theory calibration table (the evidence the verdict rests on)
    python scripts/p3_2b_mresponse_gt.py --verify
    # or all of it in one ~20 min process
    python scripts/p3_2b_mresponse_gt.py --all

Only per-mode bandwidths are persisted (~a few hundred kB). The 240 fields themselves are
transient: nothing downstream reads them, and writing them would add ~4 GB of HDF5 to a
corpus that already has 3300 files.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from aaf.eval.p3_2b_mresponse import (
    DATA_DIR,
    GT_JSON,
    KAPPA,
    format_calibration,
    gt_calibration,
    gt_document,
    gt_geometry,
    merge_gt,
    select_geometries,
    sim_common,
    sweep_points,
)
from aaf.walls import WALLS_2D

PARTS_DIR = "outputs/p3_2b/mresponse_parts"
CALIB_JSON = "outputs/p3_2b/mresponse_gt_calibration.json"


def cells(n_geoms: int):
    """The flat (geometry index, wall) work list -- 12 cells, 20 simulations each."""
    return [(gi, w) for gi in range(n_geoms) for w in WALLS_2D]


def run_shard(shard: int, n_shards: int, data_dir: str, parts_dir: str,
              n_points: int) -> Path:
    geoms = select_geometries()
    points = sweep_points()[:n_points] if n_points else sweep_points()
    common = sim_common()
    mine = cells(len(geoms))[shard::n_shards]
    by_geom = {}
    for gi, wall in mine:
        by_geom.setdefault(gi, []).append(wall)
    print("[shard {}/{}] {} cells: {}".format(
        shard, n_shards, len(mine),
        ", ".join("{}:{}".format(geoms[g]["role"], w) for g, w in mine)), flush=True)

    out_geoms = []
    for gi in sorted(by_geom):
        out_geoms.append(gt_geometry(geoms[gi], by_geom[gi], points, common, data_dir))
    doc = gt_document(out_geoms, points, common)
    doc["shard"] = {"index": shard, "n_shards": n_shards,
                    "cells": [[g, w] for g, w in mine]}
    Path(parts_dir).mkdir(parents=True, exist_ok=True)
    p = Path(parts_dir) / "part_{:02d}_of_{:02d}.json".format(shard, n_shards)
    p.write_text(json.dumps(doc, default=float))
    print("[shard {}] wrote {}".format(shard, p), flush=True)
    return p


def do_merge(parts_dir: str, out: str) -> dict:
    parts = [json.loads(p.read_text()) for p in sorted(Path(parts_dir).glob("part_*.json"))]
    if not parts:
        raise SystemExit(f"no part_*.json under {parts_dir}")
    doc = merge_gt(parts)
    n_cells = sum(len(g["walls"]) for g in doc["geometries"])
    expected = len(doc["geometries"]) * len(WALLS_2D)
    if n_cells != expected:
        raise SystemExit("merged {} cells, expected {} -- a shard is missing".format(
            n_cells, expected))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, default=float))
    n_sim = sum(1 for g in doc["geometries"] for w in g["walls"].values()
                for p in w["points"] if p["source"] == "simulated")
    n_re = sum(1 for g in doc["geometries"] for w in g["walls"].values()
               for p in w["points"] if p["source"] == "reused")
    print("[merge] {} geometries x {} walls, {} points ({} simulated, {} reused) -> {} "
          "({:.0f} kB)".format(len(doc["geometries"]), len(WALLS_2D), n_sim + n_re, n_sim,
                               n_re, out, Path(out).stat().st_size / 1024.0))
    return doc


def do_verify(gt_path: str, calib_path: str) -> int:
    doc = json.load(open(gt_path))
    rows = gt_calibration(doc, kappa=KAPPA)
    print(format_calibration(rows))
    worst = max(abs(r["pct_err_vs_kappa_theory"]) for r in rows)
    drops = [r for r in rows if r["n_points_with_dropped_modes"]]
    print("\nworst |a_fit/a_theory - 1| = {:.2f}%  over {} cells".format(worst, len(rows)))
    if drops:
        print("estimator breakdown (points where a mode's -3 dB width went unresolvable):")
        for r in drops:
            ms = ", ".join("m={m} ({n_modes}/{n_modes_total})".format(**d)
                           for d in r["dropped"])
            print("  {:<14s} {:<5s} L{:.2f} W{:.2f}: {} of {} points -- {}".format(
                r["role"], r["wall"], r["L"], r["W"], r["n_points_with_dropped_modes"],
                r["n_points"], ms))
    else:
        print("estimator breakdown: none -- every mode resolved at every sweep point")
    Path(calib_path).parent.mkdir(parents=True, exist_ok=True)
    Path(calib_path).write_text(json.dumps(
        {"schema": "p3_2b.m_response_gt_calibration/1", "kappa": KAPPA,
         "gt_source": gt_path, "worst_abs_pct_err": worst, "cells": rows},
        indent=1, default=float))
    print("[verify] wrote {}".format(calib_path))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="P3-2b ground-truth m-response sweep")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--n-shards", type=int, default=4)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--all", action="store_true", help="every shard, then merge + verify")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--parts-dir", default=PARTS_DIR)
    ap.add_argument("--out", default=GT_JSON)
    ap.add_argument("--calib", default=CALIB_JSON)
    ap.add_argument("--n-points", type=int, default=0,
                    help="truncate the sweep (smoke tests only; 0 = all 20)")
    ap.add_argument("--plan", action="store_true", help="print the work plan and exit")
    a = ap.parse_args()

    if a.plan:
        geoms = select_geometries()
        pts = sweep_points()[:a.n_points] if a.n_points else sweep_points()
        print(json.dumps({
            "geometries": geoms, "n_walls": len(WALLS_2D), "n_points": len(pts),
            "alphas": [p[0] for p in pts], "m": [round(p[1], 4) for p in pts],
            "n_simulations": len(geoms) * len(WALLS_2D) * len(pts),
            "cells": len(geoms) * len(WALLS_2D),
            "array_range": "0-{}".format(a.n_shards - 1),
        }, indent=1))
        return 0

    t0 = time.time()
    if a.all:
        for s in range(a.n_shards):
            run_shard(s, a.n_shards, a.data_dir, a.parts_dir, a.n_points)
        do_merge(a.parts_dir, a.out)
        do_verify(a.out, a.calib)
    elif a.shard is not None:
        run_shard(a.shard, a.n_shards, a.data_dir, a.parts_dir, a.n_points)
    elif a.merge:
        do_merge(a.parts_dir, a.out)
        if a.verify:
            do_verify(a.out, a.calib)
    elif a.verify:
        do_verify(a.out, a.calib)
    else:
        ap.error("nothing to do: pass --shard, --merge, --verify, --all or --plan")
    print("[total] {:.1f}s".format(time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
