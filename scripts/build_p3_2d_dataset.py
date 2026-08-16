"""Build the P3-2d simulations that do not already exist.

The five grids share their frozen preset test rows with P3-2b (reused byte-for-byte) and
share nothing else, since a different interval assigns different values. A single
de-duplicated worklist across all five manifests is therefore both correct and minimal, and
`.done` sentinels make it idempotent under preemption.

The worklist returns the FULL stable list. Filtering on `.done` here would make it shrink as
the build progresses, so array tasks starting later would slice a different list and the
index->config mapping would race -- that bug left 79 of 479 P3-2c configs unbuilt while every
task reported COMPLETED. `_build_one` already short-circuits on the sentinel, so skipping is
free and belongs there. `pending_only=True` is for reporting only, never for indexing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from aaf.data.mat_configs_cont import MatConfigM
from aaf.data.mat_configs_grid import GRID_ORDER
from scripts.build_2d_mat_dataset import _build_one, make_receiver_grid_2d  # noqa: F401

MANIFESTS = [f"configs/sweeps_2d_mat/p3_2d_{r}_manifest.json" for r in GRID_ORDER]


def worklist(manifests=MANIFESTS, data_dir="data/track_c_2d", pending_only=False):
    d, seen, out = Path(data_dir), set(), []
    for mf in manifests:
        for r in json.load(open(mf))["configs"]:
            fn = r["filename"]
            if fn in seen:
                continue
            seen.add(fn)
            if pending_only and (d / (fn + ".done")).exists():
                continue
            out.append(MatConfigM(L=float(r["L"]), W=float(r["W"]),
                                  alphas=tuple(float(x) for x in r["alphas"]),
                                  kind=r["kind"], edited=tuple(r["edited"]),
                                  split=r["split"], geom_id=int(r["geom_id"])))
    out.sort(key=lambda c: c.filename)          # stable across array tasks
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--out-dir", default="data/track_c_2d")
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()
    common = yaml.safe_load(open("configs/sweeps_2d_mat/p3_2_train.yaml"))
    work = worklist(data_dir=a.out_dir)                       # full, stable, index-safe
    if a.plan or a.idx is None:
        pending = worklist(data_dir=a.out_dir, pending_only=True)
        n_tasks = (len(work) + a.chunk - 1) // a.chunk
        print(json.dumps({"n_total": len(work), "n_pending": len(pending),
                          "chunk": a.chunk,
                          "array_range": f"0-{max(n_tasks - 1, 0)}"}, indent=1))
        return
    lo = a.idx * a.chunk
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, cfg in enumerate(work[lo:lo + a.chunk]):
        _build_one(lo + j, cfg.split, cfg, common, out_dir, False)


if __name__ == "__main__":
    main()
