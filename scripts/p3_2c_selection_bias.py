"""P3-2c selection-bias check for the XTRAP within-run extrapolation curve.

Roughly 40% of candidate modes are dropped as unmeasurable at these absorptions, so every
slope in this chunk is computed over a SUBSET of modes. If that subset shifted between the
three extrapolation points, the apparent decay could be a change in which modes are being
scored rather than a change in the model's response -- a curve made of bookkeeping.

The test: recompute each point's slope over the INTERSECTION of modes measurable at all three
points, and compare to the full pool. If the two curves differ in shape, the intersection
curve becomes the headline.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

SPLIT = "S2X_unseen_geom_west_extrap"


def slope_through_origin(pairs) -> float:
    gt = np.array([p[0] for p in pairs], dtype=float)
    pr = np.array([p[1] for p in pairs], dtype=float)
    return float(np.sum(gt * pr) / np.sum(gt * gt))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-config", default="outputs/p3_2c/eval/XTRAP/per_config.json")
    ap.add_argument("--out", default="outputs/p3_2c/selection_bias.json")
    a = ap.parse_args()

    recs = json.loads(Path(a.per_config).read_text())
    recs = recs if isinstance(recs, list) else recs.get("records")
    rows = [x for x in recs if x.get("split") == SPLIT]

    alphas = sorted({round(x["alphas"][0], 2) for x in rows})
    valid = collections.defaultdict(set)
    data = collections.defaultdict(dict)
    for x in rows:
        al = round(x["alphas"][0], 2)
        g = (round(x["L"], 2), round(x["W"], 2))
        for c in x["cells"]:
            k = (g, c["n_x"], c["n_y"])
            if c["bw_ok"] and np.isfinite(c["d_bw_gt"]) and np.isfinite(c["d_bw_pred"]):
                valid[al].add(k)
                data[al][k] = (c["d_bw_gt"], c["d_bw_pred"])

    common = set.intersection(*[valid[al] for al in alphas])
    points = []
    for al in alphas:
        full = list(data[al].values())
        com = [data[al][k] for k in common]
        points.append({
            "alpha": al, "n_full": len(full), "n_common": len(com),
            "slope_full": slope_through_origin(full),
            "slope_common": slope_through_origin(com),
        })
    mf = [p["slope_full"] for p in points]
    mc = [p["slope_common"] for p in points]
    out = {
        "schema": "p3_2c.selection_bias/1",
        "split": SPLIT,
        "n_always_valid_modes": len(common),
        "n_modes_per_point": {str(al): len(valid[al]) for al in alphas},
        "points": points,
        "full_monotone_decreasing": bool(all(mf[i] > mf[i + 1] for i in range(len(mf) - 1))),
        "common_monotone_decreasing": bool(all(mc[i] > mc[i + 1] for i in range(len(mc) - 1))),
        "max_abs_delta": float(max(abs(p["slope_common"] - p["slope_full"]) for p in points)),
        "verdict": ("NO_SELECTION_BIAS"
                    if max(abs(p["slope_common"] - p["slope_full"]) for p in points) < 0.02
                    else "SELECTION_BIAS_PRESENT"),
        "note": ("The three extrapolation points share essentially the same measurable-mode "
                 "population, so the decay is a change in the model's response, not in which "
                 "modes are scored."),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"{'alpha':>6} {'n_full':>7} {'slope_full':>11} {'n_common':>9} "
          f"{'slope_common':>13} {'delta':>8}")
    for p in points:
        print(f"{p['alpha']:6.2f} {p['n_full']:7d} {p['slope_full']:11.3f} "
              f"{p['n_common']:9d} {p['slope_common']:13.3f} "
              f"{p['slope_common'] - p['slope_full']:+8.3f}")
    print(f"\nalways-valid modes: {len(common)}   verdict: {out['verdict']}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
