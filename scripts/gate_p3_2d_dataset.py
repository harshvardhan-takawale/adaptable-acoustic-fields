"""P3-2d dataset gate H1-H6, run per grid before any training compute is committed.

Deliberately NOT P3-2b's dataset gate. That one gates a CONTINUOUS sampler: no draw inside a
held-out slab, no draw on a preset, and a coverage test that bins M_RANGE into 12 cells and
demands max/mean <= 2.0 outside the slab. Every one of those is wrong here:

  * there is no slab (midpoints are the hold-out), so the slab test is vacuous;
  * a grid value is chosen, not drawn, so "never lands on a preset" is a property of the grid
    and belongs at construction;
  * the coverage test FAILS BY CONSTRUCTION on a coarse grid -- 4 values leave 8 of 12 bins
    empty -- so it would reject a perfectly correct dataset.

The grid analogue of coverage is "every draw is ON the grid and every grid value is used".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from aaf.data.mat_configs_cont import configs_from_rows, m_of_alpha
from aaf.data.mat_configs_grid import (
    GRID_ORDER,
    GRID_SPECS,
    M_BASELINE,
    PRESET_ALPHA_EPS,
    TRAINED_VALUE_TOL,
    assert_grid_invariants,
    grid_alphas,
    midpoints,
    near_preset_grid_values,
    realized_delta,
)
from aaf.data.mat_configs import PRESET_ALPHAS
from aaf.walls import WALL_INDEX

MANIFEST_FMT = "configs/sweeps_2d_mat/p3_2d_{run}_manifest.json"
EXPECTED_MIX = {"baseline": 40, "single": 440, "two": 320, "four": 160}


def gate_run(run: str, data_dir: Path) -> dict:
    man = json.loads(Path(MANIFEST_FMT.format(run=run)).read_text())
    n = man["n_grid_points"]
    train = configs_from_rows(man["configs"], split="train")

    # ---- H1/H2: on-grid, all values used, no duplicate filenames, exact mix
    try:
        rep = assert_grid_invariants(train, n)
        h12 = {"pass": True, "detail": {k: rep[k] for k in
                                        ("n_configs", "kinds", "n_grid_values_used",
                                         "n_edited_draws", "per_wall_n")}}
    except AssertionError as e:
        h12 = {"pass": False, "error": str(e)}
    mix_ok = h12.get("detail", {}).get("kinds") == EXPECTED_MIX
    h2 = {"pass": bool(h12["pass"] and mix_ok), "expected_mix": EXPECTED_MIX,
          "got_mix": h12.get("detail", {}).get("kinds")}

    # ---- H3: no grid value may collide with a preset alpha (filename-token flip)
    ga = grid_alphas(n)
    collide = [{"alpha": a, "preset": p} for a in ga for p in PRESET_ALPHAS
               if abs(a - p) <= PRESET_ALPHA_EPS]
    h3 = {"pass": not collide, "collisions": collide,
          "note": ("a grid alpha within 1e-9 of a preset would format as 2 dp instead of 6 dp "
                   "and the training room would share a filename with a frozen test room")}

    # ---- H4: grid values merely CLOSE to a preset -- not an error, but that preset's S1/S4
    #          configs stop being a genuine hold-out in this run alone.
    near = near_preset_grid_values(n)
    h4 = {"pass": True, "near_preset": near,
          "affected_presets": sorted({x["preset_alpha"] for x in near}),
          "action": ("excluded from the cross-run S1/S4 comparison and reported" if near
                     else "none")}

    # ---- H5: midpoints must be far from every trained value, including the baseline
    mps = midpoints(n)
    train_m = sorted({m_of_alpha(c.alphas[WALL_INDEX[w]]) for c in train for w in c.edited}
                     | {M_BASELINE})
    rows = []
    for mp in mps:
        d = min(abs(mp["m"] - x) for x in train_m)
        rows.append({**mp, "d_to_any_trained_m": float(d)})
    headline = [r for r in rows if r["headline"]]
    h5 = {"pass": bool(headline) and all(r["d_to_any_trained_m"] > TRAINED_VALUE_TOL
                                         for r in headline),
          "n_midpoints": len(rows), "n_headline": len(headline),
          "n_excluded_as_trained_value": len(rows) - len(headline),
          "min_d_headline": min((r["d_to_any_trained_m"] for r in headline), default=None),
          "midpoints": rows}

    # ---- H6: every simulation present
    missing = [r["filename"] for r in man["configs"]
               if not (data_dir / (r["filename"] + ".done")).exists()]
    h6 = {"pass": not missing, "n_missing": len(missing), "missing": missing[:10],
          "n_configs": len(man["configs"])}

    res = {
        "run": run, "n_grid_points": n,
        "realized_delta_m": realized_delta(n),
        "nominal_delta_m": man["nominal_delta_m"],
        "H1_on_grid": h12, "H2_count_and_mix": h2, "H3_preset_collision": h3,
        "H4_near_preset": h4, "H5_midpoint_margin": h5, "H6_sims_built": h6,
        "rows_sha256": man["rows_sha256"],
    }
    res["pass"] = all(res[k]["pass"] for k in res if k.startswith("H"))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=list(GRID_ORDER))
    ap.add_argument("--data-dir", default="data/track_c_2d")
    ap.add_argument("--out", default="outputs/p3_2d/dataset_gate.json")
    a = ap.parse_args()
    results = [gate_run(r, Path(a.data_dir)) for r in a.runs]
    for r in results:
        flags = " ".join("{}:{}".format(k.split("_")[0], "ok" if v["pass"] else "FAIL")
                         for k, v in r.items() if isinstance(v, dict) and "pass" in v)
        print("[{:6s}] {}  realD={:.4f}  {}".format(
            r["run"], "PASS" if r["pass"] else "FAIL", r["realized_delta_m"], flags))
        h5 = r["H5_midpoint_margin"]
        print("         midpoints {} ({} headline, {} excluded as trained-value), "
              "min margin {:.4f}".format(h5["n_midpoints"], h5["n_headline"],
                                         h5["n_excluded_as_trained_value"],
                                         h5["min_d_headline"] or float("nan")))
        if r["H4_near_preset"]["near_preset"]:
            print("         H4 near-preset: {}".format(
                ", ".join("m={:.4f}~a{:.2f}(d={:.4f})".format(
                    x["grid_m"], x["preset_alpha"], x["d_m"])
                    for x in r["H4_near_preset"]["near_preset"])))
    ok = all(r["pass"] for r in results)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"pass": ok, "runs": results}, open(out, "w"), indent=1)
    print("\n{} -> {}".format("GATE PASS" if ok else "GATE FAIL", out))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
