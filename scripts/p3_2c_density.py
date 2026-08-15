"""P3-2c: turn the five arm evals into the density curve, the breakpoint, and the rule.

The question: how wide a hole in the absorption axis can the model bridge? Each arm withheld
a wider band of west absorptions; everything else -- geometry, the other three walls, the
sampler, the recipe, the estimator, the thresholds -- is identical by construction.

Three things this script is careful about, each of which could otherwise manufacture a curve:

* **x is the REALIZED gap, not the nominal slab width.** Draws do not land on slab edges, so
  the nominal width is only an upper bound. Reported alongside is ``d_support`` -- the actual
  distance from the test point to the nearest training draw -- which is the quantity the model
  experiences and comes out at almost exactly half the gap.
* **north is the within-run control.** Its slab is identical in every arm and its draws are
  byte-identical, so any drift in the north slope is drift in TRAINING REALIZATION, not in the
  thing being manipulated. Pre-registered tolerance: 0.15. Without it, a one-seed-per-arm sweep
  cannot separate an effect from run-to-run noise.
* **The breakpoint is never extrapolated.** If the slope never crosses the threshold inside the
  sampled range, that is reported as a bound, not as a fitted crossing.

XTRAP is deliberately NOT on the same axis: its test points sit beyond all training support,
so it carries a per-point beyond-edge distance instead of a single interior gap. Placing it on
the interior-gap axis would put the extrapolation arm at the dense end of the curve.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from aaf.eval.p3_2b_splits import S2
from aaf.eval.p3_2c_splits import S2X_EXTRAP, curve_point

SWEEP = ("W015", "W030", "W060", "W100")
XTRAP = "XTRAP"

SLOPE_THRESHOLD = 0.80          # the frozen P3-2b acceptance threshold
RHO_TOLERANCE = 0.25
NORTH_CONTROL_TOL = 0.15        # pre-registered
M_SPAN = 1.5852                 # sampled m range, for the draws-per-wall rule

# P3-2's own failing holdout, for annotation. Discrete presets + raw-alpha conditioning at an
# effective gap of ~1.04 scored 0.133 -- the comparison that gives this chunk its punchline.
P3_2_REFERENCE = {"gap_m": 1.04, "slope": 0.133,
                  "note": "P3-2: discrete presets, raw-alpha conditioning"}

BOOT_SEED = 20260815
N_BOOT = 4000


def _summary(root: Path, arm: str) -> dict:
    return json.loads((root / arm / "summary.json").read_text())


def _wall_cells(s: dict, wall: str) -> List[dict]:
    """Own-family, fitted, finite-rho cells for one wall -- one per geometry."""
    return [c for c in s["slope_fit"]["per_cell"]
            if c["wall"] == wall and c["own_family"] and c["fitted"]
            and np.isfinite(c["rho"])]


def _combo_slope(s: dict, split: str, key: str) -> Optional[float]:
    pc = (s["splits"].get(split) or {}).get("per_combo") or {}
    e = (pc.get(key) or {}).get("edit") or {}
    v = e.get("edit_bw_slope")
    return float(v) if v is not None and np.isfinite(v) else None


def arm_row(root: Path, arm: str) -> dict:
    s = _summary(root, arm)
    west, north = _wall_cells(s, "west"), _wall_cells(s, "north")
    slab = s["slope_fit"]["aggregate"]["own_family"]["slab_local"]
    s2 = s["splits"][S2]
    cp = curve_point(arm)

    row = {
        "arm": arm,
        "axis": cp["axis"],
        "x_gap_m": cp["x"],
        "gap_detail": cp["detail"],
        # headline: the west measurement
        "west_edit_slope": _combo_slope(s, S2, "west0.50"),
        "west_rho_median": float(np.median([c["rho"] for c in west])) if west else float("nan"),
        "west_n_cells": len(west),
        "west_rho_per_geom": {f"{c['L']:.2f}x{c['W']:.2f}": float(c["rho"]) for c in west},
        # control: north, identical slab and identical draws in every arm
        "north_edit_slope": _combo_slope(s, S2, "north0.70"),
        "north_rho_median": float(np.median([c["rho"] for c in north])) if north else float("nan"),
        "north_n_cells": len(north),
        "north_rho_per_geom": {f"{c['L']:.2f}x{c['W']:.2f}": float(c["rho"]) for c in north},
        # pooled S2 (west+north), which is what the acceptance gate scores
        "s2_edit_slope": float(s2["edit"]["edit_bw_slope"]),
        "s2_rho_slab_local": float(slab["rho_median"]),
        "s2_frac_modes_dropped": float(slab.get("frac_modes_dropped", float("nan"))),
        "s2_n_cells": int(slab.get("n_cells", 0)),
        "in_dist_val_lsd_db": s.get("in_dist_val_lsd_db"),
        "verdict": "PASS" if "PASS" in s["verdict"]["one_line"] else "FAIL",
    }
    p3c = (s.get("meta") or {}).get("p3_2c") or {}
    ann = p3c.get("annotations") or {}
    ds = [a["per_wall"]["west"]["d_support_m"]
          for a in ann.values()
          if "west" in a.get("per_wall", {})
          and abs(a["per_wall"]["west"]["alpha"] - 0.50) < 1e-9]
    row["west_d_support_m"] = float(np.median(ds)) if ds else None
    row["s2_west_is_arm_holdout"] = (p3c.get("s2_designation") or {}).get(
        "s2_west_is_arm_holdout")
    return row


def xtrap_row(root: Path) -> dict:
    """XTRAP's own axis: per-alpha beyond-edge distance, not an interior gap."""
    s = _summary(root, XTRAP)
    cp = curve_point(XTRAP)
    pts = []
    for p in cp["detail"]["points"]:
        key = "west{:.2f}".format(p["alpha"])
        sl = _combo_slope(s, S2X_EXTRAP, key)
        pts.append({"alpha": p["alpha"], "m": p["m"],
                    "beyond_edge_m": p["beyond_edge_m"], "edit_slope": sl})
    base = arm_row(root, XTRAP)
    base.update({
        "axis": "beyond_edge",
        "x_gap_m": None,
        "extrap_points": pts,
        "train_edge_m": cp["detail"]["train_edge_m"],
        "note": ("west@0.50 (m=0.6931) is BELOW this arm's exclusion threshold of 1.10 and is "
                 "therefore TRAINED here -- its S2 west entry is an interpolation control, "
                 "not a holdout. The extrapolation measurement is S2X."),
    })
    return base


# --------------------------------------------------------------------------- breakpoint
def _crossing(xs: List[float], ys: List[float], thr: float) -> Optional[float]:
    """First x where y crosses from >= thr to < thr, by linear interpolation."""
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if y0 >= thr > y1:
            t = (y0 - thr) / (y0 - y1)
            return float(xs[i] + t * (xs[i + 1] - xs[i]))
    return None


def breakpoint_analysis(rows: List[dict], metric: str = "west_rho_median") -> dict:
    """Where the curve crosses the acceptance threshold, with a PAIRED geometry bootstrap.

    Paired: one geometry index set is drawn and applied to EVERY arm, because the arms share
    their test geometries and it is the shape ACROSS arms that the breakpoint depends on.
    Resampling each arm independently would break that pairing and inflate the interval.
    """
    sweep = [r for r in rows if r["axis"] == "interior_gap"]
    sweep.sort(key=lambda r: r["x_gap_m"])
    xs = [r["x_gap_m"] for r in sweep]
    ys = [r[metric] for r in sweep]

    point = _crossing(xs, ys, SLOPE_THRESHOLD)
    geoms = sorted(set.intersection(*[set(r["west_rho_per_geom"]) for r in sweep]))
    rng = np.random.default_rng(BOOT_SEED)
    boots, n_cross = [], 0
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(geoms), len(geoms))
        picked = [geoms[i] for i in idx]
        yb = [float(np.median([r["west_rho_per_geom"][g] for g in picked])) for r in sweep]
        c = _crossing(xs, yb, SLOPE_THRESHOLD)
        if c is not None:
            boots.append(c)
            n_cross += 1
    frac = n_cross / float(N_BOOT)
    out = {
        "metric": metric,
        "threshold": SLOPE_THRESHOLD,
        "x": xs, "y": ys,
        "geometries_used": geoms,
        "n_boot": N_BOOT,
        "frac_bootstrap_with_crossing": frac,
        "censoring_fraction": 1.0 - frac,
        "point_estimate": point,
    }
    if boots:
        out["ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    if point is None:
        # Pre-registered: never interpolate outside the bracket.
        out["verdict"] = "NO_CROSSING_IN_RANGE"
        out["bound"] = f"> {max(xs):.4f}"
        out["statement"] = (
            f"The S2-west curve never crosses {SLOPE_THRESHOLD:.2f} within the sampled range; "
            f"the smallest value observed is {min(ys):.3f} at gap {xs[int(np.argmin(ys))]:.4f}. "
            f"The breakpoint is therefore bounded BELOW at > {max(xs):.4f} and is not "
            f"estimated by interpolation.")
    else:
        out["verdict"] = "CROSSING"
        out["statement"] = f"Breakpoint at realized gap {point:.4f} m."
    return out


def xtrap_breakpoint(rows: List[dict]) -> dict:
    """XTRAP's WITHIN-RUN extrapolation curve -- the one measurement immune to the cross-arm
    realization noise, because all three points come from a single trained model.

    The interior-gap sweep compares five separately-trained models, so it inherits a
    run-to-run noise floor. These three points share every weight; the only thing that varies
    is how far beyond the training edge the test material sits.
    """
    r = next((r for r in rows if r["arm"] == XTRAP), None)
    if not r or not r.get("extrap_points"):
        return {"available": False}
    pts = sorted((p for p in r["extrap_points"] if p["edit_slope"] is not None),
                 key=lambda p: p["beyond_edge_m"])
    xs = [p["beyond_edge_m"] for p in pts]
    ys = [p["edit_slope"] for p in pts]
    cross = _crossing(xs, ys, SLOPE_THRESHOLD)
    monotone = all(ys[i] > ys[i + 1] for i in range(len(ys) - 1))
    return {
        "available": True,
        "axis": "beyond_edge_m",
        "train_edge_m": r.get("train_edge_m"),
        "points": [{"alpha": p["alpha"], "beyond_edge_m": p["beyond_edge_m"],
                    "edit_slope": p["edit_slope"]} for p in pts],
        "monotone_decreasing": bool(monotone),
        "threshold": SLOPE_THRESHOLD,
        "breakpoint_beyond_edge_m": cross,
        "statement": (
            f"Within a single trained model, the edit slope decays monotonically with distance "
            f"beyond the training edge ({' -> '.join(f'{y:.3f}' for y in ys)} at "
            f"+{'/+'.join(f'{x:.3f}' for x in xs)}), crossing {SLOPE_THRESHOLD:.2f} at "
            f"+{cross:.3f} in m." if cross is not None else
            f"No crossing of {SLOPE_THRESHOLD:.2f} within the sampled beyond-edge range."),
    }


def dataset_rule(bp: dict) -> dict:
    """The deliverable sentence: how densely to sample the absorption axis."""
    if bp["verdict"] == "NO_CROSSING_IN_RANGE":
        x = max(bp["x"])
        n = math.ceil(M_SPAN / x)
        return {
            "bounded": True, "delta_m": x, "draws_per_wall": n,
            "sentence": (
                f"No breakpoint was found up to a realized gap of {x:.3f} in m. Sampling the "
                f"absorption axis at intervals no coarser than dm = {x:.3f} -- i.e. >= {n} "
                f"draws per wall over m in [0.02, 1.61] -- is SUFFICIENT, and is an upper "
                f"bound on what is necessary: the true requirement is looser than this and "
                f"was not reached by this sweep."),
        }
    x = bp["point_estimate"]
    n = math.ceil(M_SPAN / x)
    return {"bounded": False, "delta_m": x, "draws_per_wall": n,
            "sentence": (f"Sample the absorption axis at intervals no coarser than "
                         f"dm = {x:.3f}, i.e. >= {n} draws per wall over m in [0.02, 1.61].")}


def north_control(rows: List[dict]) -> dict:
    """Pre-registered interpretability gate. north is identical in every arm by construction,
    so its spread across arms measures training-realization noise, not the manipulation."""
    ref = next(r for r in rows if r["arm"] == "W015")
    ref_v = ref["north_rho_median"]
    per = {}
    ok = True
    for r in rows:
        d = abs(r["north_rho_median"] - ref_v)
        per[r["arm"]] = {"north_rho_median": r["north_rho_median"], "delta_vs_W015": float(d),
                         "within_tol": bool(d <= NORTH_CONTROL_TOL)}
        ok &= bool(d <= NORTH_CONTROL_TOL)
    slopes = [r["north_edit_slope"] for r in rows if r["north_edit_slope"] is not None]
    return {"pass": bool(ok), "tolerance": NORTH_CONTROL_TOL, "reference_arm": "W015",
            "per_arm": per,
            "north_edit_slope_range": [float(min(slopes)), float(max(slopes))] if slopes else None,
            "interpretation": (
                "north's slab and draws are byte-identical across arms, so this spread is the "
                "run-to-run floor. A west effect smaller than it is not attributable to the "
                "manipulation with one seed per arm.")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="outputs/p3_2c/eval")
    ap.add_argument("--out", default="outputs/p3_2c/density.json")
    a = ap.parse_args()
    root = Path(a.eval_root)

    rows = [arm_row(root, arm) for arm in SWEEP]
    if (root / XTRAP / "summary.json").exists():
        rows.append(xtrap_row(root))

    # BOTH metrics, always. The gate scores edit_bw_slope AND |rho-1|, and here they disagree
    # about whether a breakpoint exists at all -- rho never crosses, slope does. Reporting
    # whichever one happens to give a cleaner story would be a choice dressed as a result.
    bp = breakpoint_analysis(rows, metric="west_rho_median")
    bp_slope = breakpoint_analysis(rows, metric="west_edit_slope")
    nc = north_control(rows)
    out = {
        "schema": "p3_2c.density/1",
        "arms": rows,
        "breakpoint": bp,
        "breakpoint_by_slope": bp_slope,
        "breakpoint_metric_agreement": {
            "agree": bp["verdict"] == bp_slope["verdict"],
            "rho": bp["verdict"], "slope": bp_slope["verdict"],
            "note": ("The two gated metrics disagree about whether the curve crosses at all. "
                     "Neither is 'the' answer; both are reported and the disagreement is "
                     "itself evidence that the interior-gap sweep is under-powered."),
        },
        "within_run_extrapolation": xtrap_breakpoint(rows),
        "reportable": {
            "pass": bool(nc["pass"]),
            "note": ("PRE-REGISTERED: the interior-gap curve is reportable only if the north "
                     "control holds. north's slab and draws are byte-identical across arms, "
                     "so its spread IS the one-seed-per-arm realization noise. If the control "
                     "fails, cross-arm differences of comparable size cannot be attributed to "
                     "gap width, and the curve needs replicate seeds before it means anything. "
                     "The within-run XTRAP curve is unaffected -- it is a single model."),
        },
        "dataset_rule": dataset_rule(bp),
        "north_control": nc,
        "p3_2_reference": P3_2_REFERENCE,
        "thresholds": {"slope": SLOPE_THRESHOLD, "rho_tol": RHO_TOLERANCE,
                       "north_control_tol": NORTH_CONTROL_TOL},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)

    print(f"{'arm':6s} {'axis':13s} {'gap_m':>7s} {'d_supp':>7s} "
          f"{'westSlp':>8s} {'westRho':>8s} {'nrthSlp':>8s} {'nrthRho':>8s} {'drop':>6s}  v")
    for r in rows:
        g = f"{r['x_gap_m']:.4f}" if r["x_gap_m"] is not None else "  --   "
        ds = f"{r['west_d_support_m']:.4f}" if r.get("west_d_support_m") else "  --   "
        ws = f"{r['west_edit_slope']:.3f}" if r["west_edit_slope"] is not None else "   -- "
        ns = f"{r['north_edit_slope']:.3f}" if r["north_edit_slope"] is not None else "   -- "
        print(f"{r['arm']:6s} {r['axis']:13s} {g:>7s} {ds:>7s} {ws:>8s} "
              f"{r['west_rho_median']:>8.3f} {ns:>8s} {r['north_rho_median']:>8.3f} "
              f"{r['s2_frac_modes_dropped']:>6.3f}  {r['verdict']}")
    for r in rows:
        for p in r.get("extrap_points", []):
            sl = f"{p['edit_slope']:.3f}" if p["edit_slope"] is not None else "--"
            print(f"       XTRAP west@{p['alpha']:.2f}  beyond_edge=+{p['beyond_edge_m']:.4f}"
                  f"  edit_slope={sl}")
    print(f"\nnorth control: {'PASS' if nc['pass'] else 'FAIL'} "
          f"(tol {nc['tolerance']}), rho spread "
          f"{max(v['delta_vs_W015'] for v in nc['per_arm'].values()):.4f}")
    print(f"breakpoint [rho]  : {bp['verdict']} -- {bp['statement']}")
    print(f"breakpoint [slope]: {bp_slope['verdict']} -- {bp_slope['statement']}")
    print(f"censoring fraction: rho {bp['censoring_fraction']:.3f}  "
          f"slope {bp_slope['censoring_fraction']:.3f}")
    wr = out["within_run_extrapolation"]
    if wr.get("available"):
        print(f"\nWITHIN-RUN (XTRAP, one model): monotone={wr['monotone_decreasing']}  "
              f"{wr['statement']}")
    print(f"\nREPORTABLE: {'yes' if out['reportable']['pass'] else 'NO -- north control failed'}")
    print(f"\nRULE: {out['dataset_rule']['sentence']}")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
