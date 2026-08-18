"""Is the reported LSD measuring the physics, or the nulls?

Track A/A2/B all plateaued at 4.5-5.7 dB band-limited LSD, which reads as a poor fit against
P3-2b's ~1.0 dB. But that comparison was already invalidated once (the FDTD corpus spans ~75-88
dB against the ISM corpus's ~22 dB), and a stronger objection follows: measured over 12 Track A
test configs, 97.3% of in-band bins sit MORE THAN 40 dB BELOW their config's own peak. Plain
LSD averages |20log10|pred| - 20log10|gt|| over every bin equally, so it is dominated by deep
nulls and inter-modal floor -- regions where the log-difference is large, unstable, and
physically uninteresting, because nobody hears a -70 dB null.

This recomputes LSD over bins ABOVE a floor set per config relative to its own peak, at several
thresholds, so the reported number can be read as "error on the content that carries the
physics" rather than "error on the nulls". Raw LSD is reported beside every floored value; the
floor is a REPORTING choice and is never used to select a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from aaf.data.seg_configs import configs_from_rows
from aaf.eval.p3_2_eval import band_limit, load_gt, load_model
from aaf.eval.p3_2b_eval import render_config_arm

FLOORS_DB = (None, -60.0, -50.0, -40.0, -30.0, -20.0)
BAND_HI_HZ, DF = 300.0, 0.5


def lsd_floored(pred: np.ndarray, gt: np.ndarray, floor_db):
    """Mean |dB(pred) - dB(gt)| over bins where GT is above `floor_db` from its own peak."""
    p = np.abs(np.asarray(pred)); g = np.abs(np.asarray(gt))
    gdb = 20.0 * np.log10(np.maximum(g, 1e-30))
    pdb = 20.0 * np.log10(np.maximum(p, 1e-30))
    if floor_db is None:
        mask = np.ones_like(gdb, dtype=bool)
    else:
        mask = gdb >= (gdb.max() + floor_db)
    if not mask.any():
        return float("nan"), 0.0
    return float(np.mean(np.abs(pdb[mask] - gdb[mask]))), float(mask.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="configs/sweeps_2d_mat/p3_3fast_trackA_manifest.json")
    ap.add_argument("--data-dir", default="data/track_p3_3fast_A")
    ap.add_argument("--out", default="outputs/p3_3fast/floored_lsd.json")
    ap.add_argument("--limit", type=int, default=24)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, renderer, cfg, meta, it = load_model(Path(a.checkpoint), dev)
    model.eval(); renderer.eval()          # renderer.eval() is required -- D49 C3
    cond_source = str(cfg["cond_source"])
    hi = int(round(BAND_HI_HZ / DF)) + 1

    rows = [r for r in json.load(open(a.manifest))["configs"] if r["split"] == "test"]
    cfgs = configs_from_rows(rows, split="test")[:a.limit]
    per, agg = [], {str(f): [] for f in FLOORS_DB}
    for c in cfgs:
        H_gt, rx, src, _ = load_gt(Path(a.data_dir) / c.filename)
        gt = band_limit(H_gt, hi)
        with torch.no_grad():
            pr = render_config_arm(model, renderer, cond_source, c.L, c.W,
                                   list(c.alphas), rx, src, dev)
        pr = band_limit(np.asarray(pr)[:, :gt.shape[1]], hi)
        rec = {"label": c.label, "kind": c.kind}
        for f in FLOORS_DB:
            v, frac = lsd_floored(pr, gt, f)
            rec["lsd_raw" if f is None else "lsd_floor{:.0f}".format(f)] = v
            rec["frac_raw" if f is None else "frac_floor{:.0f}".format(f)] = frac
            agg[str(f)].append(v)
        per.append(rec)

    summary = {}
    for f in FLOORS_DB:
        v = np.array(agg[str(f)], float); v = v[np.isfinite(v)]
        summary["raw" if f is None else "floor_{:.0f}_db".format(f)] = {
            "mean_lsd_db": float(v.mean()), "median_lsd_db": float(np.median(v)),
            "n_configs": int(v.size),
            "mean_frac_bins_used": float(np.mean(
                [r["frac_raw" if f is None else "frac_floor{:.0f}".format(f)] for r in per])),
        }
    out = {"checkpoint": a.checkpoint, "iter": int(it), "cond_source": cond_source,
           "n_configs": len(per), "floors_db": [str(f) for f in FLOORS_DB],
           "summary": summary,
           "note": ("the floor is a REPORTING choice, never a model-selection criterion; raw is "
                    "reported beside every floored value. 97.3% of in-band bins sit >40 dB "
                    "below their config's peak, so plain LSD is dominated by nulls."),
           "per_config": per}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print("{:14s} {:>10s} {:>12s} {:>14s}".format("floor", "mean LSD", "median", "bins used"))
    for k, v in summary.items():
        print("{:14s} {:10.3f} {:12.3f} {:13.1%}".format(
            k, v["mean_lsd_db"], v["median_lsd_db"], v["mean_frac_bins_used"]))
    print("-> {}".format(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
