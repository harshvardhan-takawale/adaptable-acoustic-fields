"""P3-2b acceptance gate. Frozen thresholds, hashed, emitted before any figure is drawn.

The thresholds below were fixed in the chunk spec BEFORE any P3-2b arm was evaluated. They
are hashed and the hash is written into ``verdict.json`` and into the one-line verdict
string, and ``tests/test_p3_2b_accept.py`` pins the literal hash. Editing a number here
therefore breaks CI and shows up as a diff on every artifact that carries the hash --
which is the point. P3-2's failure was not that a threshold was too strict; it was that
"~13% of the edit magnitude" had no pre-registered line to fall on the wrong side of.

The gate is evaluated on **S2 only** (unseen geometry x held-out slab combo). S1/S3/S4/S5
are diagnostics: S1 says the model renders unseen rooms, S3 isolates combo novelty from
geometry novelty, S4 says the material axis is continuous, S5 says two edits superpose.
None of them can substitute for S2, and a strong S1 with a dead S2 is precisely the P3-2
result being re-tested.

Criteria (all four must hold):

  edit_bw_slope   >= 0.80   predicted delta-BW regressed on GT delta-BW, pooled over modes
  edit_bw_pearson >= 0.80   the same association, scale-free -- slope alone can be inflated
                            by a few large-|delta| modes
  edit_gain        > 1.00   the model's EDITED render must beat its own BASELINE render as
                            an explanation of the edited ground truth. Below 1.0 the
                            material channel is worse than useless.
  |rho - 1|       <= 0.25   the physics number: fitted d(BW)/d(m) over kappa-scaled ISM-ray
                            theory, on the slab walls. rho is taken from
                            slope_fit.aggregate.own_family.slab_local.rho_median.

Blockers (measurability, not performance) force ``passed = False`` and are reported
separately from criterion failures, because "the arm is wrong" and "the arm cannot be
measured" are different findings and must not be confused in a results table:

  n_cells                 >= 40   paired-valid mode observations in S2
  frac_modes_dropped      <= 0.5  in S2 and in the slab_local slope group

Without the second blocker an arm could pass by rendering peaks so degraded that the
estimator rejects nearly all of them, leaving a handful of survivors that happen to line
up. That is the failure mode most likely to look like success.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional

import numpy as np

# ------------------------------------------------------------------ FROZEN. Do not edit.
THRESHOLDS: Dict[str, object] = {
    "spec": "p3_2b.accept/1",
    "split": "S2_unseen_geom_slab",
    "edit_bw_slope_min": 0.80,
    "edit_bw_pearson_min": 0.80,
    "edit_gain_min_exclusive": 1.00,
    "rho_abs_dev_max": 0.25,
    "rho_source": "slope_fit.aggregate.own_family.slab_local.rho_median",
    "blocker_min_n_cells": 40,
    "blocker_max_frac_modes_dropped": 0.5,
}


def thresholds_sha256(thresholds: Optional[Dict[str, object]] = None) -> str:
    t = THRESHOLDS if thresholds is None else thresholds
    return hashlib.sha256(
        json.dumps(t, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _crit(name: str, value, op: str, threshold: float, note: str) -> dict:
    v = _f(value)
    if not np.isfinite(v):
        ok = False
    elif op == ">=":
        ok = v >= threshold
    elif op == ">":
        ok = v > threshold
    elif op == "<=":
        ok = v <= threshold
    else:
        raise ValueError("unknown comparison {!r}".format(op))
    return {"name": name, "value": v, "op": op, "threshold": float(threshold),
            "pass": bool(ok), "note": note}


def verdict(arm: str, s2: dict, slope_fit: dict, iter_: Optional[int] = None,
            mid_training: bool = False) -> dict:
    """Build the acceptance verdict. ``s2`` is ``summary["splits"]["S2_..."]``.

    A nan value fails its criterion. Nothing here is softenable at call time: the caller
    passes measurements, not thresholds.
    """
    slab = ((slope_fit or {}).get("aggregate", {}).get("own_family", {})
            .get("slab_local", {}))
    rho = _f(slab.get("rho_median"))
    edit = (s2 or {}).get("edit", {})

    criteria = [
        _crit("edit_bw_slope", edit.get("edit_bw_slope"), ">=",
              float(THRESHOLDS["edit_bw_slope_min"]),
              "predicted delta-BW regressed on GT delta-BW, pooled over S2 modes"),
        _crit("edit_bw_pearson", edit.get("edit_bw_pearson"), ">=",
              float(THRESHOLDS["edit_bw_pearson_min"]),
              "association between predicted and GT delta-BW on S2"),
        _crit("edit_gain", edit.get("edit_gain"), ">",
              float(THRESHOLDS["edit_gain_min_exclusive"]),
              "LSD(model baseline render vs edited GT) / LSD(model edited render vs edited GT)"),
        _crit("abs_rho_minus_1", abs(rho - 1.0) if np.isfinite(rho) else float("nan"), "<=",
              float(THRESHOLDS["rho_abs_dev_max"]),
              "|rho - 1| with rho = a_fit / a_theory (kappa-scaled) from {}".format(
                  THRESHOLDS["rho_source"])),
    ]

    n_cells = _f((s2 or {}).get("n_cells"))
    frac_s2 = _f((s2 or {}).get("frac_modes_dropped"))
    frac_slab = _f(slab.get("frac_modes_dropped"))
    max_drop = float(THRESHOLDS["blocker_max_frac_modes_dropped"])
    blockers: List[dict] = []
    if not np.isfinite(n_cells) or n_cells < float(THRESHOLDS["blocker_min_n_cells"]):
        blockers.append({"name": "insufficient_S2_cells", "value": n_cells,
                         "threshold": float(THRESHOLDS["blocker_min_n_cells"]),
                         "note": "fewer than 40 paired-valid mode observations in S2"})
    if np.isfinite(frac_s2) and frac_s2 > max_drop:
        blockers.append({"name": "S2_modes_unmeasurable", "value": frac_s2,
                         "threshold": max_drop,
                         "note": "over half of S2's candidate modes failed paired validity"})
    if np.isfinite(frac_slab) and frac_slab > max_drop:
        blockers.append({"name": "slab_slope_modes_unmeasurable", "value": frac_slab,
                         "threshold": max_drop,
                         "note": "over half of the slab_local slope-fit modes were dropped"})

    passed = bool(all(c["pass"] for c in criteria) and not blockers)
    failed = [c["name"] for c in criteria if not c["pass"]]
    sha = thresholds_sha256()

    bits = []
    for c in criteria:
        bits.append("{}={:.3f}{}{:.2f}{}".format(
            c["name"], c["value"], c["op"], c["threshold"], "" if c["pass"] else " FAIL"))
    tag = "PASS" if passed else "FAIL"
    prefix = "MID-TRAINING (iter {}) ".format(iter_) if mid_training and iter_ is not None \
        else ("iter {} ".format(iter_) if iter_ is not None else "")
    one_line = "P3-2b {} {}{}: {} | {} | blockers: {} | thr {}".format(
        arm, prefix, THRESHOLDS["split"], tag, "; ".join(bits),
        ",".join(b["name"] for b in blockers) or "none", sha[:12])

    return {
        "arm": arm,
        "iter": iter_,
        "mid_training": bool(mid_training),
        "split": THRESHOLDS["split"],
        "passed": passed,
        "criteria": {c["name"]: c for c in criteria},
        "criteria_failed": failed,
        "blockers": blockers,
        "rho_used": rho,
        "thresholds": dict(THRESHOLDS),
        "thresholds_sha256": sha,
        "one_line": one_line,
    }
