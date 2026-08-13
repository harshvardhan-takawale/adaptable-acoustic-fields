"""P3-2b cross-arm ablation: build ``ablation.json`` and ``EVAL.md``.

This is the attribution result of the chunk. P3-2 failed compositional transfer (a
never-trained ``(wall, alpha)`` pair recovered ~13% of the edit magnitude) and four
independent causes were identified. Each P3-2b arm removes one of them, so the table below
is read as a ladder rather than as a leaderboard:

    A  preset configs  + geom_alpha_fourier   -- P3-2's data, P3-2b's renderer/eval
    B  continuous configs + geom_alpha_fourier -- A + continuous alpha sampling
    C  continuous configs + m_linear           -- B + the m = -ln(1-alpha) coordinate
    D  single-wall configs + m_linear          -- C without multi-wall training

so ``B - A`` isolates continuous sampling, ``C - B`` isolates the m-coordinate, and
``C - D`` says whether multi-wall training was necessary at all. ``A`` alone isolates the
renderer/eval change against the P3-2 baseline row.

Everything is read from ``outputs/p3_2b/eval/<arm>/summary.json``; nothing is recomputed
from a model here. The acceptance verdict is taken from the arm's own ``verdict`` block
when the eval wrote one, and otherwise recomputed through the FROZEN
:mod:`aaf.eval.p3_2b_accept` -- never with thresholds restated in this file.

Usage
-----
    python scripts/p3_2b_ablation.py                 # real eval artifacts
    python scripts/p3_2b_ablation.py --synthetic     # fabricated inputs, layout check only

``--synthetic`` writes to ``outputs/p3_2b/synthetic/`` (gitignored) and stamps every
artifact with a watermark. Those numbers are fabricated and must never be quoted.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from aaf.eval import p3_2b_accept

WATERMARK = "SYNTHETIC - layout check only"

# The ladder order. This is the ablation's argument structure, not a display preference:
# "first arm to clear S2" is only meaningful walking A -> B -> C, and D hangs off C.
ARMS: Tuple[str, ...] = (
    "p3_2b_A_preset_fourier",
    "p3_2b_B_cont_fourier",
    "p3_2b_C_cont_mlinear",
    "p3_2b_D_single_mlinear",
)
ARM_LETTER = {a: a.split("_")[2] for a in ARMS}
ARM_ROLE = {
    "p3_2b_A_preset_fourier": "P3-2 preset configs, geom_alpha_fourier -- isolates the "
                              "renderer/eval change alone",
    "p3_2b_B_cont_fourier": "continuous configs, geom_alpha_fourier -- adds continuous alpha "
                            "sampling",
    "p3_2b_C_cont_mlinear": "continuous configs, m_linear -- adds the m = -ln(1-alpha) "
                            "coordinate (proposed design)",
    "p3_2b_D_single_mlinear": "single-wall configs, m_linear -- drops multi-wall training",
}

SPLITS: Tuple[str, ...] = (
    "S1_unseen_geom_nonslab_1wall",
    "S2_unseen_geom_slab",
    "S3_seen_geom_slab",
    "S4_unseen_geom_alpha030",
    "S5_unseen_geom_2wall",
)
SPLIT_SHORT = {
    "S1_unseen_geom_nonslab_1wall": "S1 unseen geom, non-slab",
    "S2_unseen_geom_slab": "S2 unseen geom, HELD-OUT slab",
    "S3_seen_geom_slab": "S3 seen geom, HELD-OUT slab",
    "S4_unseen_geom_alpha030": "S4 unseen geom, alpha=0.30",
    "S5_unseen_geom_2wall": "S5 unseen geom, two walls",
}
GATE_SPLIT = str(p3_2b_accept.THRESHOLDS["split"])

# P3-2's split names against their nearest P3-2b analogue. The pairing is by experimental
# ROLE, not by construction: P3-2 held out (wall, preset-material) pairs and P3-2b holds out
# an m-slab, so these rows are context, never a like-for-like comparison. Flagged as such in
# both the JSON and EVAL.md.
P3_2_SPLIT_MAP = {
    "S1_unseen_geom_nonslab_1wall": "i_unseen_geom_seen_combo",
    "S2_unseen_geom_slab": "iii_unseen_geom_heldout_combo",
    "S3_seen_geom_slab": "ii_seen_geom_heldout_combo",
    "S4_unseen_geom_alpha030": "iv_unseen_alpha",
}

# The metrics the ladder is read on: the three S2 gate numbers plus in-distribution fidelity.
LADDER_METRICS = ("s2_edit_bw_slope", "s2_edit_bw_pearson", "s2_edit_gain", "rho_slab_local",
                  "in_dist_val_lsd_db")


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _fin(x) -> bool:
    v = _f(x)
    return bool(np.isfinite(v))


def _fmt(x, nd: int = 3) -> str:
    return ("{:." + str(nd) + "f}").format(_f(x)) if _fin(x) else "n/a"


def _signed(x, nd: int = 3) -> str:
    return ("{:+." + str(nd) + "f}").format(_f(x)) if _fin(x) else "n/a"


# --------------------------------------------------------------------------- verdict shim
def normalize_verdict(v: Optional[dict]) -> dict:
    """One shape for the verdict regardless of which writer produced it.

    ``aaf.eval.p3_2b_accept.verdict`` emits ``criteria`` as a name-keyed dict whose entries
    carry ``pass``, and hashes the frozen thresholds into ``thresholds_sha256``. The chunk
    spec describes a list of entries carrying ``passed`` and a ``spec_sha``. Both are
    accepted here so the table does not depend on which one landed; the values are never
    re-derived, only re-keyed.
    """
    if not v:
        return {"passed": None, "criteria": [], "blockers": [], "one_line": None,
                "spec_sha": None, "criteria_failed": []}
    crit = v.get("criteria", [])
    items = list(crit.values()) if isinstance(crit, dict) else list(crit)
    out_crit = []
    for c in items:
        out_crit.append({
            "name": c.get("name"),
            "value": _f(c.get("value")),
            "op": c.get("op"),
            "threshold": _f(c.get("threshold")),
            "passed": bool(c.get("passed", c.get("pass", False))),
            "note": c.get("note"),
        })
    return {
        "passed": (None if v.get("passed") is None else bool(v.get("passed"))),
        "criteria": out_crit,
        "criteria_failed": v.get("criteria_failed",
                                 [c["name"] for c in out_crit if not c["passed"]]),
        "blockers": v.get("blockers", []),
        "one_line": v.get("one_line"),
        "spec_sha": v.get("spec_sha") or v.get("thresholds_sha256"),
        "spec": (v.get("thresholds") or {}).get("spec"),
        "mid_training": v.get("mid_training"),
    }


# --------------------------------------------------------------------------- row extraction
def slope_group(summary: dict, group: str) -> dict:
    return (((summary.get("slope_fit") or {}).get("aggregate", {})
             .get("own_family", {}).get(group, {})) or {})


def split_row(block: Optional[dict]) -> dict:
    """The per-split columns of the ablation table."""
    if not block:
        return {"present": False}
    edit = block.get("edit", {}) or {}
    fid = block.get("fidelity", {}) or {}
    return {
        "present": True,
        "n_configs": block.get("n_configs"),
        "n_cells": block.get("n_cells"),
        "frac_modes_dropped": _f(block.get("frac_modes_dropped")),
        "edit_bw_slope": _f(edit.get("edit_bw_slope")),
        "edit_bw_pearson": _f(edit.get("edit_bw_pearson")),
        "edit_gain": _f(edit.get("edit_gain")),
        "E_BW_hz": _f(edit.get("E_BW_hz")),
        "E_LVL_db": _f(edit.get("E_LVL_db")),
        "gt_effect_size_hz": _f(edit.get("gt_effect_size_hz")),
        "pred_effect_size_hz": _f(edit.get("pred_effect_size_hz")),
        "band_lsd_db": _f(fid.get("band_lsd_db")),
    }


def arm_row(arm: str, summary_path: Path) -> dict:
    """One ablation row from one arm's ``summary.json``."""
    s = json.loads(summary_path.read_text())
    verdict = normalize_verdict(s.get("verdict"))
    if verdict["passed"] is None and s.get("splits", {}).get(GATE_SPLIT):
        # The eval did not write a verdict. Recompute through the FROZEN gate module rather
        # than restating any threshold here -- the hash in the output must stay comparable.
        verdict = normalize_verdict(p3_2b_accept.verdict(
            arm, s["splits"][GATE_SPLIT], s.get("slope_fit") or {},
            iter_=s.get("iter"), mid_training=bool((s.get("meta") or {}).get("mid_training"))))
        verdict["recomputed"] = True

    meta = s.get("meta") or {}
    sf = s.get("slope_fit") or {}
    row = {
        "arm": arm,
        "letter": ARM_LETTER.get(arm, arm),
        "role": ARM_ROLE.get(arm, ""),
        "present": True,
        "summary_path": str(summary_path),
        "checkpoint": s.get("checkpoint"),
        "iter": s.get("iter"),
        "total_iters": meta.get("total_iters"),
        "mid_training": bool(meta.get("mid_training")),
        "cond_source": s.get("cond_source"),
        "cond_dim": s.get("cond_dim"),
        "in_dist_val_lsd_db": _f(s.get("in_dist_val_lsd_db")),
        "n_train_configs": meta.get("n_train_configs"),
        "manifest_sha": meta.get("manifest_sha"),
        "kappa": _f(sf.get("kappa", meta.get("kappa"))),
        "rho_vs_raw_theory_median": _f(sf.get("rho_vs_raw_theory_median")),
        "splits": {sp: split_row(s.get("splits", {}).get(sp)) for sp in SPLITS},
        "verdict": verdict,
    }
    for group in ("all", "non_slab", "slab_local"):
        g = slope_group(s, group)
        row["rho_" + group] = _f(g.get("rho_median"))
        row["rho_ci95_" + group] = g.get("rho_ci95")
        row["a_fit_" + group] = _f(g.get("a_fit_median"))
        row["a_theory_" + group] = _f(g.get("a_theory_median"))
        row["n_cells_slope_" + group] = g.get("n_cells")
    # Flat aliases so the ladder deltas and the figure-E table read one key each.
    s2 = row["splits"][GATE_SPLIT]
    row["s2_edit_bw_slope"] = s2.get("edit_bw_slope", float("nan"))
    row["s2_edit_bw_pearson"] = s2.get("edit_bw_pearson", float("nan"))
    row["s2_edit_gain"] = s2.get("edit_gain", float("nan"))
    row["rho_slab_local"] = row["rho_slab_local"]
    return row


def missing_row(arm: str) -> dict:
    return {
        "arm": arm, "letter": ARM_LETTER.get(arm, arm), "role": ARM_ROLE.get(arm, ""),
        "present": False,
        "note": "no summary.json yet -- arm not evaluated at the time this table was built",
        "splits": {sp: {"present": False} for sp in SPLITS},
        "verdict": normalize_verdict(None),
    }


def baseline_row(path: Path) -> Optional[dict]:
    """The P3-2 baseline row, split names remapped by experimental role."""
    if not path.exists():
        return None
    s = json.loads(path.read_text())
    splits = {}
    for sp in SPLITS:
        src = P3_2_SPLIT_MAP.get(sp)
        blk = s.get("splits", {}).get(src) if src else None
        r = split_row(blk)
        r["p3_2_source_split"] = src
        splits[sp] = r
    s2 = splits[GATE_SPLIT]
    return {
        "arm": "p3_2_main",
        "letter": "P3-2",
        "role": "P3-2 baseline (preset configs, geom_alpha conditioning) -- the failure this "
                "chunk is diagnosing",
        "present": True,
        "is_baseline": True,
        "comparability": "APPROXIMATE. P3-2 held out (wall, preset-material) pairs; P3-2b "
                         "holds out an m-slab. Splits are matched by role, not by "
                         "construction, and P3-2 ran no slope fit, so rho is unavailable.",
        "summary_path": str(path),
        "checkpoint": s.get("checkpoint"),
        "iter": s.get("iter"),
        "total_iters": (s.get("meta") or {}).get("total_iters"),
        "mid_training": False,
        "cond_source": "geom_alpha",
        "cond_dim": None,
        "in_dist_val_lsd_db": _f(s.get("in_dist_val_lsd_db")),
        "kappa": _f((s.get("meta") or {}).get("theory_slope_ism_ray")),
        "rho_all": float("nan"), "rho_non_slab": float("nan"), "rho_slab_local": float("nan"),
        "rho_ci95_slab_local": None, "a_fit_slab_local": float("nan"),
        "a_theory_slab_local": float("nan"), "rho_vs_raw_theory_median": float("nan"),
        "splits": splits,
        "s2_edit_bw_slope": s2.get("edit_bw_slope", float("nan")),
        "s2_edit_bw_pearson": s2.get("edit_bw_pearson", float("nan")),
        "s2_edit_gain": s2.get("edit_gain", float("nan")),
        "verdict": normalize_verdict(None),
    }


# --------------------------------------------------------------------------- attribution
def delta(a: Optional[dict], b: Optional[dict]) -> Dict[str, float]:
    """``b - a`` on the ladder metrics. NaN wherever either side is missing."""
    out: Dict[str, float] = {}
    for k in LADDER_METRICS:
        va = _f((a or {}).get(k))
        vb = _f((b or {}).get(k))
        out[k] = float(vb - va) if (np.isfinite(va) and np.isfinite(vb)) else float("nan")
    return out


def build_ladder(by_arm: Dict[str, dict], base: Optional[dict]) -> List[dict]:
    A, B, C, D = (by_arm.get(a) for a in ARMS)
    steps = [
        {"step": "renderer + eval protocol", "comparison": "A vs P3-2 baseline",
         "from": "p3_2_main", "to": ARMS[0],
         "isolates": "the P3-2b renderer/eval on P3-2's own data and conditioning family",
         "caveat": "arm A inherits the P3-2 dataset, in which one holdout was an "
                   "EXTRAPOLATION (alpha above every trained value on that wall), so a null "
                   "here cannot separate 'the renderer did not help' from 'that holdout was "
                   "unfair'.",
         "delta": delta(base, A)},
        {"step": "continuous alpha sampling", "comparison": "B - A",
         "from": ARMS[0], "to": ARMS[1],
         "isolates": "960 continuous configs instead of 440 preset configs, same "
                     "geom_alpha_fourier conditioning",
         "caveat": None, "delta": delta(A, B)},
        {"step": "m = -ln(1-alpha) coordinate", "comparison": "C - B",
         "from": ARMS[1], "to": ARMS[2],
         "isolates": "m_linear conditioning instead of geom_alpha_fourier, same continuous "
                     "configs",
         "caveat": None, "delta": delta(B, C)},
        {"step": "multi-wall training", "comparison": "C - D",
         "from": ARMS[3], "to": ARMS[2],
         "isolates": "whether multi-wall training configs were necessary (D trains on "
                     "single-wall edits only)",
         "caveat": "a near-zero delta here means multi-wall training was NOT necessary, "
                   "which is a positive finding about data cost, not a failure.",
         "delta": delta(D, C)},
    ]
    for st in steps:
        st["both_present"] = bool(by_arm.get(st["to"], {}).get("present")
                                  or (st["to"] == "p3_2_main" and base))
    return steps


def first_clearing_arm(by_arm: Dict[str, dict]) -> Optional[str]:
    for a in ARMS:
        r = by_arm.get(a)
        if r and r.get("present") and r["verdict"].get("passed"):
            return a
    return None


def biggest_mover(steps: Sequence[dict]) -> Optional[dict]:
    """The ladder step that moved the S2 slope most, used only when nothing passed.

    Slope is the metric picked because it is the one P3-2 failed on most starkly
    (0.133 on the never-trained pair). It is reported as movement, never as a pass.
    """
    best, best_v = None, -np.inf
    for st in steps:
        v = _f(st["delta"].get("s2_edit_bw_slope"))
        if np.isfinite(v) and v > best_v:
            best, best_v = st, v
    return best if best is not None and np.isfinite(best_v) else None


# --------------------------------------------------------------------------- EVAL.md
def write_eval_md(path: Path, doc: dict) -> None:
    rows: List[dict] = doc["rows"]
    by_arm = {r["arm"]: r for r in rows}
    steps = doc["attribution"]
    first = doc["s2"]["first_clearing_arm"]
    thr = doc["thresholds"]
    out: List[str] = []
    A = out.append

    if doc.get("synthetic"):
        A("> # {}".format(WATERMARK.upper()))
        A(">")
        A("> Every number in this file was FABRICATED by `--synthetic` to check the table "
          "layout. Nothing here is a measurement. Do not quote, do not commit.")
        A("")

    A("# P3-2b — cross-arm ablation")
    A("")
    # ---- the first lines are the answer. Everything else is support.
    if first:
        r = by_arm[first]
        A("**S2 (unseen geometry x HELD-OUT m-slab): PASSED, first by arm {} (`{}`).**".format(
            r["letter"], first))
        step = next((s for s in steps if s["to"] == first), None)
        if step:
            A("**The change that produced it: {} ({}).**".format(step["step"],
                                                                 step["comparison"]))
    else:
        evaluated = [r for r in rows if r.get("present") and not r.get("is_baseline")]
        if not evaluated:
            A("**S2 (unseen geometry x HELD-OUT m-slab): NOT DETERMINED — no arm has been "
              "evaluated yet.**")
            A("**No attribution is possible from this table.**")
        else:
            A("**S2 (unseen geometry x HELD-OUT m-slab): NOT PASSED by any evaluated arm "
              "({} of {} arms evaluated).**".format(len(evaluated), len(ARMS)))
            mv = biggest_mover(steps)
            if mv and _fin(mv["delta"].get("s2_edit_bw_slope")):
                A("**Largest movement in the S2 edit slope came from {} ({}): {} — movement, "
                  "not a pass.**".format(mv["step"], mv["comparison"],
                                         _signed(mv["delta"]["s2_edit_bw_slope"])))
            else:
                A("**No ladder step can yet be credited: too few arms are evaluated to form "
                  "a delta.**")
    A("")

    missing = [r["letter"] for r in rows if not r.get("present")]
    if missing:
        A("> **Partial ladder.** Arm(s) {} have no `summary.json` yet, so the attribution below "
          "is provisional and every delta that touches them is `n/a`. Re-run this script when "
          "their evals land.".format(", ".join(missing)))
        A("")
    mid = [r["letter"] for r in rows if r.get("present") and r.get("mid_training")]
    if mid:
        A("> **Mid-training snapshot.** Arm(s) {} were scored on a checkpoint taken before the "
          "end of training ({}). Any claim from this table must carry the iteration "
          "number.".format(
              ", ".join(mid),
              "; ".join("{} at iter {}/{}".format(r["letter"], r.get("iter"),
                                                  r.get("total_iters"))
                        for r in rows if r.get("present") and r.get("mid_training"))))
        A("")

    # ---- the gate
    A("## The S2 gate")
    A("")
    A("Spec `{}`, thresholds sha256 `{}` (frozen in `aaf/eval/p3_2b_accept.py` before any arm "
      "was evaluated). All four criteria must hold on **{}** and no blocker may "
      "fire.".format(thr.get("spec"), (doc.get("spec_sha") or "n/a")[:12], GATE_SPLIT))
    A("")
    A("| criterion | op | threshold | {} |".format(
        " | ".join(r["letter"] for r in rows if not r.get("is_baseline"))))
    A("|---|---|---|{}".format("---|" * len([r for r in rows if not r.get("is_baseline")])))
    crit_names = ["edit_bw_slope", "edit_bw_pearson", "edit_gain", "abs_rho_minus_1"]
    ops = {"edit_bw_slope": (">=", thr.get("edit_bw_slope_min")),
           "edit_bw_pearson": (">=", thr.get("edit_bw_pearson_min")),
           "edit_gain": (">", thr.get("edit_gain_min_exclusive")),
           "abs_rho_minus_1": ("<=", thr.get("rho_abs_dev_max"))}
    for cn in crit_names:
        cells = []
        for r in rows:
            if r.get("is_baseline"):
                continue
            c = next((c for c in r["verdict"]["criteria"] if c["name"] == cn), None)
            if c is None:
                cells.append("n/a")
            else:
                cells.append("{} {}".format(_fmt(c["value"]), "PASS" if c["passed"] else "FAIL"))
        op, tv = ops[cn]
        A("| `{}` | {} | {} | {} |".format(cn, op, _fmt(tv, 2), " | ".join(cells)))
    verd = []
    for r in rows:
        if r.get("is_baseline"):
            continue
        p = r["verdict"]["passed"]
        verd.append("**PASS**" if p else ("FAIL" if p is not None else "not evaluated"))
    A("| **verdict** |  |  | {} |".format(" | ".join(verd)))
    A("")
    for r in rows:
        if r.get("is_baseline") or not r.get("present"):
            continue
        if r["verdict"].get("blockers"):
            A("- Arm {} blockers: {}".format(
                r["letter"], "; ".join("`{}` ({}={} vs {})".format(
                    b.get("name"), b.get("name"), _fmt(b.get("value")), _fmt(b.get("threshold")))
                    for b in r["verdict"]["blockers"])))
    A("")

    # ---- the headline table
    A("## Ablation table")
    A("")
    A("`slope` is `edit_bw_slope` (predicted delta-BW regressed on GT delta-BW) and `gain` is "
      "`edit_gain` (>1 means the edited render beats the model's own baseline render as an "
      "explanation of the edited ground truth). `rho` is `a_fit / a_theory` on the slab walls "
      "with **kappa-scaled** theory, kappa = {}; the raw-Lorentzian comparison is in the "
      "column after it, for transparency only.".format(_fmt(doc.get("kappa"), 6)))
    A("")
    hdr = ["arm", "cond", "val LSD (dB)"]
    for sp in SPLITS:
        hdr += ["{} slope".format(sp.split("_")[0]), "{} gain".format(sp.split("_")[0])]
    hdr += ["rho (slab)", "rho vs raw", "S2"]
    A("| " + " | ".join(hdr) + " |")
    A("|" + "---|" * len(hdr))
    for r in rows:
        cells = ["**{}**".format(r["letter"]),
                 "{}{}".format(r.get("cond_source") or "n/a",
                               "/{}".format(r.get("cond_dim")) if r.get("cond_dim") else ""),
                 _fmt(r.get("in_dist_val_lsd_db"), 3)]
        for sp in SPLITS:
            b = r["splits"].get(sp, {})
            cells += [_fmt(b.get("edit_bw_slope")), _fmt(b.get("edit_gain"))]
        p = r["verdict"]["passed"]
        cells += [_fmt(r.get("rho_slab_local")), _fmt(r.get("rho_vs_raw_theory_median")),
                  ("**PASS**" if p else ("FAIL" if p is not None else "n/a"))]
        A("| " + " | ".join(cells) + " |")
    A("")
    A("Marker: {}".format(
        "arm **{}** is the first arm in ladder order to clear the S2 thresholds.".format(
            by_arm[first]["letter"]) if first
        else "**no arm clears the S2 thresholds.**"))
    A("")

    # ---- attribution
    A("## Attribution ladder")
    A("")
    A("| step | comparison | S2 slope | S2 pearson | S2 gain | rho (slab) | val LSD |")
    A("|---|---|---|---|---|---|---|")
    for st in steps:
        d = st["delta"]
        A("| {} | {} | {} | {} | {} | {} | {} |".format(
            st["step"], st["comparison"], _signed(d.get("s2_edit_bw_slope")),
            _signed(d.get("s2_edit_bw_pearson")), _signed(d.get("s2_edit_gain")),
            _signed(d.get("rho_slab_local")), _signed(d.get("in_dist_val_lsd_db"))))
    A("")
    A("Deltas are `to - from`; for val LSD lower is better, so a negative delta is an "
      "improvement. For every other column higher is better and rho is best at 1.0.")
    A("")
    for st in steps:
        if st.get("caveat"):
            A("- **{}** ({}): {}".format(st["step"], st["comparison"], st["caveat"]))
    A("")

    # ---- the honesty note, stated rather than buried
    A("## Honesty notes")
    A("")
    A("1. **Arm A cannot cleanly isolate the renderer.** It inherits the P3-2 dataset, in "
      "which one holdout was an EXTRAPOLATION rather than an interpolation: the held-out "
      "alpha lay above every alpha that wall was trained on. A null result on arm A is "
      "therefore ambiguous between *the renderer did not help* and *that holdout was unfair*. "
      "Arms B/C/D use the P3-2b manifest, whose held-out slabs (west m in [0.62, 0.77], north "
      "m in [1.13, 1.28]) are strictly INTERIOR to the sampled range m in [0.02, 1.61], so "
      "only they test interpolation.")
    A("2. **The P3-2 baseline row is not a like-for-like comparison.** Its splits are matched "
      "to P3-2b's by experimental role, not by construction, and P3-2 ran no slope fit, so its "
      "rho column is empty. Read it as context for the size of the failure, not as a fifth "
      "arm.")
    A("3. **rho is reported against kappa-scaled theory.** The bandwidth estimator measures a "
      "calibrated -3 dB width, not the raw Lorentzian width; the gate's T5 fit gives "
      "`BW = 0.302 + {} * (gamma/pi)`. The intercept cancels in a paired delta, the slope "
      "does not, so `a_theory = kappa * c / (4 pi D)`. Scoring against the raw value would "
      "hand a perfect model rho ~ 0.60 and fail it.".format(_fmt(doc.get("kappa"), 4)))
    A("4. **A high S1/S3 with a dead S2 is exactly the P3-2 result.** Do not read a strong "
      "in-distribution or seen-geometry column as progress on the chunk's question.")
    A("")

    # ---- per-split detail
    A("## Per-split detail")
    A("")
    for sp in SPLITS:
        A("### {}".format(SPLIT_SHORT[sp]))
        A("")
        A("| arm | n cfg | n cells | frac modes dropped | band LSD (dB) | E_BW (Hz) | slope | "
          "pearson | gain | GT effect (Hz) | model effect (Hz) |")
        A("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            b = r["splits"].get(sp, {})
            if not b.get("present"):
                A("| {} | — | — | — | — | — | — | — | — | — | — |".format(r["letter"]))
                continue
            A("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["letter"], b.get("n_configs"), b.get("n_cells"),
                _fmt(b.get("frac_modes_dropped")), _fmt(b.get("band_lsd_db")),
                _fmt(b.get("E_BW_hz")), _fmt(b.get("edit_bw_slope")),
                _fmt(b.get("edit_bw_pearson")), _fmt(b.get("edit_gain")),
                _fmt(b.get("gt_effect_size_hz")), _fmt(b.get("pred_effect_size_hz"))))
        A("")

    # ---- provenance
    A("## Provenance")
    A("")
    A("| arm | checkpoint | iter | cond | train configs | manifest sha |")
    A("|---|---|---|---|---|---|")
    for r in rows:
        A("| {} | `{}` | {}{} | {}{} | {} | `{}` |".format(
            r["letter"], r.get("checkpoint"), r.get("iter"),
            "/{}".format(r.get("total_iters")) if r.get("total_iters") else "",
            r.get("cond_source"),
            "({})".format(r.get("cond_dim")) if r.get("cond_dim") else "",
            r.get("n_train_configs") or "—",
            (r.get("manifest_sha") or "—")[:12]))
    A("")
    A("Held-out slabs (m = -ln(1-alpha)): west m in [0.62, 0.77] (brackets alpha=0.50), north "
      "m in [1.13, 1.28] (brackets alpha=0.70). Both interior to the sampled range "
      "m in [0.02, 1.61].")
    A("")
    A("## Reproduction")
    A("")
    A("```bash")
    A("export PYTHONPATH=\"$PWD\"")
    A("python scripts/p3_2b_ablation.py          # this table (CPU, reads eval JSONs)")
    A("python scripts/make_p3_2b_figures.py      # the meeting pack, incl. figure E")
    A("```")
    A("")
    path.write_text("\n".join(out) + "\n")


# --------------------------------------------------------------------------- synthetic
def synthesize(dest: Path) -> Tuple[Path, Path]:
    """Fabricate schema-valid ``eval/<arm>/summary.json`` for all four arms + a P3-2 baseline.

    The values are arbitrary but are made to walk the intended ladder (A/B fail, C passes,
    D falls back) so the "first arm to clear S2" branch and the attribution text are both
    exercised. Nothing here is a measurement.
    """
    rng = np.random.RandomState(7)
    root = dest / "eval"
    kappa = 1.6607564051417665
    # (slope, pearson, gain, rho) per arm -- the ladder the layout must be able to display.
    profile = {
        ARMS[0]: (0.21, 0.44, 0.98, 0.30),
        ARMS[1]: (0.55, 0.71, 1.06, 0.58),
        ARMS[2]: (0.91, 0.88, 1.31, 0.94),
        ARMS[3]: (0.62, 0.75, 1.11, 0.66),
    }
    cond = {ARMS[0]: ("geom_alpha_fourier", 64), ARMS[1]: ("geom_alpha_fourier", 64),
            ARMS[2]: ("m_linear", 60), ARMS[3]: ("m_linear", 60)}
    for arm in ARMS:
        slope, pear, gain, rho = profile[arm]
        cs, cd = cond[arm]
        splits = {}
        for i, sp in enumerate(SPLITS):
            k = 1.0 if sp == GATE_SPLIT else float(1.0 + 0.12 * rng.randn())
            splits[sp] = {
                "n_configs": [100, 20, 80, 40, 40][i],
                "n_cells": [756, 162, 1055, 330, 317][i],
                "n_modes_candidate": 1540,
                "frac_modes_dropped": float(0.30 + 0.05 * rng.rand()),
                "fidelity": {"mag_corr": 0.89, "band_lsd_db": float(2.4 + 0.5 * rng.rand()),
                             "phase_corr_mw": 0.88, "rir_pearson": 0.90,
                             "t20_rel_err": 0.61},
                "null_fidelity": {"mag_corr": 0.88, "band_lsd_db": 3.1,
                                  "phase_corr_mw": 0.87, "rir_pearson": 0.89,
                                  "t20_rel_err": 0.66},
                "edit": {"E_BW_hz": float(2.0 + rng.rand()),
                         "edit_bw_slope": float(slope * k),
                         "edit_bw_pearson": float(min(0.99, pear * k)),
                         "edit_gain": float(gain * (1.0 + 0.02 * rng.randn())),
                         "E_LVL_db": 1.4,
                         "gt_effect_size_hz": 5.0, "pred_effect_size_hz": float(5.0 * slope)},
                "by_family": {f: {"E_BW_hz": float(1.5 + rng.rand()), "n": 200,
                                  "gt_d_bw": 2.6, "pred_d_bw": float(2.6 * slope)}
                              for f in ("x_axial", "y_axial", "tangential")},
                "per_combo": {"west0.50": {"n_configs": 10}, "north0.70": {"n_configs": 10}},
            }
        sf_group = lambda r: {                                            # noqa: E731
            "rho_median": float(r), "rho_ci95": [float(r - 0.09), float(r + 0.09)],
            "a_fit_median": float(r * 11.22), "a_theory_median": 11.22,
            "n_cells": 16, "frac_modes_dropped": 0.28}
        summary = {
            "arm": arm,
            "checkpoint": "outputs/p3_2/{}/ckpt_iter0010000.pt".format(arm),
            "iter": 10000,
            "in_dist_val_lsd_db": float(1.7 + 0.3 * rng.rand()),
            "cond_source": cs, "cond_dim": cd,
            "splits": splits,
            "slope_fit": {"aggregate": {"own_family": {"all": sf_group(rho),
                                                       "non_slab": sf_group(rho * 0.98),
                                                       "slab_local": sf_group(rho)},
                                        "orthogonal_family": {"a_fit_median": 0.5,
                                                              "n_cells": 40}},
                          "per_cell": [], "kappa": kappa,
                          "rho_vs_raw_theory_median": float(rho * kappa)},
            "controls": {"C2_floor_hz": 0.04, "C3_conditioning_identity": True},
            "slabs_m": {"holdout_slabs_m": {"west": [0.62, 0.77], "north": [1.13, 1.28]}},
            "meta": {"band_hz": [0.0, 300.0], "kappa": kappa, "total_iters": 60000,
                     "mid_training": True, "n_train_configs": 960 if arm != ARMS[0] else 440,
                     "manifest_sha": "0" * 64, "n_geometries": 50},
        }
        summary["verdict"] = p3_2b_accept.verdict(
            arm, splits[GATE_SPLIT], summary["slope_fit"], iter_=10000, mid_training=True)
        d = root / arm
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(summary, indent=1))

    # The P3-2 baseline row, in P3-2's own split vocabulary so the remap gets exercised.
    base = {
        "checkpoint": "outputs/p3_2/p3_2_main/ckpt_iter0060000.pt", "iter": 60000,
        "in_dist_val_lsd_db": 2.6869,
        "splits": {src: {"n_configs": 40, "n_cells": 300,
                         "frac_modes_dropped": 0.35,
                         "fidelity": {"band_lsd_db": 3.2},
                         "edit": {"E_BW_hz": 4.9, "edit_bw_slope": 0.13,
                                  "edit_bw_pearson": 0.53, "edit_gain": 0.87,
                                  "gt_effect_size_hz": 5.2, "pred_effect_size_hz": 0.78}}
                   for src in P3_2_SPLIT_MAP.values()},
        "meta": {"theory_slope_ism_ray": kappa, "band_hz": [0.0, 300.0]},
    }
    bp = dest / "p3_2_summary.json"
    bp.write_text(json.dumps(base, indent=1))
    return root, bp


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", default="outputs/p3_2b/eval")
    ap.add_argument("--baseline", default="outputs/p3_2/eval/summary.json",
                    help="P3-2 summary.json for the baseline row (skipped if absent)")
    ap.add_argument("--out-dir", default="outputs/p3_2b")
    ap.add_argument("--arms", default=None, help="comma-separated subset of the four arm ids")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--synthetic", action="store_true",
                    help="fabricate schema-valid inputs; writes watermarked output to "
                         "<out-dir>/synthetic/ and never to the real paths")
    args = ap.parse_args()

    tmp = None
    if args.synthetic:
        tmp = Path(tempfile.mkdtemp(prefix="p3_2b_ablation_synth_"))
        eval_root, baseline = synthesize(tmp)
        out_dir = Path(args.out_dir) / "synthetic"
        if args.no_baseline:
            baseline = None
        print("SYNTHETIC inputs in {}".format(eval_root))
    else:
        eval_root = Path(args.eval_root)
        out_dir = Path(args.out_dir)
        baseline = None if args.no_baseline else Path(args.baseline)

    arms = tuple(a.strip() for a in args.arms.split(",")) if args.arms else ARMS
    rows: List[dict] = []
    base = baseline_row(baseline) if baseline else None
    if base:
        rows.append(base)
    for arm in arms:
        sp = eval_root / arm / "summary.json"
        rows.append(arm_row(arm, sp) if sp.exists() else missing_row(arm))
        print("[{}] {}".format(arm, "read {}".format(sp) if sp.exists() else "MISSING"))

    by_arm = {r["arm"]: r for r in rows}
    steps = build_ladder(by_arm, base)
    first = first_clearing_arm(by_arm)
    present = [r for r in rows if r.get("present") and not r.get("is_baseline")]
    kappa = next((r["kappa"] for r in present if _fin(r.get("kappa"))), float("nan"))
    spec_sha = next((r["verdict"].get("spec_sha") for r in present
                     if r["verdict"].get("spec_sha")), p3_2b_accept.thresholds_sha256())

    doc = {
        "schema": "p3_2b.ablation/1",
        "synthetic": bool(args.synthetic),
        "watermark": WATERMARK if args.synthetic else None,
        "eval_root": str(eval_root),
        "arms_expected": list(ARMS),
        "arms_evaluated": [r["arm"] for r in present],
        "arms_missing": [r["arm"] for r in rows if not r.get("present")],
        "kappa": kappa,
        "thresholds": dict(p3_2b_accept.THRESHOLDS),
        "spec_sha": spec_sha,
        "gate_split": GATE_SPLIT,
        "splits": list(SPLITS),
        "rows": rows,
        "attribution": steps,
        "s2": {
            "first_clearing_arm": first,
            "passed_any": bool(first),
            "biggest_slope_mover": (biggest_mover(steps) or {}).get("comparison"),
            "note": ("arm A inherits the P3-2 dataset in which one holdout was an "
                     "EXTRAPOLATION, so arm A cannot separate 'the renderer did not help' "
                     "from 'that holdout was unfair'"),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "ablation.json"
    mp = out_dir / "EVAL.md"
    jp.write_text(json.dumps(doc, indent=1, allow_nan=True))
    write_eval_md(mp, doc)
    print("wrote {}".format(jp))
    print("wrote {}".format(mp))
    print("S2 first clearing arm: {}".format(first or "none"))

    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
