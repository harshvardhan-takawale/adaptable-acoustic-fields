"""P3-2c evaluation splits: one arm of the density sweep, with DESIGNATED S2.

The density curve compares one number (the S2-west edit slope) across five arms that differ
only in how wide a band of west absorptions was withheld from training. That comparison is
meaningful only if S2 is the SAME POPULATION OF ROOMS in every arm. It is not automatic:

    W100's training slab is [0.193, 1.193] in m, which contains alpha = 0.30 (m = 0.3567).
    A split rule that asked "is this config inside THIS ARM's held-out slab?" would therefore
    move the 10 ``west@0.30`` rooms from S4 into S2 for W100 alone -- so W100's headline would
    be computed over 30 rooms, of which 10 are a different material, while every other arm's
    is over 20. The curve would then confound gap width with population change, and nothing
    would flag it: all five numbers would still be well-formed slopes.

So split assignment is **designated by the test protocol, not inferred from the arm**. Every
arm reuses ``aaf.eval.p3_2b_splits.classify``, which keys on the FROZEN W015 slabs. That is
not a workaround -- it is the definition: S2 means "the 20 rooms carrying west@0.50 or
north@0.70", a fixed protocol, and the arms differ in what training saw, never in what is
tested.

Two arm-specific facts are then attached as ANNOTATIONS rather than as reassignment:

* ``d_support_m`` -- distance in m from a config's edited value to the nearest TRAINING draw
  on that wall in that arm. This is the honest per-config x-axis: it is what "how far outside
  support" actually means, and unlike the nominal slab width it is a property of the realized
  dataset. For W100, ``west@0.30`` carries d_support = 0.164 and is reported as the S2x extra
  curve point -- a free measurement, kept out of the headline.
* ``arm_holdout`` -- whether this arm's training actually excluded the value. For XTRAP,
  ``west@0.50`` is TRAINED, so S2's west half is an interpolation *control* there rather than
  a holdout; reporting it unflagged would read as a spectacular pass.

XTRAP is the one arm whose test set differs at all: it adds 20 rooms at alpha = 0.75 / 0.80.
Those land in S1 under the frozen predicate (they are non-slab single-wall presets), which
would silently inflate S1 from 100 to 120 and bury the extrapolation measurement inside the
easy split. They are pulled into ``S2X_EXTRAP`` together with ``west@0.70``, giving three
beyond-edge distances on one arm.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from aaf.data.mat_configs_cont import m_of_alpha
from aaf.data.mat_configs_p3_2c import SPECS, ArmSpec
from aaf.eval.p3_2b_splits import (
    S1,
    S2,
    S3,
    S4,
    S5,
    SPLIT_ORDER,
    EvalConfig,
    build_splits,
)
from aaf.walls import ALPHA_BASELINE, WALL_INDEX

MANIFEST_FMT = "configs/sweeps_2d_mat/p3_2c_{arm}_manifest.json"
W015_MANIFEST = "configs/sweeps_2d_mat/p3_2b_manifest.json"

S2X_EXTRAP = "S2X_unseen_geom_west_extrap"
"""XTRAP only: west edits beyond the training edge (alpha 0.70 / 0.75 / 0.80)."""

EXTRAP_ALPHAS: Tuple[float, ...] = (0.70, 0.75, 0.80)

# Per-arm expected counts. XTRAP moves 30 west configs out of S1 (10 of which are the
# pre-existing west@0.70 rooms, 20 of which are its new test sims).
BASE_COUNTS: Dict[str, int] = {S1: 100, S2: 20, S3: 80, S4: 40, S5: 40}
XTRAP_COUNTS: Dict[str, int] = {S1: 90, S2: 20, S3: 80, S4: 40, S5: 40, S2X_EXTRAP: 30}


def arm_manifest(arm: str) -> str:
    """W015 is the frozen P3-2b arm C and keeps its original manifest path."""
    return W015_MANIFEST if arm == "W015" else MANIFEST_FMT.format(arm=arm)


def expected_counts(arm: str) -> Dict[str, int]:
    return dict(XTRAP_COUNTS) if arm == "XTRAP" else dict(BASE_COUNTS)


def split_order(arm: str) -> Tuple[str, ...]:
    return SPLIT_ORDER + (S2X_EXTRAP,) if arm == "XTRAP" else SPLIT_ORDER


# --------------------------------------------------------------------------- support map
def training_support(arm: str) -> Dict[str, List[float]]:
    """Sorted m-values actually drawn for each wall in this arm's TRAINING rows.

    Read from the manifest rather than recomputed from the slab: the realized draws are what
    the model saw, and they do not land on the slab edges.
    """
    man = json.loads(Path(arm_manifest(arm)).read_text())
    out: Dict[str, List[float]] = {}
    for r in man["configs"]:
        if r["split"] != "train":
            continue
        for w in r["edited"]:
            out.setdefault(w, []).append(m_of_alpha(float(r["alphas"][WALL_INDEX[w]])))
    return {w: sorted(v) for w, v in out.items()}


def d_support_m(support: Dict[str, List[float]], wall: str, alpha: float) -> Optional[float]:
    """Distance from this value to the nearest training draw on the same wall.

    Zero would mean the exact value was trained; every test point should be strictly positive
    because the sampler rejects the test presets outright.
    """
    ms = support.get(wall)
    if not ms:
        return None
    m = m_of_alpha(float(alpha))
    return float(min(abs(m - x) for x in ms))


def annotate(cfg: EvalConfig, arm: str, spec: ArmSpec,
             support: Dict[str, List[float]]) -> dict:
    """Arm-specific facts about one eval config -- annotation only, never reassignment."""
    per_wall = {}
    for w in cfg.edited:
        a = float(cfg.alphas[WALL_INDEX[w]])
        per_wall[w] = {
            "alpha": a,
            "m": m_of_alpha(a),
            "d_support_m": d_support_m(support, w, a),
            "arm_holdout": bool(spec.rejects(w, a)),
        }
    # The binding distance for a config is its FURTHEST-outside wall: a two-wall edit is only
    # as interpolable as its hardest component.
    ds = [v["d_support_m"] for v in per_wall.values() if v["d_support_m"] is not None]
    return {
        "arm": arm,
        "per_wall": per_wall,
        "d_support_m": max(ds) if ds else None,
        "any_arm_holdout": any(v["arm_holdout"] for v in per_wall.values()),
        "all_arm_holdout": bool(per_wall) and all(v["arm_holdout"] for v in per_wall.values()),
    }


# --------------------------------------------------------------------------- driver
def _is_west_extrap(c: EvalConfig) -> bool:
    if len(c.edited) != 1 or c.edited[0] != "west":
        return False
    a = float(c.alphas[WALL_INDEX["west"]])
    return any(abs(a - x) <= 1e-9 for x in EXTRAP_ALPHAS)


def build_splits_p3_2c(arm: str, train_yaml: Optional[str] = None,
                       test_yaml: Optional[str] = None):
    """Splits for one arm, with S2 designated and XTRAP's extrapolation split separated."""
    if arm not in SPECS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(SPECS)}")
    kw = {}
    if train_yaml:
        kw["train_yaml"] = train_yaml
    if test_yaml:
        kw["test_yaml"] = test_yaml
    splits, ctx = build_splits(manifest=arm_manifest(arm), **kw)

    if arm == "XTRAP":
        moved = [c for c in splits[S1] if _is_west_extrap(c)]
        splits[S1] = [c for c in splits[S1] if not _is_west_extrap(c)]
        # Re-label so assert_split_counts' "config labelled X in split Y" check stays honest.
        splits[S2X_EXTRAP] = [
            EvalConfig(L=c.L, W=c.W, alphas=c.alphas, edited=c.edited, split=S2X_EXTRAP,
                       geom_seen=c.geom_seen, source=c.source)
            for c in moved
        ]

    spec = SPECS[arm]
    support = training_support(arm)
    ctx = dict(ctx)
    ctx["arm"] = arm
    ctx["arm_spec"] = {
        "name": spec.name, "run_id": spec.run_id,
        "west_slab": list(spec.west) if spec.west else None,
        "west_max": spec.west_max, "north_slab": list(spec.north),
        "nominal_width": spec.nominal_width,
    }
    ctx["manifest_path"] = arm_manifest(arm)
    ctx["training_support_m"] = {w: {"n": len(v), "min": v[0], "max": v[-1]}
                                 for w, v in support.items()}
    ctx["annotations"] = {
        c.label: annotate(c, arm, spec, support)
        for name in split_order(arm) for c in splits.get(name, [])
    }
    ctx["s2_designation"] = {
        "policy": "frozen_w015_predicate",
        "note": ("S2 is the fixed 20-room protocol split (west@0.50, north@0.70) in every "
                 "arm. Arm-specific holdout status is an annotation, never a reassignment; "
                 "inferring membership from each arm's own slab would give W100 a 30-room "
                 "S2 and make the density curve compare different populations."),
        "s2_west_is_arm_holdout": bool(spec.rejects("west", 0.50)),
        "s2_north_is_arm_holdout": bool(spec.rejects("north", 0.70)),
    }
    # S2x: rooms this arm's slab additionally swallows, reported beside the curve but never in
    # the headline. For W100 this is west@0.30 at d_support 0.164.
    # Scanned over the UNSEEN-geometry splits only. S3 is excluded by definition, not by
    # accident: it *is* "slab material on a training geometry", so every S3 config is rejected
    # by every arm's predicate and including it would report all 80 as extra curve points in
    # all five arms -- swamping the one real migration (W100's west@0.30) 8:1.
    s2x = []
    for name in (S1, S4):
        for c in splits.get(name, []):
            if len(c.edited) != 1:
                continue
            w = c.edited[0]
            if spec.rejects(w, float(c.alphas[WALL_INDEX[w]])):
                s2x.append({"label": c.label, "home_split": name, "wall": w,
                            "alpha": float(c.alphas[WALL_INDEX[w]]),
                            "d_support_m": d_support_m(
                                support, w, float(c.alphas[WALL_INDEX[w]]))})
    ctx["s2x_extra_curve_points"] = s2x
    return splits, ctx


def assert_split_counts_p3_2c(splits: Dict[str, List[EvalConfig]], arm: str) -> None:
    """Blocking, per arm. A wrong count means the density curve compares different
    populations across arms -- which produces five plausible slopes and no error."""
    want = expected_counts(arm)
    bad = []
    for name, n in want.items():
        got = len(splits.get(name, []))
        if got != n:
            bad.append(f"{name}: expected {n}, got {got}")
    extra = [k for k in splits if k not in want]
    if extra:
        bad.append(f"unexpected split keys: {sorted(extra)}")
    seen = set()
    for name in want:
        for c in splits.get(name, []):
            if c.is_baseline:
                bad.append(f"{name} contains a baseline ({c.label}) -- baselines are anchors")
            if c.split != name:
                bad.append(f"{name} contains a config labelled {c.split}")
            k = (c.geom_key, c.alphas)
            if k in seen:
                bad.append(f"config {c.label} assigned to more than one split")
            seen.add(k)
    if bad:
        raise AssertionError(f"[{arm}] split assignment is wrong:\n  " + "\n  ".join(bad))


def curve_point(arm: str) -> dict:
    """This arm's x-axis value for the density curve, with its axis name.

    Interior-gap arms and the edge-excluded arm are NOT on the same axis, and the manifest
    records which one applies. Mixing them would place XTRAP -- whose test points sit beyond
    all training support -- at the dense end of the curve.
    """
    man = json.loads(Path(arm_manifest(arm)).read_text())
    if arm == "W015":
        # The frozen P3-2b manifest predates the p3_2c schema; recompute from its own rows.
        from aaf.data.mat_configs_p3_2c import realized_gap
        rows = [r for r in man["configs"] if r["split"] == "train"]
        g = realized_gap(rows, "west")
        return {"arm": arm, "axis": "interior_gap", "x": g["max_gap_m"], "detail": g}
    if man.get("gap_axis") == "beyond_edge":
        return {"arm": arm, "axis": "beyond_edge", "x": None,
                "detail": man["edge_distances_west"],
                "note": "per-test-point distances; this arm has no single x"}
    g = man["realized_gap_west"]
    return {"arm": arm, "axis": "interior_gap", "x": g["max_gap_m"], "detail": g}
