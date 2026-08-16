"""P3-2d evaluation splits: the midpoint hold-out is the headline.

P3-2c's headline was a slab hold-out and its x-axis was "how wide a hole". P3-2d asks the
dataset-design question directly -- "how coarsely may I sample" -- so the hold-out is the set
of grid MIDPOINTS, the points maximally distant from every training value at that interval.
The score at interval Delta is therefore the worst case that interval admits, which is what a
sampling rule needs.

Two membership rules are load-bearing:

* **The midpoint split is enumerated from the manifest's own grid, not inferred.** Each run has
  a different interval, hence different midpoints, hence a different M split -- unlike P3-2c,
  where the comparison required an identical population across arms. Here the population MUST
  differ, because the midpoints are the manipulation. What is held identical across runs is
  the 40/10 geometries, the config count, the mix, the encoder and the estimator.
* **A midpoint that sits near an always-trained value is NOT a hold-out and is excluded from
  the headline.** alpha = 0.15 (m = 0.16252) is exempt from the grid and appears on every
  non-edited wall of every config, so it is densely trained in every run. The n=16 and n=6
  grids each place a midpoint at m ~ 0.179, 0.0165 away. Those are reported separately as a
  trained-value control -- including them would hand exactly those two runs a free point.

S1/S4/S5 are carried over from the P3-2b definitions on the frozen preset test rows, which are
reused byte-for-byte, so the easy/continuity/superposition splits stay comparable to every
earlier chunk. S2/S3 are reported but are NOT slab splits here: P3-2d has no held-out band, so
west@0.50 and north@0.70 are ordinary untrained presets in every run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from aaf.data.mat_configs import PRESET_ALPHAS
from aaf.data.mat_configs_cont import m_of_alpha
from aaf.data.mat_configs_grid import (
    GRID_SPECS,
    M_BASELINE,
    TRAINED_VALUE_TOL,
    midpoints,
    realized_delta,
)
from aaf.eval.p3_2b_splits import (
    S1,
    S2,
    S3,
    S4,
    S5,
    SPLIT_ORDER,
    EvalConfig,
    build_splits,
    edited_walls,
)
from aaf.walls import WALL_INDEX

MANIFEST_FMT = "configs/sweeps_2d_mat/p3_2d_{run}_manifest.json"

M_HEADLINE = "M_unseen_geom_midpoint"
"""The hold-out this chunk exists to measure."""

M_TRAINED_CTRL = "Mx_midpoint_near_trained_value"
"""Midpoints too close to the always-trained baseline to count as hold-outs."""


def manifest_path(run: str) -> str:
    return MANIFEST_FMT.format(run=run)


def _all_preset(c: EvalConfig, tol: float = 1e-9) -> bool:
    """Every edited wall carries a canonical preset alpha (so this is a P3-2b preset split)."""
    return all(
        any(abs(float(c.alphas[WALL_INDEX[w]]) - p) <= tol for p in PRESET_ALPHAS)
        for w in c.edited)


def _mk_from_row(r: dict, split: str) -> EvalConfig:
    a = tuple(round(float(x), 6) for x in r["alphas"])
    return EvalConfig(L=round(float(r["L"]), 2), W=round(float(r["W"]), 2), alphas=a,
                      edited=edited_walls(a), split=split, geom_seen=False,
                      source="manifest")


def build_splits_p3_2d(run: str):
    """S1..S5 on the frozen preset rows, plus this run's midpoint splits."""
    if run not in GRID_SPECS:
        raise ValueError("unknown run {!r}; expected one of {}".format(
            run, sorted(GRID_SPECS)))
    mp = manifest_path(run)
    man = json.loads(Path(mp).read_text())

    # build_splits classifies EVERY test row in the manifest, and P3-2b's `classify` sees a
    # midpoint edit as an ordinary non-slab single-wall edit -- so without this filter every
    # midpoint lands in S1 as well as in M, S1 inflates from 100 to 670, and each midpoint is
    # counted twice. S1..S5 are preset splits by definition, so restrict them to configs whose
    # every edited wall carries a canonical preset alpha.
    splits, ctx = build_splits(manifest=mp)
    for name in SPLIT_ORDER:
        splits[name] = [c for c in splits[name] if _all_preset(c)]

    n = man["n_grid_points"]
    mids = {round(x["alpha"], 6): x for x in midpoints(n)}
    head: List[EvalConfig] = []
    ctrl: List[EvalConfig] = []
    for r in man["configs"]:
        if r["split"] != "test" or r.get("kind") != "midpoint":
            continue
        ed = tuple(r["edited"])
        if len(ed) != 1:
            continue
        a = round(float(r["alphas"][WALL_INDEX[ed[0]]]), 6)
        info = mids.get(a)
        if info is None:
            raise AssertionError(
                "midpoint row alpha={} is not one of this run's midpoints".format(a))
        (head if info["headline"] else ctrl).append(
            _mk_from_row(r, M_HEADLINE if info["headline"] else M_TRAINED_CTRL))
    splits[M_HEADLINE] = head
    if ctrl:                 # only grids whose midpoints collide with the baseline have one
        splits[M_TRAINED_CTRL] = ctrl

    ctx = dict(ctx)
    ctx["run"] = run
    ctx["n_grid_points"] = n
    ctx["realized_delta_m"] = realized_delta(n)
    ctx["nominal_delta_m"] = man["nominal_delta_m"]
    ctx["grid_m"] = man["grid_m"]
    ctx["midpoints"] = man["midpoints"]
    ctx["near_preset_grid_values"] = man["near_preset_grid_values"]
    ctx["delta_axis_note"] = man["delta_axis_note"]
    ctx["midpoint_policy"] = {
        "headline_split": M_HEADLINE,
        "control_split": M_TRAINED_CTRL,
        "tolerance_m": TRAINED_VALUE_TOL,
        "baseline_m": M_BASELINE,
        "note": ("a midpoint within {} of m={:.5f} is excluded from the headline: alpha=0.15 "
                 "is exempt from the grid and sits on every non-edited wall of every config, "
                 "so it is densely trained and such a midpoint is not a hold-out".format(
                     TRAINED_VALUE_TOL, M_BASELINE)),
    }
    return splits, ctx


def split_order(run: str) -> Tuple[str, ...]:
    n = GRID_SPECS[run]
    order = SPLIT_ORDER + (M_HEADLINE,)
    if any(not x["headline"] for x in midpoints(n)):
        order = order + (M_TRAINED_CTRL,)
    return order


def expected_counts(run: str) -> Dict[str, int]:
    n = GRID_SPECS[run]
    mids = midpoints(n)
    n_head = sum(1 for x in mids if x["headline"])
    n_ctrl = len(mids) - n_head
    out = {S1: 100, S2: 20, S3: 80, S4: 40, S5: 40,
           M_HEADLINE: n_head * 4 * 10}          # 4 walls x 10 test geometries
    if n_ctrl:
        out[M_TRAINED_CTRL] = n_ctrl * 4 * 10
    return out


def assert_split_counts_p3_2d(splits, run: str) -> None:
    """Blocking. A wrong midpoint count means the headline is measured over the wrong
    population, and the sampling law is exactly a comparison of those populations."""
    want = expected_counts(run)
    bad = []
    for name, k in want.items():
        got = len(splits.get(name, []))
        if got != k:
            bad.append("{}: expected {}, got {}".format(name, k, got))
    extra = [k for k in splits if k not in want]
    if extra:
        bad.append("unexpected split keys: {}".format(sorted(extra)))
    seen = set()
    for name in want:
        for c in splits.get(name, []):
            if c.is_baseline:
                bad.append("{} contains a baseline ({})".format(name, c.label))
            key = (c.geom_key, c.alphas)
            if key in seen:
                bad.append("config {} in more than one split".format(c.label))
            seen.add(key)
    if bad:
        raise AssertionError("[{}] split assignment is wrong:\n  {}".format(
            run, "\n  ".join(bad)))


def curve_point(run: str) -> dict:
    """This run's x for the sampling-law curve: the REALIZED interval."""
    n = GRID_SPECS[run]
    return {
        "run": run, "n_grid_points": n,
        "x_realized_delta_m": realized_delta(n),
        "nominal_delta_m": float(run[1:]) / 100.0,
        "axis": "realized_grid_interval_m",
        "note": ("nominal is a LABEL. Anchoring n points inclusively on M_RANGE is what gives "
                 "the intended counts, and it makes the realized interval differ (D53(c))."),
    }
