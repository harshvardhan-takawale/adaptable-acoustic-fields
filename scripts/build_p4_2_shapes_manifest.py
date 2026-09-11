"""Freeze the P4-2 notch-family manifest.

Ships as a script deliberately. `p3_2b_manifest.json` and `p4_1_lroom_manifest.json` were both
committed with NO builder in the tree -- every later chunk reads them as immutable inputs and
nobody can regenerate or audit them. Not repeating that.

Every invariant is asserted HERE, at freeze time, so a broken manifest never reaches the
cluster: no training shape inside the held-out band, >= 6 test shapes inside it, unique
filenames, unique labels, >= 8 rectangles, origin-anchored vertices, and sampling no coarser
than 0.125 of each normalized range (excluding the deliberate holes).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aaf.data.shape_configs import (
    ALPHA, D_HAT_HOLDOUT, D_FRAC_MAX, L_RANGE, MAX_GAP_FRAC, MIN_PASSAGE_M, MIN_RECTANGLES,
    N_TEST, N_TEST_IN_SLAB, N_TRAIN, SEED, SRC, TEST_SEED, W_FRAC_MAX, W_RANGE,
    assert_invariants, manifest_rows, rows_sha256, sample_test_shapes, sample_train_shapes,
)

OUT = "configs/sweeps_2d_mat/p4_2_shapes_manifest.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and not a.force:
        raise SystemExit("{} exists and is FROZEN; pass --force to overwrite".format(out))

    train = sample_train_shapes()
    test = sample_test_shapes(train)
    report = assert_invariants(train, test)          # blocking
    rows = manifest_rows(train, test)

    man = {
        "schema": "p4_2.shape/1",
        "seed": SEED, "test_seed": TEST_SEED,
        "family": "rectangle with a rectangular notch removed from the TOP-LEFT corner; "
                  "d = w = 0 is a pure rectangle, d, w > 0 is an L",
        "L_range": list(L_RANGE), "W_range": list(W_RANGE),
        "d_frac_max": D_FRAC_MAX, "w_frac_max": W_FRAC_MAX,
        "d_hat_def": "d / (d_frac_max * W)",
        "w_hat_def": "w / (w_frac_max * L)",
        "d_hat_holdout": list(D_HAT_HOLDOUT),
        "d_hat_holdout_note": (
            "The only reading under which the band is reachable. Under d_hat = d/W the sampled "
            "range is [0, 0.45] and the band [0.45, 0.60] would be EMPTY -- no training shape "
            "excluded, no test shape inside."),
        "alpha": ALPHA, "alpha_note": "fixed everywhere; this chunk isolates SHAPE",
        "src": list(SRC),
        "src_note": (
            "A fixed source must satisfy sy < min(W - d) = 0.55 * W_min = 2.20 m or it falls "
            "inside the notch of the tightest shapes. NLOS is ~3% at the slab centre and ~25% "
            "at maximum notch -- thin in the band by construction, since shallow notches cast "
            "little shadow. Reported with receiver counts, never averaged in silently."),
        "min_passage_m": MIN_PASSAGE_M,
        "min_passage_note": (
            "VACUOUS at these ranges -- minimum passage is 0.55 * W_min = 2.20 m, so the rule "
            "never rejects. Implemented because it becomes live if the ranges move."),
        "max_gap_frac": MAX_GAP_FRAC,
        "max_gap_note": (
            "0.125, NOT the 0.2 the chunk spec cites. P3-2d's Delta* ~ 0.275 is unratified "
            "(no DECISIONS entry; sampling_law.json records delta_star.point_estimate = null), "
            "rests on a rho definition P3-2d explicitly escalated rather than decided, and does "
            "NOT replicate at a second seed (G030 rho 1.3083 FAIL -> 1.1753 PASS). 0.2 x 1.59 = "
            "0.318 in m is exactly that G030 arm. The arm passing under both rho definitions at "
            "both seeds is G020 = 0.125 of the normalized range. P3-2d also measured ONE axis "
            "in its own linearizing coordinate; nothing establishes transfer to geometry."),
        "n_train": N_TRAIN, "n_test": N_TEST,
        "n_test_in_holdout_target": N_TEST_IN_SLAB, "min_rectangles": MIN_RECTANGLES,
        "invariants": report,
        "rows_sha256": rows_sha256(rows),
        "configs": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, indent=1))
    g = report["max_normalized_gap"]
    print("train {} (rect {}) | test {} (in-slab {}) | sha {}".format(
        report["n_train"], report["n_rect"], report["n_test"],
        report["n_test_in_holdout"], man["rows_sha256"][:12]))
    print("max normalized gap (excl. deliberate holes, target <= {}): {}".format(
        MAX_GAP_FRAC, {k: round(v, 4) for k, v in g.items()}))
    print("-> {}".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
