"""Freeze the P4-4 Task 3 manifest: rectangles, L-rooms and TWO-reflex-corner rooms.

The question is whether corner count extrapolates one step at a time. Train on 0-, 1- and
2-corner rooms and test held-out instances of each; only if that holds does the 3-corner
Z-corridor become Stage 3.

Token count is the specific thing Z will hinge on -- a rectangle is 4 boundary tokens, an L is 6,
a U/T/DN is 8 -- so the manifest records `n_tokens` per row and the evaluator reports accuracy
against it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aaf.data.multi_notch import (
    D_HAT_HOLDOUT_FAM, FAMILIES, FAMILY_SEED, N_FAM_TEST, N_FAM_TEST_IN_SLAB, N_FAM_TRAIN,
    N_TOKENS, geometry_ok, sample_family_shapes,
)
from aaf.data.shape_configs import ALPHA, L_RANGE, SRC, W_RANGE, rows_sha256

OUT = "configs/sweeps_2d_mat/p4_4_family_manifest.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n-train", type=int, default=N_FAM_TRAIN)
    ap.add_argument("--n-test", type=int, default=N_FAM_TEST)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and not a.force:
        raise SystemExit("{} exists and is FROZEN; pass --force".format(out))

    train = sample_family_shapes(a.n_train, split="train", seed=FAMILY_SEED)
    test = sample_family_shapes(a.n_test, split="test", seed=FAMILY_SEED + 1,
                                n_in_slab=N_FAM_TEST_IN_SLAB,
                                exclude=[c.filename for c in train])

    # ---- blocking invariants ----
    slab = [c.label for c in train
            if D_HAT_HOLDOUT_FAM[0] <= c.d_hat <= D_HAT_HOLDOUT_FAM[1]]
    if slab:
        raise AssertionError("{} TRAINING rooms in the held-out slab: {}".format(
            len(slab), slab[:4]))
    fns = [c.filename for c in train + test]
    if len(set(fns)) != len(fns):
        dup = sorted({f for f in fns if fns.count(f) > 1})
        raise AssertionError("duplicate filenames (a U and a T would alias): {}".format(dup[:4]))
    if len({c.label for c in train + test}) != len(fns):
        raise AssertionError("duplicate labels")
    for c in train + test:
        ok, why = geometry_ok(c.L, c.W, c.notches)
        if not ok:
            raise AssertionError("{} is not a legal room: {}".format(c.label, why))
        if c.n_tokens != N_TOKENS[c.kind]:
            raise AssertionError("{}: {} tokens, expected {}".format(
                c.label, c.n_tokens, N_TOKENS[c.kind]))
        for v in (c.L, c.W) + tuple(x for n in c.notches for x in (n.d, n.w, n.x0)):
            if abs(v / 0.02 - round(v / 0.02)) > 1e-9:
                raise AssertionError("{}: {} is off the 0.02 m grid (D67c)".format(c.label, v))
    for fam in FAMILIES:
        n_tr = sum(1 for c in train if c.kind == fam)
        n_te = sum(1 for c in test if c.kind == fam)
        if n_tr != a.n_train or n_te != a.n_test:
            raise AssertionError("family {}: {} train / {} test, expected {} / {}".format(
                fam, n_tr, n_te, a.n_train, a.n_test))

    rows = [c.row(i) for i, c in enumerate(list(train) + list(test))]
    man = {
        "schema": "p4_4.family/1",
        "families": {k: {"sides": list(v), "n_tokens": N_TOKENS[k]} for k, v in FAMILIES.items()},
        "seed": FAMILY_SEED, "test_seed": FAMILY_SEED + 1,
        "why": ("Does corner count extrapolate one step at a time? Train on 0-, 1- and "
                "2-reflex-corner rooms, test held-out instances of each. Z (3 corners) becomes "
                "Stage 3 only if this holds."),
        "notch_placement_note": (
            "Notches are restricted to NW / NE / midN / SE, each proved to leave the frozen "
            "SRC = (0.5, 1.6) in air for every legal draw. The trainer keeps only the LAST "
            "config's source_pos, so a family that swallowed the source would train every room "
            "against the wrong one with no error anywhere."),
        "L_range": list(L_RANGE), "W_range": list(W_RANGE),
        "d_hat_holdout": list(D_HAT_HOLDOUT_FAM),
        "d_hat_def": "the PRIMARY notch's d / (0.45 W); NW, else midN, else NE, else SE",
        "alpha": ALPHA, "src": list(SRC),
        "n_train": len(train), "n_test": len(test),
        "n_train_per_family": a.n_train, "n_test_per_family": a.n_test,
        "token_counts": {k: N_TOKENS[k] for k in FAMILIES},
        "rows_sha256": rows_sha256(rows), "configs": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(man, open(out, "w"), indent=1)
    print("[ok] {} train + {} test -> {}".format(len(train), len(test), out))
    print("     rows_sha256 {}".format(man["rows_sha256"][:12]))
    print("     {:6s} {:>6s} {:>6s} {:>8s} {:>14s}".format(
        "family", "train", "test", "tokens", "mean d_hat"))
    for fam in sorted(FAMILIES):
        tr = [c for c in train if c.kind == fam]
        te = [c for c in test if c.kind == fam]
        ins = sum(1 for c in te
                  if D_HAT_HOLDOUT_FAM[0] <= c.d_hat <= D_HAT_HOLDOUT_FAM[1])
        print("     {:6s} {:>6d} {:>6d} {:>8d} {:>14.3f}   ({} test in-slab)".format(
            fam, len(tr), len(te), N_TOKENS[fam],
            float(np.mean([c.d_hat for c in tr])), ins))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
