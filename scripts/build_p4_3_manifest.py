"""Freeze the P4-3 Arm-D manifest: P4-2's corpus plus a shallow-band extension.

P4-2's sweep found accuracy WORST at shallow notches (+0.741 at d_hat 0.21) and BEST at deep
ones (+0.906 at 0.90), while the corpus holds only 13 of 60 training shapes in [0.10, 0.45)
against 10-15 per decile at both ends (D72). Arm D asks whether that weakness is a data problem
by filling the thin band and changing nothing else.

TWO INVARIANTS THIS SCRIPT EXISTS TO ENFORCE
--------------------------------------------
1. **The P4-2 rows must reproduce EXACTLY.** The 75 existing rooms are already simulated; if a
   single L, W, d or w drifted, its derived filename would change and the builder would silently
   re-simulate a different room -- or worse, reuse a `.h5` that no longer matches its row. The
   original manifest is read back and compared row for row, and this script refuses to write if
   anything differs.
2. **No training shape inside the held-out slab.** The extension band stops at 0.45 precisely
   because the slab is [0.45, 0.60]; a shape there would invalidate the in-slab Gate-2 number
   for EVERY arm, not just for D. Asserted after quantization, not before.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aaf.data.shape_configs import (
    ALPHA, D_FRAC_MAX, D_HAT_HOLDOUT, EXTRA_SEED, L_RANGE, MIN_RECTANGLES, N_TEST,
    N_TEST_IN_SLAB, N_TRAIN, SEED, SRC, TEST_SEED, W_FRAC_MAX, W_RANGE, in_holdout,
    manifest_rows, rows_sha256, sample_extra_shapes, sample_test_shapes, sample_train_shapes,
)

P4_2 = "configs/sweeps_2d_mat/p4_2_shapes_manifest.json"
OUT = "configs/sweeps_2d_mat/p4_3_shapes_manifest.json"
N_EXTRA = 32
BAND = (0.10, 0.45)
KEYS = ("L", "W", "d", "w", "filename", "label", "split", "d_hat", "w_hat")


def _assert_reproduces_p4_2(train, test, base_path: str):
    """The existing corpus must come back bit-for-bit, or its 75 built rooms cannot be reused."""
    base = json.load(open(base_path))["configs"]
    old_tr = [r for r in base if r["split"] == "train"]
    old_te = [r for r in base if r["split"] == "test"]
    if len(old_tr) != len(train) or len(old_te) != len(test):
        raise AssertionError("P4-2 had {}/{} train/test, regenerated {}/{}".format(
            len(old_tr), len(old_te), len(train), len(test)))
    bad = []
    for old, new in list(zip(old_tr, train)) + list(zip(old_te, test)):
        row = new.row(0)
        for k in KEYS:
            if k in ("d_hat", "w_hat"):
                if abs(float(old[k]) - float(row[k])) > 1e-9:
                    bad.append((old["filename"], k, old[k], row[k]))
            elif old[k] != row[k]:
                bad.append((old["filename"], k, old[k], row[k]))
    if bad:
        raise AssertionError(
            "regenerated P4-2 rows DIFFER from the frozen manifest at {} field(s); the existing "
            ".h5 corpus could not be reused safely. First 5: {}".format(len(bad), bad[:5]))
    print("[ok] P4-2's {} train + {} test rows reproduce exactly".format(len(train), len(test)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--base", default=P4_2)
    ap.add_argument("--n-extra", type=int, default=N_EXTRA)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and not a.force:
        raise SystemExit("{} exists and is FROZEN; pass --force to overwrite".format(out))

    train = sample_train_shapes()
    test = sample_test_shapes(train)
    _assert_reproduces_p4_2(train, test, a.base)

    extra = sample_extra_shapes(list(train) + list(test), a.n_extra, band=BAND)
    all_train = list(train) + list(extra)

    # ---- blocking invariants -------------------------------------------------------------
    slab = [c.label for c in all_train if in_holdout(c.d_hat)]
    if slab:
        raise AssertionError("{} TRAINING shapes landed in the held-out slab {}: {}".format(
            len(slab), D_HAT_HOLDOUT, slab[:5]))
    n_slab_test = sum(1 for c in test if in_holdout(c.d_hat))
    if n_slab_test < N_TEST_IN_SLAB:
        raise AssertionError("only {} test shapes in the slab, need >= {}".format(
            n_slab_test, N_TEST_IN_SLAB))
    fns = [c.filename for c in all_train + list(test)]
    if len(set(fns)) != len(fns):
        dupes = sorted({f for f in fns if fns.count(f) > 1})
        raise AssertionError("duplicate filenames: {}".format(dupes))
    labels = [c.label for c in all_train + list(test)]
    if len(set(labels)) != len(labels):
        raise AssertionError("duplicate labels")
    n_rect = sum(1 for c in all_train if c.is_rect)
    if n_rect < MIN_RECTANGLES:
        raise AssertionError("only {} rectangles, need >= {}".format(n_rect, MIN_RECTANGLES))
    # D67c: every parameter must land on the FDTD grid or the tokens and the solver describe
    # different rooms. Re-checked here because the extension introduces new draws.
    off = [(c.label, v) for c in all_train + list(test)
           for v in (c.L, c.W, c.d, c.w) if abs(v / 0.02 - round(v / 0.02)) > 1e-9]
    if off:
        raise AssertionError("{} parameter(s) off the 0.02 grid (D67c): {}".format(
            len(off), off[:5]))
    for c in extra:
        if not (BAND[0] <= c.d_hat < BAND[1]):
            raise AssertionError("extra shape {} has d_hat {:.4f} outside {}".format(
                c.label, c.d_hat, BAND))

    rows = manifest_rows(all_train, test)
    dh = np.array([c.d_hat for c in all_train])
    hist, edges = np.histogram(dh, bins=np.linspace(0.0, 1.0, 11))
    man = {
        "schema": "p4_2.shape/1",
        "derived_from": a.base,
        "derived_from_sha256": rows_sha256(json.load(open(a.base))["configs"]),
        "extension": {
            "n_extra": len(extra), "band": list(BAND), "seed": EXTRA_SEED,
            "why": ("P4-2's sweep put the accuracy MINIMUM at shallow notches (+0.741 at d_hat "
                    "0.21) and the corpus held only 13 of 60 training shapes in [0.10, 0.45) "
                    "(D72). Arm D fills that band and changes nothing else, so a flattened U "
                    "means data."),
            "band_note": ("Stops at 0.45, NOT the 0.5 the chunk spec names: the held-out slab "
                          "is [0.45, 0.60] and a training shape there would invalidate the "
                          "in-slab Gate-2 metric for every arm, not only for D."),
        },
        "seed": SEED, "test_seed": TEST_SEED,
        "family": "rectangle with a rectangular notch removed from the TOP-LEFT corner; "
                  "d = w = 0 is a pure rectangle, d, w > 0 is an L",
        "L_range": list(L_RANGE), "W_range": list(W_RANGE),
        "d_frac_max": D_FRAC_MAX, "w_frac_max": W_FRAC_MAX,
        "d_hat_def": "d / (d_frac_max * W)", "w_hat_def": "w / (w_frac_max * L)",
        "d_hat_holdout": list(D_HAT_HOLDOUT),
        "alpha": ALPHA, "src": list(SRC),
        "n_train": len(all_train), "n_train_original": N_TRAIN, "n_test": N_TEST,
        "n_rectangles": n_rect,
        "d_hat_hist": {"counts": hist.tolist(), "edges": [round(e, 3) for e in edges]},
        "rows_sha256": rows_sha256(rows),
        "configs": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(man, open(out, "w"), indent=1)
    print("[ok] {} train ({} original + {} extra) + {} test -> {}".format(
        len(all_train), N_TRAIN, len(extra), len(test), out))
    print("     rows_sha256 {}".format(man["rows_sha256"][:12]))
    print("     d_hat histogram (training):")
    for lo, hi, c in zip(edges[:-1], edges[1:], hist):
        tag = "  <-- HELD-OUT SLAB" if abs(lo - 0.5) < 1e-9 else (
            "  <-- extension band" if BAND[0] <= lo < BAND[1] else "")
        print("       [{:.1f},{:.1f})  {:2d} {}{}".format(lo, hi, c, "#" * int(c), tag))
    n_band = int(((dh >= BAND[0]) & (dh < BAND[1])).sum())
    print("     shapes in {}: {} (was 13)".format(BAND, n_band))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
