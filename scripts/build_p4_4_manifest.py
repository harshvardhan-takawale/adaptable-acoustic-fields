"""Freeze the P4-4 corpus manifests: P4-3's 92 shapes topped up to 250 and 500.

P4-3 established that corpus density along a shape parameter is a first-order driver of accuracy
along it (D75): 32 extra shapes in one thin band eliminated the U-shaped accuracy curve and
passed Gate 2. This scales that finding into a data-scaling CURVE, which is the shape analogue of
P3-2d's material sampling law.

THREE INVARIANTS THIS SCRIPT EXISTS TO ENFORCE
----------------------------------------------
1. **P4-3's 92 training rows and 15 test rows must reproduce EXACTLY**, so the 107 already-built
   rooms are reused and every P4-2/P4-3 number stays comparable. Asserted row-for-row.
2. **`sample_test_shapes` must be called on the ORIGINAL 60.** It derives its interior bounds and
   its maximin rejection from whatever training set it is handed, so calling it with 250 shapes
   would silently produce a DIFFERENT frozen test set and invalidate every cross-chunk
   comparison. The ordering below is load-bearing, not incidental.
3. **No training shape inside the held-out slab**, checked after quantization, and every
   parameter on the 0.02 m FDTD grid (D67c) -- otherwise the tokens and the solver describe
   different rooms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aaf.data.shape_configs import (
    ALPHA, BALANCE_SEED, D_FRAC_MAX, D_HAT_HOLDOUT, EXTRA_SEED, L_RANGE, MIN_RECTANGLES,
    N_TEST, N_TEST_IN_SLAB, N_TRAIN, SEED, SRC, TEST_SEED, W_FRAC_MAX, W_RANGE, in_holdout,
    manifest_rows, rows_sha256, sample_balanced_shapes, sample_extra_shapes, sample_test_shapes,
    sample_train_shapes,
)
from scripts.build_p4_3_manifest import KEYS, N_EXTRA, BAND

P4_3 = "configs/sweeps_2d_mat/p4_3_shapes_manifest.json"


def _assert_reproduces(rows_new, base_path, n_expect):
    base = json.load(open(base_path))["configs"]
    if len(base) != n_expect:
        raise AssertionError("{} has {} rows, expected {}".format(base_path, len(base), n_expect))
    by_fn = {r["filename"]: r for r in rows_new}
    bad = []
    for old in base:
        new = by_fn.get(old["filename"])
        if new is None:
            bad.append((old["filename"], "MISSING", None, None)); continue
        for k in KEYS:
            if k in ("d_hat", "w_hat"):
                if abs(float(old[k]) - float(new[k])) > 1e-9:
                    bad.append((old["filename"], k, old[k], new[k]))
            elif old[k] != new[k]:
                bad.append((old["filename"], k, old[k], new[k]))
    if bad:
        raise AssertionError(
            "{} P4-3 row(s) DIFFER or are missing; the built corpus could not be reused "
            "safely. First 5: {}".format(len(bad), bad[:5]))
    print("[ok] all {} P4-3 rows reproduce exactly".format(len(base)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, required=True, help="total training shapes (250/500)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.out or "configs/sweeps_2d_mat/p4_4_shapes_{}.json".format(a.n_train))
    if out.exists() and not a.force:
        raise SystemExit("{} exists and is FROZEN; pass --force".format(out))

    # ORDER IS LOAD-BEARING: test shapes come from the ORIGINAL 60 (see the module docstring).
    train60 = sample_train_shapes()
    test = sample_test_shapes(train60)
    extra32 = sample_extra_shapes(list(train60) + list(test), N_EXTRA, band=BAND)
    base92 = list(train60) + list(extra32)

    n_new = a.n_train - len(base92)
    if n_new < 0:
        raise SystemExit("--n-train {} is below the frozen base of {}".format(
            a.n_train, len(base92)))
    balanced = sample_balanced_shapes(list(base92) + list(test), n_new)
    all_train = base92 + balanced
    rows = manifest_rows(all_train, test)
    _assert_reproduces(rows, P4_3, 107)

    # ---- blocking invariants ----
    slab = [c.label for c in all_train if in_holdout(c.d_hat)]
    if slab:
        raise AssertionError("{} training shapes in the held-out slab: {}".format(
            len(slab), slab[:5]))
    if sum(1 for c in test if in_holdout(c.d_hat)) < N_TEST_IN_SLAB:
        raise AssertionError("too few in-slab test shapes")
    fns = [c.filename for c in all_train + list(test)]
    if len(set(fns)) != len(fns):
        raise AssertionError("duplicate filenames")
    if len({c.label for c in all_train + list(test)}) != len(fns):
        raise AssertionError("duplicate labels")
    if sum(1 for c in all_train if c.is_rect) < MIN_RECTANGLES:
        raise AssertionError("too few rectangles")
    off = [(c.label, v) for c in all_train + list(test)
           for v in (c.L, c.W, c.d, c.w) if abs(v / 0.02 - round(v / 0.02)) > 1e-9]
    if off:
        raise AssertionError("{} parameter(s) off the 0.02 grid (D67c): {}".format(
            len(off), off[:5]))

    hist = {}
    for key, lo, hi in (("d_hat", 0, 1), ("w_hat", 0, 1), ("L", 5.0, 7.0), ("W", 4.0, 5.5)):
        v = np.array([getattr(c, key) for c in all_train])
        h, e = np.histogram(v, bins=np.linspace(lo, hi, 11))
        hist[key] = {"counts": h.tolist(), "edges": [round(x, 3) for x in e]}

    man = {
        "schema": "p4_2.shape/1",
        "derived_from": P4_3, "derived_from_sha256": rows_sha256(json.load(open(P4_3))["configs"]),
        "extension": {
            "n_balanced": len(balanced), "seed": BALANCE_SEED,
            "method": "greedy maximin in normalized (L, W, d_hat, w_hat)",
            "why": ("D75 showed corpus density along a shape parameter is a first-order driver "
                    "of accuracy along it. This scales that into a data-scaling curve. Maximin "
                    "fills the sparsest regions and equalizes the marginals as a consequence, "
                    "rather than binning any axis by hand."),
        },
        "seed": SEED, "test_seed": TEST_SEED, "extra_seed": EXTRA_SEED,
        "family": "rectangle with a rectangular notch removed from the TOP-LEFT corner",
        "L_range": list(L_RANGE), "W_range": list(W_RANGE),
        "d_frac_max": D_FRAC_MAX, "w_frac_max": W_FRAC_MAX,
        "d_hat_def": "d / (d_frac_max * W)", "w_hat_def": "w / (w_frac_max * L)",
        "d_hat_holdout": list(D_HAT_HOLDOUT), "alpha": ALPHA, "src": list(SRC),
        "n_train": len(all_train), "n_train_p4_3": len(base92), "n_test": N_TEST,
        "n_rectangles": sum(1 for c in all_train if c.is_rect),
        "marginal_hist": hist,
        "rows_sha256": rows_sha256(rows), "configs": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(man, open(out, "w"), indent=1)
    print("[ok] {} train ({} from P4-3 + {} balanced) + {} test -> {}".format(
        len(all_train), len(base92), len(balanced), len(test), out))
    print("     rows_sha256 {}".format(man["rows_sha256"][:12]))
    # Report the imbalance EXCLUDING the deliberately-unsampled bins, or the number is an
    # artifact of the design rather than a coverage defect -- the same mistake P4-2's raw gap
    # metric made (it read 0.166 and flagged "too coarse" for the hold-out band itself).
    # d_hat bin 5 is the hold-out slab; w_hat bin 0 is below the sliver floor, where only the
    # pure rectangles (w_hat = 0) sit.
    skip = {"d_hat": {5}, "w_hat": {0}}
    for key in ("d_hat", "w_hat", "L", "W"):
        h = hist[key]["counts"]
        keep = [c for i, c in enumerate(h) if i not in skip.get(key, set())]
        note = {"d_hat": "  (excl. hold-out slab)",
                "w_hat": "  (excl. sliver floor)"}.get(key, "")
        print("     {:6s} {}   sampled bins {}-{}  ratio {:.2f}x{}".format(
            key, " ".join("{:>3d}".format(c) for c in h), min(keep), max(keep),
            max(keep) / max(min(keep), 1), note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
