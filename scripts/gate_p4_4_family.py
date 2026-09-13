"""Blocking dataset gate for the P4-4 multi-family corpus.

Follows the P4-2 pattern where MOST items check bookkeeping and ONE checks the SIMULATOR. The
bookkeeping items would all pass on a corpus whose masks were silently wrong -- filenames would
still be unique, counts would still balance, and the manifest would still parse. The simulator
item would not: it asserts that a two-corner room and a one-corner room with the SAME bounding
box produce measurably different fields. If `solid_blocks` had dropped the second notch, or put
it a cell from where the polygon says it is, the two would agree far too closely.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

import aaf.sim.fdtd_2d as F
from aaf.data.multi_notch import (
    D_HAT_HOLDOUT_FAM, FAMILIES, N_TOKENS, Notch, configs_from_rows, geometry_ok,
    reflex_corners, solid_blocks,
)
from aaf.data.shape_configs import ALPHA, SRC
from aaf.sim.polygon_geom import polygon_contains
from scripts.build_p4_2_shapes import DX, FS, N_RX

MANIFEST = "configs/sweeps_2d_mat/p4_4_family_manifest.json"
DATA = "data/track_p4_4_family"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--out", default="outputs/p4_4/family/DATASET_GATE.json")
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    cfgs = configs_from_rows(man["configs"])
    d = Path(a.data_dir)
    items = []

    def item(name, ok, detail):
        items.append({"name": name, "pass": bool(ok), "detail": detail})
        print("  [{}] {:34s} {}".format("PASS" if ok else "FAIL", name, detail), flush=True)

    built = [c for c in cfgs if (d / c.filename).exists()]
    item("all_rooms_built", len(built) == len(cfgs),
         "{} of {} built".format(len(built), len(cfgs)))

    nbins, nrx, shapes = set(), set(), []
    for c in built:
        with h5py.File(d / c.filename) as f:
            H = f["ism/H_complex"]
            nbins.add(H.shape[1]); nrx.add(H.shape[0])
            shapes.append((c, dict(f.attrs)))
    item("uniform_freq_bins", len(nbins) == 1, "bins {}".format(sorted(nbins)))
    item("uniform_receiver_count", len(nrx) == 1,
         "receiver counts {} (np.stack needs exactly one)".format(sorted(nrx)))

    fns = [c.filename for c in cfgs]
    item("unique_filenames", len(set(fns)) == len(fns),
         "{} unique of {} (a U and a T must not alias)".format(len(set(fns)), len(fns)))
    labels = [c.label for c in cfgs]
    item("unique_labels", len(set(labels)) == len(labels),
         "{} unique of {}".format(len(set(labels)), len(labels)))

    tr = [c for c in cfgs if c.split == "train"]
    te = [c for c in cfgs if c.split == "test"]
    slab = [c.label for c in tr if D_HAT_HOLDOUT_FAM[0] <= c.d_hat <= D_HAT_HOLDOUT_FAM[1]]
    item("holdout_slab_empty_in_train", not slab,
         "{} training rooms in {}".format(len(slab), list(D_HAT_HOLDOUT_FAM)))
    ct, cte = Counter(c.kind for c in tr), Counter(c.kind for c in te)
    item("families_balanced", len(set(ct.values())) == 1 and len(set(cte.values())) == 1,
         "train {} | test {}".format(dict(ct), dict(cte)))
    bad_tok = [(c.label, c.n_tokens) for c in cfgs if c.n_tokens != N_TOKENS[c.kind]]
    item("token_counts_match_family", not bad_tok,
         "rect 4 / L 6 / U,T,DN 8; {} mismatches".format(len(bad_tok)))

    bad_geom = [(c.label, geometry_ok(c.L, c.W, c.notches)[1]) for c in cfgs
                if not geometry_ok(c.L, c.W, c.notches)[0]]
    item("all_rooms_legal", not bad_geom, "{} illegal".format(len(bad_geom)))

    off = [(c.label, v) for c in cfgs
           for v in (c.L, c.W) + tuple(x for n in c.notches for x in (n.d, n.w, n.x0))
           if abs(v / 0.02 - round(v / 0.02)) > 1e-9]
    item("grid_aligned_D67c", not off,
         "{} parameters off the 0.02 m grid".format(len(off)))

    # the per-room node-for-node check the builder already ran, re-verified on a sample
    worst = 0
    for c in built[::17]:
        nx, dxx = F._fit_axis(c.L, DX, "L")
        ny, dxy = F._fit_axis(c.W, DX, "W")
        m = solid_blocks(nx, ny, dxx, dxy, c.L, c.W, c.notches)
        geom = F.build_geometry(c.L, c.W, [ALPHA] * 4, dx=DX,
                                extra_walls=([{"type": "mask", "solid": m, "alpha": ALPHA}]
                                             if m.any() else None))
        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        want = polygon_contains(c.verts, np.stack([ii * dxx, jj * dxy], axis=-1))
        worst = max(worst, int((geom.air != want).sum()))
    item("solver_matches_polygon", worst == 0,
         "worst disagreement {} nodes over {} sampled rooms".format(worst, len(built[::17])))

    src_bad = [c.label for c in cfgs
               if not bool(polygon_contains(c.verts, np.asarray([SRC], float))[0])]
    item("source_in_air_everywhere", not src_bad,
         "{} rooms with the source inside solid".format(len(src_bad)))

    refl = [(c.label, len(reflex_corners(c.L, c.W, c.notches))) for c in cfgs]
    want_refl = {"rect": 0, "L": 1, "U": 2, "T": 2, "DN": 2}
    bad_refl = [(l, n) for (l, n), c in zip(refl, cfgs) if n != want_refl[c.kind]]
    item("reflex_corner_counts", not bad_refl, "{} mismatches".format(len(bad_refl)))

    # ---------------- THE SIMULATOR-LEVEL ITEM ----------------
    # A two-corner room must differ MEASURABLY from a one-corner room with the same bounding box.
    # Every item above passes on a corpus whose second notch was silently dropped; this does not.
    L, W = 6.00, 5.00
    one = (Notch("NW", 1.20, 1.60),)
    two = (Notch("NW", 1.20, 1.60), Notch("NE", 1.00, 1.40))
    # must be in AIR for BOTH rooms: outside NW [0,1.6]x[3.8,5] and outside NE [4.6,6]x[4,5].
    # The solver refuses a receiver on a solid node, which is how the first choice here was
    # caught -- it sat inside the very notch the item exists to detect.
    rx = [[4.20, 1.00], [3.00, 3.00], [2.50, 4.50]]
    out = []
    for ns in (one, two):
        nx, dxx = F._fit_axis(L, DX, "L")
        ny, dxy = F._fit_axis(W, DX, "W")
        m = solid_blocks(nx, ny, dxx, dxy, L, W, ns)
        r = F.simulate(L, W, [ALPHA] * 4, SRC, rx, dx=DX, fs=FS, n=int(2 * FS), c=343.0,
                       extra_walls=[{"type": "mask", "solid": m, "alpha": ALPHA}])
        out.append(np.asarray(r["H_complex"])[:, :601])
    db = 20 * np.log10(np.maximum(np.abs(out[0]), 1e-30)) - \
        20 * np.log10(np.maximum(np.abs(out[1]), 1e-30))
    delta = float(np.mean(np.abs(db)))
    item("second_notch_changes_the_physics", delta >= 0.5,
         "L vs U with the same bbox differ by {:.3f} dB mean (>= 0.5)".format(delta))

    ok = all(i["pass"] for i in items)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"passed": ok, "n_items": len(items), "items": items,
               "n_train": len(tr), "n_test": len(te),
               "manifest_sha": man.get("rows_sha256", "")},
              open(a.out, "w"), indent=1)
    print("\nGATE: {}  ({}/{} items)".format(
        "PASS" if ok else "FAIL", sum(i["pass"] for i in items), len(items)))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
