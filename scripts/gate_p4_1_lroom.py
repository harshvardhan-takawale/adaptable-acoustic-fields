"""Blocking dataset gate for the P4-1 Stage 1 L-room. Exit 3 = STOP.

Four items, following the Track B gate's design. Items 1-3 are bookkeeping; **item 4 tests the
SIMULATOR**, and it is the one that matters: if `extra_walls` were silently dropped -- or the
notch mask landed in the wrong corner, which actually happened on the first build -- items 1-3
would every one of them still pass and we would train on a plain 6x5 shoebox while believing it
was an L-room.

Item 4 works because a rigid notch (alpha = 0) and an absorbing notch (alpha = 0.15) can only
differ if the notch faces exist at all. On a shoebox the two files would be byte-identical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import aaf.sim.fdtd_2d as F
from scripts.build_p4_1_lroom import (
    ALPHA, DX, L, NOTCH_X, NOTCH_Y, SRC, VERTS, W,
    inside_polygon, line_of_sight, notch_mask,
)

MIN_RX = 1000
MIN_NLOS_FRAC = 0.15
MIN_NOTCH_EFFECT_DB = 0.05


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/track_p4_1_lroom")
    ap.add_argument("--out", default="outputs/p4_1/stage1/DATASET_GATE.json")
    a = ap.parse_args()
    d = Path(a.data_dir)
    items, ok = [], True

    def add(name, passed, detail):
        nonlocal ok
        items.append({"item": name, "pass": bool(passed), "detail": detail})
        ok = ok and bool(passed)
        print("  [{}] {} -- {}".format("PASS" if passed else "FAIL", name, detail), flush=True)

    print("P4-1 Stage 1 L-room dataset gate")

    # ---- 1. both files exist, open, and have the right shape -----------------------------
    paths = {t: d / "{}.h5".format(t) for t in ("lroom", "lroom_rigidnotch")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    add("files_present", not missing, "missing: {}".format(missing) if missing else
        "both built: " + ", ".join(p.name for p in paths.values()))
    if missing:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"passed": False, "items": items}, open(a.out, "w"), indent=1)
        return 3

    with h5py.File(paths["lroom"]) as f:
        H = np.asarray(f["ism/H_complex"])
        at = dict(f.attrs)
        rx = np.asarray(json.loads(at["receiver_pos"]))
        los = np.asarray(json.loads(at["line_of_sight"]), dtype=bool)
    add("shape", H.shape == (len(rx), 601) and H.dtype == np.complex64,
        "H {} {} | {} receivers".format(H.shape, H.dtype, len(rx)))

    # ---- 2. receivers: enough of them, all inside, enough NLOS ---------------------------
    add("n_receivers", len(rx) >= MIN_RX, "{} >= {}".format(len(rx), MIN_RX))
    n_out = int((~inside_polygon(rx)).sum())
    add("receivers_inside_polygon", n_out == 0,
        "{} of {} outside the L".format(n_out, len(rx)))
    nlos = 1.0 - los.mean()
    add("nlos_fraction", nlos >= MIN_NLOS_FRAC,
        "{:.1%} NLOS ({} of {}) >= {:.0%}".format(nlos, int((~los).sum()), len(rx),
                                                  MIN_NLOS_FRAC))
    # recompute the split rather than trusting the stored one
    los2 = line_of_sight(np.asarray(SRC, float), rx)
    add("los_split_reproduces", bool(np.array_equal(los, los2)),
        "stored split matches a fresh recomputation")

    # ---- 3. the geometry the solver actually built --------------------------------------
    nx, dxx = F._fit_axis(L, DX, "L")
    ny, dxy = F._fit_axis(W, DX, "W")
    m = notch_mask(nx, ny, dxx, dxy)
    geom = F.build_geometry(L, W, [ALPHA] * 4, dx=DX,
                            extra_walls=[{"type": "mask", "solid": m, "alpha": ALPHA}])
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    want_air = inside_polygon(np.stack([ii * dxx, jj * dxy], axis=-1))
    disagree = int((geom.air != want_air).sum())
    add("air_mask_matches_polygon", disagree == 0,
        "{} nodes disagree (notch solid {} of {})".format(disagree, int(m.sum()), nx * ny))
    add("notch_node_count", int(at["notch_solid_nodes"]) == int(m.sum()),
        "recorded {} == rebuilt {}".format(at["notch_solid_nodes"], int(m.sum())))
    add("exact_fit_grid", abs(dxx - DX) < 1e-12 and abs(dxy - DX) < 1e-12,
        "dx=({:.6f}, {:.6f}) both exactly {}".format(dxx, dxy, DX))
    add("six_edges", len(VERTS) == 6, "{} polygon vertices".format(len(VERTS)))

    # ---- 4. THE SIMULATOR ITEM: the notch faces are physically present -------------------
    # A rigid notch and an absorbing notch can only differ if those faces exist. On a plain
    # 6x5 shoebox -- i.e. if `extra_walls` were dropped -- these two files would be identical.
    with h5py.File(paths["lroom_rigidnotch"]) as f:
        Hr = np.asarray(f["ism/H_complex"])
    eps = 1e-12
    band = slice(1, None)                                   # skip DC
    lvl_a = 20.0 * np.log10(np.abs(H[:, band]).mean() + eps)
    lvl_r = 20.0 * np.log10(np.abs(Hr[:, band]).mean() + eps)
    delta = abs(lvl_r - lvl_a)
    identical = bool(np.array_equal(H, Hr))
    add("notch_is_physically_present", (not identical) and delta >= MIN_NOTCH_EFFECT_DB,
        "rigid vs absorbing notch differ by {:.3f} dB (>= {}); byte-identical: {}".format(
            delta, MIN_NOTCH_EFFECT_DB, identical))

    # ---- report -------------------------------------------------------------------------
    out = {
        "gate": "p4_1.stage1.lroom/1", "passed": ok, "items": items,
        "geometry": {"L": L, "W": W, "notch": [NOTCH_X, NOTCH_Y], "verts": VERTS,
                     "src": list(SRC), "alpha": ALPHA, "area_m2": L * W - NOTCH_X * NOTCH_Y},
        "receivers": {"n": int(len(rx)), "n_los": int(los.sum()),
                      "n_nlos": int((~los).sum()), "nlos_fraction": float(nlos)},
        "solver": {"dx_x": dxx, "dx_y": dxy, "fs": float(at["fs"]),
                   "n_time_samples": int(at["n_time_samples"]),
                   "air_nodes": int(geom.air.sum()), "notch_solid_nodes": int(m.sum())},
        "notch_effect_db": float(delta),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("\n{} -> {}".format("GATE PASS" if ok else "GATE FAIL (exit 3)", a.out))
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
