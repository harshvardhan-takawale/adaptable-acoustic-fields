"""Blocking dataset gate for the P4-2 notch family. Exit 3 = STOP.

Six items. Items 1-4 are bookkeeping and geometry; **item 5 tests the SIMULATOR**, and it is the
only one that can catch the failure that matters. If `extra_walls` were silently dropped -- or
the notch mask landed in the wrong place -- every bookkeeping item would still pass and we would
train on 75 plain shoeboxes while believing they were L-rooms. Item 5 compares a rigid notch
against an absorbing one: on a shoebox those two files are byte-identical.

Item 6 is a REPORTED census, not a gate. The NLOS fraction across this family turns out to be
very thin (shallow or narrow notches cast almost no shadow), and Gate 2's NLOS criterion depends
on it, so the real distribution is measured here against built data rather than left to the
analytic estimate made while planning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import aaf.sim.fdtd_2d as F
from aaf.data.shape_configs import (
    ALPHA, D_HAT_HOLDOUT, GRID_DX, MIN_RECTANGLES, N_TEST_IN_SLAB, SRC,
    configs_from_rows, in_holdout,
)
from aaf.sim.polygon_geom import line_of_sight, notch_solid_mask, polygon_contains

DX = 0.02
FS, N = 30720.0, 61440
MIN_NOTCH_EFFECT_DB = 0.05


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/sweeps_2d_mat/p4_2_shapes_manifest.json")
    ap.add_argument("--data-dir", default="data/track_p4_2_shapes")
    ap.add_argument("--out", default="outputs/p4_2/stage2/DATASET_GATE.json")
    a = ap.parse_args()

    man = json.load(open(a.manifest))
    rows = man["configs"]
    cfgs = configs_from_rows(rows)
    d = Path(a.data_dir)
    items, ok = [], True

    def add(name, passed, detail):
        nonlocal ok
        items.append({"item": name, "pass": bool(passed), "detail": detail})
        ok = ok and bool(passed)
        print("  [{}] {} -- {}".format("PASS" if passed else "FAIL", name, detail), flush=True)

    print("P4-2 notch-family dataset gate  (manifest sha {})".format(man["rows_sha256"][:12]))

    # ---- 1. every config built, opens, right shape ---------------------------------------
    missing, badshape = [], []
    n_rx_set = set()
    for c in cfgs:
        p = d / c.filename
        if not p.exists():
            missing.append(c.label); continue
        with h5py.File(p) as f:
            H = f["ism/H_complex"]
            n_rx_set.add(H.shape[0])
            if H.shape[1] != 601 or H.dtype != np.complex64:
                badshape.append((c.label, H.shape, str(H.dtype)))
    add("all_built", not missing, "{} of {} built{}".format(
        len(cfgs) - len(missing), len(cfgs),
        "" if not missing else "; missing {}".format(missing[:3])))
    add("shape_and_dtype", not badshape, "601 bins complex64 everywhere"
        if not badshape else str(badshape[:3]))
    # the trainer's np.stack REQUIRES one receiver count across the whole corpus
    add("uniform_receiver_count", len(n_rx_set) == 1,
        "receiver counts present: {} (np.stack needs exactly one)".format(sorted(n_rx_set)))
    if missing:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"passed": False, "items": items}, open(a.out, "w"), indent=1)
        return 3

    # ---- 2. identity: filenames and labels unique ----------------------------------------
    fns = [c.filename for c in cfgs]
    lbs = [c.label for c in cfgs]
    add("filenames_unique", len(set(fns)) == len(fns),
        "{} unique of {}".format(len(set(fns)), len(fns)))
    # PolyConfig.label is "L[6v]" for EVERY 6-edge room; a collision here would make
    # train_meta's config_labels non-identifying and collapse anything keyed on label.
    add("labels_unique", len(set(lbs)) == len(lbs),
        "{} unique of {}".format(len(set(lbs)), len(lbs)))

    # ---- 3. the held-out band, both directions -------------------------------------------
    tr_in = [c.label for c in cfgs if c.split == "train" and in_holdout(c.d_hat)]
    te_in = [c for c in cfgs if c.split == "test" and in_holdout(c.d_hat)]
    add("no_train_in_holdout", not tr_in,
        "0 training shapes in d_hat {}{}".format(list(D_HAT_HOLDOUT),
                                                 "" if not tr_in else "; got " + str(tr_in[:3])))
    add("enough_test_in_holdout", len(te_in) >= N_TEST_IN_SLAB,
        "{} test shapes in the band (need >= {})".format(len(te_in), N_TEST_IN_SLAB))
    n_rect = sum(1 for c in cfgs if c.split == "train" and c.is_rect)
    add("rectangles_anchor_the_family", n_rect >= MIN_RECTANGLES,
        "{} pure rectangles in training (need >= {})".format(n_rect, MIN_RECTANGLES))

    # ---- 4. geometry: air mask == polygon, node for node, and faces ON the grid -----------
    worst_nodes, worst_face = 0, 0.0
    offenders = []
    for c in cfgs:
        nx, dxx = F._fit_axis(c.L, DX, "L")
        ny, dxy = F._fit_axis(c.W, DX, "W")
        m = notch_solid_mask(nx, ny, dxx, dxy, c.W, c.d, c.w)
        geom = F.build_geometry(c.L, c.W, [ALPHA] * 4, dx=DX,
                                extra_walls=([{"type": "mask", "solid": m, "alpha": ALPHA}]
                                             if m.any() else None))
        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        want = polygon_contains(c.verts, np.stack([ii * dxx, jj * dxy], axis=-1))
        bad = int((geom.air != want).sum())
        if bad:
            offenders.append((c.label, bad))
        worst_nodes = max(worst_nodes, bad)
        if not c.is_rect:
            worst_face = max(worst_face,
                             abs(round(c.w / dxx) * dxx - c.w),
                             abs(round((c.W - c.d) / dxy) * dxy - (c.W - c.d)))
    add("air_mask_matches_polygon", worst_nodes == 0,
        "worst disagreement {} nodes over {} rooms{}".format(
            worst_nodes, len(cfgs), "" if not offenders else "; " + str(offenders[:3])))
    add("notch_faces_on_grid", worst_face < 1e-9,
        "worst face displacement {:.2e} m (must be ~0, or the tokens describe a different "
        "room than the solver simulates)".format(worst_face))

    # ---- 5. THE SIMULATOR ITEM ------------------------------------------------------------
    # A rigid notch and an absorbing notch can only differ if the notch FACES exist. On a plain
    # shoebox -- i.e. if extra_walls were dropped -- these two runs would be identical.
    probe = max((c for c in cfgs if not c.is_rect), key=lambda c: c.d_hat * c.w_hat)
    nx, dxx = F._fit_axis(probe.L, DX, "L")
    ny, dxy = F._fit_axis(probe.W, DX, "W")
    m = notch_solid_mask(nx, ny, dxx, dxy, probe.W, probe.d, probe.w)
    with h5py.File(d / probe.filename) as f:
        rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
    lv = []
    for alpha_notch in (ALPHA, 0.0):
        res = F.simulate(probe.L, probe.W, [ALPHA] * 4, SRC, rx, dx=DX, fs=FS, n=N,
                         extra_walls=[{"type": "mask", "solid": m, "alpha": alpha_notch}])
        H = np.asarray(res["H_complex"])[:, 1:601]
        lv.append(20.0 * np.log10(np.abs(H).mean() + 1e-12))
    delta = abs(lv[1] - lv[0])
    add("notch_is_physically_present", delta >= MIN_NOTCH_EFFECT_DB,
        "rigid vs absorbing notch differ by {:.3f} dB on {} (>= {})".format(
            delta, probe.label, MIN_NOTCH_EFFECT_DB))

    # ---- 6. NLOS census: REPORTED, not gated ----------------------------------------------
    cens = []
    for c in cfgs:
        with h5py.File(d / c.filename) as f:
            los = np.asarray(json.loads(f.attrs["line_of_sight"]), dtype=bool)
        cens.append({"label": c.label, "split": c.split, "d_hat": round(c.d_hat, 4),
                     "w_hat": round(c.w_hat, 4), "in_holdout": bool(in_holdout(c.d_hat)),
                     "n_rx": int(len(los)), "n_nlos": int((~los).sum()),
                     "nlos_fraction": float(1.0 - los.mean())})
    te = [r for r in cens if r["split"] == "test"]
    ins = [r for r in te if r["in_holdout"]]
    out_ = [r for r in te if not r["in_holdout"]]
    zero = [r["label"] for r in te if r["n_nlos"] == 0]
    print("\n  [INFO] NLOS census (reported, NOT gated):")
    print("     in-slab  test: mean {:.1%}, pooled {} NLOS receivers over {} shapes".format(
        float(np.mean([r["nlos_fraction"] for r in ins])),
        sum(r["n_nlos"] for r in ins), len(ins)))
    print("     out-slab test: mean {:.1%}, pooled {} NLOS receivers over {} shapes".format(
        float(np.mean([r["nlos_fraction"] for r in out_])),
        sum(r["n_nlos"] for r in out_), len(out_)))
    print("     test shapes with ZERO NLOS receivers: {}{}".format(
        len(zero), "" if not zero else " -- no per-shape NLOS Pearson is computable for these"))
    print("     => Gate 2's NLOS criterion must be evaluated on the POOLED in-slab receiver "
          "set;\n        no single in-slab shape supports an independent estimate.")

    out = {"gate": "p4_2.shapes/1", "passed": ok, "items": items,
           "manifest_sha": man["rows_sha256"],
           "n_train": sum(1 for c in cfgs if c.split == "train"),
           "n_test": sum(1 for c in cfgs if c.split == "test"),
           "n_rx_per_room": sorted(n_rx_set),
           "notch_effect_db": float(delta), "notch_probe": probe.label,
           "nlos_census": cens,
           "nlos_summary": {
               "in_slab_mean": float(np.mean([r["nlos_fraction"] for r in ins])),
               "in_slab_pooled_nlos_rx": int(sum(r["n_nlos"] for r in ins)),
               "out_slab_mean": float(np.mean([r["nlos_fraction"] for r in out_])),
               "out_slab_pooled_nlos_rx": int(sum(r["n_nlos"] for r in out_)),
               "test_shapes_with_zero_nlos": zero,
               "note": ("Thin by construction: shadow needs BOTH depth and width, and the "
                        "family samples w_hat down to 0.18. Gate 2's NLOS deficit must be read "
                        "on the pooled in-slab receiver set, with per-shape counts shown; no "
                        "single in-slab shape supports an independent NLOS Pearson.")}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print("\n{} -> {}".format("GATE PASS" if ok else "GATE FAIL (exit 3)", a.out))
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
