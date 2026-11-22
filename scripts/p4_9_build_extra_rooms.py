"""P4-9 extra dense rooms (D181 E8: S-5's exception and S-3).

(1) CONFIRMATION SET (S-5 exception, D181 c6): 8 U + 8 C3, NEW instances from the SAME generators as the frozen unseen
    set (`scripts/build_p4_8_unseen.py`), seed 20261122 in place of the originals:
      U : `sample_family_shapes(10, split="test", seed=20261122, n_in_slab=N_FAM_TEST_IN_SLAB)` -- the FAM test split's
          own call with a new seed -- excluding every FAM / frozen-unseen filename; its U rooms sorted by
          (d_hat, filename), then the frozen set's picks (4, 5, 3, 6, 2, 7, 1, 8).
      C3: `draw_c3(8, seed=20261122)` (NW + NE + SE, the frozen set's ranges and constraints).
    Simulated dense exactly as the frozen set (`build_p4_5_rooms.build_famdense`, source (0.5, 1.6)).
    The manifest's rows_sha256 is frozen in a D-entry BEFORE the first score; the set is scored ONCE, for S2X @240K vs
    R0 @120K (D181 c6).
      python scripts/p4_9_build_extra_rooms.py --confirm-manifest
      python scripts/p4_9_build_extra_rooms.py --confirm --idx I        (2 rooms per index, 0-7)
(2) SECOND SOURCE for the 16 frozen unseen rooms (S-3, descriptive): the same dense receivers, the source moved by a
    fixed rule: the first of (L - 0.5, 1.6), (L / 2, 1.0), (L / 2, W / 2) that is an air node >= 0.3 m from every wall.
    The geometry is proven first: the rerun with the ORIGINAL source reproduces 50 stored receivers (<= 1e-6).
      python scripts/p4_9_build_extra_rooms.py --src2 --idx I           (one room per index, 0-15)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np

CONFIRM_SEED = 20261122
CONFIRM_MAN = Path("configs/sweeps_2d_mat/p4_9_confirm_manifest.json")
CONFIRM_OUT = Path("data/track_p4_9_confirm")
SRC2_OUT = Path("data/track_p4_9_unseen_src2")
UNSEEN_MAN = "configs/sweeps_2d_mat/p4_8_unseen_manifest.json"
UNSEEN_DIR = Path("data/track_p4_8_unseen")


def confirm_rooms():
    from aaf.data.multi_notch import sample_family_shapes
    from scripts.build_p4_4_family_manifest import N_FAM_TEST_IN_SLAB
    from scripts.build_p4_8_unseen import FAM_MANIFEST, U_PICKS, draw_c3
    fam = json.load(open(FAM_MANIFEST))["configs"]
    frozen = json.load(open(UNSEEN_MAN))["configs"]
    used = {r["filename"] for r in fam} | {r["filename"] for r in frozen}
    test = sample_family_shapes(10, split="test", seed=CONFIRM_SEED, n_in_slab=N_FAM_TEST_IN_SLAB, exclude=sorted(used))
    us = sorted([c for c in test if c.kind == "U"], key=lambda c: (c.d_hat, c.filename))
    assert len(us) == 10, len(us)
    u = [(i, us[i]) for i in U_PICKS]
    c3, tried = draw_c3(8, seed=CONFIRM_SEED)
    for c in c3:
        assert c.filename not in used, c.filename
    return u, c3, tried


def write_confirm_manifest():
    from scripts.build_p4_8_unseen import row
    u, c3, tried = confirm_rooms()
    rows = [row(c, "U", {"u_sorted_index": i, "origin": "confirm"}) for i, c in u]
    rows += [row(c, "C3", {"draw": j}) for j, c in enumerate(c3)]
    assert len({r["filename"] for r in rows}) == 16
    man = {"schema": "p4_1.poly/1", "set": "P4-9 confirmation set (D181 c6; manager E8 / S-5 exception)",
           "built": "2026-11-22",
           "u_rule": "sample_family_shapes(10, 'test', seed={}, n_in_slab=N_FAM_TEST_IN_SLAB, exclude=FAM + frozen "
                     "unseen); U sorted by (d_hat, filename); picks (4, 5, 3, 6, 2, 7, 1, 8)".format(CONFIRM_SEED),
           "c3_rule": "NW+NE+SE, first 8 valid draws of default_rng([{}, k]) ({} tried)".format(CONFIRM_SEED, tried),
           "use": "scored ONCE: S2X @240K vs R0 @120K at the 2a final read",
           "rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(), "configs": rows}
    CONFIRM_MAN.write_text(json.dumps(man, indent=1))
    print("wrote", CONFIRM_MAN, man["rows_sha256"][:16])
    for r in rows:
        print("  ", r["family"], r["label"], "d_hat {:.3f}".format(r["d_hat"]))


def build_confirm(idx, chunk=2):
    from scripts.build_p4_5_rooms import build_famdense
    man = json.loads(CONFIRM_MAN.read_text())
    u, c3, _ = confirm_rooms()
    cfgs = [c for _, c in u] + list(c3)
    assert [c.filename for c in cfgs] == [r["filename"] for r in man["configs"]], "the manifest is not this draw"
    CONFIRM_OUT.mkdir(parents=True, exist_ok=True)
    for r, c in list(zip(man["configs"], cfgs))[idx * chunk:(idx + 1) * chunk]:
        print(build_famdense(r["family"], c, CONFIRM_OUT), c.filename, flush=True)


def src2_for(r, geom, dxx, dxy):
    from aaf.sim.polygon_geom import polygon_contains
    V = r["verts"]
    for cand in ((r["L"] - 0.5, 1.6), (r["L"] / 2, 1.0), (r["L"] / 2, r["W"] / 2)):
        ij = np.rint(np.asarray(cand) / [dxx, dxy]).astype(int)
        p = ij * np.array([dxx, dxy])
        if not polygon_contains(V, p[None])[0] or not geom.air[ij[0], ij[1]]:
            continue
        clear = min(_seg_dist(p, a, b) for a, b in zip(V, V[1:] + V[:1]))
        if clear >= 0.3:
            return tuple(float(x) for x in p)
    raise RuntimeError("no second-source candidate fits")


def _seg_dist(p, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0.0, 1.0)
    return float(np.hypot(*(p - (a + t * (b - a)))))


def build_src2(idx):
    import aaf.sim.fdtd_2d as F
    from aaf.sim.polygon_geom import line_of_sight
    from scripts.p4_9_fdtd_stencils import ALPHA, C, DX, NB, geometry, load_room, simulate
    row = json.load(open(UNSEEN_MAN))["configs"][idx]
    path = UNSEEN_DIR / row["filename"]
    r = load_room(path)
    geom, extra, dxx, dxy = geometry(r)
    # proves the geometry: one chunk (one node + the 50 check receivers) must reproduce the stored H
    simulate(r, extra, np.rint(r["rx"][:1] / [dxx, dxy]).astype(int), dxx, dxy)
    s2 = src2_for(r, geom, dxx, dxy)
    t0 = time.time()
    H = np.zeros((len(r["rx"]), NB), np.complex64)
    per = 5000
    for s in range(0, len(r["rx"]), per):
        res = F.simulate(r["L"], r["W"], [ALPHA] * 4, s2, r["rx"][s:s + per], dx=DX, fs=r["fs"], n=r["n"], c=C,
                         extra_walls=extra)
        H[s:s + per] = np.asarray(res["H_complex"])[:, :NB].astype(np.complex64)
        assert np.abs(np.asarray(res["meta"]["rx_pos_snapped"]) - r["rx"][s:s + per]).max() < 1e-9
        assert np.abs(np.asarray(res["meta"]["src_pos_snapped"]) - np.asarray(s2)).max() < 1e-9
    los = line_of_sight(r["verts"], s2, r["rx"])
    SRC2_OUT.mkdir(parents=True, exist_ok=True)
    dst = SRC2_OUT / row["filename"]
    tmp = dst.with_suffix(".h5.tmp")
    with h5py.File(path) as f0, h5py.File(tmp, "w") as f:
        f.create_dataset("ism/H_complex", data=H, compression="gzip", compression_opts=4)
        for k, v in f0.attrs.items():
            f.attrs[k] = v
        f.attrs["source_pos"] = json.dumps(list(s2))
        f.attrs["line_of_sight"] = json.dumps([bool(x) for x in los])
        f.attrs["loop_s"] = round(time.time() - t0, 1)
        f.attrs["src_rule"] = ("P4-9 S-3: first of (L-0.5, 1.6), (L/2, 1.0), (L/2, W/2) on an air node >= 0.3 m from "
                               "every wall; original source (0.5, 1.6)")
    os.replace(tmp, dst)
    print("built", dst, "src2", s2, "{:.0f}s".format(time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-manifest", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--src2", action="store_true")
    ap.add_argument("--idx", type=int, default=0)
    a = ap.parse_args()
    if a.confirm_manifest:
        write_confirm_manifest()
    elif a.confirm:
        build_confirm(a.idx)
    elif a.src2:
        build_src2(a.idx)


if __name__ == "__main__":
    main()
