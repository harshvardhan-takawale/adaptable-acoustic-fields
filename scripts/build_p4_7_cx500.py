"""P4-7 m4 (D169): build the configuration-diverse corpus CX500 and the C500 + CX500 training manifest.

  python scripts/build_p4_7_cx500.py --manifest        # freeze configs/sweeps_2d_mat/p4_7_cx500_manifest.json
                                                       # and the combined p4_7_c500_cx500_manifest.json (+ symlink dir)
  python scripts/build_p4_7_cx500.py --plan            # array size
  python scripts/build_p4_7_cx500.py --idx 0 --chunk 8 # FDTD for rooms [0, 8) of the frozen manifest

Each room: FDTD at the corpus settings (dx 0.02, fs 30720, n 61440, alpha 0.15 on every face) with its
OWN source; 800 receivers drawn from the room's own 0.02 m nodes (B6 style: off any shared lattice),
>= 0.12 m from every wall / notch face, 0.15 m from reflex corners, 0.10 m from the source, seed
[RX_SEED_CX, shape_id]. The solver's air mask is asserted node-for-node against the polygon.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

import aaf.sim.fdtd_2d as F
from aaf.data.cx_rooms import (ALPHA, CX_SEED, CxNotch, CxRoom, edge_distance, reflex_vertices,
                               sample_cx_rooms)
from aaf.sim.polygon_geom import line_of_sight, polygon_contains

MANIFEST = Path("configs/sweeps_2d_mat/p4_7_cx500_manifest.json")
MIX_MANIFEST = Path("configs/sweeps_2d_mat/p4_7_c500_cx500_manifest.json")
C500_MANIFEST = Path("configs/sweeps_2d_mat/p4_4_shapes_500.json")
C500_DIR = Path("data/track_p4_2_shapes")
OUT = Path("data/track_p4_7_cx500")
MIX_DIR = Path("data/track_p4_7_mix")
N_ROOMS = 500
N_RX = 800
RX_SEED_CX = 20261017
WALL_CLEAR, CORNER_CLEAR, SRC_RX_CLEAR = 0.12, 0.15, 0.10
DX, FS, N, C = 0.02, 30720.0, 61440, 343.0
BAND_HI = 601


def _sha(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def write_manifests():
    rooms = sample_cx_rooms(N_ROOMS, seed=CX_SEED)
    rows = [r.row(i) for i, r in enumerate(rooms)]
    assert len({r["filename"] for r in rows}) == len(rows)
    man = {"schema": "p4_1.poly/1", "corpus": "CX500 (P4-7 m4, D169)", "seed": CX_SEED, "n_train": len(rows),
           "rx_seed": RX_SEED_CX, "per_room_source": True,
           "why": "configuration diversity: notches at any corner / side midpoint (U topology excluded), "
                  "source uniform in air >= 0.5 m from walls and reflex corners, per-room off-lattice receivers",
           "kinds": dict(Counter(r["kind"] for r in rows)),
           "positions": dict(Counter(n["pos"] for r in rows for n in r["notches"])),
           "rows_sha256": _sha(rows), "configs": rows}
    MANIFEST.write_text(json.dumps(man, indent=1))
    # The combined training manifest: C500's 500 training rows re-expressed as polygon rows (same verts,
    # edge alphas, filename and strata, so the geom_token conditioning is identical), then CX500.
    c5 = json.loads(C500_MANIFEST.read_text())
    mix = []
    for r in c5["configs"]:
        if r["split"] != "train":
            continue
        mix.append({"i": len(mix), "split": "train", "kind": r["kind"], "shape": r["kind"], "strata": r["strata"],
                    "verts": r["verts"], "edge_alphas": r["edge_alphas"], "alphas": r["alphas"],
                    "filename": r["filename"], "src": [0.5, 1.6], "origin": "C500"})
    for r in rows:
        mix.append(dict(r, i=len(mix), origin="CX500"))
    assert len({r["filename"] for r in mix}) == len(mix) == 1000
    mm = {"schema": "p4_1.poly/1", "corpus": "C500 train + CX500 (P4-7 m4, D169)", "per_room_source": True,
          "data_dir": str(MIX_DIR), "derived_from": [str(C500_MANIFEST), str(MANIFEST)],
          "derived_from_sha256": [c5.get("rows_sha256", ""), man["rows_sha256"]],
          "d_hat_holdout": c5.get("d_hat_holdout"), "n_train": len(mix), "rows_sha256": _sha(mix), "configs": mix}
    MIX_MANIFEST.write_text(json.dumps(mm, indent=1))
    MIX_DIR.mkdir(parents=True, exist_ok=True)
    for r in mix:
        src = (C500_DIR if r["origin"] == "C500" else OUT) / r["filename"]
        link = MIX_DIR / r["filename"]
        if not link.is_symlink():
            os.symlink(os.path.relpath(src, MIX_DIR), link)
    print("wrote", MANIFEST, MIX_MANIFEST, "and", len(mix), "links in", MIX_DIR)
    print(json.dumps({k: man[k] for k in ("kinds", "positions")}))


def load_rooms():
    rows = json.loads(MANIFEST.read_text())["configs"]
    return [CxRoom(r["L"], r["W"], tuple(CxNotch(n["pos"], n["d"], n["w"], n["off"]) for n in r["notches"]),
                   tuple(r["src"]), r["shape_id"]) for r in rows], rows


def build(room: CxRoom, row: dict, force=False):
    path, done = OUT / room.filename, OUT / (room.filename + ".done")
    if done.exists() and path.exists() and not force:
        return "skip"
    verts = [tuple(v) for v in row["verts"]]
    assert verts == [tuple(v) for v in room.verts], room.label
    nx, dxx = F._fit_axis(room.L, DX, "L")
    ny, dxy = F._fit_axis(room.W, DX, "W")
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    P = np.stack([ii * dxx, jj * dxy], axis=-1)
    inside = polygon_contains(verts, P)
    solid = ~inside
    extra = [{"type": "mask", "solid": solid, "alpha": ALPHA}] if solid.any() else None
    geom = F.build_geometry(room.L, room.W, [ALPHA] * 4, dx=DX, extra_walls=extra)
    bad = int((geom.air != inside).sum())
    if bad:
        raise AssertionError("{}: air mask disagrees with polygon at {} nodes".format(room.label, bad))
    Pf = P.reshape(-1, 2)
    keep = geom.air.ravel() & (edge_distance(verts, Pf) >= WALL_CLEAR)
    for cx, cy in reflex_vertices(verts):
        keep &= np.hypot(Pf[:, 0] - cx, Pf[:, 1] - cy) > CORNER_CLEAR
    keep &= np.hypot(Pf[:, 0] - room.src[0], Pf[:, 1] - room.src[1]) >= SRC_RX_CLEAR
    pool = Pf[keep]
    rng = np.random.default_rng([RX_SEED_CX, int(room.shape_id)])
    rx = pool[np.sort(rng.choice(len(pool), size=N_RX, replace=False))]
    t0 = time.time()
    res = F.simulate(room.L, room.W, [ALPHA] * 4, room.src, rx, dx=DX, fs=FS, n=N, c=C, extra_walls=extra)
    loop_s = time.time() - t0
    H = np.asarray(res["H_complex"])[:, :BAND_HI].astype(np.complex64)
    rx_snap = np.asarray(res["meta"]["rx_pos_snapped"])
    src_snap = np.asarray(res["meta"]["src_pos_snapped"])
    assert np.abs(rx_snap - rx).max() < 1e-9 and np.abs(src_snap - np.asarray(room.src)).max() < 1e-9
    k = (rx_snap - 0.12) / 0.06
    on = float(np.all(np.abs(k - np.round(k)) < 1e-6, axis=1).mean())
    los = line_of_sight(verts, tuple(room.src), rx_snap)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("ism/H_complex", data=H, compression="gzip", compression_opts=4)
        a = f.attrs
        a["source_pos"] = json.dumps([float(x) for x in src_snap])
        a["receiver_pos"] = json.dumps(rx_snap.tolist())
        a["line_of_sight"] = json.dumps([bool(x) for x in los])
        a["split"], a["kind"], a["label"] = room.split, room.kind, room.label
        a["alphas"] = json.dumps([ALPHA] * 4)
        a["edge_alphas"] = json.dumps([ALPHA] * len(verts))
        a["verts"] = json.dumps([list(v) for v in verts])
        a["notches"] = json.dumps(row["notches"])
        a["L"], a["W"], a["shape_id"] = room.L, room.W, int(room.shape_id)
        a["n_air_nodes"], a["notch_solid_nodes"] = int(geom.air.sum()), int(solid.sum())
        a["nlos_fraction"] = float(1.0 - np.mean(los))
        a["solver"] = "fdtd_2d_slf_kw"
        a["dx"], a["dx_x"], a["dx_y"], a["fs"], a["n_time_samples"] = DX, dxx, dxy, FS, N
        a["band_hi_hz"], a["n_freq_bins"], a["n_rx"] = 300.0, H.shape[1], N_RX
        a["loop_s"] = round(loop_s, 1)
        a["rx_construction"] = ("800 drawn uniformly from every in-room 0.02 m node, >= {} m from walls, {} m from "
                                "reflex corners, {} m from the source; seed [{}, shape_id] (P4-7 m4, D169)".format(
                                    WALL_CLEAR, CORNER_CLEAR, SRC_RX_CLEAR, RX_SEED_CX))
        a["rx_candidate_pool"], a["frac_on_old_lattice"] = int(len(pool)), on
    os.replace(tmp, path)
    done.touch()
    print("built {:7s} {:44s} src ({:.2f}, {:.2f}) pool {:6d} on-lattice {:.3f} nlos {:.2f} {:.0f}s".format(
        room.kind, room.label, room.src[0], room.src[1], len(pool), on, 1 - np.mean(los), loop_s), flush=True)
    return "built"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=8)
    a = ap.parse_args()
    if a.manifest:
        write_manifests()
        return
    rooms, rows = load_rooms()
    if a.plan or a.idx is None:
        print("n_rooms {} -> --array=0-{} at chunk {}".format(len(rooms), (len(rooms) - 1) // a.chunk, a.chunk))
        return
    for r, row in list(zip(rooms, rows))[a.idx * a.chunk:(a.idx + 1) * a.chunk]:
        print(build(r, row), r.filename, flush=True)


if __name__ == "__main__":
    main()
