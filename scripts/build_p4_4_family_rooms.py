"""P4-4 Task 3: FDTD ground truth for the multi-family corpus (rect / L / U / T / DN).

Generalizes `build_p4_2_shapes.py` from one top-left notch to any legal notch set. Two things it
keeps verbatim because they earned their place, and one it must extend:

  * the polygon is cut with `mask`, never `wall_segments` (D65);
  * every room is reconstructed analytically from its vertex list and asserted **node-for-node**
    against `geom.air`. That assert caught P4-1's wrong-corner mask, and it matters more here:
    a mid-wall (T) notch has TWO interior x-faces, a case the single-notch index convention never
    exercised, and getting the `+1` wrong on one side puts a wall one cell from where the polygon
    says it is;
  * **the reflex-corner clearance now loops over EVERY reflex vertex.** The single-notch builder
    clears one hard-coded corner; with two, the receivers nearest the second one would sit in the
    unresolved r^(2/3) singular field and be scored as if they were fine.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

import aaf.sim.fdtd_2d as F
from aaf.data.multi_notch import configs_from_rows, reflex_corners, solid_blocks
from aaf.data.shape_configs import ALPHA, SRC
from aaf.sim.polygon_geom import assert_extents_match_polygon, line_of_sight, polygon_contains
from scripts.build_p4_2_shapes import (
    BAND_HI_HZ, C, DX, FS, N, N_RX, RX_CORNER_CLEAR, RX_SEED, RX_WALL_CLEAR,
)

MANIFEST = "configs/sweeps_2d_mat/p4_4_family_manifest.json"


def receivers(cfg, geom, dxx, dxy):
    """Exactly N_RX interior points, seeded so a rebuild is identical.

    Two notches remove up to ~40% of the bounding box AND add clearance bands around a second
    reflex corner, so the candidate pool is markedly smaller than in the single-notch family.
    The `len(P) < N_RX` raise below is the check most likely to fire; it is a hard error rather
    than a silent thinning because the trainer stacks targets and needs a constant count.
    """
    V = cfg.verts
    step = 0.06
    xs = np.arange(RX_WALL_CLEAR, cfg.L, step)
    ys = np.arange(RX_WALL_CLEAR, cfg.W, step)
    P = np.array([[x, y] for x in xs for y in ys], dtype=float)
    P = P[polygon_contains(V, P)]
    keep = np.ones(len(P), dtype=bool)
    for (x0, y0), (x1, y1) in zip(V, list(V[1:]) + [V[0]]):
        if abs(x1 - x0) < 1e-9:                                   # vertical wall
            keep &= ~((np.abs(P[:, 0] - x0) < RX_WALL_CLEAR)
                      & (P[:, 1] >= min(y0, y1) - RX_WALL_CLEAR)
                      & (P[:, 1] <= max(y0, y1) + RX_WALL_CLEAR))
        else:                                                     # horizontal wall
            keep &= ~((np.abs(P[:, 1] - y0) < RX_WALL_CLEAR)
                      & (P[:, 0] >= min(x0, x1) - RX_WALL_CLEAR)
                      & (P[:, 0] <= max(x0, x1) + RX_WALL_CLEAR))
    P = P[keep]
    for (cx, cy) in reflex_corners(cfg.L, cfg.W, cfg.notches):    # EVERY reflex corner
        P = P[np.hypot(P[:, 0] - cx, P[:, 1] - cy) > RX_CORNER_CLEAR]
    i = np.rint(P[:, 0] / dxx).astype(int)
    j = np.rint(P[:, 1] / dxy).astype(int)
    P = P[geom.air[i, j]]
    if len(P) < N_RX:
        raise RuntimeError("{}: only {} candidate receivers, need {}".format(
            cfg.label, len(P), N_RX))
    rng = np.random.default_rng([RX_SEED, cfg.shape_id])
    return P[np.sort(rng.choice(len(P), size=N_RX, replace=False))]


def build_one(cfg, out_dir: Path, force: bool = False) -> str:
    out_path = out_dir / cfg.filename
    done = out_dir / (cfg.filename + ".done")
    if done.exists() and out_path.exists() and not force:
        return "skip"

    nx, dxx = F._fit_axis(cfg.L, DX, "L")
    ny, dxy = F._fit_axis(cfg.W, DX, "W")
    m = solid_blocks(nx, ny, dxx, dxy, cfg.L, cfg.W, cfg.notches)
    extra = [{"type": "mask", "solid": m, "alpha": ALPHA}] if m.any() else None
    geom = F.build_geometry(cfg.L, cfg.W, [ALPHA] * 4, dx=DX, extra_walls=extra)

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    want = polygon_contains(cfg.verts, np.stack([ii * dxx, jj * dxy], axis=-1))
    bad = int((geom.air != want).sum())
    if bad:
        raise AssertionError(
            "{}: solver air mask disagrees with the polygon at {} nodes -- check the notch "
            "indices (a mid-wall notch needs the +1 on BOTH x faces)".format(cfg.label, bad))

    rx = receivers(cfg, geom, dxx, dxy)
    los = line_of_sight(cfg.verts, SRC, rx)
    t0 = time.time()
    res = F.simulate(cfg.L, cfg.W, [ALPHA] * 4, SRC, rx, dx=DX, fs=FS, n=N, c=C,
                     extra_walls=extra)
    assert_extents_match_polygon(res["meta"].get("extra_walls", []), cfg.verts)
    hi = int(round(BAND_HI_HZ / (FS / N))) + 1
    H = np.asarray(res["H_complex"])[:, :hi].astype(np.complex64)
    loop_s = time.time() - t0

    with h5py.File(out_path, "w") as f:
        f.create_dataset("ism/H_complex", data=H, compression="gzip", compression_opts=4)
        a = f.attrs
        a["source_pos"] = json.dumps(list(SRC))
        a["receiver_pos"] = json.dumps(np.asarray(res["meta"]["rx_pos_snapped"]).tolist())
        a["line_of_sight"] = json.dumps([bool(x) for x in los])
        a["split"], a["kind"], a["label"] = cfg.split, cfg.kind, cfg.label
        a["alphas"] = json.dumps(list(cfg.alphas))
        a["edge_alphas"] = json.dumps(list(cfg.edge_alphas))
        a["verts"] = json.dumps([list(v) for v in cfg.verts])
        a["notches"] = json.dumps([{"side": n.side, "d": n.d, "w": n.w, "x0": n.x0}
                                   for n in cfg.notches])
        a["L"], a["W"] = cfg.L, cfg.W
        a["d_hat"], a["w_hat"] = cfg.d_hat, cfg.w_hat
        a["n_tokens"], a["n_reflex"] = cfg.n_tokens, len(reflex_corners(cfg.L, cfg.W, cfg.notches))
        a["shape_id"] = cfg.shape_id
        a["notch_solid_nodes"] = int(m.sum())
        a["n_air_nodes"] = int(geom.air.sum())
        a["nlos_fraction"] = float(1.0 - los.mean())
        a["solver"] = "fdtd_2d_slf_kw"
        a["dx"], a["dx_x"], a["dx_y"] = DX, dxx, dxy
        a["fs"], a["n_time_samples"] = FS, N
        a["band_hi_hz"], a["n_freq_bins"], a["n_rx"] = BAND_HI_HZ, H.shape[1], int(len(rx))
        a["loop_s"] = round(loop_s, 1)
        a["extra_walls"] = json.dumps(res["meta"].get("extra_walls", []))
    done.touch()
    print("built {:6s} {:2d}tok {:42s} d_hat {:.3f} NLOS {:5.1%} {:.0f}s".format(
        cfg.kind, cfg.n_tokens, cfg.label[:42], cfg.d_hat, 1 - los.mean(), loop_s), flush=True)
    return "built"


def worklist(manifest: str):
    """The FULL stable list, never filtered on .done -- filtering races the array index against
    the config mapping, which silently left 79 of 479 P3-2c configs unbuilt."""
    rows = json.load(open(manifest))["configs"]
    seen, out = set(), []
    for c in configs_from_rows(rows):
        if c.filename not in seen:
            seen.add(c.filename)
            out.append(c)
    return sorted(out, key=lambda c: c.filename)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--out-dir", default="data/track_p4_4_family")
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    work = worklist(a.manifest)
    d = Path(a.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    if a.plan or a.idx is None:
        pend = sum(1 for c in work if not (d / (c.filename + ".done")).exists())
        print(json.dumps({"n_total": len(work), "n_pending": pend, "chunk": a.chunk,
                          "array_range": "0-{}".format((len(work) - 1) // a.chunk)}))
        return 0
    for c in work[a.idx * a.chunk:(a.idx + 1) * a.chunk]:
        print(build_one(c, d, force=a.force), c.filename, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
