"""P4-4 demo v2: FDTD ground truth for the morph-figure expansion sweeps.

`build_p4_2_sweep.py` builds ONE line through the single-notch family -- fixed box, fixed notch
width, depth swept -- and that line is figN. This builds the other lines, and it has to handle
rooms with TWO reflex corners, which that script cannot: its receiver stage clears exactly one
corner and its solid mask takes scalar `(d, w)`.

WHAT IS CARRIED OVER VERBATIM, BECAUSE IT IS WHAT MAKES THE DATA TRUSTWORTHY:

  * the polygon is cut with an `extra_walls` MASK (D65), never by editing the bounding box;
  * the solver's own air mask is cross-checked NODE FOR NODE against `polygon_contains` for
    every room -- this is the check that catches an off-by-one in the mask, which otherwise
    simulates a different room that still looks entirely plausible (the P4-1 failure);
  * every reported wall extent is asserted against the polygon's own edges;
  * receivers are 0.08 m apart with 0.12 m of wall clearance, as figN's corpus is, so a v2 strip
    and figN are directly comparable.

WHAT IS GENERALISED:

  * `multi_notch.solid_blocks` cuts any legal notch set, and it is the module that documents the
    asymmetric index convention the mask depends on;
  * `multi_notch.reflex_corners` returns EVERY reflex vertex, so the r^(2/3) singularity is
    cleared at both corners of a U, a T or a double-notch. Clearing only the first would leave
    the receivers nearest the second one sitting in an unresolved field and score them as if
    they were fine.

Rooms are keyed by `FamilyConfig.filename`, which encodes every notch, and all sweeps share one
output directory on purpose: `s3a` (rect -> L -> U) and `s3b` (rect -> L -> DN) have the SAME
first four frames, so the shared rooms are simulated once and the `.done` sentinel makes the
second sweep a no-op over them.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

import aaf.sim.fdtd_2d as F
from aaf.data.multi_notch import reflex_corners, solid_blocks
from aaf.data.shape_configs import ALPHA, SRC
from aaf.sim.polygon_geom import (
    assert_extents_match_polygon, line_of_sight, polygon_contains,
)
from scripts.build_p4_2_shapes import BAND_HI_HZ, C, DX, FS, N
from scripts.build_p4_4_sweeps_common import RX_CORNER_CLEAR, RX_STEP, RX_WALL_CLEAR
from scripts.p4_4_sweeps import SWEEPS, gallery_configs, validate

OUT_DIR = "data/track_p4_4_sweeps"


def receivers(cfg, geom, dxx, dxy):
    """A dense interior grid, clear of every wall and of EVERY reflex corner."""
    V = cfg.verts
    xs = np.arange(RX_WALL_CLEAR, cfg.L, RX_STEP)
    ys = np.arange(RX_WALL_CLEAR, cfg.W, RX_STEP)
    P = np.array([[x, y] for x in xs for y in ys], dtype=float)
    P = P[polygon_contains(V, P)]
    keep = np.ones(len(P), dtype=bool)
    for (x0, y0), (x1, y1) in zip(V, V[1:] + V[:1]):
        if abs(x1 - x0) < 1e-9:                                    # vertical edge
            keep &= ~((np.abs(P[:, 0] - x0) < RX_WALL_CLEAR)
                      & (P[:, 1] >= min(y0, y1) - RX_WALL_CLEAR)
                      & (P[:, 1] <= max(y0, y1) + RX_WALL_CLEAR))
        else:                                                      # horizontal edge
            keep &= ~((np.abs(P[:, 1] - y0) < RX_WALL_CLEAR)
                      & (P[:, 0] >= min(x0, x1) - RX_WALL_CLEAR)
                      & (P[:, 0] <= max(x0, x1) + RX_WALL_CLEAR))
    P = P[keep]
    for cx, cy in reflex_corners(cfg.L, cfg.W, cfg.notches):
        P = P[np.hypot(P[:, 0] - cx, P[:, 1] - cy) > RX_CORNER_CLEAR]
    if len(P) == 0:
        raise RuntimeError("{}: no receivers survived clearance".format(cfg.label))
    i = np.rint(P[:, 0] / dxx).astype(int)
    j = np.rint(P[:, 1] / dxy).astype(int)
    return P[geom.air[i, j]]                                       # geom.air is the authority


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
        raise AssertionError("{}: solver air mask disagrees with the polygon at {} nodes".format(
            cfg.label, bad))

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
        a["L"], a["W"] = cfg.L, cfg.W
        a["notches"] = json.dumps([{"side": n.side, "d": n.d, "w": n.w, "x0": n.x0}
                                   for n in cfg.notches])
        a["n_tokens"] = int(cfg.n_tokens)
        a["d_hat"], a["w_hat"] = cfg.d_hat, cfg.w_hat
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
    print("built {:52s} {} tok  n_rx {:5d}  NLOS {:5.1%}  {:.0f}s".format(
        cfg.label, cfg.n_tokens, len(rx), 1 - los.mean(), loop_s), flush=True)
    return "built"


def work_list(names):
    """Every frame of every named sweep, DEDUPED by filename.

    s3a and s3b share their first four rooms. Simulating them twice would be ~4 wasted minutes
    and, worse, two `.h5` files that must agree byte-for-byte for the two figures to be the
    controlled comparison they claim to be.
    """
    seen, out = set(), []
    for nm in names:
        # "gallery" re-simulates the five family rooms at figN's RECEIVER density. The physics
        # is identical -- same solver, same dx, same geometry, same source -- and only the
        # receiver sampling changes (800 scattered points in the family corpus, ~3-4k on a
        # 0.08 m grid here). It writes to this sweep directory, NOT over
        # `data/track_p4_4_family/`, because the published family eval was computed on those
        # 800 receivers and overwriting them would silently invalidate it.
        frames = gallery_configs() if nm == "gallery" else validate(nm)[0]
        for _lab, c in frames:
            if c.filename in seen:
                continue
            seen.add(c.filename)
            out.append(c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", default=",".join(SWEEPS))
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    names = [s for s in a.sweeps.split(",") if s]
    work = work_list(names)
    d = Path(a.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    if a.plan or a.idx is None:
        pend = [c.filename for c in work if not (d / (c.filename + ".done")).exists()]
        print(json.dumps({"sweeps": names, "n_unique_rooms": len(work),
                          "n_pending": len(pend), "chunk": a.chunk,
                          "array_range": "0-{}".format((len(work) - 1) // a.chunk)}, indent=1))
        return 0
    for c in work[a.idx * a.chunk:(a.idx + 1) * a.chunk]:
        print(build_one(c, d, force=a.force), c.filename, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
