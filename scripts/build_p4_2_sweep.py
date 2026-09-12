"""P4-2 Stage 2: FDTD ground truth for the SHAPE-EDIT SWEEP -- the money figure's x-axis.

The headline claim a shape-conditioned model would have to support is not "it fits 75 rooms".
It is: *move one shape parameter continuously and the predicted field moves the way physics
says it does.* This builds that axis -- `L, W, w` held fixed, `d` swept across 20 values none of
which the model ever saw, passing straight through the held-out slab.

Three departures from `build_p4_2_shapes.py`, all deliberate:

  * **No sampler, no hold-out logic.** The sweep is a deterministic line through the family, not
    a draw from it. `L = 6.00, W = 5.00` is a bounding box no training or test shape occupies.
  * **Two FIXED probe receivers at indices 0 and 1**, identical in every room: the waterfall
    needs one receiver whose spectrum can be stacked against `d`, and a receiver that MOVES
    between rooms would confound the shape edit with a position change. Index 1 is chosen to be
    swallowed by the growing shadow (LOS at small `d`, NLOS at large `d`) -- that transition is
    the thing the renderer either represents or does not. Index 0 stays LOS throughout as the
    control.
  * **A denser receiver grid** (step 0.08 vs the trainer's 800-point subsample). Nothing stacks
    these across rooms, so the eval is free of the trainer's equal-count constraint, and the
    field-map strip wants the resolution.

Everything that earned its place upstream is kept verbatim: the polygon is cut with `mask`
(D65), the solver's air mask is cross-checked node-for-node against the analytic polygon per
room, and every reported wall extent is asserted against the polygon's own edges.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

import aaf.sim.fdtd_2d as F
from aaf.data.shape_configs import ALPHA, D_FRAC_MAX, GRID_DX, SRC, ShapeConfig
from aaf.sim.polygon_geom import (
    assert_extents_match_polygon, line_of_sight, notch_solid_mask, polygon_contains,
)
from scripts.build_p4_2_shapes import BAND_HI_HZ, C, DX, FS, N

# -- the line through the family ------------------------------------------------------------
SWEEP_L, SWEEP_W, SWEEP_W_NOTCH = 6.00, 5.00, 2.00   # bounding box occupied by no corpus shape
N_SWEEP = 20
RX_STEP = 0.08
RX_WALL_CLEAR = 0.12
RX_CORNER_CLEAR = 0.15

# Index 0: LOS for every d. Index 1: crosses LOS -> NLOS as the notch deepens; x > w keeps it
# inside the room at every d, so the receiver itself never moves.
PROBE_LOS = (4.00, 2.50)
PROBE_EDGE = (2.20, 4.80)


def sweep_configs():
    """20 depths from a pure rectangle to the deepest notch, all on the 0.02 m grid."""
    d_max = D_FRAC_MAX * SWEEP_W                                   # 2.25
    raw = np.linspace(0.0, d_max, N_SWEEP)
    ds = np.round(np.round(raw / GRID_DX) * GRID_DX, 2)            # grid-snapped (correction 6)
    out = []
    for i, d in enumerate(ds):
        w = 0.0 if d <= 0.0 else SWEEP_W_NOTCH                     # d = 0 IS a rectangle
        out.append(ShapeConfig(L=SWEEP_L, W=SWEEP_W, d=float(d), w=w,
                               split="sweep", shape_id=900000 + i))
    fns = [c.filename for c in out]
    assert len(set(fns)) == len(fns), "sweep filenames collide: {}".format(
        sorted({f for f in fns if fns.count(f) > 1}))
    return out


def receivers(cfg, geom, dxx, dxy):
    """The two fixed probes, then a dense interior grid. Probes stay at indices 0 and 1."""
    V = cfg.verts
    xs = np.arange(RX_WALL_CLEAR, cfg.L, RX_STEP)
    ys = np.arange(RX_WALL_CLEAR, cfg.W, RX_STEP)
    P = np.array([[x, y] for x in xs for y in ys], dtype=float)
    P = P[polygon_contains(V, P)]
    keep = np.ones(len(P), dtype=bool)
    for (x0, y0), (x1, y1) in zip(V, V[1:] + V[:1]):
        if abs(x1 - x0) < 1e-9:
            keep &= ~((np.abs(P[:, 0] - x0) < RX_WALL_CLEAR)
                      & (P[:, 1] >= min(y0, y1) - RX_WALL_CLEAR)
                      & (P[:, 1] <= max(y0, y1) + RX_WALL_CLEAR))
        else:
            keep &= ~((np.abs(P[:, 1] - y0) < RX_WALL_CLEAR)
                      & (P[:, 0] >= min(x0, x1) - RX_WALL_CLEAR)
                      & (P[:, 0] <= max(x0, x1) + RX_WALL_CLEAR))
    P = P[keep]
    if not cfg.is_rect:
        P = P[np.hypot(P[:, 0] - cfg.w, P[:, 1] - (cfg.W - cfg.d)) > RX_CORNER_CLEAR]
    probes = np.array([PROBE_LOS, PROBE_EDGE], dtype=float)
    if not polygon_contains(V, probes).all():
        raise RuntimeError("{}: a fixed probe fell outside the room".format(cfg.label))
    P = np.concatenate([probes, P], axis=0)
    i = np.rint(P[:, 0] / dxx).astype(int)
    j = np.rint(P[:, 1] / dxy).astype(int)
    if not geom.air[i[:2], j[:2]].all():
        raise RuntimeError("{}: a fixed probe snapped onto a solid node".format(cfg.label))
    return P[geom.air[i, j]]


def build_one(cfg, out_dir: Path, force: bool = False) -> str:
    out_path = out_dir / cfg.filename
    done = out_dir / (cfg.filename + ".done")
    if done.exists() and out_path.exists() and not force:
        return "skip"

    nx, dxx = F._fit_axis(cfg.L, DX, "L")
    ny, dxy = F._fit_axis(cfg.W, DX, "W")
    m = notch_solid_mask(nx, ny, dxx, dxy, cfg.W, cfg.d, cfg.w)
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
        a["L"], a["W"], a["d"], a["w"] = cfg.L, cfg.W, cfg.d, cfg.w
        a["d_hat"], a["w_hat"] = cfg.d_hat, cfg.w_hat
        a["shape_id"] = cfg.shape_id
        a["probe_los"], a["probe_edge"] = json.dumps(list(PROBE_LOS)), json.dumps(list(PROBE_EDGE))
        a["probe_edge_is_los"] = bool(los[1])
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
    print("built {:34s} d_hat {:.3f} n_rx {:5d} NLOS {:5.1%} probe1_los {} {:.0f}s".format(
        cfg.label, cfg.d_hat, len(rx), 1 - los.mean(), bool(los[1]), loop_s), flush=True)
    return "built"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/track_p4_2_sweep")
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    work = sweep_configs()
    d = Path(a.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    if a.plan or a.idx is None:
        n_pend = sum(1 for c in work if not (d / (c.filename + ".done")).exists())
        print(json.dumps({"n_total": len(work), "n_pending": n_pend, "chunk": a.chunk,
                          "array_range": "0-{}".format((len(work) - 1) // a.chunk),
                          "d_hats": [round(c.d_hat, 4) for c in work]}))
        return 0
    for c in work[a.idx * a.chunk:(a.idx + 1) * a.chunk]:
        print(build_one(c, d, force=a.force), c.filename, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
