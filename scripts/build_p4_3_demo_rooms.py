"""P4-3 Task 1: FDTD ground truth for the scoped shape-edit demo.

Two unseen test shapes, four notch depths each, on a REGULAR 64x64 lattice so the demo can show
field maps as images rather than scatters. The lattice is masked to the polygon and the notch
comes back as NaN, which is the honest way to grid a non-convex room -- P4-1 and P4-2 used
scatters precisely because gridding a polygon otherwise invents values inside the removed corner.

WHY THESE TWO SHAPES. The chunk scopes the demo to d_hat >= 0.5, the regime P4-2's sweep showed
working (+0.797 at 0.58 rising to +0.906 at 0.90). Ten test shapes qualify; these two are chosen
for HIGH w_hat, because two of the ten have w_hat ~ 0.18-0.24 and literally zero NLOS receivers
-- a shape-edit demo on a notch that casts no shadow would show nothing changing but the outline.

WHY THE DEPTHS ARE SWEPT AND NOT JUST THE SHAPE'S OWN. The claim being demonstrated is that the
model responds to a shape EDIT, so the bounding box and notch width are pinned and only d moves.
Each of the eight geometries is therefore new, and none is a training shape (asserted).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

import aaf.sim.fdtd_2d as F
from aaf.data.shape_configs import (
    ALPHA, D_FRAC_MAX, GRID_DX, SRC, ShapeConfig, configs_from_rows,
)
from aaf.sim.polygon_geom import (
    assert_extents_match_polygon, line_of_sight, notch_solid_mask, polygon_contains,
)
from scripts.build_p4_2_shapes import BAND_HI_HZ, C, DX, FS, N

MANIFEST = "configs/sweeps_2d_mat/p4_2_shapes_manifest.json"
GRID_N = 64                   # 64 x 64 lattice, notch masked to NaN
WALL_CLEAR = 0.12
CORNER_CLEAR = 0.15
D_HAT_SHOWN = (0.50, 0.65, 0.80, 0.95)

# (L, W, w) of the two chosen test shapes; d is swept.
DEMO_SHAPES = (
    {"tag": "A", "L": 6.06, "W": 4.52, "w": 2.44},    # test d_hat 0.531, w_hat 0.895
    {"tag": "B", "L": 5.52, "W": 4.96, "w": 1.98},    # test d_hat 0.699, w_hat 0.797
)


def demo_configs():
    """8 configs: 2 shapes x 4 depths, all grid-snapped (D67c)."""
    out = []
    for si, sh in enumerate(DEMO_SHAPES):
        L, W, w = sh["L"], sh["W"], sh["w"]
        d_max = D_FRAC_MAX * W
        for di, dh in enumerate(D_HAT_SHOWN):
            d = round(round(dh * d_max / GRID_DX) * GRID_DX, 2)
            out.append((sh["tag"], ShapeConfig(L=L, W=W, d=float(d), w=w,
                                               split="demo", shape_id=800000 + 10 * si + di)))
    fns = [c.filename for _, c in out]
    assert len(set(fns)) == len(fns), "demo filenames collide: {}".format(fns)
    return out


def assert_not_a_training_shape(cfgs, manifest=MANIFEST):
    """A demo panel that happened to land on a training room would flatter the model silently."""
    rows = json.load(open(manifest))["configs"]
    train = {r["filename"] for r in rows if r["split"] == "train"}
    hit = [c.filename for _, c in cfgs if c.filename in train]
    if hit:
        raise AssertionError("demo geometries collide with TRAINING shapes: {}".format(hit))


def lattice(cfg):
    """A regular GRID_N x GRID_N lattice inside the bounding box, then masked to the polygon.

    Returns (pts_kept, keep_mask_flat) so the figure can scatter the kept values back onto the
    full lattice and leave the notch as NaN.
    """
    xs = np.linspace(WALL_CLEAR, cfg.L - WALL_CLEAR, GRID_N)
    ys = np.linspace(WALL_CLEAR, cfg.W - WALL_CLEAR, GRID_N)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], axis=-1)
    keep = polygon_contains(cfg.verts, P)
    # clear of the notch faces and of the reflex corner's r^(2/3) singularity
    keep &= ~((np.abs(P[:, 0] - cfg.w) < WALL_CLEAR) & (P[:, 1] > cfg.W - cfg.d - WALL_CLEAR))
    keep &= ~((np.abs(P[:, 1] - (cfg.W - cfg.d)) < WALL_CLEAR) & (P[:, 0] < cfg.w + WALL_CLEAR))
    keep &= np.hypot(P[:, 0] - cfg.w, P[:, 1] - (cfg.W - cfg.d)) > CORNER_CLEAR
    return P[keep], keep


def build_one(tag, cfg, out_dir: Path, force=False) -> str:
    out_path = out_dir / cfg.filename
    done = out_dir / (cfg.filename + ".done")
    if done.exists() and out_path.exists() and not force:
        return "skip"

    nx, dxx = F._fit_axis(cfg.L, DX, "L")
    ny, dxy = F._fit_axis(cfg.W, DX, "W")
    m = notch_solid_mask(nx, ny, dxx, dxy, cfg.W, cfg.d, cfg.w)
    extra = [{"type": "mask", "solid": m, "alpha": ALPHA}]
    geom = F.build_geometry(cfg.L, cfg.W, [ALPHA] * 4, dx=DX, extra_walls=extra)

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    want = polygon_contains(cfg.verts, np.stack([ii * dxx, jj * dxy], axis=-1))
    bad = int((geom.air != want).sum())
    if bad:
        raise AssertionError("{}: solver air mask disagrees with the polygon at {} nodes".format(
            cfg.label, bad))

    rx, keep = lattice(cfg)
    i = np.rint(rx[:, 0] / dxx).astype(int)
    j = np.rint(rx[:, 1] / dxy).astype(int)
    if not geom.air[i, j].all():
        raise RuntimeError("{}: {} lattice points snapped onto solid nodes".format(
            cfg.label, int((~geom.air[i, j]).sum())))
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
        a["lattice_keep"] = json.dumps([bool(x) for x in keep])
        a["lattice_n"] = GRID_N
        a["lattice_extent"] = json.dumps(
            [WALL_CLEAR, cfg.L - WALL_CLEAR, WALL_CLEAR, cfg.W - WALL_CLEAR])
        a["demo_tag"] = tag
        a["split"], a["kind"], a["label"] = cfg.split, cfg.kind, cfg.label
        a["alphas"] = json.dumps(list(cfg.alphas))
        a["edge_alphas"] = json.dumps(list(cfg.edge_alphas))
        a["verts"] = json.dumps([list(v) for v in cfg.verts])
        a["L"], a["W"], a["d"], a["w"] = cfg.L, cfg.W, cfg.d, cfg.w
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
    print("built {} {:30s} d_hat {:.3f} n_rx {:5d} NLOS {:5.1%} {:.0f}s".format(
        tag, cfg.label, cfg.d_hat, len(rx), 1 - los.mean(), loop_s), flush=True)
    return "built"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/track_p4_3_demo")
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    work = demo_configs()
    assert_not_a_training_shape(work)
    d = Path(a.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    if a.plan or a.idx is None:
        pend = sum(1 for _, c in work if not (d / (c.filename + ".done")).exists())
        print(json.dumps({"n_total": len(work), "n_pending": pend, "chunk": a.chunk,
                          "array_range": "0-{}".format((len(work) - 1) // a.chunk),
                          "shapes": [{"tag": t, "d_hat": round(c.d_hat, 4),
                                      "file": c.filename} for t, c in work]}, indent=1))
        return 0
    for t, c in work[a.idx * a.chunk:(a.idx + 1) * a.chunk]:
        print(build_one(t, c, d, force=a.force), c.filename, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
