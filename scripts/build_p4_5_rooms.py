"""P4-5: the three new FDTD ground-truth sets, one builder, three modes.

  rx4x       B4's training corpus: the 250 training rooms of `p4_4_shapes_250` re-simulated at
             3200 receivers. (D93: the solver records ONLY the requested receivers, so a denser
             corpus cannot be resampled from the existing files.)
  testdense  A2's dense ground truth: three FROZEN test shapes on a 64x64 lattice.
  a3         A3's convergence pair: one test shape at dx = 0.02 AND dx = 0.01.

Every mode writes to its OWN directory and never over `data/track_p4_2_shapes`, which backs
C92/C250/C500 and every published P4 evaluation. The same room filename then exists in two
directories at different receiver densities, which is exactly the D89 hazard; every consumer
must key its caches on the data directory.

THE B4 SUPERSET, AND HOW IT IS VERIFIED. `build_p4_2_shapes.receivers()` draws 800 points from
a 0.06 m candidate grid with a seeded `choice`. Re-drawing 3200 with the same seed would NOT
contain those 800, so this rebuilds the candidate pool, reproduces the original draw, ASSERTS it
equals the receiver positions stored in the original file, and appends 2400 more from the
remainder. Because the FDTD is deterministic and each receiver is recorded independently, the
new file's first 800 columns must reproduce the old `H` -- and that is asserted too, which makes
the whole re-simulation self-checking end to end.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

import aaf.sim.fdtd_2d as F
from aaf.data.shape_configs import ALPHA, D_HAT_HOLDOUT, SRC, configs_from_rows
from aaf.sim.polygon_geom import (
    assert_extents_match_polygon, line_of_sight, notch_solid_mask, polygon_contains,
)
from scripts.build_p4_2_shapes import (
    BAND_HI_HZ, C, FS_FOR_DX, N_RX, RX_CORNER_CLEAR, RX_SEED, RX_WALL_CLEAR,
)
from scripts.build_p4_3_demo_rooms import GRID_N, lattice

TRAIN_MANIFEST = "configs/sweeps_2d_mat/p4_4_shapes_250.json"
TEST_MANIFEST = "configs/sweeps_2d_mat/p4_2_shapes_manifest.json"
SRC_DIR = "data/track_p4_2_shapes"
OUT = {"rx4x": "data/track_p4_5_rx4x", "testdense": "data/track_p4_5_testdense",
       # all 15 frozen test shapes densely, for the D99 Gate-2 re-score; same directory as
       # `testdense`, so A2's three rooms are reused (their .done sentinels skip them)
       "testdense15": "data/track_p4_5_testdense", "a3": "data/track_p4_5_a3",
       # B6 (D99, Q22): the 250 training rooms with receivers OFF the shared lattice
       "offlat": "data/track_p4_5_offlat",
       # manager item 3: one dense held-out room per family, to BOUND the family eval (D102(d))
       "famdense": "data/track_p4_5_famdense",
       # manager (D112): the SECOND room per family -- the other median room; same directory,
       # distinct filenames, so the first five are untouched
       "famdense2": "data/track_p4_5_famdense"}
FAMILY_MANIFEST = "configs/sweeps_2d_mat/p4_4_family_manifest.json"

N_RX_4X = 4 * N_RX                       # 3200 -- identical in every room: the trainer stacks
RX_SEED_4X = 20261013                    # seeds ONLY the 2400 appended receivers
RX_SEED_OFFLAT = 20261014                # B6: per-room draw over EVERY FDTD node
A3_STEP = 0.08                           # node-aligned at BOTH dx = 0.02 and dx = 0.01
H_REPRO_TOL = 1e-5                       # relative; FFT batching may reorder a few float ops


# --------------------------------------------------------------------------- shape selection
def test_shapes_for_a2():
    """The three frozen test shapes A2 evaluates densely. PRE-REGISTERED RULE, not score-based:
    the minimum-d_hat shape, the in-slab shape nearest the slab centre d_hat = 0.525, and the
    maximum-d_hat shape. Ties broken by lexicographic filename (two in-slab shapes share
    d_hat = 0.531 -- same d and W, different notch width)."""
    rows = json.load(open(TEST_MANIFEST))["configs"]
    t = sorted(configs_from_rows(rows, split="test"), key=lambda c: (c.d_hat, c.filename))
    lo, hi = D_HAT_HOLDOUT
    ins = [c for c in t if lo - 1e-9 <= c.d_hat <= hi + 1e-9]
    mid = min(ins, key=lambda c: (abs(c.d_hat - 0.525), c.filename))
    return [("shallow", t[0]), ("inslab", mid), ("deep", t[-1])]


def a3_shape():
    """A3 uses A2's in-slab shape, so the GT-validity limit applies directly to A2's curves."""
    return [s for s in test_shapes_for_a2() if s[0] == "inslab"][0][1]


# --------------------------------------------------------------------------- geometry
def _geometry(cfg, dx):
    nx, dxx = F._fit_axis(cfg.L, dx, "L")
    ny, dxy = F._fit_axis(cfg.W, dx, "W")
    m = notch_solid_mask(nx, ny, dxx, dxy, cfg.W, cfg.d, cfg.w)
    extra = [{"type": "mask", "solid": m, "alpha": ALPHA}] if m.any() else None
    geom = F.build_geometry(cfg.L, cfg.W, [ALPHA] * 4, dx=dx, extra_walls=extra)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    want = polygon_contains(cfg.verts, np.stack([ii * dxx, jj * dxy], axis=-1))
    bad = int((geom.air != want).sum())
    if bad:
        raise AssertionError("{} @ dx={}: solver air mask disagrees with the polygon at {} "
                             "nodes".format(cfg.label, dx, bad))
    return geom, extra, m, dxx, dxy


def _wall_clear(cfg, P):
    """`build_p4_2_shapes.receivers`' clearance, verbatim: every wall, then the reflex corner."""
    V = cfg.verts
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
    return P


def _air_only(P, geom, dxx, dxy):
    i = np.rint(P[:, 0] / dxx).astype(int)
    j = np.rint(P[:, 1] / dxy).astype(int)
    return P[geom.air[i, j]]


def _grid(cfg, step):
    """The candidate grid, built EXACTLY as `build_p4_2_shapes.receivers` builds it.

    DO NOT ROUND. An earlier version rounded to 1e-9 "for tidiness" and 37 of 250 rooms then
    failed to reproduce their stored receivers by 4-5 m: a point sitting exactly 0.12 m from a
    wall carries float error either side of the `< RX_WALL_CLEAR` test, rounding moved it across,
    the pool size changed, and the seeded `choice` mapped onto entirely different points. Left
    unguarded that would have built a B4 corpus whose first 800 receivers were NOT the
    baseline's in 15% of rooms, quietly breaking the superset property the arm depends on.
    Unrounded, all 250 reproduce exactly. Node alignment for A3 does not need rounding either:
    the solver snaps with `rint`, and the snap error is asserted.
    """
    xs = np.arange(RX_WALL_CLEAR, cfg.L, step)
    ys = np.arange(RX_WALL_CLEAR, cfg.W, step)
    return np.array([[x, y] for x in xs for y in ys], dtype=float)


def rx_4x(cfg, geom, dxx, dxy, stored_rx):
    """The original 800, reproduced and VERIFIED, then 2400 appended from the remainder."""
    P = _air_only(_wall_clear(cfg, _grid(cfg, 0.06)), geom, dxx, dxy)
    rng = np.random.default_rng([RX_SEED, cfg.shape_id])
    idx800 = np.sort(rng.choice(len(P), size=N_RX, replace=False))
    orig = P[idx800]
    err = float(np.abs(orig - stored_rx).max())
    if orig.shape != stored_rx.shape or err > 1e-6:
        raise AssertionError("{}: reproduced 800 receivers differ from the stored ones by {:.3g} m "
                             "-- the candidate pool is not the original".format(cfg.label, err))
    rest = np.setdiff1d(np.arange(len(P)), idx800)
    need = N_RX_4X - N_RX
    if len(rest) < need:
        raise RuntimeError("{}: only {} spare candidates, need {}".format(cfg.label, len(rest), need))
    add = np.sort(np.random.default_rng([RX_SEED_4X, cfg.shape_id]).choice(
        len(rest), size=need, replace=False))
    return np.concatenate([orig, P[rest[add]]], axis=0), len(P)


# --------------------------------------------------------------------------- one simulation
def _simulate(cfg, rx, dx, geom, extra):
    fs = FS_FOR_DX[dx]
    n = int(2 * fs)                                  # T = 2 s, df = 0.5 Hz at every dx
    t0 = time.time()
    res = F.simulate(cfg.L, cfg.W, [ALPHA] * 4, SRC, rx, dx=dx, fs=fs, n=n, c=C,
                     extra_walls=extra)
    assert_extents_match_polygon(res["meta"].get("extra_walls", []), cfg.verts)
    hi = int(round(BAND_HI_HZ / (fs / n))) + 1
    H = np.asarray(res["H_complex"])[:, :hi].astype(np.complex64)
    return H, np.asarray(res["meta"]["rx_pos_snapped"]), res, fs, n, time.time() - t0


def _write(path, cfg, H, rx_snap, los, res, dx, dxx, dxy, fs, n, m, geom, loop_s, extra_attrs):
    with h5py.File(path, "w") as f:
        f.create_dataset("ism/H_complex", data=H, compression="gzip", compression_opts=4)
        a = f.attrs
        a["source_pos"] = json.dumps(list(SRC))
        a["receiver_pos"] = json.dumps(rx_snap.tolist())
        a["line_of_sight"] = json.dumps([bool(x) for x in los])
        a["split"], a["kind"], a["label"] = cfg.split, cfg.kind, cfg.label
        a["alphas"] = json.dumps(list(cfg.alphas))
        a["edge_alphas"] = json.dumps(list(cfg.edge_alphas))
        a["verts"] = json.dumps([list(v) for v in cfg.verts])
        a["L"], a["W"], a["d"], a["w"] = cfg.L, cfg.W, cfg.d, cfg.w
        a["d_hat"], a["w_hat"] = cfg.d_hat, cfg.w_hat
        a["shape_id"] = cfg.shape_id
        a["notch_solid_nodes"] = int(m.sum()) if m is not None else 0
        a["n_air_nodes"] = int(geom.air.sum())
        a["nlos_fraction"] = float(1.0 - los.mean())
        a["solver"] = "fdtd_2d_slf_kw"
        a["dx"], a["dx_x"], a["dx_y"] = dx, dxx, dxy
        a["fs"], a["n_time_samples"] = fs, n
        a["band_hi_hz"], a["n_freq_bins"], a["n_rx"] = BAND_HI_HZ, H.shape[1], int(len(rx_snap))
        a["loop_s"] = round(loop_s, 1)
        a["extra_walls"] = json.dumps(res["meta"].get("extra_walls", []))
        for k, v in extra_attrs.items():
            a[k] = v


# --------------------------------------------------------------------------- the three modes
def build_rx4x(cfg, out: Path, force=False):
    path, done = out / cfg.filename, out / (cfg.filename + ".done")
    if done.exists() and path.exists() and not force:
        return "skip"
    dx = 0.02
    geom, extra, m, dxx, dxy = _geometry(cfg, dx)
    with h5py.File(Path(SRC_DIR) / cfg.filename) as f:
        H_old = np.asarray(f["ism/H_complex"])
        rx_old = np.asarray(json.loads(f.attrs["receiver_pos"]))
    rx, pool = rx_4x(cfg, geom, dxx, dxy, rx_old)
    H, rx_snap, res, fs, n, loop_s = _simulate(cfg, rx, dx, geom, extra)
    if H.shape[0] != N_RX_4X:
        raise AssertionError("{}: {} receivers, need exactly {}".format(cfg.label, H.shape[0],
                                                                       N_RX_4X))
    # THE END-TO-END CHECK: the first 800 columns must reproduce the original file's H.
    rel = float(np.abs(H[:N_RX] - H_old).max() / max(np.abs(H_old).max(), 1e-30))
    if rel > H_REPRO_TOL:
        raise AssertionError("{}: first 800 receivers reproduce the original H only to {:.3g} "
                             "(tolerance {:.0e})".format(cfg.label, rel, H_REPRO_TOL))
    los = line_of_sight(cfg.verts, SRC, rx_snap)
    _write(path, cfg, H, rx_snap, los, res, dx, dxx, dxy, fs, n, m, geom, loop_s, {
        "rx_construction": ("indices 0-799 = {} verbatim (seed [RX_SEED, shape_id]); 800-3199 "
                            "appended from the remaining 0.06 m candidates, seed [{}, shape_id]"
                            .format(SRC_DIR, RX_SEED_4X)),
        "rx_candidate_pool": int(pool), "h_first800_rel_maxdiff": rel})
    done.touch()
    print("built {:34s} n_rx {} pool {:5d} first800 rel.diff {:.2e} {:.0f}s".format(
        cfg.label, N_RX_4X, pool, rel, loop_s), flush=True)
    return "built"


def rx_offlat(cfg, geom, dxx, dxy):
    """B6: 800 receivers drawn per room, uniformly, from EVERY in-room 0.02 m FDTD node.

    D99 found the model memorises the absolute receiver lattice (0.12 + 0.06 k) because every P4
    room puts its receivers there. Drawing from all nodes -- with the same 0.12 m wall and
    0.15 m reflex-corner clearance as the corpus builder -- leaves ~1 in 9 receivers on the old
    lattice, its natural rate, so no absolute position is privileged: a given point is a
    receiver in ~3 training rooms instead of ~30. The count stays at 800, so the trainer's
    stacking, its validation set and its per-iteration cost are unchanged."""
    P = _air_only(_wall_clear(cfg, _grid(cfg, 0.02)), geom, dxx, dxy)
    if len(P) < N_RX:
        raise RuntimeError("{}: only {} node candidates".format(cfg.label, len(P)))
    rng = np.random.default_rng([RX_SEED_OFFLAT, cfg.shape_id])
    return P[np.sort(rng.choice(len(P), size=N_RX, replace=False))], len(P)


def build_offlat(cfg, out: Path, force=False):
    path, done = out / cfg.filename, out / (cfg.filename + ".done")
    if done.exists() and path.exists() and not force:
        return "skip"
    dx = 0.02
    geom, extra, m, dxx, dxy = _geometry(cfg, dx)
    rx, pool = rx_offlat(cfg, geom, dxx, dxy)
    H, rx_snap, res, fs, n, loop_s = _simulate(cfg, rx, dx, geom, extra)
    if H.shape[0] != N_RX:
        raise AssertionError("{}: {} receivers, need exactly {}".format(cfg.label, H.shape[0], N_RX))
    k = (rx_snap - 0.12) / 0.06
    on = float(np.all(np.abs(k - np.round(k)) < 1e-6, axis=1).mean())
    los = line_of_sight(cfg.verts, SRC, rx_snap)
    _write(path, cfg, H, rx_snap, los, res, dx, dxx, dxy, fs, n, m, geom, loop_s, {
        "rx_construction": "800 drawn uniformly from every in-room 0.02 m node, seed [{}, "
                           "shape_id] (B6, D99)".format(RX_SEED_OFFLAT),
        "rx_candidate_pool": int(pool), "frac_on_old_lattice": on})
    done.touch()
    print("built {:34s} pool {:5d}  on-old-lattice {:.3f} {:.0f}s".format(
        cfg.label, pool, on, loop_s), flush=True)
    return "built"


def family_test_rooms(second: bool = False):
    """One held-out room per family, PRE-REGISTERED, not score-based: the median-d_hat test room
    of each family (rectangles: the median-AREA test room). Ties by filename.

    ``second=True`` gives the OTHER median room: each family has 10 test rooms, so the sorted list
    has two medians, indices 4 (the first pick) and 5. Fixed in D112 before any score."""
    from aaf.data.multi_notch import configs_from_rows as fam_rows
    rows = json.load(open(FAMILY_MANIFEST))["configs"]
    out = []
    for fam in ("rect", "L", "T", "DN", "U"):
        cs = [c for c in fam_rows(rows, split="test") if c.kind == fam]
        key = (lambda c: (c.L * c.W, c.filename)) if fam == "rect" else (lambda c: (c.d_hat, c.filename))
        cs = sorted(cs, key=key)
        out.append((fam, cs[(len(cs) - 1) // 2 + (1 if second else 0)]))
    return out


def build_famdense(tag, cfg, out: Path, force=False):
    """A family room on a 64x64 lattice, multi-notch geometry (`solid_blocks`, every reflex
    corner cleared), with the node-for-node polygon assert. Same lattice construction as A2's."""
    from aaf.data.multi_notch import reflex_corners, solid_blocks
    from scripts.build_p4_3_demo_rooms import CORNER_CLEAR, WALL_CLEAR
    path, done = out / cfg.filename, out / (cfg.filename + ".done")
    if done.exists() and path.exists() and not force:
        return "skip"
    dx = 0.02
    nx, dxx = F._fit_axis(cfg.L, dx, "L")
    ny, dxy = F._fit_axis(cfg.W, dx, "W")
    m = solid_blocks(nx, ny, dxx, dxy, cfg.L, cfg.W, cfg.notches)
    extra = [{"type": "mask", "solid": m, "alpha": ALPHA}] if m.any() else None
    geom = F.build_geometry(cfg.L, cfg.W, [ALPHA] * 4, dx=dx, extra_walls=extra)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    want = polygon_contains(cfg.verts, np.stack([ii * dxx, jj * dxy], axis=-1))
    bad = int((geom.air != want).sum())
    if bad:
        raise AssertionError("{}: air mask disagrees with polygon at {} nodes".format(cfg.label, bad))
    xs = np.linspace(WALL_CLEAR, cfg.L - WALL_CLEAR, GRID_N)
    ys = np.linspace(WALL_CLEAR, cfg.W - WALL_CLEAR, GRID_N)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], axis=-1)
    keep = polygon_contains(cfg.verts, P)
    V = cfg.verts
    for (x0, y0), (x1, y1) in zip(V, V[1:] + V[:1]):                # every wall, notch faces too
        if abs(x1 - x0) < 1e-9:
            keep &= ~((np.abs(P[:, 0] - x0) < WALL_CLEAR) & (P[:, 1] >= min(y0, y1) - WALL_CLEAR)
                      & (P[:, 1] <= max(y0, y1) + WALL_CLEAR))
        else:
            keep &= ~((np.abs(P[:, 1] - y0) < WALL_CLEAR) & (P[:, 0] >= min(x0, x1) - WALL_CLEAR)
                      & (P[:, 0] <= max(x0, x1) + WALL_CLEAR))
    for cx, cy in reflex_corners(cfg.L, cfg.W, cfg.notches):
        keep &= np.hypot(P[:, 0] - cx, P[:, 1] - cy) > CORNER_CLEAR
    rx = P[keep]
    i, j = np.rint(rx[:, 0] / dxx).astype(int), np.rint(rx[:, 1] / dxy).astype(int)
    if not geom.air[i, j].all():
        raise RuntimeError("{}: lattice points on solid".format(cfg.label))
    H, rx_snap, res, fs, n, loop_s = _simulate(cfg, rx, dx, geom, extra)
    los = line_of_sight(cfg.verts, SRC, rx_snap)
    with h5py.File(path, "w") as f:
        f.create_dataset("ism/H_complex", data=H, compression="gzip", compression_opts=4)
        a = f.attrs
        a["source_pos"] = json.dumps(list(SRC))
        a["receiver_pos"] = json.dumps(rx_snap.tolist())
        a["line_of_sight"] = json.dumps([bool(x) for x in los])
        a["lattice_keep"] = json.dumps([bool(x) for x in keep])
        a["label"], a["kind"], a["family_tag"] = cfg.label, cfg.kind, tag
        a["verts"] = json.dumps([list(v) for v in cfg.verts])
        a["L"], a["W"], a["d_hat"] = cfg.L, cfg.W, cfg.d_hat
        a["n_rx"], a["loop_s"], a["dx"] = int(len(rx)), round(loop_s, 1), dx
    done.touch()
    print("built famdense {:5s} {:52s} n_rx {:5d} {:.0f}s".format(tag, cfg.label, len(rx), loop_s),
          flush=True)
    return "built"


def build_testdense(tag, cfg, out: Path, force=False):
    path, done = out / cfg.filename, out / (cfg.filename + ".done")
    if done.exists() and path.exists() and not force:
        return "skip"
    dx = 0.02
    geom, extra, m, dxx, dxy = _geometry(cfg, dx)
    rx, keep = lattice(cfg)
    i, j = np.rint(rx[:, 0] / dxx).astype(int), np.rint(rx[:, 1] / dxy).astype(int)
    if not geom.air[i, j].all():
        raise RuntimeError("{}: {} lattice points snapped onto solid".format(
            cfg.label, int((~geom.air[i, j]).sum())))
    H, rx_snap, res, fs, n, loop_s = _simulate(cfg, rx, dx, geom, extra)
    los = line_of_sight(cfg.verts, SRC, rx_snap)
    _write(path, cfg, H, rx_snap, los, res, dx, dxx, dxy, fs, n, m, geom, loop_s, {
        "lattice_keep": json.dumps([bool(x) for x in keep]), "lattice_n": GRID_N,
        "a2_tag": tag, "frozen_test_shape": True})
    done.touch()
    print("built {:8s} {:30s} d_hat {:.3f} n_rx {:5d} {:.0f}s".format(
        tag, cfg.label, cfg.d_hat, len(rx), loop_s), flush=True)
    return "built"


def build_a3(dx, out: Path, force=False):
    cfg = a3_shape()
    name = "{}_dx{:.2f}.h5".format(cfg.label, dx)
    path, done = out / name, out / (name + ".done")
    if done.exists() and path.exists() and not force:
        return "skip"
    geom, extra, m, dxx, dxy = _geometry(cfg, dx)
    rx = _air_only(_wall_clear(cfg, _grid(cfg, A3_STEP)), geom, dxx, dxy)
    H, rx_snap, res, fs, n, loop_s = _simulate(cfg, rx, dx, geom, extra)
    # Node alignment is the whole point of the 0.08 m grid: snapping must be the identity.
    snap_err = float(np.abs(rx_snap - rx).max())
    if snap_err > 1e-6:
        raise AssertionError("{} @ dx={}: receivers moved {:.3g} m when snapped -- not "
                             "node-aligned".format(cfg.label, dx, snap_err))
    los = line_of_sight(cfg.verts, SRC, rx_snap)
    _write(path, cfg, H, rx_snap, los, res, dx, dxx, dxy, fs, n, m, geom, loop_s, {
        "a3_step": A3_STEP, "snap_err_m": snap_err})
    done.touch()
    print("built a3 {} dx={} n_rx {} fs {:.0f} n {} {:.0f}s".format(
        cfg.label, dx, len(rx), fs, n, loop_s), flush=True)
    return "built"


def worklist(mode):
    if mode in ("rx4x", "offlat"):
        rows = json.load(open(TRAIN_MANIFEST))["configs"]
        return sorted(configs_from_rows(rows, split="train"), key=lambda c: c.filename)
    if mode == "testdense":
        return test_shapes_for_a2()
    if mode == "famdense":
        return family_test_rooms()
    if mode == "famdense2":
        return family_test_rooms(second=True)
    if mode == "testdense15":
        rows = json.load(open(TEST_MANIFEST))["configs"]
        return [(c.filename[:-3], c) for c in sorted(configs_from_rows(rows, split="test"),
                                                     key=lambda c: c.filename)]
    return [0.02, 0.01]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=sorted(OUT))
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(OUT[a.mode])
    out.mkdir(parents=True, exist_ok=True)
    work = worklist(a.mode)
    if a.plan or a.idx is None:
        desc = ([{"tag": t, "file": c.filename, "d_hat": round(c.d_hat, 4)} for t, c in work]
                if a.mode in ("testdense", "testdense15", "famdense", "famdense2") else len(work))
        print(json.dumps({"mode": a.mode, "out": str(out), "n_total": len(work),
                          "array_range": "0-{}".format((len(work) - 1) // a.chunk),
                          "work": desc}, indent=1))
        return 0
    for w in work[a.idx * a.chunk:(a.idx + 1) * a.chunk]:
        if a.mode == "rx4x":
            print(build_rx4x(w, out, a.force), w.filename, flush=True)
        elif a.mode in ("famdense", "famdense2"):
            print(build_famdense(w[0], w[1], out, a.force), w[1].filename, flush=True)
        elif a.mode == "offlat":
            print(build_offlat(w, out, a.force), w.filename, flush=True)
        elif a.mode in ("testdense", "testdense15"):
            print(build_testdense(w[0], w[1], out, a.force), w[1].filename, flush=True)
        else:
            print(build_a3(w, out, a.force), "dx", w, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
