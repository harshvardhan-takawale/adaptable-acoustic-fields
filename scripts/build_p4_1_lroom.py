"""P4-1 Stage 1: the L-shaped room ground truth. FDTD, one scene, dense receivers.

Stage 1 asks nothing about generalization. It asks whether the architecture can fit a single
non-convex room AT ALL when handed dense data -- the cheapest way to find an architectural
blocker before the shape family is built.

Geometry: 6.0 x 5.0 with a 3.0 x 2.5 notch removed from the top-left corner. Six edges, one
reflex (270 degree) corner at (3.0, 2.5). alpha = 0.15 everywhere including the notch faces.

    (0,5)                 (6,5)
      +--------+  . . . . . +          the notch (x < 3, y > 2.5) is SOLID
      | notch  |            |
    (0,2.5)    +------------+ ...  reflex corner at (3, 2.5)
      |        (3,2.5)      |
      +---------------------+
    (0,0)                 (6,0)

WHY `mask` AND NOT `wall_segments`
----------------------------------
The chunk spec called for the `wall_segments` partition primitive. That primitive is hard-coded
to the four outer edges of the bounding RECTANGLE (`_apply_wall_segments`, fdtd_2d.py:482-536)
and never consults `solid`. Applied to this room it does not raise -- it reports
`tiles_exactly: True` and extents summing to 5.0 m for an east wall that physically exists for
only 2.5 m, and writes half the requested absorption onto inert solid nodes. That `True` would
land in the HDF5 attrs and look like a passing check. So the notch is cut with the `mask`
escape hatch instead, which sets the Kowalczyk-van Walstijn face admittance correctly on the
newly-created interfaces (fdtd_2d.py:657-671, the same four lines `_apply_slab` uses).

Since Stage 1 is uniform alpha = 0.15, one mask carries the whole boundary and the per-segment
limitation never bites. The six SEGMENTS matter only as model tokens, and those are built from
the polygon directly by `aaf.models.conditioning_2d.polygon_edge_tokens`.

THE NODE INDICES THAT MATTER
----------------------------
The removed block is the TOP-LEFT corner, so the solid slice is `m[:i3, j25+1:]`.

The reflecting plane sits at the first AIR node on each side, one dx outside the solid block
(fdtd_2d.py:427-430), so node `i3` (x = 3.0) must stay AIR -- it IS the vertical notch face --
and likewise node `j25` (y = 2.5) is the horizontal face. Get either index wrong by one and the
notch walls land at 2.98 / 2.52 m and every modal frequency is quietly wrong; get the CORNER
wrong and you simulate a different room entirely that still looks plausible.

Neither failure is left to inspection: `build` reconstructs the room analytically from `VERTS`
and asserts node-for-node against `geom.air`. That assert fired on the first run of this script
(37500 nodes = exactly twice the block, the signature of a mask in the wrong corner) and is the
reason the geometry is right.

Receivers are filtered through `geom.air` -- `_snap_nodes` raises on a solid node and there is
no in-repo helper for "is this point inside the room". They are additionally kept clear of the
reflex corner, which carries an r^(2/3) pressure singularity that no 2nd-order scheme resolves.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

import aaf.sim.fdtd_2d as F

# ------------------------------------------------------------------ frozen protocol (Track A)
C = 343.0
DX = 0.02
#: fs MUST scale with 1/dx or the anisotropic CFL check raises. Same table the other tracks use.
FS_FOR_DX = {0.05: 12288.0, 0.02: 30720.0, 0.01: 61440.0}
FS = FS_FOR_DX[DX]
N = int(2 * FS)                      # T = 2.000 s, df = 0.5 Hz
BAND_HI_HZ = 300.0

# ------------------------------------------------------------------------------ the L-room
L, W = 6.0, 5.0
NOTCH_X, NOTCH_Y = 3.0, 2.5          # removed block occupies x < NOTCH_X, y > NOTCH_Y
ALPHA = 0.15
#: Counter-clockwise, no repeated first vertex. Edge i runs VERTS[i] -> VERTS[i+1].
VERTS = [(0.0, 0.0), (L, 0.0), (L, W), (NOTCH_X, W), (NOTCH_X, NOTCH_Y), (0.0, NOTCH_Y)]

#: Source in the bottom arm. Swept over the interior during planning: ~32% NLOS is the geometric
#: ceiling for this room, and any source with >= 0.5 m wall clearance reaches 25-32%. This one
#: sits well inside the fluid (no near-field wall coupling) and still shadows a large part of
#: the upper vertical arm behind the reflex corner.
SRC = (0.8, 1.5)

RX_STEP = 0.10                       # dense interior grid
RX_WALL_CLEAR = 0.12                 # keep off the boundary layer
RX_CORNER_CLEAR = 0.15               # >= 2 nodes clear of the r^(2/3) reflex singularity


def inside_polygon(p: np.ndarray) -> np.ndarray:
    """Analytic membership for THIS room. Kept separate from `geom.air` on purpose: the two are
    cross-checked in `build`, so a mask built at the wrong index would be caught rather than
    silently producing a room of the wrong shape."""
    x, y = p[..., 0], p[..., 1]
    return (x >= 0) & (x <= L) & (y >= 0) & (y <= W) & ~((x < NOTCH_X) & (y > NOTCH_Y))


def notch_mask(nx: int, ny: int, dxx: float, dxy: float) -> np.ndarray:
    """The solid block: TOP-LEFT corner, x < NOTCH_X and y > NOTCH_Y.

    Node i3 (x = 3.0) and node j25 (y = 2.5) stay AIR -- they are the notch's two wall faces.
    See the module docstring."""
    i3 = int(round(NOTCH_X / dxx))
    j25 = int(round(NOTCH_Y / dxy))
    m = np.zeros((nx, ny), dtype=bool)
    m[:i3, j25 + 1:] = True
    return m


def line_of_sight(src: np.ndarray, rx: np.ndarray, n: int = 600) -> np.ndarray:
    """True where the straight src->rx segment stays inside the room.

    This is the Stage 1 evaluation split. Note it is a property of the GEOMETRY, not of the
    renderer -- FreqRenderer2D never traces a source->receiver ray (it fans rays out from the
    receiver and feeds the source in as a network input). The split is still the right one:
    it separates receivers the source can reach directly from those it can only reach by
    diffraction, which is what the architecture is being asked to represent.
    """
    t = np.linspace(0.0, 1.0, n)[None, :, None]
    seg = src[None, None, :] + t * (rx[:, None, :] - src[None, None, :])
    return inside_polygon(seg).all(axis=1)


def receivers(geom) -> np.ndarray:
    xs = np.arange(RX_WALL_CLEAR, L, RX_STEP)
    ys = np.arange(RX_WALL_CLEAR, W, RX_STEP)
    P = np.array([[x, y] for x in xs for y in ys], dtype=float)
    P = P[inside_polygon(P)]
    # clear of every boundary
    keep = ((P[:, 0] > RX_WALL_CLEAR) & (P[:, 0] < L - RX_WALL_CLEAR)
            & (P[:, 1] > RX_WALL_CLEAR) & (P[:, 1] < W - RX_WALL_CLEAR))
    inner = (P[:, 0] < NOTCH_X) | (P[:, 1] > NOTCH_Y)     # near the notch faces
    keep &= ~(inner & ((np.abs(P[:, 0] - NOTCH_X) < RX_WALL_CLEAR)
                       | (np.abs(P[:, 1] - NOTCH_Y) < RX_WALL_CLEAR)))
    # clear of the reflex corner
    keep &= np.hypot(P[:, 0] - NOTCH_X, P[:, 1] - NOTCH_Y) > RX_CORNER_CLEAR
    P = P[keep]
    # final authority: the solver's own air mask
    i = np.rint(P[:, 0] / geom.dx_x).astype(int)
    j = np.rint(P[:, 1] / geom.dx_y).astype(int)
    return P[geom.air[i, j]]


def build(out_dir: Path, alpha_notch: float = ALPHA, tag: str = "lroom", force: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "{}.h5".format(tag)
    done = out_dir / "{}.h5.done".format(tag)
    if done.exists() and out_path.exists() and not force:
        print("skip {} (.done present)".format(out_path))
        return "skip"

    nx, dxx = F._fit_axis(L, DX, "L")
    ny, dxy = F._fit_axis(W, DX, "W")
    m = notch_mask(nx, ny, dxx, dxy)
    extra = [{"type": "mask", "solid": m, "alpha": float(alpha_notch)}]

    geom = F.build_geometry(L, W, [ALPHA] * 4, dx=DX, extra_walls=extra)
    # Cross-check the mask against the analytic polygon: a wrong index would otherwise produce
    # a plausible-looking room of the wrong shape and every downstream number would be silently
    # about a different geometry.
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    want_air = inside_polygon(np.stack([ii * dxx, jj * dxy], axis=-1))
    disagree = int((geom.air != want_air).sum())
    if disagree:
        raise AssertionError(
            "solver air mask disagrees with the polygon at {} nodes -- check the +1 in "
            "notch_mask".format(disagree))

    rx = receivers(geom)
    los = line_of_sight(np.asarray(SRC, float), rx)
    print("[geom] {}x{} nodes, dx=({:.5f},{:.5f}) | air {} | notch solid {}".format(
        nx, ny, dxx, dxy, int(geom.air.sum()), int(m.sum())))
    print("[rx]   {} receivers | LOS {} | NLOS {} ({:.1%})".format(
        len(rx), int(los.sum()), int((~los).sum()), 1.0 - los.mean()), flush=True)

    t0 = time.time()
    res = F.simulate(L, W, [ALPHA] * 4, SRC, rx, dx=DX, fs=FS, n=N, c=C, extra_walls=extra)
    hi = int(round(BAND_HI_HZ / (FS / N))) + 1
    H = np.asarray(res["H_complex"])[:, :hi].astype(np.complex64)
    loop_s = time.time() - t0

    with h5py.File(out_path, "w") as f:
        f.create_dataset("ism/H_complex", data=H, compression="gzip", compression_opts=4)
        a = f.attrs
        a["source_pos"] = json.dumps(list(SRC))
        a["receiver_pos"] = json.dumps(np.asarray(res["meta"]["rx_pos_snapped"]).tolist())
        a["line_of_sight"] = json.dumps([bool(x) for x in los])
        a["split"] = "train"
        a["kind"] = "lroom"
        a["alphas"] = json.dumps([ALPHA] * 4)
        a["L"], a["W"] = L, W
        a["shape"] = "L"
        # `mask` reports no node indices and `meta` reports the 6x5 BOUNDING BOX, so the notch
        # is recorded here explicitly or it is not recorded anywhere.
        a["verts"] = json.dumps([list(v) for v in VERTS])
        a["notch_x"], a["notch_y"] = NOTCH_X, NOTCH_Y
        a["notch_alpha"] = float(alpha_notch)
        a["notch_solid_nodes"] = int(m.sum())
        a["n_air_nodes"] = int(geom.air.sum())
        a["solver"] = "fdtd_2d_slf_kw"
        a["dx"], a["dx_x"], a["dx_y"] = DX, dxx, dxy
        a["fs"], a["n_time_samples"] = FS, N
        a["band_hi_hz"], a["n_freq_bins"] = BAND_HI_HZ, H.shape[1]
        a["n_rx"] = int(len(rx))
        a["nlos_fraction"] = float(1.0 - los.mean())
        a["loop_s"] = round(loop_s, 1)
        a["extra_walls"] = json.dumps(res["meta"]["extra_walls"])
    done.touch()
    print("built {}  H{}  {:.0f}s".format(out_path, H.shape, loop_s), flush=True)
    return "built"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/track_p4_1_lroom")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rigid-notch", action="store_true",
                    help="build the alpha=0 twin used by the dataset gate")
    a = ap.parse_args()
    d = Path(a.out_dir)
    build(d, force=a.force)
    if a.rigid_notch:
        build(d, alpha_notch=0.0, tag="lroom_rigidnotch", force=a.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
