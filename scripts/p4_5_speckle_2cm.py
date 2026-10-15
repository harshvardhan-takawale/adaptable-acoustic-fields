"""P4-5 speckle thread (D150) d6: shape 1000 on EVERY in-room 2 cm FDTD node, B5 @60K against FDTD.

The 64x64 dense test grid (~8-10 cm by ~6-8 cm) may alias the speckle; the 2 cm grid is the FDTD
node grid itself. Nodes: every air node of shape 1000 inside the dense grid's own region and
clearances (0.12 m from the bounding box, the notch faces and the reflex corner, exactly as
`build_p4_3_demo_rooms.lattice`), so the two grids cover the same area.

  --fdtd    CPU: re-simulate with the builder's own settings (`build_p4_5_rooms._simulate`, dx 0.02,
            same source pulse), receivers in chunks; EXACTNESS: the stored test h5's receivers are
            a subset of these nodes, and H there must reproduce the stored H (rel 1e-5).
  --render  GPU: B5 @60K at the same nodes (the dense evaluator's render loop).
  --figure  CPU: dB maps at 200 and 280 Hz (FDTD | B5 at 2 cm | B5 at the 64x64 nodes), the d1
            speckle ratio on the 2 cm grid against the 64x64 grid, and the correlation length of
            the dB error field.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path("outputs/p4_5/speckle")
CACHE = OUT / "cache"
SHAPE_ID = 1000
BINS = {"200 Hz": 400, "280 Hz": 560}
CK = "outputs/p4_5/arms/B5/ckpt_iter0060000.pt"
H5 = "data/track_p4_5_testdense/L5.54_W5.28_d1.42_w1.52.h5"
CHUNK = 6000


def shape_cfg():
    from aaf.data.shape_configs import configs_from_rows
    from scripts.build_p4_5_rooms import TEST_MANIFEST
    test = configs_from_rows(json.load(open(TEST_MANIFEST))["configs"], split="test")
    c = [c for c in test if c.shape_id == SHAPE_ID]
    assert len(c) == 1
    return c[0]


def nodes(cfg):
    """All 2 cm air nodes inside the dense lattice's region/clearances. Returns (P, i, j, geom...)."""
    from aaf.sim.polygon_geom import polygon_contains
    from scripts.build_p4_3_demo_rooms import CORNER_CLEAR, WALL_CLEAR
    from scripts.build_p4_5_rooms import _geometry
    geom, extra, _m, dxx, dxy = _geometry(cfg, 0.02)
    nx, ny = geom.air.shape
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    P = np.stack([ii.ravel() * dxx, jj.ravel() * dxy], axis=-1)
    k = geom.air.ravel().copy()
    k &= (P[:, 0] >= WALL_CLEAR - 1e-9) & (P[:, 0] <= cfg.L - WALL_CLEAR + 1e-9)
    k &= (P[:, 1] >= WALL_CLEAR - 1e-9) & (P[:, 1] <= cfg.W - WALL_CLEAR + 1e-9)
    k &= polygon_contains(cfg.verts, P)
    k &= ~((np.abs(P[:, 0] - cfg.w) < WALL_CLEAR) & (P[:, 1] > cfg.W - cfg.d - WALL_CLEAR))
    k &= ~((np.abs(P[:, 1] - (cfg.W - cfg.d)) < WALL_CLEAR) & (P[:, 0] < cfg.w + WALL_CLEAR))
    k &= np.hypot(P[:, 0] - cfg.w, P[:, 1] - (cfg.W - cfg.d)) > CORNER_CLEAR
    return P[k], ii.ravel()[k], jj.ravel()[k], geom, extra, dxx, dxy


def run_fdtd():
    import h5py
    from scripts.build_p4_5_rooms import _simulate
    cfg = shape_cfg()
    P, i, j, geom, extra, dxx, dxy = nodes(cfg)
    print("shape {} : {} nodes at 2 cm".format(SHAPE_ID, len(P)), flush=True)
    H = np.zeros((len(P), 601), np.complex64)
    for s in range(0, len(P), CHUNK):
        h, snap, _res, _fs, _n, t = _simulate(cfg, P[s:s + CHUNK], 0.02, geom, extra)
        assert np.abs(snap - P[s:s + CHUNK]).max() < 1e-9
        H[s:s + CHUNK] = h
        print("  fdtd chunk {:5d}-{:5d} {:.0f}s".format(s, s + len(h), t), flush=True)
    with h5py.File(H5) as f:
        Ts = np.asarray(f["ism/H_complex"]); rs = np.asarray(json.loads(f.attrs["receiver_pos"]))
    key = {(a, b): n for n, (a, b) in enumerate(zip(i, j))}
    idx = [key.get((int(round(x / dxx)), int(round(y / dxy)))) for x, y in rs]
    hit = np.array([n is not None for n in idx])
    sub = H[[n for n in idx if n is not None]]
    err = np.abs(sub - Ts[hit]).max() / np.abs(Ts[hit]).max()
    print("EXACTNESS: {}/{} stored receivers on the 2 cm node set; max rel |H diff| {:.2e}".format(
        hit.sum(), len(rs), err), flush=True)
    assert err < 1e-5, err
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / "d6_fdtd_shape1000.npz", P=P, i=i, j=j, H=H, exact_rel_err=err,
                        n_stored_hit=hit.sum())


def run_render():
    import torch

    from scripts.p4_5_speckle_render import load_modified
    from aaf.models.conditioning_2d import build_cond_vector_2d
    from aaf.data.shape_configs import SRC
    cfg = shape_cfg()
    P, i, j, *_ = nodes(cfg)
    dev = torch.device("cuda")
    model, renderer = load_modified(CK, dev)
    cond = build_cond_vector_2d("geom_token", cfg.L, cfg.W, cfg.edge_alphas, verts=cfg.verts, device=dev)
    room_min = torch.zeros(2, device=dev)
    room_max = torch.tensor([cfg.L, cfg.W], device=dev, dtype=torch.float32)
    R = torch.tensor(P, dtype=torch.float32, device=dev)
    S = torch.tensor(np.asarray(SRC, float), dtype=torch.float32, device=dev).unsqueeze(0)
    H = np.zeros((len(P), 601), np.complex64)
    for s0 in range(0, len(P), 8):
        r = R[s0:s0 + 8]
        with torch.no_grad():
            h = renderer(model, r, S.expand(r.shape[0], -1), room_min, room_max,
                         z_s=cond.unsqueeze(0).expand(r.shape[0], -1))
        H[s0:s0 + 8] = h[:, :601].cpu().numpy()
        if s0 % 8000 == 0:
            print("  rendered {}/{}".format(s0, len(P)), flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / "d6_B5_shape1000.npz", P=P, i=i, j=j, H=H)


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def to_grid(i, j, v):
    g = np.full((i.max() + 1, j.max() + 1), np.nan)
    g[i, j] = v
    return g


def pair_ratio(gP, gT):
    """d1 on a grid with NaN outside: mean |dB step| of P over T, 4-neighbours."""
    num, den = [], []
    for a, b in ((gP[:-1, :], gP[1:, :]), (gP[:, :-1], gP[:, 1:])):
        num.append(np.abs(a - b).ravel())
    for a, b in ((gT[:-1, :], gT[1:, :]), (gT[:, :-1], gT[:, 1:])):
        den.append(np.abs(a - b).ravel())
    n, d = np.concatenate(num), np.concatenate(den)
    ok = np.isfinite(n) & np.isfinite(d)
    return float(n[ok].mean() / d[ok].mean())


def corr_length(g, step):
    """Distance (m) at which the isotropic autocorrelation of a NaN-masked field falls to 1/e."""
    m = np.isfinite(g)
    x = np.where(m, g - np.nanmean(g), 0.0)
    n = [2 * s for s in g.shape]
    F = np.fft.fft2(x, s=n); Mk = np.fft.fft2(m.astype(float), s=n)
    ac = np.real(np.fft.ifft2(np.abs(F) ** 2)) / np.maximum(np.real(np.fft.ifft2(np.abs(Mk) ** 2)), 1)
    ac = ac / ac[0, 0]
    r = np.hypot(*np.meshgrid(np.fft.fftfreq(n[0], 1 / n[0]), np.fft.fftfreq(n[1], 1 / n[1]),
                              indexing="ij")) * step
    edges = np.arange(0, 1.5, step)
    prof = np.array([ac[(r >= e) & (r < e + step)].mean() for e in edges])
    below = np.nonzero(prof < np.exp(-1))[0]
    return float(edges[below[0]]) if len(below) else float("nan")


def run_figure():
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zt = np.load(CACHE / "d6_fdtd_shape1000.npz"); zp = np.load(CACHE / "d6_B5_shape1000.npz")
    assert np.array_equal(zt["P"], zp["P"])
    i, j, P2 = zt["i"], zt["j"], zt["P"]
    cfg = shape_cfg()
    zc = np.load("outputs/p4_5/gate2_dense/cache/B5_60K__track_p4_5_testdense.npz")
    with h5py.File(H5) as f:
        keep = np.asarray(json.loads(f.attrs["lattice_keep"]), bool).reshape(64, 64)
    from scripts.p4_5_speckle import grid_pairs
    a, b = grid_pairs(keep)
    Pd, Td = zc["P0"][:, :601], zc["T0"][:, :601]
    res = {"n_nodes_2cm": int(len(P2)), "exactness_rel_err": float(zt["exact_rel_err"]), "bins": {}}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=160, facecolor="white")
    for row, (lab, bn) in enumerate(BINS.items()):
        gT = to_grid(i, j, db(zt["H"][:, bn])); gP = to_grid(i, j, db(zp["H"][:, bn]))
        r2 = pair_ratio(gP, gT)
        dP = np.abs(db(Pd[a, bn]) - db(Pd[b, bn])); dT = np.abs(db(Td[a, bn]) - db(Td[b, bn]))
        r64 = float(dP.mean() / dT.mean())
        ok = np.isfinite(gT) & np.isfinite(gP)
        res["bins"][lab] = {
            "d1_ratio_2cm": r2, "d1_ratio_64grid": r64,
            "corr_len_m_fdtd_dB": corr_length(gT, 0.02),
            "corr_len_m_B5_dB": corr_length(gP, 0.02),
            "corr_len_m_error_dB": corr_length(np.where(ok, gP - gT, np.nan), 0.02),
            "lsd_2cm": float(np.nanmean(np.abs(gP - gT))),
            "pearson_dB_2cm": float(np.corrcoef(gP[ok], gT[ok])[0, 1])}
        lo, hi = np.nanpercentile(gT, [2, 98])
        ext = [0, (gT.shape[0] - 1) * 0.02, 0, (gT.shape[1] - 1) * 0.02]
        for col, (g, t) in enumerate(((gT, "FDTD, 2 cm"), (gP, "B5 @60K, 2 cm"))):
            ax = axes[row, col]
            im = ax.imshow(g.T, origin="lower", extent=ext, vmin=lo, vmax=hi, cmap="viridis")
            ax.set_title("{} | {}".format(t, lab), fontsize=12)
        gd = np.full(4096, np.nan); gd[keep.ravel()] = db(Pd[:, bn])
        xs = np.linspace(0.12, cfg.L - 0.12, 64); ys = np.linspace(0.12, cfg.W - 0.12, 64)
        ax = axes[row, 2]
        ax.pcolormesh(xs, ys, gd.reshape(64, 64).T, shading="nearest", vmin=lo, vmax=hi, cmap="viridis")
        ax.set_title("B5 @60K at the 64x64 test nodes | {}".format(lab), fontsize=12)
        for ax in axes[row]:
            ax.set_aspect("equal"); ax.set_xlim(0, cfg.L); ax.set_ylim(0, cfg.W)
            v = np.array(cfg.verts + [cfg.verts[0]]); ax.plot(v[:, 0], v[:, 1], "k-", lw=1)
        fig.colorbar(im, ax=axes[row].tolist(), shrink=0.85, label="dB |H|")
        axes[row, 1].text(0.02, 0.02, "d1 ratio: 2 cm {:.2f}, 64x64 {:.2f}".format(r2, r64),
                          transform=axes[row, 1].transAxes, fontsize=10, color="white")
    fig.suptitle("Shape {} (unseen), dB|H| on every 2 cm FDTD node vs the 64x64 test grid".format(
        SHAPE_ID), fontsize=14)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "d6_shape1000_2cm.png", facecolor="white")
    (OUT / "d6_shape1000_2cm.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fdtd", action="store_true")
    g.add_argument("--render", action="store_true")
    g.add_argument("--figure", action="store_true")
    a = ap.parse_args()
    run_fdtd() if a.fdtd else run_render() if a.render else run_figure()


if __name__ == "__main__":
    main()
