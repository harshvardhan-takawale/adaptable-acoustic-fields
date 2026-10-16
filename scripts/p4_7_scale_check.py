"""P4-7 prep p2 (D164, pre-registered): verify scale invariance of our FDTD before using it as augmentation.

Claim: with the frequency-independent KW wall (`fdtd_2d.wall_admittance`), the 2D pulse-free transfer
function satisfies H_alpha(f; alpha x) = H(alpha f; x), i.e. h_alpha(t) = h(t / alpha) / alpha.

Rooms: 3 notched C250 training rooms chosen by rule -- the smallest max notch-snapping error (metres) on the
scaled rooms' fitted grids over alpha in {0.8, 0.9} (ties by filename). Each original room and its alpha-
scaled copies (L, W, d, w, source all x alpha) are simulated at the corpus settings (dx 0.02, fs 30720,
n 61440, wall alpha 0.15). Receivers: original air nodes (0.12 m wall / 0.15 m corner clearance, as the
corpus) whose alpha-scaled images lie within 1 mm of a scaled-grid node, ~200 per alpha by stride.

Augmentation (the manager's): h = irfft(H_deconv) of the original (pulse removed; <= 800 Hz, sampled at
30720 Hz); h_aug(t) = h(t / alpha) / alpha by cubic interpolation, ZERO beyond the 2 s record; rfft; compared
with the scaled simulation's H_deconv at the 0.5 Hz bins. Control: alpha = 1 through the same pipeline.

Per band (0-10, 10-20, 0-20, 20-100, 100-200, 200-300 Hz): relative complex L2, dB LSD, dB spatial r
(per bin across receivers, band-averaged), the best-fit complex scalar (no amplitude factor expected), and
the original record's energy fraction in its last (1 - alpha) T. PASS (D164): above 20 Hz, rel. L2 <= 2x
D94's dx floor (1.6% / 3.8% / 4.6%) and dB r >= 0.99, for all rooms x alphas.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import aaf.sim.fdtd_2d as F
from aaf.data.shape_configs import ALPHA, SRC, configs_from_rows, verts_for
from aaf.sim.polygon_geom import notch_solid_mask, polygon_contains

MANIFEST = "configs/sweeps_2d_mat/p4_4_shapes_250.json"
OUT = Path("outputs/p4_7")
OUTNAME = __import__("os").environ.get("P2_OUT", "p2_scale.json")   # diagnostics rerun -> separate file
DX, FS, N, C = 0.02, 30720.0, 61440, 343.0
ALPHAS = (0.8, 0.9)
WALL_CLEAR, CORNER_CLEAR = 0.12, 0.15
N_RX = 200
BANDS = {"0-10": (0, 20), "10-20": (20, 40), "0-20": (0, 40), "20-100": (40, 200),
         "100-200": (200, 400), "200-300": (400, 601)}
D94_FLOOR = {"20-100": 0.008, "100-200": 0.019, "200-300": 0.023}       # max rel L2, dx .02 vs .01


def scaled_grid(L, W, a):
    nx, dxx = F._fit_axis(a * L, DX, "L")
    ny, dxy = F._fit_axis(a * W, DX, "W")
    return nx, ny, dxx, dxy


def snap_err(c, a):
    _nx, _ny, dxx, dxy = scaled_grid(c.L, c.W, a)
    ex = abs(a * c.w - round(a * c.w / dxx) * dxx)
    ey = abs((a * c.W - a * c.d) - round((a * c.W - a * c.d) / dxy) * dxy)
    return max(ex, ey)


def pick_rooms():
    cs = [c for c in configs_from_rows(json.load(open(MANIFEST))["configs"], split="train") if c.d > 0 and c.w > 0]
    scored = sorted(cs, key=lambda c: (max(snap_err(c, a) for a in ALPHAS), c.filename))
    return scored[:3], {c.filename: max(snap_err(c, a) for a in ALPHAS) for c in scored[:3]}


def geometry(L, W, d, w):
    nx, dxx = F._fit_axis(L, DX, "L")
    ny, dxy = F._fit_axis(W, DX, "W")
    m = notch_solid_mask(nx, ny, dxx, dxy, W, d, w)
    extra = [{"type": "mask", "solid": m, "alpha": ALPHA}] if m.any() else None
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    want = polygon_contains(verts_for(L, W, d, w), np.stack([ii * dxx, jj * dxy], -1))
    geom = F.build_geometry(L, W, [ALPHA] * 4, dx=DX, extra_walls=extra)
    return extra, int((geom.air != want).sum()), (nx, ny, dxx, dxy), geom


def candidates(c):
    """Original-room air nodes with the corpus clearances."""
    nx, ny, dxx, dxy = scaled_grid(c.L, c.W, 1.0)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    P = np.stack([ii.ravel() * dxx, jj.ravel() * dxy], -1)
    V = verts_for(c.L, c.W, c.d, c.w)
    k = polygon_contains(V, P)
    for (x0, y0), (x1, y1) in zip(V, V[1:] + V[:1]):
        if abs(x1 - x0) < 1e-9:
            k &= ~((np.abs(P[:, 0] - x0) < WALL_CLEAR) & (P[:, 1] >= min(y0, y1) - WALL_CLEAR) & (P[:, 1] <= max(y0, y1) + WALL_CLEAR))
        else:
            k &= ~((np.abs(P[:, 1] - y0) < WALL_CLEAR) & (P[:, 0] >= min(x0, x1) - WALL_CLEAR) & (P[:, 0] <= max(x0, x1) + WALL_CLEAR))
    k &= np.hypot(P[:, 0] - c.w, P[:, 1] - (c.W - c.d)) > CORNER_CLEAR
    return P[k]


def aligned(P, c, a, tol=1e-3):
    _nx, _ny, dxx, dxy = scaled_grid(c.L, c.W, a)
    Q = a * P
    snapped = np.stack([np.rint(Q[:, 0] / dxx) * dxx, np.rint(Q[:, 1] / dxy) * dxy], -1)
    ok = np.hypot(*(Q - snapped).T) < tol
    sel = np.nonzero(ok)[0]
    sel = sel[np.linspace(0, len(sel) - 1, min(N_RX, len(sel))).astype(int)]
    return P[sel], snapped[sel], float(np.hypot(*(Q - snapped)[sel].T).max())


def stretch(h, a):
    """h_alpha(t) = h(t/alpha)/alpha on the same sample grid, zero beyond the record (cubic)."""
    from scipy.interpolate import CubicSpline
    t = np.arange(h.shape[1]) / FS
    tau = t / a
    cs = CubicSpline(t, h, axis=1, extrapolate=False)
    out = cs(tau)
    out[:, tau > t[-1]] = 0.0
    return np.nan_to_num(out) / a


def band_metrics(Hs, Ha):
    out = {}
    for b, (lo, hi) in BANDS.items():
        S, A = Hs[:, lo:hi], Ha[:, lo:hi]
        rel = float(np.linalg.norm(S - A) / np.linalg.norm(S))
        dS = 20 * np.log10(np.maximum(np.abs(S), 1e-30)); dA = 20 * np.log10(np.maximum(np.abs(A), 1e-30))
        lsd = float(np.mean(np.abs(dS - dA)))
        a_ = dS - dS.mean(0); b_ = dA - dA.mean(0)
        r = (a_ * b_).sum(0) / np.sqrt((a_ * a_).sum(0) * (b_ * b_).sum(0))
        k = complex(np.vdot(A.ravel(), S.ravel()) / np.vdot(A.ravel(), A.ravel()))
        out[b] = {"rel_l2": rel, "lsd_db": lsd, "r_db": float(np.nanmean(r)),
                  "best_scalar_abs": abs(k), "best_scalar_phase_rad": math.atan2(k.imag, k.real)}
    return out


def main():
    rooms, errs = pick_rooms()
    rep = {"pre_registration": "D164 (p2)", "settings": {"dx": DX, "fs": FS, "n": N, "wall_alpha": ALPHA},
           "rooms": [], "d94_floor_rel_l2": D94_FLOOR}
    allpass = True
    for c in rooms:
        P = candidates(c)
        sets = {a: aligned(P, c, a) for a in ALPHAS}
        rx0 = np.unique(np.concatenate([sets[a][0] for a in ALPHAS]), axis=0)
        extra0, bad0, _g0, _ = geometry(c.L, c.W, c.d, c.w)
        res0 = F.simulate(c.L, c.W, [ALPHA] * 4, SRC, rx0, dx=DX, fs=FS, n=N, c=C, extra_walls=extra0)
        h0 = np.fft.irfft(res0["H_deconv"], n=N, axis=1)
        e_tot = (h0 ** 2).sum(1)
        room = {"room": c.label, "L": c.L, "W": c.W, "d": c.d, "w": c.w, "max_notch_snap_m": errs[c.filename],
                "orig_mask_vs_polygon_bad_nodes": bad0, "alphas": {}}
        # control: alpha = 1 through the same pipeline
        Hc = np.fft.rfft(stretch(h0, 1.0), axis=1)[:, :601]
        room["control_alpha1"] = band_metrics(res0["H_deconv"][:, :601], Hc)
        for a in ALPHAS:
            Pa, Qa, rx_err = sets[a]
            idx = np.array([np.nonzero((rx0 == p).all(1))[0][0] for p in Pa])
            extra, bad, (nx, ny, dxx, dxy), _ = geometry(a * c.L, a * c.W, a * c.d, a * c.w)
            src_a = (a * SRC[0], a * SRC[1])
            res = F.simulate(a * c.L, a * c.W, [ALPHA] * 4, src_a, Qa, dx=DX, fs=FS, n=N, c=C, extra_walls=extra)
            Hs = res["H_deconv"][:, :601]
            Ha = np.fft.rfft(stretch(h0[idx], a), axis=1)[:, :601]
            tail = float(np.mean(((h0[idx][:, int(a * N):] ** 2).sum(1)) / e_tot[idx]))
            m = band_metrics(Hs, Ha)
            # DIAGNOSTICS (labelled; they do not change the pre-registered verdict):
            # (i) remove each record's static offset (mean of its last 0.25 s) before stretching
            hs = np.fft.irfft(res["H_deconv"], n=N, axis=1)
            k = int(0.25 * FS)
            c_o, c_s = h0[idx][:, -k:].mean(1, keepdims=True), hs[:, -k:].mean(1, keepdims=True)
            Hs_i = np.fft.rfft(hs - c_s, axis=1)[:, :601]
            Ha_i = np.fft.rfft(stretch(h0[idx] - c_o, a), axis=1)[:, :601]
            # (ii) matched window: the direct record also cut at alpha T
            hs_w = hs.copy(); hs_w[:, int(a * N):] = 0.0
            Hs_ii = np.fft.rfft(hs_w, axis=1)[:, :601]
            diag = {"static_offset": {"orig_c0_x_N_median": float(np.median(np.abs(c_o)) * N),
                                       "orig_H0_median": float(np.median(np.abs(np.fft.rfft(h0[idx], axis=1)[:, 0]))),
                                       "tail_energy_frac_after_offset_removal": float(np.mean(
                                           (((h0[idx] - c_o)[:, int(a * N):]) ** 2).sum(1) / ((h0[idx] - c_o) ** 2).sum(1)))},
                    "i_offset_removed": band_metrics(Hs_i, Ha_i), "ii_matched_window": band_metrics(Hs_ii, Ha)}
            ok = all(m[b]["rel_l2"] <= 2 * D94_FLOOR[b] and m[b]["r_db"] >= 0.99 for b in D94_FLOOR)
            allpass &= ok
            room["alphas"][str(a)] = {"n_rx": int(len(Pa)), "max_rx_snap_m": rx_err,
                                      "src_offset_m": np.asarray(res["meta"].get("src_offset_m", np.nan), float).tolist(),
                                      "scaled_mask_vs_polygon_bad_nodes": bad, "grid": [nx, ny, dxx, dxy],
                                      "tail_energy_frac_last_(1-a)T": tail, "bands": m, "pass": ok,
                                      "diagnostics": diag}
            print("{} a={} n_rx {} tail {:.2e} | ".format(c.label, a, len(Pa), tail) + " | ".join(
                "{} L2 {:.4f} r {:.4f} lsd {:.3f}".format(b, v["rel_l2"], v["r_db"], v["lsd_db"]) for b, v in m.items())
                + (" | PASS" if ok else " | FAIL"), flush=True)
            for dn in ("i_offset_removed", "ii_matched_window"):
                print("   diag {}: ".format(dn) + " | ".join("{} L2 {:.4f} r {:.4f}".format(b, v["rel_l2"], v["r_db"])
                      for b, v in diag[dn].items()), flush=True)
            print("   static offset: c0*N {:.1f} vs |H0| {:.1f}; tail energy after removal {:.2e}".format(
                diag["static_offset"]["orig_c0_x_N_median"], diag["static_offset"]["orig_H0_median"],
                diag["static_offset"]["tail_energy_frac_after_offset_removal"]), flush=True)
        rep["rooms"].append(room)
    rep["pass_all"] = bool(allpass)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / OUTNAME).write_text(json.dumps(rep, indent=1, default=float))
    print("PASS_ALL", allpass, flush=True)


if __name__ == "__main__":
    main()
