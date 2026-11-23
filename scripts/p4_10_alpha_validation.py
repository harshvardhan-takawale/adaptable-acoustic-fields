"""P4-10 A5 validation (D197): is per-edge absorption in the FDTD right?

(1) EXACT rectangle reference (second method, D54). On a rectangle with one locally-reacting admittance per wall the
    problem separates: p = X(x) Y(y), X'' = -kx^2 X with X'(0) = j k b_W X(0), X'(L) = -j k b_E X(L) (e^{j w t}, passive
    for b > 0), same for Y, and k^2 = kx^2 + ky^2. Solved by complex Newton for each (nx, ny) -- NOT a small-alpha
    expansion, so it has no grazing problem (the earlier gate's reference was first order and failed at alpha 0.70).
    The FDTD's decay rate per mode is measured from the projected mode's band-limited energy decay curve.
(2) ROBIN WALL RESIDUAL on three polygons (L, U, C3) with a different alpha on every edge: the FDTD's own wall samples
    (2 cm, one-sided stencil) must satisfy dp/dn = -j k b_edge p. Reported for the right b per edge and for the WRONG b
    (0.15 everywhere): the right one must be smaller.
  PYTHONPATH=$PWD python scripts/p4_10_alpha_validation.py --rect | --robin -> outputs/p4_10/alpha_validation.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import aaf.sim.fdtd_2d as F

OUT = Path("outputs/p4_10/alpha_validation.json")
C = 343.0


def admittance(a):
    return float(F.wall_admittance(a))


def exact_k(L, W, b, nx, ny, iters=60):
    """complex k = w/c (decay rate Im(w) = C Im(k)) of mode (nx, ny); b = (west, east, south, north) admittances."""
    bw, be, bs, bn = b

    def axis(Ln, n, b0, b1, k):
        """root kx near n pi / Ln of F = -kx sin + j k b0 cos + j k b1 cos - (k^2 b0 b1 / kx) sin, complex Newton"""
        kx = complex(n * np.pi / Ln) if n else complex(np.sqrt(abs(1j * k * (b0 + b1) / Ln)) * (1 + 0.2j))
        f = lambda q: -q * np.sin(q * Ln) + 1j * k * (b0 + b1) * np.cos(q * Ln) - (k * k * b0 * b1 / q) * np.sin(q * Ln)  # noqa: E731
        for _ in range(iters):
            h = 1e-6 * max(abs(kx), 1e-3)
            d = (f(kx + h) - f(kx - h)) / (2 * h)
            s = f(kx) / d
            if abs(s) > 0.5 * np.pi / Ln:
                s *= 0.5 * np.pi / Ln / abs(s)
            kx = kx - s
            if abs(s) < 1e-13 * max(1.0, abs(kx)):
                break
        return kx

    k = complex(np.sqrt((nx * np.pi / L) ** 2 + (ny * np.pi / W) ** 2), 1e-3)
    for _ in range(iters):
        g = lambda kk: axis(L, nx, bw, be, kk) ** 2 + axis(W, ny, bs, bn, kk) ** 2 - kk * kk  # noqa: E731
        h = 1e-6 * abs(k)
        d = (g(k + h) - g(k - h)) / (2 * h)
        s = g(k) / d
        k = k - s
        if abs(s) < 1e-12 * abs(k):
            break
    return k


def measure(L, W, alphas, dx=0.02, fs=30720.0, n=61440, nmax=40.0):
    """FDTD modal decay rates (1/s) of the low modes, through the `edges` spec on a rectangle."""
    from aaf.eval.modal_projection import enumerate_modes, mode_shape_matrix
    from scripts.p3_3_boundary_validation import _edc_slope
    V = [(0.0, 0.0), (L, 0.0), (L, W), (0.0, W)]                      # edges: south, east, north, west
    extra = [{"type": "edges", "verts": V, "alphas": [alphas[2], alphas[1], alphas[3], alphas[0]]}]
    rx = np.array([[x, y] for x in np.linspace(L / 16, L - L / 16, 8) for y in np.linspace(W / 16, W - W / 16, 8)])
    rx = np.round(rx / dx) * dx
    src = (round(0.37 * L / dx) * dx, round(0.29 * W / dx) * dx)
    res = F.simulate(L, W, alphas, src=src, rx=rx, dx=dx, fs=fs, n=n, c=C, extra_walls=extra)
    res2 = F.simulate(L, W, alphas, src=src, rx=rx, dx=dx, fs=fs, n=n, c=C)   # the established per-wall path
    same = float(np.abs(np.asarray(res["H_complex"]) - np.asarray(res2["H_complex"])).max())
    modes = [m for m in enumerate_modes(L, W, f_max=nmax * 5) if not (m.n_x == 0 and m.n_y == 0)]
    phi = mode_shape_matrix(modes, rx, L, W)
    pf = np.linalg.pinv(phi) @ np.asarray(res["H_complex"])
    out = []
    for k, m in enumerate(modes[:24]):
        e = _edc_slope(np.fft.irfft(pf[k], n=n), fs, m.f, 4.0)
        if e and e["r2"] >= 0.98:
            out.append((m.n_x, m.n_y, float(m.f), float(e["gamma"])))
    return out, same


def esprit_poles(X, fs_new, keep_hz=125.0, order=56, L=220):
    """Common complex poles of several decaying channels (stacked Hankel ESPRIT): X [n_ch, n_t] band-limited, real, noise-free
    -> (f_hz, gamma) with gamma > 0 (decay rate)."""
    nch, nt = X.shape
    K = nt - L
    H = np.concatenate([np.array([x[j:j + L] for j in range(K)]).T for x in X], axis=0)      # [n_ch * L, K]
    U = np.linalg.svd(H, full_matrices=False)[0][:, :order]
    up = np.concatenate([U[c * L:(c + 1) * L - 1] for c in range(nch)])
    dn = np.concatenate([U[c * L + 1:(c + 1) * L] for c in range(nch)])
    z = np.linalg.eigvals(np.linalg.lstsq(up, dn, rcond=None)[0])
    lam = np.log(z) * fs_new
    f, g = lam.imag / (2 * np.pi), -lam.real
    ok = (f > 5) & (f < keep_hz) & (g > 0.1) & (g < 80)
    return np.stack([f[ok], g[ok]], 1)


def measure_esprit(L, W, alphas, dx=0.02, fs=30720.0, n=61440, n_ch=16):
    """FDTD poles (f, decay rate) below ~125 Hz from the impulse responses of 16 receivers; the first 0.12 s is dropped."""
    V = [(0.0, 0.0), (L, 0.0), (L, W), (0.0, W)]
    extra = [{"type": "edges", "verts": V, "alphas": [alphas[2], alphas[1], alphas[3], alphas[0]]}]
    rng = np.random.default_rng(1)
    rx = np.round(np.stack([rng.uniform(0.2, L - 0.2, n_ch), rng.uniform(0.2, W - 0.2, n_ch)], 1) / dx) * dx
    src = (round(0.37 * L / dx) * dx, round(0.29 * W / dx) * dx)
    res = F.simulate(L, W, alphas, src=src, rx=rx, dx=dx, fs=fs, n=n, c=C, extra_walls=extra)
    x = np.asarray(res["ir"])                                     # [n_ch, n]
    X = np.fft.rfft(x, axis=1)
    fr = np.arange(X.shape[1]) * fs / n
    taper = np.clip((160.0 - fr) / 30.0, 0, 1)                    # flat to 130 Hz, zero above 160 Hz
    nt = 1024
    m = nt // 2 + 1
    y = np.fft.irfft((X * taper)[:, :m], n=nt, axis=1) * (nt / n)
    fs_new = 512.0                                                # 2 s record -> 0.5 Hz bins -> 1024 samples at 512 Hz
    y = y[:, int(0.12 * fs_new):]
    return esprit_poles(y, fs_new)


def rect_esprit():
    rows = []
    cases = [(5.0, 4.2, (0.15, 0.15, 0.15, 0.15)), (5.0, 4.2, (0.30, 0.15, 0.15, 0.15)), (5.0, 4.2, (0.60, 0.15, 0.15, 0.15)),
             (5.0, 4.2, (0.08, 0.60, 0.30, 0.15)), (6.0, 3.0, (0.50, 0.15, 0.15, 0.50)), (6.0, 3.0, (0.60, 0.60, 0.08, 0.08)),
             (4.5, 4.0, (0.70, 0.15, 0.15, 0.15))]
    for L, W, al in cases:
        poles = measure_esprit(L, W, al)
        b = [admittance(a) for a in al]
        mine = []
        for nx in range(0, 8):
            for ny in range(0, 8):
                if nx == ny == 0:
                    continue
                try:
                    k = exact_k(L, W, b, nx, ny)
                except Exception:
                    continue
                f = C * k.real / 2 / np.pi
                if 15 < f < 115:
                    mine.append((f, C * k.imag, nx, ny))
        mine.sort()
        for f, g, nx, ny in mine:
            d = np.abs(poles[:, 0] - f)
            j = int(np.argmin(np.where((poles[:, 1] > 0.3), d + 0.02 * np.abs(poles[:, 1] - g) / max(g, 1), 9e9)))
            if d[j] < 0.8:
                rows.append({"L": L, "W": W, "alphas": al, "mode": [nx, ny], "f_exact": f, "gamma_exact": g,
                             "f_fdtd": float(poles[j, 0]), "gamma_fdtd": float(poles[j, 1]),
                             "rel_gamma": float(poles[j, 1] / g - 1), "df": float(poles[j, 0] - f)})
        cur = [r for r in rows if r["alphas"] == al and r["L"] == L and r["W"] == W]
        rel = sorted(abs(r["rel_gamma"]) for r in cur)
        print(L, W, al, "matched", len(cur), "of", len(mine), "modes; |rel decay| median {:.3f} p90 {:.3f} max {:.3f}; |df| max {:.2f} Hz".format(
            np.median(rel), np.percentile(rel, 90), max(rel), max(abs(r["df"]) for r in cur)), flush=True)
    return rows


def rect():
    rows = []
    cases = [(5.0, 4.2, (0.15, 0.15, 0.15, 0.15)), (5.0, 4.2, (0.60, 0.15, 0.15, 0.15)), (5.0, 4.2, (0.08, 0.60, 0.30, 0.15)),
             (6.0, 3.0, (0.50, 0.15, 0.15, 0.50)), (6.0, 3.0, (0.60, 0.60, 0.08, 0.08)), (4.5, 4.0, (0.70, 0.15, 0.15, 0.15))]
    for L, W, al in cases:
        m, same = measure(L, W, al)
        b = [admittance(a) for a in al]
        for nx, ny, f, g in m:
            k = exact_k(L, W, b, nx, ny)
            rows.append({"L": L, "W": W, "alphas": al, "mode": [nx, ny], "f_hz": f, "f_exact": float(C * k.real / 2 / np.pi),
                         "gamma_fdtd": g, "gamma_exact": float(C * k.imag), "rel": float(g / (C * k.imag) - 1.0)})
        print(L, W, al, "modes", len(m), "edges-vs-perwall max |dH|", "{:.1e}".format(same), "worst |rel|",
              "{:.3f}".format(max(abs(r["rel"]) for r in rows if r["alphas"] == al and r["L"] == L and r["W"] == W)), flush=True)
    worst = max(abs(r["rel"]) for r in rows)
    lo = max(abs(r["rel"]) for r in rows if max(r["alphas"]) <= 0.60)
    return {"rows": rows, "worst_rel": worst, "worst_rel_alpha_le_0.60": lo, "n": len(rows)}


def robin():
    import h5py
    from scripts.p4_10_edits import base_rooms
    from aaf.sim.polygon_geom import polygon_contains
    rep = []
    names = {"L": None, "U": None, "C3": None}
    for r in base_rooms():
        if names.get(r["family"], "x") is None:
            names[r["family"]] = r
    rng = np.random.default_rng(20261123)
    for fam, r in names.items():
        V = [tuple(map(float, v)) for v in r["verts"]]
        L, W = r["L"], r["W"]
        al = [float(np.round(x, 2)) for x in rng.uniform(0.08, 0.60, len(V))]
        nx, dxx = F._fit_axis(L, 0.02, "L")
        ny, dxy = F._fit_axis(W, 0.02, "W")
        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        solid = ~polygon_contains(V, np.stack([ii * dxx, jj * dxy], -1))
        extra = ([{"type": "mask", "solid": solid, "alpha": 0.15}] if solid.any() else []) + [{"type": "edges", "verts": V, "alphas": al}]
        geom = F.build_geometry(L, W, [0.15] * 4, dx=0.02, extra_walls=extra)
        from aaf.sim.fdtd_2d import XM, XP, YM, YP
        pts, nb, eid = [], [], []                                       # air node on the wall, the two nodes inwards, its edge
        for e in range(len(V)):
            (x0, y0), (x1, y1) = V[e], V[(e + 1) % len(V)]
            d = (XP if y1 > y0 else XM) if abs(x1 - x0) < 1e-9 else (YM if x1 > x0 else YP)
            inw = {XP: (-1, 0), XM: (1, 0), YM: (0, 1), YP: (0, -1)}[d]
            a, b = sorted(((x0, y0), (x1, y1)))
            n_nodes = int(round((abs(x1 - x0) + abs(y1 - y0)) / 0.02)) + 1
            for t in np.linspace(0.15, 0.85, 12):
                i = int(round((x0 + t * (x1 - x0)) / 0.02))
                j = int(round((y0 + t * (y1 - y0)) / 0.02))
                if geom.blocked[d, i, j] and geom.air[i, j] and geom.air[i + inw[0], j + inw[1]] and geom.air[i + 2 * inw[0], j + 2 * inw[1]]:
                    pts += [[i * 0.02, j * 0.02], [(i + inw[0]) * 0.02, (j + inw[1]) * 0.02], [(i + 2 * inw[0]) * 0.02, (j + 2 * inw[1]) * 0.02]]
                    eid.append(e)
        P = np.round(np.array(pts), 6)
        res = F.simulate(L, W, [0.15] * 4, (0.5, 1.6), P, dx=0.02, fs=30720.0, n=61440, c=C, extra_walls=extra)
        H = np.asarray(res["H_complex"])[:, :601].reshape(-1, 3, 601)
        k = 2 * np.pi * np.arange(601) * 0.5 / C
        dn = -(-3 * H[:, 0] + 4 * H[:, 1] - H[:, 2]) / (2 * 0.02)

        def rr(b):
            b = np.asarray(b)[:, None] if np.ndim(b) else b
            R = dn + 1j * k[None] * b * H[:, 0]
            return (np.linalg.norm(R, axis=0) / np.maximum(np.linalg.norm(k[None] * H[:, 0], axis=0), 1e-300))
        right = rr([admittance(al[e]) for e in eid])
        wrong = rr(admittance(0.15))
        sel = slice(50, 600)
        rep.append({"family": fam, "room": r["filename"], "alphas": al, "n_points": len(eid),
                    "rel_right_band": float(np.mean(right[sel])), "rel_wrong_band": float(np.mean(wrong[sel])),
                    "ratio": float(np.mean(wrong[sel]) / np.mean(right[sel]))})
        print(fam, r["filename"][:40], "right", "{:.4f}".format(rep[-1]["rel_right_band"]), "wrong (0.15 everywhere)",
              "{:.4f}".format(rep[-1]["rel_wrong_band"]), flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rect", action="store_true")
    ap.add_argument("--robin", action="store_true")
    ap.add_argument("--esprit", action="store_true")
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    if a.rect:
        d["rect"] = rect()
        print("worst |rel|, all:", round(d["rect"]["worst_rel"], 4), "alpha <= 0.60:", round(d["rect"]["worst_rel_alpha_le_0.60"], 4))
    if a.esprit:
        rows = rect_esprit()
        d["rect_esprit"] = rows
    if a.robin:
        d["robin"] = robin()
    OUT.write_text(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()
