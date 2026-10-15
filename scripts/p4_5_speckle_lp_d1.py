"""P4-5 speckle thread (D152): does the physical projection remove the VISIBLE speckle?

d5 reported accuracy before / after projecting B5 onto |k| <= 1.5 k0. This adds the d1 speckle ratio
of the projected field LP(P) against T:
  (a) on the 64x64 dense grid, 15 shapes, per band and per FDTD-amplitude quintile (200-300 Hz),
      as `p4_5_speckle.analyse` computes it for P;
  (b) on the 2 cm FDTD node grid of shape 1000 (d6), at 200 and 280 Hz -- where the speckle is
      visible and the 64x64 grid aliases it.
If LP(P)'s d1 at 2 cm falls to ~1, the visible speckle IS supra-physical content.
"""
from __future__ import annotations

import json
import math

import numpy as np

from scripts.p4_5_gate2_dense import on_lattice
from scripts.p4_5_speckle import (BANDS, CACHE, CUT, C, DF, OUT, db, grid_pairs, lowpass_all,
                                  project_lowpass, shape_geometry)


def d1_64(name):
    z = np.load(CACHE.format(name), allow_pickle=False)
    out = {"P": {b: [] for b in BANDS}, "LP": {b: [] for b in BANDS}}
    q_acc = {"P": np.zeros((5, 3)), "LP": np.zeros((5, 3))}
    i = 0
    while "P%d" % i in z.files:
        P, T, rx = z["P%d" % i][:, :601], z["T%d" % i][:, :601], z["rx%d" % i]
        keep, K = shape_geometry(i, rx)
        off = ~on_lattice(rx)
        a, b = grid_pairs(keep)
        both = off[a] & off[b]
        LP, _ = lowpass_all(P, keep, K, "ls")
        LP[:, :BANDS["25-100"][0]] = P[:, :BANDS["25-100"][0]]
        dT = np.abs(db(T[a]) - db(T[b]))[both]
        lvl = np.minimum(db(T[a]), db(T[b]))[both][:, 400:601]
        q = np.argsort(np.argsort(lvl, axis=0), axis=0) * 5 // lvl.shape[0]
        for lab, F in (("P", P), ("LP", LP)):
            dF = np.abs(db(F[a]) - db(F[b]))[both]
            rat = dF.mean(0) / dT.mean(0)
            for bnd, (lo, hi) in BANDS.items():
                out[lab][bnd].append(float(rat[lo:hi].mean()))
            for k in range(5):
                q_acc[lab][k] += [dF[:, 400:601][q == k].sum(), dT[:, 400:601][q == k].sum(),
                                  float((q == k).sum())]
        print("  {} shape {:2d} d1 200-300 P {:.3f} LP {:.3f}".format(
            name, i, out["P"]["200-300"][-1], out["LP"]["200-300"][-1]), flush=True)
        i += 1
    res = {lab: {b: float(np.mean(v)) for b, v in out[lab].items()} for lab in out}
    res["quintile_ratio_200_300"] = {lab: (q_acc[lab][:, 0] / q_acc[lab][:, 1]).tolist() for lab in q_acc}
    res["quintile_excess_step_db_200_300"] = {
        lab: ((q_acc[lab][:, 0] - q_acc[lab][:, 1]) / q_acc[lab][:, 2]).tolist() for lab in q_acc}
    return res


def d1_2cm():
    from scripts.p4_5_speckle_2cm import BINS, CACHE as C2, corr_length, pair_ratio, to_grid
    zt = np.load(C2 / "d6_fdtd_shape1000.npz"); zp = np.load(C2 / "d6_B5_shape1000.npz")
    i, j = zt["i"], zt["j"]
    keep = np.zeros((i.max() + 1, j.max() + 1), bool); keep[i, j] = True
    n = 4 * max(keep.shape)
    kx = 2 * np.pi * np.fft.fftfreq(n, 0.02)
    K = np.hypot(*np.meshgrid(kx, kx, indexing="ij"))
    order = np.lexsort((j, i))                      # keep-order (ij raster) for project_lowpass
    assert np.array_equal(np.flatnonzero(keep.ravel()), (i * keep.shape[1] + j)[order])
    res = {}
    for lab, bn in BINS.items():
        k0 = 2 * math.pi * bn * DF / C
        vals = zp["H"][order, bn][None, :]
        lp = project_lowpass(vals, keep, K, np.array([CUT * k0]), iters=1000)[0]
        lp_t = project_lowpass(zt["H"][order, bn][None, :], keep, K, np.array([CUT * k0]), iters=1000)[0]
        gT = to_grid(i[order], j[order], db(zt["H"][order, bn]))
        gP = to_grid(i[order], j[order], db(vals[0]))
        gL = to_grid(i[order], j[order], db(lp))
        gLT = to_grid(i[order], j[order], db(lp_t))
        ok = np.isfinite(gT)
        res[lab] = {"d1_2cm_P": pair_ratio(gP, gT), "d1_2cm_LP": pair_ratio(gL, gT),
                    "d1_2cm_LP_of_T": pair_ratio(gLT, gT),
                    "pearson_dB_P": float(np.corrcoef(gP[ok], gT[ok])[0, 1]),
                    "pearson_dB_LP": float(np.corrcoef(gL[ok], gT[ok])[0, 1]),
                    "lsd_P": float(np.nanmean(np.abs(gP - gT))), "lsd_LP": float(np.nanmean(np.abs(gL - gT))),
                    "hp_energy_frac_P": float(np.sum(np.abs(vals[0] - lp) ** 2) / np.sum(np.abs(vals[0]) ** 2)),
                    "hp_energy_frac_T": float(np.sum(np.abs(zt["H"][order, bn] - lp_t) ** 2)
                                              / np.sum(np.abs(zt["H"][order, bn]) ** 2)),
                    "corr_len_m_LP_dB": corr_length(gL, 0.02)}
        print(lab, json.dumps(res[lab]), flush=True)
    return res


def main():
    rep = {"grid64_B5_60K": d1_64("B5_60K"), "grid2cm_shape1000": d1_2cm()}
    (OUT / "d5_d1_after_projection.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["grid64_B5_60K"], indent=1))


if __name__ == "__main__":
    main()
