"""P4-5 speckle thread (D150): field-map statistics on the 15 dense test shapes, from cached renders.

  d1  speckle ratio. 4-neighbour pairs on the 64x64 dense grid, both nodes in the room. Per bin and
      shape: mean |dB step| of the prediction / the same for FDTD. PRIMARY: pairs with both nodes
      off the training lattice (Rule 11); secondary: all pairs. In 200-300 Hz, also by FDTD
      amplitude quintile: a pair's level is min(dB|T_a|, dB|T_b|) (a pair straddling a null counts as
      near-null), quintiles formed per shape per bin; each quintile's pooled ratio and its share of
      the excess sum(|dB step_P| - |dB step_T|).
  d2  fine-scale coherence. HP = f - LP(f) on in-room nodes, for prediction and FDTD; complex
      correlation of the two over off-lattice nodes, per bin, band-averaged (|c| and Re c), plus
      the HP energy ratio P/T and each field's HP energy fraction.
      LP (PRIMARY) = least-squares projection of the in-room samples onto fields band-limited to
      |k| <= 1.5 k0, by CG on the 4x zero-padded grid (`project_lowpass`). It respects the
      boundary like normalised convolution but passes |k| <= 1.5 k0 at unity, so it does not blur
      the physical band. LP (SECONDARY, the spec's literal method) = order-0 normalised convolution
      with a Gaussian applicability at half power at 1.5 k0 (`normconv_lowpass`).
  d5  physical projection: off-lattice r / LSD per band of LP(P) against T, before and after, with
      the FDTD self-projection ceiling LP(T) against T. (`--project`)
  acc off-lattice band r (column Pearson of dB per bin, band-averaged, as p4_5_hf_gap) and the
      frozen LSD, so every cache's numbers stand alone.

Every band figure is a 15-shape mean with a shape-bootstrap 95% CI (10k, seed 20261013).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

from scripts.p4_5_gate2_dense import on_lattice

CACHE = "outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz"
DENSE = Path("data/track_p4_5_testdense")
OUT = Path("outputs/p4_5/speckle")
BANDS = {"25-100": (50, 200), "100-200": (200, 400), "200-300": (400, 601)}
BANDS_0 = "25-100"
DF, C, CUT, WALL, NG, PAD = 0.5, 343.0, 1.5, 0.12, 64, 4      # PAD 4: k-step fine enough at 25 Hz
SEED, NB = 20261013, 10000
CG_ITERS = 300


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def colr(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def room_k_grid(L, W):
    """K on the PADx zero-padded grid (p4_5_kspace_diag uses 2x) and the node coordinates."""
    dx, dy = (L - 2 * WALL) / (NG - 1), (W - 2 * WALL) / (NG - 1)
    n = NG * PAD
    kx = 2 * np.pi * np.fft.fftfreq(n, dx)
    ky = 2 * np.pi * np.fft.fftfreq(n, dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return np.hypot(KX, KY), WALL + dx * np.arange(NG), WALL + dy * np.arange(NG)


def grid_pairs(keep):
    """Receiver-index pairs (a, b) of in-room 4-neighbours on the [ix, iy] grid."""
    idx = -np.ones(keep.shape, int)
    idx[keep] = np.arange(keep.sum())
    a, b = [], []
    for A, B in ((idx[:-1, :], idx[1:, :]), (idx[:, :-1], idx[:, 1:])):
        ok = (A >= 0) & (B >= 0)
        a.append(A[ok]); b.append(B[ok])
    return np.concatenate(a), np.concatenate(b)


from aaf.eval.physical_projection import _embed, _torch, project_lowpass  # noqa: E402,F401
# (moved to the library in P4-6, D160; identical code)


def normconv_lowpass(vals, keep, K, kc):
    """Order-0 normalised convolution, Gaussian applicability with |G| = 1/sqrt(2) at |k| = kc."""
    torch, dev = _torch()
    n = K.shape[0]
    G = torch.as_tensor(np.exp(-0.5 * math.log(2.0) * (K[None] / np.asarray(kc)[:, None, None]) ** 2),
                        device=dev)
    m = torch.zeros((1, n, n), dtype=torch.complex128, device=dev)
    ix, iy = np.nonzero(keep)
    m[0, ix, iy] = 1.0
    num = torch.fft.ifft2(torch.fft.fft2(_embed(vals, keep, n)) * G)
    den = torch.fft.ifft2(torch.fft.fft2(m) * G).real
    return (num[:, ix, iy] / den[:, ix, iy]).cpu().numpy()


def shape_geometry(i, rx):
    """keep [64,64] (ij) and K for test shape i, from its h5 (matched by receiver positions)."""
    for f in sorted(DENSE.glob("*.h5")):
        with h5py.File(f) as h:
            if int(h.attrs["shape_id"]) != 1000 + i:
                continue
            keep = np.asarray(json.loads(h.attrs["lattice_keep"]), bool).reshape(NG, NG)
            L, W = float(h.attrs["L"]), float(h.attrs["W"])
            assert np.allclose(np.asarray(json.loads(h.attrs["receiver_pos"])), rx)
            return keep, room_k_grid(L, W)[0]
    raise FileNotFoundError(i)


def boot(vals):
    v = np.asarray(vals, float)
    idx = np.random.default_rng(SEED).integers(0, len(v), (NB, len(v)))
    m = v[idx].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def lowpass_all(F, keep, K, method):
    bins = np.arange(BANDS["25-100"][0], 601)
    kc = CUT * 2 * np.pi * bins * DF / C
    fn = project_lowpass if method == "ls" else normconv_lowpass
    out = np.zeros_like(F, dtype=np.complex64)
    for s in range(0, len(bins), 128):
        sl = slice(s, s + 128)
        out[:, bins[sl]] = fn(F[:, bins[sl]].T, keep, K, kc[sl]).T.astype(np.complex64)
    return out, bins


def analyse(name, project, t_cache, d1_only=False):
    z = np.load(CACHE.format(name), allow_pickle=False)
    per = {"d1_off": {b: [] for b in BANDS}, "d1_all": {b: [] for b in BANDS},
           "r": {b: [] for b in BANDS}, "lsd": {b: [] for b in BANDS}}
    for meth in ("ls", "nc"):
        for k in ("c_abs", "c_re", "hp_ratio", "hp_frac_P", "hp_frac_T"):
            per["d2_%s_%s" % (meth, k)] = {b: [] for b in BANDS}
    if project:
        for k in ("r_lp", "lsd_lp", "r_ceil", "lsd_ceil", "d1_off_proj"):
            per[k] = {b: [] for b in BANDS}
    qsum = np.zeros((15, 5, 3))              # shape x quintile x (sum dP, sum dT, n pairs), 200-300 Hz
    i = 0
    while "P%d" % i in z.files:
        P, T, rx = z["P%d" % i][:, :601], z["T%d" % i][:, :601], z["rx%d" % i]
        keep, K = shape_geometry(i, rx)
        off = ~on_lattice(rx)
        a, b = grid_pairs(keep)
        dP = np.abs(db(P[a]) - db(P[b])); dT = np.abs(db(T[a]) - db(T[b]))
        both_off = off[a] & off[b]
        rat_off = dP[both_off].mean(0) / dT[both_off].mean(0)
        rat_all = dP.mean(0) / dT.mean(0)
        r = colr(db(P[off]), db(T[off]))
        for bnd, (lo, hi) in BANDS.items():
            per["d1_off"][bnd].append(float(rat_off[lo:hi].mean()))
            per["d1_all"][bnd].append(float(rat_all[lo:hi].mean()))
            per["r"][bnd].append(float(np.nanmean(r[lo:hi])))
            per["lsd"][bnd].append(float(np.mean(np.abs(db(P[off, lo:hi]) - db(T[off, lo:hi])))))
        lo, hi = BANDS["200-300"]
        lvl = np.minimum(db(T[a]), db(T[b]))[both_off][:, lo:hi]
        dPo, dTo = dP[both_off][:, lo:hi], dT[both_off][:, lo:hi]
        q = np.argsort(np.argsort(lvl, axis=0), axis=0) * 5 // lvl.shape[0]
        for k in range(5):
            qsum[i, k] = [dPo[q == k].sum(), dTo[q == k].sum(), float((q == k).sum())]
        # d2 / d5
        key = str(i)
        for meth in (() if d1_only else ("ls", "nc")):
            if "T_%s_%s" % (meth, key) in t_cache:
                LT = t_cache["T_%s_%s" % (meth, key)]
            else:
                LT, _ = lowpass_all(T, keep, K, meth)
                t_cache["T_%s_%s" % (meth, key)] = LT
            LP, _ = lowpass_all(P, keep, K, meth)
            hpP, hpT = (P - LP)[off], (T - LT)[off]
            c = (hpP * hpT.conj()).sum(0) / np.sqrt((np.abs(hpP) ** 2).sum(0) * (np.abs(hpT) ** 2).sum(0))
            eP, eT = (np.abs(hpP) ** 2).sum(0), (np.abs(hpT) ** 2).sum(0)
            fP, fT = eP / (np.abs(P[off]) ** 2).sum(0), eT / (np.abs(T[off]) ** 2).sum(0)
            for bnd, (lo2, hi2) in BANDS.items():
                per["d2_%s_c_abs" % meth][bnd].append(float(np.nanmean(np.abs(c[lo2:hi2]))))
                per["d2_%s_c_re" % meth][bnd].append(float(np.nanmean(c[lo2:hi2].real)))
                per["d2_%s_hp_ratio" % meth][bnd].append(float(eP[lo2:hi2].sum() / eT[lo2:hi2].sum()))
                per["d2_%s_hp_frac_P" % meth][bnd].append(float(np.mean(fP[lo2:hi2])))
                per["d2_%s_hp_frac_T" % meth][bnd].append(float(np.mean(fT[lo2:hi2])))
            if project and meth == "ls":
                rl = colr(db(LP[off]), db(T[off])); rc = colr(db(LT[off]), db(T[off]))
                for bnd, (lo2, hi2) in BANDS.items():
                    per["r_lp"][bnd].append(float(np.nanmean(rl[lo2:hi2])))
                    per["r_ceil"][bnd].append(float(np.nanmean(rc[lo2:hi2])))
                    per["lsd_lp"][bnd].append(
                        float(np.mean(np.abs(db(LP[off, lo2:hi2]) - db(T[off, lo2:hi2])))))
                    per["lsd_ceil"][bnd].append(
                        float(np.mean(np.abs(db(LT[off, lo2:hi2]) - db(T[off, lo2:hi2])))))
                # P4-6 standard report (D160(e)): d1 of the PROJECTED field (post-processing), same pairs
                dPp = np.abs(db(LP[a]) - db(LP[b]))
                rat_p = dPp[both_off].mean(0) / dT[both_off].mean(0)
                for bnd, (lo2, hi2) in BANDS.items():
                    per["d1_off_proj"][bnd].append(float(np.nanmean(rat_p[max(lo2, 50):hi2])))
        print("  {} shape {:2d}: d1_off 200-300 {:.3f}  r {:.3f}".format(
            name, i, per["d1_off"]["200-300"][-1], per["r"]["200-300"][-1]), flush=True)
        i += 1
    rep = {"name": name, "n_shapes": i, "per_shape": per,
           "summary": {k: {b: boot(v[b]) for b in BANDS} for k, v in per.items() if v[BANDS_0]}}
    qs = qsum[:i]
    exc = qs[:, :, 0] - qs[:, :, 1]

    def qstats(sel):
        s = qs[sel].sum(0); e = exc[sel].sum(0)
        return s[:, 0] / s[:, 1], e / e.sum()

    ratio_q, share_q = qstats(np.arange(i))
    idx = np.random.default_rng(SEED).integers(0, i, (NB, i))
    share_b = np.array([qstats(ix)[1] for ix in idx])
    rep["quintiles_200_300"] = {
        "ratio": ratio_q.tolist(), "excess_share": share_q.tolist(),
        "mean_step_db_P": (qs[:, :, 0].sum(0) / qs[:, :, 2].sum(0)).tolist(),
        "mean_step_db_T": (qs[:, :, 1].sum(0) / qs[:, :, 2].sum(0)).tolist(),
        "mean_excess_step_db": (exc.sum(0) / qs[:, :, 2].sum(0)).tolist(),
        "excess_share_ci": [np.percentile(share_b[:, k], [2.5, 97.5]).tolist() for k in range(5)],
        "gate_a3": bool(share_q[0] >= 0.40 and ratio_q[0] == ratio_q.max()),
        "note": "Q1 = lowest FDTD level (min of the pair, dB). D150(b): a3 runs iff Q1 holds >= 40% "
                "of the excess AND has the largest ratio."}
    if project:
        d = {b: boot(np.array(per["r_lp"][b]) - np.array(per["r"][b])) for b in BANDS}
        dl = {b: boot(np.array(per["lsd_lp"][b]) - np.array(per["lsd"][b])) for b in BANDS}
        rep["d5_paired"] = {"r_after_minus_before": d, "lsd_after_minus_before": dl}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True)
    ap.add_argument("--project", action="store_true")
    ap.add_argument("--d1-only", action="store_true", help="d1 + accuracy only (d3/d4 variants)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tpath = OUT / "cache" / "T_lowpass.npz"
    tpath.parent.mkdir(parents=True, exist_ok=True)
    t_cache = dict(np.load(tpath)) if tpath.exists() else {}
    n0 = len(t_cache)
    for name in a.names.replace("+", ",").split(","):  # "+": survives sbatch --export (D133)
        rep = analyse(name, a.project, t_cache, a.d1_only)
        (OUT / "{}.json".format(name)).write_text(json.dumps(rep, indent=1))
        s = rep["summary"]
        for bnd in BANDS:
            c2 = s.get("d2_ls_c_abs", {bnd: [float("nan")]})[bnd][0]
            h2 = s.get("d2_ls_hp_ratio", {bnd: [float("nan")]})[bnd][0]
            print("{:14s} {:8s} d1_off {:.3f} [{:.3f},{:.3f}]  |c|ls {:.3f}  hp P/T {:.2f}  "
                  "r {:.3f}  lsd {:.2f}".format(name, bnd, *s["d1_off"][bnd], c2, h2,
                                               s["r"][bnd][0], s["lsd"][bnd][0]), flush=True)
        qz = rep["quintiles_200_300"]
        print("   quintile ratio", np.round(qz["ratio"], 3), "excess share", np.round(qz["excess_share"], 3),
              "gate_a3", qz["gate_a3"], flush=True)
    if len(t_cache) > n0:
        np.savez_compressed(tpath, **t_cache)


if __name__ == "__main__":
    main()
