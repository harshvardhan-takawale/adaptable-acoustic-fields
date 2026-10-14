"""P4-5 k-space diagnostic (D107, pre-registered): is the off-lattice error SUPRA-PHYSICAL?

A pressure field at frequency f satisfies the Helmholtz equation, so its spatial spectrum is
concentrated at |k| ~ k0 = 2 pi f / c (plus the room's finite-size and corner spread). Energy far
above k0 is structure no physical field at that frequency can carry. D99 found that off the
training lattice the attention arms predict the right energy and contrast with almost no
correlation; if that error is NOISE, it lives at |k| >> k0. If it is a physical field in the
wrong place, it does not -- and then no band-limit loss can help.

Per 0.5 Hz bin, per room: the COMPLEX field (predicted and FDTD) placed on its 64x64 lattice
(`lattice_keep`), zero outside the room, zero-padded 2x, 2D FFT. S = energy at |k| > 1.5 k0 over
total. The FDTD field's own S is the mask's spectral leakage; the metric is the EXCESS. The
on/off-lattice split is done in real space: the supra-physical component R = IFFT(F x 1[|k| >
1.5 k0]), as a fraction of field energy at on-lattice vs off-lattice points.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.build_p4_5_rooms import test_shapes_for_a2
from scripts.build_p4_3_demo_rooms import GRID_N, WALL_CLEAR
from scripts.p4_5_eval import DENSE_DIR

C = 343.0
DF = 0.5
CUT = 1.5
PAD = 2
BANDS = {"25-100": (50, 200), "100-200": (200, 400), "200-300": (400, 601)}
SCORED = (200, 601)                                      # 100-300 Hz
RADIAL_EDGES = np.linspace(0, 6, 25)                    # |k| / k0 histogram edges


def _onlat(rx):
    k = (rx - 0.12) / 0.06
    return np.all(np.abs(k - np.round(k)) < 1e-6, axis=1)


def room_geometry(c):
    with h5py.File(Path(DENSE_DIR) / c.filename) as f:
        keep = np.asarray(json.loads(f.attrs["lattice_keep"]), dtype=bool)
        rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
    dx = (c.L - 2 * WALL_CLEAR) / (GRID_N - 1)
    dy = (c.W - 2 * WALL_CLEAR) / (GRID_N - 1)
    n = GRID_N * PAD
    kx = 2 * np.pi * np.fft.fftfreq(n, dx)
    ky = 2 * np.pi * np.fft.fftfreq(n, dy)
    K = np.hypot(*np.meshgrid(kx, ky, indexing="ij"))
    on = np.zeros(GRID_N * GRID_N, bool)
    on[np.flatnonzero(keep)] = _onlat(rx)
    return keep, on.reshape(GRID_N, GRID_N), K


def field_stats(H, keep, on_grid, K, b):
    """S, the radial |k|/k0 energy histogram, and the real-space supra fraction on/off lattice."""
    g = np.zeros(GRID_N * GRID_N, complex)
    g[keep] = H[:, b]
    g = g.reshape(GRID_N, GRID_N)
    n = GRID_N * PAD
    F = np.fft.fft2(g, s=(n, n))
    E = np.abs(F) ** 2
    k0 = 2 * np.pi * b * DF / C
    hi = K > CUT * k0
    S = float(E[hi].sum() / E.sum())
    hist = np.histogram(K.ravel() / k0, bins=RADIAL_EDGES, weights=E.ravel())[0] / E.sum()
    R = np.fft.ifft2(F * hi)[:GRID_N, :GRID_N]
    m = keep.reshape(GRID_N, GRID_N)
    eg, er = np.abs(g) ** 2, np.abs(R) ** 2
    on, off = m & on_grid, m & ~on_grid
    return S, hist, float(er[on].sum() / eg[on].sum()), float(er[off].sum() / eg[off].sum())


def analyse(cache):
    z = np.load(cache, allow_pickle=False)
    per = {"P": {"S": [], "hist": [], "on": [], "off": []}, "T": {"S": [], "hist": [], "on": [], "off": []}}
    for i, (_tag, c) in enumerate(test_shapes_for_a2()):
        keep, on_grid, K = room_geometry(c)
        for key, H in (("P", z["dP%d" % i]), ("T", z["dT%d" % i])):
            rows = [field_stats(H, keep, on_grid, K, b) for b in range(50, 601)]
            per[key]["S"].append([r[0] for r in rows])
            per[key]["hist"].append([r[1] for r in rows])
            per[key]["on"].append([r[2] for r in rows])
            per[key]["off"].append([r[3] for r in rows])
    out = {}
    for key in ("P", "T"):
        for q in ("S", "on", "off"):
            out[key + "_" + q] = np.mean(np.array(per[key][q]), axis=0)   # mean over rooms, per bin
        out[key + "_hist"] = np.mean(np.array(per[key]["hist"]), axis=0)
    return out


def band(x, a, b):
    return float(np.mean(x[a - 50:b - 50]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="baseline_60K,P4-4_C92_60K,P4-3_D_60K")
    ap.add_argument("--out", default="outputs/p4_5/diagnostics")
    a = ap.parse_args()
    rep = {"pre_registration": "D107", "cut": "|k| > 1.5 k0", "scored_band_hz": [100, 300], "models": {}}
    print("supra-physical fraction S (energy at |k| > 1.5 k0), complex field, 3 frozen dense shapes")
    print("{:16s} {:>8s} | {:>21s} | {:>21s} | {:>21s} | real-space supra frac (pred) on / off lattice, 100-300 Hz".format(
        "model", "", "25-100 Hz pred/GT/exc", "100-200 Hz pred/GT/exc", "200-300 Hz pred/GT/exc"))
    for name in a.names.split(","):
        r = analyse("outputs/p4_5/eval/cache/render_{}.npz".format(name))
        m = {"bands": {}}
        for bn, (lo, hi) in BANDS.items():
            p, t = band(r["P_S"], lo, hi), band(r["T_S"], lo, hi)
            m["bands"][bn] = {"S_pred": p, "S_gt": t, "excess": p - t,
                              "realspace_on": band(r["P_on"], lo, hi), "realspace_off": band(r["P_off"], lo, hi),
                              "gt_realspace_on": band(r["T_on"], lo, hi), "gt_realspace_off": band(r["T_off"], lo, hi)}
        sp = band(r["P_S"], *SCORED) - band(r["T_S"], *SCORED)
        m["scored_excess_100_300"] = sp
        m["scored_realspace_on_off"] = [band(r["P_on"], *SCORED), band(r["P_off"], *SCORED)]
        m["radial_hist_100_300"] = {"pred": r["P_hist"][150:].mean(0).tolist(),
                                    "gt": r["T_hist"][150:].mean(0).tolist(),
                                    "edges_k_over_k0": RADIAL_EDGES.tolist()}
        m["per_bin"] = {"S_pred": r["P_S"].tolist(), "S_gt": r["T_S"].tolist()}
        rep["models"][name] = m
        b = m["bands"]
        print("{:16s} {:>8s} | {:.3f}/{:.3f}/{:+.3f} | {:.3f}/{:.3f}/{:+.3f} | {:.3f}/{:.3f}/{:+.3f} | {:.3f} / {:.3f}   (GT {:.3f} / {:.3f})".format(
            name, "", *[x for bn in BANDS for x in (b[bn]["S_pred"], b[bn]["S_gt"], b[bn]["excess"])],
            *m["scored_realspace_on_off"], band(r["T_on"], *SCORED), band(r["T_off"], *SCORED)))
    if not {"baseline_60K", "P4-3_D_60K"} <= set(rep["models"]):
        # per-checkpoint use (the standing metric, D108): no D107 verdict to score
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(out / "kspace_diag.json", "w"), indent=1)
        for name, m in rep["models"].items():
            print("{}: excess supra-physical fraction 100-300 Hz {:+.3f}".format(name, m["scored_excess_100_300"]))
        return 0
    c, d = rep["models"]["baseline_60K"]["scored_excess_100_300"], rep["models"]["P4-3_D_60K"]["scored_excess_100_300"]
    holds = bool(c >= 0.10 and d <= 0.5 * c)
    rep["verdict"] = {"C250_excess": c, "D_excess": d, "criterion": "C250 >= 0.10 and D <= 0.5 x C250",
                      "holds": holds}
    print("\nD107 scored (100-300 Hz excess S): C250 {:+.3f}, Arm D {:+.3f} -> prediction {}".format(
        c, d, "HOLDS" if holds else "DOES NOT HOLD"))
    out = Path(a.out)
    json.dump(rep, open(out / "kspace_diag.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), dpi=160)
    f = np.arange(50, 601) * DF
    col = {"baseline_60K": "0.15", "P4-4_C92_60K": "0.55", "P4-3_D_60K": "#7A1F5C"}
    for name, m in rep["models"].items():
        ax[0].plot(f, np.convolve(m["per_bin"]["S_pred"], np.ones(11) / 11, "same"), color=col.get(name, "b"),
                   lw=2, label=name + " (pred)")
    gt = rep["models"]["baseline_60K"]["per_bin"]["S_gt"]
    ax[0].plot(f, np.convolve(gt, np.ones(11) / 11, "same"), color="#009E73", lw=2, ls="--",
               label="FDTD ground truth (mask leakage floor)")
    ax[0].axvspan(25, 100, color="0.93", zorder=0)
    ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("supra-physical fraction S (|k| > 1.5 k0)")
    ax[0].set_title("energy the Helmholtz equation forbids (grey: resolution-limited, not scored)",
                    loc="left", fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)
    ctr = 0.5 * (RADIAL_EDGES[1:] + RADIAL_EDGES[:-1])
    for name, m in rep["models"].items():
        ax[1].semilogy(ctr, m["radial_hist_100_300"]["pred"], color=col.get(name, "b"), lw=2, label=name)
    ax[1].semilogy(ctr, rep["models"]["baseline_60K"]["radial_hist_100_300"]["gt"], color="#009E73",
                   ls="--", lw=2, label="FDTD")
    ax[1].axvline(CUT, color="#D55E00", ls=":", lw=1.5)
    ax[1].set_xlabel("|k| / k0"); ax[1].set_ylabel("energy fraction per bin (100-300 Hz mean)")
    ax[1].set_title("radial wavenumber spectrum, 100-300 Hz (dotted: the 1.5 k0 cut)", loc="left",
                    fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25, which="both")
    fig.suptitle("D107 k-space diagnostic -- C250 excess {:+.3f}, Arm D excess {:+.3f} -> {}".format(
        c, d, "prediction HOLDS" if holds else "prediction does NOT hold"), fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "kspace_diag.png", facecolor="white")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
