"""P4-5 A3: is the FDTD ground truth trustworthy where A2 is about to evaluate?

One frozen test room (A2's in-slab shape) solved at dx = 0.02 -- the resolution of every corpus
-- and at dx = 0.01, with IDENTICAL receivers on a 0.08 m grid that lands on nodes at both
resolutions (asserted at build time; a 64x64 linspace lattice would snap differently on the two
grids and fold a position error into the comparison). Both runs use T = 2 s, so the 0.5 Hz bins
coincide exactly and every bin can be compared directly.

The criterion is D94's, fixed before this ran: ground truth is VALID at f where the dB spatial
Pearson between the two solutions, averaged over a 5 Hz window, is >= 0.99. The GT-validity
limit is the first frequency, scanning up from 25 Hz, where that fails. It is the vertical line
on the headline figure.

The dx = 0.01 solution is itself only an approximation, so this bounds the dx = 0.02 error
rather than measuring it exactly: for a second-order scheme, Richardson extrapolation puts the
dx = 0.02 error at ~(4/3) x |H(0.02) - H(0.01)|.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

DF = 0.5
F_LO = 25.0
WIN_HZ = 5.0
R_VALID = 0.99


def _db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def _pearson(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _window(x, n):
    """Centred moving mean over n bins, NaN-aware, same length as x."""
    out = np.full_like(x, np.nan, dtype=float)
    h = n // 2
    for i in range(len(x)):
        seg = x[max(0, i - h):i + h + 1]
        seg = seg[np.isfinite(seg)]
        out[i] = seg.mean() if len(seg) else np.nan
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/track_p4_5_a3")
    ap.add_argument("--out", default="outputs/p4_5/diagnostics")
    a = ap.parse_args()
    files = sorted(Path(a.dir).glob("*_dx*.h5"))
    by = {}
    for f in files:
        with h5py.File(f) as h:
            by[round(float(h.attrs["dx"]), 3)] = {
                "H": np.asarray(h["ism/H_complex"]), "label": str(h.attrs["label"]),
                "rx": np.asarray(json.loads(h.attrs["receiver_pos"])),
                "fs": float(h.attrs["fs"]), "n": int(h.attrs["n_time_samples"]),
                "loop_s": float(h.attrs["loop_s"])}
    c, f_ = by[0.02], by[0.01]
    if c["rx"].shape != f_["rx"].shape or np.abs(c["rx"] - f_["rx"]).max() > 1e-6:
        raise AssertionError("receivers differ between the two resolutions")
    if c["H"].shape != f_["H"].shape:
        raise AssertionError("bin grids differ: {} vs {}".format(c["H"].shape, f_["H"].shape))
    Hc, Hf = c["H"], f_["H"]
    nb = Hc.shape[1]
    freqs = np.arange(nb) * DF
    r_db = np.array([_pearson(_db(Hc[:, b]), _db(Hf[:, b])) for b in range(nb)])
    r_lin = np.array([_pearson(np.abs(Hc[:, b]), np.abs(Hf[:, b])) for b in range(nb)])
    lsd = np.mean(np.abs(_db(Hc) - _db(Hf)), axis=0)
    level = 20 * np.log10(np.abs(Hc).mean(0) / np.abs(Hf).mean(0))
    rel = np.linalg.norm(Hc - Hf, axis=0) / np.linalg.norm(Hf, axis=0)
    win = int(round(WIN_HZ / DF)) + 1                    # 11 bins = +-2.5 Hz
    r_w = _window(r_db, win)
    lo = int(round(F_LO / DF))
    bad = [b for b in range(lo, nb) if np.isfinite(r_w[b]) and r_w[b] < R_VALID]
    limit = float(freqs[bad[0]]) if bad else None
    marks = {}
    for f in (50, 100, 150, 200, 250, 299.5):
        b = int(round(f / DF))
        marks["{:g}".format(f)] = {"r_conv_db_win": float(r_w[b]), "lsd_db": float(lsd[b]),
                                   "level_db": float(level[b]), "rel_err": float(rel[b])}
    rep = {"room": c["label"], "n_rx": int(Hc.shape[0]), "criterion": {
               "r_valid": R_VALID, "window_hz": WIN_HZ, "scan_from_hz": F_LO,
               "rule": "first f >= 25 Hz where the 5 Hz-window dB Pearson(dx0.02, dx0.01) < 0.99"},
           "gt_validity_limit_hz": limit, "valid_to_300hz": limit is None,
           "n_bins_below_criterion": len(bad),
           "worst_window_r": float(np.nanmin(r_w[lo:])),
           "worst_window_r_hz": float(freqs[lo + int(np.nanargmin(r_w[lo:]))]),
           "loop_s": {"dx0.02": c["loop_s"], "dx0.01": f_["loop_s"]},
           "at": marks, "freqs_hz": freqs.tolist(), "r_conv_db": r_db.tolist(),
           "r_conv_db_window": r_w.tolist(), "r_conv_lin": r_lin.tolist(),
           "lsd_db": lsd.tolist(), "level_db": level.tolist(), "rel_err": rel.tolist(),
           "richardson_note": "for a 2nd-order scheme the dx=0.02 error is ~(4/3) x "
                              "|H(0.02) - H(0.01)|"}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(out / "a3_dx_convergence.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), dpi=160, sharex=True)
    ax[0].plot(freqs, r_db, color="0.75", lw=0.8, label="per bin")
    ax[0].plot(freqs, r_w, color="#0072B2", lw=2, label="5 Hz window")
    ax[0].axhline(R_VALID, color="#D55E00", ls="--", lw=1.2, label="validity criterion 0.99")
    if limit:
        for a_ in ax:
            a_.axvline(limit, color="#D55E00", lw=1.5)
    ax[0].set_ylabel("dB spatial Pearson\n dx 0.02 vs 0.01")
    ax[0].set_ylim(min(0.9, np.nanmin(r_w[lo:]) - 0.01), 1.001)
    ax[0].legend(fontsize=9, loc="lower left")
    ax[1].plot(freqs, lsd, color="#009E73")
    ax[1].set_ylabel("LSD between\nresolutions (dB)")
    ax[2].plot(freqs, rel, color="#CC79A7")
    ax[2].set_ylabel("relative L2\n|H.02-H.01|/|H.01|")
    ax[2].set_xlabel("frequency (Hz)")
    for a_ in ax:
        a_.axvspan(0, F_LO, color="0.92", zorder=0)
        a_.grid(alpha=0.25)
    fig.suptitle("A3 -- FDTD ground-truth convergence, {} ({} receivers on a node-aligned "
                 "0.08 m grid)\nGT-validity limit: {}".format(
                     c["label"], Hc.shape[0],
                     "{:.1f} Hz".format(limit) if limit else "none -- valid to 300 Hz"),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "a3_dx_convergence.png", facecolor="white")
    print("GT-validity limit: {}".format("{:.1f} Hz".format(limit) if limit else
                                         "NONE -- r_conv >= 0.99 to 300 Hz"))
    print("worst 5-Hz-window r_conv {:.5f} at {:.1f} Hz".format(rep["worst_window_r"],
                                                                rep["worst_window_r_hz"]))
    for k, v in marks.items():
        print("  {:>6s} Hz  r_win {:.5f}  LSD {:.3f} dB  level {:+.3f} dB  rel.err {:.4f}".format(
            k, v["r_conv_db_win"], v["lsd_db"], v["level_db"], v["rel_err"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
