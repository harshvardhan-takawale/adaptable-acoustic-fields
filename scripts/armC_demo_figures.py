"""Arm C demo pack, stage 2: figures from the cached fields. No GPU, no re-simulation.

Stage 1 (`armC_demo_metrics.py`) rendered 12 scenarios at 4096 receivers and re-simulated ISM
ground truth on the same grid, caching both as complex64 npz. It also enforced the abort rule:
spatial Pearson came out worst 0.920 / mean 0.951 against a 0.70 threshold, so the spatial claim
is supportable and these figures may be drawn. Nothing here re-derives a number -- every value
plotted is read from `metrics.json` or recomputed from the same cached arrays.

Three figures:

  A  headline -- median geometry, one mode, 4 scenarios x (predicted / GT) field grid, shared
     colour scale per row so panels are visually comparable.
  B  signals -- per scenario, spectrum overlay (0-300 Hz, centre receiver) and band-limited RIR
     with a 50 ms zoom.
  C  the metric that decides the claim -- spatial Pearson across all 12 scenarios against the
     0.70 abort line, with the FDTD-corpus model's 0.24-0.60 band drawn for contrast. Same
     metric, same code path, two corpora; it is the clearest single slide available.

Two honesty rules baked in rather than footnoted:
  * MAGNITUDE correlation (0.61-0.76) pools every bin including deep nulls, while spatial
    Pearson is evaluated at modal bins. This chunk has twice been misled by null-dominated
    metrics, so both are shown and labelled, never just the flattering one.
  * `mode_shape_invariance` = 0.9921, recorded by P3-2b for this arm, is agreement with the
    ANALYTIC cosine shape on the 8x8 grid -- a different quantity from the pointwise pred-vs-GT
    Pearson here. It is annotated as such, not merged in.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 200
SCEN = ["a_baseline", "b_east_curtain", "c_north_absorber", "d_two_wall"]
TITLES = {
    "a_baseline": "(a) baseline\nall walls $\\alpha$=0.15",
    "b_east_curtain": "(b) east curtain\n$\\alpha$=0.50",
    "c_north_absorber": "(c) north absorber\n$\\alpha$=0.70  ★ held-out combo",
    "d_two_wall": "(d) two-wall edit\neast 0.50 + south 0.70",
}
# Okabe-Ito, CVD-safe
C_PRED, C_GT, C_ACC = "#0072B2", "#D55E00", "#009E73"
FDTD_BAND = (0.24, 0.60)      # the FDTD-corpus model on this same metric
ABORT = 0.70


def _load(tag, scen, d):
    return np.load(Path(d) / "{}_{}.npz".format(tag, scen))


def _grid(vals, rx, n=64):
    """4096 receivers were emitted x-major (x outer, y inner) -> [ny, nx] for imshow."""
    return np.asarray(vals, float).reshape(n, n).T


def fig_a(meta, fdir, out, tag="median", mode_idx=0):
    d0 = _load(tag, SCEN[0], fdir)
    L, W = float(d0["L"]), float(d0["W"])
    b = int(d0["mode_bins"][mode_idx])
    f = float(d0["mode_f"][mode_idx])
    nx_, ny_ = int(d0["mode_nx"][mode_idx]), int(d0["mode_ny"][mode_idx])
    rx = d0["rx"]

    # aspect="equal" on a 4-wide x 2-high grid of rooms whose L/W is ~1.9 means the content is
    # ~3.7:1; a 2.1:1 canvas leaves the panels stranded in whitespace. Size the canvas to the
    # content and raise dpi to hold the pixel floor.
    fig, axes = plt.subplots(2, 4, figsize=(19.2, 7.4), dpi=260)
    rows = {}
    for j, s in enumerate(SCEN):
        d = _load(tag, s, fdir)
        rows[s] = (20 * np.log10(np.maximum(np.abs(d["pred"][:, b]), 1e-30)),
                   20 * np.log10(np.maximum(np.abs(d["gt"][:, b]), 1e-30)))
    allv = np.concatenate([np.concatenate(v) for v in rows.values()])
    vmax = np.percentile(allv, 99.5)
    vmin = vmax - 40.0

    by_scen = {r["scenario"]: r for r in meta["scenarios"] if r["geometry"] == tag}
    for j, s in enumerate(SCEN):
        p, g = rows[s]
        pr = by_scen[s]["modes"][mode_idx]["spatial_pearson_db"]
        for i, (v, lab) in enumerate(((p, "PREDICTED (zero-shot)"), (g, "ISM ground truth"))):
            ax = axes[i, j]
            im = ax.imshow(_grid(v, rx), origin="lower", extent=[0, L, 0, W],
                           vmin=vmin, vmax=vmax, cmap="magma", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(lab, fontsize=12, fontweight="bold")
            if i == 0:
                ax.set_title(TITLES[s], fontsize=11)
                ax.text(0.5, -0.06, "spatial r = {:+.3f}".format(pr), transform=ax.transAxes,
                        ha="center", va="top", fontsize=11, color=C_ACC, fontweight="bold")
    fig.subplots_adjust(hspace=0.22, wspace=0.06, top=0.86, bottom=0.13)
    cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.012)
    cb.set_label("|H| (dB, shared scale)", fontsize=11)
    fig.suptitle(
        "One INR, four rooms it never saw — dense field at mode ({},{}), {:.1f} Hz  |  "
        "{:.2f} × {:.2f} m unseen geometry".format(nx_, ny_, f, L, W),
        fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.015,
             "Zero-shot: one forward pass per panel from a single checkpoint, no measurements of "
             "these rooms, no per-room fitting. Queried on a 64×64 grid; the model trained on "
             "8×8.\n(c) is doubly zero-shot — north@0.70 lies in the held-out slab and "
             "appears in no training config. Ground truth re-simulated by ISM on the identical "
             "grid.",
             ha="center", va="bottom", fontsize=10.5)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"geometry": tag, "L": L, "W": W, "mode": [nx_, ny_], "f_hz": f,
            "vmin_db": float(vmin), "vmax_db": float(vmax),
            "spatial_pearson": {s: by_scen[s]["modes"][mode_idx]["spatial_pearson_db"]
                                for s in SCEN}}


def fig_b(meta, fdir, out, tag="median", fs=4096.0, n_time=8192):
    df = fs / n_time
    fig, axes = plt.subplots(2, 4, figsize=(19.2, 8.4), dpi=DPI)
    by_scen = {r["scenario"]: r for r in meta["scenarios"] if r["geometry"] == tag}
    info = {}
    for j, s in enumerate(SCEN):
        d = _load(tag, s, fdir)
        c = int(d["centre"])
        pr, gt = d["pred"][c], d["gt"][c]
        fr = np.arange(pr.size) * df
        ax = axes[0, j]
        ax.plot(fr, 20 * np.log10(np.maximum(np.abs(gt), 1e-30)), color=C_GT, lw=1.9,
                label="ISM GT", zorder=2)
        ax.plot(fr, 20 * np.log10(np.maximum(np.abs(pr), 1e-30)), color=C_PRED, lw=1.1,
                ls="--", label="predicted", zorder=3)
        ax.set_xlim(0, 300); ax.set_title(TITLES[s], fontsize=11)
        ax.set_xlabel("frequency (Hz)")
        if j == 0:
            ax.set_ylabel("|H| (dB)\ncentre receiver", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9, loc="lower left")
        r = by_scen[s]
        ax.text(0.97, 0.95, "LSD {:.2f} dB".format(r["band_lsd_db"]), transform=ax.transAxes,
                ha="right", va="top", fontsize=10, color=C_ACC, fontweight="bold")

        rp = np.fft.irfft(pr, n=n_time); rg = np.fft.irfft(gt, n=n_time)
        t = np.arange(n_time) / fs * 1e3
        m = t <= 50.0
        ax = axes[1, j]
        ax.plot(t[m], rg[m], color=C_GT, lw=1.7, label="ISM GT")
        ax.plot(t[m], rp[m], color=C_PRED, lw=1.0, ls="--", label="predicted")
        ax.set_xlabel("time (ms)"); ax.set_xlim(0, 50)
        if j == 0:
            ax.set_ylabel("band-limited RIR\n(0–300 Hz, 50 ms zoom)", fontsize=11,
                          fontweight="bold")
        ax.text(0.97, 0.95, "RIR r = {:+.3f}".format(r["rir_pearson"]), transform=ax.transAxes,
                ha="right", va="top", fontsize=10, color=C_ACC, fontweight="bold")
        info[s] = {"band_lsd_db": r["band_lsd_db"], "rir_pearson": r["rir_pearson"],
                   "phase_circ_corr": r["phase_circ_corr"],
                   "magnitude_corr": r["magnitude_corr"]}
    L, W = by_scen[SCEN[0]]["L"], by_scen[SCEN[0]]["W"]
    fig.suptitle("Signal-level reconstruction, same checkpoint — {:.2f} × {:.2f} m "
                 "unseen geometry".format(L, W), fontsize=15, fontweight="bold")
    fig.text(0.5, 0.005,
             "Zero-shot, one forward pass per scenario. RIR is the inverse transform of the "
             "band-limited prediction — no time-domain fitting anywhere.",
             ha="center", fontsize=10.5)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return info


def fig_c(meta, out):
    rows = meta["scenarios"]
    labels = ["{}\n{}".format(r["geometry"], r["scenario"].split("_", 1)[1]) for r in rows]
    sp = [r["spatial_pearson_mean"] for r in rows]
    mg = [r["magnitude_corr"] for r in rows]
    held = [r["held_out_combo"] for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(19.2, 8.4), dpi=DPI)
    ax.axhspan(FDTD_BAND[0], FDTD_BAND[1], color="#999999", alpha=0.30, zorder=0)
    ax.text(-0.42, np.mean(FDTD_BAND),
            "FDTD-corpus model\non this same metric\n(0.24 – 0.60)",
            ha="left", va="center", fontsize=10.5, color="#333333", style="italic", zorder=5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", alpha=0.94))
    ax.axhline(ABORT, color=C_GT, lw=2.2, ls="--", zorder=1,
               label="abort threshold (0.70) — fixed before the run")
    ax.bar(x - 0.19, sp, 0.38, color=[C_ACC if h else C_PRED for h in held], zorder=3,
           label="spatial Pearson (modal bins)")
    ax.bar(x + 0.19, mg, 0.38, color="#CC79A7", zorder=3,
           label="magnitude corr (ALL bins, null-dominated)")
    for xi, v, h in zip(x, sp, held):
        ax.text(xi - 0.19, v + 0.012, "{:.3f}".format(v), ha="center", fontsize=9.5,
                fontweight="bold" if h else "normal")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("correlation with ISM ground truth", fontsize=12)
    ax.set_ylim(0, 1.30)
    ax.legend(fontsize=11, loc="upper center", ncol=3, framealpha=0.95)
    ax.set_title("Spatial reconstruction on a 64×64 grid — worst {:.3f}, mean {:.3f} "
                 "across 12 unseen scenarios".format(meta["spatial_pearson_worst"],
                                                     meta["spatial_pearson_mean"]),
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.005,
             "Green bars = held-out wall/material combo (north@0.70), excluded from every "
             "training config. Magnitude correlation pools all 601 bins including deep nulls "
             "and is shown to keep the comparison honest;\nspatial Pearson is evaluated at modal "
             "bins. Separately, P3-2b recorded mode_shape_invariance 0.9921 for this arm — "
             "agreement with the analytic cosine shape on the 8×8 grid, a DIFFERENT quantity "
             "from the pointwise Pearson plotted here.",
             ha="center", fontsize=10.5)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"worst": meta["spatial_pearson_worst"], "mean": meta["spatial_pearson_mean"],
            "per_scenario": {r["scenario"] + "@" + r["geometry"]: r["spatial_pearson_mean"]
                             for r in rows}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="outputs/armC_demo/metrics.json")
    ap.add_argument("--fields", default="outputs/armC_demo/fields")
    ap.add_argument("--outdir", default="outputs/armC_demo")
    a = ap.parse_args()
    meta = json.load(open(a.metrics))
    if not meta.get("proceed_to_figures", False):
        print("metrics.json says the abort rule FAILED; not drawing figures.")
        return 2
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    ia = fig_a(meta, a.fields, out / "figA_spatial_fields.png")
    ib = fig_b(meta, a.fields, out / "figB_signals.png")
    ic = fig_c(meta, out / "figC_spatial_pearson.png")
    json.dump({"figA": ia, "figB": ib, "figC": ic}, open(out / "figures.json", "w"),
              indent=1, default=float)
    for f in ("figA_spatial_fields.png", "figB_signals.png", "figC_spatial_pearson.png"):
        p = out / f
        print("  {}  {:.1f} MB".format(p, p.stat().st_size / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
