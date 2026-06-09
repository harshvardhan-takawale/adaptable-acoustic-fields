"""P2-3.5 meeting figures: (1) all methods incl oracle ceiling fail equally on unseen
rooms (coverage, not search); (2) known-geometry rendering works at training density
(LOO) but collapses in the sparse 45-room gaps. 1920x1080, traceable."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("outputs/known_geometry")
plt.rcParams.update({"font.size": 15, "axes.titlesize": 19, "axes.labelsize": 16,
                     "figure.titlesize": 23, "savefig.facecolor": "white", "figure.facecolor": "white"})
INTERP = ["L4.50_W4.00_H3.25", "L4.40_W4.09_H3.26", "L3.52_W4.31_H3.40", "L4.82_W3.81_H2.92"]
EXTRAP = ["L4.10_W3.01_H3.93", "L5.94_W4.93_H2.51", "L5.92_W3.06_H2.55", "L5.91_W4.17_H3.72",
          "L3.17_W3.00_H3.49", "L5.99_W3.96_H2.54", "L3.14_W3.08_H2.51"]


def _pb(path, key="mag_corr_full"):
    try:
        return json.loads(Path(path).read_text())["per_band_mag_corr"][key]
    except Exception:
        return np.nan


def fig1_methods():
    """Per interpolative room: 8-recv vs lookup-RBF vs lookup-lin vs oracle (full + 0-250)."""
    p23 = json.loads((ROOT / "p2_3_8recv_per_band.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(19.2, 10.8), constrained_layout=True)
    methods = ["8-recv search", "lookup-RBF", "lookup-linear", "oracle (best latent)"]
    colors = ["#7f7f7f", "#1f77b4", "#17becf", "#d62728"]
    for ax, band, bl in zip(axes, ["mag_corr_full", "mag_corr_0_250"], ["full spectrum", "0-250 Hz (modal band)"]):
        x = np.arange(len(INTERP)); w = 0.2
        for k, (mth, col) in enumerate(zip(methods, colors)):
            vals = []
            for r in INTERP:
                if mth == "8-recv search":
                    vals.append(p23.get(r, {}).get(band, np.nan))
                elif mth == "lookup-RBF":
                    vals.append(_pb(ROOT / f"lookup/{r}__rbf/metrics.json", band))
                elif mth == "lookup-linear":
                    vals.append(_pb(ROOT / f"lookup/{r}__linear/metrics.json", band))
                else:
                    vals.append(_pb(ROOT / f"oracle/{r}/metrics.json", band))
            ax.bar(x + (k - 1.5) * w, vals, w, label=mth, color=col)
        ax.axhline(0.9, color="green", ls="--", lw=1, label="target (0.9)")
        ax.set_xticks(x); ax.set_xticklabels([r.replace("_", "\n") for r in INTERP], fontsize=11)
        ax.set_ylabel("magnitude correlation"); ax.set_ylim(0, 1.0)
        ax.set_title(f"Interpolative rooms — {bl}"); ax.grid(True, axis="y", alpha=0.3)
        if band == "mag_corr_full":
            ax.legend(loc="upper right", fontsize=12)
    fig.suptitle("Even the oracle (best possible latent) can't render unseen rooms — it's training-set coverage, not the search",
                 fontweight="bold")
    fig.text(0.5, 0.01, "All four routes collapse to ~0.27 (full) / ~0.4 (modal): the 45-room decoder has no good latent "
             "for an unseen geometry. Source: outputs/known_geometry/{lookup,oracle}, p2_3_8recv_per_band.json.",
             ha="center", fontsize=12, style="italic", color="#444")
    p = ROOT / "fig_methods_comparison.png"; fig.savefig(p, dpi=100); plt.close(fig)
    return p


def fig2_density():
    """The positive contrast: LOO (training density) 0.89 vs test (sparse gaps) 0.27."""
    loo = json.loads((ROOT / "lookup_summary.json").read_text())["loo"]
    interp_test = np.nanmean([_pb(ROOT / f"lookup/{r}__rbf/metrics.json") for r in INTERP])
    interp_test_m = np.nanmean([_pb(ROOT / f"lookup/{r}__rbf/metrics.json", "mag_corr_0_250") for r in INTERP])
    all_test = np.nanmean([_pb(ROOT / f"lookup/{r}__rbf/metrics.json") for r in INTERP + EXTRAP])
    fig, ax = plt.subplots(figsize=(19.2, 10.8), constrained_layout=True)
    cats = ["LOO held-out\n(training density, ~0.34 m)", "Interpolative test\n(sparse gaps, ~0.43 m)",
            "All test rooms\n(~0.61 m from training)"]
    full = [loo["rbf"]["mean_mag_corr_full"], interp_test, all_test]
    modal = [loo["rbf"]["mean_mag_corr_0_250"], interp_test_m,
             np.nanmean([_pb(ROOT / f"lookup/{r}__rbf/metrics.json", "mag_corr_0_250") for r in INTERP + EXTRAP])]
    x = np.arange(len(cats)); w = 0.35
    ax.bar(x - w / 2, full, w, label="mag corr (full)", color="#1f77b4")
    ax.bar(x + w / 2, modal, w, label="mag corr (0-250 Hz)", color="#ff7f0e")
    for i, (f, m) in enumerate(zip(full, modal)):
        ax.text(i - w / 2, f + 0.02, f"{f:.2f}", ha="center", fontweight="bold")
        ax.text(i + w / 2, m + 0.02, f"{m:.2f}", ha="center", fontweight="bold")
    ax.axhline(0.9, color="green", ls="--", lw=1.2, label="target (0.9)")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("magnitude correlation (known-geometry lookup)"); ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3); ax.legend(loc="upper right", fontsize=13)
    fig.suptitle("Known-geometry rendering WORKS at training density (LOO 0.89) — it collapses only in the sparse 45-room gaps",
                 fontweight="bold")
    fig.text(0.5, 0.01, "Leave-one-out: predict a held-out room's latent from the other 44 and render — 0.89 mag corr / "
             "2.6 dB LSD, no measurements. The route is sound; the binding constraint is training density. "
             "Source: outputs/known_geometry/lookup_summary.json + lookup/.",
             ha="center", fontsize=12, style="italic", color="#444")
    p = ROOT / "fig_density_contrast.png"; fig.savefig(p, dpi=100); plt.close(fig)
    return p


def main():
    for fn in (fig1_methods, fig2_density):
        p = fn()
        from PIL import Image
        print(f"wrote {p}  {Image.open(p).size}")


if __name__ == "__main__":
    main()
