"""P4-1 Stage 1 figures: the LOS/NLOS field map and the sigma profile.

Both are required deliverables regardless of how Gate 1 lands -- a failure is a result here, not
an absence of one, and these two panels are what make the failure mode legible rather than just
a number. Drawn from `stage1_fields.npz`, which the eval wrote AFTER emitting its verdict.

The sigma panel is the diagnostic that separates the candidate explanations. sigma can only
occlude the receiver->point leg (transmittance accumulates outward from the receiver), so a ray
from an NLOS receiver toward the source is the one direction in which the representation is
structurally ABLE to place the wall. If sigma stays flat through the notch there, the model did
not find the occluder even where it could -- which points at the representation rather than at
ray sampling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from scripts.build_p4_1_lroom import NOTCH_X, NOTCH_Y, VERTS, inside_polygon

DPI = 200
C_PRED, C_GT, C_ACC, C_BAD = "#0072B2", "#D55E00", "#009E73", "#CC79A7"


def _db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def _room_outline(ax, lw=2.0):
    ax.add_patch(MplPolygon(VERTS, closed=True, fill=False, ec="black", lw=lw, zorder=6))


def fig_fields(z, g, out):
    """Predicted vs FDTD at one mode, scattered on the true receiver positions.

    A scatter rather than an image: the receivers are the interior of a NON-convex polygon, and
    gridding them would either invent values inside the notch or need masking that hides which
    points were actually measured.
    """
    rx, los = z["rx"], z["los"].astype(bool)
    bins, freqs = z["mode_bins"], z["mode_f"]
    k = 0
    b, f = int(bins[k]), float(freqs[k])
    p, t = _db(z["pred"][:, b]), _db(z["gt"][:, b])
    vmax = float(np.percentile(np.concatenate([p, t]), 99.5)); vmin = vmax - 40.0

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.0), dpi=DPI)
    for ax, v, title in ((axes[0], t, "FDTD ground truth"),
                         (axes[1], p, "predicted (per-scene fit)")):
        s = ax.scatter(rx[:, 0], rx[:, 1], c=v, s=9, vmin=vmin, vmax=vmax, cmap="magma")
        ax.set_title(title, fontsize=13, fontweight="bold")
        _room_outline(ax)
    cb = fig.colorbar(s, ax=axes[:2], fraction=0.03, pad=0.01)
    cb.set_label("|H| (dB, shared scale)", fontsize=11)

    ax = axes[2]
    ax.scatter(rx[los, 0], rx[los, 1], c=C_ACC, s=9, label="LOS ({})".format(int(los.sum())))
    ax.scatter(rx[~los, 0], rx[~los, 1], c=C_BAD, s=9,
               label="NLOS ({})".format(int((~los).sum())))
    ax.set_title("line-of-sight split", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    _room_outline(ax)

    for ax in axes:
        ax.plot(*z["sigma_rx"], marker="o", ms=9, mfc="none", mec="#00E5FF", mew=2.2, zorder=8)
        ax.set_xlim(-0.3, 6.3); ax.set_ylim(-0.3, 5.3); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        # the notch, and the reflex corner everything hinges on
        ax.add_patch(plt.Rectangle((0, NOTCH_Y), NOTCH_X, 5.0 - NOTCH_Y, fc="#dddddd",
                                   ec="none", zorder=5, alpha=0.55))
        ax.plot([NOTCH_X], [NOTCH_Y], marker="*", ms=15, color="black", zorder=9)

    fig.suptitle("L-room per-scene fit, mode ({},{}) at {:.1f} Hz  |  NLOS spatial R = {:+.3f}, "
                 "LOS {:+.3f}, gap {:+.3f}".format(
                     g["per_mode"][k]["mode"][0], g["per_mode"][k]["mode"][1], f,
                     g["mean_pearson_nlos"], g["mean_pearson_los"], g["los_nlos_gap"]),
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.015,
             "Grey = the removed corner (solid). Black star = the reflex corner. Cyan ring = the "
             "receiver whose sigma profile is plotted in the companion figure.\n"
             "Scatter, not an image: the receivers fill a non-convex polygon, and gridding "
             "would invent values inside the notch.",
             ha="center", fontsize=10.5)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"mode": g["per_mode"][k]["mode"], "f_hz": f, "vmin_db": vmin, "vmax_db": vmax}


def fig_sigma(z, g, out):
    t, pts, sig, rx0 = z["sigma_t"], z["sigma_pts"], z["sigma"], z["sigma_rx"]
    solid = ~inside_polygon(pts)
    sp = g["sigma_probe"]

    fig, axes = plt.subplots(1, 2, figsize=(18.0, 6.2), dpi=DPI)
    ax = axes[0]
    ax.plot(t, sig, color=C_PRED, lw=2.2)
    if solid.any():
        # shade every contiguous solid stretch the ray passes through
        d = np.diff(np.concatenate([[0], solid.astype(int), [0]]))
        for s0, s1 in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            ax.axvspan(t[s0], t[min(s1, len(t) - 1)], color="#999999", alpha=0.45,
                       label="_" if s0 else "inside the notch (solid)")
    ax.set_xlabel("distance from the receiver along the ray (m)", fontsize=12)
    ax.set_ylabel(r"learned $\sigma$ (mean over the band)", fontsize=12)
    ax.set_title(r"$\sigma$ along a ray from an NLOS receiver toward the source",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.text(0.98, 0.95,
            "mean $\\sigma$  solid {:.3g}\n           air {:.3g}\n        ratio {:.2f}".format(
                sp["mean_sigma_solid"], sp["mean_sigma_air"], sp["solid_over_air"]),
            transform=ax.transAxes, ha="right", va="top", fontsize=11, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999999"))

    ax = axes[1]
    ax.plot(pts[:, 0], pts[:, 1], color=C_PRED, lw=2.0, zorder=7, label="probe ray")
    if solid.any():
        ax.scatter(pts[solid, 0], pts[solid, 1], c="#444444", s=6, zorder=8,
                   label="ray inside solid")
    ax.plot(*rx0, marker="o", ms=10, mfc="none", mec="#00E5FF", mew=2.5, zorder=9)
    ax.add_patch(plt.Rectangle((0, NOTCH_Y), NOTCH_X, 5.0 - NOTCH_Y, fc="#dddddd", ec="none",
                               zorder=5, alpha=0.55))
    _room_outline(ax)
    ax.plot([NOTCH_X], [NOTCH_Y], marker="*", ms=15, color="black", zorder=9)
    ax.set_xlim(-0.3, 6.3); ax.set_ylim(-0.3, 5.3); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=10, loc="lower right")
    ax.set_title("where that ray goes", fontsize=13, fontweight="bold")

    fig.suptitle(r"Did the renderer discover the occluder?  $\sigma$ ratio (solid / air) = "
                 "{:.2f}".format(sp["solid_over_air"]), fontsize=15, fontweight="bold")
    fig.text(0.5, 0.015,
             "Ratio > 1 means the model placed attenuation where the wall is. This is the one "
             "direction in which sigma CAN represent the occluder: transmittance accumulates "
             "outward from the receiver,\nso a wall between the SOURCE and a sample point has no "
             "structural representation at all and could only be absorbed into the emitted "
             "signal.",
             ha="center", fontsize=10.5)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return sp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/p4_1/stage1")
    a = ap.parse_args()
    d = Path(a.dir)
    z = np.load(d / "stage1_fields.npz")
    g = json.load(open(d / "GATE1.json"))
    info = {"figG": fig_fields(z, g, d / "figG_lroom_fields.png"),
            "figH": fig_sigma(z, g, d / "figH_sigma_profile.png")}
    json.dump(info, open(d / "figures_stage1.json", "w"), indent=1, default=float)
    for f in ("figG_lroom_fields.png", "figH_sigma_profile.png"):
        p = d / f
        print("  {}  {:.1f} MB".format(p, p.stat().st_size / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
