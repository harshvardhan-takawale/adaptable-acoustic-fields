"""Arm C v2: the doorway-physics figure. GROUND TRUTH ONLY -- every panel is FDTD simulation.

Draws what `armC_v2_doorway.py` simulated: a two-room domain at three aperture widths, showing
how energy crosses a doorway. This motivates the next phase (aperture as a trainable edit axis).
It is NOT a model result and the figure says so in the title, in a per-row banner and in the
caption, because a reader who mistakes a simulated field for a prediction draws exactly the
wrong conclusion.

Two rows, both on a colour scale shared across all three apertures so the transfer is directly
comparable panel to panel:
  * the 60 Hz sub-room (1,1) field -- sealed shows two independent fields, open shows one
  * band-integrated (20-300 Hz) level -- the literal "how much energy is here" view

The sealed panel's right-hand room is EXACTLY zero, not merely quiet: a full-span slab
disconnects the domain, so H_B is identically 0 and its level is -inf. It is clipped to the
floor for display and labelled as exact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

DPI = 200
L, W, DIV_X = 8.0, 4.0, 4.0
SRC = (0.5, 0.5)
TITLES = {0.0: "SEALED  ($a$ = 0)", 1.0: "DOORWAY  ($a$ = 1.0 m)",
          4.0: "FULLY OPEN  ($a$ = $W$ = 4.0 m)"}
SUB = {0.0: "full-span divider — the two rooms are\nEXACTLY disconnected",
       1.0: "one centred 1.0 m opening", 4.0: "no divider at all"}
DYN_MODE, DYN_BAND = 45.0, 35.0


def _grid(v, nx=128, ny=64):
    """Receivers are emitted x-major (x outer, y inner) -> [ny, nx] for imshow."""
    return np.asarray(v, float).reshape(nx, ny).T


def _draw_divider(ax, a, lw=5.0):
    """Solid parts of the divider. Nothing is drawn when the aperture spans the full width."""
    if a >= W:
        return
    lo, hi = 0.5 * W - 0.5 * a, 0.5 * W + 0.5 * a
    for y0, y1 in (((0.0, lo), (hi, W)) if a > 0.0 else ((0.0, W),)):
        if y1 > y0:
            ax.plot([DIV_X, DIV_X], [y0, y1], color="#00E5FF", lw=lw, solid_capstyle="butt",
                    zorder=6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/armC_demo/v2/doorway")
    ap.add_argument("--out", default="outputs/armC_demo/v2/figF_doorway_physics.png")
    a_ = ap.parse_args()
    d = Path(a_.dir)
    meta = json.load(open(d / "doorway_meta.json"))
    aps = [float(x) for x in meta["apertures"]]
    runs = {float(r["a"]): r for r in meta["runs"]}

    dat = {}
    for a in aps:
        z = np.load(d / "a{:04.0f}.npz".format(1000 * a))
        dat[a] = {"mode": np.abs(z["H_mode"]), "band": z["e_band"], "f": float(z["f_mode"])}

    def to_db(v, ref):
        return 20.0 * np.log10(np.maximum(v, 1e-30) / ref)

    ref_m = max(float(dat[a]["mode"].max()) for a in aps)
    ref_b = max(float(dat[a]["band"].max()) for a in aps)
    rows = [("mode", ref_m, DYN_MODE,
             "field at {:.0f} Hz\nsub-room mode (1,1)".format(dat[aps[0]]["f"])),
            ("band", ref_b, DYN_BAND, "band-integrated level\n20–300 Hz")]

    fig, axes = plt.subplots(2, 3, figsize=(20.4, 8.6), dpi=DPI)
    info = {"rows": {}, "note": meta["source"]}
    for ri, (key, ref, dyn, ylab) in enumerate(rows):
        info["rows"][key] = {"ref_abs": ref, "dynamic_range_db": dyn, "panels": {}}
        for ci, a in enumerate(aps):
            ax = axes[ri, ci]
            v = to_db(dat[a][key], ref)
            im = ax.imshow(_grid(v), origin="lower", extent=[0, L, 0, W], vmin=-dyn, vmax=0.0,
                           cmap="magma", aspect="equal", interpolation="bilinear")
            _draw_divider(ax, a)
            ax.plot(*SRC, marker="*", ms=20, color="#00E5FF", mec="black", mew=1.1, zorder=7)
            ax.add_patch(Rectangle((0, 0), L, W, fill=False, ec="black", lw=1.4, zorder=5))
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.015, 0.965, "A", transform=ax.transAxes, fontsize=15, color="white",
                    fontweight="bold", va="top", zorder=8)
            ax.text(0.985, 0.965, "B", transform=ax.transAxes, fontsize=15, color="white",
                    fontweight="bold", va="top", ha="right", zorder=8)
            if ri == 0:
                ax.set_title("{}\n{}".format(TITLES[a], SUB[a]), fontsize=13,
                             fontweight="bold", pad=10)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=12, fontweight="bold")
            r = runs[a]
            if a == 0.0:
                lab = "room B is EXACTLY zero\n(exact disconnection, not a floor)"
            else:
                lab = "inter-room level difference\n{:+.2f} dB".format(
                    r["ld_db_amplitude_ftb_def"])
            ax.text(0.5, -0.055, lab, transform=ax.transAxes, ha="center", va="top",
                    fontsize=11, fontweight="bold", color="#0072B2")
            info["rows"][key]["panels"][str(a)] = {
                "ld_db_amplitude_ftb_def": r["ld_db_amplitude_ftb_def"],
                "ld_db_power": r["ld_db_power"],
                "room_b_exactly_zero": r["room_b_exactly_zero"]}
        cb = fig.colorbar(im, ax=list(axes[ri]), fraction=0.016, pad=0.012)
        cb.set_label("dB re loudest panel in this row", fontsize=10)

    fig.suptitle("SIMULATION, NOT MODEL OUTPUT — how sound crosses a doorway  "
                 "(FDTD, {:.0f} × {:.0f} m, $dx$ = {} m)".format(L, W, meta["dx"]),
                 fontsize=16.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.012,
             "Every panel is a 2-D FDTD simulation on an 8192-point dense grid — there is NO "
             "neural network anywhere in this figure. It motivates the next phase (doorway "
             "aperture as a trainable edit axis); it is not a result.\n"
             "Source (cyan star) is always in room A. The divider is drawn in cyan; its gap is "
             "the doorway. Sealed → room B is identically zero. Domain, absorption, source and "
             "$dx$ are FT-B's frozen setup, so these panels sit on an already-validated "
             "configuration.",
             ha="center", fontsize=11)
    fig.savefig(a_.out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    json.dump(info, open(Path(a_.out).with_suffix(".json"), "w"), indent=1, default=float)
    print("-> {}".format(a_.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
