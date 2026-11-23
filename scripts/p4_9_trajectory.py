"""P4-9 training-trajectory figure (plan section 4): HF and UF 25-100 Hz against training step for every arm read so far,
redrawn at every read so progress is visible between verdicts.

Arms: R0 (P4-8: 30 / 60 / 90 / 120K); S2 (P4-8: 60 / 120K) continued as S2X (150 / 180 / 210 / 240K); CTL, DIR, DIR-G
(fine-tunes of R0 @120K: 130 / 140 / 150K). Numbers from the standard report (`p4_8_primary_table.load`: HF = in-family
test r 200-300 Hz, 15 shapes; UF = unseen 25-100 Hz, 16 rooms; off-lattice). Dashed: S2X's pre-registered bars
(HF >= 0.56, UF >= 0.49). Reads before 180K carry the restart dip (interim, D181 A8).
Writes outputs/p4_9/meeting/F0_trajectory.{png,json}.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.p4_8_primary_table import load  # noqa: E402

OUT = Path("outputs/p4_9/meeting")
# the dataviz reference categorical palette, slots 1-5 in fixed order (validated default instance)
COL = {"R0": "#2a78d6", "S2 / S2X": "#eb6834", "CTL": "#1baf7a", "DIR": "#eda100", "DIR-G": "#e87ba4"}
SERIES = {
    "R0": [("R0_30K", 30), ("R0_60K", 60), ("R0_90K", 90), ("R0_120K", 120)],
    "S2 / S2X": [("S2_60K", 60), ("S2_120K", 120), ("S2X_150K", 150), ("S2X_180K", 180), ("S2X_210K", 210),
                 ("S2X_240K", 240)],
    "CTL": [("R0_120K", 120), ("CTL_130K", 130), ("CTL_140K", 140), ("CTL_150K", 150)],
    "DIR": [("R0_120K", 120), ("DIR_130K", 130), ("DIR_140K", 140), ("DIR_150K", 150)],
    "DIR-G": [("R0_120K", 120), ("DIRG_130K", 130), ("DIRG_140K", 140), ("DIRG_150K", 150)],
}


def main():
    pts = {}
    for s, items in SERIES.items():
        pts[s] = []
        for n, k in items:
            a = load(n)
            if a is not None:
                pts[s].append({"name": n, "kstep": k, "HF": a["test_r"]["200-300"], "UF": a["uf"]["25-100"]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "F0_trajectory.json").write_text(json.dumps(pts, indent=1))
    plt.rcParams.update({"font.size": 15, "axes.edgecolor": "#888888", "axes.labelcolor": "#333333",
                         "xtick.color": "#555555", "ytick.color": "#555555"})
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=160, facecolor="white")
    for ax, key, bar, title in ((axes[0], "HF", 0.56, "In-family HF (test r, 200-300 Hz, 15 shapes)"),
                                (axes[1], "UF", 0.49, "Unseen UF (r, 25-100 Hz, 16 rooms)")):
        ax.set_facecolor("white")
        ax.grid(True, color="#e6e6e6", linewidth=1)
        ax.set_axisbelow(True)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.axhline(bar, color="#888888", linestyle=(0, (5, 4)), linewidth=1.5,
                   label="S2X pre-registered bar (HF 0.56 / UF 0.49)")
        ax.axvspan(120, 180, color="#f2f2f2", zorder=0)
        ends = []
        for s, P in pts.items():
            if len(P) < 2:                     # a branch arm is drawn once it has a read of its own
                continue
            x = [p["kstep"] for p in P]
            y = [p[key] for p in P]
            ax.plot(x, y, color=COL[s], linewidth=2, marker="o", markersize=8, markeredgecolor="white",
                    markeredgewidth=2, label=s, zorder=3)
            ends.append([y[-1], x[-1], s])
        ax.ends_ = ends
        ax.set_xlabel("training step (thousands)")
        ax.set_title(title, fontsize=16, color="#222222", loc="left")
        ax.set_xlim(20, 265)
    lo = min(p[k] for P in pts.values() for p in P for k in ("HF",)) if any(pts.values()) else 0.3
    axes[0].set_ylim(min(0.40, lo - 0.02), 0.65)
    axes[1].set_ylim(0.40, 0.56)
    for ax in axes:                            # end labels, pushed apart so arms that end together stay legible
        lo_, hi_ = ax.get_ylim()
        gap = 0.045 * (hi_ - lo_)
        ends = sorted(ax.ends_, key=lambda e: (e[0], e[2]))
        ys = [e[0] for e in ends]
        for i in range(1, len(ys)):                # against EVERY lower label that ends within 15K steps
            for j in range(i):
                if abs(ends[i][1] - ends[j][1]) < 15:
                    ys[i] = max(ys[i], ys[j] + gap)
        for (y0, x0, s), yl in zip(ends, ys):
            ax.text(x0 + 3, yl, s, color="#333333", fontsize=12, va="center")
    for ax in axes:
        ax.text(150, ax.get_ylim()[0] + 0.004, "restart dip (interim)", ha="center", color="#777777", fontsize=11)
    axes[1].legend(loc="upper left", frameon=False, fontsize=13)
    fig.suptitle("P4-9 training trajectories -- off-lattice, standard report at every read (redrawn as reads land)",
                 fontsize=17, color="#222222", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "F0_trajectory.png", facecolor="white")
    print("wrote", OUT / "F0_trajectory.png", {s: len(P) for s, P in pts.items()})


if __name__ == "__main__":
    main()
