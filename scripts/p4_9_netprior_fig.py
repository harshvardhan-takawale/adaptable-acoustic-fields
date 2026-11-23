"""P4-9 F8 (X1 / X1b, D189 / D192): how many microphones does a room need, with and without the network as a prior?

Unseen 16 rooms, off-lattice, pooled strata, r per band against the number of measured receivers K. Top row: the exact
polygon is known (X1). Bottom row: only the bounding rectangle is known, for every method (X1b, A1-matched).
Reads outputs/p4_9/netprior/netprior.json -> outputs/p4_9/meeting/F8_netprior.{png,json}.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC = Path("outputs/p4_9/netprior/netprior.json")
OUT = Path("outputs/p4_9/meeting/F8_netprior.png")
B = ("25-100", "100-200", "200-300")
KS = (8, 16, 32, 64)
NET, PHYS, KRR = "#2a78d6", "#1baf7a", "#eda100"       # deck entity colours: R0, physics, FDTD / KRR


def main():
    d = json.loads(SRC.read_text())
    top = d["curves"]["R0_120K|unseen16"]
    bot = d["curves"]["bbox|unseen16"]
    phys = bot["M3"]
    rows = [
        ("Exact room shape known", [
            ("microphones only (Helmholtz KRR)", top["M5"], KRR, "-", "o"),
            ("network fine-tuned on the mics (58 s GPU)", top["M2"], NET, (0, (5, 3)), "s"),
            ("network as prior + mics (0.8 s CPU)", top["M6s"], NET, "-", "o")],
         [("network alone, no mics", top["M0"], NET), ("physics, exact shape, no mics", phys, PHYS)]),
        ("Only the bounding box known", [
            ("microphones only (Helmholtz KRR)", bot["M5"], KRR, "-", "o"),
            ("physics on the box as prior + mics", bot["M7-iii"], PHYS, "-", "o"),
            ("network on the box as prior + mics", bot["M6-iii"], NET, "-", "o")],
         [("network on the box, no mics", bot["M0-iii"], NET), ("physics on the box, beta fitted", bot["M4-iii"], PHYS)]),
    ]
    plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#888888", "axes.labelcolor": "#333333",
                         "xtick.color": "#555555", "ytick.color": "#555555"})
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5), dpi=150, facecolor="white", sharey=True)
    meta = {}
    for i, (rtitle, series, refs) in enumerate(rows):
        for j, b in enumerate(B):
            ax = axes[i, j]
            ax.set_facecolor("white")
            ax.grid(True, color="#e6e6e6", linewidth=1)
            ax.set_axisbelow(True)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
            for lab, cur, col in refs:
                y = cur["32"][b]
                ax.axhline(y, color=col, linestyle=(0, (1, 2)), linewidth=1.8, alpha=0.9, label=lab if j == 0 else None)
            for lab, cur, col, ls, mk in series:
                ys = [cur[str(K)][b] for K in KS]
                ax.plot(KS, ys, color=col, linestyle=ls, linewidth=2.2, marker=mk, markersize=7,
                        markeredgecolor="white", markeredgewidth=1.5, label=lab if j == 0 else None, zorder=3)
                meta.setdefault(rtitle, {}).setdefault(lab, {})[b] = ys
            ax.set_xscale("log", base=2)
            ax.set_xticks(KS)
            ax.set_xticklabels([str(K) for K in KS])
            ax.set_ylim(0.0, 1.0)
            if i == 1:
                ax.set_xlabel("microphones per room (K)")
            if j == 0:
                ax.set_ylabel("r (unseen rooms)")
            ax.set_title("{}  |  {} Hz".format(rtitle, b) if j == 0 else "{} Hz".format(b), fontsize=13,
                         color="#222222", loc="left")
        axes[i, 0].legend(loc="lower right", frameon=False, fontsize=10)
    fig.suptitle("F8 -- the network as a prior for few-microphone reconstruction (16 unseen rooms, off-lattice; "
                 "dotted = no-microphone references)", fontsize=15, color="#222222", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor="white")
    OUT.with_suffix(".json").write_text(json.dumps({"source": str(SRC), "series": meta}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
