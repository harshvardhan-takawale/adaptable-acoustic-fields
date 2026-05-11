"""Chunk 3.8: regenerate `06_latent_manifold.png` from the C1 FiLM latent_probe
JSON at a presentation-ready resolution (≥ 1920 px on the long edge).

The original `aaf/eval/latent_probing.py` writes a 7×4 inch figure at dpi=110,
producing 758×429 px — below the meeting deck's 1920×1080 spec. This script
re-emits the same PC1-vs-L scatter at figsize (10, 6) and dpi 200 → ~2000 px
wide. Pulls all data straight from `latent_probe.json` (no model needed).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE = REPO_ROOT / "outputs/multi_room/sweep/C1_film/latent_probe/latent_probe.json"
OUT = REPO_ROOT / "outputs/meeting_assets/06_latent_manifold.png"


def main() -> int:
    d = json.loads(PROBE.read_text())
    Ls_train = np.asarray(d["Ls_train"], dtype=float)
    pc1_train = np.asarray(d["pc1_train"], dtype=float)
    Ls_test = np.asarray(d.get("Ls_test", []), dtype=float)
    pc1_test = np.asarray(d.get("pc1_test", []), dtype=float)
    r2 = float(d["pc1_vs_L_r2"])
    slope = float(d["slope_PC1_per_m"])

    # Re-fit intercept from train data to draw the line.
    intercept = float(pc1_train.mean() - slope * Ls_train.mean())

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    ax.scatter(Ls_train, pc1_train, s=140, color="steelblue", marker="o",
               label="train (7 rooms)", edgecolors="black", linewidth=0.6, zorder=3)
    if Ls_test.size:
        ax.scatter(Ls_test, pc1_test, s=140, color="indianred", marker="^",
                   label="zero-shot test", edgecolors="black", linewidth=0.6, zorder=3)

    L_lo, L_hi = float(Ls_train.min() - 0.2), float(Ls_train.max() + 0.2)
    L_grid = np.linspace(L_lo, L_hi, 100)
    ax.plot(L_grid, slope * L_grid + intercept, "k--", lw=1.4,
            label=f"linear fit  (R² = {r2:.3f})", zorder=2)

    ax.set_xlabel("Room length L (m)", fontsize=13)
    ax.set_ylabel("Latent PC1", fontsize=13)
    ax.set_title(
        "C1 FiLM — latent PC1 vs room L:  did z_s learn the geometry?\n"
        f"PC1-vs-L  R² = {r2:.3f}   (target was > 0.7 — Chunk 3.6 met it for the first time)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=11)
    # Annotate the R² prominently in the body of the plot for at-a-glance legibility.
    ax.text(0.97, 0.05, f"R² = {r2:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=14, weight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="black", alpha=0.9))

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"# wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
