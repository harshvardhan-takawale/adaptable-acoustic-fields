"""Chunk 3.8: generate `08_progress_trajectory.png` — the closing optimism
slide. Pure-modal (0-250 Hz) zero-shot LSD on the 6 unseen L, plotted across
the four project chunks where it was measured (or estimable).

Numbers (single source of truth):
- Chunk 3 (R0):        3.70 dB  — retrospective via Chunk 3.6 Track A
                                  band-limited recompute on R0; the original
                                  Chunk 3 didn't measure modal at the time.
                                  Chunk 3.6 Track A gives R0 modal = 3.69 dB
                                  (rounded to 3.70 for the bar).
- Chunk 3.5 (R6 + B1): 3.66 dB  — best of the 9-run sweep, R6_tiny_lhead with
                                  the baseline inner loop; see
                                  outputs/multi_room/sweep/band_limited_summary.md
- Chunk 3.6 (C2 + B6): 3.51 dB  — best of the FiLM/jitter retrains with the
                                  Track-B-winner simplex inner loop; see
                                  tasks/CHUNK_3_6_RESULTS.md.
- Chunk 3.7 (D1 + B1): 2.55 dB  — denser-sweep retrain (15 rooms @ 0.2 m)
                                  with baseline inner loop; see
                                  tasks/CHUNK_3_7_RESULTS.md.

The horizontal target line is at 2 dB ("Phase 1 target"); the bracket
annotation marks the cumulative 1.15 dB drop across chunks.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "outputs/meeting_assets/08_progress_trajectory.png"

LABELS = [
    "Chunk 3*\n(R0)",
    "Chunk 3.5\n(R6 + B1)",
    "Chunk 3.6\n(C2 + B6)",
    "Chunk 3.7\n(D1 + B1)",
]
MODAL_LSD = [3.70, 3.66, 3.51, 2.55]
COLORS = ["#888888", "#9ba0a8", "#7eb9b0", "#2a9d8f"]   # neutral → highlight


def main() -> int:
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    x = np.arange(len(LABELS))
    ax.bar(x, MODAL_LSD, color=COLORS, edgecolor="black", linewidth=0.8, width=0.6)
    # Phase-1 target line.
    ax.axhline(2.0, ls="--", color="green", lw=1.6, label="Phase 1 target (2 dB)")
    # Bar value labels.
    for xi, v in zip(x, MODAL_LSD):
        ax.text(xi, v + 0.05, f"{v:.2f}", ha="center", va="bottom",
                fontsize=12, weight="bold")
    # Annotation bracket: −1.15 dB across project (3.70 → 2.55).
    drop = MODAL_LSD[0] - MODAL_LSD[-1]
    ax.annotate(
        "",
        xy=(3, MODAL_LSD[-1] + 0.4),
        xytext=(0, MODAL_LSD[0] + 0.4),
        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.6,
                        connectionstyle="arc3,rad=-0.18"),
    )
    ax.text(
        1.5, MODAL_LSD[0] + 0.62,
        f"−{drop:.2f} dB across chunks  (data density is the lever)",
        ha="center", color="darkred", fontsize=11.5, weight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_ylabel("Modal-regime zero-shot LSD (0-250 Hz, dB)", fontsize=12)
    ax.set_title(
        "Modal-regime zero-shot LSD across project iterations\n"
        "(same 6 unseen L throughout: {3.25, 3.75, 4.25, 4.75, 5.25, 5.75} m)",
        fontsize=13,
    )
    ax.set_ylim(0, 4.6)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower left", fontsize=11)
    # Footnote: source-of-truth + caveat on retrospective Chunk-3 number.
    ax.text(
        0.005, -0.16,
        "*Retrospective modal estimate from Chunk 3.6 Track A band-limited analysis applied to "
        "Chunk 3's R0 architecture family (modal wasn't measured at the time).",
        transform=ax.transAxes, fontsize=8.5, color="gray", style="italic",
    )

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"# wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
