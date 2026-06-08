"""Figure 5 - 05_phase2_progress.png

Phase-2 trajectory slide: in-distribution val LSD across M1 -> P2-2.5-B -> P3.

HONESTY-CRITICAL: P3 is in progress. Its bar is drawn as a PROJECTED band
(1.8-2.2 dB target), hatched + translucent + distinct edge, with the CURRENT
latest live val LSD labeled. The two real bars (M1, P2-2.5-B) are solid.

Every plotted number is loaded from the named source scalars.json at run time
and PRINTED before plotting.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = "/fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields"

# ----- shared style -----
plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 19,
        "axes.labelsize": 16,
        "figure.titlesize": 24,
        "legend.fontsize": 14,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)

# Dimension color convention is not used here (no L/W/H series); use a neutral
# real-bar color plus a distinct projection color.
REAL_COLOR = "#1f77b4"  # blue
PROJ_COLOR = "#9467bd"  # purple, visually distinct from real bars
TARGET = 2.5


def latest_val(path):
    """Return (lsd_db, iter, n_val_rows) for the latest phase=='val' row."""
    with open(path) as f:
        rows = json.load(f)
    vals = [r for r in rows if r.get("phase") == "val" and "lsd_db" in r]
    if not vals:
        return None, None, 0
    last = vals[-1]
    return float(last["lsd_db"]), int(last["iter"]), len(vals)


def main():
    src_m1 = os.path.join(REPO, "outputs/multi_room_3d/M1_45rooms/scalars.json")
    src_b = os.path.join(REPO, "outputs/diag_p2_2_5/B_45rm_ddp/scalars.json")
    src_p3 = os.path.join(REPO, "outputs/multi_room_3d/P3_45rooms_4gpu/scalars.json")

    m1_lsd, m1_iter, m1_n = latest_val(src_m1)
    b_lsd, b_iter, b_n = latest_val(src_b)

    p3_exists = os.path.exists(src_p3)
    if p3_exists:
        p3_lsd, p3_iter, p3_n = latest_val(src_p3)
    else:
        p3_lsd, p3_iter, p3_n = None, None, 0

    # --- PRINT every number read from source ---
    print("[READ] M1 (final)        :", m1_lsd, "dB @ iter", m1_iter, f"({m1_n} val rows)")
    print("[READ] P2-2.5-B (final)  :", b_lsd, "dB @ iter", b_iter, f"({b_n} val rows)")
    if p3_exists:
        print("[READ] P3 (in progress)  :", p3_lsd, "dB @ iter", p3_iter, f"({p3_n} val rows)")
    else:
        print("[READ] P3 scalars.json missing - projection band only")
    print("[CONST] target line      :", TARGET, "dB")

    # P3 projection band (target, NOT a result)
    P3_LO, P3_HI = 1.8, 2.2

    # ----- figure -----
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, constrained_layout=True)
    # Reserve a strip at the bottom for the 2-line italic caption so it never
    # collides with the x-tick labels.
    fig.get_layout_engine().set(rect=(0.0, 0.055, 1.0, 0.945))
    ax = fig.add_subplot(111)

    x = [0, 1, 2]
    labels = ["P2-2 M1\n(45 rooms)", "P2-2.5-B\n(DDP, 45 rooms)", "P3\n(45 rooms, 4-GPU)"]
    width = 0.55

    # Real solid bars
    ax.bar(
        x[0], m1_lsd, width=width, color=REAL_COLOR, edgecolor="black", linewidth=1.2,
        zorder=3, label="Final (real)",
    )
    ax.bar(
        x[1], b_lsd, width=width, color=REAL_COLOR, edgecolor="black", linewidth=1.2,
        zorder=3,
    )

    # Projected band for P3: a bar spanning [P3_LO, P3_HI], hatched + translucent
    # + distinct dashed edge. Drawn as bottom=P3_LO, height=range.
    ax.bar(
        x[2], P3_HI - P3_LO, bottom=P3_LO, width=width,
        color=PROJ_COLOR, alpha=0.35, edgecolor=PROJ_COLOR, linewidth=2.5,
        linestyle="--", hatch="////", zorder=3,
        label=f"PROJECTED band ({P3_LO:.1f}-{P3_HI:.1f} dB target)",
    )

    # Value labels on the real bars
    ax.text(x[0], m1_lsd + 0.08, f"{m1_lsd:.2f} dB", ha="center", va="bottom",
            fontsize=18, fontweight="bold", zorder=5)
    ax.text(x[1], b_lsd + 0.08, f"{b_lsd:.2f} dB", ha="center", va="bottom",
            fontsize=18, fontweight="bold", zorder=5)

    # Projection annotation box (live current value)
    if p3_exists:
        proj_txt = (
            f"PROJECTED\n(training in progress,\n"
            f"latest {p3_lsd:.2f} dB @ iter {p3_iter})"
        )
    else:
        proj_txt = "PROJECTED\n(training in progress)"
    ax.annotate(
        proj_txt,
        xy=(x[2], P3_HI), xytext=(x[2], P3_HI + 1.35),
        ha="center", va="bottom", fontsize=14, color=PROJ_COLOR, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=PROJ_COLOR, lw=1.8),
        arrowprops=dict(arrowstyle="->", color=PROJ_COLOR, lw=1.8),
        zorder=6,
    )
    # Mark the current live P3 value as a tick on the projected bar
    if p3_exists:
        ax.plot([x[2] - width / 2, x[2] + width / 2], [p3_lsd, p3_lsd],
                color=PROJ_COLOR, lw=2.0, ls=":", zorder=4)
        ax.text(x[2] + width / 2 + 0.04, p3_lsd, f"current {p3_lsd:.2f} dB",
                ha="left", va="center", fontsize=12, color=PROJ_COLOR, style="italic")

    # 2.5 dB target dashed line
    ax.axhline(TARGET, color="#d62728", ls="--", lw=2.0, zorder=2)
    ax.text(2.42, TARGET + 0.06, f"2.5 dB target", ha="right", va="bottom",
            fontsize=14, color="#d62728", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("In-distribution val LSD (dB)  -  lower is better")
    ax.set_xlim(-0.6, 2.7)
    ax.set_ylim(0, max(m1_lsd + 1.2, 7.0))
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend including a proxy for the projection
    legend_handles = [
        Patch(facecolor=REAL_COLOR, edgecolor="black", label="Final val LSD (real)"),
        Patch(facecolor=PROJ_COLOR, alpha=0.35, edgecolor=PROJ_COLOR, hatch="////",
              label=f"P3 PROJECTED band ({P3_LO:.1f}-{P3_HI:.1f} dB target)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.95)

    fig.suptitle("Rapid progress once the bottleneck was identified", fontweight="bold")

    caption = (
        "In-distribution val LSD across Phase 2. M1 and P2-2.5-B are final values; the "
        "P3 bar is a PROJECTION (training in progress at the time of plotting -\n"
        "current value labeled), shown as a band, not a result. "
        "Sources: M1 + diag_p2_2_5 + P3_45rooms_4gpu scalars.json."
    )
    fig.text(0.5, 0.010, caption, ha="center", fontsize=12, style="italic", color="#444")

    out = os.path.join(REPO, "outputs/phase2_meeting_assets/05_phase2_progress.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=100)
    print("[SAVED]", out)

    from PIL import Image

    w, h = Image.open(out).size
    print("[VERIFY] PNG size:", w, "x", h, "ok=", (w >= 1920 and h >= 1080))


if __name__ == "__main__":
    main()
