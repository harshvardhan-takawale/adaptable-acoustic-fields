"""Figure 2 — 02_modal_density_2d_vs_3d.png.

Quantifies why 3D is harder: the modal regime (below Schroeder) carries far
more distinct eigenfrequencies in 3D than in the Phase-1 2D family.

Sources (every plotted number is READ AT RUN TIME):
  - aaf.sim.analytical_modal_3d.eigenfrequencies_3d  → recompute the distinct
    3D mode count for the box-center room (L=4.5, W=4.0, H=3.25), f_max=250.
  - tasks/CHUNK_P2_1_RESULTS.md §5 → 2D ~12 modes (Phase-1, approximate) and
    f_Schroeder ≈ 217 Hz (parsed from the file, not hardcoded).
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from aaf.sim.analytical_modal_3d import eigenfrequencies_3d

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path("/fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields")
P2_1_RESULTS = REPO / "tasks" / "CHUNK_P2_1_RESULTS.md"
OUT = REPO / "outputs" / "phase2_meeting_assets" / "02_modal_density_2d_vs_3d.png"

# Box-center room dims (Phase-2 reference room) and analysis band.
L, W, H = 4.5, 4.0, 3.25
F_MAX = 250.0

# ---------------------------------------------------------------------------
# 1) Recompute the 3D distinct-mode count from the analytical model.
# ---------------------------------------------------------------------------
eigs = eigenfrequencies_3d(L, W, H, f_max=F_MAX)
# Entries are already deduplicated/distinct. Keep 0 < f <= 250 (drop DC).
freqs_3d = np.array([e.f for e in eigs if 0.0 < e.f <= F_MAX])
freqs_3d.sort()
n_3d = int(freqs_3d.size)
print(f"[3D] eigenfrequencies_3d({L},{W},{H},f_max={F_MAX}) -> "
      f"{len(eigs)} dedup entries; distinct with 0<f<=250: {n_3d}")

# ---------------------------------------------------------------------------
# 2) Parse the 2D count (~12, Phase-1) and f_Schroeder from P2-1 results §5.
# ---------------------------------------------------------------------------
text = P2_1_RESULTS.read_text()

m2d = re.search(r"Phase-1 2D ~(\d+)\s*modes", text)
n_2d = int(m2d.group(1))
print(f"[2D] parsed Phase-1 2D modes from {P2_1_RESULTS.name}: ~{n_2d}")

msch = re.search(r"f_Schroeder\s*[≈~]\s*([0-9]+(?:\.[0-9]+)?)\s*Hz", text)
f_schroeder = float(msch.group(1))
print(f"[Schroeder] parsed f_Schroeder from {P2_1_RESULTS.name}: ~{f_schroeder} Hz")

ratio = n_3d / n_2d
print(f"[ratio] 3D/2D = {n_3d}/{n_2d} = {ratio:.2f} -> ~{round(ratio)}x")

# ---------------------------------------------------------------------------
# 3) Shared deck style.
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 19,
    "axes.labelsize": 16,
    "figure.titlesize": 24,
    "legend.fontsize": 14,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

COL_2D = "#9aa7b1"   # muted grey-blue for the easier 2D regime
COL_3D = "#d62728"   # strong red for the hard 3D modal regime

fig = plt.figure(figsize=(19.2, 10.8), dpi=100, constrained_layout=True)
ax_l, ax_r = fig.subplots(1, 2)

# ---- LEFT: grouped bar ----------------------------------------------------
labels = ["2D (Phase 1)", "3D (box-center)"]
values = [n_2d, n_3d]
colors = [COL_2D, COL_3D]
x = np.arange(len(labels))
bars = ax_l.bar(x, values, width=0.55, color=colors, edgecolor="black",
                linewidth=0.8, zorder=3)

for b, v, lab in zip(bars, values, labels):
    prefix = "~" if lab.startswith("2D") else ""
    ax_l.text(b.get_x() + b.get_width() / 2, v + n_3d * 0.015,
              f"{prefix}{v}", ha="center", va="bottom",
              fontsize=17, fontweight="bold")

# Ratio annotation spanning the two bars.
y_arrow = n_3d * 1.12
ax_l.annotate("", xy=(x[1], y_arrow), xytext=(x[0], y_arrow),
              arrowprops=dict(arrowstyle="<->", color="#333", lw=1.8))
ax_l.text(0.5, y_arrow * 1.02, f"~{round(ratio)}x",
          ha="center", va="bottom", fontsize=20, fontweight="bold",
          color=COL_3D)

ax_l.set_xticks(x)
ax_l.set_xticklabels(labels)
ax_l.set_ylabel("Distinct modes  ≤ 250 Hz")
ax_l.set_title("Modal density below 250 Hz: 2D vs 3D")
ax_l.set_ylim(0, n_3d * 1.30)
ax_l.grid(axis="y", alpha=0.3, zorder=0)
ax_l.spines["top"].set_visible(False)
ax_l.spines["right"].set_visible(False)

# ---- RIGHT: cumulative staircase of distinct 3D eigenfrequencies ----------
# Step from 0 to n_3d as frequency increases 0 -> 250 Hz.
xx = np.concatenate(([0.0], freqs_3d, [F_MAX]))
yy = np.concatenate(([0], np.arange(1, n_3d + 1), [n_3d]))
ax_r.step(xx, yy, where="post", color=COL_3D, lw=2.4, zorder=3,
          label="3D distinct eigenfrequencies")
ax_r.fill_between(xx, yy, step="post", color=COL_3D, alpha=0.12, zorder=1)

# Schroeder vertical marker.
ax_r.axvline(f_schroeder, color="#333", ls="--", lw=2.0, zorder=4)
# count of distinct 3D modes at/below Schroeder, for an honest annotation.
n_below_sch = int(np.count_nonzero(freqs_3d <= f_schroeder))
ax_r.text(f_schroeder - 4, n_3d * 0.45, f"Schroeder\n≈ {f_schroeder:.0f} Hz",
          ha="right", va="center", fontsize=14, color="#333",
          fontweight="bold")

ax_r.set_xlabel("Frequency (Hz)")
ax_r.set_ylabel("Cumulative # distinct 3D eigenfrequencies")
ax_r.set_title("3D modal accumulation, 0–250 Hz (box-center room)")
ax_r.set_xlim(0, F_MAX)
ax_r.set_ylim(0, n_3d * 1.08)
ax_r.grid(alpha=0.3, zorder=0)
ax_r.spines["top"].set_visible(False)
ax_r.spines["right"].set_visible(False)
ax_r.legend(loc="upper left", frameon=True)
ax_r.annotate(f"{n_below_sch} distinct modes\n≤ Schroeder",
              xy=(f_schroeder, n_below_sch), xytext=(f_schroeder * 0.45, n_3d * 0.86),
              fontsize=13, color=COL_3D,
              arrowprops=dict(arrowstyle="->", color=COL_3D, lw=1.5))

# ---- Suptitle + honest caption -------------------------------------------
fig.suptitle(
    f"3D rooms have ~{round(ratio)}x higher modal density below Schroeder — "
    "the modal regime is the hardest band to reconstruct",
    fontweight="bold",
)

CAPTION = (
    f"3D distinct eigenfrequencies 0<f≤250 Hz = {n_3d} (recomputed via "
    f"aaf.sim.analytical_modal_3d.eigenfrequencies_3d for box-center room "
    f"L={L}, W={W}, H={H} m). "
    f"2D count ~{n_2d} modes is an approximate Phase-1 (2D) reference from "
    f"tasks/CHUNK_P2_1_RESULTS.md §5; f_Schroeder ≈ {f_schroeder:.0f} Hz (P2-1). "
    f"Ratio {n_3d}/{n_2d} ≈ {round(ratio)}x."
)
fig.text(0.5, 0.012, CAPTION, ha="center", fontsize=12, style="italic",
         color="#444")

# ---------------------------------------------------------------------------
# 4) Save at exactly dpi=100 (no bbox_inches).
# ---------------------------------------------------------------------------
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=100)
plt.close(fig)
print(f"[saved] {OUT}")
