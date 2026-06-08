#!/usr/bin/env python
"""Figure 3 — 03_diagnostic_convergence.png.

The controlled diagnostic that isolated the Phase-2 bottleneck: three runs varying
ONLY the number of rooms and effective batch (coverage), plotted as val LSD (dB) vs
iteration. Shows that coverage, not model capacity, was the wall.

Every plotted number is READ FROM SOURCE at run time (load + print, then plot).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----- shared deck style -----
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

REPO = "/fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields"

# ----- source files -----
SRC = {
    "A": os.path.join(REPO, "outputs/diag_p2_2_5/A_10rm_b16/scalars.json"),
    "B": os.path.join(REPO, "outputs/diag_p2_2_5/B_45rm_ddp/scalars.json"),
    "C": os.path.join(REPO, "outputs/diag_p2_2_5/C_10rm_b64/scalars.json"),
    "M1": os.path.join(REPO, "outputs/multi_room_3d/M1_45rooms/scalars.json"),
}


def load_val(path):
    """Return (iters, lsd_db) lists for phase==val rows that carry lsd_db."""
    rows = json.load(open(path))
    val = [r for r in rows if r.get("phase") == "val" and "lsd_db" in r]
    iters = [r["iter"] for r in val]
    lsd = [r["lsd_db"] for r in val]
    return iters, lsd


# ----- load + PRINT every number we will plot -----
curves = {}
for k in ("A", "B", "C"):
    it, lsd = load_val(SRC[k])
    curves[k] = (it, lsd)
    print(f"[{k}] n_val={len(it)}  first=({it[0]}, {lsd[0]:.4f})  "
          f"final=({it[-1]}, {lsd[-1]:.4f})")

# M1 plateau: read its final val LSD to confirm ~6.16
m1_it, m1_lsd = load_val(SRC["M1"])
M1_PLATEAU = m1_lsd[-1]
print(f"[M1] n_val={len(m1_it)}  final=({m1_it[-1]}, {M1_PLATEAU:.4f})  "
      f"-> P2-2 M1 plateau reference")

TARGET = 2.5
print(f"[target] success target = {TARGET} dB (brief constant)")

A_FINAL = curves["A"][1][-1]
B_FINAL = curves["B"][1][-1]
C_FINAL = curves["C"][1][-1]
print(f"FINALS: A={A_FINAL:.4f}  B={B_FINAL:.4f}  C={C_FINAL:.4f}  "
      f"M1={M1_PLATEAU:.4f}  target={TARGET}")

# ----- colors -----
COL_A = "#1f77b4"  # blue
COL_B = "#9467bd"  # purple
COL_C = "#2ca02c"  # green
COL_M1 = "#d62728"  # red (plateau, the wall)
COL_TGT = "#555555"  # grey dashed target

# ----- build figure -----
fig = plt.figure(figsize=(19.2, 10.8), dpi=100, constrained_layout=True)
ax = fig.add_subplot(111)

ax.plot(
    curves["A"][0], curves["A"][1], color=COL_A, lw=2.6, marker="o", ms=5,
    label=f"Run A — 10 rm, eff-batch 16 ({A_FINAL:.2f} dB)",
)
ax.plot(
    curves["B"][0], curves["B"][1], color=COL_B, lw=2.6, marker="s", ms=5,
    label=f"Run B — 45 rm, eff-batch 32, DDP ({B_FINAL:.2f} dB)",
)
ax.plot(
    curves["C"][0], curves["C"][1], color=COL_C, lw=2.8, marker="D", ms=5,
    label=f"Run C — 10 rm, high cov, eff-batch 64 ({C_FINAL:.2f} dB)",
)

# horizontal references
ax.axhline(M1_PLATEAU, color=COL_M1, lw=2.2, ls="-", alpha=0.9)
ax.text(
    curves["B"][0][-1], M1_PLATEAU + 0.12,
    f"P2-2 M1 plateau (45 rm, low coverage) = {M1_PLATEAU:.2f} dB",
    color=COL_M1, fontsize=14, fontweight="bold", ha="right", va="bottom",
)

ax.axhline(TARGET, color=COL_TGT, lw=2.0, ls="--", alpha=0.9)
ax.text(
    curves["B"][0][-1], TARGET + 0.12, f"target = {TARGET:.1f} dB",
    color=COL_TGT, fontsize=14, fontweight="bold", ha="right", va="bottom",
)

# annotate near Run C's tail
c_x, c_y = curves["C"][0][-1], curves["C"][1][-1]
ax.annotate(
    "Capacity is not the wall —\n10 rooms at high coverage → ~1 dB",
    xy=(c_x, c_y),
    xytext=(c_x * 0.62, c_y + 2.0),
    color=COL_C, fontsize=15, fontweight="bold", ha="center", va="bottom",
    arrowprops=dict(arrowstyle="->", color=COL_C, lw=2.0),
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=COL_C, alpha=0.9),
)

ax.set_xlabel("training iteration")
ax.set_ylabel("validation log-spectral distance (dB)")
ax.set_title("Validation LSD vs iteration — three coverage-controlled runs")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(M1_PLATEAU, max(curves["A"][1]), max(curves["B"][1]),
                   max(curves["C"][1])) + 1.2)
ax.set_xlim(0, max(curves["A"][0][-1], curves["B"][0][-1], curves["C"][0][-1]))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", framealpha=0.95)

fig.suptitle(
    "Controlled diagnostic: coverage, not capacity, was the bottleneck",
    fontweight="bold",
)

CAPTION = (
    f"Val LSD vs iter for three coverage-controlled diagnostic runs. "
    f"Run A: 10 rooms, eff-batch 16 → {A_FINAL:.2f} dB "
    f"(A_10rm_b16/scalars.json). "
    f"Run B: 45 rooms, eff-batch 32, converged DDP → {B_FINAL:.2f} dB "
    f"(B_45rm_ddp/scalars.json). "
    f"Run C: 10 rooms, high coverage, eff-batch 64 → {C_FINAL:.2f} dB "
    f"(C_10rm_b64/scalars.json). "
    f"P2-2 M1 plateau {M1_PLATEAU:.2f} dB from outputs/multi_room_3d/M1_45rooms/"
    f"scalars.json (45 rm, low coverage); target {TARGET:.1f} dB."
)
fig.text(0.5, 0.012, CAPTION, ha="center", fontsize=12, style="italic",
         color="#444", wrap=True)

OUT = os.path.join(REPO, "outputs/phase2_meeting_assets/03_diagnostic_convergence.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=100)
print(f"SAVED {OUT}")

from PIL import Image

w, h = Image.open(OUT).size
print(f"PNG size = {w}x{h}")
assert w >= 1920 and h >= 1080, f"PNG too small: {w}x{h}"
print("SIZE OK (>=1920x1080)")
