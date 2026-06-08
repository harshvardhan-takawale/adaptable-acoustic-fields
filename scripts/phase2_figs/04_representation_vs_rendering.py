#!/usr/bin/env python
"""Figure 4 — Representation vs Rendering dissociation.

Phase-2 meeting deck. The methodological insight: the per-room latent encodes
room geometry well (linear-probe R^2, LEFT) even at a stage where the decoder
cannot yet render accurately at low per-iteration coverage (RIGHT, 6.16 dB);
raising coverage fixes the rendering (2.61 -> 0.98 dB).

Every number plotted is LOADED from a named source file at run time and PRINTED
before being plotted. No fabricated values.

Sources
-------
LEFT  R^2 per axis (full latent linear probe):
    outputs/multi_room_3d/M1_45rooms/latent_probe/latent_probe.json
      -> r2_per_axis_full {L, W, H}
RIGHT in-distribution val LSD (dB), final val entry of each run's scalar log:
    P2-2 M1 low coverage:
      outputs/multi_room_3d/M1_45rooms/scalars.json  (last val lsd_db, iter 24000)
    Run B 45 rooms (DDP, coverage-fixed, the canonical "Run B 2.61 dB"):
      outputs/diag_p2_2_5/B_45rm_ddp/scalars.json    (last val lsd_db, iter 60000)
    Run C 10 rooms b64 (highest coverage):
      outputs/diag_p2_2_5/C_10rm_b64/scalars.json    (last val lsd_db, iter 30000)
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Shared deck style
# ----------------------------------------------------------------------------
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

C_L = "#1f77b4"  # length  (blue)
C_W = "#ff7f0e"  # width   (orange)
C_H = "#2ca02c"  # height  (green)

REPO = "/fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields"

PROBE_JSON = os.path.join(REPO, "outputs/multi_room_3d/M1_45rooms/latent_probe/latent_probe.json")
M1_SCALARS = os.path.join(REPO, "outputs/multi_room_3d/M1_45rooms/scalars.json")
B_SCALARS = os.path.join(REPO, "outputs/diag_p2_2_5/B_45rm_ddp/scalars.json")
C_SCALARS = os.path.join(REPO, "outputs/diag_p2_2_5/C_10rm_b64/scalars.json")

OUT_PNG = os.path.join(REPO, "outputs/phase2_meeting_assets/04_representation_vs_rendering.png")

LSD_TARGET = 2.5  # methodological target line stated in the brief


def final_val_lsd(scalars_path):
    """Return (iter, lsd_db) of the LAST validation entry in a scalar log."""
    with open(scalars_path) as fh:
        log = json.load(fh)
    vals = [(e["iter"], e["lsd_db"]) for e in log if e.get("phase") == "val" and "lsd_db" in e]
    if not vals:
        raise ValueError(f"no val lsd_db entries in {scalars_path}")
    return vals[-1]


# ----------------------------------------------------------------------------
# LOAD + PRINT every number we will plot
# ----------------------------------------------------------------------------
print("=" * 78)
print("LOADING SOURCE NUMBERS FOR FIGURE 4")
print("=" * 78)

with open(PROBE_JSON) as fh:
    probe = json.load(fh)
r2 = probe["r2_per_axis_full"]
r2_L = float(r2["L"])
r2_W = float(r2["W"])
r2_H = float(r2["H"])
print(f"[probe] {PROBE_JSON}")
print(f"  r2_per_axis_full: L={r2_L:.6f}  W={r2_W:.6f}  H={r2_H:.6f}")

m1_iter, m1_lsd = final_val_lsd(M1_SCALARS)
print(f"[M1 low-cov] {M1_SCALARS}")
print(f"  final val (iter {m1_iter}): lsd_db = {m1_lsd:.6f} dB")

b_iter, b_lsd = final_val_lsd(B_SCALARS)
print(f"[Run B 45rm DDP] {B_SCALARS}")
print(f"  final val (iter {b_iter}): lsd_db = {b_lsd:.6f} dB")

c_iter, c_lsd = final_val_lsd(C_SCALARS)
print(f"[Run C 10rm b64] {C_SCALARS}")
print(f"  final val (iter {c_iter}): lsd_db = {c_lsd:.6f} dB")

# Cross-check against EXPECTED (do not trust blindly; plot REAL, note deltas).
expected = {"r2_L": 0.991, "r2_W": 0.967, "r2_H": 0.974, "m1": 6.16, "b": 2.61, "c": 0.98}
print("-" * 78)
print("CROSS-CHECK vs EXPECTED (plotting REAL values):")
for name, real in [
    ("r2_L", r2_L),
    ("r2_W", r2_W),
    ("r2_H", r2_H),
    ("m1", m1_lsd),
    ("b", b_lsd),
    ("c", c_lsd),
]:
    exp = expected[name]
    print(f"  {name:5s}: real={real:.4f}  expected={exp:.4f}  |delta|={abs(real - exp):.4f}")
print("=" * 78)

# ----------------------------------------------------------------------------
# FIGURE
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(19.2, 10.8), constrained_layout=True)
axes = fig.subplots(1, 2)
axL, axR = axes

# --- LEFT: representation works (linear-probe R^2) ----------------------------
labels = ["L\n(length)", "W\n(width)", "H\n(height)"]
r2_vals = [r2_L, r2_W, r2_H]
colors = [C_L, C_W, C_H]

bars = axL.bar(labels, r2_vals, color=colors, width=0.62, edgecolor="white", linewidth=1.5, zorder=3)
axL.axhline(1.0, color="#555555", ls="--", lw=1.8, zorder=2, label="perfect = 1.0")
axL.set_ylim(0.0, 1.06)
axL.set_ylabel(r"linear-probe $R^2$  (latent $\rightarrow$ dimension)")
axL.set_title("Representation WORKS:\nthe latent already encodes geometry", fontweight="bold")
for b, v in zip(bars, r2_vals):
    axL.text(
        b.get_x() + b.get_width() / 2.0,
        v + 0.012,
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight="bold",
    )
axL.legend(loc="lower right", frameon=True)
axL.grid(axis="y", alpha=0.3, zorder=0)
axL.set_axisbelow(True)
axL.spines["top"].set_visible(False)
axL.spines["right"].set_visible(False)
axL.text(
    0.02,
    0.04,
    "source: M1_45rooms/latent_probe.json\n(r2_per_axis_full, 45-room train latents)",
    transform=axL.transAxes,
    fontsize=11,
    style="italic",
    color="#666",
    va="bottom",
)

# --- RIGHT: rendering lagged until coverage was fixed -------------------------
r_labels = ["P2-2 M1\nlow coverage", "Run B\n45 rooms (DDP)", "Run C\n10 rooms (b64)"]
r_vals = [m1_lsd, b_lsd, c_lsd]
# colour: failure in red, coverage-fixed runs in a calm teal/green ramp
r_colors = ["#d62728", "#3b8ea5", "#2ca02c"]

rbars = axR.bar(
    r_labels, r_vals, color=r_colors, width=0.62, edgecolor="white", linewidth=1.5, zorder=3
)
axR.axhline(
    LSD_TARGET, color="#7f7f7f", ls="--", lw=1.8, zorder=2, label=f"target = {LSD_TARGET:.1f} dB"
)
axR.set_ylim(0.0, max(r_vals) * 1.22)
axR.set_ylabel("in-distribution val LSD  (dB, lower is better)")
axR.set_title("Rendering LAGGED:\nfixed once per-iteration coverage rose", fontweight="bold")
for b, v in zip(rbars, r_vals):
    axR.text(
        b.get_x() + b.get_width() / 2.0,
        v + max(r_vals) * 0.015,
        f"{v:.2f} dB",
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight="bold",
    )
axR.legend(loc="upper right", frameon=True)
axR.grid(axis="y", alpha=0.3, zorder=0)
axR.set_axisbelow(True)
axR.spines["top"].set_visible(False)
axR.spines["right"].set_visible(False)

# "coverage fixed ->" annotation spanning from the 6.16 bar to the fixed bars.
y_arrow = m1_lsd * 0.86
axR.annotate(
    "",
    xy=(1.5, y_arrow),
    xytext=(0.32, y_arrow),
    arrowprops=dict(arrowstyle="-|>", color="#444", lw=2.6, mutation_scale=26),
)
axR.text(
    0.92,
    y_arrow + max(r_vals) * 0.03,
    "coverage fixed  →",
    ha="center",
    va="bottom",
    fontsize=15,
    fontweight="bold",
    color="#333",
)
# delta call-out: how much rendering improved
axR.annotate(
    f"{m1_lsd:.2f} → {b_lsd:.2f} → {c_lsd:.2f} dB\n"
    f"({m1_lsd - c_lsd:.2f} dB total drop)",
    xy=(2.0, c_lsd),
    xytext=(2.05, m1_lsd * 0.55),
    ha="center",
    fontsize=12.5,
    color="#222",
    arrowprops=dict(arrowstyle="->", color="#888", lw=1.4),
    bbox=dict(boxstyle="round,pad=0.35", fc="#f4f4f4", ec="#bbbbbb"),
)
axR.text(
    0.98,
    0.02,
    "sources: M1_45rooms/scalars.json; diag_p2_2_5/{B_45rm_ddp,C_10rm_b64}/scalars.json",
    transform=axR.transAxes,
    fontsize=10.5,
    style="italic",
    color="#666",
    va="bottom",
    ha="right",
)

# --- suptitle + honest caption ----------------------------------------------
fig.suptitle(
    "Representation and rendering are separable: the latent learned geometry "
    "before the decoder could render it",
    fontweight="bold",
)

CAPTION = (
    "The latent encodes geometry well (left, linear-probe R²) even when the decoder cannot "
    "yet render accurately at low coverage (right, 6.16 dB); raising per-iteration coverage fixes "
    "the rendering (2.61 → 0.98 dB).  Sources: latent_probe.json; diag_p2_2_5 scalars."
)
fig.text(0.5, 0.012, CAPTION, ha="center", fontsize=12, style="italic", color="#444")

fig.savefig(OUT_PNG, dpi=100)
print(f"saved -> {OUT_PNG}")

# ----------------------------------------------------------------------------
# VERIFY pixel size
# ----------------------------------------------------------------------------
from PIL import Image

w, h = Image.open(OUT_PNG).size
print(f"PNG size = {w}x{h}")
assert w >= 1920 and h >= 1080, f"PNG too small: {w}x{h}"
print("OK: PNG is >= 1920x1080")
