"""Phase-2 headliner figure: 01_latent_manifold_3d.png.

The 3D analog of Phase 1's latent plot. Loads the 45 trained 16-D per-room
latents from the M1 auto-decoder checkpoint, PCA-projects them onto their top
two principal components, and shows the SAME PC1-vs-PC2 scatter in three
side-by-side panels colored by true room L, W, H respectively. Each panel is
annotated with that axis's full-16-D linear-probe R^2.

ALL numbers are read at run time from:
  - outputs/multi_room_3d/M1_45rooms/ckpt_iter0024000.pt   (latents.weight [45,16])
  - outputs/multi_room_3d/M1_45rooms/latent_probe/latent_probe.json
    (dims_train_LWH [45,3], r2_per_axis_full {L,W,H}, pca_explained_variance [15])

No fabricated numbers: every value plotted is loaded + printed first.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CKPT = REPO_ROOT / "outputs/multi_room_3d/M1_45rooms/ckpt_iter0024000.pt"
PROBE = REPO_ROOT / "outputs/multi_room_3d/M1_45rooms/latent_probe/latent_probe.json"
OUT = REPO_ROOT / "outputs/phase2_meeting_assets/01_latent_manifold_3d.png"

# Shared deck style.
DIM_COLORS = {"L": "#1f77b4", "W": "#ff7f0e", "H": "#2ca02c"}


def main() -> int:
    # ---- Load sources -------------------------------------------------------
    state = torch.load(CKPT, map_location="cpu")
    latents = state["model"]["latents.weight"].detach().cpu().numpy().astype(float)
    print(f"# loaded latents.weight from {CKPT}")
    print(f"#   latents shape = {latents.shape}")
    assert latents.shape == (45, 16), f"unexpected latent shape {latents.shape}"

    probe = json.loads(PROBE.read_text())
    dims = np.asarray(probe["dims_train_LWH"], dtype=float)  # [45,3] L,W,H
    print(f"# loaded latent_probe.json from {PROBE}")
    print(f"#   dims_train_LWH shape = {dims.shape}")
    assert dims.shape == (45, 3), f"unexpected dims shape {dims.shape}"

    r2 = probe["r2_per_axis_full"]
    r2_L = float(r2["L"])
    r2_W = float(r2["W"])
    r2_H = float(r2["H"])
    print(f"# r2_per_axis_full: L={r2_L:.6f}  W={r2_W:.6f}  H={r2_H:.6f}")

    pca_ev = np.asarray(probe["pca_explained_variance"], dtype=float)  # [15]
    print(f"# pca_explained_variance length = {len(pca_ev)}")
    print(f"#   stored EV top2 (fraction) = {pca_ev[0]:.6f}, {pca_ev[1]:.6f}")

    # ---- PCA the 16-D latents ourselves (standardize/center -> SVD) ---------
    # Center (and standardize each dim) the 45x16 latents, then SVD for PCs.
    X = latents - latents.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=0, keepdims=True)
    std[std == 0] = 1.0
    Xs = X / std  # standardized
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    scores = U * S  # 45 x 16 projection onto PCs
    pc1 = scores[:, 0]
    pc2 = scores[:, 1]
    # Explained variance ratio recomputed from our own SVD (cross-check).
    var = (S ** 2)
    ev_ratio = var / var.sum()
    print(f"#   recomputed EV ratio top2 (standardized SVD) = "
          f"{ev_ratio[0]:.6f}, {ev_ratio[1]:.6f}")

    # Use the stored probe EV for the axis labels (that is the canonical source
    # the brief points to); print both so any discrepancy is visible.
    ev1_pct = pca_ev[0] * 100.0
    ev2_pct = pca_ev[1] * 100.0
    print(f"# axis labels use stored EV: PC1={ev1_pct:.1f}% var  PC2={ev2_pct:.1f}% var")

    L = dims[:, 0]
    W = dims[:, 1]
    H = dims[:, 2]

    # ---- rcParams (shared deck style) --------------------------------------
    plt.rcParams.update({
        "font.size": 15,
        "axes.titlesize": 19,
        "axes.labelsize": 16,
        "figure.titlesize": 24,
        "legend.fontsize": 14,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })

    fig, axes = plt.subplots(1, 3, figsize=(19.2, 10.8), dpi=100,
                             constrained_layout=True)

    panels = [
        ("L", L, "L (m)", r2_L),
        ("W", W, "W (m)", r2_W),
        ("H", H, "H (m)", r2_H),
    ]

    for ax, (name, vals, cbar_label, r2v) in zip(axes, panels):
        sc = ax.scatter(pc1, pc2, c=vals, cmap="viridis", s=140,
                        edgecolors="black", linewidths=0.7, zorder=3)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(cbar_label, fontsize=15)
        ax.set_xlabel(f"PC1 ({ev1_pct:.1f}% var)")
        ax.set_ylabel(f"PC2 ({ev2_pct:.1f}% var)")
        ax.set_title(f"Colored by {name}", color=DIM_COLORS[name], weight="bold")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # R^2 annotation, upper-left text box.
        ax.text(0.04, 0.96, f"linear probe (16-D)\n{name}  R² = {r2v:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=15,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                          edgecolor=DIM_COLORS[name], linewidth=1.8, alpha=0.95))

    fig.suptitle("The latent space autonomously encodes all three room dimensions",
                 weight="bold")

    caption = (
        "45 per-room latents (16-D) from the M1 auto-decoder, projected onto their top two principal "
        "components; color = true room dimension. A linear probe on the full 16-D latent recovers each "
        "dimension with the R² shown. Source: outputs/multi_room_3d/M1_45rooms "
        "(latent_probe.json + checkpoint)."
    )
    fig.text(0.5, 0.012, caption, ha="center", fontsize=12, style="italic",
             color="#444")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=100)
    plt.close(fig)
    print(f"# wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
