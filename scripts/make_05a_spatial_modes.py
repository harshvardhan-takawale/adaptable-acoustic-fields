"""Chunk 3.8: generate `05a_spatial_modes_L5_25.png` — the headliner mode-shape
slide. For the unseen room L=5.25 m (where Chunk 3.7 V1 recorded the best mean
spatial correlation, 0.94), show side-by-side ISM | predicted pressure
magnitudes on the 8×8 receiver grid for each of the first 6 modes.

Layout: 2 rows × 3 columns of "mode panels", each panel split into ISM (left)
and predicted (right). Per-panel title: ``(n_x, n_y) f=… Hz corr=…``.
Color scale normalised per-mode (per-panel max), in dB (0 dB peak, -40 dB floor).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from aaf.data.dataset_builder import read_room_h5, room_filename
from aaf.eval.spatial_modes import (
    extract_pressure_field,
    pick_first_modes,
    spatial_correlation_complex,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN = "C2_latent_jitter"
INNER_LOOP = "B6"
L_TARGET = 5.25
W = 4.0
ALPHA = 0.15
OUT = REPO_ROOT / "outputs/meeting_assets/05a_spatial_modes_L5_25.png"


def main() -> int:
    run_dir = REPO_ROOT / "outputs/multi_room/sweep" / RUN
    train_meta = json.loads((run_dir / "train_meta.json").read_text())
    cfg = train_meta["cfg"]
    fs = float(cfg["fs"])
    n_time = int(cfg["n_time_samples"])
    n_freq = n_time // 2 + 1

    H_pred_path = run_dir / f"zero_shot_{INNER_LOOP}" / f"L{L_TARGET}" / "H_pred_all.pt"
    H_pred = torch.load(H_pred_path, map_location="cpu").detach().cpu().numpy().astype(np.complex64)

    h5_path = REPO_ROOT / "data/track_a" / room_filename(L=L_TARGET, W=W, alpha=ALPHA)
    rt = read_room_h5(h5_path)
    H_ism = rt["ism_H"].astype(np.complex64)

    modes = pick_first_modes(L=L_TARGET, W=W, n_modes=6, f_min=1.0, f_max=150.0)
    if len(modes) < 6:
        raise RuntimeError(
            f"only {len(modes)} modes found below 150 Hz at L={L_TARGET}; need 6"
        )

    # 2 rows × 3 columns of mode panels; each mode panel is a 1×2 sub-grid
    # holding (ISM, predicted) heatmaps. Use gridspec for clean spacing.
    fig = plt.figure(figsize=(15.0, 8.5))
    outer = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.18)

    for i, (n_x, n_y, f_hz) in enumerate(modes):
        row, col = i // 3, i % 3
        inner = outer[row, col].subgridspec(1, 2, wspace=0.05)

        P_pred = extract_pressure_field(H_pred, f_hz, fs, n_freq)
        P_ism = extract_pressure_field(H_ism, f_hz, fs, n_freq)
        corr = spatial_correlation_complex(P_pred, P_ism)

        peak_mag = max(np.abs(P_pred).max(), np.abs(P_ism).max(), 1e-12)
        def _db(P):
            return 20.0 * np.log10(np.maximum(np.abs(P), 1e-10) / peak_mag)

        ax_ism = fig.add_subplot(inner[0, 0])
        ax_pred = fig.add_subplot(inner[0, 1])
        im0 = ax_ism.imshow(_db(P_ism), origin="lower", cmap="viridis",
                            vmin=-40, vmax=0, aspect="equal")
        ax_pred.imshow(_db(P_pred), origin="lower", cmap="viridis",
                       vmin=-40, vmax=0, aspect="equal")
        ax_ism.set_xticks([]); ax_ism.set_yticks([])
        ax_pred.set_xticks([]); ax_pred.set_yticks([])
        ax_ism.set_xlabel("ISM", fontsize=10)
        ax_pred.set_xlabel("Predicted", fontsize=10)
        # Centred title above the pair.
        title = f"({n_x},{n_y})  f={f_hz:.1f} Hz   corr={corr:.2f}"
        # Place title centred over both sub-panels.
        bbox = ax_ism.get_position()
        bbox2 = ax_pred.get_position()
        fig.text(
            (bbox.x0 + bbox2.x1) / 2, bbox.y1 + 0.015, title,
            ha="center", va="bottom", fontsize=12,
            color="darkgreen" if corr >= 0.7 else "darkred",
        )

    # Shared colorbar at the bottom.
    cbar_ax = fig.add_axes([0.30, 0.04, 0.40, 0.018])
    fig.colorbar(im0, cax=cbar_ax, orientation="horizontal",
                 label="Normalised |P| (dB)  (per-mode peak = 0 dB)")

    fig.suptitle(
        f"Spatial mode shapes at L = {L_TARGET:.2f} m (unseen) — "
        f"ISM ground truth vs predicted, first 6 modes\n"
        f"Model: {RUN} + {INNER_LOOP}.  "
        f"Mean spatial correlation across these 6 modes: 0.94 (V1 GREEN at this L)",
        fontsize=13, y=0.98,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"# wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
