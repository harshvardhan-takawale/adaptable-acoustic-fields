"""Per-3D-room evaluator: traditional metrics + signal-level eval suite.

3D analogue of `aaf.eval.single_room_eval`. Same structure (modal + full-band
metrics + figures), plus the new ``aaf.eval.signal_level.compute_signal_metrics``
+ ``make_signal_plots`` calls (Dolby-requested).

Modal MAE is reported only within f < f_Schroeder (DECISIONS.md D18): above
f_Schroeder, 3D modal density exceeds the RFFT resolution Δf = 0.5 Hz.

CLI
---
    python -m aaf.eval.single_room_3d_eval --L 4.5 --W 4.0 --H 3.25 \
        --rooms-yaml configs/sweeps_3d/derisk_rooms.yaml \
        --output_dir outputs/single_room_3d/L4.50_W4.00_H3.25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from aaf.data.loader import Shoebox3DDataset
from aaf.eval.modal_verifier import (
    pick_peaks,
    plot_modal_overlay,
    modal_error_metrics,
)
from aaf.eval.signal_level import (
    DEFAULT_BANDS,
    compute_signal_metrics,
    make_signal_plots,
)
from aaf.models.inr_3d import INR3D_Single
from aaf.renderers.freq_3d import FreqRenderer3D
from aaf.sim.analytical_modal_3d import eigenfrequencies_3d, modal_rir_3d


C_DEFAULT = 343.0


def _load_latest_ckpt(output_dir: Path) -> tuple[Path, dict]:
    ckpts = sorted(
        output_dir.glob("ckpt_iter*.pt"),
        key=lambda p: int(p.stem.split("ckpt_iter")[-1]),
        reverse=True,
    )
    for p in ckpts:
        try:
            state = torch.load(p, map_location="cuda")
            return p, state
        except Exception:
            continue
    raise FileNotFoundError(f"no usable ckpt in {output_dir}")


def _representative_receiver_idx(
    receiver_pos: np.ndarray, L: float, W: float, H: float
) -> int:
    centre = np.array([L / 2.0, W / 2.0, H / 2.0])
    return int(np.argmin(np.linalg.norm(receiver_pos - centre[None, :], axis=1)))


def evaluate_single_room_3d(
    L: float, W: float, H: float,
    rooms_yaml: str, output_dir: str, device: str = "cuda",
) -> dict:
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    dataset = Shoebox3DDataset(
        rooms_yaml=rooms_yaml, room_filter=[(L, W, H)]
    )
    n_freq = dataset.n_freq_bins
    n_time = dataset.n_time_samples
    fs = dataset.fs

    items = [dataset[i] for i in range(len(dataset))]
    rx_pos = np.stack([it["rx_pos"].numpy() for it in items])         # [N_rx, 3]
    tx_pos = np.stack([it["tx_pos"].numpy() for it in items])         # [N_rx, 3]
    H_target = np.stack([it["H_complex"].numpy() for it in items])    # [N_rx, n_freq]
    rir_target = np.stack([it["rir_time"].numpy() for it in items])   # [N_rx, n_time]
    n_rx = rx_pos.shape[0]

    f_axis = np.arange(n_freq) * (fs / n_time)

    # Recreate renderer with the training-time cfg.
    train_meta_path = output_dir / "train_meta.json"
    if train_meta_path.exists():
        train_cfg = json.loads(train_meta_path.read_text())["cfg"]
        n_azi = int(train_cfg.get("n_azi", 16))
        n_ele = int(train_cfg.get("n_ele", 16))
        n_pts_per_ray = int(train_cfg.get("n_pts_per_ray", 32))
    else:
        n_azi, n_ele, n_pts_per_ray = 16, 16, 32
    renderer = FreqRenderer3D(
        n_azi=n_azi, n_ele=n_ele, n_pts_per_ray=n_pts_per_ray, near=1e-3,
        fs=fs, n_time_samples=n_time, c=C_DEFAULT, use_geometric_attn=False,
    ).to(device).eval()

    model = INR3D_Single(n_freq_bins=n_freq).to(device).eval()
    ckpt_path, state = _load_latest_ckpt(output_dir)
    model.load_state_dict(state["model"])
    model.eval()

    # Forward all receivers in chunks (avoid OOM at 512 receivers).
    chunk = 8
    H_pred_chunks = []
    rx_t = torch.from_numpy(rx_pos).to(device)
    tx_t = torch.from_numpy(tx_pos).to(device)
    room_min = torch.tensor([0.0, 0.0, 0.0], device=device)
    room_max = torch.tensor([float(L), float(W), float(H)], device=device)
    with torch.no_grad():
        for s in range(0, rx_t.size(0), chunk):
            H_pred_chunks.append(
                renderer(model, rx_t[s:s+chunk], tx_t[s:s+chunk],
                         room_min, room_max).cpu().numpy()
            )
    H_pred = np.concatenate(H_pred_chunks, axis=0).astype(np.complex64)
    rir_pred = np.fft.irfft(H_pred, n=n_time, axis=-1).astype(np.float32)

    # ---------------- Frequency-domain summary ----------------
    eps = 1e-8
    mag_pred = np.abs(H_pred)
    mag_target = np.abs(H_target)
    lsd_db = float(np.mean(np.abs(20 * np.log10(
        np.maximum(mag_pred, eps) / np.maximum(mag_target, eps)))))
    complex_l1 = float(np.mean(np.abs(H_pred - H_target)))
    magnitude_l1 = float(np.mean(np.abs(mag_pred - mag_target)))
    phase_diff = np.angle(H_pred) - np.angle(H_target)
    phase_wrapped = np.minimum(np.abs(phase_diff), 2 * np.pi - np.abs(phase_diff))
    phase_l1 = float(np.mean(phase_wrapped))

    # ---------------- Signal-level eval (Dolby-requested suite) ----------------
    signal_metrics = compute_signal_metrics(
        H_pred, H_target, fs=fs, n_time_samples=n_time,
        bands=DEFAULT_BANDS, early_late_split_ms=50.0,
        rir_pred=rir_pred, rir_target=rir_target,
    )

    # ---------------- Modal metrics (within f_Schroeder only; D18) ----------------
    rep = _representative_receiver_idx(rx_pos, L, W, H)
    attrs = dataset.get_room_attrs(0)
    f_schroeder = float(attrs.get("schroeder_freq_hz", 200.0))
    # Hard upper bound on the modal-MAE band: never go above f_Schroeder, never
    # below 100 Hz (need at least ~10-15 modes to pick).
    f_modal_cap = max(100.0, min(f_schroeder, 250.0))
    f_max_plot = 2000.0
    f_mask_full = (f_axis >= 0) & (f_axis <= f_max_plot)
    f_mask_modal = (f_axis >= 0) & (f_axis <= f_modal_cap)

    modes_full = [
        m for m in eigenfrequencies_3d(L=L, W=W, H=H, c=C_DEFAULT, f_max=f_max_plot)
        if m.f > 0
    ]
    modes_modal = [m for m in modes_full if m.f <= f_modal_cap]

    peaks_pred_full = pick_peaks(
        H_pred[rep, f_mask_full], f_axis[f_mask_full],
        prominence_db=3.0, min_distance_hz=10.0,
    )
    peaks_pred_modal = [p for p in peaks_pred_full if p.f <= f_modal_cap]
    peaks_ism_full = pick_peaks(
        H_target[rep, f_mask_full], f_axis[f_mask_full],
        prominence_db=3.0, min_distance_hz=10.0,
    )
    peaks_ism_modal = [p for p in peaks_ism_full if p.f <= f_modal_cap]

    metrics_modal = modal_error_metrics(
        peaks_pred_modal, modes_modal, tolerance_hz=4.0, tolerance_pct=0.02
    )

    eval_dict = {
        "L": float(L), "W": float(W), "H": float(H),
        "alpha": float(dataset.alpha),
        "fs": int(fs), "n_freq_bins": int(n_freq), "n_time_samples": int(n_time),
        "ckpt_path": str(ckpt_path.relative_to(output_dir.parent.parent)),
        "ckpt_iter": int(state["iter"]),
        "f_schroeder_hz": f_schroeder,
        "f_modal_cap_hz": f_modal_cap,
        "n_receivers": int(n_rx),
        "n_modes_total_below_2k": len(modes_full),
        "n_modes_modal_band": len(modes_modal),

        "modal": {
            "mae_hz": metrics_modal["mae_hz"],
            "recall_at_tol": metrics_modal["recall_at_tol"],
            "n_picked": metrics_modal["n_picked"],
            "n_analytical": metrics_modal["n_analytical"],
            "n_matched": metrics_modal["n_matched"],
            "n_spurious": metrics_modal["n_spurious"],
            "per_band": metrics_modal["per_mode_breakdown"],
        },
        "full_band": {
            "lsd_db": lsd_db,
            "complex_l1": complex_l1,
            "magnitude_l1": magnitude_l1,
            "phase_l1": phase_l1,
        },
        "signal_metrics": signal_metrics,
        "ism_modal_recall_at_centre": modal_error_metrics(
            peaks_ism_modal, modes_modal, tolerance_hz=4.0, tolerance_pct=0.02
        )["recall_at_tol"],
    }
    (output_dir / "eval.json").write_text(json.dumps(eval_dict, indent=2))

    # ---------------- Figures ----------------
    # 1. training curves (from scalars.json)
    scalars_path = output_dir / "scalars.json"
    if scalars_path.exists():
        scalars = json.loads(scalars_path.read_text())
        train_rows = [r for r in scalars if r.get("phase") == "train"]
        val_rows = [r for r in scalars if r.get("phase") == "val"]
        fig, axs = plt.subplots(2, 4, figsize=(14, 6))
        loss_keys = ["L_spec_real", "L_spec_imag", "L_amp", "L_phase"]
        for ax, k in zip(axs[0], loss_keys):
            ts = [r["iter"] for r in train_rows if k in r]
            vs = [r[k] for r in train_rows if k in r]
            ax.semilogy(ts, vs, color="steelblue", lw=0.5, label="train")
            if val_rows:
                ts_v = [r["iter"] for r in val_rows if k in r]
                vs_v = [r[k] for r in val_rows if k in r]
                ax.semilogy(ts_v, vs_v, "o-", color="indianred", ms=3, label="val")
            ax.set_xlabel("iter")
            ax.set_title(k)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        metric_keys = ["lsd_db", "complex_l1", "phase_l1"]
        for ax, k in zip(axs[1], metric_keys):
            if val_rows:
                ts_v = [r["iter"] for r in val_rows if k in r]
                vs_v = [r[k] for r in val_rows if k in r]
                ax.plot(ts_v, vs_v, "o-", color="indianred", ms=3)
            ax.set_xlabel("iter")
            ax.set_title(f"val {k}")
            ax.grid(True, alpha=0.3)
        axs[1, 3].axis("off")
        fig.suptitle(f"L={L:.2f} W={W:.2f} H={H:.2f} m — training curves (iter={state['iter']})")
        fig.tight_layout()
        fig.savefig(fig_dir / "training_curves.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    # 2. modal tracking (predicted |H| with analytical modes overlay)
    fig, ax = plt.subplots(figsize=(11, 4))
    plot_modal_overlay(
        H_pred[rep], f_axis, modes_full, peaks_pred_full, ax,
        title=f"L={L:.2f} W={W:.2f} H={H:.2f} — predicted |H(f)| (centre rx)",
        f_min=0, f_max=int(f_modal_cap * 1.5), db_floor=-100,
    )
    # Overlay ISM peaks as blue triangles for reference.
    for p in peaks_ism_full:
        if p.f <= f_modal_cap * 1.5:
            ax.plot(p.f, p.magnitude_db, "v", color="tab:blue", ms=5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(fig_dir / "modal_tracking.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 3. spectrum overlay (predicted vs ISM vs analytical-modal-sum, full band)
    cfg_modal = {
        "L": L, "W": W, "H": H,
        "source_pos": tuple(tx_pos[0]),
        "receiver_pos": rx_pos[rep:rep+1],
        "alpha": dataset.alpha,
        "fs": fs, "n_time_samples": n_time,
        "f_max_modes": f_max_plot,
    }
    ana = modal_rir_3d(cfg_modal)
    H_ana = ana["H_complex"][0]

    fig, ax = plt.subplots(figsize=(11, 4))
    mag_db_pred = 20 * np.log10(np.maximum(np.abs(H_pred[rep]), eps))
    mag_db_ism = 20 * np.log10(np.maximum(np.abs(H_target[rep]), eps))
    mag_db_ana = 20 * np.log10(np.maximum(np.abs(H_ana), eps))
    ax.plot(f_axis, mag_db_ism, color="steelblue", lw=0.5, label="ISM target")
    ax.plot(f_axis, mag_db_pred, color="indianred", lw=0.5, label="predicted")
    ax.plot(f_axis, mag_db_ana, color="tab:green", lw=0.4, alpha=0.6, label="analytical modal")
    ax.set_xlim(0, f_max_plot)
    ax.set_xlabel("f (Hz)"); ax.set_ylabel("|H| (dB)")
    ax.set_title(f"L={L:.2f} W={W:.2f} H={H:.2f} m — spectrum overlay (centre rx)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "spectrum_overlay.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 4. receiver slice grid: 8x8 mini-plots at the middle z-plane
    # (showing 512 individual subplots would be unreadable; one z-slice is
    # representative.)
    rx_per_axis = 8
    if n_rx == rx_per_axis ** 3:
        # Receiver grid is row-major over z, y, x (per dataset_builder convention).
        z_mid_idx = rx_per_axis // 2
        slice_start = z_mid_idx * rx_per_axis * rx_per_axis
        slice_end = slice_start + rx_per_axis * rx_per_axis
        fig, axs = plt.subplots(rx_per_axis, rx_per_axis, figsize=(11, 9), sharex=True)
        for iy in range(rx_per_axis):
            for ix in range(rx_per_axis):
                idx = slice_start + (rx_per_axis - 1 - iy) * rx_per_axis + ix
                ax = axs[iy, ix]
                mp = 20 * np.log10(np.maximum(np.abs(H_pred[idx]), eps))
                mt = 20 * np.log10(np.maximum(np.abs(H_target[idx]), eps))
                ax.plot(f_axis, mt, color="steelblue", lw=0.4)
                ax.plot(f_axis, mp, color="indianred", lw=0.4, alpha=0.8)
                ax.set_xlim(0, f_max_plot)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylim(-80, 30)
                if iy == rx_per_axis - 1:
                    ax.set_xlabel(f"x={rx_pos[idx,0]:.1f}", fontsize=6)
                if ix == 0:
                    ax.set_ylabel(f"y={rx_pos[idx,1]:.1f}", fontsize=6,
                                  rotation=0, labelpad=12)
        fig.suptitle(
            f"L={L:.2f} W={W:.2f} H={H:.2f} m — receiver slice at z={rx_pos[slice_start,2]:.2f} m: "
            f"predicted (red) vs ISM (blue), 0–{int(f_max_plot)} Hz",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(fig_dir / "receiver_grid.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    # 5-9. Signal-level figures (Dolby's suite)
    make_signal_plots(
        H_pred, H_target, fs=fs, n_time_samples=n_time,
        output_dir=fig_dir, representative_rx_idx=rep,
        bands=DEFAULT_BANDS, f_max_plot=f_max_plot,
        rir_pred=rir_pred, rir_target=rir_target,
    )

    return eval_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, required=True)
    ap.add_argument("--W", type=float, required=True)
    ap.add_argument("--H", type=float, required=True)
    ap.add_argument("--rooms-yaml", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    args = ap.parse_args()
    out = evaluate_single_room_3d(
        L=args.L, W=args.W, H=args.H,
        rooms_yaml=args.rooms_yaml, output_dir=args.output_dir,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
