"""Per-room evaluator: compute dual-metric report + 4 figures.

CLI
---
    python -m aaf.eval.single_room_eval --L 3.0 --output_dir outputs/single_room/L3.0

Reads from `output_dir/ckpt_iter*.pt` (latest), evaluates the trained model on
the same room's 64 receivers, and writes:
  - eval.json (modal + full-band metrics, per-mode-band breakdown)
  - figures/training_curves.png
  - figures/modal_tracking.png
  - figures/spectrum_overlay.png
  - figures/receiver_grid.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from aaf.data.loader import ShoeboxDataset
from aaf.eval.modal_verifier import (
    pick_peaks,
    plot_modal_overlay,
    modal_error_metrics,
)
from aaf.models.inr_2d import INR2D_Single
from aaf.renderers.freq_2d import FreqRenderer2D
from aaf.sim.analytical_modal_2d import eigenfrequencies_2d, modal_rir_2d


C_DEFAULT = 343.0


def _load_latest_ckpt(output_dir: Path) -> tuple[Path, dict]:
    ckpts = sorted(output_dir.glob("ckpt_iter*.pt"),
                   key=lambda p: int(p.stem.split("ckpt_iter")[-1]),
                   reverse=True)
    for p in ckpts:
        try:
            state = torch.load(p, map_location="cuda")
            return p, state
        except Exception:
            continue
    raise FileNotFoundError(f"no usable ckpt in {output_dir}")


def _representative_receiver_idx(receiver_pos: np.ndarray, L: float, W: float) -> int:
    centre = np.array([L / 2.0, W / 2.0])
    return int(np.argmin(np.linalg.norm(receiver_pos - centre[None, :], axis=1)))


def _third_octave_smooth(mag_db: np.ndarray, f_axis: np.ndarray) -> np.ndarray:
    """Symmetric 1/3-octave moving-average smoothing on log-magnitude.

    For each frequency f, average over [f / 2^(1/6), f * 2^(1/6)]. At low freq
    this is < 1 bin and we fall back to no smoothing.
    """
    out = np.zeros_like(mag_db)
    df = float(f_axis[1] - f_axis[0])
    factor = 2 ** (1 / 6)
    for i, f in enumerate(f_axis):
        if f <= df:
            out[i] = mag_db[i]
            continue
        lo, hi = f / factor, f * factor
        lo_i = max(0, int(np.searchsorted(f_axis, lo)))
        hi_i = min(len(mag_db), int(np.searchsorted(f_axis, hi)) + 1)
        if hi_i - lo_i <= 1:
            out[i] = mag_db[i]
        else:
            out[i] = mag_db[lo_i:hi_i].mean()
    return out


def evaluate_single_room(L: float, sweep_yaml: str, output_dir: str,
                         device: str = "cuda") -> dict:
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    dataset = ShoeboxDataset(sweep_yaml=sweep_yaml, split="train", room_filter=[L])
    n_freq = dataset.n_freq_bins
    n_time = dataset.n_time_samples
    fs = dataset.fs

    # Load all 64 receivers' targets once.
    items = [dataset[i] for i in range(len(dataset))]
    rx_pos = np.stack([it["rx_pos"].numpy() for it in items])         # [64, 2]
    tx_pos = np.stack([it["tx_pos"].numpy() for it in items])         # [64, 2]
    H_target = np.stack([it["H_complex"].numpy() for it in items])    # [64, n_freq]

    f_axis = np.arange(n_freq) * (fs / n_time)
    W = dataset.W

    model = INR2D_Single(n_freq_bins=n_freq).to(device).eval()
    # Read renderer params from train_meta.json so we evaluate with the same renderer as training.
    train_meta_path = output_dir / "train_meta.json"
    if train_meta_path.exists():
        train_cfg = json.loads(train_meta_path.read_text())["cfg"]
        n_azi = int(train_cfg.get("n_azi", 64))
        n_pts_per_ray = int(train_cfg.get("n_pts_per_ray", 32))
    else:
        n_azi, n_pts_per_ray = 64, 32  # match memory_check fallback
    renderer = FreqRenderer2D(
        n_azi=n_azi, n_pts_per_ray=n_pts_per_ray, near=1e-3,
        fs=fs, n_time_samples=n_time, c=C_DEFAULT, use_geometric_attn=False,
    ).to(device).eval()
    ckpt_path, state = _load_latest_ckpt(output_dir)
    model.load_state_dict(state["model"])
    model.eval()

    # Forward all receivers in chunks (avoid OOM).
    chunk = 8
    H_pred_chunks = []
    rx_t = torch.from_numpy(rx_pos).to(device)
    tx_t = torch.from_numpy(tx_pos).to(device)
    room_min = torch.tensor([0.0, 0.0], device=device)
    room_max = torch.tensor([float(L), W], device=device)
    with torch.no_grad():
        for s in range(0, rx_t.size(0), chunk):
            H_pred_chunks.append(
                renderer(model, rx_t[s:s+chunk], tx_t[s:s+chunk], room_min, room_max).cpu().numpy()
            )
    H_pred = np.concatenate(H_pred_chunks, axis=0).astype(np.complex64)

    # ---------------- Metrics ----------------
    eps = 1e-8
    mag_pred = np.abs(H_pred)
    mag_target = np.abs(H_target)
    lsd_db = float(np.mean(np.abs(20 * np.log10(np.maximum(mag_pred, eps) / np.maximum(mag_target, eps)))))
    complex_l1 = float(np.mean(np.abs(H_pred - H_target)))
    magnitude_l1 = float(np.mean(np.abs(mag_pred - mag_target)))
    phase_diff = np.angle(H_pred) - np.angle(H_target)
    phase_wrapped = np.minimum(np.abs(phase_diff), 2 * np.pi - np.abs(phase_diff))
    phase_l1 = float(np.mean(phase_wrapped))

    # Envelope LSD via 1/3-octave smoothing (per-receiver, then mean).
    env_lsd_per_rx = []
    for r in range(rx_pos.shape[0]):
        mp = _third_octave_smooth(20 * np.log10(np.maximum(mag_pred[r], eps)), f_axis)
        mt = _third_octave_smooth(20 * np.log10(np.maximum(mag_target[r], eps)), f_axis)
        env_lsd_per_rx.append(np.mean(np.abs(mp - mt)))
    envelope_lsd_db = float(np.mean(env_lsd_per_rx))

    # Modal-regime metrics on the room-centre receiver.
    rep = _representative_receiver_idx(rx_pos, L, W)
    f_max = 2000.0
    f_mask = (f_axis >= 0) & (f_axis <= f_max)
    schroeder = float(0.161 * (L * W) / (dataset.alpha * 2 * (L + W))) ** 0.5  # not used; pra preserves attrs
    # Use the room's stored Schroeder approx for cut.
    attrs = dataset.get_room_attrs(0)
    f_schroeder = float(attrs.get("schroeder_freq_approx_hz", 500.0))
    modes_full = [m for m in eigenfrequencies_2d(L=L, W=W, c=C_DEFAULT, f_max=f_max) if m.f > 0]
    modes_modal = [m for m in modes_full if m.f <= f_schroeder]

    peaks_pred_full = pick_peaks(H_pred[rep, f_mask], f_axis[f_mask],
                                 prominence_db=3.0, min_distance_hz=10.0)
    peaks_pred_modal = [p for p in peaks_pred_full if p.f <= f_schroeder]
    peaks_ism_full = pick_peaks(H_target[rep, f_mask], f_axis[f_mask],
                                 prominence_db=3.0, min_distance_hz=10.0)
    peaks_ism_modal = [p for p in peaks_ism_full if p.f <= f_schroeder]

    metrics_modal = modal_error_metrics(peaks_pred_modal, modes_modal,
                                        tolerance_hz=4.0, tolerance_pct=0.02)
    metrics_full = modal_error_metrics(peaks_pred_full, modes_full,
                                       tolerance_hz=4.0, tolerance_pct=0.02)

    eval_dict = {
        "L": float(L), "W": float(W), "alpha": float(dataset.alpha),
        "fs": int(fs), "n_freq_bins": int(n_freq), "n_time_samples": int(n_time),
        "ckpt_path": str(ckpt_path.relative_to(output_dir.parent.parent)),
        "ckpt_iter": int(state["iter"]),
        "f_schroeder_hz": f_schroeder,
        "n_receivers": int(rx_pos.shape[0]),

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
            "envelope_lsd_db": envelope_lsd_db,
            "modal_mae_full_hz": metrics_full["mae_hz"],
            "modal_recall_full": metrics_full["recall_at_tol"],
        },
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
        # Bottom row: 3 acoustic metrics from val
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
        fig.suptitle(f"L={L:.2f} m — training curves (ckpt_iter={state['iter']})")
        fig.tight_layout()
        fig.savefig(fig_dir / "training_curves.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    # 2. modal tracking plot
    fig, ax = plt.subplots(figsize=(11, 4))
    plot_modal_overlay(H_pred[rep], f_axis, modes_full, peaks_pred_full, ax,
                       title=f"L={L:.2f} m — predicted |H(f)| 0–200 Hz",
                       f_min=0, f_max=200, db_floor=-100)
    # Overlay ISM peaks as blue triangles for reference.
    for p in peaks_ism_full:
        if p.f <= 200:
            ax.plot(p.f, p.magnitude_db, "v", color="tab:blue", ms=6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(fig_dir / "modal_tracking.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 3. spectrum overlay (predicted vs ISM vs analytical-modal-sum)
    cfg_modal = {
        "L": L, "W": W,
        "source_pos": tuple(tx_pos[0]),
        "receiver_pos": rx_pos[rep:rep+1],
        "alpha": dataset.alpha,
        "fs": fs, "n_time_samples": n_time,
        "f_max_modes": f_max,
    }
    ana = modal_rir_2d(cfg_modal)
    H_ana = ana["H_complex"][0]

    fig, ax = plt.subplots(figsize=(11, 4))
    mag_db_pred = 20 * np.log10(np.maximum(np.abs(H_pred[rep]), eps))
    mag_db_ism = 20 * np.log10(np.maximum(np.abs(H_target[rep]), eps))
    mag_db_ana = 20 * np.log10(np.maximum(np.abs(H_ana), eps))
    ax.plot(f_axis, mag_db_ism, color="steelblue", lw=0.5, label="ISM target")
    ax.plot(f_axis, mag_db_pred, color="indianred", lw=0.5, label="predicted")
    ax.plot(f_axis, mag_db_ana, color="tab:green", lw=0.4, alpha=0.6, label="analytical modal")
    ax.set_xlim(0, f_max)
    ax.set_xlabel("f (Hz)"); ax.set_ylabel("|H| (dB)")
    ax.set_title(f"L={L:.2f} m — spectrum overlay (centre rx)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "spectrum_overlay.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 4. receiver grid (8x8 mini-plots)
    n = 8
    fig, axs = plt.subplots(n, n, figsize=(11, 9), sharex=True)
    for iy in range(n):
        for ix in range(n):
            idx = (n - 1 - iy) * n + ix
            ax = axs[iy, ix]
            mp = 20 * np.log10(np.maximum(np.abs(H_pred[idx]), eps))
            mt = 20 * np.log10(np.maximum(np.abs(H_target[idx]), eps))
            ax.plot(f_axis, mt, color="steelblue", lw=0.4)
            ax.plot(f_axis, mp, color="indianred", lw=0.4, alpha=0.8)
            ax.set_xlim(0, f_max)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim(-80, 30)
            if iy == n - 1:
                ax.set_xlabel(f"x={rx_pos[idx,0]:.1f}", fontsize=6)
            if ix == 0:
                ax.set_ylabel(f"y={rx_pos[idx,1]:.1f}", fontsize=6, rotation=0, labelpad=12)
    fig.suptitle(f"L={L:.2f} m — receiver grid: predicted (red) vs ISM (blue), 0–{int(f_max)} Hz",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(fig_dir / "receiver_grid.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    return eval_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, required=True)
    ap.add_argument("--sweep", type=str, default="configs/sweeps/dense.yaml")
    ap.add_argument("--output_dir", type=str, required=True)
    args = ap.parse_args()
    out = evaluate_single_room(L=args.L, sweep_yaml=args.sweep, output_dir=args.output_dir)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
