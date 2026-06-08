"""Zero-shot adaptation at unseen 3D room (L, W, H).

P2-2 analog of `aaf.eval.zero_shot`. Adapts a fresh `z_star` to an unseen
test room by optimising against 8 sparse observed receivers, then evaluates
the predicted field on the (up to) 504 held-out receivers.

Pipeline (mirrors Phase 1 zero_shot.py):
  1. Load `INR3D_AutoDecoder` + `FreqRenderer3D` from train_meta.json.
  2. Read target HDF5: 512 receivers × 4097 freq complex.
  3. Subsample obs_idx (8 corners) and held_idx (504 complement, or a subset).
  4. Freeze model. Init z_star ~ N(0, 1/√d) at the given seed.
  5. Inner loop (2000 iters): 4-term spectral loss on obs + λ·‖z‖²; Adam lr=1e-2.
  6. Final render of all 512 receivers; compute headline signal-level metrics
     via `aaf.eval.signal_level.compute_signal_metrics` on the held-out subset
     plus modal LSD on the f<f_modal_cap band.
  7. predict_geometry from z_star → compare to truth, report per-axis MAE.
  8. Save z_star.pt, H_pred_all.pt, metrics.json, loss_curve.json, figures.

CLI
---
    python -m aaf.eval.zero_shot_3d \
        --target-h5 data/track_a_3d/L4.50_W4.00_H3.25.h5 \
        --train-output-dir outputs/multi_room_3d/M1_45rooms \
        --output_dir outputs/multi_room_3d/M1_45rooms/zero_shot/L4.50_W4.00_H3.25
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
import torch.nn as nn
import torch.nn.functional as F

from aaf.data.dataset_builder import read_room_h5
from aaf.eval.band_limited import compute_band_limited_metrics
from aaf.eval.modal_verifier import (
    modal_error_metrics,
    pick_peaks,
    plot_modal_overlay,
)
from aaf.eval.zero_shot_diagnosis import compute_manifold_distances
from aaf.eval.signal_level import (
    DEFAULT_BANDS,
    compute_signal_metrics,
    make_signal_plots,
)
from aaf.models.inr_3d import INR3D_AutoDecoder
from aaf.renderers.freq_3d import FreqRenderer3D
from aaf.sim.analytical_modal_3d import eigenfrequencies_3d


C_DEFAULT = 343.0


# Deterministic 8-of-512 observed receivers: the 8 corners of the 8×8×8 grid.
# Receivers are stored row-major over (z, y, x) per build_3d_dataset.py:
#   index = iz * 64 + iy * 8 + ix
# 8 corners: {(0,0,0), (0,0,7), (0,7,0), (0,7,7), (7,0,0), (7,0,7), (7,7,0), (7,7,7)}
# → [0, 7, 56, 63, 448, 455, 504, 511].
OBS_INDICES_3D = np.array([0, 7, 56, 63, 448, 455, 504, 511], dtype=np.int64)


def select_obs_indices_3d(n_obs: int, total: int = 512) -> np.ndarray:
    """Deterministic sub-sampling of receiver indices for 3D zero-shot."""
    if n_obs == 8 and total == 512:
        return OBS_INDICES_3D.copy()
    if n_obs <= 0 or n_obs > total:
        raise ValueError(f"n_obs must be in (0, {total}]; got {n_obs}")
    return np.unique(np.linspace(0, total - 1, n_obs).round().astype(np.int64))


def _losses(H_pred: torch.Tensor, H_target: torch.Tensor) -> dict:
    eps = 1e-6
    return {
        "L_spec_real": F.l1_loss(H_pred.real, H_target.real),
        "L_spec_imag": F.l1_loss(H_pred.imag, H_target.imag),
        "L_amp": F.l1_loss(
            torch.log10(H_pred.abs() + eps),
            torch.log10(H_target.abs() + eps),
        ),
        "L_phase": (1.0 - torch.cos(H_pred.angle() - H_target.angle())).mean(),
    }


def _load_trained_model(
    train_output_dir: Path, device: str = "cuda"
) -> tuple[INR3D_AutoDecoder, dict]:
    train_meta = json.loads((train_output_dir / "train_meta.json").read_text())
    n_rooms = int(train_meta["n_rooms"])
    cfg = train_meta["cfg"]
    n_freq_bins = int(cfg["n_time_samples"]) // 2 + 1

    hg_cfg = {
        "otype": "HashGrid",
        "n_levels": int(cfg.get("n_levels", 16)),
        "n_features_per_level": 2,
        "log2_hashmap_size": int(cfg.get("log2_hashmap_size", 18)),
        "base_resolution": 16,
        "per_level_scale": float(cfg.get("per_level_scale", 1.38)),
    }
    model = INR3D_AutoDecoder(
        n_rooms=n_rooms,
        latent_dim=int(cfg["latent_dim"]),
        n_freq_bins=n_freq_bins,
        hash_grid_config=hg_cfg,
        l_head_enabled=bool(cfg.get("l_head_enabled", True)),
        conditioning_type=str(cfg.get("conditioning_type", "film")),
        latent_jitter_sigma=float(cfg.get("latent_jitter_sigma", 0.0)),
    ).to(device)

    ckpts = sorted(
        train_output_dir.glob("ckpt_iter*.pt"),
        key=lambda p: int(p.stem.split("ckpt_iter")[-1]),
        reverse=True,
    )
    state = None
    for p in ckpts:
        try:
            state = torch.load(p, map_location=device)
            break
        except Exception as e:
            print(f"[zero-shot-3d] skipping corrupted ckpt {p.name}: {e!r}")
    if state is None:
        raise FileNotFoundError(f"no usable checkpoint in {train_output_dir}")
    model.load_state_dict(state["model"])
    model.eval()
    return model, train_meta


def zero_shot_adapt_3d(
    train_output_dir: Path,
    target_h5: Path,
    output_dir: Path,
    n_obs_receivers: int = 8,
    n_adapt_iters: int = 2000,
    lr: float = 1e-2,
    device: str = "cuda",
    weights: tuple = (1.0, 1.0, 1.0, 0.1),
    lambda_latent: float = 1e-4,
    random_seed: int = 0,
    held_out_subset_size=None,
    bands: tuple = DEFAULT_BANDS,
    eval_chunk: int = 4,
    z_init: str = "randn",
) -> dict:
    """3D zero-shot inner-loop adaptation. Returns metrics dict."""
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load trained model + train meta.
    model, train_meta = _load_trained_model(Path(train_output_dir), device=device)
    cfg = train_meta["cfg"]
    fs = float(cfg["fs"])
    n_time = int(cfg["n_time_samples"])
    n_freq = n_time // 2 + 1
    latent_dim = int(cfg["latent_dim"])

    # Load target (unseen) room.
    rt = read_room_h5(target_h5)
    attrs = rt["attrs"]
    L = float(attrs["L"])
    W = float(attrs["W"])
    H_dim = float(attrs["H"])
    receiver_pos = np.asarray(attrs["receiver_pos"], dtype=np.float32)        # [512, 3]
    H_target_all_np = rt["ism_H"].astype(np.complex64)                        # [512, n_freq]
    rir_target_all_np = rt["ism_rir"].astype(np.float32)                      # [512, n_time]
    src = np.asarray(attrs["source_pos"], dtype=np.float32)                   # [3]
    f_schroeder = float(attrs.get("schroeder_freq_hz", 200.0))

    # Obs / held-out indices.
    obs_idx = select_obs_indices_3d(int(n_obs_receivers), total=receiver_pos.shape[0])
    held_idx_full = np.array(
        [i for i in range(receiver_pos.shape[0]) if i not in set(obs_idx.tolist())],
        dtype=np.int64,
    )
    if held_out_subset_size is not None and held_out_subset_size < held_idx_full.size:
        # Deterministic stride sample.
        stride = max(1, held_idx_full.size // int(held_out_subset_size))
        held_idx = held_idx_full[::stride][: int(held_out_subset_size)]
    else:
        held_idx = held_idx_full

    # Renderer.
    renderer = FreqRenderer3D(
        n_azi=int(cfg.get("n_azi", 16)),
        n_ele=int(cfg.get("n_ele", 16)),
        n_pts_per_ray=int(cfg.get("n_pts_per_ray", 16)),
        near=float(cfg.get("near", 1e-3)),
        fs=int(fs),
        n_time_samples=n_time,
        c=float(cfg.get("c", C_DEFAULT)),
        use_geometric_attn=False,
    ).to(device).train()

    # Freeze model weights.
    for p in model.parameters():
        p.requires_grad_(False)

    # Tensors for the observed batch.
    rx_obs_t = torch.from_numpy(receiver_pos[obs_idx]).to(device)             # [n_obs, 3]
    tx_obs_t = torch.from_numpy(np.tile(src, (rx_obs_t.size(0), 1))).to(device)
    H_obs_t = torch.from_numpy(H_target_all_np[obs_idx]).to(device)
    room_min = torch.tensor([0.0, 0.0, 0.0], device=device)
    room_max = torch.tensor([L, W, H_dim], device=device)
    w_r, w_i, w_a, w_p = weights
    eps_lsd = 1e-8

    # Init z_star. Default "randn" ~ N(0, 1/√d) → ‖z*‖≈1, the historical behaviour.
    # The Run C probe (2026-06-08) showed the trained latents live at ‖z‖≈6.6, so a
    # ‖z*‖≈1 start is far *below* the manifold and the optimiser overshoots past it
    # to a high-norm off-manifold region. "mean" inits z* at the training-latent
    # centroid (on the manifold) and anchors the regulariser there (λ‖z*−z̄‖²) so the
    # search stays on the manifold. Opt-in; "randn" preserves the exact old path.
    torch.manual_seed(int(random_seed))
    z_anchor = torch.zeros(latent_dim, device=device)
    if z_init == "mean":
        z_anchor = model.latents.weight.detach().mean(0).to(device)
        z_star = nn.Parameter(z_anchor.clone())
    elif z_init == "randn":
        z_star = nn.Parameter(torch.randn(latent_dim, device=device) / math.sqrt(latent_dim))
    else:
        raise ValueError(f"z_init must be 'randn' or 'mean', got {z_init!r}")
    optimizer = torch.optim.Adam([z_star], lr=lr)
    loss_curve: list[dict] = []

    # Chunk the obs receivers so we don't OOM on 12 GB cards. Each chunk does
    # its own forward+backward; per-chunk loss scaled by (chunk_n / n_obs) so
    # the accumulated gradient equals the full-batch gradient (mean over rx).
    # λ_latent is added once per outer step (independent of receivers).
    n_obs = rx_obs_t.size(0)
    chunks = [(c0, min(c0 + eval_chunk, n_obs)) for c0 in range(0, n_obs, eval_chunk)]
    for step in range(int(n_adapt_iters)):
        optimizer.zero_grad(set_to_none=True)
        spec_loss_log = 0.0
        real_loss_log = 0.0
        amp_loss_log = 0.0
        for c0, c1 in chunks:
            z_chunk = z_star
            z_s_c = z_chunk.unsqueeze(0).expand(c1 - c0, -1)
            H_pred_c = renderer(
                model, rx_obs_t[c0:c1], tx_obs_t[c0:c1],
                room_min, room_max, z_s=z_s_c,
            )
            losses_c = _losses(H_pred_c, H_obs_t[c0:c1])
            weight = (c1 - c0) / n_obs
            loss_c = weight * (
                w_r * losses_c["L_spec_real"]
                + w_i * losses_c["L_spec_imag"]
                + w_a * losses_c["L_amp"]
                + w_p * losses_c["L_phase"]
            )
            loss_c.backward()
            spec_loss_log += float(loss_c.detach())
            real_loss_log += weight * float(losses_c["L_spec_real"].detach())
            amp_loss_log += weight * float(losses_c["L_amp"].detach())
        l_latent = ((z_star - z_anchor) ** 2).mean()
        (lambda_latent * l_latent).backward()
        if z_star.grad is not None:
            z_star.grad = torch.nan_to_num(
                z_star.grad, nan=0.0, posinf=0.0, neginf=0.0
            )
        torch.nn.utils.clip_grad_norm_([z_star], 1.0)
        optimizer.step()
        if step % 20 == 0 or step == n_adapt_iters - 1:
            loss_curve.append({
                "iter": step,
                "loss": spec_loss_log + float((lambda_latent * l_latent).detach()),
                "L_spec_real": real_loss_log,
                "L_amp": amp_loss_log,
                "z_norm": float(z_star.detach().norm()),
            })

    z_star_tensor = z_star.detach()

    # Final render of all 512 receivers (eval mode for determinism).
    renderer.eval()
    with torch.no_grad():
        z_s_static = z_star_tensor.unsqueeze(0)
        rx_all_t = torch.from_numpy(receiver_pos).to(device)
        tx_all_t = torch.from_numpy(np.tile(src, (rx_all_t.size(0), 1))).to(device)
        H_pred_chunks = []
        for s in range(0, rx_all_t.size(0), eval_chunk):
            sub_rx = rx_all_t[s : s + eval_chunk]
            sub_tx = tx_all_t[s : s + eval_chunk]
            z_s = z_s_static.expand(sub_rx.size(0), -1)
            H_pred_c = renderer(model, sub_rx, sub_tx, room_min, room_max, z_s=z_s)
            H_pred_chunks.append(H_pred_c.cpu().numpy())
        H_pred_all = np.concatenate(H_pred_chunks, axis=0).astype(np.complex64)

    # Metrics on the held-out set.
    H_target = H_target_all_np
    H_pred = H_pred_all
    rir_pred_all = np.fft.irfft(H_pred, n=n_time, axis=-1).astype(np.float32)
    rir_target_all = rir_target_all_np

    eps = 1e-8
    obs_lsd = float(np.mean(np.abs(20 * np.log10(
        np.maximum(np.abs(H_pred[obs_idx]), eps) / np.maximum(np.abs(H_target[obs_idx]), eps)
    ))))
    held_lsd = float(np.mean(np.abs(20 * np.log10(
        np.maximum(np.abs(H_pred[held_idx]), eps) / np.maximum(np.abs(H_target[held_idx]), eps)
    ))))
    held_complex = float(np.mean(np.abs(H_pred[held_idx] - H_target[held_idx])))
    held_mag = float(np.mean(np.abs(np.abs(H_pred[held_idx]) - np.abs(H_target[held_idx]))))
    pd = np.angle(H_pred[held_idx]) - np.angle(H_target[held_idx])
    pd_w = np.minimum(np.abs(pd), 2 * np.pi - np.abs(pd))
    held_phase = float(np.mean(pd_w))

    # Signal-level metrics (Dolby headline) — on held-out subset.
    signal_metrics = compute_signal_metrics(
        H_pred[held_idx], H_target[held_idx],
        fs=fs, n_time_samples=n_time,
        bands=bands,
        rir_pred=rir_pred_all[held_idx],
        rir_target=rir_target_all[held_idx],
    )

    # Band-limited LSD (held-out + observed).
    band_metrics_held = compute_band_limited_metrics(
        H_pred[held_idx], H_target[held_idx], fs, n_freq, bands,
    )
    band_metrics_obs = compute_band_limited_metrics(
        H_pred[obs_idx], H_target[obs_idx], fs, n_freq, bands,
    )

    # Modal MAE on f<f_modal_cap (D18 convention: cap=clip(f_S, 100, 250)).
    f_modal_cap = max(100.0, min(f_schroeder, 250.0))
    centre_idx = int(np.argmin(np.linalg.norm(
        receiver_pos - np.array([L / 2.0, W / 2.0, H_dim / 2.0]), axis=1
    )))
    f_axis = np.arange(n_freq) * (fs / n_time)
    f_mask = f_axis <= f_modal_cap
    modes_modal = [
        m for m in eigenfrequencies_3d(L=L, W=W, H=H_dim, c=C_DEFAULT, f_max=f_modal_cap)
        if m.f > 0
    ]
    if f_mask.sum() > 0 and modes_modal:
        peaks_pred_modal = pick_peaks(
            H_pred[centre_idx, f_mask], f_axis[f_mask],
            prominence_db=3.0, min_distance_hz=2.0,
        )
        modal_metrics = modal_error_metrics(
            peaks_pred_modal, modes_modal, tolerance_hz=4.0, tolerance_pct=0.02
        )
    else:
        peaks_pred_modal = []
        modal_metrics = {
            "mae_hz": float("nan"), "recall_at_tol": 0.0,
            "n_picked": 0, "n_analytical": 0, "n_matched": 0,
        }

    # Geometry head: predict (L, W, H) from z_star and report per-axis MAE.
    geom_extras = {}
    if model.l_head is not None:
        with torch.no_grad():
            geom_pred = (
                model.predict_geometry(z_star_tensor.unsqueeze(0))[0].cpu().numpy().tolist()
            )
        geom_true = [L, W, H_dim]
        geom_extras = {
            "geom_pred_LWH": geom_pred,
            "geom_true_LWH": geom_true,
            "geom_err_L_m": float(abs(geom_pred[0] - L)),
            "geom_err_W_m": float(abs(geom_pred[1] - W)),
            "geom_err_H_m": float(abs(geom_pred[2] - H_dim)),
            "geom_err_max_m": float(max(abs(geom_pred[0] - L),
                                        abs(geom_pred[1] - W),
                                        abs(geom_pred[2] - H_dim))),
        }

    # Manifold-distance self-diagnosis (P2-3 D37): is z* near where the trained
    # latents sit? Two distances:
    #   latent_min_dist        — ‖z* − nearest training latent‖₂
    #   geom_nearest_train_dist — ‖z* − latent of the training room whose TRUE
    #                             (L,W,H) is closest to this test room‖₂
    # Together with geom_err these classify the 3-way verdict: a z* that's far
    # from the manifold (large latent_min_dist) AND geometrically misplaced
    # (large geom_err) => the inner loop couldn't reach the right region =>
    # manifold-coverage problem (more training rooms). A well-placed z* with a
    # still-poor spectrum => decoder-at-interpolated-latent problem.
    z_train = model.latents.weight.detach().cpu().numpy()                 # [n_rooms, d]
    z_np = z_star_tensor.cpu().numpy()
    L_list = train_meta.get("L_list", [])
    W_list = train_meta.get("W_list", [])
    H_list = train_meta.get("H_list", [])
    train_LWH = None
    if L_list and W_list and H_list and len(L_list) == z_train.shape[0]:
        train_LWH = list(zip(L_list, W_list, H_list))
    manifold_extras = compute_manifold_distances(
        z_np, z_train, train_LWH=train_LWH, test_LWH=[L, W, H_dim]
    )

    # Save artifacts.
    torch.save(z_star_tensor.cpu(), output_dir / "z_star.pt")
    torch.save(torch.from_numpy(H_pred_all), output_dir / "H_pred_all.pt")
    (output_dir / "loss_curve.json").write_text(json.dumps(loss_curve, indent=2))
    (output_dir / "band_limited_metrics.json").write_text(json.dumps(
        {"held": band_metrics_held, "obs": band_metrics_obs,
         "L": L, "W": W, "H": H_dim,
         "n_obs_receivers": int(n_obs_receivers),
         "n_held_out": int(len(held_idx))}, indent=2
    ))

    metrics = {
        "L": L, "W": W, "H": H_dim,
        "obs_idx": obs_idx.tolist(),
        "held_idx": held_idx.tolist(),
        "n_held_out": int(len(held_idx)),
        "n_adapt_iters": int(n_adapt_iters),
        "n_obs_receivers": int(n_obs_receivers),
        "random_seed": int(random_seed),
        "z_init": str(z_init),
        "lambda_latent": float(lambda_latent),
        "obs_lsd_db": obs_lsd,
        "held_out_lsd_db": held_lsd,
        "held_out_complex_l1": held_complex,
        "held_out_magnitude_l1": held_mag,
        "held_out_phase_l1": held_phase,
        "held_out_modal_mae_hz": modal_metrics["mae_hz"],
        "held_out_modal_recall": modal_metrics["recall_at_tol"],
        "n_picked": modal_metrics["n_picked"],
        "n_analytical": modal_metrics["n_analytical"],
        "n_matched": modal_metrics["n_matched"],
        "f_modal_cap_hz": f_modal_cap,
        "f_schroeder_hz": f_schroeder,
        "z_star_norm": float(z_star_tensor.norm()),
        "signal_metrics": signal_metrics,
        "band_metrics_held": band_metrics_held,
        "band_metrics_obs": band_metrics_obs,
        "loss_curve_first_last": [loss_curve[0], loss_curve[-1]] if loss_curve else [],
        **geom_extras,
        **manifold_extras,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---------- Figures ----------
    f_max_plot = 2000.0
    # Reuse the signal-level plot suite as the headline (5 PNGs).
    make_signal_plots(
        H_pred, H_target, fs=fs, n_time_samples=n_time,
        output_dir=fig_dir, representative_rx_idx=centre_idx,
        bands=bands, f_max_plot=f_max_plot,
        rir_pred=rir_pred_all, rir_target=rir_target_all,
    )

    # Modal tracking (zoomed into the matched modal band).
    if modes_modal:
        fig, ax = plt.subplots(figsize=(11, 4))
        plot_modal_overlay(
            H_pred[centre_idx], f_axis, modes_modal, peaks_pred_modal, ax,
            title=f"L={L:.2f} W={W:.2f} H={H_dim:.2f} (UNSEEN) — predicted |H|",
            f_min=0, f_max=int(f_modal_cap * 1.5), db_floor=-100,
        )
        fig.tight_layout()
        fig.savefig(fig_dir / "zero_shot_modal_tracking.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    # Adaptation loss curve.
    if loss_curve:
        iters = [r["iter"] for r in loss_curve]
        vals = [r["loss"] for r in loss_curve]
        z_norms = [r["z_norm"] for r in loss_curve]
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.semilogy(iters, vals, color="indianred", label="adaptation loss")
        ax1.set_xlabel("inner iter"); ax1.set_ylabel("loss"); ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(iters, z_norms, color="steelblue", lw=0.6, label="‖z_star‖")
        ax2.set_ylabel("‖z_star‖", color="steelblue")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.set_title(
            f"L={L:.2f} W={W:.2f} H={H_dim:.2f} UNSEEN — z_star adaptation "
            f"(final norm {z_star_tensor.norm():.3f})"
        )
        fig.tight_layout()
        fig.savefig(fig_dir / "adapt_loss_curve.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-h5", type=str, required=True)
    ap.add_argument("--train-output-dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--n_obs_receivers", type=int, default=8)
    ap.add_argument("--n_adapt_iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--random_seed", type=int, default=0)
    ap.add_argument(
        "--held_out_subset_size", type=int, default=None,
        help="If given, sub-sample the held-out set for headline metrics.",
    )
    ap.add_argument("--eval_chunk", type=int, default=4)
    ap.add_argument(
        "--z_init", type=str, default="randn", choices=["randn", "mean"],
        help="z* init + reg anchor. 'randn' (default, historical): ‖z*‖≈1, λ‖z*‖². "
             "'mean': init at the training-latent centroid (on the manifold) with "
             "λ‖z*−z̄‖² — keeps the search on the manifold (manifold-anchored adaptation).",
    )
    ap.add_argument("--lambda_latent", type=float, default=1e-4)
    args = ap.parse_args()

    out = zero_shot_adapt_3d(
        train_output_dir=Path(args.train_output_dir),
        target_h5=Path(args.target_h5),
        output_dir=Path(args.output_dir),
        n_obs_receivers=args.n_obs_receivers,
        n_adapt_iters=args.n_adapt_iters,
        lr=args.lr,
        random_seed=args.random_seed,
        held_out_subset_size=args.held_out_subset_size,
        eval_chunk=args.eval_chunk,
        z_init=args.z_init,
        lambda_latent=args.lambda_latent,
    )
    # Print a compact summary.
    print(json.dumps(
        {k: v for k, v in out.items() if not isinstance(v, (list, dict))},
        indent=2,
    ))


if __name__ == "__main__":
    main()
