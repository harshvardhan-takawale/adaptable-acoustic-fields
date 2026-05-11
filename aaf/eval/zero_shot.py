"""Zero-shot adaptation at unseen room L.

Given a trained ``INR2D_AutoDecoder`` and a target room (not in the training
set), this module:

  1. Loads the target room's H_target [64, n_freq] and rx_pos [64, 2].
  2. Subsamples 8 observed receivers via a deterministic 3×3-minus-corner
     pattern (indices [0, 3, 7, 24, 31, 56, 59, 63]).
  3. Freezes the trained network weights.
  4. Initialises a fresh ``z_star ∈ ℝ^32`` and optimises it for 2,000 iters
     using the same 5-term loss as training, restricted to the 8 observed
     receivers.
  5. Forwards the trained network with z_star on all 64 receivers; reports
     held-out metrics on the 56 unobserved positions plus modal MAE on the
     centre receiver against deduplicated analytical eigenfreqs for THIS L.
  6. Writes z_star.pt, metrics.json, and 4 figures.

CLI
---
    python -m aaf.eval.zero_shot --L 3.25 \
        --train_output_dir outputs/multi_room/dense \
        --output_dir outputs/multi_room/dense/zero_shot/L3.25
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aaf.data.dataset_builder import read_room_h5, room_filename
from aaf.eval.band_limited import DEFAULT_BANDS, compute_band_limited_metrics
from aaf.eval.modal_verifier import (
    pick_peaks,
    plot_modal_overlay,
)
from aaf.models.inr_2d import INR2D_AutoDecoder
from aaf.renderers.freq_2d import FreqRenderer2D
from aaf.sim.analytical_modal_2d import eigenfrequencies_2d


C_DEFAULT = 343.0

# Deterministic 8-of-64 observed receivers: 3×3 grid (without centre).
# Receivers are stored row-major with `for y in ys: for x in xs:` (per
# scripts/build_datasets.py line 67). 8×8 grid → flat index = iy*8 + ix.
# We pick (iy, ix) ∈ {(0,0), (0,3), (0,7), (3,0), (3,7), (7,0), (7,3), (7,7)}
# → flat indices [0, 3, 7, 24, 31, 56, 59, 63]. The centre (3, 3) is dropped on
# purpose so we can hold it out for evaluation.
OBS_INDICES = np.array([0, 3, 7, 24, 31, 56, 59, 63], dtype=np.int64)


def select_obs_indices(n_obs: int, total: int = 64) -> np.ndarray:
    """Deterministic sub-sampling of receiver indices for zero-shot conditioning.

    n_obs=8 reproduces the Chunk-3 ``OBS_INDICES`` (3×3 minus centre). n_obs=32
    on a 64-receiver grid uses the standard 8×8 checkerboard (half the grid).
    Other counts fall back to a deduped ``np.linspace`` over [0, total-1].
    """
    if n_obs == 8 and total == 64:
        return OBS_INDICES.copy()
    if n_obs == 32 and total == 64:
        return np.array(
            [i for i in range(total) if (i // 8 + i % 8) % 2 == 0],
            dtype=np.int64,
        )
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


def _load_trained_model(train_output_dir: Path, device: str = "cuda") -> tuple[INR2D_AutoDecoder, dict]:
    """Load the latest checkpoint into a fresh INR2D_AutoDecoder.

    Reads HashGrid + L-head settings from ``train_meta.json["cfg"]`` so the
    rebuilt model matches the architecture used at training. Uses ``.get(...)``
    defaults so Chunk-3 train_meta files (which lack these keys) still load.
    """
    train_meta = json.loads((train_output_dir / "train_meta.json").read_text())
    n_rooms = int(train_meta["n_rooms"])
    cfg = train_meta["cfg"]
    n_freq_bins = int(cfg["n_time_samples"]) // 2 + 1

    hg_cfg = {
        "otype": "HashGrid",
        "n_levels": int(cfg.get("n_levels", 20)),
        "n_features_per_level": 2,
        "log2_hashmap_size": int(cfg.get("log2_hashmap_size", 18)),
        "base_resolution": 16,
        "per_level_scale": 1.5,
    }
    l_head_enabled = float(cfg.get("l_head_weight", 0.0)) > 0
    l_head_arch = str(cfg.get("l_head_arch", "mlp_32"))
    conditioning_type = str(cfg.get("conditioning_type", "concat"))
    latent_jitter_sigma = float(cfg.get("latent_jitter_sigma", 0.0))

    model = INR2D_AutoDecoder(
        n_rooms=n_rooms,
        latent_dim=int(cfg["latent_dim"]),
        n_freq_bins=n_freq_bins,
        hash_grid_config=hg_cfg,
        l_head_enabled=l_head_enabled,
        l_head_arch=l_head_arch,
        conditioning_type=conditioning_type,
        latent_jitter_sigma=latent_jitter_sigma,
    ).to(device)

    ckpts = sorted(
        train_output_dir.glob("ckpt_iter*.pt"),
        key=lambda p: int(p.stem.split("ckpt_iter")[-1]),
        reverse=True,
    )
    state = None
    chosen = None
    for p in ckpts:
        try:
            state = torch.load(p, map_location=device)
            chosen = p
            break
        except Exception as e:
            print(f"[zero-shot] skipping corrupted ckpt {p.name}: {e!r}")
    if state is None:
        raise FileNotFoundError(f"no usable checkpoint in {train_output_dir}")
    model.load_state_dict(state["model"])
    model.eval()
    return model, train_meta


def _build_latent_init(
    init_strategy: str,
    latent_dim: int,
    train_latents: torch.Tensor,
    L_target: float,
    train_L_list: list,
    seed: int,
    device: str,
):
    """Construct the inner-loop optimization variable per ``init_strategy``.

    Returns ``(get_z_fn, optim_params, simplex_logits_param_or_None)``.

    ``get_z_fn()`` returns the current latent ``[latent_dim]`` (called every
    inner-loop step). For 'random'/'nearest_train' this is just the underlying
    nn.Parameter; for 'simplex' it computes ``softmax(logits) @ Z_train``.
    """
    if init_strategy == "random":
        torch.manual_seed(int(seed))
        z = nn.Parameter(torch.randn(latent_dim, device=device) / math.sqrt(latent_dim))
        return (lambda: z), [z], None
    if init_strategy == "nearest_train":
        if not train_L_list:
            raise ValueError("nearest_train init requires train_meta['L_list'] to be non-empty")
        idx = int(np.argmin(np.abs(np.asarray(train_L_list) - float(L_target))))
        z = nn.Parameter(train_latents[idx].detach().clone().to(device))
        return (lambda: z), [z], None
    if init_strategy == "simplex":
        # Lazy import to avoid circulars.
        from aaf.eval.zero_shot_variants import SimplexLatent
        sx = SimplexLatent(train_latents.detach().to(device))
        return (lambda: sx()), list(sx.parameters()), sx.logits
    raise ValueError(f"unknown init_strategy={init_strategy!r}")


def zero_shot_adapt(
    train_output_dir: Path,
    target_h5: Path,
    output_dir: Path,
    n_obs_receivers: int = 8,
    n_adapt_iters: int = 2000,
    lr: float = 1e-2,
    device: str = "cuda",
    weights: tuple = (1.0, 1.0, 1.0, 0.1),
    lambda_latent: float = 1e-4,
    init_strategy: str = "random",
    n_restarts: int = 1,
    random_seed: int = 0,
    bands: Optional[tuple] = None,
) -> dict:
    """Inner-loop adaptation of z_star at an unseen room L.

    Chunk-3.6 generalises the original Chunk-3 zero-shot pipeline:
    - ``n_obs_receivers`` may be 8 (default) or 32 (Variant B2).
    - ``init_strategy`` ∈ {'random', 'nearest_train', 'simplex'} (B5/B6).
    - ``n_restarts`` runs the inner loop multiple times with seeds
      ``random_seed, random_seed+1, ...`` and keeps the lowest-obs-LSD winner (B4).
    - ``bands`` (default: modal/transition/diffuse + full) is forwarded to
      ``compute_band_limited_metrics`` and the per-band LSDs are saved alongside
      the existing full-band metrics.
    """
    if bands is None:
        bands = DEFAULT_BANDS
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
    train_L_list = list(train_meta.get("L_list", []))
    train_latents = model.latents.weight.detach().clone()             # [n_train, latent_dim]

    # Load target (unseen) room.
    rt = read_room_h5(target_h5)
    attrs = rt["attrs"]
    L = float(attrs["L"])
    W = float(attrs["W"])
    receiver_pos = np.asarray(attrs["receiver_pos"], dtype=np.float32)        # [64, 2]
    H_target_all_np = rt["ism_H"].astype(np.complex64)                        # [64, n_freq]
    src = np.asarray(attrs["source_pos"], dtype=np.float32)                   # [2]

    # Subsample observed/held-out indices.
    obs_idx = select_obs_indices(int(n_obs_receivers), total=receiver_pos.shape[0])
    held_idx = np.array(
        [i for i in range(receiver_pos.shape[0]) if i not in set(obs_idx.tolist())],
        dtype=np.int64,
    )

    # Renderer (n_pts/n_azi from training cfg).
    renderer = FreqRenderer2D(
        n_azi=int(cfg["n_azi"]),
        n_pts_per_ray=int(cfg["n_pts_per_ray"]),
        near=float(cfg["near"]),
        fs=int(fs),
        n_time_samples=n_time,
        c=float(cfg["c"]),
        use_geometric_attn=False,
    ).to(device).train()  # train mode keeps ray jitter on during inner loop

    # Freeze model weights.
    for p in model.parameters():
        p.requires_grad_(False)

    # Tensors for the observed batch.
    rx_obs_t = torch.from_numpy(receiver_pos[obs_idx]).to(device)   # [n_obs, 2]
    tx_obs_t = torch.from_numpy(np.tile(src, (rx_obs_t.size(0), 1))).to(device)
    H_obs_t = torch.from_numpy(H_target_all_np[obs_idx]).to(device)  # [n_obs, n_freq]
    room_min = torch.tensor([0.0, 0.0], device=device)
    room_max = torch.tensor([L, W], device=device)
    eps_lsd = 1e-8
    w_r, w_i, w_a, w_p = weights

    # ---------------- inner loop, n_restarts times ----------------
    best = {"obs_lsd_db": float("inf"), "z_tensor": None,
            "loss_curve": [], "simplex_logits": None, "restart": -1, "seed": -1}
    for r in range(int(max(1, n_restarts))):
        seed = int(random_seed) + r
        get_z, opt_params, simplex_logits_param = _build_latent_init(
            init_strategy=init_strategy,
            latent_dim=latent_dim,
            train_latents=train_latents,
            L_target=L,
            train_L_list=train_L_list,
            seed=seed,
            device=device,
        )
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        loss_curve: list[dict] = []
        for step in range(int(n_adapt_iters)):
            z_now = get_z()                                  # [latent_dim]
            z_s = z_now.unsqueeze(0).expand(rx_obs_t.size(0), -1)
            H_pred = renderer(model, rx_obs_t, tx_obs_t, room_min, room_max, z_s=z_s)
            losses = _losses(H_pred, H_obs_t)
            l_latent = (z_now ** 2).mean()
            loss = (
                w_r * losses["L_spec_real"]
                + w_i * losses["L_spec_imag"]
                + w_a * losses["L_amp"]
                + w_p * losses["L_phase"]
                + lambda_latent * l_latent
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for p in opt_params:
                if p.grad is not None:
                    p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
            optimizer.step()
            if step % 20 == 0 or step == n_adapt_iters - 1:
                loss_curve.append({
                    "iter": step,
                    "loss": float(loss.detach()),
                    "L_spec_real": float(losses["L_spec_real"].detach()),
                    "L_amp": float(losses["L_amp"].detach()),
                    "z_norm": float(z_now.detach().norm()),
                })

        # Score this restart by obs-LSD on the optimised z (in eval mode for
        # determinism — no ray jitter during scoring).
        renderer.eval()
        with torch.no_grad():
            z_final = get_z().detach()
            z_s = z_final.unsqueeze(0).expand(rx_obs_t.size(0), -1)
            H_obs_pred = renderer(model, rx_obs_t, tx_obs_t, room_min, room_max, z_s=z_s)
            H_obs_pred_np = H_obs_pred.cpu().numpy()
            obs_lsd_this = float(np.mean(np.abs(20 * np.log10(
                np.maximum(np.abs(H_obs_pred_np), eps_lsd)
                / np.maximum(np.abs(H_target_all_np[obs_idx]), eps_lsd)
            ))))
        renderer.train()

        if obs_lsd_this < best["obs_lsd_db"]:
            best.update({
                "obs_lsd_db": obs_lsd_this,
                "z_tensor": z_final.clone(),
                "loss_curve": loss_curve,
                "simplex_logits": (simplex_logits_param.detach().clone()
                                   if simplex_logits_param is not None else None),
                "restart": r,
                "seed": seed,
            })

    z_star_tensor = best["z_tensor"]
    loss_curve = best["loss_curve"]

    # ---------------- final render of all 64 receivers ----------------
    renderer.eval()
    with torch.no_grad():
        z_s_static = z_star_tensor.unsqueeze(0)
        rx_all_t = torch.from_numpy(receiver_pos).to(device)
        tx_all_t = torch.from_numpy(np.tile(src, (rx_all_t.size(0), 1))).to(device)
        H_pred_chunks = []
        chunk = 8
        for s in range(0, rx_all_t.size(0), chunk):
            sub_rx = rx_all_t[s : s + chunk]
            sub_tx = tx_all_t[s : s + chunk]
            z_s = z_s_static.expand(sub_rx.size(0), -1)
            H_pred = renderer(model, sub_rx, sub_tx, room_min, room_max, z_s=z_s)
            H_pred_chunks.append(H_pred.cpu().numpy())
        H_pred_all = np.concatenate(H_pred_chunks, axis=0).astype(np.complex64)

    # Metrics.
    eps = 1e-8
    H_target = H_target_all_np
    H_pred = H_pred_all
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

    # Modal MAE on the centre receiver against analytical eigenfreqs for THIS L.
    centre_idx = int(np.argmin(np.linalg.norm(receiver_pos - np.array([L/2.0, W/2.0]), axis=1)))
    f_axis = np.arange(n_freq) * (fs / n_time)
    f_max = 2000.0
    f_mask = f_axis <= f_max
    modes = [m for m in eigenfrequencies_2d(L=L, W=W, c=C_DEFAULT, f_max=f_max) if m.f > 0]
    peaks_pred = pick_peaks(H_pred[centre_idx, f_mask], f_axis[f_mask],
                            prominence_db=3.0, min_distance_hz=10.0)
    from aaf.eval.modal_verifier import modal_error_metrics
    modal_metrics = modal_error_metrics(peaks_pred, modes, tolerance_hz=4.0, tolerance_pct=0.02)

    # Save z_star + H_pred_all + metrics + figures.
    torch.save(z_star_tensor.cpu(), output_dir / "z_star.pt")
    # H_pred_all is saved going forward so future band-limited recomputation
    # doesn't need to re-forward through the model. Complex64 64×n_freq ≈ 2 MB.
    torch.save(torch.from_numpy(H_pred_all), output_dir / "H_pred_all.pt")
    if best["simplex_logits"] is not None:
        torch.save(best["simplex_logits"].cpu(), output_dir / "simplex_logits.pt")

    # Band-limited metrics on the held-out set (Chunk 3.6 standard).
    band_metrics_held = compute_band_limited_metrics(
        H_pred_all[held_idx], H_target_all_np[held_idx], fs, n_freq, bands,
    )
    band_metrics_obs = compute_band_limited_metrics(
        H_pred_all[obs_idx], H_target_all_np[obs_idx], fs, n_freq, bands,
    )

    metrics = {
        "L": L, "W": W,
        "obs_idx": obs_idx.tolist(),
        "held_idx": held_idx.tolist(),
        "n_adapt_iters": int(n_adapt_iters),
        "n_obs_receivers": int(n_obs_receivers),
        "init_strategy": str(init_strategy),
        "n_restarts": int(n_restarts),
        "random_seed": int(random_seed),
        "best_restart": int(best["restart"]),
        "best_seed": int(best["seed"]),
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
        "z_star_norm": float(z_star_tensor.norm()),
        "loss_curve_first_last": [loss_curve[0], loss_curve[-1]] if loss_curve else [],
        "band_metrics_held": band_metrics_held,
        "band_metrics_obs": band_metrics_obs,
    }
    # L-head sanity check: if the model has an L-head, predict L from z_star and
    # compare to the true target L. If the latent encodes L well, prediction
    # error should be small (tens of cm at most).
    if model.l_head is not None:
        with torch.no_grad():
            L_pred = float(model.predict_L(z_star_tensor.unsqueeze(0)).item())
        metrics["lhead_predicted_L"] = L_pred
        metrics["lhead_L_error_m"] = float(abs(L_pred - L))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "band_limited_metrics.json").write_text(
        json.dumps({"held": band_metrics_held, "obs": band_metrics_obs,
                    "L": L, "n_obs_receivers": int(n_obs_receivers)}, indent=2)
    )
    (output_dir / "loss_curve.json").write_text(json.dumps(loss_curve, indent=2))

    # ---------- Figures ----------

    # 1. zero_shot_overlay (predicted vs ISM, one held-out receiver — centre)
    fig, ax = plt.subplots(figsize=(11, 4))
    mag_db_pred = 20 * np.log10(np.maximum(np.abs(H_pred[centre_idx]), eps))
    mag_db_ism = 20 * np.log10(np.maximum(np.abs(H_target[centre_idx]), eps))
    ax.plot(f_axis, mag_db_ism, color="steelblue", lw=0.5, label="ISM target")
    ax.plot(f_axis, mag_db_pred, color="indianred", lw=0.5, label="zero-shot predicted")
    ax.set_xlim(0, f_max)
    ax.set_xlabel("f (Hz)"); ax.set_ylabel("|H| (dB)")
    ax.set_title(f"L={L:.2f} m (UNSEEN) — held-out centre receiver  |  held-out LSD={held_lsd:.2f} dB")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "zero_shot_overlay.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 2. zero_shot_modal_tracking (0–200 Hz overlay with analytical ticks)
    fig, ax = plt.subplots(figsize=(11, 4))
    plot_modal_overlay(H_pred[centre_idx], f_axis, modes, peaks_pred, ax,
                       title=f"L={L:.2f} m UNSEEN — predicted |H(f)| 0–200 Hz",
                       f_min=0, f_max=200, db_floor=-100)
    fig.tight_layout()
    fig.savefig(fig_dir / "zero_shot_modal_tracking.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 3. zero_shot_receiver_grid (8x8, predicted vs ISM; obs marked)
    n = 8
    fig, axs = plt.subplots(n, n, figsize=(11, 9), sharex=True)
    obs_set = set(obs_idx.tolist())
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
            if idx in obs_set:
                # Yellow border = observed receiver
                for s in ax.spines.values():
                    s.set_edgecolor("gold"); s.set_linewidth(2.0)
            if iy == n - 1:
                ax.set_xlabel(f"x={receiver_pos[idx,0]:.1f}", fontsize=6)
            if ix == 0:
                ax.set_ylabel(f"y={receiver_pos[idx,1]:.1f}", fontsize=6, rotation=0, labelpad=12)
    fig.suptitle(
        f"L={L:.2f} m UNSEEN — receiver grid (gold border = 8 observed; "
        f"56 held-out shown predicted vs ISM)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(fig_dir / "zero_shot_receiver_grid.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 4. adapt_loss_curve
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
        ax1.set_title(f"L={L:.2f} m UNSEEN — z_star adaptation (final norm {z_star_tensor.norm():.3f})")
        fig.tight_layout()
        fig.savefig(fig_dir / "adapt_loss_curve.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, required=True)
    ap.add_argument("--train_output_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--n_adapt_iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--data_dir", type=str, default="data/track_a")
    ap.add_argument("--n_obs_receivers", type=int, default=8)
    ap.add_argument("--init_strategy", type=str, default="random",
                    choices=["random", "nearest_train", "simplex"])
    ap.add_argument("--n_restarts", type=int, default=1)
    ap.add_argument("--random_seed", type=int, default=0)
    args = ap.parse_args()

    L = float(args.L)
    target_h5 = Path(args.data_dir) / room_filename(L=L, W=4.0, alpha=0.15)
    if not target_h5.exists():
        raise FileNotFoundError(f"missing target room HDF5: {target_h5}")

    out = zero_shot_adapt(
        train_output_dir=args.train_output_dir,
        target_h5=target_h5,
        output_dir=args.output_dir,
        n_adapt_iters=args.n_adapt_iters,
        lr=args.lr,
        n_obs_receivers=args.n_obs_receivers,
        init_strategy=args.init_strategy,
        n_restarts=args.n_restarts,
        random_seed=args.random_seed,
    )
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}, indent=2))


if __name__ == "__main__":
    main()
