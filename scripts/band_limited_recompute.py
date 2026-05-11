"""Track A: recompute band-limited zero-shot LSDs from saved z_star.pt.

For each (run, L) in the existing R0/R6/R7/R8 zero-shot output dirs, this
script:

1. Loads the run's checkpoint and the saved z_star.pt for that L.
2. Re-forwards through the model+renderer to produce H_pred for all 64 receivers
   (no inner-loop adaptation — z_star is fixed at its saved value).
3. Computes band-limited LSDs (0-250 Hz, 250-500 Hz, 500-2000 Hz, full) for the
   held-out 56 receivers and writes ``band_limited_metrics.json`` next to the
   existing ``metrics.json``.

This is the cheap version of Track A: it reuses the saved z_star, so it
matches the existing held-out predictions exactly. The only cost is one forward
pass per (run, L). Running on a single scavenger GPU finishes in <15 min.

Pre-Chunk-3.6 zero-shot runs don't save ``H_pred_all.pt`` so the model+renderer
forward is unavoidable for those legacy outputs. Chunk-3.6 zero-shot runs do
save H_pred_all and could be band-recomputed without GPU (but that's a future
optimisation; this script handles both paths uniformly via re-forward).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from aaf.data.dataset_builder import read_room_h5, room_filename
from aaf.eval.band_limited import DEFAULT_BANDS, compute_band_limited_metrics
from aaf.eval.zero_shot import _load_trained_model, select_obs_indices
from aaf.renderers.freq_2d import FreqRenderer2D


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS = ("R0_central", "R6_tiny_lhead", "R7_medium_hash", "R8_tiny_latent")
DEFAULT_LS = (3.25, 3.75, 4.25, 4.75, 5.25, 5.75)


def _recompute_one(
    run_dir: Path,
    zs_dir: Path,
    target_h5: Path,
    n_obs_receivers: int,
    bands: tuple,
    device: str,
) -> dict:
    model, train_meta = _load_trained_model(run_dir, device=device)
    cfg = train_meta["cfg"]
    fs = float(cfg["fs"])
    n_time = int(cfg["n_time_samples"])
    n_freq = n_time // 2 + 1

    z_star = torch.load(zs_dir / "z_star.pt", map_location=device)
    if z_star.dim() == 0 or z_star.numel() != int(cfg["latent_dim"]):
        raise ValueError(
            f"z_star.pt at {zs_dir} has unexpected shape {tuple(z_star.shape)}; "
            f"expected ({int(cfg['latent_dim'])},)"
        )
    z_star = z_star.to(device)

    rt = read_room_h5(target_h5)
    attrs = rt["attrs"]
    L = float(attrs["L"])
    W = float(attrs["W"])
    receiver_pos = np.asarray(attrs["receiver_pos"], dtype=np.float32)
    H_target = rt["ism_H"].astype(np.complex64)
    src = np.asarray(attrs["source_pos"], dtype=np.float32)

    obs_idx = select_obs_indices(int(n_obs_receivers), total=receiver_pos.shape[0])
    held_idx = np.array(
        [i for i in range(receiver_pos.shape[0]) if i not in set(obs_idx.tolist())],
        dtype=np.int64,
    )

    renderer = FreqRenderer2D(
        n_azi=int(cfg["n_azi"]),
        n_pts_per_ray=int(cfg["n_pts_per_ray"]),
        near=float(cfg["near"]),
        fs=int(fs),
        n_time_samples=n_time,
        c=float(cfg["c"]),
        use_geometric_attn=False,
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rx_all = torch.from_numpy(receiver_pos).to(device)
    tx_all = torch.from_numpy(np.tile(src, (rx_all.size(0), 1))).to(device)
    room_min = torch.tensor([0.0, 0.0], device=device)
    room_max = torch.tensor([L, W], device=device)

    H_pred_chunks = []
    chunk = 8
    z_s_static = z_star.unsqueeze(0)
    with torch.no_grad():
        for s in range(0, rx_all.size(0), chunk):
            rx_sub = rx_all[s : s + chunk]
            tx_sub = tx_all[s : s + chunk]
            z_s = z_s_static.expand(rx_sub.size(0), -1)
            H_pred = renderer(model, rx_sub, tx_sub, room_min, room_max, z_s=z_s)
            H_pred_chunks.append(H_pred.cpu().numpy())
    H_pred_all = np.concatenate(H_pred_chunks, axis=0).astype(np.complex64)

    band_held = compute_band_limited_metrics(
        H_pred_all[held_idx], H_target[held_idx], fs, n_freq, bands,
    )
    band_obs = compute_band_limited_metrics(
        H_pred_all[obs_idx], H_target[obs_idx], fs, n_freq, bands,
    )

    eps = 1e-8
    full_held_lsd = float(np.mean(np.abs(20 * np.log10(
        np.maximum(np.abs(H_pred_all[held_idx]), eps)
        / np.maximum(np.abs(H_target[held_idx]), eps)
    ))))
    return {
        "L": L, "W": W,
        "n_obs_receivers": int(n_obs_receivers),
        "obs_idx": obs_idx.tolist(),
        "held_idx": held_idx.tolist(),
        "held": band_held,
        "obs": band_obs,
        "full_band_held_lsd_db_check": full_held_lsd,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", type=str,
                    default=str(REPO_ROOT / "outputs/multi_room/sweep"))
    ap.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "data/track_a"))
    ap.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    ap.add_argument("--Ls", nargs="+", type=float, default=list(DEFAULT_LS))
    ap.add_argument("--n_obs_receivers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    data_dir = Path(args.data_dir)
    bands = DEFAULT_BANDS

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; pass --device cpu (slow, untested).")

    failures: list[str] = []
    for run_id in args.runs:
        run_dir = sweep_root / run_id
        if not (run_dir / "train_meta.json").exists():
            print(f"# skipping {run_id}: no train_meta.json")
            continue
        for L in args.Ls:
            zs_dir = run_dir / "zero_shot" / f"L{L}"
            zstar_path = zs_dir / "z_star.pt"
            if not zstar_path.exists():
                print(f"# skipping {run_id} L={L}: no z_star.pt")
                continue
            target_h5 = data_dir / room_filename(L=L, W=4.0, alpha=0.15)
            if not target_h5.exists():
                print(f"# skipping {run_id} L={L}: missing target {target_h5.name}")
                continue
            try:
                out = _recompute_one(
                    run_dir=run_dir, zs_dir=zs_dir, target_h5=target_h5,
                    n_obs_receivers=args.n_obs_receivers, bands=bands, device=args.device,
                )
            except Exception as e:
                msg = f"{run_id} L={L}: {type(e).__name__}: {e}"
                failures.append(msg)
                print(f"# FAIL: {msg}")
                continue
            (zs_dir / "band_limited_metrics.json").write_text(
                json.dumps(out, indent=2)
            )
            held = out["held"]
            print(
                f"# {run_id} L={L:.2f}  "
                f"0-250: {held['lsd_band_0_250_db']:.2f}  "
                f"250-500: {held['lsd_band_250_500_db']:.2f}  "
                f"500-2000: {held['lsd_band_500_2000_db']:.2f}  "
                f"full: {held['lsd_band_0_2000_db']:.2f}"
            )
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
