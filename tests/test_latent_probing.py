"""Latent probing: PCA recovers known 1D manifold; random latents have high intrinsic_dim."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _write_minimal_train_artifacts(root: Path, latent_dim: int, n_rooms: int, Ls):
    """Create a fake train_meta.json + a fake checkpoint with a known latent table.

    The checkpoint is constructed so it can be loaded by torch.load(map_location='cpu').
    """
    import torch
    root.mkdir(parents=True, exist_ok=True)
    # Synthetic latents lying on a 1D manifold (linear in L).
    direction = np.random.RandomState(0).randn(latent_dim).astype(np.float32)
    direction /= np.linalg.norm(direction)
    latents = np.outer(np.array(Ls, dtype=np.float32), direction)        # [n_rooms, latent_dim]
    latents += np.random.RandomState(1).randn(n_rooms, latent_dim).astype(np.float32) * 0.01

    # Fake train_meta.
    meta = {
        "n_rooms": n_rooms,
        "L_list": Ls,
        "n_iters_target": 1, "n_iters_actual": 1, "stopped_early": False,
        "stop_iter": None, "stop_reason": "",
        "wall_clock_seconds": 0.0,
        "cfg": {
            "n_iters": 1, "batch_size": 1, "lr_network": 2e-4, "lr_latent": 1e-3,
            "weights": [1.0, 1.0, 1.0, 0.1], "lambda_latent": 1e-4,
            "n_azi": 8, "n_pts_per_ray": 4, "near": 1e-3,
            "fs": 4096, "n_time_samples": 256, "c": 343.0,
            "val_every": 1, "ckpt_every": 1, "log_every": 1,
            "early_stop_warmup": 1, "early_stop_patience": 1,
            "early_stop_min_rel_improvement": 0.01, "seed": 0,
            "latent_dim": latent_dim,
        },
    }
    (root / "train_meta.json").write_text(json.dumps(meta))

    # Stub state_dict that only contains the latent table — probe loads model
    # weights into INR2D_AutoDecoder, but tcnn requires CUDA at instantiation,
    # so this test path is GPU-only. We mock by providing a real state_dict
    # whose `latents.weight` matches our synthetic latents.
    return latents, meta


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="tcnn requires CUDA")
def test_pca_recovers_1d_manifold(tmp_path: Path):
    import torch

    from aaf.models.inr_2d import INR2D_AutoDecoder
    from aaf.eval.latent_probing import probe_latents

    latent_dim, n_freq = 16, 129
    Ls_train = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    Ls_test = [3.25, 3.75, 4.25]
    latents, _ = _write_minimal_train_artifacts(tmp_path / "train", latent_dim, len(Ls_train), Ls_train)

    # Build an actual model (CUDA), inject the synthetic latents, save a checkpoint.
    model = INR2D_AutoDecoder(n_rooms=len(Ls_train), latent_dim=latent_dim,
                              n_freq_bins=n_freq).cuda()
    with torch.no_grad():
        model.latents.weight.copy_(torch.from_numpy(latents).cuda())
    state = {"iter": 1, "model": model.state_dict(),
             "optimizer": {}, "scheduler": {}}
    torch.save(state, tmp_path / "train" / "ckpt_iter0000001.pt")
    # Override n_freq_bins in the saved meta so the loader rebuilds with this size.
    meta_path = tmp_path / "train" / "train_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["cfg"]["n_time_samples"] = 2 * (n_freq - 1)
    meta_path.write_text(json.dumps(meta))

    # Synthesize zero-shot latents on the same 1D manifold.
    direction = (latents[1] - latents[0])
    direction = direction / np.linalg.norm(direction)
    zs_root = tmp_path / "train" / "zero_shot"
    for L in Ls_test:
        d = zs_root / f"L{L}"
        d.mkdir(parents=True)
        z_test = (L * direction + np.random.RandomState(2).randn(latent_dim) * 0.01).astype(np.float32)
        torch.save(torch.from_numpy(z_test), d / "z_star.pt")

    out_dir = tmp_path / "probe"
    res = probe_latents(train_output_dir=tmp_path / "train",
                        output_dir=out_dir, zero_shot_root=zs_root)

    assert res["pc1_vs_L_r2"] > 0.95, f"R²={res['pc1_vs_L_r2']:.3f}"
    assert res["intrinsic_dim_95pct"] == 1, f"intrinsic_dim={res['intrinsic_dim_95pct']}"
    # Figures and JSON exist.
    assert (out_dir / "latent_probe.json").exists()
    assert (out_dir / "figures" / "latent_pca_1d.png").exists()
    assert (out_dir / "figures" / "latent_pca_2d.png").exists()
    assert (out_dir / "figures" / "latent_variance.png").exists()


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="tcnn requires CUDA")
def test_random_latents_high_intrinsic_dim(tmp_path: Path):
    import torch

    from aaf.models.inr_2d import INR2D_AutoDecoder
    from aaf.eval.latent_probing import probe_latents

    latent_dim, n_freq = 16, 129
    Ls_train = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)
    rng = np.random.RandomState(42)
    latents = rng.randn(len(Ls_train), latent_dim).astype(np.float32)

    meta = {
        "n_rooms": len(Ls_train), "L_list": Ls_train,
        "n_iters_target": 1, "n_iters_actual": 1, "stopped_early": False,
        "stop_iter": None, "stop_reason": "", "wall_clock_seconds": 0.0,
        "cfg": {"latent_dim": latent_dim, "n_time_samples": 2 * (n_freq - 1)},
    }
    (train_dir / "train_meta.json").write_text(json.dumps(meta))
    model = INR2D_AutoDecoder(n_rooms=len(Ls_train), latent_dim=latent_dim,
                              n_freq_bins=n_freq).cuda()
    with torch.no_grad():
        model.latents.weight.copy_(torch.from_numpy(latents).cuda())
    state = {"iter": 1, "model": model.state_dict(), "optimizer": {}, "scheduler": {}}
    torch.save(state, train_dir / "ckpt_iter0000001.pt")

    out_dir = tmp_path / "probe"
    res = probe_latents(train_output_dir=train_dir, output_dir=out_dir,
                        zero_shot_root=tmp_path / "train" / "zero_shot")
    assert res["intrinsic_dim_95pct"] >= 4, (
        f"random {latent_dim}D latents should have high intrinsic_dim, got "
        f"{res['intrinsic_dim_95pct']}"
    )
    assert res["pc1_vs_L_r2"] < 0.6, f"random latents shouldn't fit L well; R²={res['pc1_vs_L_r2']:.3f}"
