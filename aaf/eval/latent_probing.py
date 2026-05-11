"""Latent manifold analysis for the trained INR2D_AutoDecoder.

Combines the trained per-room latents (from ``model.latents``) with the
zero-shot adapted latents (loaded from each ``zero_shot/L*/z_star.pt``),
runs PCA, fits a linear regression of PC1 vs L, and produces three
diagnostic figures.

Headline question: did the auto-decoder's latent space learn physics?
A useful answer is **(1) PC1 is monotonic in L (R² close to 1) and (2)
intrinsic_dim ≈ 1**, meaning the latents lie on a 1-D manifold parameterised
by L. That's the strong claim for the Phase-1 result.

CLI
---
    python -m aaf.eval.latent_probing \
        --train_output_dir outputs/multi_room/dense \
        --output_dir outputs/multi_room/dense/latent_probe
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
from sklearn.decomposition import PCA

from aaf.models.inr_2d import INR2D_AutoDecoder


def _load_train_latents(train_output_dir: Path, device: str = "cuda") -> tuple[np.ndarray, list[float]]:
    """Load the trained model's latent embedding and return (latents, L_per_room).

    Returns:
        latents:    [n_rooms, latent_dim] np.float32
        Ls_train:   list of L values aligned with the rows of `latents`
    """
    train_meta = json.loads((train_output_dir / "train_meta.json").read_text())
    n_rooms = int(train_meta["n_rooms"])
    cfg = train_meta["cfg"]
    n_freq_bins = int(cfg["n_time_samples"]) // 2 + 1
    Ls_train = list(train_meta["L_list"])

    # Match the trained architecture exactly (Chunk 3.5 sweep configs vary HashGrid + L-head).
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
    for p in ckpts:
        try:
            state = torch.load(p, map_location=device)
            break
        except Exception as e:
            print(f"[probe] skipping corrupted ckpt {p.name}: {e!r}")
    if state is None:
        raise FileNotFoundError(f"no usable checkpoint in {train_output_dir}")
    model.load_state_dict(state["model"])
    latents = model.latents.weight.detach().cpu().numpy().astype(np.float32)
    return latents, Ls_train


def _load_zero_shot_latents(zero_shot_root: Path) -> tuple[np.ndarray, list[float]]:
    """Walk `zero_shot_root/L*/z_star.pt`, load each, and return latents + Ls."""
    L_star_pairs: list[tuple[float, np.ndarray]] = []
    for d in sorted(zero_shot_root.glob("L*/")):
        z_path = d / "z_star.pt"
        if not z_path.exists():
            continue
        try:
            z = torch.load(z_path, map_location="cpu").numpy().astype(np.float32)
        except Exception as e:
            print(f"[probe] failed to load {z_path}: {e!r}")
            continue
        # Parse L from directory name "L3.25" → 3.25
        try:
            L = float(d.name[1:])
        except ValueError:
            print(f"[probe] cannot parse L from {d.name}")
            continue
        L_star_pairs.append((L, z))

    L_star_pairs.sort(key=lambda t: t[0])
    if not L_star_pairs:
        return np.zeros((0, 0), dtype=np.float32), []
    Ls = [L for L, _ in L_star_pairs]
    Z = np.stack([z for _, z in L_star_pairs], axis=0)
    return Z, Ls


def probe_latents(
    train_output_dir: Path,
    output_dir: Path,
    zero_shot_root: Path,
    device: str = "cuda",
) -> dict:
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    z_train, Ls_train = _load_train_latents(Path(train_output_dir), device=device)
    z_test, Ls_test = _load_zero_shot_latents(Path(zero_shot_root))
    if z_test.size > 0 and z_test.shape[1] != z_train.shape[1]:
        raise ValueError(
            f"latent dim mismatch: train {z_train.shape[1]} vs zero-shot {z_test.shape[1]}"
        )

    # PCA on the combined set (so the projection axes reflect both train + test).
    if z_test.size > 0:
        z_all = np.concatenate([z_train, z_test], axis=0)
    else:
        z_all = z_train
    n_components = min(z_all.shape) - 1 if z_all.shape[0] > 1 else 1
    pca = PCA(n_components=n_components)
    z_all_pca = pca.fit_transform(z_all)
    expl_var = pca.explained_variance_ratio_.tolist()
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    intrinsic_dim = int(np.searchsorted(cum_var, 0.95) + 1)

    # PC1 train vs test
    pc1_train = z_all_pca[: z_train.shape[0], 0].tolist()
    pc1_test = z_all_pca[z_train.shape[0]:, 0].tolist() if z_test.size > 0 else []
    pc2_train = z_all_pca[: z_train.shape[0], 1].tolist() if n_components >= 2 else []
    pc2_test = z_all_pca[z_train.shape[0]:, 1].tolist() if (z_test.size > 0 and n_components >= 2) else []

    # PC1-vs-L regression: fit on TRAIN only, then evaluate R² on (train + test).
    L_arr = np.array(Ls_train + Ls_test, dtype=np.float64)
    pc1_arr = np.array(pc1_train + pc1_test, dtype=np.float64)
    if len(Ls_train) >= 2:
        slope, intercept = np.polyfit(np.array(Ls_train), np.array(pc1_train), 1)
        pred = slope * L_arr + intercept
        ss_res = np.sum((pc1_arr - pred) ** 2)
        ss_tot = np.sum((pc1_arr - pc1_arr.mean()) ** 2)
        pc1_vs_L_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    else:
        slope, intercept, pc1_vs_L_r2 = float("nan"), float("nan"), float("nan")

    out = {
        "pca_explained_variance": expl_var,
        "intrinsic_dim_95pct": intrinsic_dim,
        "pc1_vs_L_r2": pc1_vs_L_r2,
        "slope_PC1_per_m": float(slope) if not np.isnan(slope) else float("nan"),
        "Ls_train": Ls_train,
        "Ls_test": Ls_test,
        "pc1_train": pc1_train,
        "pc1_test": pc1_test,
        "latent_dim": int(z_all.shape[1]),
        "n_train": int(z_train.shape[0]),
        "n_test": int(z_test.shape[0]),
    }
    (output_dir / "latent_probe.json").write_text(json.dumps(out, indent=2))

    # ---------- Figures ----------

    # 1. PC1 vs L (1D projection)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(Ls_train, pc1_train, s=80, color="steelblue", marker="o", label="train")
    if Ls_test:
        ax.scatter(Ls_test, pc1_test, s=80, color="indianred", marker="^", label="zero-shot test")
    if not np.isnan(slope):
        L_grid = np.linspace(min(L_arr) - 0.1, max(L_arr) + 0.1, 100)
        ax.plot(L_grid, slope * L_grid + intercept, "k--", lw=0.8,
                label=f"linear fit (R²={pc1_vs_L_r2:.3f})")
    ax.set_xlabel("L (m)"); ax.set_ylabel("PC1")
    ax.set_title("Latent PC1 vs room L  —  did z_s learn the geometry?")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "latent_pca_1d.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 2. (PC1, PC2) coloured by L
    fig, ax = plt.subplots(figsize=(7, 6))
    if n_components >= 2:
        all_L = np.array(Ls_train + Ls_test)
        all_pc1 = np.concatenate([np.array(pc1_train), np.array(pc1_test)]) if Ls_test else np.array(pc1_train)
        all_pc2 = np.concatenate([np.array(pc2_train), np.array(pc2_test)]) if Ls_test else np.array(pc2_train)
        sc = ax.scatter(all_pc1, all_pc2, c=all_L, cmap="viridis", s=80, edgecolors="black", lw=0.5)
        # Mark train/test by border colour.
        ax.scatter(np.array(pc1_train), np.array(pc2_train),
                   facecolors="none", edgecolors="steelblue", s=120, lw=1.5, label="train")
        if Ls_test:
            ax.scatter(np.array(pc1_test), np.array(pc2_test),
                       facecolors="none", edgecolors="indianred", s=120, lw=1.5, marker="^", label="test")
        plt.colorbar(sc, ax=ax, label="L (m)")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title("Latent PCA (PC1, PC2) coloured by L")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "n_components=1 → only 1D projection available",
                ha="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(fig_dir / "latent_pca_2d.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 3. Cumulative explained variance
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(expl_var) + 1), expl_var, color="steelblue", alpha=0.6, label="per PC")
    ax.plot(range(1, len(expl_var) + 1), np.cumsum(expl_var), "o-", color="indianred",
            label="cumulative")
    ax.axhline(0.95, color="k", lw=0.5, linestyle="--", alpha=0.7)
    ax.set_xlabel("PC index"); ax.set_ylabel("explained variance ratio")
    ax.set_title(f"Latent PCA variance — intrinsic_dim (95%) = {intrinsic_dim}")
    ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "latent_variance.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_output_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--zero_shot_root", type=str, default=None,
                    help="defaults to <train_output_dir>/zero_shot")
    args = ap.parse_args()

    train_dir = Path(args.train_output_dir)
    zs_root = Path(args.zero_shot_root) if args.zero_shot_root else (train_dir / "zero_shot")
    out = probe_latents(train_output_dir=train_dir,
                        output_dir=Path(args.output_dir),
                        zero_shot_root=zs_root)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}, indent=2))


if __name__ == "__main__":
    main()
