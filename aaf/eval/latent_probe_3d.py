"""3D latent manifold probe.

Generalization of `aaf.eval.latent_probing` to 3 geometry axes (L, W, H).

Loads the 45 trained latents from `INR3D_AutoDecoder.latents.weight` plus the
N test-room zero-shot `z_star.pt` files, runs sklearn PCA on the combined
set, and reports per-axis regression R² in two ways:

  - **R²_full**: R² of fitting axis_value vs the full latent (multi-variate
    linear regression). The right measure of "does the latent encode this axis"
    — captures geometry encoded in any direction.
  - **R²_best_PC**: R² of fitting axis_value vs the single best PC. Tells you
    how aligned the manifold's principal directions are with (L, W, H).

If R²_full is high but R²_best_PC is low for an axis, the axis is encoded but
spread across multiple PCs (entangled). If both are high, the manifold is
disentangled.

Headline expectations:
  - Intrinsic dim (95% variance) ≤ 5: the 3D geometry manifold is low-dim.
  - R²_full per axis ≥ 0.85: latent encodes (L, W, H).

CLI
---
    python -m aaf.eval.latent_probe_3d \
        --train-output-dir outputs/multi_room_3d/M1_45rooms \
        --output_dir outputs/multi_room_3d/M1_45rooms/latent_probe
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from aaf.models.inr_3d import INR3D_AutoDecoder


# Parse "L4.50_W4.00_H3.25" style directory name.
LWH_RE = re.compile(r"^L([0-9.]+)_W([0-9.]+)_H([0-9.]+)$")


def _load_train_latents_and_dims(
    train_output_dir: Path, device: str = "cuda"
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (latents [n_rooms, d], dims [n_rooms, 3] with columns L, W, H)."""
    train_meta = json.loads((train_output_dir / "train_meta.json").read_text())
    n_rooms = int(train_meta["n_rooms"])
    cfg = train_meta["cfg"]
    n_freq_bins = int(cfg["n_time_samples"]) // 2 + 1
    L_list = train_meta["L_list"]
    W_list = train_meta["W_list"]
    H_list = train_meta["H_list"]
    dims = np.asarray(
        list(zip(L_list, W_list, H_list)),
        dtype=np.float32,
    )

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
            print(f"[probe-3d] skipping corrupted ckpt {p.name}: {e!r}")
    if state is None:
        raise FileNotFoundError(f"no usable checkpoint in {train_output_dir}")
    model.load_state_dict(state["model"])
    latents = model.latents.weight.detach().cpu().numpy().astype(np.float32)
    return latents, dims


def _load_zero_shot_latents(zero_shot_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Walk `zero_shot_root/L*_W*_H*/z_star.pt`. Returns (latents, dims)."""
    triples: list[tuple[float, float, float, np.ndarray]] = []
    for d in sorted(zero_shot_root.glob("L*_W*_H*/")):
        m = LWH_RE.match(d.name)
        if m is None:
            print(f"[probe-3d] cannot parse L/W/H from {d.name}")
            continue
        L, W, H = (float(x) for x in m.groups())
        z_path = d / "z_star.pt"
        if not z_path.exists():
            continue
        try:
            z = torch.load(z_path, map_location="cpu").numpy().astype(np.float32)
        except Exception as e:
            print(f"[probe-3d] failed to load {z_path}: {e!r}")
            continue
        triples.append((L, W, H, z))
    triples.sort(key=lambda t: (t[0], t[1], t[2]))
    if not triples:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    dims = np.asarray([(L, W, H) for L, W, H, _ in triples], dtype=np.float32)
    Z = np.stack([z for _, _, _, z in triples], axis=0)
    return Z, dims


def _r2_full_latent(Z: np.ndarray, y: np.ndarray) -> float:
    """Multivariate linear regression: fit y ≈ Z @ w + b on the full set;
    return R² on the same set (in-sample).
    """
    if Z.shape[0] < 2 or Z.shape[1] < 1:
        return float("nan")
    A = np.concatenate([Z, np.ones((Z.shape[0], 1), dtype=Z.dtype)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _r2_per_pc(z_pca: np.ndarray, y: np.ndarray) -> list[float]:
    """For each PC column, fit y ≈ a * pc + b and return R²."""
    out = []
    for k in range(z_pca.shape[1]):
        pc = z_pca[:, k]
        if y.size < 2 or np.allclose(pc, pc[0]):
            out.append(float("nan"))
            continue
        slope, intercept = np.polyfit(pc, y, 1)
        pred = slope * pc + intercept
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        out.append(1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
    return out


def probe_latents_3d(
    train_output_dir: Path,
    output_dir: Path,
    zero_shot_root: Path = None,
    device: str = "cuda",
) -> dict:
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    z_train, dims_train = _load_train_latents_and_dims(
        Path(train_output_dir), device=device
    )
    if zero_shot_root is not None and Path(zero_shot_root).exists():
        z_test, dims_test = _load_zero_shot_latents(Path(zero_shot_root))
    else:
        z_test = np.zeros((0, z_train.shape[1]), dtype=np.float32)
        dims_test = np.zeros((0, 3), dtype=np.float32)

    if z_test.size > 0 and z_test.shape[1] != z_train.shape[1]:
        raise ValueError(
            f"latent dim mismatch: train {z_train.shape[1]} vs test {z_test.shape[1]}"
        )

    z_all = np.concatenate([z_train, z_test], axis=0) if z_test.size > 0 else z_train
    dims_all = (
        np.concatenate([dims_train, dims_test], axis=0)
        if z_test.size > 0 else dims_train
    )

    # PCA on combined set.
    n_components = min(z_all.shape) - 1 if z_all.shape[0] > 1 else 1
    pca = PCA(n_components=n_components)
    z_all_pca = pca.fit_transform(z_all)
    expl_var = pca.explained_variance_ratio_.tolist()
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    intrinsic_dim = int(np.searchsorted(cum_var, 0.95) + 1)

    # Per-axis R² — TRAIN-ONLY fit + TRAIN-AND-TEST evaluation.
    axis_names = ["L", "W", "H"]
    r2_full = {}
    r2_per_pc_dict: dict[str, list[float]] = {}
    best_pc_per_axis: dict[str, dict] = {}
    for k, name in enumerate(axis_names):
        # R²_full on combined set (this is what matters for "manifold encodes axis").
        r2_full[name] = _r2_full_latent(z_all, dims_all[:, k])
        per_pc = _r2_per_pc(z_all_pca, dims_all[:, k])
        r2_per_pc_dict[name] = per_pc
        valid = [(i, r) for i, r in enumerate(per_pc) if not np.isnan(r)]
        if valid:
            best_idx, best_r2 = max(valid, key=lambda t: t[1])
            best_pc_per_axis[name] = {"pc_index": int(best_idx), "r2": float(best_r2)}
        else:
            best_pc_per_axis[name] = {"pc_index": -1, "r2": float("nan")}

    out = {
        "pca_explained_variance": expl_var,
        "intrinsic_dim_95pct": intrinsic_dim,
        "r2_per_axis_full": r2_full,
        "r2_per_axis_per_pc": r2_per_pc_dict,
        "best_pc_per_axis": best_pc_per_axis,
        "n_train": int(z_train.shape[0]),
        "n_test": int(z_test.shape[0]),
        "latent_dim": int(z_all.shape[1]),
        "dims_train_LWH": dims_train.tolist(),
        "dims_test_LWH": dims_test.tolist(),
    }
    (output_dir / "latent_probe.json").write_text(json.dumps(out, indent=2))

    # ---------- Figures ----------
    n_train = z_train.shape[0]
    pc1_train = z_all_pca[:n_train, 0]
    pc2_train = z_all_pca[:n_train, 1] if n_components >= 2 else np.zeros(n_train)
    pc3_train = z_all_pca[:n_train, 2] if n_components >= 3 else np.zeros(n_train)
    pc1_test = z_all_pca[n_train:, 0] if z_test.size > 0 else np.zeros(0)
    pc2_test = z_all_pca[n_train:, 1] if (z_test.size > 0 and n_components >= 2) else np.zeros(0)
    pc3_test = z_all_pca[n_train:, 2] if (z_test.size > 0 and n_components >= 3) else np.zeros(0)

    # Variance bar/cum curve.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(expl_var) + 1), expl_var, color="steelblue", alpha=0.6, label="per PC")
    ax.plot(
        range(1, len(expl_var) + 1), np.cumsum(expl_var),
        "o-", color="indianred", label="cumulative",
    )
    ax.axhline(0.95, color="k", lw=0.5, ls="--", alpha=0.7)
    ax.set_xlabel("PC index"); ax.set_ylabel("explained variance ratio")
    ax.set_title(f"Latent PCA — intrinsic_dim (95%) = {intrinsic_dim}")
    ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "latent_variance.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 3-panel scatter of (PC1, PC2) colored by L, W, H separately.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, k, name in zip(axes, range(3), axis_names):
        sc = ax.scatter(
            pc1_train, pc2_train, c=dims_train[:, k], cmap="viridis", s=80,
            edgecolors="black", lw=0.4, label="train",
        )
        if z_test.size > 0:
            ax.scatter(
                pc1_test, pc2_test, c=dims_test[:, k], cmap="viridis", s=110,
                marker="^", edgecolors="indianred", lw=1.5, label="test",
            )
        plt.colorbar(sc, ax=ax, label=f"{name} (m)")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title(f"colored by {name}")
        ax.grid(True, alpha=0.3)
        if z_test.size > 0:
            ax.legend(loc="best", fontsize=8)
    fig.suptitle("Latent PCA (PC1, PC2) projected by each geometry axis")
    fig.tight_layout()
    fig.savefig(fig_dir / "latent_pca_3d_by_axis.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Per-axis R² bar chart.
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_pos = np.arange(3)
    width = 0.35
    r2_full_vals = [r2_full[n] if not np.isnan(r2_full[n]) else 0.0 for n in axis_names]
    r2_best_vals = [
        best_pc_per_axis[n]["r2"] if not np.isnan(best_pc_per_axis[n]["r2"]) else 0.0
        for n in axis_names
    ]
    ax.bar(bar_pos - width / 2, r2_full_vals, width, label="R²_full (multi-PC)",
           color="steelblue")
    ax.bar(bar_pos + width / 2, r2_best_vals, width, label="R²_best PC",
           color="indianred")
    ax.set_xticks(bar_pos); ax.set_xticklabels(axis_names)
    ax.set_ylabel("R²"); ax.set_ylim(0, 1.05)
    ax.axhline(0.85, color="k", lw=0.5, ls="--", alpha=0.7, label="P2-2 expectation (0.85)")
    ax.set_title("Latent encodes geometry? Per-axis R² (linear regression)")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "latent_per_axis_r2.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-output-dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--zero-shot-root", type=str, default=None)
    args = ap.parse_args()

    train_dir = Path(args.train_output_dir)
    zs_root = (
        Path(args.zero_shot_root)
        if args.zero_shot_root else (train_dir / "zero_shot")
    )
    out = probe_latents_3d(
        train_output_dir=train_dir,
        output_dir=Path(args.output_dir),
        zero_shot_root=zs_root,
    )
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, (list, dict))},
                     indent=2))


if __name__ == "__main__":
    main()
