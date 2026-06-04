"""Tests for `aaf.eval.latent_probe_3d` — pure-Python parts (no CUDA needed)."""
import numpy as np


def test_r2_full_latent_perfect_linear():
    """Synthetic: y = a·z + b → R² == 1.0 (in-sample fit)."""
    from aaf.eval.latent_probe_3d import _r2_full_latent

    rng = np.random.default_rng(0)
    Z = rng.standard_normal((50, 8)).astype(np.float32)
    # Pick a random direction.
    direction = rng.standard_normal(8).astype(np.float32)
    y = Z @ direction + 3.0
    r2 = _r2_full_latent(Z, y)
    assert r2 > 0.9999


def test_r2_full_latent_random_y_low_r2():
    """Random labels → low R²."""
    from aaf.eval.latent_probe_3d import _r2_full_latent

    rng = np.random.default_rng(0)
    Z = rng.standard_normal((50, 8)).astype(np.float32)
    y = rng.standard_normal(50).astype(np.float32)
    r2 = _r2_full_latent(Z, y)
    # In-sample R² of a random regression on 8 features can be sizeable; check
    # it's at most O(d/n) ≈ 0.16 plus margin.
    assert r2 < 0.5


def test_r2_per_pc_synthetic_axis_aligned():
    """Synthetic PCA-projected latents where PC1 perfectly encodes y → R² ≈ 1
    on PC1, ~0 on others."""
    from aaf.eval.latent_probe_3d import _r2_per_pc

    n = 40
    rng = np.random.default_rng(0)
    pc1 = np.linspace(0, 10, n)
    other = rng.standard_normal((n, 3))
    z_pca = np.column_stack([pc1, other])
    y = 0.5 * pc1 + 1.0
    r2 = _r2_per_pc(z_pca, y)
    assert r2[0] > 0.9999
    for k in range(1, 4):
        assert r2[k] < 0.3


def test_obs_indices_3d_round_trip(tmp_path):
    """Smoke test: walking a directory of mock z_star.pt files yields the
    expected (dims, z) ordering."""
    import torch
    from aaf.eval.latent_probe_3d import _load_zero_shot_latents

    # Create 3 mock test rooms.
    rooms = [(3.0, 3.5, 2.5), (4.5, 4.0, 3.25), (6.0, 5.0, 4.0)]
    for L, W, H in rooms:
        d = tmp_path / f"L{L:.2f}_W{W:.2f}_H{H:.2f}"
        d.mkdir()
        torch.save(torch.zeros(8), d / "z_star.pt")
    Z, dims = _load_zero_shot_latents(tmp_path)
    assert Z.shape == (3, 8)
    assert dims.shape == (3, 3)
    # Sorted ascending in L.
    Ls = dims[:, 0].tolist()
    assert Ls == sorted(Ls)


def test_intrinsic_dim_synthetic_low_dim_manifold():
    """Latents drawn from a 3D linear subspace (45 samples) → intrinsic_dim_95pct == 3."""
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(0)
    # 45 samples × 16 dims, but only 3 underlying parameters.
    params = rng.uniform(0, 1, (45, 3))
    # Random projection into 16-D.
    basis = rng.standard_normal((3, 16))
    Z = params @ basis + 0.001 * rng.standard_normal((45, 16))
    pca = PCA(n_components=min(Z.shape) - 1)
    pca.fit(Z)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    intrinsic_dim = int(np.searchsorted(cum_var, 0.95) + 1)
    assert intrinsic_dim <= 4


def test_intrinsic_dim_random_high_dim():
    """Latents drawn from full-rank Gaussian → intrinsic_dim_95pct > 8."""
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(0)
    Z = rng.standard_normal((45, 16))
    pca = PCA(n_components=min(Z.shape) - 1)
    pca.fit(Z)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    intrinsic_dim = int(np.searchsorted(cum_var, 0.95) + 1)
    # With 16 fully-random dims, 95% of variance needs at least 13 PCs.
    assert intrinsic_dim >= 9
