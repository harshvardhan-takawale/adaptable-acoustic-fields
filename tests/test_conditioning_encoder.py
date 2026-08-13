"""P3-2 conditioning encoder: shape, range, block layout, and material identity.

CPU-only (imports torch + the dependency-free aaf.walls; never tcnn).
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from aaf.data.mat_configs import make_config
from aaf.models.conditioning_2d import (  # noqa: E402
    COND_SOURCE,
    FOURIER_DIM_2D,
    N_K_ALPHA,
    N_K_GEOM,
    build_cond_vector_2d,
    fourier_features_2d,
    normalize_params_2d,
)
from aaf.walls import ALPHA_NORM, WALLS_2D, alphas_for  # noqa: E402


def test_dim_is_64_and_matches_layout():
    assert FOURIER_DIM_2D == 64
    assert FOURIER_DIM_2D == 2 * 2 * N_K_GEOM + len(WALLS_2D) * 2 * N_K_ALPHA
    f = fourier_features_2d(4.5, 4.0, alphas_for("west", "M3"))
    assert f.shape == (FOURIER_DIM_2D,)
    assert torch.isfinite(f).all()
    assert float(f.abs().max()) <= 1.0 + 1e-6


def test_normalization_constants():
    u = normalize_params_2d(6.0, 5.0, (0.7, 0.7, 0.7, 0.7))
    assert u.shape == (6,)
    assert u[0] == pytest.approx(1.0)      # (6-3)/3
    assert u[1] == pytest.approx(1.0)      # (5-3)/2
    assert all(float(x) == pytest.approx(1.0) for x in u[2:])   # 0.7 / ALPHA_NORM
    u0 = normalize_params_2d(3.0, 3.0, (0.0, 0.0, 0.0, 0.0))
    assert all(float(x) == pytest.approx(0.0) for x in u0)
    assert ALPHA_NORM == 0.7


def test_block_offsets_are_load_bearing():
    """[0:16]=L [16:32]=W [32:40]=a_west [40:48]=a_east [48:56]=a_south [56:64]=a_north."""
    L, W = 4.5, 4.0
    alphas = (0.05, 0.50, 0.70, 0.15)          # distinct per wall so blocks can't alias
    u = normalize_params_2d(L, W, alphas)
    f = fourier_features_2d(L, W, alphas)
    for blk, (start, n_k) in enumerate(
        [(0, N_K_GEOM), (16, N_K_GEOM), (32, N_K_ALPHA), (40, N_K_ALPHA),
         (48, N_K_ALPHA), (56, N_K_ALPHA)]
    ):
        for k in range(n_k):
            ang = (2.0 ** k) * math.pi * float(u[blk])
            assert float(f[start + k]) == pytest.approx(math.sin(ang), abs=1e-5)
            assert float(f[start + n_k + k]) == pytest.approx(math.cos(ang), abs=1e-5)


def test_wall_slot_order_matches_walls_2d():
    """alpha_west must land in slot 2 of u, in WALLS_2D order."""
    for i, wall in enumerate(WALLS_2D):
        u = normalize_params_2d(4.5, 4.0, alphas_for(wall, "M3"))
        assert float(u[2 + i]) == pytest.approx(0.70 / ALPHA_NORM), wall
        others = [float(u[2 + j]) for j in range(len(WALLS_2D)) if j != i]
        assert all(o == pytest.approx(0.15 / ALPHA_NORM) for o in others), wall


def test_editing_a_wall_to_baseline_is_identical_to_baseline():
    """Consistency check C3 at the encoding level: (wall=k, M0) IS the baseline."""
    base = fourier_features_2d(4.3, 3.7, alphas_for())
    for wall in WALLS_2D:
        assert torch.equal(base, fourier_features_2d(4.3, 3.7, alphas_for(wall, "M0")))
        # and the config layer collapses it to a baseline config too
        assert make_config(4.3, 3.7, wall=wall, material="M0").is_baseline


def test_materials_stay_distinguishable():
    feats = {m: fourier_features_2d(4.5, 4.0, alphas_for("east", m))
             for m in ("M0", "M1", "M2", "M3")}
    keys = sorted(feats)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            d = float((feats[keys[i]] - feats[keys[j]]).norm())
            assert d > 0.5, f"{keys[i]} vs {keys[j]} too close: {d}"


def test_deterministic_and_dispatch():
    a = alphas_for("south", "M2")
    assert torch.equal(fourier_features_2d(4.1, 3.3, a), fourier_features_2d(4.1, 3.3, a))
    v = build_cond_vector_2d(COND_SOURCE, 4.1, 3.3, a)
    assert torch.equal(v, fourier_features_2d(4.1, 3.3, a))
    with pytest.raises(ValueError):
        build_cond_vector_2d("nope", 4.1, 3.3, a)
    with pytest.raises(ValueError):
        build_cond_vector_2d(COND_SOURCE, 4.1, 3.3, None)
    with pytest.raises(ValueError):
        normalize_params_2d(4.1, 3.3, (0.1, 0.2, 0.3))     # wrong length
