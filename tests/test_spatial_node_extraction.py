"""Spatial mode extraction (Chunk 3.7 V0): grid layout + correlation sanity."""
from __future__ import annotations

import numpy as np
import pytest

from aaf.eval.spatial_modes import (
    analytical_mode_shape,
    bin_index_for_freq,
    extract_pressure_field,
    mode_shape_fit_error,
    node_match_score,
    pick_first_modes,
    receiver_grid_xy,
    spatial_correlation_complex,
)


def test_bin_index_basic():
    # fs=4096, n_time=8192 → df = 0.5 Hz, n_freq_bins=4097.
    assert bin_index_for_freq(0.0,   fs=4096.0, n_freq_bins=4097) == 0
    assert bin_index_for_freq(0.5,   fs=4096.0, n_freq_bins=4097) == 1
    assert bin_index_for_freq(100.0, fs=4096.0, n_freq_bins=4097) == 200
    # Clamped.
    assert bin_index_for_freq(99999.0, fs=4096.0, n_freq_bins=4097) == 4096


def test_extract_pressure_field_layout():
    """Synthesize a (1, 0) axial mode at one bin; assert recovery and orientation."""
    L, W = 4.25, 4.0
    n_freq = 4097
    fs = 4096.0
    n_time = 2 * (n_freq - 1)
    f_target = 40.0  # arbitrary
    bin_idx = bin_index_for_freq(f_target, fs, n_freq)

    Phi = analytical_mode_shape(1, 0, L=L, W=W)        # [8, 8] real
    H = np.zeros((64, n_freq), dtype=np.complex64)
    H[:, bin_idx] = Phi.ravel().astype(np.complex64)   # (iy, ix) flattened

    P = extract_pressure_field(H, f_target, fs, n_freq)
    assert P.shape == (8, 8)
    # The extracted [8,8] should be exactly Phi (no rotation/transpose).
    assert np.allclose(P.real, Phi)
    assert np.allclose(P.imag, 0.0)


def test_spatial_correlation_self_is_one():
    P = analytical_mode_shape(1, 0, L=4.25, W=4.0).astype(np.complex128) * (1 + 2j)
    assert spatial_correlation_complex(P, P) == pytest.approx(1.0, abs=1e-9)


def test_spatial_correlation_random_is_near_zero():
    rng = np.random.default_rng(0)
    P_ism = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    P_other = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    # Two independent random 64-vectors have |corr| ~ 1/sqrt(64) ≈ 0.125.
    assert spatial_correlation_complex(P_ism, P_other) < 0.35


def test_spatial_correlation_phase_invariant():
    """Multiplying P_pred by a global complex factor must not change correlation."""
    P_ism = analytical_mode_shape(2, 1, L=5.0, W=4.0).astype(np.complex128)
    P_pred = P_ism * (0.3 - 1.7j)
    assert spatial_correlation_complex(P_pred, P_ism) == pytest.approx(1.0, abs=1e-9)


def test_node_match_score_perfect_when_pred_is_ism():
    # The 8×8 discretisation rarely hits -20 dB on these mode shapes; relax to
    # -10 dB to ensure the test exercises the comparison branch.
    P = analytical_mode_shape(2, 0, L=4.25, W=4.0).astype(np.complex128)
    score = node_match_score(P, P, threshold_db=-10.0)
    assert score == pytest.approx(1.0, abs=1e-9)


def test_node_match_score_zero_when_pred_constant():
    """A flat predicted field has NO nodes, so 0 of the ISM-node positions match."""
    P_ism = analytical_mode_shape(2, 0, L=4.25, W=4.0).astype(np.complex128)
    P_pred = np.ones_like(P_ism)
    score = node_match_score(P_pred, P_ism, threshold_db=-10.0)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_node_match_score_nan_when_no_ism_nodes():
    """At -20 dB the (1,0) shape on this discretisation has no node positions
    → function returns nan rather than a meaningless 0/0."""
    P = analytical_mode_shape(1, 0, L=4.25, W=4.0).astype(np.complex128)
    score = node_match_score(P, P, threshold_db=-20.0)
    assert np.isnan(score)


def test_mode_shape_fit_error_self_is_perfect():
    P = analytical_mode_shape(1, 1, L=4.0, W=4.0).astype(np.complex128) * 2.7
    fit = mode_shape_fit_error(P, 1, 1, L=4.0, W=4.0)
    assert fit["snr_db"] > 80.0   # essentially infinite
    assert abs(fit["alpha"] - 2.7) < 1e-6


def test_pick_first_modes_drops_dc_and_orders_by_frequency():
    modes = pick_first_modes(L=4.25, W=4.0, n_modes=4, f_min=1.0, f_max=150.0)
    assert 1 <= len(modes) <= 4
    # Strictly increasing frequencies.
    fs = [f for _, _, f in modes]
    assert all(fs[i] < fs[i + 1] for i in range(len(fs) - 1))
    # (0, 0) should not appear.
    assert all((nx, ny) != (0, 0) for nx, ny, _ in modes)


def test_receiver_grid_layout_matches_build_datasets():
    """build_datasets.py uses `for y in ys: for x in xs`. The (X, Y) helper here
    should produce the same flat sequence when ravelled."""
    L, W = 4.25, 4.0
    n = 8
    margin = 0.5
    X, Y = receiver_grid_xy(L=L, W=W, n_grid=n, margin=margin)
    # Manual reconstruction matching build_datasets.py:60-63.
    xs = np.linspace(margin, L - margin, n)
    ys = np.linspace(margin, W - margin, n)
    expected = np.array([[x, y] for y in ys for x in xs])
    actual = np.stack([X.ravel(), Y.ravel()], axis=1)
    assert np.allclose(actual, expected)
