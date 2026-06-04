"""Tests for the 3D eigenfrequency enumerator + cubic-degeneracy handling."""
import math

import numpy as np
import pytest

from aaf.sim.analytical_modal_3d import (
    EigenFreq3D,
    eigenfrequencies_3d,
    modal_rir_3d,
    sabine_damping_3d,
)

C = 343.0


def test_first_axial_modes_match_closed_form():
    """For L=4, W=3, H=2.5 the three first axial modes have known values."""
    L, W, H = 4.0, 3.0, 2.5
    modes = eigenfrequencies_3d(L=L, W=W, H=H, c=C, f_max=200.0)
    # Drop DC.
    modes_nz = [m for m in modes if m.f > 0]
    # First axial modes: f_x = c/(2L), f_y = c/(2W), f_z = c/(2H).
    f_x = C / (2 * L)              # = 42.875
    f_y = C / (2 * W)              # = 57.166...
    f_z = C / (2 * H)              # = 68.6
    # The 3 lowest non-DC modes should be approximately these (within dedup tol).
    fs_low = sorted([m.f for m in modes_nz])[:3]
    assert math.isclose(fs_low[0], f_x, abs_tol=0.05)
    assert math.isclose(fs_low[1], f_y, abs_tol=0.05)
    assert math.isclose(fs_low[2], f_z, abs_tol=0.05)


def test_cubic_room_triple_degeneracy():
    """In a perfect cubic room L=W=H=3, the first axial modes (1,0,0), (0,1,0),
    (0,0,1) all coincide → multiplicity must be 3."""
    L = W = H = 3.0
    modes = eigenfrequencies_3d(L=L, W=W, H=H, c=C, f_max=100.0)
    nz = [m for m in modes if m.f > 0]
    # The lowest non-DC entry should have multiplicity 3 (three axial modes
    # all at c/(2L)).
    assert len(nz) > 0
    first = nz[0]
    assert first.multiplicity == 3, (
        f"expected multiplicity 3 for cubic-room first axial; got {first}"
    )
    triples = set(first.triples)
    assert triples == {(1, 0, 0), (0, 1, 0), (0, 0, 1)}


def test_dedup_tolerance_default():
    """Modes within 0.01 Hz must collapse to one EigenFreq3D entry."""
    # Pick L slightly perturbed from W so axial modes are almost identical.
    L = 4.0
    W = 4.0 + 1e-4
    H = 2.5
    modes = eigenfrequencies_3d(L=L, W=W, H=H, c=C, f_max=200.0,
                                dedup_tol_hz=0.1)
    # The 1-along-L and 1-along-W modes should merge.
    nz = [m for m in modes if m.f > 0]
    has_x = (1, 0, 0)
    has_y = (0, 1, 0)
    merged = any(
        has_x in m.triples and has_y in m.triples for m in nz
    )
    assert merged, "1-along-L and 1-along-W should merge under 0.1 Hz dedup"


def test_eigenfrequencies_returns_sorted_unique():
    modes = eigenfrequencies_3d(L=4.5, W=4.0, H=3.25, c=C, f_max=300.0)
    freqs = [m.f for m in modes]
    assert freqs == sorted(freqs)
    # Distinct entries — multiplicities encode the dups.
    assert len(set(freqs)) == len(freqs)


def test_modal_rir_3d_shape_and_rfft_symmetry():
    cfg = {
        "L": 4.5, "W": 4.0, "H": 3.25,
        "source_pos": (0.5, 0.5, 0.5),
        "receiver_pos": np.array([[2.25, 2.0, 1.5]], dtype=np.float64),
        "alpha": 0.15,
        "fs": 4096,
        "n_time_samples": 4096,
        "f_max_modes": 800.0,
    }
    out = modal_rir_3d(cfg)
    H = out["H_complex"]
    rir = out["rir_time"]
    n_freq = cfg["n_time_samples"] // 2 + 1
    assert H.shape == (1, n_freq)
    assert rir.shape == (1, cfg["n_time_samples"])
    assert np.allclose(H[0, 0].imag, 0, atol=1e-5)
    if cfg["n_time_samples"] % 2 == 0:
        assert np.allclose(H[0, -1].imag, 0, atol=1e-5)
    assert np.all(np.isfinite(rir))


def test_sabine_damping_positive():
    g = sabine_damping_3d(L=4.0, W=3.0, H=2.5, alpha=0.15, c=C)
    assert g > 0
    # T60 = 6.91 / g; for α=0.15 should be in [0.5, 1.5] s.
    t60 = 6.91 / g
    assert 0.3 < t60 < 2.0


def test_sabine_damping_bad_alpha():
    with pytest.raises(ValueError):
        sabine_damping_3d(L=4.0, W=3.0, H=2.5, alpha=1.5, c=C)
