"""The -3 dB bandwidth estimator, validated against closed-form damping. Pure CPU.

The analytic modal model's per-mode damping is known exactly, so BW = gamma/pi is a real
reference rather than a regression baseline.
"""
from __future__ import annotations

import numpy as np
import pytest

from aaf.eval.modal_bandwidth import (
    CAP_LEFT,
    CAP_RIGHT,
    FLOOR,
    OK,
    caps_from_mode_spacing,
    caps_from_predicted_bw,
    measure_modes,
    peak_level_and_bw,
)
from aaf.eval.modal_projection import enumerate_modes
from aaf.sim.analytical_modal_2d import (
    C_DEFAULT,
    damping_to_bandwidth_hz,
    eigenfrequencies_2d,
    modal_damping_2d,
    modal_rir_2d,
    sabine_damping_2d,
)

FS, N = 4096, 8192
F_AXIS = np.arange(N // 2 + 1) * FS / N


def _lorentzian(f0, gamma):
    k = 2 * np.pi * F_AXIS / C_DEFAULT
    km = 2 * np.pi * f0 / C_DEFAULT
    return np.abs(1.0 / (km ** 2 - k ** 2 - 2j * gamma * km / C_DEFAULT + 1e-30))


def test_bandwidth_matches_gamma_over_pi():
    """V1: synthetic Lorentzians, including a deliberately mid-bin centre frequency."""
    for f0 in (38.11, 76.22, 114.33, 38.36):
        for gamma in (4.0, 8.0, 16.0, 32.0):
            bw, level, f_peak, flag = peak_level_and_bw(
                _lorentzian(f0, gamma), F_AXIS, f0, cap_hz=40.0)
            true_bw = damping_to_bandwidth_hz(gamma)
            assert flag == OK, (f0, gamma, flag)
            assert bw == pytest.approx(true_bw, rel=0.05), (f0, gamma, bw, true_bw)
            assert f_peak == pytest.approx(f0, abs=0.06)      # sub-bin (df = 0.5 Hz)
            assert np.isfinite(level)


def test_unresolvable_peak_returns_nan_not_a_fabricated_width():
    """gamma=2 -> BW 0.637 Hz ~ 1.3 bins: below the resolvability floor."""
    bw, _, _, flag = peak_level_and_bw(_lorentzian(38.11, 2.0), F_AXIS, 38.11, cap_hz=40.0)
    assert flag == FLOOR
    assert np.isnan(bw), "must not clamp an unresolvable width to one bin"


def test_walk_is_capped_and_flags_which_side():
    """V3: a wide peak measured with a deliberately tight cap must flag, not merge."""
    bw, _, _, flag = peak_level_and_bw(_lorentzian(100.0, 32.0), F_AXIS, 100.0, cap_hz=2.0)
    assert flag in (CAP_LEFT, CAP_RIGHT)
    assert np.isnan(bw)


def test_two_close_modes_do_not_merge():
    mag = _lorentzian(100.0, 4.0) + _lorentzian(103.0, 4.0)
    caps = caps_from_mode_spacing([100.0, 103.0])
    bw, _, _, flag = peak_level_and_bw(mag, F_AXIS, 100.0, cap_hz=caps[0])
    if flag == OK:
        assert bw < 3.0, "a merged pair would report a grossly inflated width"
    else:
        assert np.isnan(bw)


def test_cap_helpers():
    caps = caps_from_mode_spacing([10.0, 14.0, 40.0])
    assert caps[0] == pytest.approx(0.45 * 4.0)
    # predicted-BW caps must be generous enough for strongly damped modes: the M3 x-axial
    # width is ~13.8 Hz, needing a +/-6.9 Hz walk, which a spacing-based cap would refuse.
    assert caps_from_predicted_bw([13.758])[0] >= 2 * 13.758
    assert caps_from_predicted_bw([4.0])[0] == pytest.approx(12.0)   # 3x, un-clipped
    assert caps_from_predicted_bw([0.01])[0] == 4.0          # clipped to the floor
    assert caps_from_predicted_bw([100.0])[0] == 40.0        # clipped to the ceiling


def test_against_analytic_model_where_gamma_is_known_exactly():
    """V2: modal_rir_2d builds Lorentzians with gamma = sabine_damping_2d exactly."""
    L, W, alpha = 4.5, 4.0, 0.15
    rx = np.array([[L - 0.3, W - 0.3]])          # far corner: all low modes near-antinodal
    res = modal_rir_2d(dict(L=L, W=W, source_pos=np.array([0.5, 0.5]), receiver_pos=rx,
                            alpha=alpha, fs=FS, n_time_samples=N))
    mag = np.abs(res["H_complex"][0])
    true_bw = damping_to_bandwidth_hz(sabine_damping_2d(L, W, alpha))
    assert true_bw == pytest.approx(3.867, abs=0.01)

    # a well-isolated mode: (2,0) at 76.22 Hz has no close neighbour
    modes = [e for e in eigenfrequencies_2d(L, W, f_max=90.0) if e.f > 1.0]
    f20 = [e.f for e in modes if (2, 0) in e.pairs][0]
    bw, _, _, flag = peak_level_and_bw(mag, F_AXIS, f20, cap_hz=8.0)
    assert flag == OK
    assert bw == pytest.approx(true_bw, rel=0.08)


def test_measure_modes_reports_family_and_validity():
    L, W = 4.5, 4.0
    modes = [m for m in enumerate_modes(L, W, f_max=120.0)][:4]
    alphas = (0.15, 0.15, 0.15, 0.15)
    bw_pred = [damping_to_bandwidth_hz(
        modal_damping_2d(L, W, alphas, m.n_x, m.n_y, model="ism_ray")) for m in modes]
    spectra = np.stack([_lorentzian(m.f, np.pi * b) for m, b in zip(modes, bw_pred)])
    peaks = measure_modes(spectra, F_AXIS, modes, caps=caps_from_predicted_bw(bw_pred))
    assert len(peaks) == len(modes)
    for p, m, b in zip(peaks, modes, bw_pred):
        assert p.family == m.family and (p.n_x, p.n_y) == (m.n_x, m.n_y)
        if p.bw_valid:
            assert p.bw_3db_hz == pytest.approx(b, rel=0.08)
            assert p.q_factor == pytest.approx(p.f_mode / p.bw_3db_hz, rel=1e-6)
        else:
            assert np.isnan(p.bw_3db_hz)


def test_ism_ray_and_kuttruff_disagree_on_selectivity():
    """The two damping laws the gate discriminates (D48)."""
    L, W = 4.5, 4.0
    base = (0.15,) * 4
    west_m3 = (0.70, 0.15, 0.15, 0.15)
    dx_ray = (modal_damping_2d(L, W, west_m3, 1, 0, model="ism_ray")
              - modal_damping_2d(L, W, base, 1, 0, model="ism_ray"))
    dy_ray = (modal_damping_2d(L, W, west_m3, 0, 1, model="ism_ray")
              - modal_damping_2d(L, W, base, 0, 1, model="ism_ray"))
    assert dx_ray > 0 and dy_ray == pytest.approx(0.0, abs=1e-12)  # no grazing absorption

    dx_k = (modal_damping_2d(L, W, west_m3, 1, 0, model="kuttruff")
            - modal_damping_2d(L, W, base, 1, 0, model="kuttruff"))
    dy_k = (modal_damping_2d(L, W, west_m3, 0, 1, model="kuttruff")
            - modal_damping_2d(L, W, base, 0, 1, model="kuttruff"))
    assert dx_k / dy_k == pytest.approx(2.0, rel=1e-6)             # exactly 2:1


def test_kuttruff_reduces_to_sabine_and_alpha_zero_is_allowed():
    L, W, a = 4.5, 4.0, 0.15
    assert modal_damping_2d(L, W, (a,) * 4, 1, 1, model="kuttruff") == pytest.approx(
        sabine_damping_2d(L, W, a))
    assert modal_damping_2d(L, W, (0.0,) * 4, 1, 0, model="ism_ray") == 0.0
    with pytest.raises(ValueError):
        modal_damping_2d(L, W, (a,) * 4, 0, 0, model="ism_ray")     # DC has no rate
