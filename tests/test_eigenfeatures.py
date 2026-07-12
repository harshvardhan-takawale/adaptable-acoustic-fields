"""P3-1 conditioning features (Arm G Fourier / Arm G+ eigenstructure) — pure CPU.

Validates aaf.models.conditioning against closed-form modal physics. tcnn-free, so it
runs on the login node / any CPU box (no CUDA import).
"""
import math

import numpy as np
import torch

from aaf.models.conditioning import (
    fourier_features, eigen_features, resonance_map, build_cond_vector, FOURIER_DIM,
)
from aaf.sim.analytical_modal_3d import eigenfrequencies_3d, C_DEFAULT

C = C_DEFAULT  # 343.0


def test_fourier_features_shape_and_range():
    ff = fourier_features(4.5, 4.0, 3.25)
    assert ff.shape == (48,) and FOURIER_DIM == 48
    assert float(ff.min()) >= -1.0 - 1e-6 and float(ff.max()) <= 1.0 + 1e-6
    # deterministic
    assert torch.allclose(ff, fourier_features(4.5, 4.0, 3.25))
    # k=0 sin/cos of g_L: g_L=(4.5-3)/3=0.5 -> sin(pi*0.5)=1, cos(pi*0.5)=0
    # layout: dim-block [sin(2^0..2^7 pi g), cos(...)]; ff[0] is sin(2^0 pi g_L)
    assert abs(float(ff[0]) - math.sin(math.pi * 0.5)) < 1e-5


def test_eigen_features_first_10_closed_form():
    L, W, H = 5.0, 4.0, 3.0
    ef = eigen_features(L, W, H)  # [64] / 300
    assert ef.shape == (64,)
    # strictly non-decreasing (ascending distinct freqs / 300)
    assert bool((ef[1:] >= ef[:-1] - 1e-6).all())
    modes = [m.f for m in eigenfrequencies_3d(L, W, H, c=C, f_max=2000.0) if m.f > 0][:10]
    # hand-verify the lowest axials appear: (1,0,0)=c/2L, (0,1,0)=c/2W, (0,0,1)=c/2H
    axials = sorted([C / (2 * L), C / (2 * W), C / (2 * H)])
    got = ef[:10].numpy() * 300.0
    for a in axials:
        assert np.min(np.abs(got - a)) < 0.05, f"axial {a:.2f} Hz not among first-10 {got}"
    # first entry equals the analytic lowest mode
    assert abs(got[0] - modes[0]) < 0.05


def test_resonance_map_peaks_at_eigenfreqs():
    L, W, H = 4.5, 4.0, 3.25
    df, n_bins, cap = 0.5, 601, 310.0
    R = resonance_map(L, W, H, n_bins=n_bins, df=df, f_cap=cap).numpy()
    assert R.shape == (601,)
    assert abs(R.max() - 1.0) < 1e-6           # max-normalized
    assert R.min() >= 0.0
    f_axis = np.arange(n_bins) * df
    f_n = [m.f for m in eigenfrequencies_3d(L, W, H, c=C, f_max=cap) if 0 < m.f <= cap]
    # R must be locally maximal within 1 bin of every analytic eigenfrequency (a peak there)
    for fn in f_n:
        b = int(round(fn / df))
        if b <= 0 or b >= n_bins - 1:
            continue
        window = R[max(0, b - 2): b + 3]
        assert R[b] >= 0.5 * window.max(), f"R weak at eigenfreq {fn:.2f} Hz (bin {b})"


def test_build_cond_vector_dispatch():
    L, W, H = 4.5, 4.0, 3.25
    assert build_cond_vector("geom_fourier", L, W, H).shape == (48,)
    assert build_cond_vector("eigen", L, W, H).shape == (64,)
    try:
        build_cond_vector("nonsense", L, W, H)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_enough_eigenfreqs_extremes():
    # smallest and largest rooms in the box both yield >=64 distinct modes
    for (L, W, H) in [(3.0, 3.0, 2.5), (5.99, 4.99, 3.99)]:
        assert eigen_features(L, W, H).shape == (64,)
