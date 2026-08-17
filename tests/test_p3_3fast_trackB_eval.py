"""Unit tests for the P3-3-FAST Track 2b eval harness that need neither a GPU nor a model.

The four things that would silently corrupt every headline number if they were wrong:
the held-out/seen split, the room-A/room-B receiver assignment, the third-octave banding,
and the sealed-config guards against log(0).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aaf.data.aperture_configs import (
    A_HOLDOUT,
    A_HOLDOUT_TEST_VALUES,
    ApertureConfig,
    in_holdout,
)
from scripts.p3_3fast_ftb import BAND_HI, BAND_LO, THIRD_OCTAVE_HZ
from scripts.p3_3fast_trackB_eval import (
    DF_HZ,
    GROUP_HELDOUT,
    GROUP_SEEN,
    N_BINS_BAND,
    N_FREQ_FULL,
    N_TIME,
    continuity_fit,
    dynamic_range_db,
    group_of,
    level_difference,
    pad_to_full,
    room_masks,
    subroom_frame,
    topological_reference,
)


def _cfg(a, L=8.0, W=4.0, x0=4.0, kind="t_aperture", gid=0):
    return ApertureConfig(L=L, W=W, x0=x0, a=a, kind=kind, split="test", geom_id=gid)


# ------------------------------------------------------------------------- the split logic
@pytest.mark.parametrize("a", A_HOLDOUT_TEST_VALUES)
def test_holdout_values_are_in_the_holdout_group(a):
    assert in_holdout(a)
    assert group_of(_cfg(a)) == GROUP_HELDOUT


@pytest.mark.parametrize("a", (0.15, 0.30, 0.50, 0.70, 1.50, 2.00, 2.50, 4.0))
def test_test_apertures_outside_the_band_are_seen(a):
    assert not in_holdout(a)
    assert group_of(_cfg(a)) == GROUP_SEEN


def test_holdout_band_is_closed_on_both_ends():
    """The band is [0.9, 1.1] CLOSED; the nearest training draws are 0.8969 and 1.1044."""
    assert in_holdout(A_HOLDOUT[0]) and in_holdout(A_HOLDOUT[1])
    assert not in_holdout(0.8969) and not in_holdout(1.1044)


# ----------------------------------------------------- room-A / room-B receiver assignment
def test_room_masks_partition_by_divider_and_report_on_divider_receivers():
    rx = np.array([[0.5, 1.0], [3.9, 2.0], [4.0, 1.0], [4.1, 2.0], [7.5, 3.0]])
    a, b, n_on = room_masks(rx, 4.0)
    assert a.tolist() == [True, True, False, False, False]
    assert b.tolist() == [False, False, False, True, True]
    assert n_on == 1                                   # never folded silently into a room
    assert not np.any(a & b)


def test_subroom_frame_puts_room_b_in_its_own_zero_based_frame():
    rx = np.array([[1.0, 1.0], [3.0, 2.0], [5.0, 1.0], [7.0, 2.0]])
    loc_a, L_a, sel_a = subroom_frame(rx, 8.0, 4.0, "A")
    loc_b, L_b, sel_b = subroom_frame(rx, 8.0, 4.0, "B")
    assert L_a == pytest.approx(4.0) and L_b == pytest.approx(4.0)
    assert np.allclose(loc_a[:, 0], [1.0, 3.0])
    assert np.allclose(loc_b[:, 0], [1.0, 3.0])        # 5 - 4 and 7 - 4
    assert int(sel_a.sum()) == 2 and int(sel_b.sum()) == 2
    assert np.all(loc_b[:, 0] >= 0.0) and np.all(loc_b[:, 0] <= L_b)


# --------------------------------------------------------------------- third-octave banding
def test_third_octave_bands_lie_inside_the_analysis_band_and_do_not_overlap():
    edges = [(fc * 2 ** (-1 / 6), fc * 2 ** (1 / 6)) for fc in THIRD_OCTAVE_HZ]
    assert edges[0][0] >= BAND_LO and edges[-1][1] <= BAND_HI
    for (lo0, hi0), (lo1, _) in zip(edges, edges[1:]):
        assert lo0 < hi0
        assert hi0 <= lo1 * 2 ** (1 / 6)               # adjacent, ordered, no gaps
    assert len(THIRD_OCTAVE_HZ) == 11


def test_level_difference_reproduces_a_known_ratio_per_band_and_broadband():
    """Room B at exactly 1/10 of room A must read -20 dB in every band and pooled."""
    freqs = np.arange(N_BINS_BAND) * DF_HZ
    rx = np.array([[1.0, 1.0], [3.0, 2.0], [5.0, 1.0], [7.0, 2.0]])
    sel_a, sel_b, _ = room_masks(rx, 4.0)
    H = np.ones((4, N_BINS_BAND), dtype=np.complex64)
    H[sel_b] *= 0.1
    out = level_difference(H, freqs, sel_a, sel_b)
    # complex64 storage, so 1e-4 dB rather than machine epsilon.
    assert out["ld_broadband_db"] == pytest.approx(-20.0, abs=1e-4)
    assert len(out["bands"]) == len(THIRD_OCTAVE_HZ)
    for b in out["bands"]:
        assert b["ld_db"] == pytest.approx(-20.0, abs=1e-4)
        assert BAND_LO <= b["lo_hz"] < b["hi_hz"] <= BAND_HI


# --------------------------------------------------------------------- sealed-config guards
def test_sealed_room_b_gives_minus_inf_not_a_crash_or_a_floored_number():
    freqs = np.arange(N_BINS_BAND) * DF_HZ
    rx = np.array([[1.0, 1.0], [3.0, 2.0], [5.0, 1.0], [7.0, 2.0]])
    sel_a, sel_b, _ = room_masks(rx, 4.0)
    H = np.ones((4, N_BINS_BAND), dtype=np.complex64)
    H[sel_b] = 0.0                                     # room B disconnected, EXACTLY zero
    out = level_difference(H, freqs, sel_a, sel_b)
    assert out["ld_broadband_db"] == float("-inf")
    assert all(b["ld_db"] == float("-inf") for b in out["bands"])


def test_sealed_configs_are_excluded_from_the_continuous_fit():
    """The sqrt(a) fit must see only a > 0; a sealed row would be a -inf y-value."""
    rows = []
    for gid in range(2):
        rows.append({"a": 0.0, "sealed": True, "group": "sealed", "geom_id": gid,
                     "ld_gt_db": float("-inf"), "gt_roomB_all_zero": True})
        for a in (0.15, 0.5, 1.5, 2.5, 0.95, 1.0):
            rows.append({"a": a, "sealed": False, "geom_id": gid,
                         "group": group_of(_cfg(a)),
                         "ld_gt_db": 6.8 * math.sqrt(a) - 14.2,
                         "gt_roomB_all_zero": False})
    fit = continuity_fit(rows, with_pred=False)
    assert fit["pooled"]["gt"]["n"] == 12              # 2 x 6 non-sealed, 0 sealed
    assert fit["pooled"]["gt"]["slope"] == pytest.approx(6.8, abs=1e-6)
    assert fit["pooled"]["gt"]["r2"] == pytest.approx(1.0, abs=1e-9)
    assert fit["by_group"][GROUP_HELDOUT]["gt"]["n"] == 4
    assert fit["by_group"][GROUP_SEEN]["gt"]["n"] == 8
    # A synthetic perfect sqrt(a) law leaves the held-out points exactly on the seen line.
    assert fit["seen_line"]["gt"]["rms_residual_heldout_db"] == pytest.approx(0.0, abs=1e-8)

    topo = topological_reference(rows, with_pred=False)
    assert topo["n_configs"] == 2
    assert topo["gt_room_b_energy_is_zero"] is True
    assert all(v == float("-inf") for v in topo["gt_ld_db"])


def test_dynamic_range_ignores_the_exact_zeros_a_sealed_room_produces():
    H = np.zeros((4, N_BINS_BAND), dtype=np.complex64)
    H[:, 40:] = 1.0
    H[0, 40:] = 100.0
    dr = dynamic_range_db(H, 40, N_BINS_BAND)
    assert dr["max_minus_min_db"] == pytest.approx(40.0, abs=1e-4)
    assert dr["n_cells"] == 4 * (N_BINS_BAND - 40)      # the sub-20 Hz zeros never counted


# ------------------------------------------------------------------- rfft / irfft alignment
def test_pad_to_full_matches_the_renderer_grid_so_the_two_spectra_are_bin_aligned():
    """GT is 601 bins of a df = 0.5 Hz grid; the renderer emits 4097 of the SAME grid."""
    assert N_FREQ_FULL == N_TIME // 2 + 1
    assert DF_HZ == pytest.approx(4096.0 / N_TIME)
    assert N_BINS_BAND == int(round(BAND_HI / DF_HZ)) + 1
    H = np.ones((3, N_BINS_BAND), dtype=np.complex64)
    P = pad_to_full(H)
    assert P.shape == (3, N_FREQ_FULL)
    assert np.allclose(P[:, :N_BINS_BAND], H)
    assert np.allclose(P[:, N_BINS_BAND:], 0.0)
    rir = np.fft.irfft(P, n=N_TIME, axis=-1)
    assert rir.shape == (3, N_TIME)
    assert N_TIME / 4096.0 == pytest.approx(2.0)       # same 2.0 s record as the FDTD data
