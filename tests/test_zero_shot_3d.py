"""Tests for `aaf.eval.zero_shot_3d` — pure-Python parts (no CUDA required)."""
import numpy as np
import pytest


def test_obs_indices_3d_is_eight_corners():
    from aaf.eval.zero_shot_3d import OBS_INDICES_3D, select_obs_indices_3d

    expected = np.array([0, 7, 56, 63, 448, 455, 504, 511], dtype=np.int64)
    assert np.array_equal(OBS_INDICES_3D, expected)


def test_obs_indices_3d_in_range_and_unique():
    from aaf.eval.zero_shot_3d import OBS_INDICES_3D

    assert OBS_INDICES_3D.size == 8
    assert OBS_INDICES_3D.min() >= 0
    assert OBS_INDICES_3D.max() < 512
    assert len(set(OBS_INDICES_3D.tolist())) == 8


def test_select_obs_indices_3d_default_returns_corners():
    from aaf.eval.zero_shot_3d import OBS_INDICES_3D, select_obs_indices_3d

    out = select_obs_indices_3d(n_obs=8, total=512)
    assert np.array_equal(out, OBS_INDICES_3D)


def test_select_obs_indices_3d_fallback_linspace():
    from aaf.eval.zero_shot_3d import select_obs_indices_3d

    out = select_obs_indices_3d(n_obs=16, total=512)
    assert out.size <= 16
    assert out.min() >= 0
    assert out.max() < 512
    # Should be increasing.
    assert np.all(np.diff(out) > 0)


def test_select_obs_indices_3d_bad_n():
    from aaf.eval.zero_shot_3d import select_obs_indices_3d

    with pytest.raises(ValueError):
        select_obs_indices_3d(n_obs=0, total=512)
    with pytest.raises(ValueError):
        select_obs_indices_3d(n_obs=513, total=512)


def test_obs_indices_3d_are_the_8_grid_corners():
    """Verify each index decodes to a (z, y, x) corner of an 8×8×8 grid."""
    from aaf.eval.zero_shot_3d import OBS_INDICES_3D

    corners = []
    for idx in OBS_INDICES_3D:
        iz = int(idx) // 64
        iy = (int(idx) % 64) // 8
        ix = int(idx) % 8
        corners.append((iz, iy, ix))
    expected = [
        (0, 0, 0), (0, 0, 7), (0, 7, 0), (0, 7, 7),
        (7, 0, 0), (7, 0, 7), (7, 7, 0), (7, 7, 7),
    ]
    assert set(corners) == set(expected)
