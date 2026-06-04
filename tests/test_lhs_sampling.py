"""Tests for the 3D LHS sampler + de-risk/test room generators."""
import pytest

from aaf.data.sample_rooms_3d import (
    DEFAULT_LHS_SEED,
    DEFAULT_RANGES,
    DEFAULT_REJECT_CUBIC_TOL,
    Room3D,
    derisk_rooms,
    sample_test_rooms,
    sample_train_rooms_lhs,
)


def test_derisk_rooms_count_and_spec_match():
    rooms = derisk_rooms()
    assert len(rooms) == 5
    coords = {r.as_tuple() for r in rooms}
    assert (4.5, 4.0, 3.25) in coords     # box center
    assert (3.0, 3.0, 2.5) in coords      # extreme small
    assert (6.0, 5.0, 4.0) in coords      # extreme large


def test_lhs_train_sample_count_and_ranges():
    rooms = sample_train_rooms_lhs(n=45)
    assert len(rooms) == 45
    Ls = [r.L for r in rooms]
    Ws = [r.W for r in rooms]
    Hs = [r.H for r in rooms]
    Llo, Lhi = DEFAULT_RANGES[0]
    Wlo, Whi = DEFAULT_RANGES[1]
    Hlo, Hhi = DEFAULT_RANGES[2]
    assert min(Ls) >= Llo and max(Ls) <= Lhi
    assert min(Ws) >= Wlo and max(Ws) <= Whi
    assert min(Hs) >= Hlo and max(Hs) <= Hhi


def test_lhs_train_no_near_cubic_draws():
    rooms = sample_train_rooms_lhs(n=45)
    for r in rooms:
        assert abs(r.L - r.W) >= DEFAULT_REJECT_CUBIC_TOL, (
            f"{r} violates |L-W| >= {DEFAULT_REJECT_CUBIC_TOL}"
        )


def test_lhs_train_no_duplicates():
    rooms = sample_train_rooms_lhs(n=45)
    coords = {r.as_tuple() for r in rooms}
    assert len(coords) == len(rooms)


def test_lhs_train_reasonable_spread_per_axis():
    """LHS over a stratified grid should give roughly uniform coverage —
    each axis's stddev should be at least 1/√12 × range × 0.7 (allowing
    for the reject-near-cubic filter and finite-sample shrinkage).

    Theoretical uniform stddev on a span of length S is S/√12 ≈ 0.289·S.
    """
    import math
    rooms = sample_train_rooms_lhs(n=45)
    import numpy as np
    Ls = np.array([r.L for r in rooms])
    Ws = np.array([r.W for r in rooms])
    Hs = np.array([r.H for r in rooms])
    threshold_factor = 0.7 / math.sqrt(12)   # ≈ 0.202
    for arr, (lo, hi), name in zip(
        (Ls, Ws, Hs), DEFAULT_RANGES, ("L", "W", "H")
    ):
        std = arr.std()
        span = hi - lo
        assert std >= threshold_factor * span, (
            f"{name} std {std:.3f} < {threshold_factor*span:.3f} "
            f"({threshold_factor*100:.0f}% of range {span:.3f})"
        )


def test_lhs_train_is_reproducible():
    rooms1 = sample_train_rooms_lhs(n=45, seed=DEFAULT_LHS_SEED)
    rooms2 = sample_train_rooms_lhs(n=45, seed=DEFAULT_LHS_SEED)
    for a, b in zip(rooms1, rooms2):
        assert a.as_tuple() == b.as_tuple()


def test_test_rooms_count_and_centered_first():
    lhs = sample_train_rooms_lhs(n=45)
    test = sample_test_rooms(n=8, lhs_rooms=lhs)
    assert len(test) == 8
    # First room is the box center.
    Llo, Lhi = DEFAULT_RANGES[0]
    Wlo, Whi = DEFAULT_RANGES[1]
    Hlo, Hhi = DEFAULT_RANGES[2]
    assert abs(test[0].L - 0.5 * (Llo + Lhi)) < 1e-6
    assert abs(test[0].W - 0.5 * (Wlo + Whi)) < 1e-6
    assert abs(test[0].H - 0.5 * (Hlo + Hhi)) < 1e-6


def test_test_rooms_inside_ranges():
    lhs = sample_train_rooms_lhs(n=45)
    test = sample_test_rooms(n=8, lhs_rooms=lhs)
    for r in test:
        for axis, (lo, hi) in zip(("L", "W", "H"), DEFAULT_RANGES):
            v = getattr(r, axis)
            assert lo <= v <= hi


def test_test_rooms_distinct_from_lhs():
    """No test room should exactly coincide with any LHS draw (within 1 mm)."""
    lhs = sample_train_rooms_lhs(n=45)
    test = sample_test_rooms(n=8, lhs_rooms=lhs)
    lhs_tuples = [r.as_tuple() for r in lhs]
    for r in test:
        for tL, tW, tH in lhs_tuples:
            d = max(abs(r.L - tL), abs(r.W - tW), abs(r.H - tH))
            assert d > 1e-3, f"{r} too close to LHS draw ({tL}, {tW}, {tH})"
