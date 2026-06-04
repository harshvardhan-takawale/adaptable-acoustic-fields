"""Tests for ``aaf.data.sample_rooms_3d.select_diag_subset_maximin`` (P2-2.5)."""
import numpy as np
import pytest

from aaf.data.sample_rooms_3d import (
    DEFAULT_RANGES,
    Room3D,
    sample_train_rooms_lhs,
    select_diag_subset_maximin,
)


def test_select_diag_subset_count_and_subset_relation():
    rooms = sample_train_rooms_lhs(n=45, seed=42)
    subset = select_diag_subset_maximin(rooms, n=10)
    assert len(subset) == 10
    # All subset rooms must be present in the input.
    input_set = {r.as_tuple() for r in rooms}
    for r in subset:
        assert r.as_tuple() in input_set


def test_select_diag_subset_no_duplicates():
    rooms = sample_train_rooms_lhs(n=45, seed=42)
    subset = select_diag_subset_maximin(rooms, n=10)
    coords = {r.as_tuple() for r in subset}
    assert len(coords) == 10


def test_select_diag_subset_deterministic():
    rooms = sample_train_rooms_lhs(n=45, seed=42)
    a = [r.as_tuple() for r in select_diag_subset_maximin(rooms, n=10)]
    b = [r.as_tuple() for r in select_diag_subset_maximin(rooms, n=10)]
    assert a == b


def test_select_diag_subset_preserves_per_axis_variation():
    """The 10 maximin rooms should have at least 70% of the full 45-room
    per-axis stddev — the subset is meant to *span* the box, not cluster."""
    rooms = sample_train_rooms_lhs(n=45, seed=42)
    subset = select_diag_subset_maximin(rooms, n=10)

    full = np.array([r.as_tuple() for r in rooms])
    sub = np.array([r.as_tuple() for r in subset])
    for k, name in enumerate(("L", "W", "H")):
        full_std = full[:, k].std()
        sub_std = sub[:, k].std()
        # Maximin selection typically *increases* per-axis spread by picking
        # corners; require ≥ 70% just in case.
        assert sub_std >= 0.7 * full_std, (
            f"{name}: subset std {sub_std:.3f} < 70% of full std {full_std:.3f}"
        )


def test_select_diag_subset_bad_n():
    rooms = sample_train_rooms_lhs(n=45, seed=42)
    with pytest.raises(ValueError):
        select_diag_subset_maximin(rooms, n=0)
    with pytest.raises(ValueError):
        select_diag_subset_maximin(rooms, n=46)


def test_select_diag_subset_seed_at_box_center():
    """First pick should be the room nearest the box center."""
    rooms = sample_train_rooms_lhs(n=45, seed=42)
    subset = select_diag_subset_maximin(rooms, n=1)
    assert len(subset) == 1
    # Manually find the room nearest the box center in normalized [0,1]^3.
    lows = np.array([r[0] for r in DEFAULT_RANGES])
    highs = np.array([r[1] for r in DEFAULT_RANGES])
    normed = np.array([
        (np.array(r.as_tuple()) - lows) / (highs - lows) for r in rooms
    ])
    dists = np.sum((normed - 0.5) ** 2, axis=1)
    expected = rooms[int(np.argmin(dists))]
    assert subset[0].as_tuple() == expected.as_tuple()
