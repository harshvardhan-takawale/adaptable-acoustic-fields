"""P2-4 nested room sets + frozen interior test set (pure numpy/scipy; no CUDA)."""
import numpy as np
import pytest

from aaf.data.sample_rooms_3d import (
    DEFAULT_RANGES, sample_train_rooms_lhs, sample_nested_supersets,
    sample_interior_test_rooms, _normalize_room,
)


def _key(r, nd=4):
    return (round(r.L, nd), round(r.W, nd), round(r.H, nd))


@pytest.fixture(scope="module")
def sets():
    base = sample_train_rooms_lhs(n=45, seed=42)
    sup = sample_nested_supersets(base, targets=(90, 150, 250), seed=7)
    return {45: base, 90: sup[90], 150: sup[150], 250: sup[250]}


def test_exact_counts(sets):
    assert [len(sets[n]) for n in (45, 90, 150, 250)] == [45, 90, 150, 250]


def test_nesting_subset_chain(sets):
    for small, big in [(45, 90), (90, 150), (150, 250)]:
        assert {_key(r) for r in sets[small]} <= {_key(r) for r in sets[big]}


def test_base_preserved_in_all(sets):
    base = {_key(r) for r in sets[45]}
    for n in (90, 150, 250):
        assert base <= {_key(r) for r in sets[n]}


def test_supersets_deterministic():
    base = sample_train_rooms_lhs(n=45, seed=42)
    s1 = sample_nested_supersets(base, targets=(90,), seed=7)[90]
    s2 = sample_nested_supersets(base, targets=(90,), seed=7)[90]
    assert [_key(r) for r in s1] == [_key(r) for r in s2]


def test_rooms_in_box(sets):
    for n, rooms in sets.items():
        for r in rooms:
            assert 3.0 <= r.L <= 6.0 and 3.0 <= r.W <= 5.0 and 2.5 <= r.H <= 4.0
            assert abs(r.L - r.W) >= 0.05      # cubic rejection preserved


def test_interior_test_set(sets):
    from scipy.spatial import Delaunay
    test = sample_interior_test_rooms(hull_rooms=sets[45], exclude_rooms=sets[250],
                                      n=15, seed=2024, min_train_dist=0.04)
    assert len(test) == 15
    hull = Delaunay(np.stack([_normalize_room(r, DEFAULT_RANGES) for r in sets[45]]))
    tn = np.stack([_normalize_room(r, DEFAULT_RANGES) for r in test])
    assert (hull.find_simplex(tn) >= 0).all()                 # strictly interior to 45-hull
    assert not ({_key(r) for r in test} & {_key(r) for r in sets[250]})  # distinct from training


def test_test_nn_distance_decreases(sets):
    test = sample_interior_test_rooms(hull_rooms=sets[45], exclude_rooms=sets[250],
                                      n=15, seed=2024, min_train_dist=0.04)
    T = np.array([r.as_tuple() for r in test])
    means = []
    for n in (45, 90, 150, 250):
        X = np.array([r.as_tuple() for r in sets[n]])
        means.append(np.sqrt(((T[:, None] - X[None]) ** 2).sum(2)).min(1).mean())
    assert all(means[i] > means[i + 1] for i in range(3)), means
