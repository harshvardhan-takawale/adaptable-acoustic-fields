"""Hand-computed first 5 unique modes for L=W=4 m, c=343 m/s, vs eigenfrequencies_2d.

Post Q9: eigenfrequencies_2d returns deduplicated EigenFreq entries with
multiplicity and pairs. Tests check the dedup'd list directly.
"""
import math

import pytest

from aaf.sim.analytical_modal_2d import eigenfrequencies_2d


def _hand_first_distinct_freqs_LW4_c343():
    """Compute the first several DISTINCT mode frequencies by hand for L=W=4, c=343.

    f_{n_x, n_y} = (c/2) * sqrt((n_x/L)^2 + (n_y/W)^2). At L=W, (n_x, n_y) and
    (n_y, n_x) coincide. Returns a sorted list of (f, expected_pairs_set) tuples.
    """
    L = W = 4.0
    c = 343.0
    by_freq: dict[float, set[tuple[int, int]]] = {}
    for nx in range(4):
        for ny in range(4):
            if nx == 0 and ny == 0:
                continue
            f = (c / 2) * math.sqrt((nx / L) ** 2 + (ny / W) ** 2)
            f = round(f, 6)
            by_freq.setdefault(f, set()).add((nx, ny))
    return sorted(by_freq.items())


def test_first_five_distinct_freqs_LW4():
    expected = _hand_first_distinct_freqs_LW4_c343()[:5]

    got = eigenfrequencies_2d(L=4.0, W=4.0, c=343.0, f_max=200.0)
    nonzero = [m for m in got if m.f > 0][:5]

    assert len(nonzero) == 5, f"expected 5 distinct freqs ≤ 200 Hz, got {len(nonzero)}"
    for (f_e, pairs_e), entry in zip(expected, nonzero):
        assert math.isclose(entry.f, f_e, rel_tol=1e-6, abs_tol=1e-6), (
            f"freq mismatch: expected {f_e}, got {entry.f}"
        )
        assert set(entry.pairs) == pairs_e, (
            f"at f={f_e}: expected pairs {pairs_e}, got {set(entry.pairs)}"
        )
        assert entry.multiplicity == len(pairs_e)


def test_degeneracy_at_LW4():
    """At L=W, modes (1,0) and (0,1) collapse into one entry with multiplicity=2."""
    modes = eigenfrequencies_2d(L=4.0, W=4.0, c=343.0, f_max=200.0)
    nonzero = [m for m in modes if m.f > 0]
    first = nonzero[0]
    assert math.isclose(first.f, 343.0 / 8.0, rel_tol=1e-9)
    assert first.multiplicity == 2
    assert set(first.pairs) == {(1, 0), (0, 1)}


def test_low_freq_no_degeneracy_at_L3_W4():
    """At L=3, W=4 the lowest few modes (≤ 160 Hz) are all non-degenerate.

    Higher modes do collide (e.g., (3, 0) and (0, 4) at 171.5 Hz) — that's
    expected for any rational L/W ratio with small integer parts.
    """
    modes = eigenfrequencies_2d(L=3.0, W=4.0, c=343.0, f_max=160.0)
    nonzero = [m for m in modes if m.f > 0]
    assert all(m.multiplicity == 1 for m in nonzero), (
        f"low-freq L=3, W=4 should be non-degenerate: "
        f"{[(m.f, m.pairs) for m in nonzero if m.multiplicity > 1]}"
    )
    by_pair = {tuple(m.pairs[0]): m.f for m in nonzero}
    assert (1, 0) in by_pair and (0, 1) in by_pair
    assert by_pair[(1, 0)] != by_pair[(0, 1)]
    # (0, 1) at f = c/(2W) = 343/8 = 42.875 Hz
    # (1, 0) at f = c/(2L) = 343/6 ≈ 57.17 Hz
    assert math.isclose(by_pair[(0, 1)], 343.0 / 8.0, rel_tol=1e-9)
    assert math.isclose(by_pair[(1, 0)], 343.0 / 6.0, rel_tol=1e-9)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        eigenfrequencies_2d(L=0.0, W=4.0, c=343.0, f_max=2000)
    with pytest.raises(ValueError):
        eigenfrequencies_2d(L=4.0, W=4.0, c=-1.0, f_max=2000)
