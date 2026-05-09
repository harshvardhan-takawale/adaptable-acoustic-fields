"""Hand-computed first 5 unique modes for L=W=4 m, c=343 m/s, vs eigenfrequencies_2d."""
import math

import numpy as np
import pytest

from aaf.sim.analytical_modal_2d import eigenfrequencies_2d


def _hand_first_modes_LW4_c343():
    """Compute the first several modes by hand for L=W=4, c=343.

    f_{n_x, n_y} = (c/2) * sqrt((n_x/L)^2 + (n_y/W)^2)

    For L=W=4 there is a degeneracy: f_{n_x, n_y} = f_{n_y, n_x}.

    Returns sorted list of (n_x, n_y, f) tuples for n_x, n_y in [0, 3], excluding (0,0).
    """
    L = W = 4.0
    c = 343.0
    out = []
    for nx in range(4):
        for ny in range(4):
            if nx == 0 and ny == 0:
                continue
            f = (c / 2) * math.sqrt((nx / L) ** 2 + (ny / W) ** 2)
            out.append((nx, ny, f))
    out.sort(key=lambda t: (t[2], t[0], t[1]))
    return out


def test_first_five_modes_LW4_match_handcomputed():
    """The first five returned modes (excluding (0,0)) should match exact analytical values."""
    expected = _hand_first_modes_LW4_c343()[:5]

    got = eigenfrequencies_2d(L=4.0, W=4.0, c=343.0, f_max=200.0)
    got = [m for m in got if not (m.n_x == 0 and m.n_y == 0)][:5]

    assert len(got) == 5, f"expected 5 modes ≤ 200 Hz, got {len(got)}"
    for (nx_e, ny_e, f_e), m in zip(expected, got):
        assert m.n_x == nx_e and m.n_y == ny_e, (
            f"mode index mismatch: expected ({nx_e},{ny_e}), got ({m.n_x},{m.n_y})"
        )
        assert math.isclose(m.f, f_e, rel_tol=1e-6, abs_tol=1e-6), (
            f"mode ({nx_e},{ny_e}): expected f={f_e}, got {m.f}"
        )


def test_degeneracy_at_LW4():
    """At L=W, modes (1,0) and (0,1) should have identical frequency, as should (2,1)/(1,2)."""
    modes = eigenfrequencies_2d(L=4.0, W=4.0, c=343.0, f_max=200.0)
    by_idx = {(m.n_x, m.n_y): m.f for m in modes}
    assert (1, 0) in by_idx and (0, 1) in by_idx
    assert math.isclose(by_idx[(1, 0)], by_idx[(0, 1)])
    assert (2, 1) in by_idx and (1, 2) in by_idx
    assert math.isclose(by_idx[(2, 1)], by_idx[(1, 2)])


def test_no_degeneracy_at_L_neq_W():
    """At L=3, W=4 the (1,0) and (0,1) modes should have different frequencies."""
    modes = eigenfrequencies_2d(L=3.0, W=4.0, c=343.0, f_max=200.0)
    by_idx = {(m.n_x, m.n_y): m.f for m in modes}
    assert by_idx[(1, 0)] != by_idx[(0, 1)]
    # (1, 0): c/(2L) = 343/6 ≈ 57.17 Hz
    # (0, 1): c/(2W) = 343/8 = 42.875 Hz
    assert math.isclose(by_idx[(0, 1)], 343.0 / 8.0, rel_tol=1e-9)
    assert math.isclose(by_idx[(1, 0)], 343.0 / 6.0, rel_tol=1e-9)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        eigenfrequencies_2d(L=0.0, W=4.0, c=343.0, f_max=2000)
    with pytest.raises(ValueError):
        eigenfrequencies_2d(L=4.0, W=4.0, c=-1.0, f_max=2000)
