"""Q9 dedup checks beyond what test_eigenfrequencies.py covers."""
import math

import numpy as np
import pytest

from aaf.sim.analytical_modal_2d import (
    EigenFreq,
    eigenfrequencies_2d,
    modal_rir_2d,
)


def test_LW4_first_degenerate_pair():
    """Sanity-restate: L=W=4 first non-zero entry is multiplicity-2 (1,0)/(0,1)."""
    modes = eigenfrequencies_2d(L=4.0, W=4.0, c=343.0, f_max=200.0)
    nonzero = [m for m in modes if m.f > 0]
    first = nonzero[0]
    assert math.isclose(first.f, 343.0 / 8.0, rel_tol=1e-9)
    assert first.multiplicity == 2
    assert set(first.pairs) == {(1, 0), (0, 1)}


def test_LW4_second_distinct_freq_is_11():
    """(1, 1) at f = c·sqrt(2)/8 ≈ 60.63 Hz is non-degenerate."""
    modes = eigenfrequencies_2d(L=4.0, W=4.0, c=343.0, f_max=200.0)
    nonzero = [m for m in modes if m.f > 0]
    second = nonzero[1]
    expected_f = 343.0 * math.sqrt(2.0) / 8.0
    assert math.isclose(second.f, expected_f, rel_tol=1e-6)
    assert second.multiplicity == 1
    assert second.pairs == [(1, 1)]


def test_L3_W4_no_degeneracy_below_160hz():
    """L=3, W=4 has no modal collisions below 160 Hz. (Above that, (3,0) and
    (0,4) collide at 171.5 Hz — see test_L3_W4_collision_at_higher_freq below.)
    """
    modes = eigenfrequencies_2d(L=3.0, W=4.0, c=343.0, f_max=160.0)
    nonzero = [m for m in modes if m.f > 0]
    assert all(m.multiplicity == 1 for m in nonzero), (
        f"L=3, W=4 below 160 Hz should be non-degenerate: "
        f"{[(m.f, m.pairs) for m in nonzero if m.multiplicity > 1]}"
    )


def test_L3_W4_collision_at_higher_freq():
    """L=3, W=4: (3, 0) at 171.5 Hz and (0, 4) at 171.5 Hz — degenerate per
    16·n_x² ≡ 9·m_y² (mod gcd(L², W²)). The dedup machinery must catch this.
    """
    modes = eigenfrequencies_2d(L=3.0, W=4.0, c=343.0, f_max=200.0)
    matching = [m for m in modes if (3, 0) in m.pairs]
    assert len(matching) == 1
    entry = matching[0]
    assert (0, 4) in entry.pairs and (3, 0) in entry.pairs
    assert entry.multiplicity == 2
    assert math.isclose(entry.f, 343.0 / 2.0, rel_tol=1e-9)


def test_L4_W2_has_expected_degeneracies():
    """L = 2W creates degeneracies: (n_x, 0) coincides with (0, n_x/2) for even n_x.

    Validates the dedup machinery on a known-degenerate non-square geometry.
    """
    modes = eigenfrequencies_2d(L=4.0, W=2.0, c=343.0, f_max=200.0)
    nonzero = [m for m in modes if m.f > 0]
    degenerate = [m for m in nonzero if m.multiplicity > 1]
    assert len(degenerate) >= 1
    # First degeneracy is (2, 0) and (0, 1) at f = c/(2W) = 85.75 Hz.
    first = degenerate[0]
    assert math.isclose(first.f, 85.75, abs_tol=0.01)
    assert (2, 0) in first.pairs and (0, 1) in first.pairs


def test_dedup_tolerance_groups_close_freqs():
    """Two modes with f < dedup_tol_hz apart must collapse."""
    # At L=8.0, W=4.0: (2, 0) at f=42.875 and (0, 1) at f=42.875 — same.
    modes = eigenfrequencies_2d(L=8.0, W=4.0, c=343.0, f_max=100.0)
    by_pairs = {tuple(sorted(m.pairs)): m for m in modes}
    # (2, 0) and (0, 1) should share an entry
    matching = [m for m in modes if (2, 0) in m.pairs]
    assert len(matching) == 1
    entry = matching[0]
    assert (0, 1) in entry.pairs and (2, 0) in entry.pairs
    assert entry.multiplicity == 2


def test_modal_rir_unchanged_at_LW4():
    """modal_rir_2d should produce a non-trivial H regardless of dedup —
    physics iterates individual (n_x, n_y) terms internally.
    """
    cfg = {
        "L": 4.0,
        "W": 4.0,
        "source_pos": (0.5, 0.5),
        "receiver_pos": np.array([[2.0, 2.0]]),
        "alpha": 0.15,
        "fs": 4096,
        "n_time_samples": 1024,
    }
    out = modal_rir_2d(cfg)
    assert out["H_complex"].shape == (1, 513)
    assert np.any(out["H_complex"] != 0)
    # Meta records both counts.
    assert out["meta"]["n_modes"] > out["meta"]["n_distinct_freqs"], (
        "L=W=4 must have more (n_x,n_y) modes than distinct freqs"
    )


def test_eigenfreq_dataclass_shape():
    e = EigenFreq(f=42.875, multiplicity=2, pairs=[(1, 0), (0, 1)])
    assert e.f == 42.875
    assert e.multiplicity == 2
    assert e.pairs == [(1, 0), (0, 1)]
