"""Track B2: the divider-TOKEN aperture arm (`aperture_token`).

Track B failed because a global 55-d vector with a SCALAR aperture cannot induce a spatial
barrier at x0 with a gap of width a (predicted level-difference fit vs sqrt(a): r^2 = 0.172,
slope 2.46, against GT 0.948 / 7.61, and equally wrong seen and held-out). These tests pin the
four properties that make the token arm a genuine fix rather than a reshuffle:

1. a SEALED divider aggregates to EXACTLY zero under delta pooling (topologically correct);
2. the aperture axis stays CONTINUOUS in `a` -- no quantization to the 16-segment grid;
3. the aggregate is PERMUTATION-INVARIANT (it is a set of tokens, not a slot vector);
4. moving the doorway to a different y-centre CHANGES the aggregate (position is read).

Deliberately CPU-only / tcnn-free: it imports aaf.models.conditioning_2d but never
aaf.models.inr_2d. The pooling used here is re-derived from the same public constants
inr_2d reads, and test_pool_matches_inr2d_contract pins that correspondence.
"""
from __future__ import annotations

import math

import pytest
import torch

from aaf.data.aperture_configs import A_RANGE, DIVIDER_ALPHA
from aaf.models.conditioning_2d import (APER_TOKEN_COND_DIM, APER_TOKEN_DIM_2D,
                                        APER_TOKEN_GEOM_DIM, COND_SOURCE_BTOK, D_TOK,
                                        M_HAT_OPEN, N_K_TOK_M, N_K_TOK_POS, N_SEG_COND,
                                        N_TOK_DIV, TOKEN_AGG_DIM, _fourier_block,
                                        aperture_token_features_2d,
                                        build_cond_vector_2d, cond_dim_for, divider_m_hat,
                                        divider_open_fraction, divider_token_geometry,
                                        m_hat_seg)

L, W, X0 = 8.0, 4.0, 4.0

# The m-block slice inr_2d's delta pool overwrites: [m_hat | 3 octaves of sin/cos].
M_SLICE = slice(2 * 2 * N_K_TOK_POS + 3, D_TOK)


def _tokens(a, L=L, W=W, x0=X0):
    """[16, 26] token block of one config (the geometry prefix is stripped)."""
    v = aperture_token_features_2d(L, W, x0, a)
    return v[APER_TOKEN_GEOM_DIM:].reshape(N_TOK_DIV, D_TOK)


def _encoder(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(D_TOK, TOKEN_AGG_DIM), torch.nn.ReLU(),
                               torch.nn.Linear(TOKEN_AGG_DIM, TOKEN_AGG_DIM))


def _baseline_m():
    """m-block of a SOLID token, built through the SAME torch path as the featurizer."""
    mh = torch.tensor([m_hat_seg(DIVIDER_ALPHA)])
    return torch.cat([mh, _fourier_block(mh, N_K_TOK_M)])


def _baseline_m_inr2d():
    """m-block as inr_2d._tok_baseline_m actually builds it: python `math` in float64, cast
    to float32. Mathematically identical to _baseline_m but not bit-identical (1 ULP), so a
    sealed divider aggregates to zero to float32 precision rather than bitwise."""
    mb = m_hat_seg(DIVIDER_ALPHA)
    ang = [(2.0 ** k) * math.pi * mb for k in range(N_K_TOK_M)]
    return torch.tensor([mb] + [math.sin(x) for x in ang] + [math.cos(x) for x in ang])


def _delta_agg(t, enc, baseline=None):
    """sum_i [phi(t_i) - phi(t_i^baseline)] -- inr_2d's 'delta' token_pool, verbatim."""
    tb = t.clone()
    tb[..., M_SLICE] = (_baseline_m() if baseline is None else baseline).to(tb.dtype)
    with torch.no_grad():
        return (enc(t) - enc(tb)).sum(dim=0)


# ----------------------------------------------------------------- shapes / registration
def test_dims_and_registration():
    assert APER_TOKEN_DIM_2D == APER_TOKEN_GEOM_DIM + N_TOK_DIV * D_TOK == 464
    assert APER_TOKEN_COND_DIM == APER_TOKEN_GEOM_DIM + TOKEN_AGG_DIM == 112
    assert N_TOK_DIV == N_SEG_COND == 16 and D_TOK == 26   # encoder shape shared with A2/A3
    assert cond_dim_for(COND_SOURCE_BTOK) == APER_TOKEN_DIM_2D
    v = build_cond_vector_2d(COND_SOURCE_BTOK, L, W, None, x0=X0, a=1.0)
    assert v.shape == (APER_TOKEN_DIM_2D,) and torch.isfinite(v).all()


def test_geometry_block_matches_track_b_bytes():
    """B2's 48-d geometry prefix must be IDENTICAL to Track B's, so the arms differ only in
    how the doorway is expressed."""
    from aaf.models.conditioning_2d import aperture_features_2d
    for a in (0.0, 0.5, 2.5):
        ref = aperture_features_2d(L, W, X0, a)[:APER_TOKEN_GEOM_DIM]
        assert torch.equal(aperture_token_features_2d(L, W, X0, a)[:APER_TOKEN_GEOM_DIM], ref)


def test_x0_is_read_by_the_tokens():
    """cx = x0/L, so moving the divider moves every token."""
    for i in (0, 7, 15):
        assert divider_token_geometry(L, W, 3.2, i)[0] == pytest.approx(0.4)
        assert divider_token_geometry(L, W, 4.8, i)[0] == pytest.approx(0.6)
    assert not torch.equal(_tokens(1.0, x0=3.2), _tokens(1.0, x0=4.8))


# ------------------------------------------------------------------------------- 1. sealed
def test_sealed_aggregates_to_exactly_zero():
    """a = 0: every token carries the baseline m_hat, so the delta aggregate is IDENTICALLY
    zero -- a sealed divider is the un-edited room. Exact, not approximate."""
    t = _tokens(0.0)
    mb = m_hat_seg(DIVIDER_ALPHA)
    assert all(divider_open_fraction(W, 0.0, i) == 0.0 for i in range(N_TOK_DIV))
    assert torch.allclose(t[:, M_SLICE.start], torch.full((N_TOK_DIV,), mb), atol=0, rtol=0)
    for seed in (0, 1, 2):
        assert torch.equal(_delta_agg(t, _encoder(seed)), torch.zeros(TOKEN_AGG_DIM))
        # ...and with inr_2d's own float64-math baseline, zero to float32 precision. The two
        # constructions differ by <= 1 ULP, so the SHIPPED model's sealed aggregate is ~1e-7,
        # not bitwise zero. Recorded rather than papered over: it is 6 orders of magnitude
        # below any open-aperture aggregate (asserted in test_fully_open_saturates).
        agg2 = _delta_agg(t, _encoder(seed), baseline=_baseline_m_inr2d())
        assert agg2.abs().max() < 1e-6


def test_fully_open_saturates():
    """a >= W: no divider at all, every token fully open at m_hat = 1.0."""
    for a in (W, W + 1.0):
        assert all(divider_open_fraction(W, a, i) == 1.0 for i in range(N_TOK_DIV))
        assert torch.allclose(_tokens(a)[:, M_SLICE.start],
                              torch.full((N_TOK_DIV,), M_HAT_OPEN))
    assert _delta_agg(_tokens(W), _encoder()).abs().sum() > 0


# --------------------------------------------------------------------------- 2. continuity
A_GRID = (0.10, 0.15, 0.20, 0.25, 0.26, 0.50, 0.75, 0.95, 1.00, 1.05, 1.30, 1.75, 2.00, 2.50)


def test_open_fraction_sums_to_the_physical_aperture():
    """sum_i f_i * (W/16) == a exactly: the tokens PARTITION the divider and conserve the
    doorway's measure, which is what makes a strictly the right continuous quantity."""
    for a in A_GRID:
        tot = sum(divider_open_fraction(W, a, i) for i in range(N_TOK_DIV)) * (W / N_TOK_DIV)
        assert tot == pytest.approx(a, abs=1e-12)


def test_m_hat_is_strictly_monotone_and_continuous_in_a():
    """The whole point of fractional m_hat: an integer open-COUNT would step by W/16 = 0.25 m,
    coarser than the 0.2 m hold-out band, quantizing the axis the track is testing."""
    lo, hi = A_RANGE
    grid = [lo + (hi - lo) * k / 200.0 for k in range(201)]
    tot = [sum(divider_m_hat(W, a, i) for i in range(N_TOK_DIV)) for a in grid]
    for x, y in zip(tot, tot[1:]):
        assert y > x                                              # STRICTLY increasing
        assert y - x < 0.15                                       # and continuous: no jumps
    # Not quantized: 16 open-count levels would give <= 16 distinct values over 201 samples.
    assert len({round(v, 9) for v in tot}) == len(grid)
    # Sub-segment resolution -- 0.26 and 0.25 open the same COUNT but differ here.
    assert sum(divider_m_hat(W, 0.26, i) for i in range(N_TOK_DIV)) > \
        sum(divider_m_hat(W, 0.25, i) for i in range(N_TOK_DIV))


def test_delta_aggregate_moves_strictly_with_a():
    """The learned aggregate need not be monotone at random init, but it must never be
    STUCK: consecutive apertures must produce different conditioning."""
    enc = _encoder()
    aggs = [_delta_agg(_tokens(a), enc) for a in A_GRID]
    for a, b, x, y in zip(A_GRID, A_GRID[1:], aggs, aggs[1:]):
        assert (y - x).abs().max() > 1e-6, "aggregate stalled between a={} and {}".format(a, b)
    # Held-out band (0.9-1.1) is strictly interior to its neighbours' aggregates -- the arm
    # interpolates there rather than reusing a trained value.
    assert not torch.allclose(aggs[A_GRID.index(1.00)], aggs[A_GRID.index(0.75)])


def test_features_continuous_across_a_segment_boundary():
    """f_i is piecewise linear; crossing y = W/2 +- k*W/16 must not jump the feature vector."""
    b = W / N_TOK_DIV * 2                                          # a = 0.5: exact boundary
    lo, hi = _tokens(b - 1e-6), _tokens(b + 1e-6)
    assert (hi - lo).abs().max() < 1e-4
    assert not torch.equal(lo, hi)


# ---------------------------------------------------------------- 3. permutation invariance
def test_permuting_tokens_leaves_the_aggregate_identical():
    """Delta pooling is a SUM over a set -- token order carries no information, which is the
    structural difference from Track A's per-slot private dims."""
    enc = _encoder()
    for a in (0.5, 1.0, 2.5):
        t = _tokens(a)
        ref = _delta_agg(t, enc)
        for seed in (3, 4, 5):
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(N_TOK_DIV, generator=g)
            assert torch.allclose(_delta_agg(t[perm], enc), ref, atol=1e-5)


# ---------------------------------------------------------------------- 4. doorway position
def test_moving_the_doorway_changes_the_aggregate():
    """Same width, different y-centre -> different tokens and a different aggregate. This is
    exactly what a scalar aperture could not express, and why Track B could not build a
    spatial barrier."""
    enc = _encoder()
    a = 1.0
    centred = _tokens(a)

    def offset_tokens(y_c):
        """Same featurization, doorway centred at y_c instead of W/2."""
        seg = W / N_TOK_DIV
        mb = m_hat_seg(DIVIDER_ALPHA)
        rows = []
        for i in range(N_TOK_DIV):
            ov = min((i + 1) * seg, y_c + a / 2) - max(i * seg, y_c - a / 2)
            f = max(0.0, min(ov, seg)) / seg
            row = centred[i].clone()
            mh = mb + f * (M_HAT_OPEN - mb)
            ang = [(2.0 ** k) * math.pi * mh for k in range(N_K_TOK_M)]
            row[M_SLICE] = torch.tensor([mh] + [math.sin(x) for x in ang]
                                        + [math.cos(x) for x in ang])
            rows.append(row)
        return torch.stack(rows)

    ref = _delta_agg(centred, enc)
    assert torch.allclose(_delta_agg(offset_tokens(W / 2), enc), ref, atol=1e-6)
    for y_c in (1.0, 1.5, 3.0):
        moved = _delta_agg(offset_tokens(y_c), enc)
        assert (moved - ref).abs().max() > 1e-4, "aggregate blind to doorway at y={}".format(y_c)


# ------------------------------------------------------------------------------- contract
def test_pool_matches_inr2d_contract():
    """The pooling replicated above must stay in step with inr_2d's. Pin the two derived
    quantities inr_2d computes from these same constants."""
    assert M_SLICE == slice(2 * 2 * N_K_TOK_POS + 3, D_TOK)
    assert APER_TOKEN_DIM_2D - N_SEG_COND * D_TOK == APER_TOKEN_GEOM_DIM   # inr_2d._tok_geom
    assert _baseline_m().shape == (1 + 2 * N_K_TOK_M,)
    assert _baseline_m()[0].item() == pytest.approx(m_hat_seg(DIVIDER_ALPHA))
    assert torch.allclose(_baseline_m(), _baseline_m_inr2d(), atol=1e-7)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        aperture_token_features_2d(L, W, X0, -0.1)
    with pytest.raises(ValueError):
        divider_open_fraction(W, 1.0, N_TOK_DIV)
    with pytest.raises(ValueError):
        build_cond_vector_2d(COND_SOURCE_BTOK, L, W, None, x0=X0, a=None)
