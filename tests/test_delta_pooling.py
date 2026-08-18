"""A3 delta-pooling: recover edit magnitude WITHOUT losing A2's transfer property.

A2 replaced Track A's private per-segment dims with a shared token encoder and mean-pooling.
That fixed transfer decisively -- held-out window recovery went from -0.069 to +0.942 -- but it
diluted magnitude: 15 of 16 tokens sit at baseline, so a single edit moves the mean by ~1/16 and
the measured discrimination ratio fell to 0.222 against a GT spread of 1.199 dB (Track A, which
could not transfer at all, scored 1.543).

Delta-pooling aggregates ``sum_i [phi(t_i) - phi(t_i^baseline)]`` instead:

  * an UNEDITED config contributes exactly zero -- the aggregate encodes the EDIT, not the room;
  * one edited segment contributes its full phi response, with no 1/16 dilution;
  * phi remains shared across segments, so the transfer property holds BY CONSTRUCTION rather
    than by re-argument.

The last two tests are the ones that matter. It would be easy to "recover magnitude" by
reintroducing per-segment structure -- which would undo A2 and reproduce Track A's failure with
extra steps. Order-invariance plus position-sensitivity together pin down that the aggregator is
still permutation-shared AND still reads position.

These run on CPU against a standalone replica of the encoder arithmetic; the full model needs a
GPU (tinycudann).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from aaf.data.seg_configs import m_of_alpha, segment_index
from aaf.models.conditioning_2d import (
    D_TOK,
    M_NORM_SEG_COND,
    N_K_TOK_M,
    N_K_TOK_POS,
    N_SEG_COND,
    TOKEN_AGG_DIM,
    TOKEN_DIM_2D,
    segment_token_features_2d,
)
from aaf.walls import ALPHA_BASELINE

L, W = 4.5, 4.0
M_SLICE = slice(2 * 2 * N_K_TOK_POS + 3, D_TOK)


def _encoder(seed: int = 0) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(D_TOK, TOKEN_AGG_DIM), nn.ReLU(),
                         nn.Linear(TOKEN_AGG_DIM, TOKEN_AGG_DIM))


def _baseline_m_block() -> torch.Tensor:
    """The m-block of a token whose absorption is the baseline. Must match inr_2d exactly."""
    mb = m_of_alpha(ALPHA_BASELINE) / M_NORM_SEG_COND
    ang = [(2.0 ** k) * math.pi * mb for k in range(N_K_TOK_M)]
    return torch.tensor([mb] + [math.sin(x) for x in ang] + [math.cos(x) for x in ang],
                        dtype=torch.float32)


def _tokens(alphas) -> torch.Tensor:
    v = segment_token_features_2d(L, W, alphas)
    n_geom = TOKEN_DIM_2D - N_SEG_COND * D_TOK
    return v[n_geom:].reshape(N_SEG_COND, D_TOK)


def _delta_pool(toks: torch.Tensor, enc: nn.Sequential) -> torch.Tensor:
    tb = toks.clone()
    tb[..., M_SLICE] = _baseline_m_block()
    with torch.no_grad():
        return (enc(toks) - enc(tb)).sum(dim=0)


def _mean_pool(toks: torch.Tensor, enc: nn.Sequential) -> torch.Tensor:
    with torch.no_grad():
        return enc(toks).mean(dim=0)


def test_baseline_config_aggregates_to_exactly_zero():
    """The defining property: no edit, no signal. Mean-pooling cannot do this -- it returns the
    room's own baseline encoding, which the FiLM generator must then learn to subtract."""
    enc = _encoder()
    agg = _delta_pool(_tokens([ALPHA_BASELINE] * N_SEG_COND), enc)
    assert torch.allclose(agg, torch.zeros_like(agg), atol=1e-6)
    # and mean-pooling emphatically does NOT
    assert _mean_pool(_tokens([ALPHA_BASELINE] * N_SEG_COND), enc).abs().max() > 1e-3


def test_the_baseline_m_block_matches_the_featurizer():
    """If this drifts from what segment_token_features_2d emits, the 'zero' above is a lie."""
    toks = _tokens([ALPHA_BASELINE] * N_SEG_COND)
    for i in range(N_SEG_COND):
        assert torch.allclose(toks[i, M_SLICE], _baseline_m_block(), atol=1e-6)


def test_single_edit_contributes_its_own_delta_alone():
    """One edited segment must produce exactly that segment's phi-delta -- no dilution, and no
    contamination from the 15 untouched tokens."""
    enc = _encoder()
    i = segment_index("west", 2)
    a = [ALPHA_BASELINE] * N_SEG_COND
    a[i] = 0.70
    toks = _tokens(a)
    agg = _delta_pool(toks, enc)
    tb = toks[i:i + 1].clone()
    tb[..., M_SLICE] = _baseline_m_block()
    with torch.no_grad():
        alone = (enc(toks[i:i + 1]) - enc(tb)).sum(dim=0)
    assert torch.allclose(agg, alone, atol=1e-6)


def test_delta_pooling_beats_mean_pooling_on_edit_magnitude():
    """The quantitative reason A3 exists. Same encoder, same edit: the delta aggregate must be
    substantially larger relative to the baseline aggregate than the mean one is."""
    enc = _encoder()
    a0 = [ALPHA_BASELINE] * N_SEG_COND
    a1 = list(a0)
    a1[segment_index("east", 3)] = 0.95
    t0, t1 = _tokens(a0), _tokens(a1)
    d_delta = (_delta_pool(t1, enc) - _delta_pool(t0, enc)).norm().item()
    d_mean = (_mean_pool(t1, enc) - _mean_pool(t0, enc)).norm().item()
    assert d_delta > 4.0 * d_mean, (
        "delta {:.4g} vs mean {:.4g}: expected roughly the 16x the mean-pool dilutes by"
        .format(d_delta, d_mean))


def test_order_invariance_survives(  ):
    """A2's property must NOT be lost. Permuting tokens leaves the aggregate identical, i.e. the
    aggregator still has no per-segment parameters."""
    enc = _encoder()
    a = [ALPHA_BASELINE] * N_SEG_COND
    a[segment_index("north", 1)] = 0.62
    a[segment_index("south", 4)] = 0.31
    toks = _tokens(a)
    ref = _delta_pool(toks, enc)
    rng = np.random.default_rng(0)
    for _ in range(5):
        perm = torch.tensor(rng.permutation(N_SEG_COND))
        assert torch.allclose(ref, _delta_pool(toks[perm], enc), atol=1e-6)


def test_position_is_still_read():
    """The companion to the test above. Without this, an aggregator that ignored position
    entirely would pass order-invariance and reproduce Track A's failure by another route."""
    enc = _encoder()
    aggs = []
    for name in ("west_2", "east_3", "south_1", "north_4"):
        wall, k = name.split("_")
        a = [ALPHA_BASELINE] * N_SEG_COND
        a[segment_index(wall, int(k))] = 0.70
        aggs.append(_delta_pool(_tokens(a), enc))
    for i in range(len(aggs)):
        for j in range(i + 1, len(aggs)):
            assert (aggs[i] - aggs[j]).abs().max().item() > 1e-4, (
                "same edit at different positions gave the same aggregate -- position-blind")


def test_additivity_over_disjoint_edits():
    """A sum of deltas is additive by construction. Worth pinning: it is what lets a multi-wall
    config be read as the superposition of its single-segment edits."""
    enc = _encoder()
    i, j = segment_index("west", 1), segment_index("north", 3)
    base = [ALPHA_BASELINE] * N_SEG_COND
    a_i = list(base); a_i[i] = 0.70
    a_j = list(base); a_j[j] = 0.42
    a_ij = list(base); a_ij[i] = 0.70; a_ij[j] = 0.42
    lhs = _delta_pool(_tokens(a_ij), enc)
    rhs = _delta_pool(_tokens(a_i), enc) + _delta_pool(_tokens(a_j), enc)
    assert torch.allclose(lhs, rhs, atol=1e-6)
