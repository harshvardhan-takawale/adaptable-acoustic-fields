"""P4-2: geometry-only tokens with a separate material channel, and the extent-weighted pool.

**Task A's hypothesis.** P4-1's `geom_token` reconstructs better than Arm C on every metric but
responds to an EDIT less linearly -- `edit_bw_slope` below Arm C in all five splits, worst S4
0.464 vs 0.789 (Q20). The suspicion is that entangling `m_hat` in the SHARED token MLP costs the
linear-in-m calibration. `geom_token_m` tests it by moving material out of the tokens and back
onto the channel where it was already proven, leaving geometry exactly where P4-1 showed it
works. The tests below pin the two properties that make that a clean experiment: the material
block must be byte-identical to `m_linear`'s, and the tokens must be the same token producer
`geom_token` uses, minus the m block.

**The pooling arm.** Token counts vary across shapes (4 edges for a rectangle, 6 for an L), so a
mean divides by a different number per shape and throws away total boundary extent. The
extent-weighted sum integrates over boundary length, which is what a boundary effect physically
does. The load-bearing test is that the two agree exactly when all extents are equal -- that is
what makes them a controlled A/B rather than two unrelated aggregators.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from aaf.models.conditioning_2d import (
    D_TOK,
    D_TOK_GEO,
    GEOM_TOKEN_M_COND_DIM,
    GEOM_TOKEN_M_DIM_2D,
    GEOM_TOKEN_M_MAT_DIM,
    MAX_SEG_POLY,
    TOKEN_AGG_DIM,
    TOK_EXTENT_IDX,
    build_cond_vector_2d,
    cond_dim_for,
    geom_token_m_features_2d,
    m_linear_features_2d,
    polygon_edge_tokens,
    rect_edge_tokens,
)
from aaf.walls import WORLD_SCALE

L_VERTS = [(0.0, 0.0), (6.0, 0.0), (6.0, 5.0), (3.0, 5.0), (3.0, 2.5), (0.0, 2.5)]
ALPHAS = [0.15, 0.50, 0.30, 0.70]


# ------------------------------------------------------------------ layout / registration
def test_widths_and_both_registration_gates():
    assert D_TOK_GEO == 19 and D_TOK == 26
    assert GEOM_TOKEN_M_DIM_2D == MAX_SEG_POLY * D_TOK_GEO + MAX_SEG_POLY + GEOM_TOKEN_M_MAT_DIM
    assert GEOM_TOKEN_M_DIM_2D == 268
    assert GEOM_TOKEN_M_COND_DIM == TOKEN_AGG_DIM + 28 == 92
    assert cond_dim_for("geom_token_m") == GEOM_TOKEN_M_DIM_2D
    # the model whitelist and the token branch are INDEPENDENT gates; read as text so this
    # runs on CPU (importing inr_2d pulls in tinycudann, which needs a GPU).
    from pathlib import Path
    src = Path(__file__).resolve().parents[1].joinpath("aaf/models/inr_2d.py").read_text()
    wl = src.split("if cond_source not in (", 1)[1].split("):", 1)[0]
    assert '"geom_token_m"' in wl, "geom_token_m missing from the model whitelist"
    br = src.split("self.cond_source in (", 1)[1].split(")", 1)[0]
    assert '"geom_token_m"' in br, "geom_token_m missing from the token-encoder branch"


def test_extent_index_matches_the_featurizer_in_both_token_widths():
    """TOK_EXTENT_IDX is read by inr_2d to build the extent-weighted pool. Two files indexing
    the same column by hand is how a silent off-by-one gets in, so it is pinned here against
    the actual feature vectors."""
    toks = rect_edge_tokens(6.0, 5.0, ALPHAS)
    # 26-d tokens (geom_token)
    v = build_cond_vector_2d("geom_token", 6.0, 5.0, ALPHAS)
    t0 = v[:MAX_SEG_POLY * D_TOK].reshape(MAX_SEG_POLY, D_TOK)[0]
    assert t0[TOK_EXTENT_IDX].item() == pytest.approx(toks[0][4])
    # 19-d tokens (geom_token_m) -- same index, shorter token
    vm = build_cond_vector_2d("geom_token_m", 6.0, 5.0, ALPHAS)
    tm0 = vm[:MAX_SEG_POLY * D_TOK_GEO].reshape(MAX_SEG_POLY, D_TOK_GEO)[0]
    assert tm0[TOK_EXTENT_IDX].item() == pytest.approx(toks[0][4])
    assert TOK_EXTENT_IDX < D_TOK_GEO, "extent must be addressable in the SHORT token too"


# ---------------------------------------------------------- the two properties that isolate A
def test_material_block_is_byte_identical_to_m_linear():
    """If this drifts, Task A stops being a controlled test of the entanglement hypothesis and
    becomes 'a different material encoding also changed'."""
    v = build_cond_vector_2d("geom_token_m", 6.0, 5.0, ALPHAS)
    ml = m_linear_features_2d(6.0, 5.0, ALPHAS)
    assert torch.allclose(v[-GEOM_TOKEN_M_MAT_DIM:], ml[32:60], atol=0.0)


def test_tokens_are_geom_tokens_minus_the_m_block():
    """Same token producer as geom_token, truncated. Guards against the two arms silently
    disagreeing about geometry, which would confound the comparison."""
    v = build_cond_vector_2d("geom_token", 6.0, 5.0, ALPHAS)
    vm = build_cond_vector_2d("geom_token_m", 6.0, 5.0, ALPHAS)
    a = v[:MAX_SEG_POLY * D_TOK].reshape(MAX_SEG_POLY, D_TOK)[:, :D_TOK_GEO]
    b = vm[:MAX_SEG_POLY * D_TOK_GEO].reshape(MAX_SEG_POLY, D_TOK_GEO)
    assert torch.allclose(a, b, atol=0.0)


def test_material_does_not_reach_the_tokens():
    """The defining property of the arm: changing a wall's absorption must move ONLY the
    material tail, never a token."""
    v1 = build_cond_vector_2d("geom_token_m", 6.0, 5.0, [0.15, 0.15, 0.15, 0.15])
    v2 = build_cond_vector_2d("geom_token_m", 6.0, 5.0, [0.15, 0.90, 0.15, 0.15])
    n = MAX_SEG_POLY * D_TOK_GEO + MAX_SEG_POLY
    assert torch.allclose(v1[:n], v2[:n], atol=0.0), "an alpha change leaked into the tokens"
    assert not torch.allclose(v1[n:], v2[n:]), "the material tail did not respond to alpha"


def test_validity_mask_sits_between_tokens_and_material():
    v = build_cond_vector_2d("geom_token_m", 6.0, 5.0, ALPHAS)
    n = MAX_SEG_POLY * D_TOK_GEO
    mask = v[n:n + MAX_SEG_POLY]
    assert mask.sum().item() == 4.0                     # a rectangle fills 4 slots
    assert torch.all(mask[:4] == 1.0) and torch.all(mask[4:] == 0.0)
    lt = geom_token_m_features_2d(polygon_edge_tokens(L_VERTS, [0.15] * 6), ALPHAS)
    assert lt[n:n + MAX_SEG_POLY].sum().item() == 6.0   # an L fills 6


def test_wrong_alpha_count_raises():
    with pytest.raises(ValueError, match="wall alphas"):
        geom_token_m_features_2d(rect_edge_tokens(6.0, 5.0, ALPHAS), [0.15] * 6)


# ------------------------------------------------------------------------- pooling behaviour
def _enc(d_tok, seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d_tok, TOKEN_AGG_DIM), nn.ReLU(),
                         nn.Linear(TOKEN_AGG_DIM, TOKEN_AGG_DIM))


def _pool(vec, enc, n_tok_d, how):
    n = MAX_SEG_POLY * n_tok_d
    t = vec[:n].reshape(MAX_SEG_POLY, n_tok_d)
    m = vec[n:n + MAX_SEG_POLY].reshape(MAX_SEG_POLY, 1)
    with torch.no_grad():
        if how == "masked_mean":
            return (enc(t) * m).sum(0) / m.sum(0).clamp(min=1.0)
        w = t[:, TOK_EXTENT_IDX:TOK_EXTENT_IDX + 1] * m
        return (enc(t) * w).sum(0)


def test_extent_sum_reduces_to_masked_mean_when_extents_are_equal():
    """THE test that makes the two a controlled A/B. With every valid extent equal to e and k
    valid slots, extent_sum = e*k*masked_mean -- i.e. the same direction, scaled. If they ever
    differ in direction here, the variant is measuring something other than the weighting."""
    enc = _enc(D_TOK)
    # a square has all four extents equal
    v = build_cond_vector_2d("geom_token", 5.0, 5.0, [0.15] * 4)
    mm = _pool(v, enc, D_TOK, "masked_mean")
    es = _pool(v, enc, D_TOK, "extent_sum")
    e, k = 5.0 / WORLD_SCALE, 4
    assert torch.allclose(es, mm * (e * k), atol=1e-5)


def test_extent_sum_differs_when_extents_differ():
    """Companion to the test above -- otherwise it would pass for an aggregator that ignores
    extent entirely."""
    enc = _enc(D_TOK)
    v = build_cond_vector_2d("geom_token", 7.0, 4.0, [0.15] * 4)   # extents 0.7, 0.4, 0.7, 0.4
    mm = _pool(v, enc, D_TOK, "masked_mean")
    es = _pool(v, enc, D_TOK, "extent_sum")
    mean_ext = (0.7 + 0.4 + 0.7 + 0.4) / 4
    assert not torch.allclose(es, mm * (mean_ext * 4), atol=1e-4)


def test_extent_sum_ignores_padding():
    """Padded slots carry extent 0 so they are already self-masking, but the mask must still be
    applied -- it is the single source of truth for validity."""
    enc = _enc(D_TOK)
    v = build_cond_vector_2d("geom_token", 6.0, 5.0, [0.15] * 4)
    full = _pool(v, enc, D_TOK, "extent_sum")
    n = MAX_SEG_POLY * D_TOK
    t = v[:n].reshape(MAX_SEG_POLY, D_TOK)
    with torch.no_grad():
        only4 = (enc(t[:4]) * t[:4, TOK_EXTENT_IDX:TOK_EXTENT_IDX + 1]).sum(0)
    assert torch.allclose(full, only4, atol=1e-6)


def test_extent_sum_grows_with_boundary_length():
    """The property the variant exists for: a bigger room has more boundary and must produce a
    larger aggregate. masked_mean cannot express this."""
    enc = _enc(D_TOK)
    small = _pool(build_cond_vector_2d("geom_token", 5.0, 4.0, [0.15] * 4), enc, D_TOK, "extent_sum")
    big = _pool(build_cond_vector_2d("geom_token", 7.0, 5.5, [0.15] * 4), enc, D_TOK, "extent_sum")
    assert big.norm() > small.norm()
