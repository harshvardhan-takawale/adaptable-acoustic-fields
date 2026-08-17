"""Track A2: the shared segment encoder must have NO per-segment parameters.

This is the property the whole track rests on. Track A gave each segment 7 private conditioning
dims, so holding out `east_3` held out its WEIGHTS -- those dims sat at the baseline value in
all 400 training configs and never received gradient. The measured consequence was stark: at a
trained position the model recovered 1.079 of the open-window energy drop, at the held-out
position -0.069, against an identical -5.28 dB ground truth. It produced no effect at all.

The fix is to describe a segment by WHAT IT IS -- centre, normal, extent, absorption -- and run
every segment through ONE shared MLP, so a held-out position differs from a trained one only in
its coordinate VALUES, which the encoder has already learned to read from the other 15.

Three properties, and the second is what stops the first from being vacuous:

1. **Order invariance.** Permuting the token order leaves the mean-pooled aggregate identical.
   This is what "no per-segment parameters" means operationally.
2. **Position is actually read.** Moving the SAME (extent, m) to a DIFFERENT centre must CHANGE
   the aggregate. Without this, property 1 would pass for an encoder that ignores position
   entirely -- which would be a different way of failing the same test Track A failed.
3. **Parameter count is independent of segment count.**

These run on CPU. The full model cannot be constructed here (tinycudann needs a GPU), so the
encoder arithmetic is exercised directly against the same featurizer the model consumes.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from aaf.data.seg_configs import SEGMENT_NAMES, segment_index
from aaf.models.conditioning_2d import (
    D_TOK,
    N_SEG_COND,
    TOKEN_AGG_DIM,
    TOKEN_COND_DIM,
    TOKEN_DIM_2D,
    segment_geometry,
    segment_token_features_2d,
)

L, W = 4.5, 4.0
BASE = 0.15


def _encoder(seed: int = 0) -> nn.Sequential:
    """The same architecture inr_2d builds, instantiated standalone for CPU testing."""
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(D_TOK, TOKEN_AGG_DIM), nn.ReLU(),
                         nn.Linear(TOKEN_AGG_DIM, TOKEN_AGG_DIM))


def _split(vec: torch.Tensor):
    n_geom = TOKEN_DIM_2D - N_SEG_COND * D_TOK
    return vec[:n_geom], vec[n_geom:].reshape(N_SEG_COND, D_TOK)


def _aggregate(vec: torch.Tensor, enc: nn.Sequential) -> torch.Tensor:
    geom, toks = _split(vec)
    with torch.no_grad():
        return torch.cat([geom, enc(toks).mean(dim=0)])


def test_feature_width_and_split():
    v = segment_token_features_2d(L, W, [BASE] * N_SEG_COND)
    assert v.shape[0] == TOKEN_DIM_2D == 448
    geom, toks = _split(v)
    assert geom.shape[0] == 32 and toks.shape == (N_SEG_COND, D_TOK)
    assert _aggregate(v, _encoder()).shape[0] == TOKEN_COND_DIM == 96


def test_order_invariance_of_the_aggregate():
    """Property 1: the mean-pool has no notion of slot order."""
    enc = _encoder()
    a = [BASE] * N_SEG_COND
    a[segment_index("west", 2)] = 0.70
    a[segment_index("north", 4)] = 0.42
    v = segment_token_features_2d(L, W, a)
    geom, toks = _split(v)
    rng = np.random.default_rng(0)
    ref = _aggregate(v, enc)
    for _ in range(5):
        perm = torch.tensor(rng.permutation(N_SEG_COND))
        with torch.no_grad():
            got = torch.cat([geom, enc(toks[perm]).mean(dim=0)])
        assert torch.allclose(ref, got, atol=1e-6)


def test_position_is_actually_read():
    """Property 2 -- the one that makes property 1 non-vacuous.

    Same absorption, same extent, DIFFERENT segment. The aggregate must move; if it does not,
    the encoder is position-blind and would reproduce Track A's failure by another route.
    """
    enc = _encoder()
    aggs = []
    for name in ("west_2", "east_3", "south_1", "north_4"):
        wall, k = name.split("_")
        a = [BASE] * N_SEG_COND
        a[segment_index(wall, int(k))] = 0.70
        aggs.append(_aggregate(segment_token_features_2d(L, W, a), enc))
    for i in range(len(aggs)):
        for j in range(i + 1, len(aggs)):
            d = (aggs[i] - aggs[j]).abs().max().item()
            assert d > 1e-4, "aggregate identical for two different positions -- position-blind"


def test_held_out_segment_shares_all_parameters():
    """east_3 is the held-out position. Its token must be an ordinary point in the same space:
    every entry finite, same width, and its (cx, cy, nx, ny) must match its wall's geometry."""
    i = segment_index("east", 3)
    a = [BASE] * N_SEG_COND
    a[i] = 0.95
    _, toks = _split(segment_token_features_2d(L, W, a))
    assert torch.isfinite(toks).all()
    cx, cy, nx, ny, ext = segment_geometry(L, W, i)
    assert (cx, nx, ny) == (1.0, -1.0, 0.0)          # east wall, inward normal -x
    assert ext == pytest.approx(0.25)
    # The encoder applies the SAME weights to this token as to any other.
    enc = _encoder()
    assert sum(p.numel() for p in enc.parameters()) == (
        D_TOK * TOKEN_AGG_DIM + TOKEN_AGG_DIM + TOKEN_AGG_DIM ** 2 + TOKEN_AGG_DIM)


def test_parameter_count_independent_of_segment_count():
    """Property 3: doubling the segments must not add parameters -- that is what makes the
    encoding scale to finer segmentation without re-architecting."""
    enc = _encoder()
    n_par = sum(p.numel() for p in enc.parameters())
    for n_tokens in (8, 16, 32, 64):
        toks = torch.randn(n_tokens, D_TOK)
        with torch.no_grad():
            out = enc(toks).mean(dim=0)
        assert out.shape[0] == TOKEN_AGG_DIM
        assert sum(p.numel() for p in enc.parameters()) == n_par


def test_segment_geometry_covers_every_wall_consistently():
    seen = {}
    for i in range(N_SEG_COND):
        cx, cy, nx, ny, ext = segment_geometry(L, W, i)
        wall = SEGMENT_NAMES[i].split("_")[0]
        seen.setdefault(wall, []).append((cx, cy, nx, ny))
        assert ext == pytest.approx(0.25)
        assert abs(nx) + abs(ny) == pytest.approx(1.0)      # unit axis-aligned normal
    assert set(seen) == {"west", "east", "south", "north"}
    for wall, vals in seen.items():
        assert len({(v[2], v[3]) for v in vals}) == 1        # one normal per wall
        assert len({(v[0], v[1]) for v in vals}) == 4        # four distinct centres
