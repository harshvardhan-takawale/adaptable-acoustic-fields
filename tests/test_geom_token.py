"""P4-1 Stage 0/1: token-ONLY geometry in absolute world coordinates (D64).

The arm exists because a Z-corridor has no meaningful ``(L, W)``. Every geometric fact must ride
on the boundary tokens, so the global geometry prefix is dropped entirely and the tokens must
carry absolute position.

The property that makes shape transfer possible is that a physical edge encodes IDENTICALLY in
every room. The pre-P4-1 tokens did the opposite: ``segment_geometry`` returns unit-square
coordinates and ignores its own ``L, W`` arguments, so a wall of a 3x3 room and a wall of a 6x5
room produced byte-identical positional channels. Tests 2-4 pin the fix.

These run on CPU against the featurizer and a standalone replica of the pooling arithmetic; the
full model needs a GPU (tinycudann).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from aaf.models.conditioning_2d import (
    D_TOK,
    GEOM_TOKEN_COND_DIM,
    GEOM_TOKEN_DIM_2D,
    MAX_SEG_POLY,
    TOKEN_AGG_DIM,
    build_cond_vector_2d,
    cond_dim_for,
    geom_token_features_2d,
    m_hat_seg,
    polygon_edge_tokens,
    rect_edge_tokens,
)
from aaf.walls import WORLD_SCALE

L_ROOM_VERTS = [(0.0, 0.0), (6.0, 0.0), (6.0, 5.0), (3.0, 5.0), (3.0, 2.5), (0.0, 2.5)]


def _encoder(seed: int = 0) -> nn.Sequential:
    """The same architecture inr_2d builds, standalone for CPU testing."""
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(D_TOK, TOKEN_AGG_DIM), nn.ReLU(),
                         nn.Linear(TOKEN_AGG_DIM, TOKEN_AGG_DIM))


def _masked_pool(vec: torch.Tensor, enc: nn.Sequential) -> torch.Tensor:
    """Replica of the model's masked_mean branch."""
    n = MAX_SEG_POLY * D_TOK
    t = vec[:n].reshape(MAX_SEG_POLY, D_TOK)
    m = vec[n:].reshape(MAX_SEG_POLY, 1)
    with torch.no_grad():
        return (enc(t) * m).sum(dim=0) / m.sum(dim=0).clamp(min=1.0)


# --------------------------------------------------------------------------- widths / wiring
def test_widths_and_registration():
    assert GEOM_TOKEN_DIM_2D == MAX_SEG_POLY * D_TOK + MAX_SEG_POLY == 324
    assert GEOM_TOKEN_COND_DIM == TOKEN_AGG_DIM == 64
    # cond_dim_for and the model whitelist are INDEPENDENT gates; registering in only one has
    # broken three launches historically, so both are asserted. inr_2d is read as TEXT rather
    # than imported -- importing it pulls in tinycudann, which needs a GPU, and this check must
    # run in the ordinary CPU test pass where such a mistake would actually be caught.
    assert cond_dim_for("geom_token") == GEOM_TOKEN_DIM_2D
    from pathlib import Path
    src = Path(__file__).resolve().parents[1].joinpath("aaf/models/inr_2d.py").read_text()
    whitelist = src.split("if cond_source not in (", 1)[1].split("):", 1)[0]
    assert '"geom_token"' in whitelist, "geom_token missing from the model-side whitelist"
    branch = src.split("self.cond_source in (", 1)[1].split(")", 1)[0]
    assert '"geom_token"' in branch, "geom_token missing from the token-encoder branch"


def test_vector_shape_and_validity_mask():
    v = build_cond_vector_2d("geom_token", 6.0, 5.0, [0.15, 0.50, 0.15, 0.70])
    assert v.shape == (GEOM_TOKEN_DIM_2D,)
    mask = v[-MAX_SEG_POLY:]
    assert mask.sum().item() == 4.0, "a rectangle must fill exactly 4 slots"
    assert torch.all(mask[:4] == 1.0) and torch.all(mask[4:] == 0.0)
    # padded slots must be exactly zero, not stale memory
    pad = v[:MAX_SEG_POLY * D_TOK].reshape(MAX_SEG_POLY, D_TOK)[4:]
    assert torch.all(pad == 0.0)


# ------------------------------------------------------------------ THE absolute-coordinate property
def test_positions_are_absolute_not_bbox_normalized():
    """The core D64 property. Under bbox normalization the east wall of ANY room maps to
    cx = 1.0; under the world scale it maps to L / WORLD_SCALE and therefore moves with L."""
    e6 = rect_edge_tokens(6.0, 5.0, [0.15] * 4)[1]
    e4 = rect_edge_tokens(4.0, 5.0, [0.15] * 4)[1]
    assert e6[0] == pytest.approx(6.0 / WORLD_SCALE)
    assert e4[0] == pytest.approx(4.0 / WORLD_SCALE)
    assert abs(e6[0] - e4[0]) > 1e-6, "east wall encodes identically in both rooms -> bbox-normalized"


def test_the_same_physical_edge_encodes_identically_in_different_rooms():
    """Companion to the test above, and the one that actually buys shape transfer: the south
    wall of a 6x5 room and of a 6x3 room IS the same physical segment, so it must produce the
    same token even though the rooms differ."""
    s_a = rect_edge_tokens(6.0, 5.0, [0.15] * 4)[0]
    s_b = rect_edge_tokens(6.0, 3.0, [0.15] * 4)[0]
    assert s_a == pytest.approx(s_b)


def test_extent_is_metres_not_a_wall_fraction():
    """The pre-P4-1 tokens hard-coded extent = 0.25 (a quarter of a wall) for every segment in
    every room -- carrying no length information at all."""
    for L, W in ((6.0, 5.0), (3.2, 4.7)):
        toks = rect_edge_tokens(L, W, [0.15] * 4)
        assert toks[0][4] == pytest.approx(L / WORLD_SCALE)   # south spans L
        assert toks[1][4] == pytest.approx(W / WORLD_SCALE)   # east spans W


# ------------------------------------------------------------------------------- polygon geometry
def test_l_room_edges_normals_and_extents():
    toks = polygon_edge_tokens(L_ROOM_VERTS, [0.15] * 6)
    assert len(toks) == 6
    ext_m = [round(t[4] * WORLD_SCALE, 6) for t in toks]
    assert ext_m == [6.0, 5.0, 3.0, 2.5, 3.0, 2.5]
    assert sum(ext_m) == pytest.approx(22.0)                 # L-room perimeter
    # inward normals, unit and axis-aligned
    for (_, _, nx, ny, _, _) in toks:
        assert abs(nx) + abs(ny) == pytest.approx(1.0)
    assert (round(toks[0][2]), round(toks[0][3])) == (0, 1)    # south -> +y
    assert (round(toks[1][2]), round(toks[1][3])) == (-1, 0)   # east  -> -x
    # the notch's vertical edge x=3 has the room on its +x side for y > 2.5
    assert (round(toks[3][2]), round(toks[3][3])) == (1, 0)


def test_clockwise_vertices_raise():
    """A clockwise list silently inverts every inward normal -- the model would learn a room
    turned inside out and nothing downstream would notice. Fail loudly instead."""
    with pytest.raises(ValueError, match="counter-clockwise"):
        polygon_edge_tokens(L_ROOM_VERTS[::-1], [0.15] * 6)


def test_too_many_edges_raises():
    n = MAX_SEG_POLY + 1
    verts = [(math.cos(2 * math.pi * i / n) + 2.0, math.sin(2 * math.pi * i / n) + 2.0)
             for i in range(n)]
    with pytest.raises(ValueError, match="MAX_SEG_POLY"):
        polygon_edge_tokens(verts, [0.15] * n)


def test_absorption_uses_the_token_arm_normalization():
    """m_hat must use M_NORM_SEG_COND = 3.0 like the other token arms, NOT walls.M_NORM = ln 5
    used by the scalar arms -- a model trained under one is not comparable to the other."""
    toks = rect_edge_tokens(4.0, 4.0, [0.15, 0.50, 0.70, 0.95])
    got = [t[5] for t in toks]                                # south, east, north, west order
    assert got == pytest.approx([m_hat_seg(0.70), m_hat_seg(0.50),
                                 m_hat_seg(0.95), m_hat_seg(0.15)])


# ------------------------------------------------------------------------------- pooling behaviour
def test_padding_does_not_move_the_aggregate():
    """The reason the mask exists. A zero token is a VALID point in token space (an edge at the
    origin, zero extent, baseline absorption); an unmasked mean would drag the aggregate toward
    it by n_pad / MAX_SEG_POLY = 8/12."""
    enc = _encoder()
    v = build_cond_vector_2d("geom_token", 6.0, 5.0, [0.15] * 4)
    masked = _masked_pool(v, enc)
    n = MAX_SEG_POLY * D_TOK
    with torch.no_grad():
        naive = enc(v[:n].reshape(MAX_SEG_POLY, D_TOK)).mean(dim=0)
    true4 = _masked_pool(torch.cat([v[:n], torch.tensor([1.0] * 4 + [0.0] * 8)]), enc)
    assert torch.allclose(masked, true4, atol=1e-6)
    assert (masked - naive).abs().max() > 1e-3, "mask had no effect -- padding is leaking in"


def test_pool_is_order_invariant():
    """No per-token parameters: permuting the slots must leave the aggregate identical."""
    enc = _encoder()
    toks = polygon_edge_tokens(L_ROOM_VERTS, [0.15, 0.5, 0.7, 0.2, 0.3, 0.9])
    ref = _masked_pool(geom_token_features_2d(toks), enc)
    import numpy as np
    rng = np.random.default_rng(0)
    for _ in range(5):
        perm = list(rng.permutation(len(toks)))
        got = _masked_pool(geom_token_features_2d([toks[i] for i in perm]), enc)
        assert torch.allclose(ref, got, atol=1e-6)


def test_geometry_survives_pooling():
    """Guards the real Gate-0 risk. Raw wall midpoints average to the room centre and raw
    extents to (L+W)/2, so a LINEAR pool would confuse rooms that differ only in shape. The
    per-token nonlinear encoder must keep them separable."""
    enc = _encoder()
    aggs = {}
    for (L, W) in ((6.0, 5.0), (5.0, 6.0), (4.0, 4.0), (6.0, 4.0)):
        aggs[(L, W)] = _masked_pool(build_cond_vector_2d("geom_token", L, W, [0.15] * 4), enc)
    keys = list(aggs)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            d = (aggs[keys[i]] - aggs[keys[j]]).abs().max().item()
            assert d > 1e-4, "rooms {} and {} pool to the same vector".format(keys[i], keys[j])


def test_rectangle_and_polygon_agree_on_a_rectangle():
    """rect_edge_tokens is a convenience wrapper; it must not drift from the general path."""
    L, W = 5.3, 3.7
    a = rect_edge_tokens(L, W, [0.15, 0.50, 0.70, 0.30])
    b = polygon_edge_tokens([(0, 0), (L, 0), (L, W), (0, W)], [0.70, 0.50, 0.30, 0.15])
    assert a == pytest.approx(b)
