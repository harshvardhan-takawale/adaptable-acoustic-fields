"""P4-7 m4 (D169): the CX500 generator. CPU only.

The corpus exists to break the fixed source and the fixed notch positions WITHOUT leaking the held-out
U family or either frozen test set's d_hat slab, so those exclusions are tested on the full 500-room draw,
and the traced polygon is checked against an independent rasterisation and against the family code's own
polygon for the configurations both can express.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from aaf.data.cx_rooms import (CORNERS, MIN_PASSAGE_M, POSITIONS, SLAB, SRC_CLEAR_M, U_PAIRS, CxNotch,
                               edge_distance, rect_gap, reflex_vertices, room_ok, sample_cx_rooms,
                               topology, trace_polygon)
from aaf.sim.polygon_geom import polygon_contains


@pytest.fixture(scope="module")
def rooms():
    return sample_cx_rooms(500)


def _area(v):
    n = len(v)
    return 0.5 * sum(v[i][0] * v[(i + 1) % n][1] - v[(i + 1) % n][0] * v[i][1] for i in range(n))


def test_no_u_topology_no_slab_and_passages(rooms):
    for r in rooms:
        assert topology(r.notches) != "U"
        assert frozenset(n.pos for n in r.notches) not in U_PAIRS
        for n in r.notches:
            assert not SLAB[0] <= n.d_hat(r.L, r.W) <= SLAB[1]
            if n.pos in CORNERS:
                assert not SLAB[0] <= n.w / (0.45 * r.L) <= SLAB[1]
        if len(r.notches) == 2:
            assert rect_gap(*r.rects) >= MIN_PASSAGE_M - 1e-9


def test_polygon_matches_rasterised_bbox_minus_rects(rooms):
    rng = np.random.default_rng(0)
    for r in rooms[::10]:
        v = r.verts
        assert _area(v) == pytest.approx(r.L * r.W - sum((x1 - x0) * (y1 - y0) for x0, x1, y0, y1 in r.rects))
        p = np.stack([rng.uniform(0, r.L, 4000), rng.uniform(0, r.W, 4000)], -1)
        in_rect = np.zeros(len(p), bool)
        for x0, x1, y0, y1 in r.rects:
            in_rect |= (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1)
        np.testing.assert_array_equal(polygon_contains(v, p), ~in_rect)
        assert min(v, key=lambda q: (q[1], q[0])) == v[0]          # canonical ring
        assert len(reflex_vertices(v)) == sum(1 if n.pos in CORNERS else 2 for n in r.notches)


def test_reflex_count_by_position():
    L, W = 6.0, 5.0
    assert len(reflex_vertices(trace_polygon(L, W, [CxNotch("NW", 1.0, 1.0).rect(L, W)]))) == 1
    assert len(reflex_vertices(trace_polygon(L, W, [CxNotch("midE", 1.0, 1.0, 2.0).rect(L, W)]))) == 2
    assert len(trace_polygon(L, W, [])) == 4


def test_nw_notch_polygon_equals_family_code():
    from aaf.data.multi_notch import Notch, verts_for_notches
    for L, W, d, w in ((6.0, 5.0, 1.2, 1.6), (5.5, 4.4, 0.4, 2.0)):
        mine = trace_polygon(L, W, [CxNotch("NW", d, w).rect(L, W)])
        theirs = verts_for_notches(L, W, [Notch("NW", d, w)])
        assert [tuple(map(float, p)) for p in theirs] == [tuple(p) for p in mine]
    for L, W in ((6.0, 5.0),):
        mine = trace_polygon(L, W, [CxNotch("midN", 1.0, 1.5, 2.0).rect(L, W), CxNotch("SE", 0.8, 1.0).rect(L, W)])
        theirs = verts_for_notches(L, W, [Notch("midN", 1.0, 1.5, x0=2.0), Notch("SE", 0.8, 1.0)])
        assert sorted(map(tuple, theirs)) == sorted(map(tuple, mine))


def test_source_in_air_with_clearance_and_on_grid(rooms):
    for r in rooms:
        assert polygon_contains(r.verts, np.asarray([r.src]))[0]
        assert edge_distance(r.verts, [r.src])[0] >= SRC_CLEAR_M - 1e-9
        for x in (r.L, r.W, r.src[0], r.src[1]) + tuple(c for n in r.notches for c in (n.d, n.w, n.off)):
            assert abs(x / 0.02 - round(x / 0.02)) < 1e-6
    # the source is NOT fixed: it spreads over the rooms (teeth for a silently constant draw)
    s = np.asarray([r.src for r in rooms])
    assert s[:, 0].std() > 0.8 and s[:, 1].std() > 0.6


def test_positions_and_topologies_are_diverse(rooms):
    pos = Counter(n.pos for r in rooms for n in r.notches)
    assert set(pos) == set(POSITIONS)
    assert min(pos.values()) > 40
    kinds = Counter(topology(r.notches) for r in rooms)
    assert set(kinds) == {"L", "T", "DN", "LT", "TT"}
    assert len({r.filename for r in rooms}) == len(rooms)


def test_prefix_stable():
    a = sample_cx_rooms(12)
    b = sample_cx_rooms(30)
    assert a == b[:12]


def test_room_ok_rejects_u_and_slab():
    L, W = 6.0, 5.0
    assert room_ok(L, W, [CxNotch("SW", 1.0, 1.0), CxNotch("SE", 1.0, 1.0)])[1] == "U topology"
    assert room_ok(L, W, [CxNotch("NE", 1.0, 1.0), CxNotch("SE", 1.0, 1.0)])[1] == "U topology"
    assert room_ok(L, W, [CxNotch("NW", 0.5 * 0.45 * W, 1.0)])[1] == "d_hat in slab"
    assert room_ok(L, W, [CxNotch("midW", 0.5 * 0.45 * L, 1.0, 2.0)])[1] == "d_hat in slab"
    assert room_ok(L, W, [CxNotch("NW", 1.0, 1.0), CxNotch("midN", 1.0, 1.0, 1.5)])[0] is False   # 0.5 m gap
    assert room_ok(L, W, [CxNotch("NW", 1.0, 1.0), CxNotch("SE", 1.0, 1.0)])[0] is True
