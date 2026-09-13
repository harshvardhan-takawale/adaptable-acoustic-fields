"""P4-4 Task 3: U / T / diagonal-double-notch rooms -- two reflex corners.

The model stack needs no changes for these (MAX_SEG_POLY = 12 already covers 4/6/8/10 tokens),
so every way this can go wrong is geometric, and most of the ways are SILENT:

* a clockwise ring inverts every inward normal and trains the model on a room turned inside out;
* a mis-indexed solid block puts a wall one cell from where the polygon says it is;
* a collapsed bounding box makes `wall_extent` measure the notch floor instead of the wall;
* a disconnected or self-intersecting domain is accepted by the solver and the dead component is
  identically zero -- and the builder's node-for-node cross-check would NOT catch it, because
  both sides of that comparison use the same bad polygon.

So the mask is checked against `polygon_contains` node for node, exactly as the builder does per
room, and every rejection guard is tested by constructing a draw that must trip it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import aaf.sim.fdtd_2d as F
from aaf.data.multi_notch import (
    FAMILIES, MIN_STEM_M, N_TOKENS, Notch, geometry_ok, reflex_corners, solid_blocks,
    verts_for_notches,
)
from aaf.data.shape_configs import SRC
from aaf.models.conditioning_2d import MAX_SEG_POLY, polygon_edge_tokens
from aaf.sim.polygon_geom import polygon_contains

L, W, DX = 6.00, 5.00, 0.02


def _fam(name):
    if name == "rect":
        return []
    if name == "L":
        return [Notch("NW", 1.20, 1.60)]
    if name == "U":
        return [Notch("NW", 1.20, 1.60), Notch("NE", 0.90, 1.40)]
    if name == "T":
        return [Notch("midN", 1.20, 2.00, x0=1.60)]
    if name == "DN":
        return [Notch("NW", 1.20, 1.60), Notch("SE", 1.00, 1.40)]
    raise KeyError(name)


# ----------------------------------------------------------------------------- the ring
@pytest.mark.parametrize("fam", sorted(FAMILIES))
def test_vertex_count_and_ccw(fam):
    v = verts_for_notches(L, W, _fam(fam))
    assert len(v) == N_TOKENS[fam], "{} gave {} vertices".format(fam, len(v))
    n = len(v)
    area2 = sum(v[i][0] * v[(i + 1) % n][1] - v[(i + 1) % n][0] * v[i][1] for i in range(n))
    assert area2 > 0, "{} ring is clockwise; every inward normal would invert".format(fam)


@pytest.mark.parametrize("fam", sorted(FAMILIES))
def test_tokenizes_within_the_existing_budget(fam):
    """The whole claim that Task 3 needs no model change rests on this."""
    v = verts_for_notches(L, W, _fam(fam))
    assert len(v) <= MAX_SEG_POLY
    toks = polygon_edge_tokens(v, [0.15] * len(v))     # raises on CW or zero-length edges
    assert len(toks) == len(v)


def test_collinear_and_coincident_vertices_are_merged():
    """A redundant vertex would emit two tokens for one physical wall -- changing the token COUNT
    without changing the room, which is exactly the confound this chunk exists to measure."""
    # a notch whose width spans the entire north edge minus nothing on one side is degenerate;
    # a zero-depth notch must vanish entirely rather than leave collinear points behind
    v = verts_for_notches(L, W, [Notch("NW", 0.0, 1.6)])
    assert len(v) == 4, "a zero-depth notch must collapse back to a rectangle, got {}".format(v)


@pytest.mark.parametrize("fam", sorted(FAMILIES))
def test_source_stays_in_air(fam):
    """SRC is frozen at (0.5, 1.6) for the whole corpus and the trainer keeps only the LAST
    config's source, so a family that swallowed it would train every room against the wrong
    source with no error anywhere."""
    v = verts_for_notches(L, W, _fam(fam))
    assert bool(polygon_contains(v, np.asarray([SRC], dtype=float))[0])


# ------------------------------------------------------------- the solid mask vs the polygon
@pytest.mark.parametrize("fam", sorted(FAMILIES))
def test_solid_mask_matches_the_polygon_node_for_node(fam):
    """THE check. It is what caught P4-1's wrong-corner mask, and the index convention here is
    asymmetric on purpose -- an interior face leaves its node air, a face on an outer wall does
    not -- so a T (two interior x-faces) exercises a case an L never reaches."""
    notches = _fam(fam)
    nx, dxx = F._fit_axis(L, DX, "L")
    ny, dxy = F._fit_axis(W, DX, "W")
    m = solid_blocks(nx, ny, dxx, dxy, L, W, notches)
    geom = F.build_geometry(L, W, [0.15] * 4, dx=DX,
                            extra_walls=([{"type": "mask", "solid": m, "alpha": 0.15}]
                                         if m.any() else None))
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    want = polygon_contains(verts_for_notches(L, W, notches),
                            np.stack([ii * dxx, jj * dxy], axis=-1))
    assert int((geom.air != want).sum()) == 0, "{}: {} nodes disagree".format(
        fam, int((geom.air != want).sum()))


def test_every_reflex_corner_is_reported():
    """Receiver sampling clears the r^(2/3) singularity at each reflex vertex. Missing one puts
    the receivers nearest it in the unresolved singular field and scores them as if fine."""
    assert reflex_corners(L, W, _fam("rect")) == []
    assert len(reflex_corners(L, W, _fam("L"))) == 1
    assert len(reflex_corners(L, W, _fam("U"))) == 2
    assert len(reflex_corners(L, W, _fam("T"))) == 2
    assert len(reflex_corners(L, W, _fam("DN"))) == 2
    for fam in ("L", "U", "T", "DN"):
        v = verts_for_notches(L, W, _fam(fam))
        for c in reflex_corners(L, W, _fam(fam)):
            assert any(abs(c[0] - p[0]) < 1e-9 and abs(c[1] - p[1]) < 1e-9 for p in v), \
                "{}: reported reflex corner {} is not a vertex".format(fam, c)


# -------------------------------------------------------------------- the rejection guards
def test_rejects_a_collapsed_bounding_box():
    """Two north notches that consume the whole north edge. `wall_extent` would then silently
    measure the notch floor instead of the north wall, and nothing else checks max(y) == W."""
    ok, why = geometry_ok(L, W, [Notch("NW", 1.2, 2.7), Notch("NE", 1.2, 2.7)])
    assert not ok and ("stem" in why or "collapsed" in why), why


def test_rejects_a_too_thin_u_stem():
    ok, why = geometry_ok(L, W, [Notch("NW", 1.2, 2.65), Notch("NE", 1.2, 2.65)])
    assert not ok and "stem" in why, why


def test_rejects_thin_t_shoulders():
    ok, why = geometry_ok(L, W, [Notch("midN", 1.2, 2.0, x0=0.2)])
    assert not ok and "shoulder" in why, why


def test_rejects_an_over_deep_notch():
    ok, why = geometry_ok(L, W, [Notch("NW", 0.45 * W + 0.5, 1.6)])
    assert not ok and "depth" in why, why


def test_accepts_every_nominal_family():
    for fam in FAMILIES:
        ok, why = geometry_ok(L, W, _fam(fam))
        assert ok, "{} rejected: {}".format(fam, why)


def test_the_domain_cannot_be_disconnected_and_the_guard_that_fires_first():
    """Disconnection turns out to be UNREACHABLE through this parameterization, and it is worth
    recording why rather than leaving a guard whose trigger nobody can describe.

    `verts_for_notches` emits a single simple ring, and the interior of a simple polygon is
    connected by definition -- so no legal draw can split the domain. An extreme draw that would
    separate two corners (a deep, wide NW plus a deep, wide SE) does not produce a disconnected
    polygon; it produces a ring whose edges cross, and the CCW/shoelace assert rejects it FIRST.
    That assert is therefore the real guard, and `_connected` is belt-and-braces for a future
    parameterization (multiple disjoint blocks, interior pillars) that could express a split.

    Also worth recording per family: with notches only on the north edge (U, T), the full-width
    strip below the deepest notch joins everything regardless of how extreme the draw is.
    """
    from aaf.data.multi_notch import _connected
    for fam in ("rect", "L", "U", "T", "DN"):
        assert _connected(L, W, _fam(fam)), fam
    # the draw that would separate the bottom-left and top-right corners
    with pytest.raises(ValueError, match="clockwise"):
        verts_for_notches(L, W, [Notch("NW", W - 0.5, 5.5), Notch("SE", 4.5, L - 0.5)])


def test_self_intersection_check_flags_a_crossed_ring():
    from aaf.data.multi_notch import _self_intersects
    assert not _self_intersects(verts_for_notches(L, W, _fam("U")))
    assert _self_intersects([(0.0, 0.0), (6.0, 5.0), (6.0, 0.0), (0.0, 5.0)])


# ------------------------------------------------------- corpus plumbing and the token control
def test_family_filenames_cannot_alias_across_families():
    """A U and a T with the same (L, W, d, w) must NOT collide. `shape_filename`'s docstring
    warns that aliasing means the trainer fits one room to another's field with no error; having
    more than one notch is what makes that hazard reachable."""
    from aaf.data.multi_notch import FamilyConfig
    u = FamilyConfig(L, W, (Notch("NW", 1.2, 2.0), Notch("NE", 1.2, 2.0)))
    t = FamilyConfig(L, W, (Notch("midN", 1.2, 2.0, x0=2.0),))
    l = FamilyConfig(L, W, (Notch("NW", 1.2, 2.0),))
    names = {u.filename, t.filename, l.filename}
    assert len(names) == 3, names
    assert u.n_tokens == 8 and t.n_tokens == 8 and l.n_tokens == 6


def test_family_config_exposes_the_trainer_interface():
    """The trainer, the builder and the evaluators consume ShapeConfig and FamilyConfig without
    a branch, so every attribute they touch must be present with the same meaning."""
    from aaf.data.multi_notch import FamilyConfig
    c = FamilyConfig(L, W, (Notch("NW", 1.2, 1.6), Notch("NE", 0.9, 1.4)), shape_id=7)
    assert len(c.alphas) == 4                      # bbox view for the drift check
    assert len(c.edge_alphas) == len(c.verts)      # one per EDGE
    assert c.kind == "U" and c.strata.startswith("U_")
    assert 0.0 <= c.d_hat <= 1.0 and 0.0 <= c.w_hat <= 1.0
    assert c.filename.endswith(".h5") and c.label == c.filename[:-3]
    r = c.row(0)
    for k in ("L", "W", "verts", "edge_alphas", "alphas", "filename", "label", "strata",
              "kind", "n_tokens", "notches"):
        assert k in r, k


def test_trainer_schema_branch_exists_for_the_family_manifest():
    """Without it the rows fall through to the shoebox reader. For this schema that raises a bare
    KeyError rather than mis-parsing -- but the branch is still mandatory, and P4-2 lost a day to
    the version of this bug that failed silently."""
    src = Path(__file__).resolve().parents[1].joinpath(
        "aaf/train/multi_room_2d_mat.py").read_text()
    assert 'startswith("p4_4.family")' in src
    assert "from aaf.data.multi_notch import configs_from_rows" in src


@pytest.mark.parametrize("fam,k,want", [("rect", 2, 8), ("rect", 3, 12), ("L", 2, 12)])
def test_refine_verts_is_the_same_room_with_more_tokens(fam, k, want):
    """The token-count control. Splitting every edge into k collinear pieces leaves the room
    identical -- same membership everywhere -- while multiplying the token count, so any accuracy
    difference is attributable to token count alone and costs no new simulation.

    The (fam, k) pairs are exactly those that fit MAX_SEG_POLY = 12, which is what makes the
    rectangle the best control in the corpus: the SAME room can be described at 4, 8 and 12
    tokens, a three-point curve at literally fixed geometry.
    """
    from aaf.data.multi_notch import refine_verts
    v = verts_for_notches(L, W, _fam(fam))
    r = refine_verts(v, k)
    assert len(r) == want
    rng = np.random.default_rng(0)
    P = np.stack([rng.uniform(0, L, 4000), rng.uniform(0, W, 4000)], axis=-1)
    assert np.array_equal(polygon_contains(v, P), polygon_contains(r, P)), \
        "refined ring describes a different room"
    n = len(r)
    area2 = sum(r[i][0] * r[(i + 1) % n][1] - r[(i + 1) % n][0] * r[i][1] for i in range(n))
    assert area2 > 0, "refined ring came out clockwise"
    polygon_edge_tokens(r, [0.15] * n)             # raises on zero-length edges


def test_refined_rings_stay_inside_the_token_budget():
    """MAX_SEG_POLY = 12 bounds the control, and the bound is worth pinning because it decides
    which comparisons are even available: rect reaches 4/8/12, L reaches 6/12, and the 8-token
    families cannot be refined at all (8 x 2 = 16). So the token-count question is answered on
    rect and L, and the U/T/DN rows contribute the across-GEOMETRY comparison instead."""
    from aaf.data.multi_notch import refine_verts
    assert len(refine_verts(verts_for_notches(L, W, _fam("rect")), 3)) == MAX_SEG_POLY
    assert len(refine_verts(verts_for_notches(L, W, _fam("L")), 2)) == MAX_SEG_POLY
    with pytest.raises(ValueError, match="MAX_SEG_POLY"):
        polygon_edge_tokens(refine_verts(verts_for_notches(L, W, _fam("U")), 2), [0.15] * 16)
