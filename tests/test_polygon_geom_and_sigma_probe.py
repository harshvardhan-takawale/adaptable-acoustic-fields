"""Coverage for the two P4-2 libraries that had none: the polygon helpers and the sigma probe.

Both are load-bearing in a way that fails SILENTLY:

* `polygon_contains` decides which receivers exist, which are NLOS, and -- via the builder's
  node-for-node cross-check -- whether the solver simulated the room the tokens describe. A
  wrong answer produces a plausible dataset for a different room.
* `assert_extents_match_polygon` is D65's guard. Its FIRST version passed the exact case it
  exists to catch, because it compared reported extents against ANY polygon edge length and for
  a top-left notch the east wall genuinely IS the full bounding-box height. The per-wall tests
  below pin that down.
* `aaf/eval/sigma_probe` replaces an analysis that existed only as ad-hoc session code and one
  JSON blob -- the single number that says whether shape transfer can work. It is anchored here
  against P4-1's recorded values so a refactor cannot quietly change the answer.

The model-dependent path is exercised with a stub: constructing a real `INR2D_AutoDecoder`
imports tinycudann and needs a GPU, and these are properties of the probe's arithmetic, not of
any trained network.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from aaf.eval.sigma_probe import (
    _interpret, pick_nlos_probe_receiver, probe_sigma_occlusion, sigma_along_ray,
)
from aaf.sim.polygon_geom import (
    assert_extents_match_polygon, edge_lengths, line_of_sight, notch_solid_mask,
    polygon_contains, wall_extent,
)

# P4-1's L-room, the regression anchor for everything below.
L1, W1 = 6.0, 5.0
NOTCH_X, NOTCH_Y = 3.0, 2.5
VERTS_L = [(0.0, 0.0), (L1, 0.0), (L1, W1), (NOTCH_X, W1), (NOTCH_X, NOTCH_Y), (0.0, NOTCH_Y)]
SRC_L = (0.8, 1.5)
RECT = [(0.0, 0.0), (6.0, 0.0), (6.0, 5.0), (0.0, 5.0)]

P4_1_STAGE1 = Path("outputs/p4_1/stage1")


# ----------------------------------------------------------------------- point in polygon
def test_polygon_contains_matches_the_analytic_l_room():
    """Against the hand-written boolean P4-1 used, on a fine grid. The general helper must agree
    with the special case everywhere, or the P4-2 dataset is not comparable to P4-1's."""
    x = np.arange(0.0, L1 + 1e-9, 0.02)
    y = np.arange(0.0, W1 + 1e-9, 0.02)
    P = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
    want = ~((P[..., 0] < NOTCH_X) & (P[..., 1] > NOTCH_Y))
    assert int((polygon_contains(VERTS_L, P) != want).sum()) == 0


def test_points_on_an_edge_count_as_inside():
    """A receiver on a wall is a legitimate probe point. Excluding it would thin the receiver
    set asymmetrically -- only on the axis-aligned walls, which here is all of them."""
    on = np.array([[0.0, 1.0], [6.0, 1.0], [3.0, 5.0], [NOTCH_X, 4.0], [1.0, NOTCH_Y]])
    assert polygon_contains(VERTS_L, on).all()


def test_the_notch_interior_is_outside_and_the_reflex_corner_is_inside():
    assert not polygon_contains(VERTS_L, np.array([[1.0, 4.0]]))[0]
    assert polygon_contains(VERTS_L, np.array([[NOTCH_X, NOTCH_Y]]))[0]


def test_rectangle_has_no_hole():
    P = np.array([[0.5, 4.5], [1.0, 4.0], [5.9, 0.1]])
    assert polygon_contains(RECT, P).all()


# ------------------------------------------------------------------------------ line of sight
def test_line_of_sight_splits_the_l_room_the_way_the_geometry_demands():
    rx = np.array([[5.0, 1.0],        # same open half -- visible
                   [5.5, 3.0],        # high but far right: the sight line clears the corner
                   [4.0, 4.5],        # looks open, but the ray crosses the notch at x = 1.87
                   [3.05, 4.9]])      # just right of the notch face, high: shadowed
    los = line_of_sight(VERTS_L, SRC_L, rx)
    assert bool(los[0]) and bool(los[1])
    assert not bool(los[2]), "the shadow reaches further right than it looks -- that is the point"
    assert not bool(los[3])


def test_every_receiver_in_a_rectangle_is_los():
    """A convex room cannot occlude. If this ever fails the sampler is broken, not the room."""
    g = np.array([[x, y] for x in np.arange(0.2, 6.0, 0.4) for y in np.arange(0.2, 5.0, 0.4)])
    assert line_of_sight(RECT, (0.8, 1.5), g).all()


# ------------------------------------------------------------- per-wall extents (the D65 guard)
def test_wall_extent_is_per_wall_not_any_edge_length():
    """The bug the first version of the guard had. For a top-left notch EAST really is the full
    5.0 m, so an "is 5.0 a real edge length?" test passes a west wall wrongly reported as 5.0."""
    assert wall_extent(VERTS_L, "east") == pytest.approx(W1)
    assert wall_extent(VERTS_L, "south") == pytest.approx(L1)
    assert wall_extent(VERTS_L, "west") == pytest.approx(NOTCH_Y)       # truncated: 2.5, not 5.0
    assert wall_extent(VERTS_L, "north") == pytest.approx(L1 - NOTCH_X)  # truncated: 3.0, not 6.0
    assert 5.0 in [pytest.approx(e) for e in edge_lengths(VERTS_L)], (
        "the premise of this test: 5.0 IS a genuine edge length here")


def test_extent_guard_accepts_the_truthful_spec():
    assert_extents_match_polygon(
        [{"type": "wall_segments", "wall": "west", "extents_m": [1.0, 1.5]},
         {"type": "wall_segments", "wall": "north", "extents_m": [3.0]}], VERTS_L)


@pytest.mark.parametrize("wall,extents,truth", [("west", [5.0], 2.5), ("north", [6.0], 3.0)])
def test_extent_guard_fires_on_the_bounding_rectangle(wall, extents, truth):
    """Exactly what `_apply_wall_segments` reports on a notched room: the bounding box's wall,
    with `tiles_exactly: True`, straight into the HDF5 attrs."""
    with pytest.raises(AssertionError, match="D65"):
        assert_extents_match_polygon(
            [{"type": "wall_segments", "wall": wall, "extents_m": extents,
              "tiles_exactly": True}], VERTS_L)
    assert wall_extent(VERTS_L, wall) == pytest.approx(truth)


def test_extent_guard_fires_on_a_single_oversized_segment():
    with pytest.raises(AssertionError, match="longer than the whole wall"):
        assert_extents_match_polygon(
            [{"wall": "west", "extents_m": [4.0, -1.5]}], VERTS_L)


def test_extent_guard_is_a_no_op_on_a_rectangle():
    """A shoebox is the case where `wall_segments` is CORRECT; the guard must not break it."""
    assert_extents_match_polygon([{"wall": "west", "extents_m": [5.0]},
                                  {"wall": "north", "extents_m": [2.0, 4.0]}], RECT)


# ------------------------------------------------------------------------------- the notch mask
def test_notch_mask_leaves_both_faces_as_air():
    """The faces ARE the walls. Off by one and they land a cell from where the polygon says."""
    dx = 0.02
    nx, ny = int(round(L1 / dx)) + 1, int(round(W1 / dx)) + 1
    m = notch_solid_mask(nx, ny, dx, dx, W1, W1 - NOTCH_Y, NOTCH_X)
    i_w, j_d = int(round(NOTCH_X / dx)), int(round(NOTCH_Y / dx))
    assert not m[i_w, j_d + 5], "x = w must stay air -- it is the vertical notch face"
    assert not m[5, j_d], "y = W - d must stay air -- it is the horizontal notch face"
    assert m[i_w - 1, j_d + 1]


def test_notch_mask_is_empty_when_either_dimension_is_zero():
    for d, w in ((0.0, 2.0), (2.0, 0.0), (0.0, 0.0)):
        assert not notch_solid_mask(301, 251, 0.02, 0.02, W1, d, w).any()


# ------------------------------------------------------ the sigma probe, against a stub model
class _StubModel:
    """Returns sigma as a caller-supplied function of position, so the probe's arithmetic can be
    checked against a field whose answer is known in advance."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, P, V, TX, tx_view=None, z_s=None):
        s = self.fn(P)                               # [1, n]
        attn = torch.complex(s, torch.zeros_like(s)).unsqueeze(-1).expand(-1, -1, 4)
        return attn, torch.zeros_like(attn)


def _probe(fn, **kw):
    return probe_sigma_occlusion(_StubModel(fn), VERTS_L, SRC_L, torch.device("cpu"),
                                 cond_source="geom_token", n=400, **kw)


def test_probe_reports_ratio_one_for_a_uniform_field():
    """The negative result the diagnostic exists to detect: a model whose sigma carries no
    geometry at all must read ~1.0, not something that can be mistaken for a weak occluder."""
    out = _probe(lambda P: torch.full(P.shape[:2], 0.7))
    assert out["probed"]
    assert out["solid_over_air"] == pytest.approx(1.0, abs=1e-6)
    assert out["transmittance_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert "NOT encoding the occluder" in out["interpretation"]


def test_probe_detects_a_genuine_occluder():
    """A field that really is opaque inside the notch must come back as one."""
    def fn(P):
        x, y = P[..., 0], P[..., 1]
        inside_notch = (x < NOTCH_X) & (y > NOTCH_Y)
        return torch.where(inside_notch, torch.full_like(x, 5.0), torch.full_like(x, 0.05))
    out = _probe(fn)
    assert out["solid_over_air"] > 10.0
    assert out["transmittance_ratio"] < 1e-3
    assert "genuine learned occluder" in out["interpretation"]


def test_probe_picks_the_most_occluded_receiver_and_derives_reach_from_the_bbox():
    """With no receiver set the probe builds its own grid, so it cannot land on P4-1's exact
    point (that anchor is the regression test below, which feeds P4-1's own receivers). What
    must hold either way: it picks a receiver deep in the shadow, and `reach` comes from the
    bounding-box diagonal rather than P4-1's hard-wired 6.5."""
    out = _probe(lambda P: torch.full(P.shape[:2], 0.7))
    x, y = out["rx"]
    assert x <= NOTCH_X + 0.2 and y >= 4.5, "probe receiver is not in the deep shadow"
    assert out["n_samples"] == 400 and out["n_samples_in_solid"] > 100
    assert max(out["t"]) == pytest.approx(np.hypot(L1, W1))


def test_probe_refuses_when_no_receiver_is_nlos():
    """On a rectangle the probe is UNDEFINED, not negative -- reporting ratio 1.0 there would be
    a false negative indistinguishable from a real one."""
    out = probe_sigma_occlusion(_StubModel(lambda P: torch.full(P.shape[:2], 0.7)),
                                RECT, (0.8, 1.5), torch.device("cpu"), cond_source="geom_token")
    assert out["probed"] is False and "no NLOS receiver" in out["reason"]


def test_probe_rejects_a_world_scale_mismatch():
    """Tokens divide positions by WORLD_SCALE; the model applies its own. If they disagree the
    probe samples the field somewhere other than where it reports."""
    with pytest.raises(ValueError, match="world_scale"):
        _probe(lambda P: torch.full(P.shape[:2], 0.7), world_scale=1.0)


def test_sigma_along_ray_needs_reach_or_verts():
    with pytest.raises(ValueError, match="reach or verts"):
        sigma_along_ray(_StubModel(lambda P: torch.zeros(P.shape[:2])), torch.zeros(4),
                        (1.0, 1.0), (1.0, 0.0), SRC_L, torch.device("cpu"))


@pytest.mark.parametrize("ratio,ts,ta,want", [
    (1.00, 0.08, 0.08, "NOT encoding"),
    (1.08, 0.068, 0.082, "PHYSICALLY NEGLIGIBLE"),   # P4-1's regime, verbatim
    (9.00, 1e-4, 0.08, "genuine learned occluder"),
])
def test_interpretation_bands(ratio, ts, ta, want):
    assert want in _interpret(ratio, ts, ta)


# ------------------------------------------------- regression: P4-1's numbers, recomputed
@pytest.mark.skipif(not (P4_1_STAGE1 / "stage1_fields.npz").exists(),
                    reason="P4-1 stage 1 artifacts not present")
def test_reproduces_p4_1_recorded_sigma_statistics():
    """The analysis this module replaces existed only as session code plus one JSON blob. Given
    P4-1's own stored ray, the library must land on P4-1's own numbers."""
    z = np.load(P4_1_STAGE1 / "stage1_fields.npz")
    rec = json.load(open(P4_1_STAGE1 / "GATE1.json"))["sigma_probe"]
    solid = ~polygon_contains(VERTS_L, z["sigma_pts"])
    a, b = z["sigma"][solid], z["sigma"][~solid]
    assert int(solid.sum()) == rec["n_samples_in_solid"] == 226
    assert float(a.mean()) == pytest.approx(rec["mean_sigma_solid"], rel=1e-9)
    assert float(b.mean()) == pytest.approx(rec["mean_sigma_air"], rel=1e-9)
    assert float(a.mean() / b.mean()) == pytest.approx(rec["solid_over_air"], rel=1e-6)
    # and the physical reading that made Gate 1 a capacity result rather than a mechanism one
    cross_m = float(solid.sum() * float(z["sigma_t"][1] - z["sigma_t"][0]))
    assert float(np.exp(-a.mean() * cross_m)) == pytest.approx(0.068, abs=5e-4)
    assert float(np.exp(-b.mean() * cross_m)) == pytest.approx(0.082, abs=5e-4)
    assert list(z["sigma_rx"]) == pytest.approx([3.12, 4.82])
    assert pick_nlos_probe_receiver(VERTS_L, SRC_L, z["rx"]) == pytest.approx([3.12, 4.82])
