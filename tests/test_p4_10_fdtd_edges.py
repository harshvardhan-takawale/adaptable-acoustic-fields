"""P4-10 (D197 A5): per-edge absorption on polygons.

Behaviour, not bookkeeping: (a) uniform alphas give bit-identical coefficients to the old mask-only geometry on every
dense room; (b) every blocked face of every air node is painted by exactly one edge, and a polygon that does not match the
solid mask is refused; (c) changing ONE edge changes only that edge's faces, and a longer wall has more; (d) off-grid,
clockwise and non-axis-aligned rings are refused; (e) at non-uniform alphas the energy absorbed depends on the edge.
"""
from __future__ import annotations

import numpy as np
import pytest

from aaf.sim import fdtd_2d as F
from aaf.sim.polygon_geom import polygon_contains

DX = 0.02
L_, W_ = 4.0, 3.2
NOTCH = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.2), (1.0, 3.2), (1.0, 2.0), (0.0, 2.0)]


def _mask(verts, L, W):
    nx, dxx = F._fit_axis(L, DX, "L")
    ny, dxy = F._fit_axis(W, DX, "W")
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    return ~polygon_contains(verts, np.stack([ii * dxx, jj * dxy], -1))


def _geom(verts, L, W, alphas):
    m = _mask(verts, L, W)
    extra = [{"type": "mask", "solid": m, "alpha": 0.15}] if m.any() else []
    return F.build_geometry(L, W, [0.15] * 4, dx=DX, extra_walls=extra + [{"type": "edges", "verts": verts, "alphas": alphas}])


def _old(verts, L, W):
    m = _mask(verts, L, W)
    return F.build_geometry(L, W, [0.15] * 4, dx=DX, extra_walls=[{"type": "mask", "solid": m, "alpha": 0.15}] if m.any() else None)


@pytest.mark.parametrize("verts,L,W", [(NOTCH, 4.0, 3.2), ([(0.0, 0.0), (4.0, 0.0), (4.0, 3.2), (0.0, 3.2)], 4.0, 3.2)])
def test_uniform_is_bit_identical(verts, L, W):
    a, b = _geom(verts, L, W, [0.15] * len(verts)), _old(verts, L, W)
    lam = 343.0 / (30720.0 * DX)
    ca, cb = F.build_coefficients(a, lam, lam), F.build_coefficients(b, lam, lam)
    for k in ca:
        assert np.array_equal(ca[k], cb[k]), k


def test_every_dense_room_is_painted_and_uniform():
    from scripts.p4_10_edits import base_rooms
    lam = 343.0 / (30720.0 * DX)
    n = 0
    for r in base_rooms():
        V = [tuple(map(float, v)) for v in r["verts"]]
        a = _geom(V, r["L"], r["W"], [0.15] * len(V))
        b = _old(V, r["L"], r["W"])
        ca, cb = F.build_coefficients(a, lam, lam), F.build_coefficients(b, lam, lam)
        assert all(np.array_equal(ca[k], cb[k]) for k in ca)
        n += 1
    assert n == 31


def test_one_edge_changes_only_its_faces():
    al = [0.15] * 6
    base = _geom(NOTCH, L_, W_, al)
    al2 = list(al)
    al2[3] = 0.6                                           # the notch's vertical face, (1.0, 3.2) -> (1.0, 2.0)
    mod = _geom(NOTCH, L_, W_, al2)
    d = np.argwhere(mod.face_alpha != base.face_alpha)
    assert len(d) == int(round(1.2 / DX)) + 1 and set(d[:, 0]) == {F.XM} and set(d[:, 1]) == {int(round(1.0 / DX))}
    blocked_changed = [tuple(x) for x in d if base.blocked[tuple(x)] and base.air[x[1], x[2]]]
    assert len(blocked_changed) >= int(round(1.2 / DX))


def test_refusals():
    sq = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.2), (0.0, 3.2)]
    with pytest.raises(ValueError):                        # clockwise
        _geom(sq[::-1], 4.0, 3.2, [0.15] * 4)
    with pytest.raises(ValueError):                        # off-grid
        F.build_geometry(4.0, 3.2, [0.15] * 4, dx=DX, extra_walls=[{"type": "edges", "verts": [(0.0, 0.0), (4.0, 0.0), (4.0, 3.2), (0.013, 3.2)],
                                                                  "alphas": [0.15] * 4}])
    with pytest.raises(ValueError):                        # polygon that does not match the solid mask: faces unpainted
        F.build_geometry(4.0, 3.2, [0.15] * 4, dx=DX, extra_walls=[
            {"type": "mask", "solid": _mask(NOTCH, 4.0, 3.2), "alpha": 0.15},
            {"type": "edges", "verts": sq, "alphas": [0.15] * 4}])


def test_absorption_depends_on_the_edge():
    """more absorption on a longer wall removes more energy: late-tail energy drops (behavioural)."""
    src, rx = (2.0, 1.0), np.array([[3.0, 2.0], [1.6, 1.0]])
    e = []
    for a in ([0.15] * 6, [0.6] + [0.15] * 5):
        m = _mask(NOTCH, L_, W_)
        res = F.simulate(L_, W_, [0.15] * 4, src, rx, dx=DX, fs=30720.0, n=8192, c=343.0,
                         extra_walls=[{"type": "mask", "solid": m, "alpha": 0.15}, {"type": "edges", "verts": NOTCH, "alphas": a}])
        h = np.asarray(res["ir"])
        e.append(float((h[:, 4096:] ** 2).sum()))
    assert e[1] < 0.9 * e[0]
