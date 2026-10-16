"""P4-7 prep p4 (D164): the D4 / reciprocity geometry of `aaf/data/augment_2d.py`, tested on BEHAVIOUR.

The decisive test is the manager's: a 90 deg rotation of a room maps its FDTD field exactly -- here for all
eight D4 elements of an L-room with NON-uniform wall alphas and a notch of its own alpha, built only from
the augmentation functions. Each check has teeth: the same comparison with the wall alphas left
unpermuted, or with the receivers mapped by the wrong element, must fail. CPU only (no tinycudann).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import aaf.sim.fdtd_2d as F
from aaf.data.augment_2d import (D4_ELEMS, TRANSPOSING, canonical_ring, d4_det, d4_dims, d4_inverse,
                                 d4_points, d4_polygon, d4_vectors, d4_wall_alphas, is_interior_node,
                                 transpose_holdout_leaks)
from aaf.models.conditioning_2d import polygon_edge_tokens
from aaf.sim.polygon_geom import polygon_contains

DX, FS, N = 0.05, 12288.0, 1024
L0, W0 = 2.0, 1.5
ROOM = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.5), (0.75, 1.5), (0.75, 1.0), (0.0, 1.0)]   # L, NW notch
WALLS = [0.10, 0.20, 0.30, 0.05]                                                     # W, E, S, N
MASK_ALPHA = 0.25
SRC = (0.4, 0.35)


def _nodes(L, W):
    nx, ny = int(round(L / DX)) + 1, int(round(W / DX)) + 1
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    return nx, ny, np.stack([ii * DX, jj * DX], -1)


def _sim(L, W, verts, walls, src, rx):
    nx, ny, P = _nodes(L, W)
    solid = ~polygon_contains(verts, P.reshape(-1, 2)).reshape(nx, ny)
    extra = [{"type": "mask", "solid": solid, "alpha": MASK_ALPHA}] if solid.any() else None
    return F.simulate(L, W, walls, src, rx, dx=DX, fs=FS, n=N, extra_walls=extra)["ir"]


def _air(verts, L, W):
    nx, ny, P = _nodes(L, W)
    P = P.reshape(-1, 2)
    return P[polygon_contains(verts, P)]


@pytest.fixture(scope="module")
def base():
    rx = _air(ROOM, L0, W0)
    return rx, _sim(L0, W0, ROOM, WALLS, SRC, rx)


@pytest.mark.parametrize("g", D4_ELEMS)
def test_d4_maps_the_fdtd_field_exactly(base, g):
    rx, ir0 = base
    Lg, Wg = d4_dims(g, L0, W0)
    vg, _ag, _ = d4_polygon(g, ROOM, [0.15] * len(ROOM), L0, W0)
    ir = _sim(Lg, Wg, vg, d4_wall_alphas(g, WALLS), tuple(d4_points(g, SRC, L0, W0)), d4_points(g, rx, L0, W0))
    if g in ("e", "fx"):
        assert np.array_equal(ir, ir0), g
    else:
        assert np.abs(ir - ir0).max() <= 1e-11 * np.abs(ir0).max(), (g, np.abs(ir - ir0).max() / np.abs(ir0).max())


def test_teeth_unpermuted_walls_or_wrong_receiver_map_fail(base):
    rx, ir0 = base
    g = "r90"
    Lg, Wg = d4_dims(g, L0, W0)
    vg, _ag, _ = d4_polygon(g, ROOM, [0.15] * len(ROOM), L0, W0)
    src = tuple(d4_points(g, SRC, L0, W0))
    bad_walls = _sim(Lg, Wg, vg, WALLS, src, d4_points(g, rx, L0, W0))
    assert np.abs(bad_walls - ir0).max() > 1e-3 * np.abs(ir0).max()
    # receivers mapped by r270 (the inverse) instead of r90: some land in solid nodes or elsewhere
    rx_wrong = d4_points("r270", rx, L0, W0)
    ok = polygon_contains(vg, rx_wrong) & (rx_wrong[:, 0] <= Lg + 1e-9) & (rx_wrong[:, 1] <= Wg + 1e-9)
    ir_wrong = _sim(Lg, Wg, vg, d4_wall_alphas(g, WALLS), src, rx_wrong[ok])
    assert np.abs(ir_wrong - ir0[ok]).max() > 1e-3 * np.abs(ir0).max()


def test_identity_is_exact_and_native_rings_are_canonical():
    from aaf.data.multi_notch import configs_from_rows as fam
    from aaf.data.shape_configs import configs_from_rows as shp
    for path, reader in (("configs/sweeps_2d_mat/p4_4_shapes_500.json", shp),
                         ("configs/sweeps_2d_mat/p4_4_family_manifest.json", fam),
                         ("configs/sweeps_2d_mat/p4_6_family_holdU_manifest.json", fam)):
        for c in reader(json.load(open(path))["configs"], split="train"):
            vin, ain = c.verts, c.edge_alphas            # properties: capture once
            v, a, perm = d4_polygon("e", vin, ain, c.L, c.W)
            assert v is vin and a is ain and perm == list(range(len(vin)))
            cv, ca = canonical_ring(c.verts, c.edge_alphas)
            assert cv == [tuple(map(float, p)) for p in c.verts] and ca == [float(x) for x in c.edge_alphas]
            for g in D4_ELEMS:                                    # every image is a valid ring
                vg, ag, _ = d4_polygon(g, c.verts, c.edge_alphas, c.L, c.W)
                polygon_edge_tokens(vg, ag)
                Lg, Wg = d4_dims(g, c.L, c.W)
                xs, ys = [p[0] for p in vg], [p[1] for p in vg]
                assert min(xs) == 0.0 and min(ys) == 0.0
                assert abs(max(xs) - Lg) < 1e-9 and abs(max(ys) - Wg) < 1e-9


def _tok(v, a):
    return np.array(polygon_edge_tokens(v, a))


@pytest.mark.parametrize("g", D4_ELEMS)
def test_tokens_transform_as_a_set(g):
    al = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    t0 = _tok(ROOM, al)
    vg, ag, _ = d4_polygon(g, ROOM, al, L0, W0)
    tg = _tok(vg, ag)
    c = d4_points(g, t0[:, :2] * 10.0, L0, W0) / 10.0
    n = d4_vectors(g, t0[:, 2:4])
    want = np.concatenate([c, n, t0[:, 4:]], 1)
    key = lambda t: sorted(map(tuple, np.round(t, 9)))  # noqa: E731
    assert key(tg) == key(want)


def test_canonical_form_depends_only_on_the_polygon():
    sq = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    imgs = {tuple(map(tuple, _tok(*d4_polygon(g, sq, [0.15] * 4, 2.0, 2.0)[:2]))) for g in D4_ELEMS}
    assert len(imgs) == 1
    Lr = [tuple(map(tuple, _tok(*d4_polygon(g, ROOM, [0.15] * 6, L0, W0)[:2]))) for g in D4_ELEMS]
    assert len(set(Lr)) == len(D4_ELEMS)                          # teeth: a non-square L's images all differ


def test_flips_of_native_rooms_are_native_rooms():
    from aaf.data.multi_notch import FamilyConfig, Notch
    L, W = 6.0, 5.0

    def ring(*notches):
        c = FamilyConfig(L=L, W=W, notches=tuple(notches))
        return canonical_ring(c.verts, c.edge_alphas)[0], c

    def img(g, c):
        return d4_polygon(g, c.verts, c.edge_alphas, L, W)[0]

    nw, cnw = ring(Notch("NW", 1.2, 2.0))
    assert img("fx", cnw) == ring(Notch("NE", 1.2, 2.0))[0]
    assert img("r180", cnw) == ring(Notch("SE", 1.2, 2.0))[0]
    _u, cu = ring(Notch("NW", 1.0, 1.5), Notch("NE", 0.6, 0.9))
    assert img("fx", cu) == ring(Notch("NW", 0.6, 0.9), Notch("NE", 1.0, 1.5))[0]
    _t, ct = ring(Notch("midN", 0.8, 1.0, x0=1.5))
    assert img("fx", ct) == ring(Notch("midN", 0.8, 1.0, x0=L - 1.5 - 1.0))[0]


def test_group_laws_and_wall_table():
    P = np.array([[0.25, 0.5], [1.75, 1.25], [0.0, 0.0], [2.0, 1.5]])
    for g in D4_ELEMS:
        Lg, Wg = d4_dims(g, L0, W0)
        back = d4_points(d4_inverse(g), d4_points(g, P, L0, W0), Lg, Wg)
        assert np.array_equal(back, P), g
        assert d4_det(g) == round(np.linalg.det(d4_vectors(g, np.eye(2))))
        # the wall table: the image of each wall's midpoint lies on the claimed image wall
        mids = {0: (0.0, W0 / 2), 1: (L0, W0 / 2), 2: (L0 / 2, 0.0), 3: (L0 / 2, W0)}
        src_of = d4_wall_alphas(g, [0, 1, 2, 3])
        for new_wall, old_wall in enumerate(src_of):
            x, y = d4_points(g, mids[old_wall], L0, W0)
            on = {0: abs(x) < 1e-12, 1: abs(x - Lg) < 1e-12, 2: abs(y) < 1e-12, 3: abs(y - Wg) < 1e-12}[new_wall]
            assert on, (g, new_wall, old_wall)


def test_transpose_leak_guard_counts():
    from aaf.data.multi_notch import configs_from_rows as fam
    from aaf.data.shape_configs import configs_from_rows as shp
    c500 = shp(json.load(open("configs/sweeps_2d_mat/p4_4_shapes_500.json"))["configs"], split="train")
    hold = fam(json.load(open("configs/sweeps_2d_mat/p4_6_family_holdU_manifest.json"))["configs"], split="train")
    assert len(transpose_holdout_leaks(c500, (0.45, 0.60))) == 88
    assert len(transpose_holdout_leaks(hold, (0.45, 0.60))) == 43
    for g in D4_ELEMS:                                           # flips keep (L, W): depths stay depths
        assert (d4_dims(g, 6.0, 5.0) == (5.0, 6.0)) == (g in TRANSPOSING)


def _swap_err(verts, a, b, band_hz=300.0):
    i1 = _sim(L0, W0, verts, WALLS, a, np.array([b]))[0]
    i2 = _sim(L0, W0, verts, WALLS, b, np.array([a]))[0]
    H1, H2 = np.fft.rfft(i1), np.fft.rfft(i2)
    k = int(band_hz / (FS / N)) + 1
    return np.abs(i1 - i2).max() / np.abs(i1).max(), np.linalg.norm(H1[1:k] - H2[1:k]) / np.linalg.norm(H1[1:k])


def test_reciprocity_exact_in_a_rectangle_bounded_in_a_notched_room():
    rect = [(0.0, 0.0), (L0, 0.0), (L0, W0), (0.0, W0)]
    a, b = (0.4, 0.35), (1.55, 1.1)
    td, _ = _swap_err(rect, a, b)
    assert td <= 1e-11, td
    _, fd = _swap_err(ROOM, a, b)
    assert 1e-6 < fd < 0.05, fd                                   # pins the D65 wall-tip residual


def test_wall_nodes_are_flagged_and_break_reciprocity():
    rect = [(0.0, 0.0), (L0, 0.0), (L0, W0), (0.0, W0)]
    a, w = (0.4, 0.35), (0.0, 0.75)
    flags = is_interior_node(rect, np.array([a, w]), DX, DX)
    assert flags.tolist() == [True, False]
    i1 = _sim(L0, W0, rect, WALLS, a, np.array([w]))[0]
    i2 = _sim(L0, W0, rect, WALLS, w, np.array([a]))[0]
    H1, H2 = np.fft.rfft(i1), np.fft.rfft(i2)
    k = int(300.0 / (FS / N)) + 1
    ratio = np.median(np.abs(H1[1:k]) / np.abs(H2[1:k]))
    assert abs(ratio - 2.0) < 0.05 or abs(ratio - 0.5) < 0.0125, ratio
