"""A0a/A0b guards: exact per-axis grid fitting, anisotropic correctness, patch extent.

These cover the two FT-A blockers whose failure mode is *silence*:

* **B1** — every FT-A gate ran L=4.5, W=4.0, the only geometry in the project that is an
  integer multiple of dx=0.05. The other 48 were snapped with a `warnings.warn`, a 0.5-0.8%
  dimension error and a proportional modal-frequency shift, ~30x the tolerance the solver was
  validated to. Exact fitting removes the error; these tests assert it over the *real*
  geometry family rather than the one convenient room.
* **B2** — the absorber patch absorbed over an extent one full cell wider than requested and
  reported the nominal span, so FT-C's independent variable was wrong at source.

The anisotropic test exists because undoing the isotropic collapse in `build_coefficients`
touches the boundary term `B`. A single `lam * sum(faces)` is correct only when dx_x == dx_y;
with unequal spacings it mis-scales absorption per direction, which yields a stable solver
with plausible output and the wrong wall physics -- exactly the class of error three
adversarial reviewers were asked to hunt for.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import yaml

import aaf.sim.fdtd_2d as F

C = 343.0
FS = 12288.0
DX = 0.05

TRAIN_YAML = "configs/sweeps_2d_mat/p3_2_train.yaml"
TEST_YAML = "configs/sweeps_2d_mat/p3_2_test_frozen.yaml"


def _all_geometries():
    out = []
    for path in (TRAIN_YAML, TEST_YAML):
        for g in yaml.safe_load(open(path))["geometries"]:
            out.append((round(float(g["L"]), 2), round(float(g["W"]), 2)))
    return out


def test_every_project_geometry_fits_exactly():
    """The blocker, stated as a test: 48 of 50 rooms used to be silently resized."""
    geoms = _all_geometries()
    assert len(geoms) == 50
    off_grid = [(L, W) for L, W in geoms
                if abs(round(L / DX) - L / DX) > 1e-9 or abs(round(W / DX) - W / DX) > 1e-9]
    assert len(off_grid) >= 45, (
        "this test is only meaningful because most rooms are off the nominal grid; "
        "got {}".format(len(off_grid)))

    for L, W in geoms:
        geom = F.build_geometry(L, W, (0.15,) * 4, dx=DX)
        assert (geom.nx - 1) * geom.dx_x == pytest.approx(L, abs=1e-12)
        assert (geom.ny - 1) * geom.dx_y == pytest.approx(W, abs=1e-12)
        assert geom.L_grid == pytest.approx(L, abs=1e-12)
        assert geom.W_grid == pytest.approx(W, abs=1e-12)


def test_realized_spacing_stays_within_one_percent():
    """Exact fitting must not drift dx far, or the dispersion characterization moves."""
    spacings = []
    for L, W in _all_geometries():
        geom = F.build_geometry(L, W, (0.15,) * 4, dx=DX)
        spacings += [geom.dx_x, geom.dx_y]
    assert max(abs(d - DX) for d in spacings) / DX < 0.01


def test_anisotropic_cfl_holds_on_every_geometry_at_frozen_fs():
    """dt stays frozen at 1/fs, so the anisotropic bound must hold unaided.

    The isotropic bound `lam <= 1/sqrt(2)` is a special case of `lam_x^2 + lam_y^2 <= 1` and
    is not valid once the spacings differ.
    """
    dt = 1.0 / FS
    worst = 0.0
    for L, W in _all_geometries():
        geom = F.build_geometry(L, W, (0.15,) * 4, dx=DX)
        cfl = math.hypot(C * dt / geom.dx_x, C * dt / geom.dx_y)
        worst = max(worst, cfl)
        assert cfl < 1.0
    assert worst < 0.85, "margin shrank; a per-room dt may now be unavoidable"


def test_off_grid_dimension_raises_when_fitting_is_bypassed():
    with pytest.raises(ValueError):
        F._fit_axis(0.03, DX, "L")          # < 2 cells
    n, dxa = F._fit_axis(3.68, DX, "L")
    assert (n - 1) * dxa == pytest.approx(3.68, abs=1e-12)
    assert n - 1 == round(3.68 / DX)


# ------------------------------------------------------------------ anisotropy
def _uniform_alpha_geom(L, W, alpha, dx):
    return F.build_geometry(L, W, (alpha,) * 4, dx=dx)


def test_anisotropic_coefficients_reduce_to_isotropic():
    """With dx_x == dx_y the per-axis form must reproduce the old numbers exactly."""
    geom = _uniform_alpha_geom(1.0, 0.8, 0.3, 0.05)      # both axes fit exactly at 0.05
    assert geom.dx_x == pytest.approx(geom.dx_y, abs=1e-15)
    lam = C / FS / geom.dx_x
    aniso = F.build_coefficients(geom, lam, lam)
    for k in ("cE", "cW", "cN", "cS", "cC", "cP", "B"):
        assert np.all(np.isfinite(aniso[k]))
    # cC must be 2*(1 - lam_x^2 - lam_y^2)/den, i.e. the old 2*(1 - 2 lam^2)/den here.
    den = 1.0 + aniso["B"]
    expect_cC = np.full(aniso["cC"].shape, 2.0 * (1.0 - 2.0 * lam * lam)) / den
    expect_cC = expect_cC * geom.air.astype(float)
    assert np.allclose(aniso["cC"], expect_cC, atol=1e-15)


def test_boundary_term_weights_each_direction_by_its_own_lambda():
    """The subtlest edit in A0a, asserted directly.

    `B` must be lam_x*(XM+XP admittance) + lam_y*(YM+YP admittance). Summing all four faces
    against one lambda is stable and looks fine, but scales x-wall and y-wall absorption by
    the wrong factors as soon as the spacings differ.
    """
    geom = _uniform_alpha_geom(1.0, 0.8, 0.4, 0.05)
    lam_x, lam_y = 0.40, 0.55                     # deliberately unequal
    coef = F.build_coefficients(geom, lam_x, lam_y)

    adm_blocked = np.where(geom.blocked, geom.adm, 0.0)
    expect = (lam_x * (adm_blocked[F.XM] + adm_blocked[F.XP])
              + lam_y * (adm_blocked[F.YM] + adm_blocked[F.YP])) * geom.air.astype(float)
    assert np.allclose(coef["B"], expect, atol=1e-15)

    # And it must NOT equal the naive single-lambda sum anywhere a wall is present.
    naive = 0.5 * (lam_x + lam_y) * adm_blocked.sum(axis=0) * geom.air.astype(float)
    walls = coef["B"] > 0
    assert walls.any()
    assert not np.allclose(coef["B"][walls], naive[walls], atol=1e-12)

    # A pure x-wall node scales with lam_x alone; a pure y-wall node with lam_y alone.
    xi = F.wall_admittance(0.4)
    mid_j = geom.ny // 2
    assert coef["B"][0, mid_j] == pytest.approx(lam_x * xi, rel=1e-12)
    mid_i = geom.nx // 2
    assert coef["B"][mid_i, 0] == pytest.approx(lam_y * xi, rel=1e-12)


def test_anisotropic_run_is_stable_and_energy_bounded():
    """A room whose axes fit to genuinely different spacings must still conserve energy."""
    L, W = 3.68, 4.03                                    # both off the nominal grid
    geom = F.build_geometry(L, W, (0.0,) * 4, dx=DX)
    assert abs(geom.dx_x - geom.dx_y) > 1e-6, "pick a room with distinct realized spacings"
    out = F.simulate(L, W, (0.0,) * 4, src=(0.9, 1.1), rx=[(2.1, 2.6)],
                     dx=DX, fs=FS, n=4096, record_energy=True)
    e = out["energy"]
    assert np.all(np.isfinite(e))
    # Energy must never exceed its post-injection peak, and must trend down. Asserting
    # monotone non-increase sample-by-sample would test the SOURCE, not the scheme: the
    # default pulse is designed in the rfft domain and is therefore periodic, so its pre-ring
    # wraps into the tail and lifts it ~0.02% (documented in FT-A gate A0). The scheme's
    # stability claim is the absence of growth, which is what is checked here.
    tail = e[len(e) // 4:]
    assert tail.max() <= e.max() * (1.0 + 1e-12)
    half = e[len(e) // 2:]
    slope = np.polyfit(np.arange(half.size, dtype=float), half, 1)[0]
    assert slope <= 0.0
    m = out["meta"]
    assert m["dx_x"] != m["dx_y"]
    assert m["lambda_CFL_aniso"] < 1.0


# ------------------------------------------------------------------ patch extent
@pytest.mark.parametrize("span,expect_m", [
    ((0.20, 0.60), 0.40),
    ((0.10, 0.30), 0.20),
    # Corner-touching: node 0 owns dx/2, so achievable extents are (n - 0.5)*dx and 0.25 is
    # NOT one of them (n = 5.5). 0.225 is the nearest achievable. Physical quantization of a
    # node-centred wall, not an error -- FT-C keeps its patches interior for this reason.
    ((0.00, 0.25), 0.225),
    ((0.35, 0.90), 0.55),
])
def test_patch_realized_extent_matches_request(span, expect_m):
    patch = {"type": "patch", "wall": "north", "span": span, "alpha": 0.8}
    geom = F.build_geometry(1.0, 0.8, (0.1,) * 4, dx=0.05, extra_walls=[patch])
    spec = geom.specs[0]
    assert spec["width_realized_m"] == pytest.approx(expect_m, abs=0.05 / 10.0)
    assert spec["width_requested_m"] == pytest.approx(abs(span[1] - span[0]))


def test_patch_extent_is_not_one_cell_too_wide():
    """The regression itself: the old rule realized request + dx for every interior patch."""
    dx = 0.05
    for a in (0.20, 0.40, 0.60):
        lo = 0.20
        patch = {"type": "patch", "wall": "north", "span": (lo, lo + a), "alpha": 0.8}
        geom = F.build_geometry(1.6, 0.8, (0.1,) * 4, dx=dx, extra_walls=[patch])
        spec = geom.specs[0]
        assert spec["width_realized_m"] == pytest.approx(a, abs=dx / 10.0)
        assert spec["width_realized_m"] != pytest.approx(a + dx, abs=1e-9)


def test_whole_wall_patch_measures_the_whole_wall():
    """The corner half-strips must be accounted for, or the reference case is wrong."""
    geom = F.build_geometry(1.0, 0.8, (0.1,) * 4, dx=0.05,
                            extra_walls=[{"type": "patch", "wall": "north", "alpha": 0.8}])
    spec = geom.specs[0]
    assert spec["n_nodes"] == geom.nx
    assert spec["width_realized_m"] == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(geom.face_alpha[F.YP, :, -1], 0.8)


def test_wall_node_extent_corner_weighting():
    dx = 0.05
    n = 21
    assert F.wall_node_extent(0, n - 1, dx, n) == pytest.approx((n - 1) * dx)
    assert F.wall_node_extent(5, 12, dx, n) == pytest.approx(8 * dx)
    assert F.wall_node_extent(0, 4, dx, n) == pytest.approx(4.5 * dx)
