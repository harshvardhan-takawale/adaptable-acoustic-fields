"""End-to-end regression tests for the slope fit -- it must actually FIT, not just return.

Why this file exists. The P3-2c audit A1 fix added a publication-policy block to name which
rho is the published one. The block was inserted into ``fit_cell`` instead of ``slope_fit``,
and it carried a ``return out`` at function-body indentation. Consequences:

  * every per-cell regression became unreachable dead code, so ``fitted`` was False for every
    cell, ``n_cells`` went to 0 and every published rho became NaN;
  * ``rho_published`` read ``out.get("aggregate")`` inside a per-CELL dict, which has no such
    key, so it was silently always None.

The A1 guard tests did not catch it because they asserted over ALREADY-STORED summary.json
files, which had been produced by the pre-A1 code. They validated documents, not the code that
generates them -- so the one change they were written to protect was invisible to them.

These tests therefore run the fit on synthetic data with a KNOWN slope and assert the numbers
come back finite and correct. A dead-code regression fails them immediately.
"""
from __future__ import annotations

import numpy as np
import pytest

from aaf.eval.p3_2b_slopefit import (
    KAPPA,
    MIN_ALPHA_POINTS,
    a_theory_hz_per_m,
    fit_cell,
    slope_fit,
)

L, W, WALL, FAMILY = 4.5, 4.0, "west", "x_axial"


def _points(a_true: float, n: int = 6, span: float = 1.5, noise: float = 0.0) -> list:
    """``n`` alpha points on a straight line of slope ``a_true`` through d_m in [0, span]."""
    rng = np.random.default_rng(0)
    d_m = np.linspace(0.0, span, n)
    eps = rng.normal(0.0, noise, size=n) if noise else np.zeros(n)
    return [
        {
            "d_m": float(x),
            "d_bw_pred": float(a_true * x + e),
            "d_bw_gt": float(a_true * x),
            "n_modes": 4,
            "n_modes_candidate": 5,
            "alpha": 0.15 + 0.1 * i,
            "in_slab": bool(i == n - 2),
        }
        for i, (x, e) in enumerate(zip(d_m, eps))
    ]


def test_fit_cell_actually_fits() -> None:
    """The core regression: a well-conditioned cell must come back fitted, not early-returned."""
    a_th = a_theory_hz_per_m(L, W, WALL, FAMILY, kappa=KAPPA)
    cell = fit_cell(_points(a_th), L, W, WALL, FAMILY)

    assert cell["fitted"] is True, f"cell was not fitted: {cell.get('reject_reason')!r}"
    assert cell["reject_reason"] is None
    assert np.isfinite(cell["a_fit"]), "a_fit is NaN -- the regression did not run"
    assert np.isfinite(cell["rho"]), "rho is NaN -- the regression did not run"
    # Data was generated at exactly the theory slope, so rho must be 1.
    assert cell["rho"] == pytest.approx(1.0, abs=1e-6)
    assert cell["a_fit"] == pytest.approx(a_th, rel=1e-6)


def test_fit_cell_recovers_a_known_non_unit_rho() -> None:
    """A cell at 70% of theory must report rho = 0.70, not merely 'something finite'."""
    a_th = a_theory_hz_per_m(L, W, WALL, FAMILY, kappa=KAPPA)
    cell = fit_cell(_points(0.7 * a_th), L, W, WALL, FAMILY)
    assert cell["rho"] == pytest.approx(0.7, abs=1e-6)


def test_slab_point_is_populated() -> None:
    """``slab_point`` is computed after the fit, so the dead-code bug erased it entirely."""
    a_th = a_theory_hz_per_m(L, W, WALL, FAMILY, kappa=KAPPA)
    cell = fit_cell(_points(a_th), L, W, WALL, FAMILY)
    assert cell["slab_point"] is not None, "slab_point missing -- post-fit code did not run"
    assert np.isfinite(cell["slab_point"]["ratio_pred_over_theory"])


def test_fit_cell_still_rejects_degenerate_input() -> None:
    """The guard rails must survive the fix -- restoring the fit must not disable rejection."""
    empty = fit_cell([{"d_m": 0.1, "d_bw_pred": 1.0, "d_bw_gt": 1.0,
                       "n_modes": 0, "n_modes_candidate": 3, "alpha": 0.3,
                       "in_slab": False}], L, W, WALL, FAMILY)
    assert empty["fitted"] is False
    assert empty["reject_reason"] == "no paired-valid modes"

    too_few = fit_cell(_points(1.0, n=MIN_ALPHA_POINTS - 1), L, W, WALL, FAMILY)
    assert too_few["fitted"] is False
    assert "alpha points" in (too_few["reject_reason"] or "")

    too_narrow = fit_cell(_points(1.0, n=6, span=0.1), L, W, WALL, FAMILY)
    assert too_narrow["fitted"] is False
    assert "span" in (too_narrow["reject_reason"] or "")


def test_slope_fit_publishes_the_gated_rho() -> None:
    """``rho_published`` must equal the slab_local median -- the A1 contract, tested on
    FRESHLY COMPUTED cells rather than on a stored summary.json."""
    a_th = a_theory_hz_per_m(L, W, WALL, FAMILY, kappa=KAPPA)
    cells = [fit_cell(_points(0.9 * a_th, noise=0.01), Lg, W, WALL, FAMILY)
             for Lg in (4.5, 4.6, 4.7, 4.8)]
    out = slope_fit(cells, n_boot=32)

    slab_local = out["aggregate"]["own_family"]["slab_local"]
    assert slab_local["n_cells"] == len(cells), "cells did not reach the aggregate"
    assert np.isfinite(slab_local["rho_median"])

    assert "rho_published" in out, "publication policy block missing from slope_fit"
    assert out["rho_published"] is not None, "rho_published is None -- wrong dict scope"
    assert out["rho_published"] == slab_local["rho_median"]
    assert out["publication_policy"]["gate_source"] == (
        "aggregate.own_family.slab_local.rho_median")
    # The pooled `all` aggregate must never be what gets published.
    assert "aggregate.own_family.all" in out["publication_policy"]["diagnostic_only"]


def test_rho_published_is_not_the_pooled_all() -> None:
    """Guard the exact A1 confusion: slab and non-slab cells at DIFFERENT rho must give a
    published value tracking slab_local, not the pooled median."""
    # Each cell's rho is measured against ITS OWN geometry's theory slope, so the synthetic
    # data has to be generated per geometry too -- generating every cell at L=4.5's slope and
    # then fitting it at L=4.6 yields rho = 0.5 * a_th(4.5)/a_th(4.6), not 0.5.
    slab = [fit_cell(_points(0.5 * a_theory_hz_per_m(Lg, W, "west", FAMILY, kappa=KAPPA)),
                     Lg, W, "west", FAMILY)
            for Lg in (4.5, 4.6, 4.7)]
    # "south" has no holdout slab, so these land in non_slab; own_family for south is y_axial.
    non = [fit_cell(_points(1.0 * a_theory_hz_per_m(Lg, W, "south", "y_axial", kappa=KAPPA)),
                    Lg, W, "south", "y_axial")
           for Lg in (4.5, 4.6, 4.7)]
    out = slope_fit(slab + non, n_boot=32)
    agg = out["aggregate"]["own_family"]
    assert out["rho_published"] == agg["slab_local"]["rho_median"]
    assert out["rho_published"] == pytest.approx(0.5, abs=1e-6)
    assert agg["all"]["rho_median"] != pytest.approx(0.5, abs=1e-6), (
        "test is not discriminating: pooled and slab_local coincide")
