"""P3-2b evaluation contract tests. CPU-only -- no GPU, no checkpoint, no rendering.

Three things are pinned here, each guarding a failure that produces plausible numbers
rather than an error:

1. **Estimator identity.** ``p3_2b_eval`` must use the FROZEN P3-2 estimator objects, not
   copies. ``is`` comparisons catch a copy-paste fork that a value comparison would not.
2. **Split assignment.** Exact counts, and the branch-order rules (two-wall before slab,
   slab keyed on (wall, m) so the trained twin stays out of the headline split).
3. **The acceptance thresholds' hash.** Pinned to a literal, so changing a threshold fails
   CI instead of silently re-scoring a published verdict.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aaf.data.mat_configs_cont import in_slab, m_of_alpha
from aaf.eval import p3_2_eval, p3_2b_eval
from aaf.eval.modal_projection import TANGENTIAL, X_AXIAL, Y_AXIAL
from aaf.eval.p3_2b_accept import THRESHOLDS, thresholds_sha256, verdict
from aaf.eval.p3_2b_slopefit import KAPPA, a_theory_hz_per_m, fit_cell, own_family
from aaf.eval.p3_2b_splits import (
    EXPECTED_COUNTS,
    S1,
    S2,
    S3,
    S4,
    S5,
    SPLIT_ORDER,
    assert_split_counts,
    build_splits,
    classify,
    edited_walls,
    s5_slab_subset,
)
from aaf.models.conditioning_2d import fourier_features_2d, m_linear_features_2d
from aaf.walls import ALPHA_BASELINE, WALL_INDEX, WALLS_2D

# The literal that makes a threshold edit a CI failure. Recomputing it here would defeat
# the purpose -- if this assertion fails, a frozen acceptance criterion was changed.
THRESHOLDS_SHA256_PINNED = "a8479c5e1dcc3ab5b2a505809e0f9d9f7dd4590f009f3549c9070e860a33caa1"


# ----------------------------------------------------------------- 1. estimator identity
@pytest.mark.parametrize("name", [
    "analyse", "paired_cells", "edit_stats", "by_family_stats", "fidelity", "db_map",
    "mode_shape_invariance", "band_limit", "load_gt", "load_model", "make_geom_ctx",
    "control_c4", "in_dist_val_lsd", "find_checkpoint", "theory_slope", "_pearson",
    "_mean", "_std", "_nan",
])
def test_estimators_are_the_frozen_objects(name):
    assert getattr(p3_2b_eval, name) is getattr(p3_2_eval, name), (
        "{} must be the frozen P3-2 object, not a re-implementation".format(name))


def test_frozen_constants_are_shared():
    assert p3_2b_eval.BAND_HI_HZ is p3_2_eval.BAND_HI_HZ
    assert p3_2b_eval.FAMILIES is p3_2_eval.FAMILIES
    assert p3_2b_eval.SELECTIVITY_FLOOR_HZ is p3_2_eval.SELECTIVITY_FLOOR_HZ


def test_kappa_matches_the_gate_slope():
    """rho is a_fit/a_theory; a kappa that drifted from the gate would rescale every rho."""
    assert p3_2_eval.THEORY_SLOPE_FALLBACK == KAPPA
    assert abs(p3_2_eval.theory_slope() - KAPPA) < 1e-12


def test_render_config_arm_matches_p3_2_conditioning_on_the_fourier_arm():
    """The only reason ``render_config`` is not reused verbatim is the encoder dispatch.
    On the fourier arms the two must build the identical vector."""
    from aaf.models.conditioning_2d import build_cond_vector_2d

    alphas = (0.50, 0.15, 0.15, 0.70)
    a = build_cond_vector_2d("geom_alpha_fourier", 4.51, 4.0, alphas)
    b = fourier_features_2d(4.51, 4.0, alphas)
    assert a.shape == b.shape == (64,)
    assert np.array_equal(a.numpy(), b.numpy())
    c = build_cond_vector_2d("m_linear", 4.51, 4.0, alphas)
    assert c.shape == (60,)
    assert np.array_equal(c.numpy(), m_linear_features_2d(4.51, 4.0, alphas).numpy())


# ---------------------------------------------------------------------- 2. split assignment
@pytest.fixture(scope="module")
def splits():
    sp, _ = build_splits()
    return sp


def test_split_counts_exact(splits):
    assert_split_counts(splits)
    assert {k: len(v) for k, v in splits.items()} == EXPECTED_COUNTS


def test_assert_split_counts_rejects_a_deviation(splits):
    bad = {k: list(v) for k, v in splits.items()}
    bad[S2] = bad[S2][:-1]
    with pytest.raises(AssertionError):
        assert_split_counts(bad)


def test_edited_walls_from_the_alpha_vector_not_a_scalar():
    a = [ALPHA_BASELINE] * 4
    a[WALL_INDEX["west"]] = 0.50
    a[WALL_INDEX["south"]] = 0.70
    assert edited_walls(a) == ("west", "south")
    assert edited_walls([ALPHA_BASELINE] * 4) == ()


def test_slab_membership_is_keyed_on_wall_and_m():
    """(west, 0.50) is held out; its trained twin (east, 0.50) must NOT be."""
    assert in_slab("west", 0.50) and not in_slab("east", 0.50)
    assert in_slab("north", 0.70) and not in_slab("south", 0.70)
    assert not in_slab("west", 0.70) and not in_slab("north", 0.50)

    def one(wall, alpha):
        a = [ALPHA_BASELINE] * 4
        a[WALL_INDEX[wall]] = alpha
        return classify(a)

    assert one("west", 0.50) == S2
    assert one("east", 0.50) == S1          # the twin stays in the easy split
    assert one("north", 0.70) == S2
    assert one("south", 0.70) == S1
    assert one("west", 0.30) == S4


def test_two_wall_configs_land_in_s5_even_when_they_contain_a_slab_value():
    a = [ALPHA_BASELINE] * 4
    a[WALL_INDEX["west"]] = 0.50            # a slab value ...
    a[WALL_INDEX["south"]] = 0.70
    assert classify(a) == S5                # ... but this is a two-wall config


def test_s5_slab_subset_is_reported_separately(splits):
    sub = s5_slab_subset(splits)
    assert sub, "the S5 slab subset must not be empty -- it is what the branch order hides"
    assert all(len(c.edited) == 2 for c in sub)
    assert all(c.split == S5 for c in sub)


def test_splits_carry_the_right_geometry_novelty(splits):
    assert all(not c.geom_seen for name in (S1, S2, S4, S5) for c in splits[name])
    assert all(c.geom_seen for c in splits[S3])
    assert all(len(c.edited) == 1 for name in (S1, S2, S3, S4) for c in splits[name])
    assert all(len(c.edited) == 2 for c in splits[S5])


def test_s3_is_the_same_combos_as_s2_on_seen_geometries(splits):
    assert (set(c.combo_key for c in splits[S3])
            == set(c.combo_key for c in splits[S2]) == {"west0.50", "north0.70"})


def test_split_order_is_contractual():
    assert SPLIT_ORDER == (S1, S2, S3, S4, S5)


# ------------------------------------------------------------------------- 3. slope fit
def test_a_theory_is_kappa_scaled():
    """Using the RAW Lorentzian slope would score a perfect model at rho = 1/kappa = 0.602."""
    a = a_theory_hz_per_m(4.5, 4.0, "west", X_AXIAL)
    raw = 343.0 / (4.0 * math.pi * 4.5)
    assert abs(raw - 6.06557) < 1e-3
    assert abs(a - KAPPA * raw) < 1e-9
    assert abs(a - 10.073) < 1e-2
    assert a_theory_hz_per_m(4.5, 4.0, "west", Y_AXIAL) == 0.0
    assert math.isnan(a_theory_hz_per_m(4.5, 4.0, "west", TANGENTIAL))
    # south/north measure on the y-axial family and scale with W, not L.
    assert abs(a_theory_hz_per_m(4.5, 4.0, "south", Y_AXIAL)
               - KAPPA * 343.0 / (4.0 * math.pi * 4.0)) < 1e-9
    assert a_theory_hz_per_m(4.5, 4.0, "south", X_AXIAL) == 0.0


def test_own_family_mapping():
    assert own_family("west") == own_family("east") == X_AXIAL
    assert own_family("south") == own_family("north") == Y_AXIAL


def _ladder(slope_hz_per_m, n_modes=5):
    m0 = m_of_alpha(ALPHA_BASELINE)
    pts = [{"alpha": 0.15, "d_m": 0.0, "in_slab": False, "n_modes": n_modes,
            "n_modes_candidate": n_modes, "d_bw_pred": 0.0, "d_bw_gt": 0.0}]
    for a in (0.05, 0.30, 0.50, 0.70):
        dm = m_of_alpha(a) - m0
        pts.append({"alpha": a, "d_m": dm, "in_slab": in_slab("west", a),
                    "n_modes": n_modes, "n_modes_candidate": n_modes,
                    "d_bw_pred": slope_hz_per_m * dm, "d_bw_gt": slope_hz_per_m * dm})
    return pts


def test_fit_cell_recovers_a_perfect_slope():
    a_th = a_theory_hz_per_m(4.5, 4.0, "west", X_AXIAL)
    out = fit_cell(_ladder(a_th), 4.5, 4.0, "west", X_AXIAL)
    assert out["fitted"]
    assert abs(out["rho"] - 1.0) < 1e-9
    assert abs(out["rho_vs_raw_theory"] - KAPPA) < 1e-9
    assert out["frac_modes_dropped"] == 0.0
    assert out["slab_point"]["alpha"] == 0.50
    assert abs(out["slab_point"]["ratio_pred_over_theory"] - 1.0) < 1e-9


def test_fit_cell_refuses_a_short_ladder():
    out = fit_cell(_ladder(10.0)[:3], 4.5, 4.0, "west", X_AXIAL)
    assert not out["fitted"] and "alpha points" in out["reject_reason"]


def test_fit_cell_counts_dropped_modes():
    pts = _ladder(10.0)
    pts[2]["n_modes"] = 0            # every mode of this alpha point failed paired validity
    out = fit_cell(pts, 4.5, 4.0, "west", X_AXIAL)
    assert out["frac_modes_dropped"] > 0.0
    assert not out["fitted"]         # 4 usable alpha points is below the floor


# -------------------------------------------------------------------------- 4. acceptance
def test_threshold_hash_is_pinned():
    assert thresholds_sha256() == THRESHOLDS_SHA256_PINNED, (
        "a frozen acceptance threshold changed; that is a research decision, not a "
        "refactor -- update DECISIONS.md and this pin deliberately")


def test_thresholds_are_the_spec_values():
    assert THRESHOLDS["edit_bw_slope_min"] == 0.80
    assert THRESHOLDS["edit_bw_pearson_min"] == 0.80
    assert THRESHOLDS["edit_gain_min_exclusive"] == 1.00
    assert THRESHOLDS["rho_abs_dev_max"] == 0.25
    assert THRESHOLDS["split"] == S2


def _summary(slope=0.9, pearson=0.9, gain=1.5, rho=1.0, n_cells=200, frac=0.1):
    s2 = {"n_cells": n_cells, "frac_modes_dropped": frac,
          "edit": {"edit_bw_slope": slope, "edit_bw_pearson": pearson, "edit_gain": gain}}
    sf = {"aggregate": {"own_family": {"slab_local": {
        "rho_median": rho, "frac_modes_dropped": frac}}}}
    return s2, sf


def test_verdict_passes_only_when_everything_passes():
    v = verdict("arm", *_summary(), iter_=60000)
    assert v["passed"] and not v["blockers"] and v["criteria_failed"] == []
    assert v["thresholds_sha256"][:12] in v["one_line"]


@pytest.mark.parametrize("kw,failed", [
    ({"slope": 0.79}, "edit_bw_slope"),
    ({"pearson": 0.5}, "edit_bw_pearson"),
    ({"gain": 1.0}, "edit_gain"),          # strictly greater than 1.0
    ({"rho": 1.26}, "abs_rho_minus_1"),
    ({"rho": 0.74}, "abs_rho_minus_1"),
])
def test_each_criterion_can_fail_alone(kw, failed):
    v = verdict("arm", *_summary(**kw))
    assert not v["passed"] and v["criteria_failed"] == [failed]


def test_nan_fails_rather_than_passing_silently():
    v = verdict("arm", *_summary(slope=float("nan")))
    assert not v["passed"] and "edit_bw_slope" in v["criteria_failed"]


@pytest.mark.parametrize("kw,name", [
    ({"n_cells": 39}, "insufficient_S2_cells"),
    ({"frac": 0.51}, "S2_modes_unmeasurable"),
])
def test_blockers_force_failure_and_are_listed_separately(kw, name):
    v = verdict("arm", *_summary(**kw))
    assert not v["passed"]
    assert name in [b["name"] for b in v["blockers"]]
    assert v["criteria_failed"] == [], "a blocker is not a criterion failure"


def test_unmeasurable_slab_slope_blocks_even_a_perfect_looking_arm():
    """The failure mode this exists for: reject nearly every mode, fit the survivors."""
    s2, sf = _summary()
    sf["aggregate"]["own_family"]["slab_local"]["frac_modes_dropped"] = 0.9
    v = verdict("arm", s2, sf)
    assert not v["passed"]
    assert "slab_slope_modes_unmeasurable" in [b["name"] for b in v["blockers"]]
