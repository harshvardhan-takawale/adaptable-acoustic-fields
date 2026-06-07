"""Tests for `aaf.eval.zero_shot_diagnosis` — the P2-3 self-diagnosis primitives
(manifold distance + the D37 3-way verdict). Pure-numpy; no CUDA."""
import numpy as np
import pytest

from aaf.eval.zero_shot_diagnosis import (
    aggregate_verdict,
    classify_zero_shot_room,
    compute_manifold_distances,
)


# ----------------------------- manifold distance -----------------------------

def test_manifold_min_dist_zero_when_zstar_is_a_train_latent():
    rng = np.random.default_rng(0)
    z_train = rng.normal(size=(45, 16))
    z_star = z_train[7].copy()
    out = compute_manifold_distances(z_star, z_train)
    assert out["latent_min_dist"] == pytest.approx(0.0, abs=1e-9)
    assert out["latent_nearest_room_idx"] == 7


def test_manifold_min_dist_large_when_zstar_far():
    z_train = np.zeros((45, 16))
    z_star = np.full(16, 10.0)
    out = compute_manifold_distances(z_star, z_train)
    # ‖[10]*16‖ = 10*sqrt(16) = 40
    assert out["latent_min_dist"] == pytest.approx(40.0, rel=1e-6)
    assert out["latent_mean_dist"] == pytest.approx(40.0, rel=1e-6)


def test_manifold_geom_nearest_picks_geometrically_closest_room():
    # 3 rooms; latent index ≠ geometry index so the test is meaningful.
    z_train = np.array([[0.0, 0.0], [5.0, 0.0], [9.0, 0.0]])
    train_LWH = [[3.0, 3.0, 3.0], [6.0, 4.0, 3.0], [4.0, 4.0, 3.0]]
    z_star = np.array([4.6, 0.0])            # nearest latent = room 1 (dist 0.4)
    test_LWH = [4.1, 4.0, 3.0]               # geometrically nearest = room 2
    out = compute_manifold_distances(z_star, z_train, train_LWH, test_LWH)
    assert out["latent_nearest_room_idx"] == 1
    assert out["geom_nearest_train_idx"] == 2
    # distance to the geometrically-nearest room's latent (room 2 at x=9)
    assert out["geom_nearest_train_dist"] == pytest.approx(abs(9.0 - 4.6), rel=1e-6)


def test_manifold_raises_on_dim_mismatch():
    with pytest.raises(ValueError):
        compute_manifold_distances(np.zeros(8), np.zeros((45, 16)))


# ------------------------------- 3-way verdict -------------------------------

def test_classify_success():
    branch, _ = classify_zero_shot_room(in_dist_lsd=2.1, mag_corr=0.93, geom_err_max_m=0.5)
    assert branch == "success"


def test_classify_precondition_unmet_when_in_dist_bad():
    # in-distribution didn't clear ≤2.5 → not interpretable, regardless of mag corr
    branch, _ = classify_zero_shot_room(in_dist_lsd=3.4, mag_corr=0.95, geom_err_max_m=0.05)
    assert branch == "precondition_unmet"


def test_classify_manifold_coverage_when_geom_misplaced():
    branch, _ = classify_zero_shot_room(in_dist_lsd=2.0, mag_corr=0.6, geom_err_max_m=0.9)
    assert branch == "manifold_coverage"


def test_classify_decoder_interp_when_geom_placed():
    branch, _ = classify_zero_shot_room(in_dist_lsd=2.0, mag_corr=0.6, geom_err_max_m=0.1)
    assert branch == "decoder_interp"


def test_classify_geom_threshold_boundary_is_placed():
    # exactly at the threshold counts as "placed" (≤), so decoder_interp
    branch, _ = classify_zero_shot_room(in_dist_lsd=2.0, mag_corr=0.6,
                                        geom_err_max_m=0.3, geom_thresh=0.3)
    assert branch == "decoder_interp"


def test_classify_unknown_when_geom_missing_for_poor_room():
    branch, _ = classify_zero_shot_room(in_dist_lsd=2.0, mag_corr=0.6, geom_err_max_m=None)
    assert branch == "unknown"


# ------------------------------- aggregate -----------------------------------

def test_aggregate_success_when_ge5():
    branches = ["success"] * 6 + ["decoder_interp"] * 2
    v = aggregate_verdict(branches, 8)
    assert v.startswith("SUCCESS")


def test_aggregate_manifold_dominant():
    branches = ["success"] * 2 + ["manifold_coverage"] * 5 + ["decoder_interp"]
    v = aggregate_verdict(branches, 8)
    assert "MANIFOLD-COVERAGE" in v and "P2-4" in v


def test_aggregate_decoder_dominant():
    branches = ["success"] * 2 + ["decoder_interp"] * 5 + ["manifold_coverage"]
    v = aggregate_verdict(branches, 8)
    assert "DECODER-AT-INTERPOLATED-LATENT" in v


def test_aggregate_all_precondition_unmet():
    v = aggregate_verdict(["precondition_unmet"] * 8, 8)
    assert "PRECONDITION UNMET" in v
