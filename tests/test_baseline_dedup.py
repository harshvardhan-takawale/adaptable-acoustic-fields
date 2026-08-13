"""P3-2 config enumeration: baseline dedup, filename uniqueness, split counts, coverage.

Guards the two silent-corruption hazards described in aaf/data/mat_configs.py:
one baseline per geometry, and no two configs sharing an HDF5 filename.
"""
from __future__ import annotations

import pytest

from aaf.data.mat_configs import (
    HELDOUT_COMBOS,
    UNSEEN_ALPHA,
    MatConfig,
    assert_unique_filenames,
    coverage_report,
    enumerate_configs,
    make_config,
    room_filename_2d_mat,
    round_dim,
)
from aaf.walls import ALPHA_BASELINE, MATERIALS, WALL_INDEX, WALLS_2D, alphas_for

G_TRAIN = [(3.0 + 0.07 * i, 3.0 + 0.05 * i) for i in range(40)]
G_TEST = [(3.5 + 0.2 * i, 3.2 + 0.15 * i) for i in range(10)]


def test_one_baseline_per_geometry():
    cfgs = enumerate_configs(G_TRAIN, exclude_combos=HELDOUT_COMBOS)
    n_base = sum(1 for c in cfgs if c.is_baseline)
    assert n_base == len(G_TRAIN), "expected exactly one baseline per geometry"
    # every baseline is a *distinct* geometry
    assert len({(c.L, c.W) for c in cfgs if c.is_baseline}) == len(G_TRAIN)


def test_wall_times_baseline_material_collapses_to_baseline():
    """(wall=k, M0) is the SAME room for every k -- it must not be emitted 4x."""
    for wall in WALLS_2D:
        c = make_config(4.5, 4.0, wall=wall, material="M0")
        assert c.is_baseline
        assert c.alphas == tuple([ALPHA_BASELINE] * 4)
    base = make_config(4.5, 4.0)
    assert len({make_config(4.5, 4.0, wall=w, material="M0").filename
                for w in WALLS_2D} | {base.filename}) == 1


def test_baseline_material_rejected_from_materials_list():
    with pytest.raises(ValueError):
        enumerate_configs(G_TEST, materials=("M0", "M1"))


def test_split_counts():
    train = enumerate_configs(G_TRAIN, exclude_combos=HELDOUT_COMBOS)
    test = enumerate_configs(G_TEST)
    split_ii = enumerate_configs(G_TRAIN, only_combos=HELDOUT_COMBOS, include_baseline=False)
    split_iv = enumerate_configs(G_TEST, unseen_alpha=UNSEEN_ALPHA)
    assert len(train) == 440          # 40 x (1 baseline + 12 - 2 held out)
    assert len(test) == 130           # 10 x 13
    assert len(split_ii) == 80        # 40 x 2
    assert len(split_iv) == 40        # 10 x 4 walls
    assert len(train) + len(test) + len(split_ii) + len(split_iv) == 690


def test_holdouts_absent_from_train_present_in_test():
    train = enumerate_configs(G_TRAIN, exclude_combos=HELDOUT_COMBOS)
    test = enumerate_configs(G_TEST)
    assert all(c.combo not in HELDOUT_COMBOS for c in train)
    for combo in HELDOUT_COMBOS:
        assert any(c.combo == combo for c in test), combo


def test_coverage_matches_spec():
    cov = coverage_report(enumerate_configs(G_TRAIN, exclude_combos=HELDOUT_COMBOS))
    assert cov["materials_per_wall"]["west"] == ["M1", "M3"]      # M2 held out
    assert cov["materials_per_wall"]["north"] == ["M1", "M2"]     # M3 held out
    assert cov["materials_per_wall"]["east"] == ["M1", "M2", "M3"]
    assert cov["materials_per_wall"]["south"] == ["M1", "M2", "M3"]
    assert cov["min_materials_per_wall"] >= 2
    assert cov["min_walls_per_material"] >= 3


def test_heldout_combos_have_a_trained_opposite_wall_twin():
    """The alpha_eff control (D44): the twin has identical mean absorption and T60."""
    from aaf.walls import WALL_TWIN

    train = enumerate_configs(G_TRAIN, exclude_combos=HELDOUT_COMBOS)
    trained = {c.combo for c in train if c.combo}
    for wall, mat in HELDOUT_COMBOS:
        twin = (WALL_TWIN[wall], mat)
        assert twin in trained, f"{(wall, mat)} has no trained twin {twin}"
        # identical multiset of absorptions => identical mean absorption
        assert sorted(alphas_for(wall, mat)) == sorted(alphas_for(*twin))


def test_filenames_unique_and_roundtrip():
    allc = (enumerate_configs(G_TRAIN, exclude_combos=HELDOUT_COMBOS)
            + enumerate_configs(G_TEST)
            + enumerate_configs(G_TEST, unseen_alpha=UNSEEN_ALPHA))
    assert_unique_filenames(allc)
    c = make_config(4.5, 4.0, wall="west", material="M1")
    assert c.filename == "L4.50_W4.00_aW0.05_aE0.15_aS0.15_aN0.15.h5"
    assert room_filename_2d_mat(4.5, 4.0, c.alphas) == c.filename


def test_dimensions_rounded_to_filename_precision():
    """Guards the latent {:.2f} collision bug: dims must be rounded AT GENERATION."""
    c = make_config(3.620434065857401, 4.550612, wall="east", material="M2")
    assert c.L == round_dim(3.620434065857401) == 3.62
    assert "L3.62_W4.55" in c.filename
    with pytest.raises(AssertionError):
        assert_unique_filenames([make_config(3.621, 4.0), make_config(3.624, 4.0)])


def test_alphas_place_edit_on_the_right_wall():
    for wall in WALLS_2D:
        c = make_config(4.5, 4.0, wall=wall, material="M3")
        a = c.alphas
        assert a[WALL_INDEX[wall]] == pytest.approx(MATERIALS["M3"])
        assert sum(1 for x in a if x != ALPHA_BASELINE) == 1


def test_unseen_alpha_split_is_labelled_and_untrained():
    s4 = enumerate_configs(G_TEST, unseen_alpha=UNSEEN_ALPHA)
    assert all(c.material == "A030" for c in s4)
    assert all(c.alpha_edit == pytest.approx(UNSEEN_ALPHA) for c in s4)
    assert UNSEEN_ALPHA not in set(MATERIALS.values()), "alpha=0.30 must be a NEW value"
    assert all(not c.is_baseline for c in s4)
