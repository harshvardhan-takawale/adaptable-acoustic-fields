"""Sweep configs: each YAML loads, the model built from it has the expected
HashGrid input dim, latent_dim, and L-head presence.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = REPO_ROOT / "configs" / "sweep"


cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="tcnn requires CUDA"
)


def _all_sweep_configs():
    return sorted(SWEEP_DIR.glob("R*.yaml"))


def test_yamls_present():
    """R0-R5 (Chunk 3.5) + R6-R8 (Chunk 3.5+ addendum) = 9 sweep YAMLs."""
    configs = _all_sweep_configs()
    assert len(configs) >= 6, f"expected ≥ 6 sweep YAMLs (R0-R5), got {len(configs)}: {[p.name for p in configs]}"
    expected = {f"R{i}_{tag}" for i, tag in enumerate([
        "central", "smaller_hash", "larger_latent",
        "no_lhead", "strong_lhead", "strong_l2",
    ])}
    expected |= {"R6_tiny_lhead", "R7_medium_hash", "R8_tiny_latent"}  # Chunk-3.5+ addendum
    present = {p.stem for p in configs}
    missing = expected - present
    assert not missing, f"missing sweep configs: {missing}"


@pytest.mark.parametrize("yaml_path", _all_sweep_configs(), ids=lambda p: p.stem)
def test_yaml_has_required_fields(yaml_path):
    cfg = yaml.safe_load(open(yaml_path))
    expected = {
        "run_id", "data_sweep",
        "log2_hashmap_size", "n_levels", "latent_dim",
        "l_head_weight", "lambda_latent_l2",
        "n_iters", "batch_size", "grad_accum_steps",
        "lr_network", "lr_latent",
        "n_pts_per_ray", "n_azi", "val_every", "ckpt_every",
    }
    missing = expected - set(cfg)
    assert not missing, f"{yaml_path.name} missing fields: {missing}"


@cuda_required
@pytest.mark.parametrize("yaml_path", _all_sweep_configs(), ids=lambda p: p.stem)
def test_model_from_yaml_matches_cfg(yaml_path):
    """Build a tiny INR2D_AutoDecoder from each YAML; verify architecture.

    Uses small n_freq_bins to keep the test fast.
    """
    from aaf.models.inr_2d import INR2D_AutoDecoder

    cfg = yaml.safe_load(open(yaml_path))
    hg = {
        "otype": "HashGrid",
        "n_levels": int(cfg["n_levels"]),
        "n_features_per_level": 2,
        "log2_hashmap_size": int(cfg["log2_hashmap_size"]),
        "base_resolution": 16,
        "per_level_scale": 1.5,
    }
    l_head_enabled = float(cfg["l_head_weight"]) > 0
    model = INR2D_AutoDecoder(
        n_rooms=7,
        latent_dim=int(cfg["latent_dim"]),
        n_freq_bins=129,                # tiny for test speed
        hash_grid_config=hg,
        l_head_enabled=l_head_enabled,
    ).cuda()

    # Latent table dim matches.
    assert model.latents.weight.shape[1] == int(cfg["latent_dim"])
    # L-head presence matches l_head_weight > 0.
    assert (model.l_head is None) == (not l_head_enabled)
    # Sigma encoder input dim = pos_emb (2*n_levels) + tx_pos_emb (2*n_levels) + latent_dim.
    expected_sigma_in = 2 * (2 * int(cfg["n_levels"])) + int(cfg["latent_dim"])
    assert model._model_encoder_sigma.n_input_dims == expected_sigma_in
