"""P3-3-FAST Track 2b: aperture configs + the 55-d aperture conditioning block.

The block offsets are load-bearing: FiLM sees one flat vector, so a mis-sized geometry block
silently shifts the aperture channels into the geometry ones and the model trains happily on
the wrong physics. These tests pin the offsets and the independence of the blocks.
"""
import math

import pytest
import torch

from aaf.data.aperture_configs import (A_HOLDOUT, ApertureConfig, DIVIDER_ALPHA, TEST_DOMAINS,
                                       enumerate_test_configs, in_holdout, manifest_rows,
                                       sample_train_configs, sample_train_domains)
from aaf.models.conditioning_2d import (APERTURE_DIM_2D, COND_SOURCE_APER, N_K_GEOM,
                                        aperture_features_2d, build_cond_vector_2d,
                                        cond_dim_for, normalize_params_aper_2d)

GEOM_BLOCK = 2 * N_K_GEOM            # 16 features per geometry dim
APER_LO = 3 * GEOM_BLOCK             # 48


def test_dim_and_registration():
    assert APERTURE_DIM_2D == 55
    assert cond_dim_for(COND_SOURCE_APER) == APERTURE_DIM_2D
    v = aperture_features_2d(8.0, 4.0, 4.0, 1.0)
    assert v.shape == (55,)
    assert torch.isfinite(v).all()


def test_model_whitelist_accepts_aperture():
    """inr_2d has its OWN cond_source check that build_cond_vector_2d does not consult.

    Read as TEXT rather than imported: ``aaf.models.inr_2d`` pulls tinycudann, which refuses
    to import without CUDA, and this whole file must stay runnable on a CPU-only node.
    """
    from pathlib import Path

    src = Path(inr_2d_path()).read_text()
    i = src.index("cond_source not in (")
    assert '"aperture"' in src[i:i + 200], "aperture missing from the inr_2d whitelist"


def inr_2d_path():
    from pathlib import Path

    import aaf

    return Path(aaf.__file__).parent / "models" / "inr_2d.py"


def test_normalization_box():
    u = normalize_params_aper_2d(8.0, 4.0, 4.0, 4.0)
    assert u.tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])
    u = normalize_params_aper_2d(9.0, 4.5, 0.6 * 9.0, 2.5)
    assert u[:3].tolist() == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)
    assert u[3].item() == pytest.approx(math.sqrt(2.5) / 2.0)


def test_block_offsets_and_independence():
    base = aperture_features_2d(8.0, 4.0, 4.0, 1.0)

    # changing a moves ONLY the aperture block [48:55]
    moved_a = aperture_features_2d(8.0, 4.0, 4.0, 2.0)
    assert torch.equal(base[:APER_LO], moved_a[:APER_LO])
    assert not torch.allclose(base[APER_LO:], moved_a[APER_LO:])

    # changing x0 moves ONLY [32:48]
    moved_x = aperture_features_2d(8.0, 4.0, 4.2, 1.0)
    assert torch.equal(base[:2 * GEOM_BLOCK], moved_x[:2 * GEOM_BLOCK])
    assert not torch.allclose(base[2 * GEOM_BLOCK:APER_LO], moved_x[2 * GEOM_BLOCK:APER_LO])
    assert torch.equal(base[APER_LO:], moved_x[APER_LO:])

    # changing W moves ONLY [16:32] -- x0 is normalized by L, not W, so its block is fixed
    moved_w = aperture_features_2d(8.0, 4.3, 4.0, 1.0)
    assert torch.equal(base[:GEOM_BLOCK], moved_w[:GEOM_BLOCK])
    assert not torch.allclose(base[GEOM_BLOCK:2 * GEOM_BLOCK],
                              moved_w[GEOM_BLOCK:2 * GEOM_BLOCK])
    assert torch.equal(base[2 * GEOM_BLOCK:], moved_w[2 * GEOM_BLOCK:])

    # the identity channel is sqrt(a)/2 and sits at the head of the aperture block
    assert base[APER_LO].item() == pytest.approx(math.sqrt(1.0) / 2.0)
    assert moved_a[APER_LO].item() == pytest.approx(math.sqrt(2.0) / 2.0)


def test_build_cond_vector_dispatch():
    c = ApertureConfig(8.0, 4.0, 4.0, 1.0, "aperture", "train", 0)
    v = build_cond_vector_2d(COND_SOURCE_APER, c.L, c.W, c.alphas, x0=c.x0, a=c.a)
    assert torch.equal(v, aperture_features_2d(c.L, c.W, c.x0, c.a))
    with pytest.raises(ValueError):          # alphas alone cannot describe a divider
        build_cond_vector_2d(COND_SOURCE_APER, c.L, c.W, c.alphas)


def test_extra_walls_three_regimes():
    sealed = ApertureConfig(8.0, 4.0, 4.0, 0.0, "sealed", "train", 0)
    spec = sealed.extra_walls
    assert len(spec) == 1 and "apertures" not in spec[0]
    assert spec[0]["pos"] == 4.0 and spec[0]["alpha"] == DIVIDER_ALPHA
    assert sealed.sealed and not sealed.fully_open

    openc = ApertureConfig(8.0, 4.0, 4.0, 4.0, "open", "train", 0)
    assert openc.extra_walls == [] and openc.fully_open and not openc.sealed

    door = ApertureConfig(8.0, 4.0, 4.0, 1.0, "aperture", "train", 0)
    (lo, hi), = door.extra_walls[0]["apertures"]
    assert (lo, hi) == pytest.approx((1.5, 2.5))     # centred on y = W/2
    assert door.alphas == (0.15,) * 4                # the four walls stay at baseline


def test_filename_and_alphas():
    c = ApertureConfig(7.25, 3.6, 3.2, 0.95, "t_holdout", "test", 0)
    assert c.filename == "L7.25_W3.60_x3.20_a0.9500.h5"
    assert len(c.alphas) == 4


def test_sampler_holdout_is_exact():
    doms = sample_train_domains()
    tr = sample_train_configs(doms)
    te = enumerate_test_configs(TEST_DOMAINS)
    assert len(tr) == 400 and len(te) == 72
    assert not [c for c in tr if in_holdout(c.a) and not c.sealed]
    assert sum(1 for c in te if in_holdout(c.a)) >= 3
    assert all(A_HOLDOUT[0] <= 1.0 <= A_HOLDOUT[1] for _ in (0,))

    rows = manifest_rows(tr, te)
    assert len({r["filename"] for r in rows}) == len(rows) == 472
    # train and test geometries must not collide -- the filename carries only (L, W, x0, a)
    gtr = {(c.L, c.W, c.x0) for c in tr}
    gte = {(c.L, c.W, c.x0) for c in te}
    assert not (gtr & gte)


def test_sampler_is_reproducible():
    assert sample_train_domains() == sample_train_domains()
    a1 = [c.a for c in sample_train_configs(sample_train_domains())]
    a2 = [c.a for c in sample_train_configs(sample_train_domains())]
    assert a1 == a2
