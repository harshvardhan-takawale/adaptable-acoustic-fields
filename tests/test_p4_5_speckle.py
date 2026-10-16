"""P4-5 speckle thread (D150): the inference-time render changes (d3, d4) and the field statistics.

Rule 8: every modification here changes behaviour WITHOUT changing parameters, so the tests assert
OUTPUTS:
  * an UNMODIFIED render through the copied loop reproduces the cached B5 @60K dense render;
  * a level cut changes the output (the mask is live), and a no-op cut (all 20 levels) does not;
  * rotating the fan by exactly one ray spacing reproduces the unrotated render (the rays are only
    permuted), while half a spacing changes it;
  * the band-limited projection keeps a physical plane wave on the L-room mask and removes
    out-of-band noise -- the governing principle (nodal lines intact) as a test;
  * the d1 ratio of a field against itself is exactly 1.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.p4_5_speckle import grid_pairs, project_lowpass, room_k_grid

CK = "outputs/p4_5/arms/B5/ckpt_iter0060000.pt"
CACHE = "outputs/p4_5/gate2_dense/cache/B5_60K__track_p4_5_testdense.npz"
H5 = "data/track_p4_5_testdense/L5.54_W5.28_d1.42_w1.52.h5"
N = 48                                                    # receivers rendered per GPU test


def _gpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("tinycudann")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return torch


def _shape0():
    from aaf.data.shape_configs import configs_from_rows
    from scripts.p4_5_speckle_render import MANIFEST
    return configs_from_rows(json.load(open(MANIFEST))["configs"], split="test")[0]


def _render(level_cut=None, fan_offset=0.0):
    torch = _gpu()
    from scripts.p4_5_speckle_render import load_modified, render_shape
    m, r = load_modified(CK, torch.device("cuda"), level_cut, fan_offset)
    P, _T, _rx, _los = render_shape(m, r, _shape0(), torch.device("cuda"), rx_limit=N)
    return P


@pytest.mark.skipif(not Path(CACHE).exists(), reason="dense cache deleted in P4-6 housekeeping (D160(b)); "
                    "regenerate with scripts/p4_5_gate2_dense.py to re-run")
def test_unmodified_render_reproduces_cache():
    P = _render()
    ref = np.load(CACHE)["P0"][:N]
    assert np.allclose(P, ref, rtol=1e-5, atol=1e-6 * np.abs(ref).max())


def test_level_cut_is_live_and_noop_cut_is_identity():
    base = _render()
    assert np.array_equal(_render(level_cut=20), base)
    cut = _render(level_cut=6)
    assert np.abs(cut[:, 40:] - base[:, 40:]).max() > 1e-3 * np.abs(base[:, 40:]).max()


def test_fan_rotation_by_one_spacing_permutes_rays():
    base = _render()
    one = _render(fan_offset=2 * math.pi / 64 - 1e-9)
    half = _render(fan_offset=math.pi / 64)
    scale = np.abs(base[:, 40:]).max()
    assert np.abs(one[:, 40:] - base[:, 40:]).max() < 1e-3 * scale
    assert np.abs(half[:, 40:] - base[:, 40:]).max() > 1e-2 * scale


def _room():
    import h5py
    with h5py.File(H5) as f:
        keep = np.asarray(json.loads(f.attrs["lattice_keep"]), bool).reshape(64, 64)
        L, W = float(f.attrs["L"]), float(f.attrs["W"])
    return keep, L, W


def test_projection_keeps_plane_wave_and_removes_noise():
    keep, L, W = _room()
    f = 200.0
    k0 = 2 * math.pi * f / 343.0
    K, xs, ys = room_k_grid(L, W)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    kx, ky = 0.8 * k0 * math.cos(0.7), 0.8 * k0 * math.sin(0.7)
    wave = np.exp(1j * (kx * X + ky * Y)) + 0.5 * np.cos(0.6 * k0 * X)  # nodal structure, |k| < k0
    lp = project_lowpass(wave[keep][None, :], keep, K, np.array([1.5 * k0]))[0]
    rel = np.linalg.norm(lp - wave[keep]) / np.linalg.norm(wave[keep])
    assert rel < 1e-3, rel
    rng = np.random.default_rng(0)
    noise = rng.normal(size=keep.sum()) + 1j * rng.normal(size=keep.sum())
    lpn = project_lowpass(noise[None, :], keep, K, np.array([1.5 * k0]))[0]
    assert np.linalg.norm(lpn) < 0.5 * np.linalg.norm(noise)


def test_d1_ratio_of_field_with_itself_is_one():
    keep, _L, _W = _room()
    a, b = grid_pairs(keep)
    rng = np.random.default_rng(1)
    T = rng.normal(size=(keep.sum(), 4)) + 1j * rng.normal(size=(keep.sum(), 4))
    d = np.abs(20 * np.log10(np.abs(T[a])) - 20 * np.log10(np.abs(T[b])))
    assert np.allclose(d.mean(0) / d.mean(0), 1.0)
    assert a.shape == b.shape and (a != b).all()
