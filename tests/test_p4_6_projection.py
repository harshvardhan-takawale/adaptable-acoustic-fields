"""P4-6 (D160): the render-time physical projection as a library default.

Behavioural: it keeps physical plane waves (|k| <= k0) on a real L-room mask -- nodal lines intact,
the governing principle of the speckle thread -- leaves bins below 25 Hz bit-identical, and is the
same function the D152 diagnostics used.
"""
from __future__ import annotations

import json
import math

import h5py
import numpy as np

H5 = "data/track_p4_5_testdense/L5.54_W5.28_d1.42_w1.52.h5"


def _room():
    with h5py.File(H5) as f:
        keep = np.asarray(json.loads(f.attrs["lattice_keep"]), bool).reshape(64, 64)
        L, W = float(f.attrs["L"]), float(f.attrs["W"])
    xs, ys = np.linspace(0.12, L - 0.12, 64), np.linspace(0.12, W - 0.12, 64)
    return keep, L, W, np.meshgrid(xs, ys, indexing="ij")


def test_project_field_keeps_plane_waves_and_leaves_low_bins_raw():
    from aaf.eval.physical_projection import project_field
    keep, L, W, (X, Y) = _room()
    rng = np.random.default_rng(0)
    F = (rng.normal(size=(keep.sum(), 601)) + 1j * rng.normal(size=(keep.sum(), 601))) * 1e-3
    bins = [50, 200, 400, 560]
    for b in bins:
        k0 = 2 * math.pi * b * 0.5 / 343.0
        F[:, b] = (np.exp(1j * 0.8 * k0 * (math.cos(0.7) * X + math.sin(0.7) * Y))
                   + 0.5 * np.cos(0.6 * k0 * X))[keep]
    P = project_field(F, keep, (L - 0.24) / 63, (W - 0.24) / 63)
    for b in bins:
        rel = np.linalg.norm(P[:, b] - F[:, b]) / np.linalg.norm(F[:, b])
        assert rel < 1e-3, (b, rel)
    assert np.array_equal(P[:, :50], F[:, :50].astype(np.complex64))
    noise = np.linalg.norm(P[:, 300]) / np.linalg.norm(F[:, 300])      # white-noise bin
    assert noise < 0.5


def test_speckle_diagnostics_use_the_library_projection():
    import aaf.eval.physical_projection as lib
    import scripts.p4_5_speckle as sp
    assert sp.project_lowpass is lib.project_lowpass
