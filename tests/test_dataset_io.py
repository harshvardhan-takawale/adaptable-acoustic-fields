"""Round-trip write/read of one HDF5 file, confirming complex storage & attrs."""
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from aaf.data.dataset_builder import (
    read_room_h5,
    room_filename,
    write_room_to_h5,
)


def _fake_results():
    """Fabricate minimal ISM + analytical result dicts (no actual simulation)."""
    n_rx = 3
    n_time = 64
    n_freq = n_time // 2 + 1
    rng = np.random.default_rng(0)
    rir = rng.standard_normal((n_rx, n_time)).astype(np.float32)
    H = np.fft.rfft(rir, n=n_time, axis=1).astype(np.complex64)
    meta = {
        "model": "fake",
        "L": 4.0,
        "W": 4.0,
        "alpha": 0.15,
        "fs": 4096.0,
        "c": 343.0,
        "source_pos": [0.5, 0.5],
        "receiver_pos": [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        "n_time_samples": n_time,
        "n_freq_bins": n_freq,
        "max_order": 5,
        "T60_sabine_2d": 0.20,
        "schroeder_freq_approx_hz": 100.0,
        "versions": {"pyroomacoustics": "0.9.0"},
    }
    return {"rir_time": rir, "H_complex": H, "meta": meta}


def test_roundtrip_complex_and_attrs(tmp_path: Path):
    ism = _fake_results()
    ana = _fake_results()
    out_path = tmp_path / room_filename(L=4.0, W=4.0, alpha=0.15)
    written = write_room_to_h5(out_path, ism, ana)
    assert written.exists()

    rt = read_room_h5(written)

    np.testing.assert_array_equal(rt["ism_H"], ism["H_complex"])
    np.testing.assert_array_equal(rt["ism_rir"], ism["rir_time"])
    np.testing.assert_array_equal(rt["ana_H"], ana["H_complex"])
    np.testing.assert_array_equal(rt["ana_rir"], ana["rir_time"])

    attrs = rt["attrs"]
    assert attrs["L"] == 4.0
    assert attrs["W"] == 4.0
    assert attrs["alpha"] == 0.15
    # source_pos was a list (JSON-encoded by writer, decoded by reader).
    assert attrs["source_pos"] == [0.5, 0.5]
    assert attrs["receiver_pos"] == [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    assert attrs["versions"]["pyroomacoustics"] == "0.9.0"
    assert "build_utc" in attrs
    assert "git_commit" in attrs


def test_complex_dtype_preserved(tmp_path: Path):
    """h5py 3.11 writes complex64 natively; verify the stored dataset's dtype."""
    ism = _fake_results()
    ana = _fake_results()
    out_path = tmp_path / "test_dtype.h5"
    write_room_to_h5(out_path, ism, ana)

    with h5py.File(out_path, "r") as f:
        assert f["ism/H_complex"].dtype == np.complex64
        assert f["analytical/H_complex"].dtype == np.complex64
        assert f["ism/rir_time"].dtype == np.float32


def test_filename_format():
    assert room_filename(L=4.0, W=4.0, alpha=0.15) == "L_4.00m_W_4.00m_alpha_0.15.h5"
    assert room_filename(L=3.25, W=4.0, alpha=0.15) == "L_3.25m_W_4.00m_alpha_0.15.h5"


def test_shape_mismatch_raises(tmp_path: Path):
    ism = _fake_results()
    ana = _fake_results()
    # Resize analytical to a different shape.
    ana["H_complex"] = ana["H_complex"][:, :-1]
    with pytest.raises(ValueError, match="H_complex shape mismatch"):
        write_room_to_h5(tmp_path / "x.h5", ism, ana)
