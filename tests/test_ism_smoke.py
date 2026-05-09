"""Smoke test: simulate one tiny 2D shoebox via pyroomacoustics ISM and assert
basic IR sanity. Tiny-room ISM is sub-second on a CPU; no slow marker needed.
"""
import numpy as np
import pytest

from aaf.sim.ism_2d import simulate_room_2d


def test_ism_2d_smoke_tiny_room():
    cfg = {
        "L": 2.0,
        "W": 2.0,
        "source_pos": (0.5, 0.5),
        "receiver_pos": np.array([[1.0, 1.0], [1.5, 0.7]]),
        "alpha": 0.20,
        "fs": 4096,
        "n_time_samples": 1024,
    }
    out = simulate_room_2d(cfg)

    assert out["rir_time"].shape == (2, 1024), f"unexpected rir_time shape {out['rir_time'].shape}"
    assert out["H_complex"].shape == (2, 513), f"unexpected H shape {out['H_complex'].shape}"
    assert out["rir_time"].dtype == np.float32
    assert out["H_complex"].dtype == np.complex64
    assert np.any(out["rir_time"] != 0), "all-zero IR — pyroomacoustics didn't emit anything"

    # DC bin must be real-valued (RFFT convention).
    assert np.all(np.imag(out["H_complex"][:, 0]) == 0), "DC bin has non-zero imag"

    # Conjugate-symmetric check on the full-FFT reconstruction (not the RFFT itself).
    full = np.fft.irfft(out["H_complex"], n=1024, axis=1)
    assert np.allclose(full, out["rir_time"], atol=1e-3), (
        "irfft(rfft(rir)) should round-trip to the original IR (modulo float noise)"
    )

    meta = out["meta"]
    assert meta["L"] == 2.0
    assert meta["W"] == 2.0
    assert 0.05 < meta["T60_sabine_2d"] < 5.0, (
        f"T60_sabine_2d {meta['T60_sabine_2d']} outside reasonable range"
    )
    assert meta["max_order"] >= 1, f"max_order {meta['max_order']} unreasonably small"
    assert meta["versions"]["pyroomacoustics"] != "unknown"
