"""Smoke tests for the 3D ISM wrapper."""
import numpy as np
import pytest

from aaf.sim.ism_3d import MAX_ORDER_CAP, simulate_room_3d


def _make_cfg(L=4.0, W=3.5, H=2.8, n_time=4096, fs=4096):
    """Small but valid 3D shoebox cfg for smoke tests."""
    rxs = np.array(
        [[1.0, 1.0, 1.0], [2.0, 1.5, 1.5], [3.0, 2.0, 2.0]],
        dtype=np.float64,
    )
    return {
        "L": L, "W": W, "H": H,
        "source_pos": (0.5, 0.5, 0.5),
        "receiver_pos": rxs,
        "alpha": 0.15,
        "fs": fs,
        "n_time_samples": n_time,
    }


def test_simulate_room_3d_shape_and_finite():
    cfg = _make_cfg()
    out = simulate_room_3d(cfg)
    rir = out["rir_time"]
    H = out["H_complex"]
    meta = out["meta"]
    assert rir.shape == (3, cfg["n_time_samples"])
    assert H.shape == (3, cfg["n_time_samples"] // 2 + 1)
    assert rir.dtype == np.float32
    assert H.dtype == np.complex64
    assert np.all(np.isfinite(rir))
    assert np.all(np.isfinite(H.real)) and np.all(np.isfinite(H.imag))
    # Meta sanity
    assert meta["L"] == cfg["L"]
    assert meta["W"] == cfg["W"]
    assert meta["H"] == cfg["H"]
    assert meta["model"] == "pyroomacoustics_ism_3d"


def test_simulate_room_3d_dc_real():
    """DC bin's imaginary part must be ~0 (rfft of real signal)."""
    out = simulate_room_3d(_make_cfg())
    assert np.allclose(out["H_complex"][:, 0].imag, 0, atol=1e-5)


def test_simulate_room_3d_nyquist_real():
    """Nyquist bin's imaginary part must be ~0 for even-length input."""
    out = simulate_room_3d(_make_cfg(n_time=4096))
    assert np.allclose(out["H_complex"][:, -1].imag, 0, atol=1e-5)


def test_max_order_cap_in_meta():
    """When auto, max_order should be bounded by MAX_ORDER_CAP."""
    cfg = _make_cfg(L=6.0, W=5.0, H=4.0)  # bigger room → would request high order
    out = simulate_room_3d(cfg)
    meta = out["meta"]
    assert meta["max_order"] <= MAX_ORDER_CAP
    assert meta["max_order_was_auto"] is True
    # T60 sanity for α=0.15 in a 120 m³ room: ~0.8-1 s.
    assert 0.3 < meta["T60_sabine_3d"] < 1.5
    # Schroeder sanity: low-end (~150-250 Hz typical).
    assert 50 < meta["schroeder_freq_hz"] < 400


def test_validate_inside_raises_on_oob_source():
    cfg = _make_cfg()
    cfg["source_pos"] = (10.0, 10.0, 10.0)  # outside the room
    with pytest.raises(ValueError):
        simulate_room_3d(cfg)


def test_validate_inside_raises_on_oob_receiver():
    cfg = _make_cfg()
    cfg["receiver_pos"] = np.array([[100.0, 100.0, 100.0]], dtype=np.float64)
    with pytest.raises(ValueError):
        simulate_room_3d(cfg)


def test_simulate_room_3d_custom_max_order():
    """When max_order_user is given, it's used verbatim (no cap)."""
    cfg = _make_cfg()
    cfg["max_order"] = 5
    out = simulate_room_3d(cfg)
    assert out["meta"]["max_order"] == 5
    assert out["meta"]["max_order_was_auto"] is False
