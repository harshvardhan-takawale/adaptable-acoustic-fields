"""Pyroomacoustics 3D ShoeBox ISM wrapper.

3D port of `aaf.sim.ism_2d`. Same interface — a single ``simulate_room_3d(cfg)``
entry point returning ``{rir_time, H_complex, meta}`` — adapted for genuine
3D shoebox geometry.

Key 3D-specific decisions (Chunk P2-1):
  - max_order is *hard-capped* at MAX_ORDER_CAP=17. The 2D auto-rule
    `ceil(c·4·T60 / min(L, W))` blows up to ~478 in 3D (~10⁸ image sources →
    OOM). At max_order=17 we get ~42 K images / 5-15 s per room on 4 CPUs;
    the IR covers ~108 ms (4× the 50 ms early-reflection envelope), which is
    sufficient for de-risk single-room overfit. Late-field tail beyond ~108
    ms is under-represented — `analytical_modal_3d` is the deterministic
    modal reference for low-freq accuracy.
  - NO `set_ray_tracing` in P2-1. Stochastic tail breaks array-task
    idempotency; analytical modal ground truth assumes deterministic ISM.
  - Schroeder frequency uses the *genuine* 3D form ``f_s = 2000·√(T60/V)``
    with ``V = L·W·H`` — no longer a 2D-slab proxy.
"""
from __future__ import annotations

import math
import warnings
from importlib import metadata as _md

import numpy as np
import pyroomacoustics as pra


C_DEFAULT = 343.0
MAX_ORDER_CAP = 17  # see DECISIONS.md (D6): 3D ISM tractability cap


def _versions() -> dict:
    """Snapshot of relevant package versions for provenance."""
    out = {}
    for pkg in ("pyroomacoustics", "numpy", "scipy", "h5py"):
        try:
            out[pkg] = _md.version(pkg)
        except _md.PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def _compute_max_order(L: float, W: float, H: float, T60: float, c: float) -> int:
    """ISM order targeting ~4·T60 coverage, capped at MAX_ORDER_CAP.

    Unbounded rule: ``ceil(c · 4·T60 / min(L, W, H))``. In 3D this can exceed
    400 for typical rooms (image-source tree explodes as (2N+1)³). We cap at
    MAX_ORDER_CAP=17 to keep simulation tractable (~42K image sources, IR
    covers ~108 ms before truncation).
    """
    raw = int(math.ceil(c * 4.0 * T60 / min(L, W, H)))
    return min(raw, MAX_ORDER_CAP)


def _validate_inside(
    pos: np.ndarray, L: float, W: float, H: float, name: str, margin: float = 0.1
):
    pos = np.atleast_2d(pos)
    if pos.shape[-1] != 3:
        raise ValueError(f"{name} must be 3D, got shape {pos.shape}")
    bad = (
        (pos[:, 0] < margin)
        | (pos[:, 0] > L - margin)
        | (pos[:, 1] < margin)
        | (pos[:, 1] > W - margin)
        | (pos[:, 2] < margin)
        | (pos[:, 2] > H - margin)
    )
    if bad.any():
        raise ValueError(
            f"{name} has {int(bad.sum())} point(s) outside [{margin}, L-{margin}] × "
            f"[{margin}, W-{margin}] × [{margin}, H-{margin}] for "
            f"L={L} W={W} H={H}: {pos[bad]}"
        )


def simulate_room_3d(cfg: dict) -> dict:
    """Simulate a 3D shoebox via pyroomacoustics ISM, return time + freq IRs.

    Args (cfg keys):
        L, W, H: room dims (m). Floats.
        source_pos: (3,) position of the omni point source (m).
        receiver_pos: (N_rx, 3) receiver positions (m).
        alpha: wall absorption coefficient (uniform, all 6 walls), in (0, 1).
        fs: sampling rate (Hz).
        n_time_samples: target IR length. Output IR is truncated/padded to this.
        max_order: optional ISM max image-source order. If None, auto-chosen
                   from 4·T60 coverage (capped at MAX_ORDER_CAP=17).
        c: speed of sound. Defaults to 343.

    Returns:
        {
          "rir_time":  (N_rx, n_time_samples)   float32,
          "H_complex": (N_rx, n_freq_bins)      complex64,
          "meta": {
              L, W, H, alpha, fs, c,
              source_pos, receiver_pos,
              n_time_samples, n_freq_bins,
              max_order, max_order_was_auto, max_order_was_capped,
              T60_sabine_3d, schroeder_freq_hz,
              ir_pra_lengths, ir_pra_length_max, ir_truncated, ir_padded,
              versions: {pyroomacoustics, numpy, scipy, h5py},
          }
        }

    Raises:
        ValueError if room.dim != 3 or any source/receiver lies outside the
        room with margin 0.1 m.
    """
    L = float(cfg["L"])
    W = float(cfg["W"])
    H = float(cfg["H"])
    source_pos = np.asarray(cfg["source_pos"], dtype=np.float64).reshape(3)
    receiver_pos = np.asarray(cfg["receiver_pos"], dtype=np.float64).reshape(-1, 3)
    alpha = float(cfg["alpha"])
    fs = float(cfg["fs"])
    n_time_samples = int(cfg["n_time_samples"])
    c = float(cfg.get("c", C_DEFAULT))
    max_order_user = cfg.get("max_order", None)

    n_freq_bins = n_time_samples // 2 + 1

    _validate_inside(source_pos, L, W, H, "source_pos")
    _validate_inside(receiver_pos, L, W, H, "receiver_pos")

    # Build the room first with max_order=0 to query T60, then re-build with
    # the chosen max_order. (Same pattern as ism_2d.)
    materials = pra.Material(energy_absorption=alpha)
    room = pra.ShoeBox(
        p=[L, W, H], fs=int(fs), materials=materials, max_order=0, ray_tracing=False
    )
    if room.dim != 3:
        raise RuntimeError(
            f"pra.ShoeBox(p=[L, W, H]) gave room.dim={room.dim}, expected 3. "
            "pyroomacoustics may have changed its 3D path."
        )
    T60_sabine_3d = float(room.rt60_theory(formula="sabine"))

    max_order_was_auto = max_order_user is None
    if max_order_was_auto:
        raw = int(math.ceil(c * 4.0 * T60_sabine_3d / min(L, W, H)))
        max_order = min(raw, MAX_ORDER_CAP)
        max_order_was_capped = raw > MAX_ORDER_CAP
    else:
        max_order = int(max_order_user)
        max_order_was_capped = False

    room = pra.ShoeBox(
        p=[L, W, H],
        fs=int(fs),
        materials=materials,
        max_order=max_order,
        ray_tracing=False,
    )
    if room.dim != 3:
        raise RuntimeError(f"second build gave room.dim={room.dim}")

    room.add_source(source_pos.tolist())
    # add_microphone_array accepts (3, N); we have (N, 3).
    mic_array = pra.MicrophoneArray(receiver_pos.T, fs=int(fs))
    room.add_microphone_array(mic_array)

    if n_time_samples < 4.0 * T60_sabine_3d * fs:
        warnings.warn(
            f"n_time_samples={n_time_samples} < 4 · T60 · fs = "
            f"{int(4.0 * T60_sabine_3d * fs)}; IR will be truncated below noise floor.",
            stacklevel=2,
        )

    room.compute_rir()

    n_rx = receiver_pos.shape[0]
    rir_time = np.zeros((n_rx, n_time_samples), dtype=np.float32)
    ir_pra_lengths = []
    truncated = False
    padded = False
    for r_idx in range(n_rx):
        ir = np.asarray(room.rir[r_idx][0], dtype=np.float32)
        ir_pra_lengths.append(int(ir.size))
        n = ir.size
        if n >= n_time_samples:
            rir_time[r_idx] = ir[:n_time_samples]
            if n > n_time_samples:
                truncated = True
        else:
            rir_time[r_idx, :n] = ir
            padded = True

    H_complex = np.fft.rfft(rir_time, n=n_time_samples, axis=1).astype(np.complex64)

    # Genuine 3D Schroeder frequency: f_s = 2000·√(T60/V), V = L·W·H.
    V = L * W * H
    schroeder_hz = float(2000.0 * math.sqrt(T60_sabine_3d / max(V, 1e-9)))

    meta = {
        "model": "pyroomacoustics_ism_3d",
        "L": L,
        "W": W,
        "H": H,
        "alpha": alpha,
        "fs": fs,
        "c": c,
        "source_pos": source_pos.tolist(),
        "receiver_pos": receiver_pos.tolist(),
        "n_time_samples": n_time_samples,
        "n_freq_bins": n_freq_bins,
        "max_order": max_order,
        "max_order_was_auto": bool(max_order_was_auto),
        "max_order_was_capped": bool(max_order_was_capped),
        "max_order_cap": int(MAX_ORDER_CAP),
        "T60_sabine_3d": T60_sabine_3d,
        "schroeder_freq_hz": schroeder_hz,
        "ir_pra_lengths": ir_pra_lengths,
        "ir_pra_length_max": int(max(ir_pra_lengths)) if ir_pra_lengths else 0,
        "ir_truncated": bool(truncated),
        "ir_padded": bool(padded),
        "versions": _versions(),
    }
    return {"rir_time": rir_time, "H_complex": H_complex, "meta": meta}
