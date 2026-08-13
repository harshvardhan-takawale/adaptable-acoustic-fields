"""Pyroomacoustics 2D ShoeBox ISM wrapper.

The library is 3D-first, but `ShoeBox(p=[L, W])` constructs a true 2D room
with a dedicated `libroom.Room2D` C++ engine. `room.dim == 2` afterwards;
ISM is dimension-agnostic.

T60 is computed via `room.rt60_theory(formula='sabine')`, which is
dimension-aware: in 2D it uses `S = 2(L+W)` (perimeter) and `sab_coef = 12`.

Schroeder frequency in 2D is not a standard quantity. We compute the 3D form
with `V = A · 1m` (treating the 2D room as a 1-m-thick slab) and label it
"approximate" in metadata. It is not used for any logic.
"""
from __future__ import annotations

import math
import warnings
from importlib import metadata as _md

import numpy as np
import pyroomacoustics as pra

from aaf.walls import WALLS_2D


C_DEFAULT = 343.0


def _alpha_eff_2d(L: float, W: float, alphas) -> float:
    """Perimeter-weighted mean absorption for per-wall configs.

    west/east span W; south/north span L (WALLS_2D order). Used so ``meta['alpha']``
    stays a float for legacy readers and so Sabine/Eyring references remain meaningful.
    NOTE this scalar deliberately CANNOT distinguish which wall was edited -- that is the
    confound P3-2's held-out combos are designed to expose, not a quantity to model with.
    """
    a_w, a_e, a_s, a_n = (float(a) for a in alphas)
    return float((a_w * W + a_e * W + a_s * L + a_n * L) / (2.0 * (L + W)))


def _versions() -> dict:
    """Snapshot of relevant package versions for provenance."""
    out = {}
    for pkg in ("pyroomacoustics", "numpy", "scipy", "h5py"):
        try:
            out[pkg] = _md.version(pkg)
        except _md.PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def _compute_max_order(L: float, W: float, T60: float, c: float) -> int:
    """Pick max_order so the ISM image-source tree covers ~4·T60 of decay.

    Rule (cf. pyroomacoustics/acoustics.py:576):
        max_order = ceil(c · 4·T60 / min(L, W))
    """
    return int(math.ceil(c * 4.0 * T60 / min(L, W)))


def _validate_inside(pos: np.ndarray, L: float, W: float, name: str, margin: float = 0.1):
    pos = np.atleast_2d(pos)
    if pos.shape[-1] != 2:
        raise ValueError(f"{name} must be 2D, got shape {pos.shape}")
    bad = (
        (pos[:, 0] < margin)
        | (pos[:, 0] > L - margin)
        | (pos[:, 1] < margin)
        | (pos[:, 1] > W - margin)
    )
    if bad.any():
        raise ValueError(
            f"{name} has {int(bad.sum())} point(s) outside [{margin}, L-{margin}] × "
            f"[{margin}, W-{margin}] for L={L} W={W}: {pos[bad]}"
        )


def simulate_room_2d(cfg: dict) -> dict:
    """Simulate a 2D shoebox via pyroomacoustics ISM, return time + freq IRs.

    Args (cfg keys):
        L, W: room dims (m). Floats.
        source_pos: (2,) position of the omni point source (m).
        receiver_pos: (N_rx, 2) receiver positions (m).
        alpha: wall absorption coefficient (uniform, all 4 walls), in (0, 1).
        fs: sampling rate (Hz).
        n_time_samples: target IR length. Output IR is truncated/padded to this.
        max_order: optional ISM max image-source order. If None, auto-chosen
                   from 4·T60 coverage.
        c: speed of sound. Defaults to 343.

    Returns:
        {
          "rir_time":  (N_rx, n_time_samples)   float32,
          "H_complex": (N_rx, n_freq_bins)      complex64,
          "meta": {
              L, W, alpha, fs, c,
              source_pos, receiver_pos,
              n_time_samples, n_freq_bins,
              max_order, T60_sabine_2d, schroeder_freq_approx_hz,
              ir_pra_length, ir_truncated, ir_padded,
              versions: {pyroomacoustics, numpy, scipy, h5py},
          }
        }

    Raises:
        ValueError if room.dim != 2 or any source/receiver lies outside the room
        with margin 0.1 m.
    """
    L = float(cfg["L"])
    W = float(cfg["W"])
    source_pos = np.asarray(cfg["source_pos"], dtype=np.float64).reshape(2)
    receiver_pos = np.asarray(cfg["receiver_pos"], dtype=np.float64).reshape(-1, 2)
    fs = float(cfg["fs"])
    n_time_samples = int(cfg["n_time_samples"])
    c = float(cfg.get("c", C_DEFAULT))
    max_order_user = cfg.get("max_order", None)

    # Per-wall absorption (P3-2). ``alphas`` is a 4-sequence in aaf.walls.WALLS_2D order
    # (west, east, south, north); when absent the legacy scalar ``alpha`` applies to all
    # four walls. Back-compat is exact, not approximate: pra's ShoeBox does
    # ``materials = dict(zip(wall_names, [material] * n_walls))`` for a single Material, so
    # the uniform-dict path below is the same code path as the scalar one.
    alphas_user = cfg.get("alphas", None)
    if alphas_user is None:
        alpha = float(cfg["alpha"])
        alphas = (alpha,) * 4
    else:
        alphas = tuple(float(a) for a in alphas_user)
        if len(alphas) != 4:
            raise ValueError(
                f"cfg['alphas'] must have 4 entries in WALLS_2D order {list(WALLS_2D)}, "
                f"got {len(alphas)}"
            )
        # Perimeter-weighted mean, so meta['alpha'] stays a float for every legacy reader
        # (write_room_to_h5 and every attrs['alpha'] consumer) and remains the right
        # scalar for Sabine/Eyring references.
        alpha = _alpha_eff_2d(L, W, alphas)

    n_freq_bins = n_time_samples // 2 + 1

    _validate_inside(source_pos, L, W, "source_pos")
    _validate_inside(receiver_pos, L, W, "receiver_pos")

    # Build the room. We pass max_order=0 first to get a valid Room object so we
    # can call rt60_theory; then re-build with the correct max_order.
    if alphas_user is None:
        materials = pra.Material(energy_absorption=alpha)
    else:
        materials = {w: pra.Material(energy_absorption=a)
                     for w, a in zip(WALLS_2D, alphas)}
    room = pra.ShoeBox(
        p=[L, W], fs=int(fs), materials=materials, max_order=0, ray_tracing=False
    )
    if room.dim != 2:
        raise RuntimeError(
            f"pra.ShoeBox(p=[L, W]) gave room.dim={room.dim}, expected 2. "
            "pyroomacoustics may have changed its 2D path."
        )
    T60_sabine_2d = float(room.rt60_theory(formula="sabine"))

    if max_order_user is None:
        max_order = _compute_max_order(L=L, W=W, T60=T60_sabine_2d, c=c)
    else:
        max_order = int(max_order_user)

    # Re-build with the chosen max_order. (pra's `change_max_order` would also work
    # but rebuilding is simpler and avoids hidden state.)
    room = pra.ShoeBox(
        p=[L, W], fs=int(fs), materials=materials, max_order=max_order, ray_tracing=False
    )
    if room.dim != 2:
        raise RuntimeError(f"second build gave room.dim={room.dim}")

    room.add_source(source_pos.tolist())
    # add_microphone_array accepts a (2, N) array; we have (N, 2).
    mic_array = pra.MicrophoneArray(receiver_pos.T, fs=int(fs))
    room.add_microphone_array(mic_array)

    if n_time_samples < 4.0 * T60_sabine_2d * fs:
        warnings.warn(
            f"n_time_samples={n_time_samples} < 4 · T60 · fs = "
            f"{int(4.0 * T60_sabine_2d * fs)}; IR will be truncated below noise floor.",
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

    # Approximate Schroeder freq using the 3D form with V = A * 1 m.
    A = L * W
    V_proxy = A * 1.0
    schroeder_hz = float(2000.0 * math.sqrt(T60_sabine_2d / max(V_proxy, 1e-9)))

    meta = {
        "model": "pyroomacoustics_ism_2d",
        "L": L,
        "W": W,
        "alpha": alpha,
        "alphas": [float(a) for a in alphas],
        "alpha_per_wall": {w: float(a) for w, a in zip(WALLS_2D, alphas)},
        "walls": list(WALLS_2D),
        "alpha_is_effective": alphas_user is not None,
        "fs": fs,
        "c": c,
        "source_pos": source_pos.tolist(),
        "receiver_pos": receiver_pos.tolist(),
        "n_time_samples": n_time_samples,
        "n_freq_bins": n_freq_bins,
        "max_order": max_order,
        "max_order_was_auto": max_order_user is None,
        "T60_sabine_2d": T60_sabine_2d,
        "schroeder_freq_approx_hz": schroeder_hz,
        "ir_pra_lengths": ir_pra_lengths,
        "ir_pra_length_max": int(max(ir_pra_lengths)) if ir_pra_lengths else 0,
        "ir_truncated": bool(truncated),
        "ir_padded": bool(padded),
        "versions": _versions(),
    }
    return {"rir_time": rir_time, "H_complex": H_complex, "meta": meta}
