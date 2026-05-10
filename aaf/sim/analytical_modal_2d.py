"""Analytical 2D rectangular-room modal model.

Independent reference for cross-checking pyroomacoustics 2D ISM. **Does not
import pyroomacoustics** — exists precisely so the two pipelines disagree
or agree on their own merits.

Conventions
-----------
- Rigid-wall (Neumann) boundary conditions: mode shapes are pure cosines.
- Mode shapes:  Φ_{n_x, n_y}(x, y) = cos(n_x π x / L) · cos(n_y π y / W)
- Mode frequencies:  f_{n_x, n_y} = (c / 2) · sqrt((n_x / L)² + (n_y / W)²)
- Modal damping (Sabine 2D, mode-independent):
      γ = c · α · P / (4 · A)        [units: 1/s]
  where P = 2(L+W) is room perimeter, A = L·W is room area, α the wall
  absorption coefficient. This is the e-folding rate of pressure; T60 ≈
  6.91 / γ (the ratio between -60 dB and ln(2) drops, scaled by 2× since
  intensity falls twice as fast as amplitude — but Sabine is already in
  intensity units so this is consistent).
- Modal sum (frequency-domain Green's function with Lorentzian per-mode):
      H(r_rx, r_tx, ω) = Σ_m  Φ_m(r_tx) · Φ_m(r_rx)  /  (k_m² - k² - 2 j γ k_m / c)
  with k = ω/c. The sign convention puts the resonance peak at k = k_m
  (i.e., ω = ω_m). Numerator does NOT include the 1/V volume normalization
  used in 3D textbooks because we're solving a 2D problem with V→A and we
  treat the constant as part of the model — only relative magnitudes matter
  for peak picking.

References
----------
- Kuttruff, "Room Acoustics" (5th ed.), §3.2 (eigenfrequencies of a
  rectangular room) and §3.4 (modal damping in the diffuse field limit).
- Pierce, "Acoustics" (3rd ed.), §10 (Helmholtz resonator and modal sums).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


C_DEFAULT = 343.0


@dataclass
class EigenFreq:
    """One distinct eigenfrequency, possibly with multiple (n_x, n_y) modes
    sharing it (e.g., (1, 0) and (0, 1) at L=W).
    """
    f: float                                # Hz
    multiplicity: int                       # number of (n_x, n_y) pairs at this freq
    pairs: list[tuple[int, int]]            # (n_x, n_y) for each contributing mode


def _enumerate_pairs(L: float, W: float, c: float, f_max: float) -> list[tuple[int, int, float]]:
    """Internal: every (n_x, n_y, f) with f ≤ f_max. No deduplication. Sorted by f."""
    if L <= 0 or W <= 0:
        raise ValueError(f"L and W must be positive, got L={L}, W={W}")
    if c <= 0 or f_max <= 0:
        raise ValueError(f"c and f_max must be positive, got c={c}, f_max={f_max}")

    n_x_max = int(np.floor(2 * f_max * L / c))
    n_y_max = int(np.floor(2 * f_max * W / c))

    pairs: list[tuple[int, int, float]] = []
    for n_x in range(n_x_max + 1):
        for n_y in range(n_y_max + 1):
            f = (c / 2.0) * np.sqrt((n_x / L) ** 2 + (n_y / W) ** 2)
            if f <= f_max:
                pairs.append((n_x, n_y, float(f)))
    pairs.sort(key=lambda t: (t[2], t[0], t[1]))
    return pairs


def eigenfrequencies_2d(
    L: float,
    W: float,
    c: float = C_DEFAULT,
    f_max: float = 2000.0,
    dedup_tol_hz: float = 0.01,
) -> list[EigenFreq]:
    """Enumerate distinct 2D eigenfrequencies up to ``f_max``, deduplicated.

    Modes that share a frequency to within ``dedup_tol_hz`` collapse into one
    ``EigenFreq`` entry; ``multiplicity`` counts them and ``pairs`` lists the
    underlying ``(n_x, n_y)`` tuples. The (0, 0) DC entry is included so callers
    can drop it explicitly (it has multiplicity 1, pairs=[(0, 0)]).

    Sorted by ``f`` ascending. Within an entry, ``pairs`` is sorted lexicographically.
    """
    raw = _enumerate_pairs(L=L, W=W, c=c, f_max=f_max)
    out: list[EigenFreq] = []
    for n_x, n_y, f in raw:
        if out and abs(f - out[-1].f) <= dedup_tol_hz:
            out[-1].pairs.append((n_x, n_y))
            out[-1].multiplicity += 1
        else:
            out.append(EigenFreq(f=f, multiplicity=1, pairs=[(n_x, n_y)]))
    for e in out:
        e.pairs.sort()
    return out


def _pair_shape(
    n_x: int, n_y: int, x: np.ndarray, y: np.ndarray, L: float, W: float
) -> np.ndarray:
    """Φ_{n_x, n_y}(x, y) = cos(n_x π x / L) · cos(n_y π y / W).

    Broadcast: if x and y are arrays of equal shape, result has the same shape.
    """
    return np.cos(n_x * np.pi * x / L) * np.cos(n_y * np.pi * y / W)


def sabine_damping_2d(L: float, W: float, alpha: float, c: float = C_DEFAULT) -> float:
    """Mode-independent Sabine damping in 1/s for 2D rooms.

    γ = c α P / (4 A),  with P = 2(L+W), A = LW.

    Returns the e-folding rate; T60 ≈ 6.91 / γ.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    P = 2.0 * (L + W)
    A = L * W
    return c * alpha * P / (4.0 * A)


def modal_rir_2d(cfg: dict) -> dict:
    """Synthesize H(f) and rir(t) by summing 2D rectangular-room eigenmodes.

    Same return signature as ``aaf.sim.ism_2d.simulate_room_2d``.

    Args (cfg keys):
        L, W: room dims (m)
        source_pos: (2,) position of point source in m, components in (0, L)x(0, W)
        receiver_pos: (N, 2) array of receiver positions
        alpha: wall absorption coefficient (uniform, all 4 walls)
        fs: sampling rate (Hz). Used to pick irfft length.
        n_time_samples: target IR length. n_freq_bins = n_time_samples // 2 + 1.
        c: speed of sound. Optional, defaults to 343.
        f_max_modes: optional cap on mode frequency to include. Defaults to fs/2.

    Returns:
        {
          "rir_time":  (N_rx, n_time_samples) float32,
          "H_complex": (N_rx, n_freq_bins) complex64,
          "meta": {...}
        }
    """
    L = float(cfg["L"])
    W = float(cfg["W"])
    source_pos = np.asarray(cfg["source_pos"], dtype=np.float64).reshape(2)
    receiver_pos = np.asarray(cfg["receiver_pos"], dtype=np.float64).reshape(-1, 2)
    alpha = float(cfg["alpha"])
    fs = float(cfg["fs"])
    n_time_samples = int(cfg["n_time_samples"])
    c = float(cfg.get("c", C_DEFAULT))
    f_max_modes = float(cfg.get("f_max_modes", fs / 2.0))

    n_freq_bins = n_time_samples // 2 + 1
    f_axis = np.arange(n_freq_bins) * (fs / n_time_samples)  # Hz, RFFT axis

    # Iterate over individual (n_x, n_y) modes — each contributes its own term
    # to the modal sum, even when degenerate. The dedup'd `eigenfrequencies_2d`
    # is the right interface for the *probe* (we want distinct freqs there);
    # the modal sum needs the full enumeration.
    pairs = _enumerate_pairs(L=L, W=W, c=c, f_max=f_max_modes)
    pairs = [p for p in pairs if p[2] > 0]  # drop (0,0) DC mode
    n_distinct_freqs = len(eigenfrequencies_2d(L=L, W=W, c=c, f_max=f_max_modes))

    gamma = sabine_damping_2d(L=L, W=W, alpha=alpha, c=c)  # 1/s

    src_x, src_y = source_pos
    n_rx = receiver_pos.shape[0]
    H = np.zeros((n_rx, n_freq_bins), dtype=np.complex128)

    omega = 2 * np.pi * f_axis  # rad/s
    k = omega / c
    eps = 1e-12

    for n_x, n_y, f_m in pairs:
        k_m = 2 * np.pi * f_m / c

        phi_src = _pair_shape(n_x, n_y, np.array(src_x), np.array(src_y), L, W)
        phi_rx = np.cos(n_x * np.pi * receiver_pos[:, 0] / L) * np.cos(
            n_y * np.pi * receiver_pos[:, 1] / W
        )

        # Lorentzian: H_m(ω) = Φ(r_tx) Φ(r_rx) / (k_m² - k² - 2 j γ k_m / c)
        denom = k_m**2 - k**2 - 2j * gamma * k_m / c + eps
        amp = phi_src * phi_rx  # shape (N_rx,)
        H += amp[:, None] / denom[None, :]

    # Force DC + Nyquist to be real (RFFT convention) — should already be real,
    # but float arithmetic can give tiny imag noise.
    H[:, 0] = H[:, 0].real
    if n_time_samples % 2 == 0:
        H[:, -1] = H[:, -1].real

    H_complex = H.astype(np.complex64)
    rir_time = np.fft.irfft(H_complex, n=n_time_samples, axis=1).astype(np.float32)

    meta = {
        "model": "analytical_modal_2d",
        "L": L,
        "W": W,
        "alpha": alpha,
        "fs": fs,
        "n_time_samples": n_time_samples,
        "n_freq_bins": n_freq_bins,
        "c": c,
        "source_pos": source_pos.tolist(),
        "receiver_pos": receiver_pos.tolist(),
        "n_modes": len(pairs),                  # individual (n_x, n_y) terms summed
        "n_distinct_freqs": n_distinct_freqs,   # entries in dedup'd EigenFreq list
        "f_max_modes": f_max_modes,
        "gamma_sabine_2d": float(gamma),
        "T60_from_gamma": float(6.91 / gamma),
        "note": (
            "Lorentzian modal sum with rigid-wall cosine modes and Sabine 2D "
            "uniform damping; numerator omits volume normalization."
        ),
    }
    return {"rir_time": rir_time, "H_complex": H_complex, "meta": meta}
