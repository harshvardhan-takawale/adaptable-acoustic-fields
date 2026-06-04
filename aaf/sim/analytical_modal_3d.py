"""Analytical 3D rectangular-room modal model.

3D port of `aaf.sim.analytical_modal_2d`. Independent reference for
cross-checking pyroomacoustics 3D ISM. **Does not import pyroomacoustics** —
exists precisely so the two pipelines disagree or agree on their own merits.

Conventions
-----------
- Rigid-wall (Neumann) boundary conditions: mode shapes are pure cosines.
- Mode shapes:
      Φ_{n_x, n_y, n_z}(x, y, z) = cos(n_x π x / L) · cos(n_y π y / W) · cos(n_z π z / H)
- Mode frequencies:
      f_{n_x, n_y, n_z} = (c / 2) · sqrt((n_x/L)² + (n_y/W)² + (n_z/H)²)
- Modal damping (Sabine 3D, mode-independent):
      γ = c · α · S / (8 · V)   [units: 1/s]
  where S = 2(LW + LH + WH) is total wall area, V = L·W·H is volume. This is
  the pressure-amplitude e-folding rate; T60 ≈ 6.91 / γ.
- Modal sum (frequency-domain Green's function with Lorentzian per-mode):
      H(r_rx, r_tx, ω) = Σ_m Φ_m(r_tx) · Φ_m(r_rx) / (k_m² - k² - 2 j γ k_m / c)
  with k = ω/c. Numerator omits volume normalization; only relative magnitudes
  matter for peak picking.

3D vs 2D: the modal density at any frequency f scales as f² in 3D (vs f in
2D), so for the same room you'll see ~30× more modes ≤ 2 kHz than 2D. Modal
MAE is only meaningful below f_Schroeder; above that the density exceeds the
RFFT resolution (Δf = 0.5 Hz at fs=4096, n_time=8192).

References
----------
- Kuttruff, "Room Acoustics" (5th ed.), §3.2 (eigenfrequencies of a
  rectangular room) and §3.4 (modal damping in the diffuse field limit).
- Pierce, "Acoustics" (3rd ed.), §10 (Helmholtz resonator and modal sums).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


C_DEFAULT = 343.0


@dataclass
class EigenFreq3D:
    """One distinct 3D eigenfrequency, possibly with multiple (n_x, n_y, n_z)
    modes sharing it (e.g., (1,0,0), (0,1,0), (0,0,1) in a cubic room).
    """
    f: float                                              # Hz
    multiplicity: int                                     # number of triples at this freq
    triples: list[tuple[int, int, int]]                   # contributing (n_x, n_y, n_z)


def _enumerate_triples(
    L: float, W: float, H: float, c: float, f_max: float
) -> list[tuple[int, int, int, float]]:
    """Internal: every (n_x, n_y, n_z, f) with f ≤ f_max. No dedup. Sorted by f."""
    if L <= 0 or W <= 0 or H <= 0:
        raise ValueError(f"L, W, H must be positive, got L={L}, W={W}, H={H}")
    if c <= 0 or f_max <= 0:
        raise ValueError(f"c and f_max must be positive, got c={c}, f_max={f_max}")

    n_x_max = int(np.floor(2 * f_max * L / c))
    n_y_max = int(np.floor(2 * f_max * W / c))
    n_z_max = int(np.floor(2 * f_max * H / c))

    triples: list[tuple[int, int, int, float]] = []
    for n_x in range(n_x_max + 1):
        for n_y in range(n_y_max + 1):
            for n_z in range(n_z_max + 1):
                f = (c / 2.0) * np.sqrt(
                    (n_x / L) ** 2 + (n_y / W) ** 2 + (n_z / H) ** 2
                )
                if f <= f_max:
                    triples.append((n_x, n_y, n_z, float(f)))
    triples.sort(key=lambda t: (t[3], t[0], t[1], t[2]))
    return triples


def eigenfrequencies_3d(
    L: float,
    W: float,
    H: float,
    c: float = C_DEFAULT,
    f_max: float = 2000.0,
    dedup_tol_hz: float = 0.01,
) -> list[EigenFreq3D]:
    """Enumerate distinct 3D eigenfrequencies up to ``f_max``, deduplicated.

    Modes that share a frequency to within ``dedup_tol_hz`` collapse into one
    ``EigenFreq3D`` entry; ``multiplicity`` counts them and ``triples`` lists
    the underlying ``(n_x, n_y, n_z)`` tuples. The (0, 0, 0) DC entry is
    included so callers can drop it explicitly.

    Sorted by ``f`` ascending. Within an entry, ``triples`` is sorted
    lexicographically.

    Handles cubic-room degeneracy correctly: at L=W=H, modes (1,0,0), (0,1,0),
    (0,0,1) collapse to one entry with multiplicity=3.
    """
    raw = _enumerate_triples(L=L, W=W, H=H, c=c, f_max=f_max)
    out: list[EigenFreq3D] = []
    for n_x, n_y, n_z, f in raw:
        if out and abs(f - out[-1].f) <= dedup_tol_hz:
            out[-1].triples.append((n_x, n_y, n_z))
            out[-1].multiplicity += 1
        else:
            out.append(EigenFreq3D(f=f, multiplicity=1, triples=[(n_x, n_y, n_z)]))
    for e in out:
        e.triples.sort()
    return out


def _triple_shape(
    n_x: int, n_y: int, n_z: int,
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    L: float, W: float, H: float,
) -> np.ndarray:
    """Φ_{n_x, n_y, n_z}(x, y, z) = cos(n_x π x / L) · cos(n_y π y / W) · cos(n_z π z / H).

    Broadcast: if x, y, z are arrays of equal shape, result has the same shape.
    """
    return (
        np.cos(n_x * np.pi * x / L)
        * np.cos(n_y * np.pi * y / W)
        * np.cos(n_z * np.pi * z / H)
    )


def sabine_damping_3d(
    L: float, W: float, H: float, alpha: float, c: float = C_DEFAULT
) -> float:
    """Mode-independent Sabine 3D damping in 1/s.

    γ = c α S / (8 V),  with S = 2(LW + LH + WH), V = LWH.

    Returns the pressure-amplitude e-folding rate; T60 ≈ 6.91 / γ.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    S = 2.0 * (L * W + L * H + W * H)
    V = L * W * H
    return c * alpha * S / (8.0 * V)


def modal_rir_3d(cfg: dict) -> dict:
    """Synthesize H(f) and rir(t) by summing 3D rectangular-room eigenmodes.

    Same return signature as ``aaf.sim.ism_3d.simulate_room_3d``.

    Args (cfg keys):
        L, W, H: room dims (m)
        source_pos: (3,) position of point source in m
        receiver_pos: (N, 3) receiver positions
        alpha: wall absorption coefficient (uniform, all 6 walls)
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
    H = float(cfg["H"])
    source_pos = np.asarray(cfg["source_pos"], dtype=np.float64).reshape(3)
    receiver_pos = np.asarray(cfg["receiver_pos"], dtype=np.float64).reshape(-1, 3)
    alpha = float(cfg["alpha"])
    fs = float(cfg["fs"])
    n_time_samples = int(cfg["n_time_samples"])
    c = float(cfg.get("c", C_DEFAULT))
    f_max_modes = float(cfg.get("f_max_modes", fs / 2.0))

    n_freq_bins = n_time_samples // 2 + 1
    f_axis = np.arange(n_freq_bins) * (fs / n_time_samples)  # Hz, RFFT axis

    # Iterate individual (n_x, n_y, n_z) modes — each contributes its own term
    # to the modal sum, even when degenerate. The dedup'd `eigenfrequencies_3d`
    # is the right interface for the *probe* (distinct freqs); the modal sum
    # needs the full enumeration.
    triples = _enumerate_triples(L=L, W=W, H=H, c=c, f_max=f_max_modes)
    triples = [t for t in triples if t[3] > 0]  # drop (0,0,0) DC
    n_distinct_freqs = len(
        eigenfrequencies_3d(L=L, W=W, H=H, c=c, f_max=f_max_modes)
    )

    gamma = sabine_damping_3d(L=L, W=W, H=H, alpha=alpha, c=c)  # 1/s

    src_x, src_y, src_z = source_pos
    n_rx = receiver_pos.shape[0]
    H_acc = np.zeros((n_rx, n_freq_bins), dtype=np.complex128)

    omega = 2 * np.pi * f_axis  # rad/s
    k = omega / c
    eps = 1e-12

    for n_x, n_y, n_z, f_m in triples:
        k_m = 2 * np.pi * f_m / c

        phi_src = float(
            np.cos(n_x * np.pi * src_x / L)
            * np.cos(n_y * np.pi * src_y / W)
            * np.cos(n_z * np.pi * src_z / H)
        )
        phi_rx = (
            np.cos(n_x * np.pi * receiver_pos[:, 0] / L)
            * np.cos(n_y * np.pi * receiver_pos[:, 1] / W)
            * np.cos(n_z * np.pi * receiver_pos[:, 2] / H)
        )

        denom = k_m**2 - k**2 - 2j * gamma * k_m / c + eps
        amp = phi_src * phi_rx  # shape (N_rx,)
        H_acc += amp[:, None] / denom[None, :]

    # Force DC + Nyquist to be real (RFFT convention).
    H_acc[:, 0] = H_acc[:, 0].real
    if n_time_samples % 2 == 0:
        H_acc[:, -1] = H_acc[:, -1].real

    H_complex = H_acc.astype(np.complex64)
    rir_time = np.fft.irfft(H_complex, n=n_time_samples, axis=1).astype(np.float32)

    meta = {
        "model": "analytical_modal_3d",
        "L": L,
        "W": W,
        "H": H,
        "alpha": alpha,
        "fs": fs,
        "n_time_samples": n_time_samples,
        "n_freq_bins": n_freq_bins,
        "c": c,
        "source_pos": source_pos.tolist(),
        "receiver_pos": receiver_pos.tolist(),
        "n_modes": len(triples),                  # individual (n_x, n_y, n_z) terms summed
        "n_distinct_freqs": n_distinct_freqs,     # entries in dedup'd EigenFreq3D list
        "f_max_modes": f_max_modes,
        "gamma_sabine_3d": float(gamma),
        "T60_from_gamma": float(6.91 / gamma),
        "note": (
            "Lorentzian modal sum with rigid-wall cosine modes and Sabine 3D "
            "uniform damping; numerator omits volume normalization."
        ),
    }
    return {"rir_time": rir_time, "H_complex": H_complex, "meta": meta}
