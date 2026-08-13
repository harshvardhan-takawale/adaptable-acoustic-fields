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


def _ln_reflection(alpha: float) -> float:
    """ln R for the PRESSURE reflection coefficient R = sqrt(1 - alpha). ≤ 0."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    return 0.5 * np.log(1.0 - alpha)


def modal_damping_2d(
    L: float,
    W: float,
    alphas,
    n_x: int,
    n_y: int,
    c: float = C_DEFAULT,
    model: str = "ism_ray",
) -> float:
    """Per-mode damping rate γ (1/s) for PER-WALL absorptions — two competing laws.

    ``alphas`` is a 4-sequence of energy absorption coefficients in ``aaf.walls.WALLS_2D``
    order: (west, east, south, north) = (x=0, x=L, y=0, y=W).

    ``model="ism_ray"`` — what pyroomacoustics ISM actually computes. The reflection
    coefficient is real and angle-independent, so the x- and y-resonance conditions
    separate and the damping is the ray's bounce rate against each wall pair::

        γ = c · [ cosθ_x · κ_x  +  cosθ_y · κ_y ]
        κ_x = (−ln R_west − ln R_east) / (2L),   κ_y = (−ln R_south − ln R_north) / (2W)
        cosθ_x = (n_x/L) / √((n_x/L)² + (n_y/W)²)

    A pure x-axial mode has cosθ_y = 0 EXACTLY, so the south/north walls contribute zero
    damping: ISM contains no grazing-incidence absorption. Averaging over an isotropic 2D
    direction distribution (⟨|cosθ|⟩ = 2/π) recovers the 2D Eyring rate, which is how this
    expression was validated.

    ``model="kuttruff"`` — the locally-reacting impedance-wall (wave) result, in which a
    grazing wave still loses energy at its pressure antinode::

        γ = (c/8) · [ (α_w + α_e)·ε_n / L  +  (α_s + α_n)·ε_m / W ],   ε = 1 if index 0 else 2

    Reduces to ``sabine_damping_2d`` for uniform α and oblique modes. It predicts only a
    ~2:1 wall selectivity and NO invariant mode family.

    The two laws disagree by ~25× in selectivity; which one the ground truth obeys is the
    subject of the P3-2 physics gate (D47/D48). Reported alongside every predicted matrix
    so the chunk's claim is explicitly scoped to the simulator's law.

    Returns the e-folding rate of pressure, in 1/s. The corresponding −3 dB modal
    bandwidth is ``γ / π`` Hz (see ``damping_to_bandwidth_hz``).
    """
    a = [float(x) for x in alphas]
    if len(a) != 4:
        raise ValueError(f"alphas must have 4 entries (west, east, south, north), got {len(a)}")
    if L <= 0 or W <= 0:
        raise ValueError(f"L and W must be positive, got L={L}, W={W}")
    if n_x < 0 or n_y < 0:
        raise ValueError(f"mode indices must be non-negative, got ({n_x}, {n_y})")
    if n_x == 0 and n_y == 0:
        raise ValueError("the (0, 0) DC mode has no damping rate")
    a_w, a_e, a_s, a_n = a

    if model == "kuttruff":
        eps_n = 1.0 if n_x == 0 else 2.0
        eps_m = 1.0 if n_y == 0 else 2.0
        return float((c / 8.0) * ((a_w + a_e) * eps_n / L + (a_s + a_n) * eps_m / W))

    if model == "ism_ray":
        kx = n_x / L
        ky = n_y / W
        k = float(np.hypot(kx, ky))
        cos_x = kx / k
        cos_y = ky / k
        kappa_x = (-_ln_reflection(a_w) - _ln_reflection(a_e)) / (2.0 * L)
        kappa_y = (-_ln_reflection(a_s) - _ln_reflection(a_n)) / (2.0 * W)
        return float(c * (cos_x * kappa_x + cos_y * kappa_y))

    raise ValueError(f"unknown model {model!r}; expected 'ism_ray' or 'kuttruff'")


def damping_to_bandwidth_hz(gamma: float) -> float:
    """−3 dB bandwidth (Hz) of a Lorentzian modal peak with damping rate ``gamma`` (1/s).

    |H| ∝ 1/√((k_m−k)² + (γ/c)²) near resonance ⇒ half-power at |k_m−k| = γ/c ⇒ full
    width 2γ in angular frequency ⇒ γ/π in Hz.
    """
    return float(gamma) / np.pi


def bandwidth_hz_to_damping(bw_hz: float) -> float:
    """Inverse of :func:`damping_to_bandwidth_hz`."""
    return float(bw_hz) * np.pi


def eyring_damping_2d(L: float, W: float, alpha: float, c: float = C_DEFAULT) -> float:
    """Direction-averaged 2D Eyring damping, γ = c·P·(−ln(1−α)) / (2πA).

    The isotropic average of ``modal_damping_2d(..., model='ism_ray')`` with uniform α
    (⟨|cosθ|⟩ = 2/π). Unlike :func:`sabine_damping_2d` it accepts α = 0 and stays finite
    as α → 1, so it is the right reference for the low-absorption (concrete) configs.
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    P = 2.0 * (L + W)
    A = L * W
    return float(c * P * (-np.log(1.0 - alpha)) / (2.0 * np.pi * A))


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
