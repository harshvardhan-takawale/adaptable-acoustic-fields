"""Mode-resolved spectra by least-squares projection onto the analytic 2D mode shapes.

The P3-2 headline claim is about *which mode family* a wall edit moves, so every
measurement must be attributable to a single (n_x, n_y). Measuring at one receiver cannot
do that:

* frequency-degenerate and near-degenerate modes are inseparable in a spectrum -- at
  L=4.5, W=4.0 the axial (3,0) sits at 114.33 Hz and the tangential (2,2) at 114.73 Hz,
  0.4 Hz apart and in DIFFERENT families;
* a single receiver may sit near a node of the very mode being measured (mode (1,0) is
  down 14 dB at the nearest-to-centre receiver of an 8x8 grid), and the receiver nearest
  the source is dominated by direct sound.

Instead, use all 64 receivers as a spatial basis. With rigid-wall mode shapes
``Phi[r, j] = cos(n_j pi x_r / L) cos(m_j pi y_r / W)`` the mode-resolved spectra are::

    a(f) = pinv(Phi) @ H(:, f)          # [n_modes, n_freq] complex

Neighbouring modes are then suppressed by their spatial orthogonality rather than by
frequency resolution, which is what makes a per-family bandwidth honest.

Two hard constraints, both enforced here:

* **f <= 200 Hz.** Measured conditioning of ``Phi`` is 1.5-1.9 for modes up to 200 Hz but
  blows up to ~1e16 by 250 Hz -- 8 points per axis cannot resolve more than a handful of
  half-wavelengths, and beyond that the projection is spatial aliasing dressed up as
  physics. ``cond(Phi) <= 5`` is asserted per geometry.
* **Receiver coordinates come from the stored ``receiver_pos``**, never reconstructed from
  a margin default. ``aaf.eval.spatial_modes.MARGIN_DEFAULT`` is 0.5 while the datasets are
  built with margin 0.3 (``scripts/build_datasets.py``); reconstructing the grid biases
  peak level by +0.66 dB and inflates cond(Phi) from 1.54 to 3.02. That mismatch is left
  untouched upstream (published P2/P3-1 numbers depend on it) and simply not inherited.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from aaf.sim.analytical_modal_2d import C_DEFAULT, eigenfrequencies_2d

F_MAX_PROJECTION_HZ = 200.0
COND_MAX = 5.0
EXCITATION_MIN = 0.2

X_AXIAL = "x_axial"
Y_AXIAL = "y_axial"
TANGENTIAL = "tangential"
FAMILIES = (X_AXIAL, Y_AXIAL, TANGENTIAL)


@dataclass(frozen=True)
class Mode:
    """One (n_x, n_y) mode. ``family`` is the P3-2 grouping."""

    n_x: int
    n_y: int
    f: float
    family: str

    @property
    def pair(self):
        return (self.n_x, self.n_y)


def classify_family(n_x: int, n_y: int) -> str:
    """x-axial: (n>0, 0) -- reflects off west/east only. y-axial: (0, m) -- south/north.
    tangential: both nonzero -- responds to every wall."""
    if n_x > 0 and n_y == 0:
        return X_AXIAL
    if n_x == 0 and n_y > 0:
        return Y_AXIAL
    if n_x > 0 and n_y > 0:
        return TANGENTIAL
    raise ValueError("the (0, 0) DC mode has no family")


def enumerate_modes(
    L: float,
    W: float,
    *,
    f_max: float = F_MAX_PROJECTION_HZ,
    c: float = C_DEFAULT,
) -> List[Mode]:
    """Every mode with 0 < f <= ``f_max``, DEGENERATES EXPANDED.

    ``eigenfrequencies_2d`` collapses modes sharing a frequency into one entry; we expand
    ``.pairs`` so each (n_x, n_y) gets its own row. (``spatial_modes.pick_first_modes``
    keeps only ``sorted(e.pairs)[0]``, which silently drops the degenerate partner and
    mislabels families on near-square rooms -- do not reuse it here.)
    """
    out: List[Mode] = []
    for e in eigenfrequencies_2d(L, W, c=c, f_max=f_max):
        if e.f <= 0.0:
            continue
        for (n_x, n_y) in e.pairs:
            if n_x == 0 and n_y == 0:
                continue
            out.append(Mode(int(n_x), int(n_y), float(e.f), classify_family(n_x, n_y)))
    out.sort(key=lambda m: (m.f, m.n_x, m.n_y))
    return out


def mode_shape_matrix(modes: Sequence[Mode], rx: np.ndarray, L: float, W: float) -> np.ndarray:
    """``Phi[r, j] = cos(n_j pi x_r / L) cos(m_j pi y_r / W)`` -> ``[n_rx, n_modes]``."""
    rx = np.asarray(rx, dtype=float)
    if rx.ndim != 2 or rx.shape[1] != 2:
        raise ValueError(f"rx must be [n_rx, 2], got {rx.shape}")
    x, y = rx[:, 0][:, None], rx[:, 1][:, None]
    n = np.array([m.n_x for m in modes], dtype=float)[None, :]
    p = np.array([m.n_y for m in modes], dtype=float)[None, :]
    return np.cos(n * np.pi * x / L) * np.cos(p * np.pi * y / W)


def excitation(modes: Sequence[Mode], src: Sequence[float], L: float, W: float) -> np.ndarray:
    """|Phi_m(source)| -- a mode with a node at the source is simply not excited.

    With the source at (0.5, 0.5) m this is exactly zero whenever ``n_x == L`` or
    ``n_y == W`` in metres (e.g. mode (0,4) at W = 4.0), so integer room dimensions need
    the mask below or those modes are measured as noise.
    """
    src = np.asarray(src, dtype=float).reshape(1, 2)
    return np.abs(mode_shape_matrix(modes, src, L, W)[0])


@dataclass
class Projection:
    """Result of projecting a 64-receiver field onto the analytic mode shapes."""

    modes: List[Mode]
    spectra: np.ndarray          # [n_modes, n_freq] complex, mode-resolved
    cond: float                  # cond(Phi)
    excitation: np.ndarray       # [n_modes] |Phi_m(src)|
    used: np.ndarray             # [n_modes] bool -- passed the excitation mask
    residual_frac: float         # ||H - Phi a|| / ||H|| over the fitted band

    def by_family(self, family: str, *, only_used: bool = True):
        """Indices of modes in ``family`` (optionally only well-excited ones)."""
        return [
            i for i, m in enumerate(self.modes)
            if m.family == family and (not only_used or self.used[i])
        ]


def project_field(
    H: np.ndarray,
    rx: np.ndarray,
    L: float,
    W: float,
    *,
    src: Optional[Sequence[float]] = None,
    f_max: float = F_MAX_PROJECTION_HZ,
    c: float = C_DEFAULT,
    fs: float = 4096.0,
    excitation_min: float = EXCITATION_MIN,
    cond_max: float = COND_MAX,
    check_cond: bool = True,
) -> Projection:
    """Project ``H`` ``[n_rx, n_freq]`` onto the analytic mode shapes.

    ``rx`` must be the receiver coordinates as stored with the data (metres).
    Raises if ``cond(Phi)`` exceeds ``cond_max`` -- an ill-conditioned basis silently turns
    the projection into noise amplification, which must fail loudly, not degrade quietly.
    """
    H = np.asarray(H)
    rx = np.asarray(rx, dtype=float)
    if H.ndim != 2 or H.shape[0] != rx.shape[0]:
        raise ValueError(f"H {H.shape} must be [n_rx, n_freq] matching rx {rx.shape}")

    modes = enumerate_modes(L, W, f_max=f_max, c=c)
    if not modes:
        raise ValueError(f"no modes below {f_max} Hz for room ({L}, {W})")
    Phi = mode_shape_matrix(modes, rx, L, W)
    cond = float(np.linalg.cond(Phi))
    if check_cond and cond > cond_max:
        raise ValueError(
            "mode-shape basis is ill-conditioned: cond(Phi)={:.3g} > {} for room "
            "({}, {}) with {} modes <= {} Hz. Lower f_max.".format(
                cond, cond_max, L, W, len(modes), f_max
            )
        )

    a = np.linalg.pinv(Phi) @ H                              # [n_modes, n_freq]

    n_freq = H.shape[1]
    df = fs / (2.0 * (n_freq - 1))
    hi = min(n_freq, int(round(f_max / df)) + 1)
    resid = float(
        np.linalg.norm(H[:, :hi] - Phi @ a[:, :hi]) / max(np.linalg.norm(H[:, :hi]), 1e-30)
    )

    if src is None:
        exc = np.ones(len(modes))
    else:
        exc = excitation(modes, src, L, W)
    return Projection(
        modes=modes, spectra=a, cond=cond, excitation=exc,
        used=exc >= excitation_min, residual_frac=resid,
    )
