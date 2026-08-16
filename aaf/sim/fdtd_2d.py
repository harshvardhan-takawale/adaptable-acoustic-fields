"""2D finite-difference time-domain (FDTD) room-acoustics solver.

Why this exists
---------------
The image-source method (``aaf.sim.ism_2d``) can only represent a convex box with
frequency-independent, angle-independent wall reflection. The next two edit axes we want to
train on -- a **doorway aperture** in an interior divider and a **partial-wall absorber
patch** -- are topological / spatially-resolved edits that ISM physically cannot express
(diffraction through an aperture; absorption that varies *along* a wall). This module is the
feasibility gate for replacing ISM with a wave solver on those axes.

Scheme
------
Node-centred **standard leapfrog** (SLF, the 2D 5-point explicit scheme) on a Cartesian grid::

    p^{n+1}_{i,j} = lam^2 (p^n_{i+1,j} + p^n_{i-1,j} + p^n_{i,j+1} + p^n_{i,j-1})
                    + 2 (1 - 2 lam^2) p^n_{i,j} - p^{n-1}_{i,j},      lam = c dt / dx

Nodes sit **on** the room boundary: with ``L / dx`` an integer there are ``nx = L/dx + 1``
node columns at ``x = 0, dx, ..., L``, and the Neumann image condition mirrors about the
boundary *node*. That makes ``cos(n_x pi x / L)`` an exact eigenvector of the discrete
Laplacian, so ``k_x = n_x pi / L`` holds exactly and the only modal-frequency error is the
temporal/spatial dispersion of the scheme::

    sin^2(pi f dt) = lam^2 [ sin^2(k_x dx / 2) + sin^2(k_y dx / 2) ]

(Using ``nx = L/dx`` instead of ``L/dx + 1`` -- the classic half-cell bug -- shifts the low
modes by about -1.1%, i.e. ~22x the 0.05% modal-frequency tolerance. ``meta['grid_shape']``
and ``meta['L_grid']`` are reported so this is checkable from the output alone.)

Frozen numerical parameters (defaults here): ``c = 343.0``, ``dx = 0.05``, ``fs = 12288``,
``n = 24576`` (T = 2.000 s exactly, rfft bin spacing exactly 0.5 Hz), giving
``lam = 0.5582682292`` against the 2D SLF stability bound ``1/sqrt(2) = 0.7071067812``.

Boundaries -- ONE code path
---------------------------
Every boundary in the domain (outer walls, faces of an interior divider, the tips of an
aperture, and every convex/concave corner) is handled by the *same* Kowalczyk-van Walstijn
locally-reacting update. Geometry is described per **face**: for each node and each of the
four directions the face is either open (neighbour is fluid) or blocked (neighbour is solid
or outside the domain). A blocked face carries a normalized admittance ``xi``.

Locally reacting wall, outward normal ``n``:  ``dp/dn = -(xi / c) dp/dt``.
Centred in space (image node) and in time, substituting into the SLF update and collecting
the ``p^{n+1}`` terms gives, for a node with *any* set of blocked faces::

    (1 + B) p^{n+1} = lam^2 * SUM_d w_d p^n_d + 2 (1 - 2 lam^2) p^n - (1 - B) p^{n-1}
    B = lam * SUM_{blocked faces f} xi_f
    w_{+x} = a_{+x} + 1 - a_{-x}      (a_d = 1 if face d is open, else 0), and cyclically.

``w_d`` is the single trick that collapses the case analysis: an open face contributes its
own neighbour with weight 1; a blocked face contributes *nothing* and instead adds +1 to the
weight of the opposite neighbour (the image node). Interior node -> all ``w_d = 1``, ``B = 0``
-> the plain SLF update. Flat wall -> one doubled neighbour. Corner -> two doubled
neighbours and ``B`` summing both admittances, which is exactly the KW corner rule; a corner
node is therefore *not* a special case in the code, which is where stable-but-wrong solvers
usually go wrong. A node with both faces blocked along the same axis (a zero-width channel)
has no consistent image condition and is rejected at construction.

Wall admittance from absorption
-------------------------------
``xi(alpha) = (1 - sqrt(1 - alpha)) / (1 + sqrt(1 - alpha))``  (the frozen FT-A formula).

Note what this quantity *is*: with the normal-incidence pressure reflection coefficient
``R = sqrt(1 - alpha)``, this is the normalized specific **admittance** ``rho c / Z``, not the
impedance -- it is 0 for a rigid wall (alpha = 0) and 1 for a perfectly matched wall
(alpha = 1), and it is what multiplies ``lam`` in ``B`` above. The normalized impedance is its
reciprocal (``inf`` at alpha = 0). Both are reported in ``meta``. Feeding the reciprocal into
``B`` would make alpha = 0 a pressure-release (Dirichlet) wall, i.e. silently the wrong
physics, so the convention is pinned by :func:`wall_admittance` and asserted in the tests.

Source
------
Soft (additive) point source: ``p^{n+1}[src] += s[n]``. Soft rather than hard so the source
node does not act as a rigid scatterer.

``S(0) = 0`` is enforced **by construction, not by cancellation**: the pulse is designed in
the rfft domain (unit magnitude to ``f_flat``, raised-cosine taper to ``f_max``, zero above,
linear phase for a delay of ``t0``), the DC coefficient is *assigned* ``0.0``, and the signal
is the inverse transform. The residual DC of the realized signal is then pure irfft round-off
(``meta['source']['dc_sum']``, ~1e-13 relative), not a near-cancellation of two large numbers.
A DC component would drive the undamped (0,0) mode of a rigid-ish room without bound.

Receivers
---------
Receivers are sampled **at grid nodes**: each requested position is snapped to the nearest
node, and ``meta['rx_offset_m']`` reports the snap distance (up to ``dx/sqrt(2) = 0.035 m``
at ``dx = 0.05``). No interpolation is applied -- interpolating pressure would bias modal
amplitudes near walls, and the gates use modal frequencies/bandwidths, which snapping does
not move. ``ir[:, k]`` holds ``p^{k+1}``, the field one leapfrog step after ``s[k]`` was
injected, so ``ir`` is exactly ``s`` convolved with the discrete Green's function and the
deconvolution ``H_deconv = rfft(ir) / S`` is unbiased in phase.

Reciprocity
-----------
The discrete operator is self-adjoint in the mu-weighted inner product, where ``mu = 1`` at
interior nodes, ``1/2`` at wall nodes and ``1/4`` at corners. Consequently
``G(b <- a) / mu_a == G(a <- b) / mu_b``: reciprocity is exact for source and receiver at
interior nodes, and carries the known factor ``mu_b / mu_a`` if one of them sits on a wall.

Known limitations
-----------------
* Frequency-independent (real) wall admittance only. Frequency-dependent boundaries would
  need a digital impedance filter per face.
* Staircase geometry: an aperture edge is resolved to ``dx``; edge diffraction is therefore
  accurate only well below the grid's dispersion limit.
* Numerical dispersion at ``lam = 0.5583`` reaches ~1% near 760 Hz, which is why the default
  source band stops at 800 Hz.
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from aaf.walls import WALLS_2D

__all__ = [
    "C_DEFAULT",
    "DX_DEFAULT",
    "FS_DEFAULT",
    "N_DEFAULT",
    "CFL_MAX_2D",
    "Geometry",
    "band_limited_pulse",
    "build_coefficients",
    "build_geometry",
    "simulate",
    "wall_admittance",
    "wall_impedance",
    "wall_reflection",
]

C_DEFAULT = 343.0
DX_DEFAULT = 0.05
FS_DEFAULT = 12288.0
N_DEFAULT = 24576

#: 2D standard-leapfrog stability bound on lam = c*dt/dx.
CFL_MAX_2D = 1.0 / math.sqrt(2.0)

# Face direction indices. Order is fixed and used by every (4, nx, ny) array below.
XM, XP, YM, YP = 0, 1, 2, 3
DIRS = ("xm", "xp", "ym", "yp")

#: Outer-wall name -> outward face direction. Mirrors aaf.walls.WALLS_2D
#: = (west, east, south, north) = (x=0, x=L, y=0, y=W).
WALL_DIR = {"west": XM, "east": XP, "south": YM, "north": YP}

# Default source band. 800 Hz is where SLF dispersion at lam = 0.5583 reaches ~1%.
SRC_F_FLAT_DEFAULT = 600.0
SRC_F_MAX_DEFAULT = 800.0


# --------------------------------------------------------------------------------------
# wall material
# --------------------------------------------------------------------------------------
def wall_reflection(alpha: float) -> float:
    """Normal-incidence *pressure* reflection coefficient ``R = sqrt(1 - alpha)``.

    ``alpha`` is an energy absorption coefficient in [0, 1] (same convention as
    ``aaf.walls`` / pyroomacoustics).
    """
    a = float(alpha)
    if not 0.0 <= a <= 1.0:
        raise ValueError("alpha must be in [0, 1], got {!r}".format(alpha))
    return float(math.sqrt(1.0 - a))


def wall_admittance(alpha: float) -> float:
    """Normalized specific admittance ``xi(alpha) = (1 - sqrt(1-a)) / (1 + sqrt(1-a))``.

    This is the FROZEN FT-A formula. It is the quantity that multiplies ``lam`` in the
    Kowalczyk-van Walstijn boundary update: ``xi(0) = 0`` (rigid), ``xi(1) = 1`` (matched).
    The normalized *impedance* is the reciprocal, see :func:`wall_impedance`.
    """
    r = wall_reflection(alpha)
    return float((1.0 - r) / (1.0 + r))


def wall_impedance(alpha: float) -> float:
    """Normalized specific impedance ``Z / (rho c) = 1 / xi(alpha)``; ``inf`` at alpha = 0."""
    xi = wall_admittance(alpha)
    if xi == 0.0:
        return float("inf")
    return float(1.0 / xi)


def _admittance_array(alpha: np.ndarray) -> np.ndarray:
    """Vectorized :func:`wall_admittance` (no per-element validation)."""
    r = np.sqrt(np.clip(1.0 - alpha, 0.0, 1.0))
    return (1.0 - r) / (1.0 + r)


# --------------------------------------------------------------------------------------
# source
# --------------------------------------------------------------------------------------
def band_limited_pulse(
    n: int,
    fs: float,
    f_flat: float = SRC_F_FLAT_DEFAULT,
    f_max: float = SRC_F_MAX_DEFAULT,
    t0: Optional[float] = None,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Zero-mean, band-limited excitation of length ``n`` with ``S(0) = 0`` by construction.

    Designed in the rfft domain: magnitude 1 up to ``f_flat``, raised-cosine taper to 0 at
    ``f_max``, 0 above; linear phase ``exp(-2j pi f t0)`` so the pulse peaks at ``t0``; the
    DC coefficient is *assigned* exactly ``0.0`` before the inverse transform. The DC null is
    therefore exact in the design and limited only by irfft round-off in the realization.

    ``t0`` defaults to ``8 / (f_max - f_flat)``, i.e. eight envelope widths of pre-ring
    headroom so the pulse does not wrap around the record.
    """
    n = int(n)
    if n < 16 or n % 2 != 0:
        raise ValueError("n must be an even integer >= 16, got {!r}".format(n))
    fs = float(fs)
    if not 0.0 <= f_flat < f_max <= fs / 2.0:
        raise ValueError(
            "require 0 <= f_flat < f_max <= fs/2, got f_flat={!r} f_max={!r} fs={!r}".format(
                f_flat, f_max, fs
            )
        )
    nf = n // 2 + 1
    f = np.arange(nf, dtype=np.float64) * (fs / n)

    mag = np.zeros(nf, dtype=np.float64)
    mag[f <= f_flat] = 1.0
    tap = (f > f_flat) & (f < f_max)
    mag[tap] = 0.5 * (1.0 + np.cos(np.pi * (f[tap] - f_flat) / (f_max - f_flat)))
    mag *= float(amplitude)

    if t0 is None:
        t0 = 8.0 / (f_max - f_flat)
    spec = mag * np.exp(-2j * np.pi * f * float(t0))

    spec[0] = 0.0  # <- the DC null, assigned not cancelled
    spec[-1] = spec[-1].real  # Nyquist coefficient of a real signal must be real
    return np.fft.irfft(spec, n=n)


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------
@dataclass
class Geometry:
    """Discrete domain: which nodes are fluid, which faces are boundaries, and their xi.

    Attributes
    ----------
    nx, ny:
        Node counts. ``nx = round(L/dx) + 1`` -- nodes sit ON the walls.
    solid:
        ``(nx, ny)`` bool. True = not simulated (interior structure). The exterior is
        implicit and always solid.
    blocked:
        ``(4, nx, ny)`` bool, direction order ``(XM, XP, YM, YP)``. True where the neighbour
        in that direction is solid or outside the domain.
    adm:
        ``(4, nx, ny)`` float. Normalized admittance of each face; meaningful only where
        ``blocked``.
    face_alpha:
        ``(4, nx, ny)`` float. The energy absorption that produced ``adm``, kept for meta.
    specs:
        Normalized echo of ``extra_walls``, with realized node indices.
    """

    nx: int
    ny: int
    dx_x: float
    dx_y: float
    L_grid: float
    W_grid: float
    solid: np.ndarray
    blocked: np.ndarray
    adm: np.ndarray
    face_alpha: np.ndarray
    specs: List[Dict[str, Any]]

    @property
    def air(self) -> np.ndarray:
        """``(nx, ny)`` bool: nodes that are actually updated."""
        return ~self.solid


def _fit_axis(length: float, dx_target: float, name: str) -> Tuple[int, float]:
    """Node count and the EXACT spacing that fits ``length`` on an integer number of cells.

    Replaces the old snap-and-warn. Snapping was a silent 0.5-0.8% dimension error on 39 of
    40 training and 9 of 10 test geometries -- none of which is an integer multiple of the
    0.05 m target -- which is ~30x the frequency tolerance the solver was validated to, and
    systematic on the very axis the model conditions on (FT-A blocker B1).

    Fitting per axis instead makes every room land on its own grid exactly. The induced
    spacing variation is tiny (measured over all 50 geometries: dx in [0.049688, 0.050328],
    max deviation 0.656%) and the dispersion knock-on is ~1e-4 % against an existing 0.020%.
    """
    length = float(length)
    if length <= 0.0:
        raise ValueError("{} must be positive, got {!r}".format(name, length))
    n_cells = int(round(length / dx_target))
    if n_cells < 2:
        raise ValueError(
            "{}={!r} spans only {} cells at dx_target={!r}; need >= 2".format(
                name, length, n_cells, dx_target)
        )
    dx_axis = length / n_cells
    if abs(n_cells * dx_axis - length) > 1e-12:
        raise AssertionError(
            "{}={!r} did not fit exactly: {} cells x {!r} = {!r}".format(
                name, length, n_cells, dx_axis, n_cells * dx_axis)
        )
    return n_cells + 1, dx_axis


def _span_nodes(lo: float, hi: float, dx: float, n_nodes: int, what: str) -> Tuple[int, int]:
    """Inclusive node index range for a metre span, snapped to the nearest nodes."""
    a = int(round(float(lo) / dx))
    b = int(round(float(hi) / dx))
    if b < a:
        a, b = b, a
    a = max(0, min(n_nodes - 1, a))
    b = max(0, min(n_nodes - 1, b))
    if b < a:
        raise ValueError("{} span [{!r}, {!r}] falls outside the grid".format(what, lo, hi))
    return a, b


def _apply_slab(
    spec: Dict[str, Any],
    solid: np.ndarray,
    face_alpha: np.ndarray,
    dx_x: float,
    dx_y: float,
) -> Dict[str, Any]:
    """Add one interior slab (optionally with apertures) to ``solid`` / ``face_alpha``.

    Takes both spacings because this one function indexes along a NORMAL axis (``pos``,
    ``thickness``) and a TANGENTIAL axis (``span``, ``apertures``); under per-axis fitting
    those are different metres-per-node.
    """
    nx, ny = solid.shape
    axis = str(spec.get("axis", "x")).lower()
    if axis not in ("x", "y"):
        raise ValueError("slab 'axis' must be 'x' or 'y', got {!r}".format(spec.get("axis")))
    n_norm, n_tan = (nx, ny) if axis == "x" else (ny, nx)
    dx_norm, dx_tan = (dx_x, dx_y) if axis == "x" else (dx_y, dx_x)

    if "pos" not in spec:
        raise ValueError("slab spec requires 'pos' (metres along its normal axis)")
    thickness = float(spec.get("thickness", dx_norm))
    k = max(1, int(round(thickness / dx_norm)))
    centre = int(round(float(spec["pos"]) / dx_norm))
    n0 = centre - (k - 1) // 2
    n1 = n0 + k - 1
    if n1 < 0 or n0 > n_norm - 1:
        raise ValueError(
            "slab at pos={!r} lies outside the room along {}".format(spec["pos"], axis)
        )
    n0 = max(0, n0)
    n1 = min(n_norm - 1, n1)

    span = spec.get("span")
    if span is None:
        t0, t1 = 0, n_tan - 1
    else:
        t0, t1 = _span_nodes(span[0], span[1], dx_tan, n_tan, "slab span")

    mask = np.zeros_like(solid)
    if axis == "x":
        mask[n0 : n1 + 1, t0 : t1 + 1] = True
    else:
        mask[t0 : t1 + 1, n0 : n1 + 1] = True

    apertures_out = []
    for ap in spec.get("apertures", ()) or ():
        ja, jb = _span_nodes(ap[0], ap[1], dx_tan, n_tan, "aperture")
        if jb - ja < 2:
            raise ValueError(
                "aperture [{!r}, {!r}] resolves to {} node(s) at dx={!r}; a usable aperture "
                "needs >= 3 nodes (the two edge nodes carry the boundary condition)".format(
                    ap[0], ap[1], jb - ja + 1, dx_tan
                )
            )
        if axis == "x":
            mask[n0 : n1 + 1, ja : jb + 1] = False
        else:
            mask[ja : jb + 1, n0 : n1 + 1] = False
        apertures_out.append(
            {
                "nodes": [int(ja), int(jb)],
                "clear_lo_m": float(ja * dx_tan),
                "clear_hi_m": float(jb * dx_tan),
                "clear_width_m": float((jb - ja) * dx_tan),
                "n_open_nodes": int(jb - ja + 1),
                "symmetric_about_m": float(0.5 * (ja + jb) * dx_tan),
            }
        )

    alpha = float(spec.get("alpha", 0.0))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("slab alpha must be in [0, 1], got {!r}".format(alpha))

    solid |= mask
    # Faces of *any* node whose neighbour belongs to this slab take the slab's alpha.
    face_alpha[XP, :-1, :][mask[1:, :]] = alpha
    face_alpha[XM, 1:, :][mask[:-1, :]] = alpha
    face_alpha[YP, :, :-1][mask[:, 1:]] = alpha
    face_alpha[YM, :, 1:][mask[:, :-1]] = alpha

    return {
        "type": "slab",
        "axis": axis,
        "pos_m": float(spec["pos"]),
        "alpha": alpha,
        "normal_nodes": [int(n0), int(n1)],
        "normal_pos_m": [float(n0 * dx_norm), float(n1 * dx_norm)],
        "tangential_nodes": [int(t0), int(t1)],
        "n_solid_nodes": int(mask.sum()),
        "apertures": apertures_out,
        "note": (
            "the reflecting plane sits at the first AIR node on each side, i.e. one dx "
            "outside the solid block; aperture clear width is (jb - ja) * dx"
        ),
    }


def wall_node_extent(j0: int, j1: int, dx: float, n_axis: int) -> float:
    """Absorbing extent of boundary nodes ``[j0, j1]`` along a flat wall.

    An interior wall node owns a ``dx`` strip; the two corner nodes own ``dx/2`` (which is
    exactly the ``mu`` energy weighting the update already applies). So a full wall of
    ``n_axis`` nodes measures ``(n_axis - 1) * dx``, as it must.
    """
    w = (j1 - j0 + 1) * dx
    if j0 == 0:
        w -= 0.5 * dx
    if j1 == n_axis - 1:
        w -= 0.5 * dx
    return w


def _patch_nodes(lo: float, hi: float, dx: float, n_axis: int) -> Tuple[int, int, float]:
    """Node range whose REALIZED extent best matches the requested span.

    The old rule took ``j0 = round(lo/dx)``, ``j1 = round(hi/dx)`` and painted the inclusive
    slice ``j0:j1+1``, i.e. ``round(a/dx) + 1`` nodes -- so the realized absorbing extent was
    ``a + dx``, one cell too wide, and it reported the nominal span anyway. The error is
    absolute in ``dx``, so it dominated short patches: a nominal 0.20 m patch absorbed like
    0.25 m (+25%) at dx=0.05. That is the FT-C independent variable, wrong at source
    (FT-A blocker B2).

    Choosing the node count that minimizes |realized - requested| fixes it for interior
    patches and for corner-touching ones alike, because the corner half-strips are folded
    into :func:`wall_node_extent`.
    """
    a = abs(float(hi) - float(lo))
    centre = 0.5 * (float(lo) + float(hi))
    jc = int(round(centre / dx))
    best = None
    for n_nodes in range(max(1, int(a / dx) - 1), int(a / dx) + 4):
        j0 = jc - (n_nodes - 1) // 2
        j1 = j0 + n_nodes - 1
        j0c, j1c = max(0, j0), min(n_axis - 1, j1)
        if j1c < j0c:
            continue
        w = wall_node_extent(j0c, j1c, dx, n_axis)
        key = (abs(w - a), abs(0.5 * (j0c + j1c) * dx - centre))
        if best is None or key < best[0]:
            best = (key, j0c, j1c, w)
    if best is None:
        raise ValueError("patch span [{!r}, {!r}] falls outside the wall".format(lo, hi))
    return best[1], best[2], best[3]


def _apply_patch(
    spec: Dict[str, Any], face_alpha: np.ndarray, dx_x: float, dx_y: float, nx: int, ny: int
) -> Dict[str, Any]:
    """Override the absorption of a segment of one outer wall (partial-wall absorber)."""
    wall = str(spec.get("wall", "")).strip().lower()
    if wall not in WALL_DIR:
        raise ValueError(
            "patch 'wall' must be one of {}, got {!r}".format(list(WALLS_2D), spec.get("wall"))
        )
    alpha = float(spec.get("alpha", 0.0))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("patch alpha must be in [0, 1], got {!r}".format(alpha))
    d = WALL_DIR[wall]
    span = spec.get("span")
    if d in (XM, XP):  # runs along y
        dxw, n_axis = dx_y, ny
        lo, hi = (0.0, (ny - 1) * dxw) if span is None else (span[0], span[1])
        j0, j1, realized = _patch_nodes(lo, hi, dxw, n_axis)
        i = 0 if d == XM else nx - 1
        face_alpha[d, i, j0 : j1 + 1] = alpha
        nodes = [int(j0), int(j1)]
    else:  # runs along x
        dxw, n_axis = dx_x, nx
        lo, hi = (0.0, (nx - 1) * dxw) if span is None else (span[0], span[1])
        j0, j1, realized = _patch_nodes(lo, hi, dxw, n_axis)
        j = 0 if d == YM else ny - 1
        face_alpha[d, j0 : j1 + 1, j] = alpha
        nodes = [int(j0), int(j1)]
    requested = abs(float(hi) - float(lo))
    return {
        "type": "patch",
        "wall": wall,
        "alpha": alpha,
        "nodes": nodes,
        "n_nodes": int(nodes[1] - nodes[0] + 1),
        "span_m": [float(nodes[0] * dxw), float(nodes[1] * dxw)],
        "width_requested_m": float(requested),
        "width_realized_m": float(realized),
        "width_error_m": float(realized - requested),
        "centre_realized_m": float(0.5 * (nodes[0] + nodes[1]) * dxw),
        "note": ("width_realized_m is the ABSORBING extent (corner nodes own dx/2); report "
                 "it, never the nominal span"),
    }


def _iter_specs(extra_walls: Any) -> List[Dict[str, Any]]:
    if extra_walls is None:
        return []
    if isinstance(extra_walls, dict):
        return [extra_walls]
    if isinstance(extra_walls, np.ndarray):
        return [{"type": "mask", "solid": extra_walls}]
    return list(extra_walls)


def build_geometry(
    L: float,
    W: float,
    alphas: Sequence[float],
    *,
    dx: float = DX_DEFAULT,
    extra_walls: Optional[Union[np.ndarray, Dict[str, Any], Sequence[Dict[str, Any]]]] = None,
) -> Geometry:
    """Discretize the room and every boundary face.

    ``alphas`` is a 4-vector of energy absorptions in ``aaf.walls.WALLS_2D`` order
    ``(west, east, south, north) = (x=0, x=L, y=0, y=W)``.

    ``extra_walls`` describes optional INTERIOR structure. It accepts a single spec dict, a
    sequence of them, or a raw ``(nx, ny)`` boolean solid mask (treated as rigid). Spec forms:

    ``{"type": "slab", "axis": "x"|"y", "pos": m, "thickness": m, "span": (lo, hi),
       "apertures": [(lo, hi), ...], "alpha": a}``
        A divider normal to ``axis``. ``span`` defaults to the full transverse extent;
        each aperture re-opens a node range, so a doorway is one slab with one aperture.
        ``thickness`` defaults to ``dx`` (one solid node column).

    ``{"type": "patch", "wall": "north", "span": (lo, hi), "alpha": a}``
        Overrides the absorption of part of one outer wall -- the partial-wall absorber.

    ``{"type": "mask", "solid": bool_array, "alpha": a}``
        Escape hatch: an arbitrary rigid/absorbing solid mask.

    Both spec kinds write into the same per-face ``alpha`` array, so the solver sees no
    difference between an outer wall, a divider face and an absorber patch.
    """
    a = [float(v) for v in alphas]
    if len(a) != 4:
        raise ValueError(
            "alphas must have 4 entries in WALLS_2D order {}, got {}".format(list(WALLS_2D), len(a))
        )
    for name, v in zip(WALLS_2D, a):
        if not 0.0 <= v <= 1.0:
            raise ValueError("alpha for wall {!r} must be in [0, 1], got {!r}".format(name, v))
    dx = float(dx)
    if dx <= 0.0:
        raise ValueError("dx must be positive, got {!r}".format(dx))

    nx, dx_x = _fit_axis(L, dx, "L")
    ny, dx_y = _fit_axis(W, dx, "W")
    L_grid, W_grid = float(L), float(W)      # exact by construction now

    solid = np.zeros((nx, ny), dtype=bool)
    face_alpha = np.zeros((4, nx, ny), dtype=np.float64)
    # Outer walls first; interior structure and patches may overwrite.
    face_alpha[XM, 0, :] = a[0]
    face_alpha[XP, nx - 1, :] = a[1]
    face_alpha[YM, :, 0] = a[2]
    face_alpha[YP, :, ny - 1] = a[3]

    specs_out: List[Dict[str, Any]] = []
    patches: List[Dict[str, Any]] = []
    for spec in _iter_specs(extra_walls):
        kind = str(spec.get("type", "slab")).lower()
        if kind == "slab":
            specs_out.append(_apply_slab(spec, solid, face_alpha, dx_x, dx_y))
        elif kind == "patch":
            patches.append(spec)  # applied last so it wins over slab-induced faces
        elif kind == "mask":
            m = np.asarray(spec["solid"], dtype=bool)
            if m.shape != (nx, ny):
                raise ValueError(
                    "mask 'solid' has shape {} but the grid is {}".format(m.shape, (nx, ny))
                )
            alpha_m = float(spec.get("alpha", 0.0))
            solid |= m
            face_alpha[XP, :-1, :][m[1:, :]] = alpha_m
            face_alpha[XM, 1:, :][m[:-1, :]] = alpha_m
            face_alpha[YP, :, :-1][m[:, 1:]] = alpha_m
            face_alpha[YM, :, 1:][m[:, :-1]] = alpha_m
            specs_out.append(
                {"type": "mask", "alpha": alpha_m, "n_solid_nodes": int(m.sum())}
            )
        else:
            raise ValueError("unknown extra_walls type {!r}".format(kind))
    for spec in patches:
        specs_out.append(_apply_patch(spec, face_alpha, dx_x, dx_y, nx, ny))

    blocked = np.zeros((4, nx, ny), dtype=bool)
    blocked[XP, :-1, :] = solid[1:, :]
    blocked[XP, -1, :] = True
    blocked[XM, 1:, :] = solid[:-1, :]
    blocked[XM, 0, :] = True
    blocked[YP, :, :-1] = solid[:, 1:]
    blocked[YP, :, -1] = True
    blocked[YM, :, 1:] = solid[:, :-1]
    blocked[YM, :, 0] = True

    air = ~solid
    if not air.any():
        raise ValueError("extra_walls filled the entire room; no fluid nodes remain")
    bad_x = air & blocked[XM] & blocked[XP]
    bad_y = air & blocked[YM] & blocked[YP]
    if bad_x.any() or bad_y.any():
        bad = np.argwhere(bad_x | bad_y)
        raise ValueError(
            "{} fluid node(s) have solid neighbours on BOTH sides of an axis (zero-width "
            "channel); the image condition is undefined there. First offenders (i, j): "
            "{}".format(int(len(bad)), bad[:8].tolist())
        )

    adm = _admittance_array(face_alpha)
    return Geometry(
        nx=nx,
        ny=ny,
        dx_x=dx_x,
        dx_y=dx_y,
        L_grid=L_grid,
        W_grid=W_grid,
        solid=solid,
        blocked=blocked,
        adm=adm,
        face_alpha=face_alpha,
        specs=specs_out,
    )


def build_coefficients(
    geom: Geometry, lam_x: float, lam_y: float, dtype: Any = np.float64
) -> Dict[str, np.ndarray]:
    """Assemble the per-node update coefficients (the single boundary code path).

    Returns ``{"cE", "cW", "cN", "cS", "cC", "cP", "B", "mu"}``, all ``(nx, ny)``, such that::

        p_next = cE*p_+x + cW*p_-x + cN*p_+y + cS*p_-y + cC*p_now + cP*p_prev

    Solid nodes get all-zero coefficients, so they stay identically zero without a mask pass.
    ``mu`` is the energy weight (1 interior, 1/2 wall, 1/4 corner).
    """
    lam_x, lam_y = float(lam_x), float(lam_y)
    lx2, ly2 = lam_x * lam_x, lam_y * lam_y
    blocked = geom.blocked
    open_f = (~blocked).astype(np.float64)
    aW, aE, aS, aN = open_f[XM], open_f[XP], open_f[YM], open_f[YP]

    # An open face contributes its own neighbour (weight 1); a blocked face contributes
    # nothing and adds 1 to the OPPOSITE neighbour (the image node).
    wE = aE + 1.0 - aW
    wW = aW + 1.0 - aE
    wN = aN + 1.0 - aS
    wS = aS + 1.0 - aN

    # Per-DIRECTION Courant weighting. A single `lam * sum(..., axis=0)` over all four faces
    # is correct only on an isotropic grid; with dx_x != dx_y it mis-scales absorption
    # anisotropically -- a stable, plausible-looking solver with the wrong wall physics.
    # XM/XP are x-normal faces (weight lam_x); YM/YP are y-normal (weight lam_y).
    adm_blocked = np.where(blocked, geom.adm, 0.0)
    B = lam_x * (adm_blocked[XM] + adm_blocked[XP]) + lam_y * (adm_blocked[YM] + adm_blocked[YP])
    den = 1.0 + B
    keep = geom.air.astype(np.float64)

    out = {
        "cE": lx2 * wE / den,
        "cW": lx2 * wW / den,
        "cN": ly2 * wN / den,
        "cS": ly2 * wS / den,
        "cC": np.full((geom.nx, geom.ny), 2.0 * (1.0 - lx2 - ly2)) / den,
        "cP": (B - 1.0) / den,
        "B": B,
    }
    for k in ("cE", "cW", "cN", "cS", "cC", "cP"):
        out[k] = (out[k] * keep).astype(dtype)
    out["B"] = out["B"] * keep
    mu_x = np.where(blocked[XM] | blocked[XP], 0.5, 1.0)
    mu_y = np.where(blocked[YM] | blocked[YP], 0.5, 1.0)
    out["mu"] = mu_x * mu_y * keep
    return out


# --------------------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------------------
def _snap_nodes(
    pos: np.ndarray, geom: Geometry, name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx_x, dx_y = geom.dx_x, geom.dx_y
    i = np.rint(pos[:, 0] / dx_x).astype(np.int64)
    j = np.rint(pos[:, 1] / dx_y).astype(np.int64)
    bad = (i < 0) | (i > geom.nx - 1) | (j < 0) | (j > geom.ny - 1)
    if bad.any():
        raise ValueError(
            "{} has {} point(s) outside the room [0, {}] x [0, {}]: {}".format(
                name, int(bad.sum()), geom.L_grid, geom.W_grid, pos[bad].tolist()
            )
        )
    on_solid = geom.solid[i, j]
    if on_solid.any():
        raise ValueError(
            "{} has {} point(s) that snap onto a solid node of the interior structure: "
            "{}".format(name, int(on_solid.sum()), pos[on_solid].tolist())
        )
    snapped = np.stack([i * dx_x, j * dx_y], axis=1)
    offset = np.linalg.norm(snapped - pos, axis=1)
    return i, j, snapped, offset


def simulate(
    L: float,
    W: float,
    alphas: Sequence[float],
    src: Sequence[float],
    rx: Sequence[Sequence[float]],
    *,
    dx: float = DX_DEFAULT,
    fs: float = FS_DEFAULT,
    n: int = N_DEFAULT,
    c: float = C_DEFAULT,
    extra_walls: Optional[Union[np.ndarray, Dict[str, Any], Sequence[Dict[str, Any]]]] = None,
    source: Optional[Dict[str, Any]] = None,
    record_energy: bool = False,
    deconv_eps: float = 1e-6,
    dtype: Any = np.float64,
) -> Dict[str, Any]:
    """Run the 2D FDTD solver and return time- and frequency-domain responses.

    Parameters
    ----------
    L, W:
        Room dimensions in metres. Should be integer multiples of ``dx`` (a warning fires
        otherwise and ``meta['L_grid']`` / ``meta['W_grid']`` report what was realized).
    alphas:
        4-vector of energy absorptions in ``aaf.walls.WALLS_2D`` order
        ``(west, east, south, north) = (x=0, x=L, y=0, y=W)``.
    src:
        ``(2,)`` source position in metres, snapped to the nearest grid node.
    rx:
        ``(n_rx, 2)`` receiver positions in metres, snapped to the nearest grid nodes.
    dx, fs, n, c:
        Frozen numerical parameters; see the module docstring.
    extra_walls:
        Interior structure -- see :func:`build_geometry`.
    source:
        Optional overrides ``{"f_flat", "f_max", "t0", "amplitude"}`` for
        :func:`band_limited_pulse`, or ``{"signal": array_of_length_n}`` to inject a
        custom excitation (rejected unless it is zero-mean).
    record_energy:
        If True, also return the discrete leapfrog energy per step. Exactly conserved for a
        rigid grid-aligned domain; a diagnostic (not a bound) at staircase corners.
    deconv_eps:
        Tikhonov floor for ``H_deconv``, relative to ``max |S|^2``.

    Returns
    -------
    dict with
        ``ir`` ``(n_rx, n)``, ``H_complex`` ``(n_rx, n//2+1)`` = ``rfft(ir)``,
        ``H_deconv`` = source-deconvolved transfer function, ``freqs`` ``(n//2+1,)``,
        ``source_signal`` ``(n,)``, ``source_spectrum`` ``(n//2+1,)``,
        ``passband`` (bool mask where ``|S| > 0.5 max|S|``), optionally ``energy`` ``(n,)``,
        and ``meta``.

    Raises
    ------
    ValueError
        If ``lam = c*dt/dx`` exceeds the 2D stability bound ``1/sqrt(2)``, if the geometry
        contains a zero-width channel, or if a source/receiver is outside the fluid domain.
    """
    fs = float(fs)
    n = int(n)
    c = float(c)
    dx = float(dx)
    if fs <= 0.0:
        raise ValueError("fs must be positive, got {!r}".format(fs))
    if n < 16 or n % 2 != 0:
        raise ValueError("n must be an even integer >= 16, got {!r}".format(n))

    dt = 1.0 / fs
    geom = build_geometry(L, W, alphas, dx=dx, extra_walls=extra_walls)
    lam_x = c * dt / geom.dx_x
    lam_y = c * dt / geom.dx_y
    # Anisotropic stability bound. `lam <= 1/sqrt(2)` is the ISOTROPIC special case of
    # lam_x^2 + lam_y^2 <= 1 and is not a valid bound once the spacings differ, so the
    # general form is asserted here. dt stays FROZEN at 1/fs rather than being recomputed
    # per room: exact fitting moves dx by at most 0.656% across the 50-room family (measured
    # worst case sqrt(lam_x^2+lam_y^2) = 0.794, a 20.6% margin), while a per-room dt would
    # make df = fs/n room-dependent and destroy the exactly-0.5 Hz rfft bin grid that
    # `fs = 12288` was frozen to provide.
    cfl = float(np.hypot(lam_x, lam_y))
    if lam_x <= 0.0 or lam_y <= 0.0:
        raise ValueError("lambda must be positive, got ({!r}, {!r})".format(lam_x, lam_y))
    if cfl > 1.0:
        raise ValueError(
            "CFL violation: sqrt(lam_x^2 + lam_y^2) = {:.10f} exceeds the 2D "
            "standard-leapfrog bound 1.0 (lam_x={:.10f}, lam_y={:.10f}, c={!r}, fs={!r}, "
            "dx_x={!r}, dx_y={!r}). Raise fs or dx.".format(
                cfl, lam_x, lam_y, c, fs, geom.dx_x, geom.dx_y)
        )
    lam = 0.5 * (lam_x + lam_y)          # reported only
    coef = build_coefficients(geom, lam_x, lam_y, dtype=dtype)
    nx, ny = geom.nx, geom.ny

    src_pos = np.asarray(src, dtype=np.float64).reshape(1, 2)
    rx_pos = np.asarray(rx, dtype=np.float64).reshape(-1, 2)
    si, sj, src_snap, src_off = _snap_nodes(src_pos, geom, "src")
    ri, rj, rx_snap, rx_off = _snap_nodes(rx_pos, geom, "rx")
    n_rx = rx_pos.shape[0]

    sopts = dict(source or {})
    user_signal = sopts.pop("signal", None)
    if user_signal is not None:
        s_sig = np.asarray(user_signal, dtype=dtype).reshape(-1)
        if s_sig.size != n:
            raise ValueError(
                "source['signal'] must have length n={}, got {}".format(n, s_sig.size)
            )
        _dc = float(np.sum(s_sig.astype(np.float64)))
        _peak = float(np.max(np.abs(s_sig.astype(np.float64))))
        if _peak > 0.0 and abs(_dc) > 1e-6 * _peak:
            raise ValueError(
                "source['signal'] must be zero-mean -- S(0) = 0 is mandatory, a DC component "
                "drives the undamped (0,0) mode without bound. sum={:.6e}, peak={:.6e}".format(
                    _dc, _peak
                )
            )
        src_desc = {"kind": "user-supplied signal (zero-mean checked)"}
    else:
        src_desc = {
            "kind": "soft additive band-limited pulse, DC assigned 0 in the rfft domain",
            "f_flat_hz": float(sopts.get("f_flat", SRC_F_FLAT_DEFAULT)),
            "f_max_hz": float(sopts.get("f_max", SRC_F_MAX_DEFAULT)),
            "amplitude": float(sopts.get("amplitude", 1.0)),
        }
        s_sig = band_limited_pulse(
            n=n,
            fs=fs,
            f_flat=float(sopts.get("f_flat", SRC_F_FLAT_DEFAULT)),
            f_max=float(sopts.get("f_max", SRC_F_MAX_DEFAULT)),
            t0=sopts.get("t0", None),
            amplitude=float(sopts.get("amplitude", 1.0)),
        ).astype(dtype)

    # ---- flat, contiguous working buffers -------------------------------------------
    # The field lives on a 1-node halo of zeros. Working on the RAVELLED array keeps every
    # stencil operand a contiguous 1D slice: +/-x is a shift of (ny+2), +/-y a shift of 1.
    stride = ny + 2
    m_tot = (nx + 2) * stride
    lo, hi = stride, m_tot - stride

    def _embed(a2d: np.ndarray) -> np.ndarray:
        pad = np.zeros((nx + 2, ny + 2), dtype=dtype)
        pad[1:-1, 1:-1] = a2d
        return pad.reshape(-1)[lo:hi].copy()

    cE = _embed(coef["cE"])
    cW = _embed(coef["cW"])
    cN = _embed(coef["cN"])
    cS = _embed(coef["cS"])
    cC = _embed(coef["cC"])
    cP = _embed(coef["cP"])

    sl_c = slice(lo, hi)
    sl_e = slice(lo + stride, hi + stride)
    sl_w = slice(lo - stride, hi - stride)
    sl_n = slice(lo + 1, hi + 1)
    sl_s = slice(lo - 1, hi - 1)

    bufs = [np.zeros(m_tot, dtype=dtype) for _ in range(3)]
    tmp = np.empty(hi - lo, dtype=dtype)

    src_flat = int((si[0] + 1) * stride + (sj[0] + 1))
    rx_flat = ((ri + 1) * stride + (rj + 1)).astype(np.intp)

    ir_t = np.zeros((n, n_rx), dtype=dtype)
    energy = np.zeros(n, dtype=np.float64) if record_energy else None
    if record_energy:
        mu = coef["mu"]
        air = geom.air
        open_x = (air[:-1, :] & ~geom.blocked[XP][:-1, :]).astype(np.float64)
        open_y = (air[:, :-1] & ~geom.blocked[YP][:, :-1]).astype(np.float64)
        # Transverse half-weights: a face lying on a wall carries half the energy.
        mu_x = np.where(geom.blocked[XM] | geom.blocked[XP], 0.5, 1.0)
        mu_y = np.where(geom.blocked[YM] | geom.blocked[YP], 0.5, 1.0)
        wxf = open_x * 0.5 * (mu_y[:-1, :] + mu_y[1:, :])
        wyf = open_y * 0.5 * (mu_x[:, :-1] + mu_x[:, 1:])
        lx2e, ly2e = lam_x * lam_x, lam_y * lam_y

    i_prev, i_cur, i_nxt = 0, 1, 2
    t_start = time.perf_counter()
    for step in range(n):
        f_cur = bufs[i_cur]
        f_prev = bufs[i_prev]
        f_nxt = bufs[i_nxt]
        out = f_nxt[sl_c]

        np.multiply(cE, f_cur[sl_e], out=out)
        np.multiply(cW, f_cur[sl_w], out=tmp)
        out += tmp
        np.multiply(cN, f_cur[sl_n], out=tmp)
        out += tmp
        np.multiply(cS, f_cur[sl_s], out=tmp)
        out += tmp
        np.multiply(cC, f_cur[sl_c], out=tmp)
        out += tmp
        np.multiply(cP, f_prev[sl_c], out=tmp)
        out += tmp

        f_nxt[src_flat] += s_sig[step]  # soft (additive) source
        np.take(f_nxt, rx_flat, out=ir_t[step])

        if record_energy:
            pn = f_nxt.reshape(nx + 2, ny + 2)[1:-1, 1:-1]
            pc = f_cur.reshape(nx + 2, ny + 2)[1:-1, 1:-1]
            d = pn - pc
            kin = 0.5 * float(np.sum(mu * d * d))
            gx = (pn[1:, :] - pn[:-1, :]) * (pc[1:, :] - pc[:-1, :])
            gy = (pn[:, 1:] - pn[:, :-1]) * (pc[:, 1:] - pc[:, :-1])
            pot = 0.5 * (lx2e * float(np.sum(wxf * gx)) + ly2e * float(np.sum(wyf * gy)))
            energy[step] = kin + pot

        i_prev, i_cur, i_nxt = i_cur, i_nxt, i_prev
    t_loop = time.perf_counter() - t_start

    ir = np.ascontiguousarray(ir_t.T)
    _src_dc = float(np.sum(s_sig.astype(np.float64)))
    H_complex = np.fft.rfft(ir, n=n, axis=1)
    S = np.fft.rfft(s_sig.astype(np.float64), n=n)
    freqs = np.arange(n // 2 + 1, dtype=np.float64) * (fs / n)

    s_max2 = float(np.max(np.abs(S)) ** 2)
    H_deconv = H_complex * np.conj(S)[None, :] / (np.abs(S)[None, :] ** 2 + deconv_eps * s_max2)
    passband = np.abs(S) > 0.5 * math.sqrt(s_max2)

    n_air = int(geom.air.sum())
    n_boundary = int((geom.air & geom.blocked.any(axis=0)).sum())
    updates = float((hi - lo) * n)  # array elements actually stepped
    meta = {
        "model": "fdtd_2d_slf_kw",
        "scheme": "node-centred standard leapfrog (SLF), 5-point, 2nd order",
        "boundary": "Kowalczyk-van Walstijn locally reacting, single code path",
        "L": float(L),
        "W": float(W),
        "L_grid": geom.L_grid,
        "W_grid": geom.W_grid,
        "dx": dx,
        "dx_target": dx,
        "dx_x": geom.dx_x,
        "dx_y": geom.dx_y,
        "dx_exact_fit": True,
        "fs": fs,
        "dt": dt,
        "n": n,
        "T_s": n * dt,
        "df_hz": fs / n,
        "c": c,
        "lambda_CFL": lam,
        "lambda_CFL_x": lam_x,
        "lambda_CFL_y": lam_y,
        "lambda_CFL_aniso": cfl,
        "lambda_CFL_aniso_max": 1.0,
        "lambda_CFL_max": CFL_MAX_2D,
        "grid_shape": [nx, ny],
        "n_nodes": int(nx * ny),
        "n_air_nodes": n_air,
        "n_boundary_nodes": n_boundary,
        "points_per_wavelength_at_300hz": float(c / 300.0 / max(geom.dx_x, geom.dx_y)),
        "walls": list(WALLS_2D),
        "alphas": [float(v) for v in alphas],
        "alpha_per_wall": {w: float(v) for w, v in zip(WALLS_2D, alphas)},
        "wall_xi_admittance": {w: wall_admittance(v) for w, v in zip(WALLS_2D, alphas)},
        "wall_impedance_normalized": {w: wall_impedance(v) for w, v in zip(WALLS_2D, alphas)},
        "wall_R_normal": {w: wall_reflection(v) for w, v in zip(WALLS_2D, alphas)},
        "face_alpha_realized": sorted(
            {float(v) for v in np.unique(geom.face_alpha[geom.blocked])}
        ),
        "face_xi_realized": sorted({float(v) for v in np.unique(geom.adm[geom.blocked])}),
        "src_pos": src_pos[0].tolist(),
        "src_pos_snapped": src_snap[0].tolist(),
        "src_node": [int(si[0]), int(sj[0])],
        "src_offset_m": float(src_off[0]),
        "rx_pos": rx_pos.tolist(),
        "rx_pos_snapped": rx_snap.tolist(),
        "rx_nodes": np.stack([ri, rj], axis=1).tolist(),
        "rx_offset_m": rx_off.tolist(),
        "rx_offset_max_m": float(rx_off.max()) if n_rx else 0.0,
        "source": dict(
            src_desc,
            t_peak_s=float(np.argmax(np.abs(s_sig)) * dt),
            dc_sum=_src_dc,
            dc_rel=float(abs(_src_dc) / max(float(np.abs(S).max()), 1e-300)),
            peak_abs=float(np.max(np.abs(s_sig))),
        ),
        "extra_walls": geom.specs,
        "dtype": np.dtype(dtype).name,
        "throughput": {
            "loop_seconds": float(t_loop),
            "node_updates": updates,
            "node_updates_per_s": float(updates / t_loop) if t_loop > 0 else float("inf"),
            "air_node_updates_per_s": (
                float(n_air * n / t_loop) if t_loop > 0 else float("inf")
            ),
        },
        "conventions": {
            "ir_alignment": "ir[:, k] = p^{k+1}, the field one step after injecting s[k]",
            "receivers": "snapped to the nearest grid node, no interpolation",
            "reciprocity": "exact for interior src/rx; carries mu_b/mu_a if one is on a wall",
            "xi": "xi is the normalized ADMITTANCE: 0 = rigid, 1 = matched",
        },
    }

    out_dict = {
        "ir": ir,
        "H_complex": H_complex,
        "H_deconv": H_deconv,
        "freqs": freqs,
        "source_signal": s_sig,
        "source_spectrum": S,
        "passband": passband,
        "meta": meta,
    }
    if record_energy:
        out_dict["energy"] = energy
    return out_dict
