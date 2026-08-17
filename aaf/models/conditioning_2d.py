"""P3-2 conditioning-vector builder: 2D geometry + per-wall material.

Reuses the pattern that worked in P3-1 Arm G — feed the physical parameters directly as
Fourier features into the shared FiLM generator, with NO latent table (D46). Here the
conditioning is the ONLY source of material information, so unlike P3-1's Arm G+ there is
no redundant path for the network to bypass.

Conditioning vector (6 physical parameters -> 64 features)::

    u = [ (L-3)/3, (W-3)/2, a_west/0.7, a_east/0.7, a_south/0.7, a_north/0.7 ]

    geometry dims: k = 0..7  ->  16 features each  (sin/cos of 2^k*pi*u)
    alpha    dims: k = 0..3  ->   8 features each  (alpha responses are smooth, so the
                                                    high octaves buy nothing and would
                                                    only invite aliasing across the 4
                                                    trained material levels)

    block layout:  [ 0:16] L        [16:32] W
                   [32:40] a_west   [40:48] a_east
                   [48:56] a_south  [56:64] a_north

The geometry box (L in [3,6], W in [3,5]) is deliberately the first two entries of
``aaf.models.conditioning._normalize_geom`` so a later 2D/3D comparison shares a
normalization. Values outside the box extrapolate rather than clamp; sin/cos stay bounded.

The layout is RAGGED (two different octave counts), which the 3D builder's single
broadcast cannot express -- so the rectangular idiom is simply applied twice and
concatenated. Block offsets are load-bearing and asserted in tests/test_conditioning_2d.py.

Deliberately **tcnn-free and numpy-free** (imports only math / torch / the dependency-free
``aaf.walls``) so it stays importable and unit-testable on CPU-only nodes. Do NOT import
``aaf.models.inr_2d`` here.

NOTE: ``aaf.models.conditioning`` (the 3D/P3-1 builder) is deliberately left untouched --
P3-1 reproducibility depends on it byte-for-byte, and its ``build_cond_vector`` is called
positionally as ``(cond_source, L, W, H, ...)`` which a 6-parameter variant cannot share.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

from aaf.walls import ALPHA_NORM, M_NORM, WALLS_2D

# k = 0..7 for geometry, k = 0..3 for the four absorptions.
N_K_GEOM = 8
N_K_ALPHA = 4

# 2 geometry dims x 2 (sin,cos) x 8 + 4 alpha dims x 2 x 4 = 32 + 32 = 64
FOURIER_DIM_2D = 2 * 2 * N_K_GEOM + len(WALLS_2D) * 2 * N_K_ALPHA

COND_SOURCE = "geom_alpha_fourier"
"""Name of this arm. Deliberately NOT 'geom_fourier', which already means the 48-d
(L,W,H) vector in every P3-1 config and checkpoint's train_meta.json -- reusing it would
make checkpoint metadata ambiguous across chunks."""


def normalize_params_2d(
    L: float,
    W: float,
    alphas: Sequence[float],
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """u = ((L-3)/3, (W-3)/2, a_w/0.7, a_e/0.7, a_s/0.7, a_n/0.7) -> [6].

    ``alphas`` must be a 4-sequence in ``aaf.walls.WALLS_2D`` order
    (west, east, south, north).
    """
    a = [float(x) for x in alphas]
    if len(a) != len(WALLS_2D):
        raise ValueError(
            "alphas must have {} entries in WALLS_2D order {}, got {}".format(
                len(WALLS_2D), list(WALLS_2D), len(a)
            )
        )
    return torch.tensor(
        [(L - 3.0) / 3.0, (W - 3.0) / 2.0] + [x / ALPHA_NORM for x in a],
        device=device,
        dtype=dtype,
    )


def _fourier_block(v: torch.Tensor, n_k: int) -> torch.Tensor:
    """[D] -> [D * 2 * n_k]; per dim ``[sin(2^0 pi v) .. sin(2^(K-1) pi v),
    cos(2^0 pi v) .. cos(2^(K-1) pi v)]``, dimension-major."""
    freqs = (2.0 ** torch.arange(n_k, device=v.device, dtype=v.dtype)) * math.pi  # [K]
    ang = v[:, None] * freqs[None, :]                                            # [D, K]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1).reshape(-1)        # [D*2K]


def fourier_features_2d(
    L: float,
    W: float,
    alphas: Sequence[float],
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """64-d Fourier features of (L, W, a_west, a_east, a_south, a_north). All in [-1, 1]."""
    u = normalize_params_2d(L, W, alphas, device=device, dtype=dtype)   # [6]
    return torch.cat(
        [
            _fourier_block(u[:2], N_K_GEOM),    # [32] geometry
            _fourier_block(u[2:], N_K_ALPHA),   # [32] absorptions
        ]
    )


# ----------------------------------------------------------------------
# P3-2b: the m-coordinate arm (D51)
# ----------------------------------------------------------------------
COND_SOURCE_M = "m_linear"
N_K_M = 3                       # pi, 2pi, 4pi -- deliberately lower than P3-2's 8 octaves
MLINEAR_DIM_2D = 2 * 2 * N_K_GEOM + len(WALLS_2D) * (1 + 2 * N_K_M)   # 32 + 28 = 60
"""Per wall: one IDENTITY channel + 3 octaves of sin/cos = 7 dims.

Why this beats P3-2's ``geom_alpha_fourier`` on the same physics:

* The target law is exactly linear in m, so an identity channel lets FiLM represent it with
  ZERO interpolation error. P3-2 conditioned on raw alpha, in which the law is
  log-curved, so every prediction between training points was an approximation.
* The top feature is 4*pi on m_hat -- one half-period per delta_m ~ 0.40. P3-2 put its top
  feature at 8*pi on normalized alpha while adjacent training points were 0.5 apart, i.e. two
  completely unconstrained cycles between samples, which is why alpha=0.30 (a pure
  interpolation) came out with the WRONG SIGN on all four walls.
"""


def m_of_alpha(alpha: float) -> float:
    """The linearizing material coordinate m = -ln(1 - alpha)."""
    a = float(alpha)
    if not 0.0 <= a < 1.0:
        raise ValueError(f"alpha must be in [0, 1), got {a}")
    return -math.log1p(-a)


def alpha_of_m(m: float) -> float:
    """Inverse of :func:`m_of_alpha`."""
    return 1.0 - math.exp(-float(m))


def m_hat_of_alpha(alpha: float) -> float:
    """Normalized material coordinate, m / ln(5). ~1.0 at the top of the sampled range."""
    return m_of_alpha(alpha) / M_NORM


def normalize_params_m_2d(L, W, alphas, device=None, dtype=torch.float32) -> torch.Tensor:
    """u = ((L-3)/3, (W-3)/2, m_hat_w, m_hat_e, m_hat_s, m_hat_n) -> [6]."""
    a = [float(x) for x in alphas]
    if len(a) != len(WALLS_2D):
        raise ValueError(
            "alphas must have {} entries in WALLS_2D order {}, got {}".format(
                len(WALLS_2D), list(WALLS_2D), len(a)))
    return torch.tensor(
        [(L - 3.0) / 3.0, (W - 3.0) / 2.0] + [m_hat_of_alpha(x) for x in a],
        device=device, dtype=dtype)


def m_linear_features_2d(L, W, alphas, device=None, dtype=torch.float32) -> torch.Tensor:
    """60-d features. Layout: [0:16] L, [16:32] W, then 7 per wall in WALLS_2D order
    ([32:39] west, [39:46] east, [46:53] south, [53:60] north), each
    ``[m_hat, sin(pi m), sin(2pi m), sin(4pi m), cos(pi m), cos(2pi m), cos(4pi m)]``.

    The geometry block is byte-identical to ``fourier_features_2d``'s, so arms sharing a
    geometry encoding differ ONLY in the material channels.
    """
    u = normalize_params_m_2d(L, W, alphas, device=device, dtype=dtype)      # [6]
    mh = u[2:]                                                              # [4]
    fb = _fourier_block(mh, N_K_M).reshape(len(WALLS_2D), 2 * N_K_M)        # [4, 6]
    per_wall = torch.cat([mh[:, None], fb], dim=1)                          # [4, 7]
    return torch.cat([_fourier_block(u[:2], N_K_GEOM), per_wall.reshape(-1)])


def cond_dim_for(cond_source: str) -> int:
    """Feature width for an arm. Guards against a config whose cond_dim contradicts its
    cond_source -- that mismatch trains happily and silently produces the wrong arm."""
    if cond_source == COND_SOURCE:
        return FOURIER_DIM_2D
    if cond_source == COND_SOURCE_M:
        return MLINEAR_DIM_2D
    if cond_source == COND_SOURCE_SEG:
        return SEGMENT_DIM_2D
    if cond_source == COND_SOURCE_APER:
        return APERTURE_DIM_2D
    raise ValueError(f"no fixed cond_dim for cond_source {cond_source!r}")


def build_cond_vector_2d(
    cond_source: str,
    L: float,
    W: float,
    alphas: Optional[Sequence[float]] = None,
    device=None,
    *,
    model=None,
    room_ids: Optional[torch.Tensor] = None,
    x0: Optional[float] = None,
    a: Optional[float] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Per-config conditioning vector fed to the FiLM generator.

    ``geom_alpha_fourier`` -> :func:`fourier_features_2d` [64] (unbatched).
    ``latent``             -> ``model.get_latent(room_ids)`` [B, latent_dim] (ablation arm
                              only; kept so a latent baseline can be trained with the same
                              trainer).

    Note the shape asymmetry between the two branches -- callers expand the analytic one.
    """
    if cond_source == COND_SOURCE:
        if alphas is None:
            raise ValueError(f"{COND_SOURCE} requires alphas")
        return fourier_features_2d(L, W, alphas, device=device, dtype=dtype)
    if cond_source == COND_SOURCE_M:
        if alphas is None:
            raise ValueError(f"{COND_SOURCE_M} requires alphas")
        return m_linear_features_2d(L, W, alphas, device=device, dtype=dtype)
    if cond_source == COND_SOURCE_SEG:
        if alphas is None:
            raise ValueError(f"{COND_SOURCE_SEG} requires 16 segment alphas")
        return segment_features_2d(L, W, alphas, device=device, dtype=dtype)
    if cond_source == COND_SOURCE_APER:
        if x0 is None or a is None:
            raise ValueError(
                f"{COND_SOURCE_APER} requires x0 (divider position) and a (aperture width); "
                "the four wall alphas carry no divider information")
        return aperture_features_2d(L, W, x0, a, device=device, dtype=dtype)
    if cond_source == "latent":
        if model is None or room_ids is None:
            raise ValueError("latent arm requires model + room_ids")
        return model.get_latent(room_ids)
    raise ValueError(
        f"unknown cond_source {cond_source!r}; expected {COND_SOURCE!r}, "
        f"{COND_SOURCE_M!r} or 'latent'"
    )


# ----------------------------------------------------------------------
# P3-3-FAST Track 1: per-segment conditioning (D56)
# ----------------------------------------------------------------------
COND_SOURCE_SEG = "m_segment"
N_SEG_COND = 16
N_K_SEG = 3
SEGMENT_DIM_2D = 2 * 2 * N_K_GEOM + N_SEG_COND * (1 + 2 * N_K_SEG)   # 32 + 112 = 144
"""Per segment: identity + 3 octaves of sin/cos = 7 dims, x 16 segments = 112, plus the
32-dim geometry block, which is BYTE-IDENTICAL to fourier_features_2d's so the geometry
encoding is shared with every earlier 2D arm.

Segment order is aaf.data.seg_configs.SEGMENT_NAMES:
  [32:39] west_1 ... [53:60] west_4 | [60:67] east_1 ... | south_* | north_*
i.e. offset 32 + 7*i for flat index i. Those offsets are load-bearing and are asserted
end-to-end in tests/test_seg_configs.py, from a manifest row through to geom.face_alpha.
"""

M_NORM_SEG_COND = 3.0
"""m_max for this chunk. P3-2b used ln(5) = 1.6094, so the normalized coordinate DIFFERS and a
model trained here is not numerically comparable to a P3-2b model. The wider range is what
admits alpha = 0.95, the open-window value."""


def m_hat_seg(alpha: float) -> float:
    return m_of_alpha(alpha) / M_NORM_SEG_COND


def segment_features_2d(L, W, alphas, device=None, dtype=torch.float32) -> torch.Tensor:
    """144-d features of (L, W, 16 segment absorptions)."""
    a = [float(x) for x in alphas]
    if len(a) != N_SEG_COND:
        raise ValueError("expected {} segment alphas, got {}".format(N_SEG_COND, len(a)))
    u = torch.tensor([(L - 3.0) / 3.0, (W - 3.0) / 2.0], device=device, dtype=dtype)
    mh = torch.tensor([m_hat_seg(x) for x in a], device=device, dtype=dtype)
    fb = _fourier_block(mh, N_K_SEG).reshape(N_SEG_COND, 2 * N_K_SEG)
    per_seg = torch.cat([mh[:, None], fb], dim=1)                     # [16, 7]
    return torch.cat([_fourier_block(u, N_K_GEOM), per_seg.reshape(-1)])


# ----------------------------------------------------------------------
# P3-3-FAST Track 2b: doorway-aperture conditioning
# ----------------------------------------------------------------------
COND_SOURCE_APER = "aperture"
N_GEOM_APER = 3                 # (L, W, x0) -- the divider position is a THIRD geometry dim
N_K_APER = 3                    # pi, 2pi, 4pi on the aperture coordinate
APERTURE_DIM_2D = N_GEOM_APER * 2 * N_K_GEOM + (1 + 2 * N_K_APER)     # 48 + 7 = 55
"""55 dims. Layout (offsets are load-bearing, asserted in tests/test_aperture_configs.py):

    [ 0:16] L      [16:32] W      [32:48] x0      [48:55] aperture

Geometry: 8 octaves of sin/cos per dim, the same idiom as every earlier 2D arm, but over
THREE dims -- the divider position x0 is geometry, not material, and a model without it
cannot tell which sub-room a receiver is in.

Aperture: one IDENTITY channel + 3 octaves =
``[sqrt_a_hat, sin(pi u), sin(2pi u), sin(4pi u), cos(pi u), cos(2pi u), cos(4pi u)]``.

The identity channel is in **sqrt(a)**, which is the linearizing coordinate FT-B measured
(pooled r^2 = 0.9870 for the inter-room level difference vs sqrt a; raw a gives 0.905, a^2
0.704). Exactly the role m = -ln(1-alpha) plays on the absorption axis: because the target law
is near-linear in sqrt(a), FiLM can represent it with near-zero interpolation error, and the
top feature at 4*pi means one half-period per delta-u ~ 0.4 rather than free cycles between
training samples (the P3-2 failure mode that flipped the sign of an interpolated prediction).

NOTE a = 0 (sealed) maps to u = 0, which is also the limit of the open configs' coordinate as
a -> 0, yet its physics is discontinuous (room B disconnects, H_B == 0). The conditioning
CANNOT separate the sealed case from a vanishing doorway, which is precisely why sealed rooms
are excluded from training rather than encoded with a flag."""

A_NORM_APER = 4.0
"""sqrt-normalizer, frozen at the FT-B domain width so the coordinate does not move with W.
u = sqrt(a)/2 in [0, ~1.06]: trained apertures reach 0.79 (a = 2.5), fully-open rooms 1.06."""


def normalize_params_aper_2d(L, W, x0, a, device=None, dtype=torch.float32) -> torch.Tensor:
    """u = ((L-8)/1, (W-4)/0.5, (x0/L - 0.5)/0.1, sqrt(a)/sqrt(4)) -> [4], all ~[-1, 1].

    The geometry box is the Track 2b sampling box (L in [7, 9], W in [3.5, 4.5],
    x0/L in [0.4, 0.6]) rather than the P3-2 box, so this arm's geometry block is NOT
    comparable to the earlier 2D arms' -- these rooms are twice as long.
    """
    if float(a) < 0.0:
        raise ValueError("aperture a must be >= 0, got {!r}".format(a))
    return torch.tensor(
        [(float(L) - 8.0) / 1.0, (float(W) - 4.0) / 0.5,
         (float(x0) / float(L) - 0.5) / 0.1,
         math.sqrt(float(a)) / math.sqrt(A_NORM_APER)],
        device=device, dtype=dtype)


def aperture_features_2d(L, W, x0, a, device=None, dtype=torch.float32) -> torch.Tensor:
    """55-d features of (L, W, x0, a). See :data:`APERTURE_DIM_2D` for the block layout."""
    u = normalize_params_aper_2d(L, W, x0, a, device=device, dtype=dtype)     # [4]
    ah = u[3:]                                                               # [1]
    fb = _fourier_block(ah, N_K_APER)                                        # [6]
    return torch.cat([_fourier_block(u[:3], N_K_GEOM), ah, fb])
