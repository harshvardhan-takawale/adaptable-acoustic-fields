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
    if cond_source == "latent":
        if model is None or room_ids is None:
            raise ValueError("latent arm requires model + room_ids")
        return model.get_latent(room_ids)
    raise ValueError(
        f"unknown cond_source {cond_source!r}; expected {COND_SOURCE!r}, "
        f"{COND_SOURCE_M!r} or 'latent'"
    )
