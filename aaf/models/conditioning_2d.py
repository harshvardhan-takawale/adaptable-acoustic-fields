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

from aaf.walls import ALPHA_BASELINE, ALPHA_NORM, M_NORM, WALLS_2D

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
    if cond_source in (COND_SOURCE_TOK, COND_SOURCE_TOK_DELTA):
        return TOKEN_DIM_2D
    if cond_source == COND_SOURCE_APER:
        return APERTURE_DIM_2D
    if cond_source == COND_SOURCE_BTOK:
        return APER_TOKEN_DIM_2D
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
    if cond_source in (COND_SOURCE_TOK, COND_SOURCE_TOK_DELTA):
        if alphas is None:
            raise ValueError(f"{cond_source} requires 16 segment alphas")
        return segment_token_features_2d(L, W, alphas, device=device, dtype=dtype)
    if cond_source == COND_SOURCE_APER:
        if x0 is None or a is None:
            raise ValueError(
                f"{COND_SOURCE_APER} requires x0 (divider position) and a (aperture width); "
                "the four wall alphas carry no divider information")
        return aperture_features_2d(L, W, x0, a, device=device, dtype=dtype)
    if cond_source == COND_SOURCE_BTOK:
        if x0 is None or a is None:
            raise ValueError(
                f"{COND_SOURCE_BTOK} requires x0 (divider position) and a (aperture width)")
        return aperture_token_features_2d(L, W, x0, a, device=device, dtype=dtype)
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


# ----------------------------------------------------------------------
# Track A2: shared-encoder TOKEN conditioning (D58)
# ----------------------------------------------------------------------
COND_SOURCE_TOK = "m_token"
COND_SOURCE_TOK_DELTA = "m_token_delta"
N_K_TOK_POS = 4          # Fourier k = 0..3 on the segment centre
N_K_TOK_M = 3            # Fourier k = 0..2 on m_hat
D_TOK = 2 * 2 * N_K_TOK_POS + 2 + 1 + (1 + 2 * N_K_TOK_M)      # 16 + 2 + 1 + 7 = 26
TOKEN_DIM_2D = 2 * 2 * N_K_GEOM + N_SEG_COND * D_TOK            # 32 + 416 = 448
TOKEN_AGG_DIM = 64
TOKEN_COND_DIM = 2 * 2 * N_K_GEOM + TOKEN_AGG_DIM               # 32 + 64 = 96 after the encoder
"""Per-segment TOKEN conditioning.

Track A gave each segment 7 PRIVATE dims. Holding out east_3 therefore held out its
PARAMETERS: those dims sat at the baseline value in all 400 training configs, so the network
never received gradient for them and produced no window effect at that position (recovered
fraction -0.069 versus 1.079 at a trained position, against an identical -5.28 dB ground
truth). That is the same pathology as P3-1's per-room latent, one level down -- a discrete
index is a memorization slot.

Here every segment is described by WHAT IT IS rather than by WHICH SLOT IT OCCUPIES:

    [cx, cy]   segment centre in normalized room coordinates -> Fourier k=0..3   16
    [nx, ny]   inward normal                                 -> raw               2
    extent     segment length as a fraction of its wall      -> raw               1
    m_hat      -ln(1-alpha)/3.0                              -> identity + k=0..2  7
                                                                             D_TOK = 26

The 16 tokens are flattened after the 32-d geometry block, so the stored width is 448 and the
trainer needs no change. `aaf.models.inr_2d` then applies ONE shared MLP to every token and
mean-pools, giving 32 + 64 = 96 into the existing FiLM generator. Because that MLP has no
per-segment parameters, a held-out position differs from a trained one only in its (cx, cy,
nx, ny) VALUES -- which the encoder has already learned to read from the other 15 segments.
"""


def segment_geometry(L: float, W: float, index: int):
    """(cx, cy, nx, ny, extent) for a segment, normalized. Order matches SEGMENT_NAMES."""
    n_per = 4
    wall = index // n_per
    k = index % n_per
    frac_lo, frac_hi = k / n_per, (k + 1) / n_per
    mid = 0.5 * (frac_lo + frac_hi)
    if wall == 0:      # west, x = 0, runs along y
        return 0.0, mid, 1.0, 0.0, 1.0 / n_per
    if wall == 1:      # east, x = L
        return 1.0, mid, -1.0, 0.0, 1.0 / n_per
    if wall == 2:      # south, y = 0, runs along x
        return mid, 0.0, 0.0, 1.0, 1.0 / n_per
    return mid, 1.0, 0.0, -1.0, 1.0 / n_per     # north


def segment_token_features_2d(L, W, alphas, device=None, dtype=torch.float32) -> torch.Tensor:
    """448-d: [32 geometry | 16 tokens x 26]. The tokens are reshaped by the model."""
    a = [float(x) for x in alphas]
    if len(a) != N_SEG_COND:
        raise ValueError("expected {} segment alphas, got {}".format(N_SEG_COND, len(a)))
    u = torch.tensor([(L - 3.0) / 3.0, (W - 3.0) / 2.0], device=device, dtype=dtype)
    geom = _fourier_block(u, N_K_GEOM)                                    # [32]
    toks = []
    for i in range(N_SEG_COND):
        cx, cy, nx, ny, ext = segment_geometry(L, W, i)
        pos = torch.tensor([cx, cy], device=device, dtype=dtype)
        mh = torch.tensor([m_hat_seg(a[i])], device=device, dtype=dtype)
        toks.append(torch.cat([
            _fourier_block(pos, N_K_TOK_POS),                              # 16
            torch.tensor([nx, ny, ext], device=device, dtype=dtype),       # 3
            mh,                                                            # 1
            _fourier_block(mh, N_K_TOK_M),                                 # 6
        ]))
    return torch.cat([geom, torch.cat(toks)])


# ----------------------------------------------------------------------
# Track B2: divider-TOKEN aperture conditioning (the A3 encoder on the aperture axis)
# ----------------------------------------------------------------------
COND_SOURCE_BTOK = "aperture_token"
N_TOK_DIV = N_SEG_COND              # 16, so segment_encoder's SHAPE is shared with A2/A3
APER_TOKEN_GEOM_DIM = N_GEOM_APER * 2 * N_K_GEOM             # 48
APER_TOKEN_DIM_2D = APER_TOKEN_GEOM_DIM + N_TOK_DIV * D_TOK  # 48 + 416 = 464
APER_TOKEN_COND_DIM = APER_TOKEN_GEOM_DIM + TOKEN_AGG_DIM    # 48 + 64 = 112 after the encoder
M_HAT_OPEN = 1.0
"""m_hat of a fully OPEN divider cell. 1.0 * M_NORM_SEG_COND = 3.0, i.e. alpha = 1-e^-3 =
0.9502 -- exactly Track A's open-window value, so 'open' means the same thing on both axes."""

APER_TOKEN_ALPHA_BASELINE = ALPHA_BASELINE
"""The divider's own absorption (aaf.data.aperture_configs.DIVIDER_ALPHA is this value).
Re-derived from aaf.walls rather than imported from aaf.data so conditioning_2d keeps its
numpy-free / data-free import surface. It MUST equal the baseline that
``inr_2d``'s delta-pool subtracts, or a sealed divider would not aggregate to zero."""

__doc_btok__ = """Track B's failure and what this replaces it with.

Track B (``aperture``, 55-d) asked a GLOBAL vector with a SCALAR aperture to induce a spatial
barrier at x0 with a gap of width a. It never learned the law: predicted inter-room level
difference vs sqrt(a) gave r^2 = 0.172 / slope 2.46 against GT's 0.948 / 7.61, and it was
EQUALLY wrong on seen and held-out apertures (residual ratio 0.933), i.e. a representation
failure rather than a transfer failure.

Here the divider is TOKENIZED along its length into 16 segments, each described by WHAT IT IS:

    [cx, cy]   segment centre, cx = x0/L, cy = (i+0.5)/16      -> Fourier k=0..3   16
    [nx, ny]   divider face normal, (+1, 0)                    -> raw               2
    extent     segment length / divider length = 1/16          -> raw               1
    m_hat      solid -> baseline, open -> 1.0, blended by cover -> identity + k=0..2  7
                                                                             D_TOK = 26

Same D_TOK = 26 and same featurization as ``segment_token_features_2d``, so ``inr_2d``'s
shared ``segment_encoder`` is reused verbatim -- only the geometry block widens from 32 (L, W)
to 48 (L, W, x0), because a model without x0 cannot tell which sub-room a receiver is in.

CONTINUITY IN ``a`` -- the load-bearing choice. ``a`` is drawn continuously on [0.1, 2.5], so
a pure integer open-COUNT would quantize the very axis the track tests (16 segments over
W ~ 4 m is a 0.25 m step, coarser than the 0.2 m hold-out band). We therefore express partial
coverage through **fractional m_hat**, NOT through ``extent``:

    f_i    = |[y_i, y_i+1] cap [W/2 - a/2, W/2 + a/2]| / (W/16)   in [0, 1]
    m_hat_i = m_hat(baseline) + f_i * (1 - m_hat(baseline))

``extent`` stays fixed at 1/16 for every token: the segments must PARTITION the divider
(the P3-3 Part-A lesson), and shrinking a solid segment's extent to make room for the doorway
would break that partition. f_i is piecewise linear and continuous in ``a``, and
sum_i f_i = 16a/W is STRICTLY increasing, so every distinct ``a`` gets a distinct token set.

Chose fractional m_hat over fractional extent because it also preserves the delta-pool's
zero: with the doorway closed every token carries exactly the baseline m_hat, so
sum_i [phi(t_i) - phi(t_i^baseline)] is IDENTICALLY ZERO. That is correct and matches the
topological reality -- a sealed divider is the un-edited room, and the a = 0 configs are
excluded from training anyway (H_B == 0 is a discontinuity no continuous coordinate holds).
"""


def divider_open_fraction(W: float, a: float, index: int, n_seg: int = N_TOK_DIV) -> float:
    """Fraction of divider segment ``index`` covered by the centred doorway of width ``a``.

    Segments PARTITION the divider: segment i spans y in [i*W/n, (i+1)*W/n]. The doorway is
    centred at y = W/2 (``ApertureConfig.extra_walls``), so it spans [W/2 - a/2, W/2 + a/2],
    clipped to the divider. Continuous and piecewise linear in ``a``; ``a >= W`` gives 1.0
    for every segment (no divider at all, matching the ``open`` kind's empty ``extra_walls``).
    """
    if int(index) < 0 or int(index) >= int(n_seg):
        raise ValueError("segment index {} out of range for n_seg={}".format(index, n_seg))
    if float(a) < 0.0:
        raise ValueError("aperture a must be >= 0, got {!r}".format(a))
    Wf = float(W)
    seg = Wf / float(n_seg)
    lo, hi = index * seg, (index + 1) * seg
    half = 0.5 * min(float(a), Wf)
    ov = min(hi, 0.5 * Wf + half) - max(lo, 0.5 * Wf - half)
    return max(0.0, min(ov, seg)) / seg


def divider_token_geometry(L: float, W: float, x0: float, index: int,
                           n_seg: int = N_TOK_DIV):
    """(cx, cy, nx, ny, extent) for divider segment ``index``, in normalized DOMAIN coords.

    The divider runs along y at x = x0, so cx = x0/L is the same for all tokens and cy sweeps
    the room's width; the face normal is (+1, 0) (pointing into room B, i.e. +x). Mirrors
    ``segment_geometry``'s contract exactly so the shared encoder reads the same channels.
    """
    return (float(x0) / float(L), (int(index) + 0.5) / float(n_seg),
            1.0, 0.0, 1.0 / float(n_seg))


def divider_m_hat(W: float, a: float, index: int, n_seg: int = N_TOK_DIV) -> float:
    """Coverage-blended m_hat: baseline when solid, 1.0 when fully open. See __doc_btok__."""
    mb = m_hat_seg(APER_TOKEN_ALPHA_BASELINE)
    return mb + divider_open_fraction(W, a, index, n_seg) * (M_HAT_OPEN - mb)


def aperture_token_features_2d(L, W, x0, a, device=None,
                               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """464-d: ``[48 geometry (L, W, x0) | 16 divider tokens x 26]``.

    The geometry block is BYTE-IDENTICAL to :func:`aperture_features_2d`'s first 48 dims
    (same ``normalize_params_aper_2d`` box), so Track B and Track B2 differ ONLY in how the
    doorway is expressed. The tokens are reshaped and pooled by ``aaf.models.inr_2d``.
    """
    u = normalize_params_aper_2d(L, W, x0, a, device=device, dtype=dtype)[:3]
    geom = _fourier_block(u, N_K_GEOM)                                     # [48]
    toks = []
    for i in range(N_TOK_DIV):
        cx, cy, nx, ny, ext = divider_token_geometry(L, W, x0, i)
        pos = torch.tensor([cx, cy], device=device, dtype=dtype)
        mh = torch.tensor([divider_m_hat(W, a, i)], device=device, dtype=dtype)
        toks.append(torch.cat([
            _fourier_block(pos, N_K_TOK_POS),                              # 16
            torch.tensor([nx, ny, ext], device=device, dtype=dtype),       # 3
            mh,                                                            # 1
            _fourier_block(mh, N_K_TOK_M),                                 # 6
        ]))
    return torch.cat([geom, torch.cat(toks)])
