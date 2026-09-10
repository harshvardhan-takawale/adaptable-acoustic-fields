"""Canonical 2D wall ordering and material presets for P3-2 (chunk: material editing).

This module is the SINGLE source of truth for the wall order. It is imported by the
simulator, the dataset filename builder, the config enumerator, the conditioning encoder,
the evaluation and the demo CLI, so a wall-order mismatch between any two of them is
structurally impossible rather than merely unlikely (that class of bug is silent and
catastrophic: it produces a model that is confidently wrong and no aggregate metric
detects it).

Deliberately dependency-free (stdlib only) — no torch, no numpy, no pyroomacoustics —
so every layer can import it without pulling anything heavy in.

Wall convention (VERIFIED against pyroomacoustics 0.9.0 ``ShoeBox.wall_names`` and by an
image-lattice probe, see ``scripts/p3_2_physics_gate.py`` assert G0.2):

    west  -> x = 0        east  -> x = L
    south -> y = 0        north -> y = W

pyroomacoustics applies the *pressure* reflection coefficient R = sqrt(1 - alpha) per
bounce; ``alpha`` throughout this project is therefore an ENERGY absorption coefficient.
"""
from __future__ import annotations

# Canonical order. Everything that carries a 4-vector of absorptions uses THIS order.
# Matches pyroomacoustics ShoeBox.wall_names for dim == 2.
WALLS_2D = ("west", "east", "south", "north")

WALL_INDEX = {name: i for i, name in enumerate(WALLS_2D)}

# Which room dimension each wall spans, and which mode index it is selective for.
# west/east are the x-normal pair (length W, selective for n_x); south/north are the
# y-normal pair (length L, selective for n_y).
WALL_AXIS = {"west": "x", "east": "x", "south": "y", "north": "y"}

# Opposite-wall twin. Load-bearing for the alpha_eff control (D44): a held-out combo and
# its twin have IDENTICAL mean absorption and T60 and differ ONLY in wall location, so a
# model that learned a scalar effective absorption cannot transfer between them.
WALL_TWIN = {"west": "east", "east": "west", "south": "north", "north": "south"}

# Material presets (D44). Flat / frequency-independent energy absorption.
ALPHA_BASELINE = 0.15

MATERIALS = {
    "M0": 0.15,   # painted brick   (baseline; every non-edited wall is always M0)
    "M1": 0.05,   # concrete (hard) -- BELOW baseline: sharpens the wall's mode family
    "M2": 0.50,   # heavy curtain
    "M3": 0.70,   # absorber panel
}

MATERIAL_NAMES = {
    "M0": "painted brick",
    "M1": "concrete",
    "M2": "heavy curtain",
    "M3": "absorber panel",
}

# CLI-friendly aliases -> material id (used by scripts/demo_edit_2d.py --material).
MATERIAL_ALIASES = {
    "m0": "M0", "baseline": "M0", "brick": "M0", "painted_brick": "M0",
    "m1": "M1", "concrete": "M1", "hard": "M1",
    "m2": "M2", "curtain": "M2", "heavy_curtain": "M2",
    "m3": "M3", "absorber": "M3", "panel": "M3", "absorber_panel": "M3",
}

# Non-baseline materials only. Keeping the baseline OUT of this list is what makes the
# "one baseline per geometry" invariant structural: enumerating walls x NON_BASELINE
# cannot re-emit the all-baseline room 4x.
NON_BASELINE_MATERIALS = ("M1", "M2", "M3")

# Alpha normalization divisor used by the P3-2 conditioning encoder (D46): u_alpha = alpha / 0.7.
ALPHA_NORM = 0.7

# P3-2b (D51): the linearizing material coordinate. The ISM-ray damping law is EXACTLY linear
# in m = -ln(1-alpha) -- gamma = c*(m_w + m_e)/(4L) for an x-axial family -- so conditioning on m
# makes the target a linear function of the input rather than something the network must learn to
# invert. M_NORM = -ln(1-0.8) = ln 5 maps the sampled range m in [0.02, 1.61] onto ~[0.012, 1.0].
M_NORM = 1.6094379124341003

# P4-1 (D64): the fixed world scale for ALL absolute positions -- boundary tokens, source,
# receivers and ray sample points alike. Every length in metres is divided by this before it
# reaches the network, so a physical point means the same thing in every room regardless of
# shape or size.
#
# It must NOT be replaced by each room's own bounding box. Under bbox normalization the same
# physical corner encodes differently in a 3x3 room and a 6x5 room, which is precisely what
# breaks shape transfer -- and it is the reason the pre-P4-1 boundary tokens (unit-square
# coordinates, ignoring L and W entirely) could not carry geometry at all.
#
# 10 m comfortably contains the P3-2 family (L in [3,6], W in [3,5]), the Track B box (L up to
# 9 m) and the planned Z-corridor, so every normalized coordinate lands in [0, 1] -- which is
# also tinycudann's documented HashGrid input domain. Changing it invalidates every checkpoint
# trained under it, so it is frozen here rather than passed around as a tunable.
WORLD_SCALE = 10.0


def normalize_position(x, world_scale=None):
    """Map a POSITION (metres) into the network's encoding domain. Works on torch or numpy.

    Two regimes, and the default is the legacy one so every pre-P4-1 checkpoint keeps rendering
    bit-identically:

    * ``world_scale is None`` -- ``(x + 1) / 2``. Inherited from the vendored reference, where
      positions had already been AABB-normalized to [-1, 1] first; FreqRenderer2D dropped that
      step, so in practice this receives raw metres and emits ~[0.5, 3.5]. Outside tinycudann's
      documented [0, 1] domain, but room-INDEPENDENT, which is why it works at all.
    * ``world_scale = s`` (D64) -- ``x / s``. Absolute metres over a fixed scale, so a physical
      point encodes identically in every room and every shape.

    DIRECTIONS must not come through here: they are unit vectors carrying no length, and
    dividing one by a world scale is a category error. They always take the legacy affine.

    It lives in `walls` rather than in the model because `aaf.models.inr_2d` imports tinycudann,
    which needs a GPU -- and the property that source and receiver share one scaling has to be
    testable in the ordinary CPU test pass, not only where a GPU happens to be attached.
    """
    if world_scale is None:
        return (x + 1.0) * 0.5
    return x / world_scale


def resolve_material(name: str) -> str:
    """Map a user-supplied material name/alias to its canonical id ('M0'..'M3')."""
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if key in MATERIAL_ALIASES:
        return MATERIAL_ALIASES[key]
    upper = str(name).strip().upper()
    if upper in MATERIALS:
        return upper
    raise ValueError(
        "unknown material {!r}; expected one of {} or an alias {}".format(
            name, sorted(MATERIALS), sorted(MATERIAL_ALIASES)
        )
    )


def resolve_wall(name: str) -> str:
    """Validate/normalize a wall name against the canonical order."""
    key = str(name).strip().lower()
    if key not in WALL_INDEX:
        raise ValueError("unknown wall {!r}; expected one of {}".format(name, list(WALLS_2D)))
    return key


def alphas_for(wall=None, material=None, baseline: float = ALPHA_BASELINE):
    """Build the 4-tuple of absorptions in ``WALLS_2D`` order.

    ``wall=None`` (or ``material`` resolving to the baseline) returns the all-baseline
    configuration. Exactly one wall may differ from baseline (single-wall-edit scope).
    """
    out = [float(baseline)] * len(WALLS_2D)
    if wall is None or material is None:
        return tuple(out)
    w = resolve_wall(wall)
    m = resolve_material(material)
    out[WALL_INDEX[w]] = float(MATERIALS[m])
    return tuple(out)
