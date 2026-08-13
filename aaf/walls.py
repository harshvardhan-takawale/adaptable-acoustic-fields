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

# Alpha normalization divisor used by the conditioning encoder (D46): u_alpha = alpha / 0.7.
ALPHA_NORM = 0.7


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
