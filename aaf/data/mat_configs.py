"""P3-2 configuration enumeration: geometry x (edited wall, material).

One "config" = one simulated room = a geometry (L, W) plus a 4-vector of wall absorptions.
Exactly one wall differs from the M0 baseline in an edited config (single-wall-edit scope
this chunk; multi-wall is future work).

Invariants enforced here rather than trusted (both are silent-corruption hazards):

1. **One baseline per geometry.** ``(wall=k, M0)`` is the SAME room for every k, so naively
   crossing 4 walls x 4 materials would emit the all-baseline room 4x under 4 different
   names. Structural fix: ``aaf.walls.NON_BASELINE_MATERIALS`` excludes M0, and the
   baseline is emitted exactly once per geometry.
2. **Filename uniqueness.** Filenames quantize L/W to 2 dp, so two geometries agreeing to
   2 dp would silently share an HDF5 file (this bug is live in the 3D path, where
   ``train_rooms_45.yaml`` stores L=3.620434... and the filename formats L3.62). We round
   at generation time and assert the rounded set is unique.

Splits (D44) -- held-out combos are (west, M2) and (north, M3):

  TRAIN  40 geom x 11 configs (baseline + 12 edits - 2 held out)   = 440
  (i)    unseen geom x seen combo          10 x 11                 = 110
  (ii)   *seen* geom x held-out combo      40 x  2                 =  80
  (iii)  unseen geom x held-out combo      10 x  2                 =  20   <- headline
  (iv)   unseen geom x unseen alpha=0.30   10 x  4 walls           =  40

(ii) decomposes the (i)->(iii) gap into combo-novelty vs geometry-novelty; (iv) proves the
material axis is continuous rather than a 4-way classifier. Both are extra simulations
only -- no extra training.

The held-out combos are not arbitrary: each has an opposite-wall twin that IS trained
((east,M2) and (south,M3)). Since west/east both span W and south/north both span L, the
twin has identical mean absorption and identical T60 -- it differs ONLY in where the
absorber sits. A model that learned a scalar effective absorption therefore cannot
transfer, which makes split (iii) a genuine test of wall identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from aaf.walls import (
    ALPHA_BASELINE,
    MATERIALS,
    NON_BASELINE_MATERIALS,
    WALL_INDEX,
    WALLS_2D,
    resolve_material,
    resolve_wall,
)

# Held-out (wall, material) combinations -- excluded from ALL training geometries (D44).
HELDOUT_COMBOS: Tuple[Tuple[str, str], ...] = (("west", "M2"), ("north", "M3"))

# Split (iv): an absorption value never seen in training, applied to each wall in turn.
UNSEEN_ALPHA = 0.30


def round_dim(x: float) -> float:
    """Round a room dimension to the 2 dp the filename encodes (see invariant 2)."""
    return round(float(x), 2)


@dataclass(frozen=True)
class MatConfig:
    """One simulated room."""

    L: float
    W: float
    wall: Optional[str]          # None -> all-baseline configuration
    alpha_edit: Optional[float]  # absorption applied to ``wall``
    material: Optional[str]      # 'M1'..'M3', or e.g. 'A030' for the unseen-alpha split

    @property
    def is_baseline(self) -> bool:
        return self.wall is None

    @property
    def alphas(self) -> Tuple[float, float, float, float]:
        """Absorptions in WALLS_2D order (west, east, south, north)."""
        out = [ALPHA_BASELINE] * len(WALLS_2D)
        if self.wall is not None:
            out[WALL_INDEX[self.wall]] = float(self.alpha_edit)
        return tuple(out)  # type: ignore[return-value]

    @property
    def combo(self) -> Optional[Tuple[str, str]]:
        return None if self.is_baseline else (self.wall, self.material)

    @property
    def filename(self) -> str:
        return room_filename_2d_mat(self.L, self.W, self.alphas)

    @property
    def label(self) -> str:
        return "L{:.2f}_W{:.2f}_{}".format(
            self.L, self.W, "baseline" if self.is_baseline else f"{self.wall}_{self.material}"
        )


def room_filename_2d_mat(L: float, W: float, alphas: Sequence[float]) -> str:
    """``L4.50_W4.00_aW0.15_aE0.15_aS0.15_aN0.15.h5`` -- self-describing and greppable.

    A pure function of the config, so the loader recomputes it rather than consulting a
    manifest. Deliberately NOT the Phase-1 ``room_filename(L, W, alpha)``: this chunk's
    rooms carry four absorptions and use max_order=60, so they must not collide with
    ``data/track_a``.
    """
    a = [float(x) for x in alphas]
    if len(a) != 4:
        raise ValueError(f"alphas must have 4 entries, got {len(a)}")
    return "L{:.2f}_W{:.2f}_aW{:.2f}_aE{:.2f}_aS{:.2f}_aN{:.2f}.h5".format(L, W, *a)


PRESET_ALPHAS: Tuple[float, ...] = (0.05, 0.15, 0.30, 0.50, 0.70)
ALPHA_QUANT_DP = 6


def _alpha_token(a: float) -> str:
    """2 dp for a canonical preset, 6 dp otherwise.

    The two alphabets are disjoint as strings (2 vs 6 decimal digits), which is what makes
    the v2 filename injective while leaving every P3-2 name byte-identical.
    """
    return "{:.2f}".format(a) if any(abs(a - p) < 1e-12 for p in PRESET_ALPHAS) \
        else "{:.6f}".format(a)


def room_filename_2d_mat_v2(L: float, W: float, alphas: Sequence[float]) -> str:
    """Back-compatible superset of :func:`room_filename_2d_mat`.

    Emits the IDENTICAL string when every alpha is a canonical preset -- so P3-2's 690 files
    are reused by name -- and 6-dp tokens for continuously drawn values. P3-2's ``{:.2f}``
    everywhere would collide under continuous sampling (0.462 and 0.464 both -> "0.46"),
    silently merging two different rooms into one HDF5 file.
    """
    a = [float(x) for x in alphas]
    if len(a) != 4:
        raise ValueError(f"alphas must have 4 entries, got {len(a)}")
    return "L{:.2f}_W{:.2f}_aW{}_aE{}_aS{}_aN{}.h5".format(
        L, W, *[_alpha_token(x) for x in a])


def make_config(L: float, W: float, wall=None, material=None, alpha=None) -> MatConfig:
    """Build one :class:`MatConfig`. ``wall=None`` -> baseline.

    Either ``material`` (an id/alias resolved via ``aaf.walls``) or an explicit ``alpha``
    must be given for an edited wall; ``alpha`` wins and is labelled ``A###`` so the
    unseen-alpha split is distinguishable from the 4 presets.
    """
    L, W = round_dim(L), round_dim(W)
    if wall is None:
        return MatConfig(L=L, W=W, wall=None, alpha_edit=None, material=None)
    w = resolve_wall(wall)
    if alpha is not None:
        a = float(alpha)
        tag = "A{:03d}".format(int(round(a * 100)))
        return MatConfig(L=L, W=W, wall=w, alpha_edit=a, material=tag)
    m = resolve_material(material)
    if MATERIALS[m] == ALPHA_BASELINE:
        # (wall, M0) IS the baseline -- collapse it rather than emitting a duplicate room.
        return MatConfig(L=L, W=W, wall=None, alpha_edit=None, material=None)
    return MatConfig(L=L, W=W, wall=w, alpha_edit=MATERIALS[m], material=m)


def enumerate_configs(
    geometries: Iterable[Sequence[float]],
    *,
    materials: Sequence[str] = NON_BASELINE_MATERIALS,
    exclude_combos: Sequence[Tuple[str, str]] = (),
    only_combos: Optional[Sequence[Tuple[str, str]]] = None,
    include_baseline: bool = True,
    unseen_alpha: Optional[float] = None,
) -> List[MatConfig]:
    """Enumerate configs for a list of ``(L, W)`` geometries.

    ``exclude_combos`` removes (wall, material) pairs globally (the training holdout).
    ``only_combos`` restricts to exactly those pairs (splits (ii)/(iii)).
    ``unseen_alpha`` instead emits every wall at that absorption (split (iv)).
    """
    if any(MATERIALS[resolve_material(m)] == ALPHA_BASELINE for m in materials):
        raise ValueError(
            "the baseline material must not appear in `materials` -- it would emit the "
            "all-baseline room once per wall (see invariant 1)"
        )
    excl = {(resolve_wall(w), resolve_material(m)) for w, m in exclude_combos}
    keep = None if only_combos is None else {
        (resolve_wall(w), resolve_material(m)) for w, m in only_combos
    }

    out: List[MatConfig] = []
    for geom in geometries:
        L, W = round_dim(geom[0]), round_dim(geom[1])
        if unseen_alpha is not None:
            for wall in WALLS_2D:
                out.append(make_config(L, W, wall=wall, alpha=unseen_alpha))
            continue
        if include_baseline:
            out.append(make_config(L, W))
        for wall in WALLS_2D:
            for m in materials:
                mid = resolve_material(m)
                if (wall, mid) in excl:
                    continue
                if keep is not None and (wall, mid) not in keep:
                    continue
                out.append(make_config(L, W, wall=wall, material=mid))

    assert_unique_filenames(out)
    return out


def assert_unique_filenames(configs: Sequence[MatConfig]) -> None:
    """Guard invariants 1 and 2 -- no duplicated room under any name."""
    names = [c.filename for c in configs]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise AssertionError(
            "duplicate config filenames ({} of {} unique); first few: {}".format(
                len(set(names)), len(names), dupes[:5]
            )
        )


def coverage_report(configs: Sequence[MatConfig]) -> dict:
    """Which materials each wall is trained with, and vice versa (the D44 coverage check)."""
    by_wall: dict = {w: set() for w in WALLS_2D}
    by_material: dict = {}
    for c in configs:
        if c.is_baseline:
            continue
        by_wall[c.wall].add(c.material)
        by_material.setdefault(c.material, set()).add(c.wall)
    return {
        "materials_per_wall": {w: sorted(v) for w, v in by_wall.items()},
        "walls_per_material": {m: sorted(v) for m, v in by_material.items()},
        "min_materials_per_wall": min(len(v) for v in by_wall.values()),
        "min_walls_per_material": min((len(v) for v in by_material.values()), default=0),
        "n_configs": len(configs),
        "n_baseline": sum(1 for c in configs if c.is_baseline),
    }
