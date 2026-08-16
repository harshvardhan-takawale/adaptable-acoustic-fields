"""P3-2d: regular-grid sampling of the material axis, and the midpoint test set.

P3-2c asked "how wide a hole can the model bridge" and could not answer it: widening a
held-out slab also re-shapes the marginal distribution of the remaining draws, so the
manipulation was not local to the edited wall (D53). For a DATASET-DESIGN rule the question
is better posed anyway -- not "how wide a hole" but "how coarsely may I sample" -- and that
is directly measurable with no slab and therefore no shoulder confound:

    train on a regular grid of interval Delta in m = -ln(1-alpha);
    test at the MIDPOINTS, which are maximally distant from every training value.

Three things here are not obvious and each would silently corrupt the sweep.

**1. The grid is endpoint-anchored, and the interval you get is not the interval you asked
for.** Anchoring n points inclusively on M_RANGE = [0.02, 1.61] is what reproduces the
intended counts (16/11/9/6/4), and it makes the realized interval
(1.61-0.02)/(n-1) = 0.1060 / 0.1590 / 0.1988 / 0.3180 / 0.5300 -- not the nominal
0.10 / 0.15 / 0.20 / 0.30 / 0.50 the run names carry. Every reported x and the fitted
Delta* use ``realized_delta``; the names are labels only. This is D53(c) ("the axis is the
REALIZED gap, never the nominal width") applied to a new axis.

**2. A discrete alphabet makes filename collisions likely, and nothing downstream catches
them.** ``sample_train_configs`` reuses (geometry, wall) slots -- 2 of the 6 wall-pairs
repeat per geometry, and each (geom, wall) carries 2-3 single-wall rows -- and a filename is
a pure function of (L, W, alphas). Continuous sampling made a collision probability-zero; a
grid does not. Measured on a naive i.i.d. substitution: **27 duplicate filenames of 960**.
The trainer does NOT check uniqueness (it only reads rows and opens files), so two configs
would quietly share one HDF5 file and one of them would train on the other's room. Draws are
therefore **collision-aware**: a value that would reproduce an already-used alphas tuple
*within the same geometry* is redrawn. Feasibility is not assumed -- the binding demand is
per slot (<=3 single-wall values per (geom, wall), 2 distinct pairs from n^2, 4 distinct
4-tuples from n^4), which is satisfiable at every grid including G050 with n = 4.

**3. Slab rejection is OFF; preset rejection becomes a grid-construction assertion.**
There is no held-out band in this design -- midpoints are held out by construction -- so
``in_slab`` must not fire. Presets must still never be trained, because the P3-2b S1/S4/S5
splits are built from them; but a grid value is fixed rather than drawn, so rejecting it
per-draw would break the "every training draw lies on the grid" invariant. The check moves
to grid construction (:func:`build_grid`), where a preset collision is a hard error.

The baseline alpha = 0.15 (m = 0.16252) is exempt from the grid and sits on every non-edited
wall of every config, so it is densely trained in every run. A midpoint that lands near it is
therefore NOT a hold-out; :func:`midpoints` flags those rather than dropping them silently.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from aaf.data.mat_configs import ALPHA_QUANT_DP, PRESET_ALPHAS
from aaf.data.mat_configs_cont import (
    M_RANGE,
    N_PER_GEOM,
    SEED,
    MatConfigM,
    _mk,
    alpha_of_m,
    m_of_alpha,
)
from aaf.walls import ALPHA_BASELINE, WALL_INDEX, WALLS_2D

# Nominal label -> number of grid points. The realized interval is derived, never assumed.
GRID_SPECS: Dict[str, int] = {
    "G010": 16,
    "G015": 11,
    "G020": 9,
    "G030": 6,
    "G050": 4,
}
GRID_ORDER: Tuple[str, ...] = ("G010", "G015", "G020", "G030", "G050")

M_BASELINE = m_of_alpha(ALPHA_BASELINE)

# A midpoint closer than this to a trained value is not a hold-out. 0.03 is ~1/5 of the
# finest realized interval (0.1060), so it flags a genuine coincidence without catching
# ordinary near-neighbours.
TRAINED_VALUE_TOL = 0.03

# A grid value this close to a preset alpha flips its filename token from 6 dp to 2 dp and
# would make a TRAINING room collide with a frozen TEST room.
PRESET_ALPHA_EPS = 1e-9


def realized_delta(n: int) -> float:
    """Interval of an ``n``-point grid anchored inclusively on ``M_RANGE``."""
    lo, hi = M_RANGE
    return (hi - lo) / (n - 1)


def build_grid(n: int) -> List[float]:
    """The ``n`` m-values, endpoint-anchored on ``M_RANGE``.

    Raises if any value collides with a preset alpha: the filename token would flip from the
    6-dp continuous form to the 2-dp preset form, and the training room would then share a
    filename with a frozen test room (see module docstring, point 3).
    """
    if n < 2:
        raise ValueError(f"grid needs >= 2 points, got {n}")
    lo, hi = M_RANGE
    d = realized_delta(n)
    grid = [lo + i * d for i in range(n)]
    for m in grid:
        a = round(alpha_of_m(m), ALPHA_QUANT_DP)
        for p in PRESET_ALPHAS:
            if abs(a - p) <= PRESET_ALPHA_EPS:
                raise AssertionError(
                    "grid value m={:.6f} -> alpha={:.6f} collides with preset {:.2f}; its "
                    "filename token would flip to 2 dp and collide with a frozen test room"
                    .format(m, a, p))
    return grid


def grid_alphas(n: int) -> List[float]:
    """Grid expressed in alpha, quantized exactly as ``_mk`` will store it."""
    return [round(alpha_of_m(m), ALPHA_QUANT_DP) for m in build_grid(n)]


def midpoints(n: int) -> List[dict]:
    """Midpoints between adjacent grid points -- the test values.

    Each carries ``near_trained_value``: True when the midpoint sits within
    ``TRAINED_VALUE_TOL`` of the always-trained baseline m. Those are reported as a labelled
    trained-value control and excluded from the headline curve, because the baseline is
    exempt from the grid and appears on every non-edited wall of every config, so such a
    midpoint is densely trained rather than held out. (Measured: this fires for the n=16 and
    n=6 grids, both at m ~ 0.179, alpha ~ 0.164.)
    """
    g = build_grid(n)
    out = []
    for i in range(len(g) - 1):
        m = 0.5 * (g[i] + g[i + 1])
        a = round(alpha_of_m(m), ALPHA_QUANT_DP)
        d_base = abs(m - M_BASELINE)
        out.append({
            "index": i,
            "m": float(m),
            "alpha": float(a),
            "bracketing_m": [float(g[i]), float(g[i + 1])],
            "d_to_nearest_train_m": float(min(abs(m - x) for x in g)),
            "d_to_baseline_m": float(d_base),
            "near_trained_value": bool(d_base <= TRAINED_VALUE_TOL),
            "headline": bool(d_base > TRAINED_VALUE_TOL),
        })
    return out


def near_preset_grid_values(n: int, tol: float = 0.02) -> List[dict]:
    """Grid values close enough to a preset m that the preset is effectively trained.

    Not an error -- the filename stays distinct, so nothing is corrupted -- but that preset's
    S1/S4 test configs are no longer a genuine hold-out in this run alone, which would make
    the cross-run comparison inconsistent if left unsaid. Measured: the n=9 grid has one at
    0.0085 in m.
    """
    out = []
    for m in build_grid(n):
        for p in PRESET_ALPHAS:
            d = abs(m - m_of_alpha(p))
            if d <= tol:
                out.append({"grid_m": float(m), "preset_alpha": float(p),
                            "preset_m": float(m_of_alpha(p)), "d_m": float(d)})
    return out


# --------------------------------------------------------------------------- sampling
def _draw(rng: np.random.Generator, grid_a: Sequence[float]) -> float:
    return float(grid_a[int(rng.integers(len(grid_a)))])


def _draw_unique(
    rng: np.random.Generator,
    grid_a: Sequence[float],
    walls: Sequence[str],
    used: set,
    max_tries: int = 10_000,
) -> List[float]:
    """Draw one value per wall such that the resulting alphas tuple is unused in this geometry.

    Redrawing inside the SAME rng stream keeps the sampler deterministic and keeps the
    P3-2b property that adding a geometry cannot perturb existing draws -- the stream is
    keyed on (seed, geom_id, local index) and collisions are resolved locally.
    """
    for _ in range(max_tries):
        vals = [_draw(rng, grid_a) for _ in walls]
        a = [ALPHA_BASELINE] * len(WALLS_2D)
        for w, v in zip(walls, vals):
            a[WALL_INDEX[w]] = v
        key = tuple(round(x, ALPHA_QUANT_DP) for x in a)
        if key not in used:
            used.add(key)
            return vals
    raise RuntimeError(
        "could not draw a collision-free config for walls {} from a {}-point grid after {} "
        "tries; the grid is too coarse for this slot's demand".format(
            list(walls), len(grid_a), max_tries))


def sample_train_configs_grid(
    geoms: Sequence[Sequence[float]], n: int, seed: int = SEED
) -> List[MatConfigM]:
    """40 geometries x 24 configs = 960 on a regular m-grid.

    Structurally identical to ``mat_configs_cont.sample_train_configs`` -- same loop, same
    ``N_PER_GEOM`` mix, same per-(geom, index) RNG keying -- so the config COUNT and the
    1/11/8/4 mix are preserved by construction. Only the value alphabet changes, plus the
    collision guard the discrete alphabet makes necessary.
    """
    grid_a = grid_alphas(n)
    pairs = [(i, j) for i in range(len(WALLS_2D)) for j in range(i + 1, len(WALLS_2D))]
    out: List[MatConfigM] = []
    for gid, geom in enumerate(geoms):
        L, W = geom[0], geom[1]
        used: set = set()
        base = _mk(L, W, [], "baseline", "train", gid)
        used.add(tuple(round(x, ALPHA_QUANT_DP) for x in base.alphas))
        out.append(base)
        for j in range(N_PER_GEOM["single"]):
            rng = np.random.default_rng([seed, gid, 100 + j])
            wall = WALLS_2D[(gid + j) % len(WALLS_2D)]
            (v,) = _draw_unique(rng, grid_a, [wall], used)
            out.append(_mk(L, W, [(wall, v)], "single", "train", gid))
        for j in range(N_PER_GEOM["two"]):
            rng = np.random.default_rng([seed, gid, 200 + j])
            i0, i1 = pairs[(j + 2 * gid) % len(pairs)]
            w0, w1 = WALLS_2D[i0], WALLS_2D[i1]
            v0, v1 = _draw_unique(rng, grid_a, [w0, w1], used)
            out.append(_mk(L, W, [(w0, v0), (w1, v1)], "two", "train", gid))
        for j in range(N_PER_GEOM["four"]):
            rng = np.random.default_rng([seed, gid, 300 + j])
            vs = _draw_unique(rng, grid_a, list(WALLS_2D), used)
            out.append(_mk(L, W, [(w, v) for w, v in zip(WALLS_2D, vs)],
                           "four", "train", gid))
    return out


def enumerate_midpoint_test_configs(
    geoms: Sequence[Sequence[float]], n: int
) -> List[MatConfigM]:
    """Single-wall edits at every midpoint, on all four walls x every test geometry.

    All four walls rather than west alone: the sampling law is a claim about the material
    axis, not about one wall, and per-wall slopes are what let the claim be checked. Kind is
    ``"single"`` so ``MatConfigM.strata`` groups these exactly like the training singles.
    """
    out: List[MatConfigM] = []
    for gid, geom in enumerate(geoms):
        L, W = geom[0], geom[1]
        for mp in midpoints(n):
            for w in WALLS_2D:
                out.append(_mk(L, W, [(w, mp["alpha"])], "midpoint", "test", gid))
    return out


def assert_grid_invariants(cfgs: Sequence[MatConfigM], n: int) -> Dict[str, object]:
    """Blocking gate assertions for one grid's TRAINING set.

    Deliberately different from ``mat_configs_cont.assert_train_invariants``: that one gates
    a continuous sampler (no draw on a preset, none in a slab) and its coverage sibling bins
    ``M_RANGE`` into 12 histogram cells and demands max/mean <= 2.0. A 4-point grid leaves 8
    of those bins empty by construction, so the continuous coverage test would fail a
    perfectly correct dataset. The grid analogue is "every draw is ON the grid and every grid
    value is used".
    """
    grid_a = grid_alphas(n)
    grid_set = {round(a, ALPHA_QUANT_DP) for a in grid_a}
    off_grid: List[dict] = []
    per_wall: Dict[str, List[float]] = {w: [] for w in WALLS_2D}
    used_values: set = set()
    for c in cfgs:
        for w in c.edited:
            a = round(c.alphas[WALL_INDEX[w]], ALPHA_QUANT_DP)
            per_wall[w].append(m_of_alpha(a))
            used_values.add(a)
            if a not in grid_set:
                off_grid.append({"label": c.label, "wall": w, "alpha": a})
    if off_grid:
        raise AssertionError(
            "{} training draws are not on the grid; first: {}".format(
                len(off_grid), off_grid[:3]))
    unused = sorted(grid_set - used_values)
    if unused:
        raise AssertionError(
            "{} grid values never used in training: {}".format(len(unused), unused))
    names = [c.filename for c in cfgs]
    if len(set(names)) != len(names):
        dupes = sorted({x for x in names if names.count(x) > 1})
        raise AssertionError(
            "duplicate filenames in the training manifest ({} of {} unique); first: {}. "
            "A discrete alphabet makes this likely and the trainer does not check it."
            .format(len(set(names)), len(names), dupes[:3]))
    kinds: Dict[str, int] = {}
    for c in cfgs:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
    return {
        "n_configs": len(cfgs),
        "n_grid_points": n,
        "realized_delta_m": realized_delta(n),
        "nominal_label": next((k for k, v in GRID_SPECS.items() if v == n), None),
        "kinds": kinds,
        "n_edited_draws": sum(len(v) for v in per_wall.values()),
        "per_wall_n": {w: len(v) for w, v in per_wall.items()},
        "grid_m": build_grid(n),
        "grid_alpha": grid_a,
        "n_grid_values_used": len(used_values),
        "near_preset_grid_values": near_preset_grid_values(n),
        "midpoints": midpoints(n),
        "n_headline_midpoints": sum(1 for mp in midpoints(n) if mp["headline"]),
    }
