"""P3-2b: continuous absorption sampling in the linearizing coordinate m = -ln(1-alpha).

P3-2 sampled alpha at 3 preset values on single walls only, giving 11 distinct alpha-vectors
in a 4-D space. The model memorized (wall, alpha) pairs instead of learning the per-wall law:
for M3 y-axial, the TRAINED wall reproduced the edit (GT 11.48 -> pred 12.67) while the
held-out wall one over produced nothing (GT 12.33 -> pred 1.07).

This module replaces that with dense continuous sampling, in the coordinate where the target
is exactly linear. Three properties are load-bearing:

* **Uniform in m, not in alpha.** The ISM-ray law is linear in m, so uniform-in-m gives
  uniform coverage of the RESPONSE space; uniform-in-alpha would over-sample the flat end.
* **Held-out slabs are INTERIOR** to the sampled range. P3-2's (north, 0.70) holdout was an
  extrapolation on that wall's own axis (north trained only up to 0.50) while (west, 0.50)
  was an interpolation -- so the two "held-out combos" were never comparable tests. Both
  slabs here are interior, making S2 a pure composition test.
* **Rejection applies to every draw**, including the multi-wall configs, or a slab value
  would leak in through a two-wall or four-wall sample.

Because alpha is drawn continuously, the demo presets (0.05 / 0.50 / 0.70) have probability
zero of appearing exactly in training -- every preset evaluation is at an unseen exact value
by construction.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from aaf.data.mat_configs import (
    ALPHA_QUANT_DP,
    PRESET_ALPHAS,
    room_filename_2d_mat_v2,
    round_dim,
)
from aaf.walls import ALPHA_BASELINE, WALL_INDEX, WALLS_2D

M_RANGE: Tuple[float, float] = (0.02, 1.61)          # alpha in [0.0198, 0.8001]
HOLDOUT_SLABS: Dict[str, Tuple[float, float]] = {
    "west": (0.62, 0.77),      # brackets alpha = 0.50 (m = 0.69315)
    "north": (1.13, 1.28),     # brackets alpha = 0.70 (m = 1.20397)
}
N_PER_GEOM = {"baseline": 1, "single": 11, "two": 8, "four": 4}   # = 24
SEED = 20260813

# Four two-wall preset combos for the test set (S5), as given in the chunk spec.
TEST_TWO_WALL: Tuple[Tuple[str, float, str, float], ...] = (
    ("west", 0.50, "south", 0.70),
    ("east", 0.05, "north", 0.50),
    ("west", 0.70, "east", 0.50),
    ("south", 0.05, "north", 0.70),
)
TEST_SINGLE_PRESETS = (0.05, 0.50, 0.70)
TEST_SMOOTHNESS_ALPHA = 0.30


def m_of_alpha(alpha: float) -> float:
    return -math.log1p(-float(alpha))


def alpha_of_m(m: float) -> float:
    return 1.0 - math.exp(-float(m))


def in_slab(wall: str, alpha: float) -> bool:
    """Is this (wall, alpha) inside that wall's held-out slab?"""
    sl = HOLDOUT_SLABS.get(wall)
    if sl is None:
        return False
    m = m_of_alpha(alpha)
    return sl[0] <= m <= sl[1]


def _is_preset(alpha: float, tol: float = 1e-6) -> bool:
    return any(abs(alpha - p) <= tol for p in PRESET_ALPHAS)


@dataclass(frozen=True)
class MatConfigM:
    """One simulated room with arbitrary per-wall absorption."""

    L: float
    W: float
    alphas: Tuple[float, float, float, float]     # WALLS_2D order
    kind: str                                     # baseline|single|two|four|preset|preset_two
    edited: Tuple[str, ...]
    split: str                                    # train | test
    geom_id: int

    @property
    def is_baseline(self) -> bool:
        return len(self.edited) == 0

    @property
    def m(self) -> Tuple[float, ...]:
        return tuple(m_of_alpha(a) for a in self.alphas)

    @property
    def filename(self) -> str:
        return room_filename_2d_mat_v2(self.L, self.W, self.alphas)

    @property
    def label(self) -> str:
        if self.is_baseline:
            return "L{:.2f}_W{:.2f}_baseline".format(self.L, self.W)
        parts = "+".join(
            "{}{:.3f}".format(w, self.alphas[WALL_INDEX[w]]) for w in self.edited)
        return "L{:.2f}_W{:.2f}_{}".format(self.L, self.W, parts)

    @property
    def strata(self) -> str:
        """Coarse validation stratum. NOT the (wall, material) combo: with continuous alpha
        every config would be its own combo, which degenerates the val subsample onto the
        first few geometries (a live bug in the P3-2 trainer)."""
        if self.is_baseline:
            return "baseline"
        if self.kind in ("single", "preset"):
            return "single_" + self.edited[0]
        return self.kind

    # Duck-type compatibility with MatConfig so the trainer/eval need no branching.
    @property
    def wall(self) -> Optional[str]:
        return self.edited[0] if len(self.edited) == 1 else None

    @property
    def material(self) -> Optional[str]:
        return None if self.is_baseline else "+".join(self.edited)


def _mk(L, W, edits, kind, split, geom_id) -> MatConfigM:
    """edits: sequence of (wall, alpha). Non-edited walls sit at the baseline."""
    a = [ALPHA_BASELINE] * len(WALLS_2D)
    for w, v in edits:
        a[WALL_INDEX[w]] = round(float(v), ALPHA_QUANT_DP)
    return MatConfigM(
        L=round_dim(L), W=round_dim(W), alphas=tuple(a), kind=kind,
        edited=tuple(w for w, _ in edits), split=split, geom_id=geom_id)


def draw_alpha(rng: np.random.Generator, wall: str) -> float:
    """Draw m ~ U(M_RANGE), reject into the wall's slab or onto a preset, return alpha.

    Quantizes ALPHA (not m) so the baseline stays exactly 0.15 and the C3 identity holds.
    """
    for _ in range(10_000):
        m = float(rng.uniform(*M_RANGE))
        a = round(alpha_of_m(m), ALPHA_QUANT_DP)
        if in_slab(wall, a) or _is_preset(a):
            continue
        return a
    raise RuntimeError(f"rejection sampling failed for wall {wall!r}")


def sample_train_configs(geoms: Sequence[Sequence[float]], seed: int = SEED) -> List[MatConfigM]:
    """40 geometries x 24 configs = 960, reproducible and order-independent.

    The RNG is keyed on (seed, geom_id, local index) rather than drawn from one sequential
    stream, so every SLURM array task rebuilds the identical manifest and adding a geometry
    cannot perturb the existing draws.
    """
    pairs = [(i, j) for i in range(len(WALLS_2D)) for j in range(i + 1, len(WALLS_2D))]
    out: List[MatConfigM] = []
    for gid, geom in enumerate(geoms):
        L, W = geom[0], geom[1]
        out.append(_mk(L, W, [], "baseline", "train", gid))
        for j in range(N_PER_GEOM["single"]):
            rng = np.random.default_rng([seed, gid, 100 + j])
            wall = WALLS_2D[(gid + j) % len(WALLS_2D)]      # balanced across walls
            out.append(_mk(L, W, [(wall, draw_alpha(rng, wall))], "single", "train", gid))
        for j in range(N_PER_GEOM["two"]):
            rng = np.random.default_rng([seed, gid, 200 + j])
            i0, i1 = pairs[(j + 2 * gid) % len(pairs)]
            w0, w1 = WALLS_2D[i0], WALLS_2D[i1]
            out.append(_mk(L, W, [(w0, draw_alpha(rng, w0)), (w1, draw_alpha(rng, w1))],
                           "two", "train", gid))
        for j in range(N_PER_GEOM["four"]):
            rng = np.random.default_rng([seed, gid, 300 + j])
            out.append(_mk(L, W, [(w, draw_alpha(rng, w)) for w in WALLS_2D],
                           "four", "train", gid))
    return out


def enumerate_test_configs(geoms: Sequence[Sequence[float]]) -> List[MatConfigM]:
    """10 geometries x 21 = 210. The first 17 per geometry use only preset alphas, so their
    v2 filenames are byte-identical to P3-2's and those simulations are reused."""
    out: List[MatConfigM] = []
    for gid, geom in enumerate(geoms):
        L, W = geom[0], geom[1]
        out.append(_mk(L, W, [], "baseline", "test", gid))
        for w in WALLS_2D:
            for a in TEST_SINGLE_PRESETS:
                out.append(_mk(L, W, [(w, a)], "preset", "test", gid))
        for w in WALLS_2D:
            out.append(_mk(L, W, [(w, TEST_SMOOTHNESS_ALPHA)], "preset", "test", gid))
        for (w0, a0, w1, a1) in TEST_TWO_WALL:
            out.append(_mk(L, W, [(w0, a0), (w1, a1)], "preset_two", "test", gid))
    return out


def assert_train_invariants(cfgs: Sequence[MatConfigM]) -> Dict[str, object]:
    """Blocking dataset-gate assertions. Returns a report for DATASET_GATE.md."""
    per_wall_m: Dict[str, List[float]] = {w: [] for w in WALLS_2D}
    n_slab = n_preset = 0
    for c in cfgs:
        for w in c.edited:
            a = c.alphas[WALL_INDEX[w]]
            per_wall_m[w].append(m_of_alpha(a))
            n_slab += in_slab(w, a)
            n_preset += _is_preset(a)
    if n_slab:
        raise AssertionError(f"{n_slab} training draws fell inside a held-out slab")
    if n_preset:
        raise AssertionError(f"{n_preset} training draws landed exactly on a preset alpha")
    names = [c.filename for c in cfgs]
    if len(set(names)) != len(names):
        raise AssertionError("duplicate filenames in the training manifest")
    return {
        "n_configs": len(cfgs),
        "n_edited_draws": sum(len(v) for v in per_wall_m.values()),
        "per_wall_n": {w: len(v) for w, v in per_wall_m.items()},
        "per_wall_m_min": {w: (min(v) if v else None) for w, v in per_wall_m.items()},
        "per_wall_m_max": {w: (max(v) if v else None) for w, v in per_wall_m.items()},
        "slab_violations": n_slab,
        "preset_collisions": n_preset,
    }


def manifest_rows(train: Sequence[MatConfigM], test: Sequence[MatConfigM]) -> List[dict]:
    rows = []
    for i, c in enumerate(list(train) + list(test)):
        rows.append({
            "i": i, "split": c.split, "kind": c.kind, "geom_id": c.geom_id,
            "L": c.L, "W": c.W, "alphas": list(c.alphas), "m": list(c.m),
            "edited": list(c.edited), "filename": c.filename, "strata": c.strata,
        })
    return rows


def rows_sha256(rows: Sequence[dict]) -> str:
    return hashlib.sha256(
        json.dumps(list(rows), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def configs_from_rows(rows: Sequence[dict], split: Optional[str] = None,
                      kinds: Sequence[str] = ()) -> List[MatConfigM]:
    """Rebuild config objects from a frozen manifest (what the trainer/eval consume)."""
    out = []
    for r in rows:
        if split is not None and r["split"] != split:
            continue
        if kinds and r["kind"] not in kinds:
            continue
        out.append(MatConfigM(
            L=float(r["L"]), W=float(r["W"]), alphas=tuple(float(x) for x in r["alphas"]),
            kind=r["kind"], edited=tuple(r["edited"]), split=r["split"],
            geom_id=int(r["geom_id"])))
    return out
