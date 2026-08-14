"""P3-2c: the slab-width sweep — arm specs and the DERIVATION sampler.

P3-2c widens the held-out band on the west absorption axis to measure how large a gap the
model can bridge. The scientific requirement is that arms differ ONLY in that band, so any
change in transfer is attributable to gap width and nothing else.

Re-running ``sample_train_configs`` per arm would violate that. ``draw_alpha`` rejects and
redraws inside a loop, so a wider slab consumes a different number of rng values; and since
``WALLS_2D`` puts west first, a west rejection perturbs east/south/north in every four-wall
config and in the three wall-pairs containing west. The arms would then differ everywhere.

So we DERIVE instead. Each arm's value for a given (geometry, row, wall) is the first entry
of ``[W015_value] + repair_stream(...)`` that the arm's own predicate accepts, where the
repair stream is keyed on POSITION ONLY and never on the arm:

    rng = default_rng([REPAIR_SEED, geom_id, row_i, wall_slot])

Two arms therefore assign different absorptions to a config **iff** at least one of them
rejects the other's value -- i.e. iff the value lies in the symmetric difference of their
holdout regions. Every other config is byte-identical, hence so is its filename, hence so is
its ``.h5`` and its ``.done`` sentinel, so the datasets share simulations automatically.

Three consequences worth stating:
  * Re-deriving with ``spec=W015`` returns the frozen P3-2b manifest with ZERO configs
    changed, so reusing the already-trained arm C as the sweep's first point is EXACT, not an
    approximation.
  * North is held at [1.13, 1.28] in every arm, so every north draw is byte-identical across
    all five manifests -- the within-run control is literally the same rooms.
  * Rejection applies ONLY to edited walls. A wall at exactly the baseline alpha = 0.15
    (m = 0.1625) is the reference configuration and is never drawn, so it can never be
    rejected even by W100 whose slab [0.193, 1.193] would otherwise contain it. This is true
    by construction (``_mk`` hard-sets non-edited walls) and is asserted in the dataset gate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from aaf.data.mat_configs import ALPHA_QUANT_DP, PRESET_ALPHAS
from aaf.data.mat_configs_cont import (
    M_RANGE,
    MatConfigM,
    alpha_of_m,
    configs_from_rows,
    m_of_alpha,
    manifest_rows,
    rows_sha256,
)
from aaf.walls import WALL_INDEX, WALLS_2D

REPAIR_SEED = 20260814
NORTH_SLAB: Tuple[float, float] = (1.13, 1.28)

# alpha values that must stay unseen in training so every test point is a genuine holdout.
# P3-2b's presets plus XTRAP's two new extrapolation alphas.
P3_2C_EXTRA_PRESETS: Tuple[float, ...] = (0.75, 0.80)
ALL_UNSEEN_ALPHAS: Tuple[float, ...] = tuple(PRESET_ALPHAS) + P3_2C_EXTRA_PRESETS


@dataclass(frozen=True)
class ArmSpec:
    """One point on the density curve."""

    name: str
    run_id: str
    west: Optional[Tuple[float, float]] = None   # interior slab (m); None if edge-excluded
    west_max: Optional[float] = None             # XTRAP: exclude m_west above this
    north: Tuple[float, float] = NORTH_SLAB
    nominal_width: Optional[float] = None
    reuse_of: Optional[str] = None               # W015 -> the frozen P3-2b arm C

    def rejects(self, wall: str, alpha: float) -> bool:
        """Does THIS arm exclude (wall, alpha) from training?"""
        m = m_of_alpha(alpha)
        if wall == "west":
            if self.west is not None and self.west[0] <= m <= self.west[1]:
                return True
            if self.west_max is not None and m > self.west_max:
                return True
        elif wall == "north":
            if self.north[0] <= m <= self.north[1]:
                return True
        return False

    def slabs(self) -> Dict[str, Tuple[float, float]]:
        out = {"north": self.north}
        if self.west is not None:
            out["west"] = self.west
        elif self.west_max is not None:
            out["west"] = (self.west_max, float(M_RANGE[1]) + 1.0)
        return out


# The sweep. W015 is P3-2b arm C, recorded with its ACTUAL slab (0.62, 0.77) rather than
# relabelled to the nominal [0.618, 0.768]: the centres differ by 0.002 in m, but no frozen
# draw lies in the symmetric difference and S2's test point (west@0.50, m=0.6931) is
# strictly interior to both, so the training set and S2 are byte-identical either way.
SPECS: Dict[str, ArmSpec] = {
    "W015": ArmSpec("W015", "p3_2b_C_cont_mlinear", west=(0.62, 0.77),
                    nominal_width=0.15, reuse_of="p3_2b"),
    "W030": ArmSpec("W030", "p3_2c_W030_mlinear", west=(0.543, 0.843), nominal_width=0.30),
    "W060": ArmSpec("W060", "p3_2c_W060_mlinear", west=(0.393, 0.993), nominal_width=0.60),
    "W100": ArmSpec("W100", "p3_2c_W100_mlinear", west=(0.193, 1.193), nominal_width=1.00),
    "XTRAP": ArmSpec("XTRAP", "p3_2c_XTRAP_mlinear", west=None, west_max=1.10),
}
SWEEP_ORDER: Tuple[str, ...] = ("W015", "W030", "W060", "W100")


def _is_unseen_preset(alpha: float, tol: float = 1e-6) -> bool:
    return any(abs(alpha - p) <= tol for p in ALL_UNSEEN_ALPHAS)


def repair_stream(geom_id: int, row_i: int, wall_slot: int) -> np.random.Generator:
    """Position-keyed and ARM-INDEPENDENT. This is the whole trick."""
    return np.random.default_rng([REPAIR_SEED, int(geom_id), int(row_i), int(wall_slot)])


def _repaired_alpha(spec: ArmSpec, wall: str, geom_id: int, row_i: int,
                    wall_slot: int, base_alpha: float) -> float:
    """First acceptable value of [base] + repair_stream, under THIS arm's predicate."""
    if not spec.rejects(wall, base_alpha) and not _is_unseen_preset(base_alpha):
        return base_alpha
    rng = repair_stream(geom_id, row_i, wall_slot)
    for _ in range(10_000):
        a = round(alpha_of_m(float(rng.uniform(*M_RANGE))), ALPHA_QUANT_DP)
        if not spec.rejects(wall, a) and not _is_unseen_preset(a):
            return a
    raise RuntimeError(f"repair failed for {spec.name} wall={wall} g={geom_id} r={row_i}")


def derive_train_rows(base_rows: Sequence[dict], spec: ArmSpec) -> List[dict]:
    """Derive this arm's training rows from the frozen W015 rows."""
    out: List[dict] = []
    for r in base_rows:
        if r["split"] != "train":
            continue
        alphas = list(r["alphas"])
        for wall in r["edited"]:
            slot = WALL_INDEX[wall]
            alphas[slot] = _repaired_alpha(
                spec, wall, r["geom_id"], r["i"], slot, float(alphas[slot]))
        row = dict(r)
        row["alphas"] = alphas
        row["m"] = [m_of_alpha(a) for a in alphas]
        cfg = MatConfigM(L=float(r["L"]), W=float(r["W"]), alphas=tuple(alphas),
                         kind=r["kind"], edited=tuple(r["edited"]), split="train",
                         geom_id=int(r["geom_id"]))
        row["filename"] = cfg.filename
        out.append(row)
    return out


def manifest_delta(base_rows: Sequence[dict], derived: Sequence[dict]) -> dict:
    """How many configs actually moved, and how many walls moved within them."""
    base = {r["i"]: r for r in base_rows if r["split"] == "train"}
    changed, wall_moves = 0, 0
    for r in derived:
        b = base[r["i"]]
        moved = [j for j in range(4) if abs(b["alphas"][j] - r["alphas"][j]) > 1e-12]
        if moved:
            changed += 1
            wall_moves += len(moved)
    return {"n_train": len(derived), "n_changed": changed,
            "n_unchanged": len(derived) - changed, "n_wall_moves": wall_moves}


def realized_gap(rows: Sequence[dict], wall: str = "west") -> dict:
    """The gap the model actually sees: the largest nearest-neighbour hole in m.

    This is the honest x-axis for the density curve -- the nominal slab width is only an
    upper bound on it, because draws do not land exactly on the slab edges.
    """
    slot = WALL_INDEX[wall]
    ms = sorted(m_of_alpha(r["alphas"][slot]) for r in rows if wall in r["edited"])
    if len(ms) < 2:
        return {"n_draws": len(ms), "max_gap_m": None, "bracketing_m": None}
    gaps = [(ms[i + 1] - ms[i], ms[i], ms[i + 1]) for i in range(len(ms) - 1)]
    g, lo, hi = max(gaps)
    return {"n_draws": len(ms), "max_gap_m": float(g),
            "bracketing_m": [float(lo), float(hi)],
            "density_per_unit_m": float(len(ms) / (M_RANGE[1] - M_RANGE[0] - g))}


def edge_distances(rows: Sequence[dict], spec: ArmSpec,
                    test_alphas: Sequence[float] = (0.70, 0.75, 0.80),
                    wall: str = "west") -> dict:
    """For an EDGE-excluded arm (XTRAP), the meaningful axis is not an interior gap but the
    distance of each test point beyond the training edge. ``realized_gap`` would report the
    ordinary sampling gap (~0.03) and be quietly meaningless here."""
    if spec.west_max is None:
        return {}
    slot = WALL_INDEX[wall]
    ms = [m_of_alpha(r["alphas"][slot]) for r in rows if wall in r["edited"]]
    edge = max(ms) if ms else None
    return {
        "train_edge_m": float(edge) if edge is not None else None,
        "exclusion_threshold_m": float(spec.west_max),
        "points": [{"alpha": float(a), "m": float(m_of_alpha(a)),
                    "beyond_edge_m": float(m_of_alpha(a) - edge) if edge else None}
                   for a in test_alphas],
    }


def build_arm_manifest(base_manifest: str, spec: ArmSpec) -> dict:
    """Full manifest for one arm: derived train rows + the unchanged frozen test rows."""
    base = json.load(open(base_manifest))
    base_rows = base["configs"]
    train = derive_train_rows(base_rows, spec)
    test = [dict(r) for r in base_rows if r["split"] == "test"]
    rows = train + test
    for k, r in enumerate(rows):
        r["i"] = k
    return {
        "schema": "p3_2c.manifest/1",
        "arm": spec.name, "run_id": spec.run_id,
        "derived_from": base_manifest,
        "repair_seed": REPAIR_SEED,
        "west_slab": list(spec.west) if spec.west else None,
        "west_max": spec.west_max,
        "north_slab": list(spec.north),
        "nominal_width": spec.nominal_width,
        "m_range": list(M_RANGE),
        "unseen_alphas": list(ALL_UNSEEN_ALPHAS),
        "n_train": len(train), "n_test": len(test),
        "delta_vs_W015": manifest_delta(base_rows, train),
        "realized_gap_west": realized_gap(train, "west"),
        "realized_gap_north": realized_gap(train, "north"),
        "edge_distances_west": edge_distances(train, spec),
        "gap_axis": ("beyond_edge" if spec.west_max is not None else "interior_gap"),
        "rows_sha256": rows_sha256(rows),
        "configs": rows,
    }
