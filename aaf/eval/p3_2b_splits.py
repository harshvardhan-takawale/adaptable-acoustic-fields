"""P3-2b evaluation splits, assigned from the FROZEN manifest -- never from the HDF5 attrs.

170 of the 210 test files are byte-identical reuses of P3-2 simulations and still carry
``split="train"`` (or P3-2's own split labels) in their HDF5 attributes. Reading the split
from the file would therefore silently misassign most of the test set, which is exactly the
kind of failure that produces a comparable-looking number from the wrong data. The manifest
(``configs/sweeps_2d_mat/p3_2b_manifest.json``, sha-pinned) is the only authority here.

Five splits, all single- or two-wall edits, all scored as paired deltas against their own
geometry's baseline:

  S1  unseen geom x non-slab single-wall preset edits          100   (transfer, easy)
  S2  unseen geom x SLAB combos (west@0.50, north@0.70)         20   <- HEADLINE
  S3  *seen* geom x slab combos                                 80   (decomposes S1->S2:
                                                                      combo novelty alone)
  S4  unseen geom x alpha=0.30                                  40   (material continuity)
  S5  unseen geom x two-wall preset edits                       40   (superposition)

Three assignment rules are load-bearing and each fixes a way P3-2's analysis could have
been wrong:

* **Edited walls are derived from the alphas vector** (``|a - 0.15| > 1e-9``), not from a
  scalar ``wall`` attribute. S5 configs edit two walls and a single-wall attribute cannot
  represent them; trusting one would drop or mislabel the whole split.
* **Slab membership is keyed on (wall, m), not on alpha.** The held-out slabs are per-wall:
  ``(west, 0.50)`` is held out while ``(east, 0.50)`` is trained. Keying on alpha alone
  would pull the trained twin into the headline split and destroy the C4 wall-identity
  control, whose whole construction is "held-out combo vs its trained opposite-wall twin".
* **S5 is tested BEFORE the slab branches.** ``(west, 0.50, south, 0.70)`` contains a slab
  value on a slab wall; it is a two-wall config and belongs in S5. It is additionally
  reported as an S5 slab subset so nothing is hidden by the ordering.

S3 is the one split not enumerable from the manifest: continuous training draws reject slab
values by construction, so no manifest row places a slab material on a training geometry.
Those 80 rooms exist on disk as P3-2's split (ii) and are re-derived here from
``p3_2_train.yaml`` x the two slab combos.

Baselines are ANCHORS, not a split: every delta in a geometry is taken against that
geometry's baseline render, and the baseline is also the ``d_m = 0`` point of every
slope fit. Counting them as split members would inflate ``n_configs`` and let a fidelity
mean be dominated by rooms with no edit in them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from aaf.data.mat_configs import PRESET_ALPHAS, room_filename_2d_mat_v2
from aaf.data.mat_configs_cont import HOLDOUT_SLABS, in_slab, m_of_alpha
from aaf.walls import ALPHA_BASELINE, WALL_INDEX, WALLS_2D

MANIFEST = "configs/sweeps_2d_mat/p3_2b_manifest.json"
TRAIN_YAML = "configs/sweeps_2d_mat/p3_2_train.yaml"
TEST_YAML = "configs/sweeps_2d_mat/p3_2_test_frozen.yaml"

# Contractual split names -- the figure script and the acceptance gate key off these.
S1 = "S1_unseen_geom_nonslab_1wall"
S2 = "S2_unseen_geom_slab"
S3 = "S3_seen_geom_slab"
S4 = "S4_unseen_geom_alpha030"
S5 = "S5_unseen_geom_2wall"
SPLIT_ORDER: Tuple[str, ...] = (S1, S2, S3, S4, S5)

# Frozen. A deviation is a mis-assignment, not a configuration choice (see assert_split_counts).
EXPECTED_COUNTS: Dict[str, int] = {S1: 100, S2: 20, S3: 80, S4: 40, S5: 40}

# The two (wall, alpha) combos whose m-slabs were withheld from every training draw.
SLAB_COMBOS: Tuple[Tuple[str, float], ...] = (("west", 0.50), ("north", 0.70))
UNSEEN_ALPHA = 0.30
EDIT_TOL = 1e-9

M_BASELINE = m_of_alpha(ALPHA_BASELINE)

# P3-2 material ids for the preset alphas. Carried forward ONLY so that the frozen
# ``p3_2_eval.control_c4`` -- which iterates ``HELDOUT_COMBOS = (("west","M2"),
# ("north","M3"))`` and looks up opposite-wall twins by that key -- can be imported and
# used verbatim. P3-2b's slabs bracket exactly those two combos, so the keys coincide.
PRESET_MATERIAL: Dict[float, str] = {0.05: "M1", 0.15: "M0", 0.50: "M2", 0.70: "M3"}


def _material_token(alpha: float) -> str:
    for a, mid in PRESET_MATERIAL.items():
        if abs(alpha - a) <= 1e-9:
            return mid
    if abs(alpha - UNSEEN_ALPHA) <= 1e-9:
        return "A030"
    return "a{:.3f}".format(float(alpha))


@dataclass(frozen=True)
class EvalConfig:
    """One room to evaluate, plus everything the driver needs to key it."""

    L: float
    W: float
    alphas: Tuple[float, float, float, float]
    edited: Tuple[str, ...]
    split: str
    geom_seen: bool          # geometry is in the training set (S3 only)
    source: str              # "manifest" | "derived_train_geom"

    @property
    def is_baseline(self) -> bool:
        return len(self.edited) == 0

    @property
    def filename(self) -> str:
        return room_filename_2d_mat_v2(self.L, self.W, self.alphas)

    @property
    def geom_key(self) -> Tuple[float, float]:
        return (round(self.L, 2), round(self.W, 2))

    @property
    def wall(self) -> Optional[str]:
        """The edited wall, or None for a multi-wall edit (C4/selectivity skip those)."""
        return self.edited[0] if len(self.edited) == 1 else None

    @property
    def material(self) -> Optional[str]:
        if self.is_baseline:
            return None
        if len(self.edited) == 1:
            return _material_token(self.alphas[WALL_INDEX[self.edited[0]]])
        return "+".join(
            "{}{:.2f}".format(w, self.alphas[WALL_INDEX[w]]) for w in self.edited)

    @property
    def combo_key(self) -> str:
        if self.is_baseline:
            return "baseline"
        return "+".join(
            "{}{:.2f}".format(w, self.alphas[WALL_INDEX[w]]) for w in self.edited)

    @property
    def label(self) -> str:
        return "L{:.2f}_W{:.2f}_{}".format(self.L, self.W, self.combo_key)

    @property
    def d_m(self) -> Dict[str, float]:
        """Per-edited-wall material step in the linearizing coordinate, vs the baseline."""
        return {w: m_of_alpha(self.alphas[WALL_INDEX[w]]) - M_BASELINE for w in self.edited}

    @property
    def touches_slab(self) -> bool:
        return any(in_slab(w, self.alphas[WALL_INDEX[w]]) for w in self.edited)


def edited_walls(alphas: Sequence[float], tol: float = EDIT_TOL) -> Tuple[str, ...]:
    """Walls whose absorption differs from the baseline. The ONLY definition of "edited".

    Derived from the 4-vector so that two-wall configs (S5) are representable; a single
    ``wall`` attribute silently collapses them onto their first wall.
    """
    return tuple(w for w in WALLS_2D if abs(float(alphas[WALL_INDEX[w]]) - ALPHA_BASELINE) > tol)


def _mk(L: float, W: float, alphas: Sequence[float], split: str, geom_seen: bool,
        source: str) -> EvalConfig:
    a = tuple(round(float(x), 6) for x in alphas)
    return EvalConfig(L=round(float(L), 2), W=round(float(W), 2), alphas=a,  # type: ignore
                      edited=edited_walls(a), split=split, geom_seen=geom_seen, source=source)


def classify(cfg_alphas: Sequence[float]) -> Optional[str]:
    """Split for an UNSEEN-geometry config, or None if it is a baseline.

    Branch order is load-bearing: the two-wall test comes first so a two-wall config that
    happens to carry a slab value lands in S5 rather than contaminating the headline split.
    """
    ed = edited_walls(cfg_alphas)
    if not ed:
        return None
    if len(ed) >= 2:
        return S5
    w = ed[0]
    a = float(cfg_alphas[WALL_INDEX[w]])
    if in_slab(w, a):
        return S2
    if abs(a - UNSEEN_ALPHA) <= 1e-9:
        return S4
    return S1


def load_manifest(path: str = MANIFEST) -> dict:
    return json.loads(Path(path).read_text())


def _geoms(path: str) -> List[Tuple[float, float]]:
    return [(round(float(g["L"]), 2), round(float(g["W"]), 2))
            for g in yaml.safe_load(Path(path).read_text())["geometries"]]


def build_splits(manifest: str = MANIFEST, train_yaml: str = TRAIN_YAML,
                 test_yaml: str = TEST_YAML) -> Tuple[Dict[str, List[EvalConfig]], dict]:
    """The five splits plus the anchor baselines and the geometry bookkeeping.

    Returns ``(splits, ctx)`` where ``ctx`` carries ``baselines`` (one EvalConfig per
    geometry that any split touches), ``test_geoms`` / ``train_geoms`` and the manifest sha.
    """
    man = load_manifest(manifest)
    rows = man["configs"]
    test_geoms = _geoms(test_yaml)
    train_geoms = _geoms(train_yaml)
    test_key_set = set(test_geoms)

    splits: Dict[str, List[EvalConfig]] = {k: [] for k in SPLIT_ORDER}
    unassigned: List[dict] = []
    baselines: Dict[Tuple[float, float], EvalConfig] = {}

    for r in rows:
        if r["split"] != "test":
            continue
        L, W = round(float(r["L"]), 2), round(float(r["W"]), 2)
        if (L, W) not in test_key_set:
            raise AssertionError(
                "manifest test row on a geometry absent from {}: {}".format(test_yaml, (L, W)))
        c = _mk(L, W, r["alphas"], "", False, "manifest")
        if c.is_baseline:
            baselines[(L, W)] = c
            continue
        name = classify(c.alphas)
        if name is None:
            unassigned.append(r)
            continue
        splits[name].append(_mk(L, W, r["alphas"], name, False, "manifest"))

    if unassigned:
        raise AssertionError(
            "{} non-baseline manifest test rows were left unassigned".format(len(unassigned)))

    # S3: slab combos on TRAINING geometries. Absent from the manifest by construction --
    # continuous training draws reject the slabs -- but present on disk as P3-2 split (ii).
    for (L, W) in train_geoms:
        for (wall, alpha) in SLAB_COMBOS:
            a = [ALPHA_BASELINE] * len(WALLS_2D)
            a[WALL_INDEX[wall]] = alpha
            splits[S3].append(_mk(L, W, a, S3, True, "derived_train_geom"))
        a0 = [ALPHA_BASELINE] * len(WALLS_2D)
        baselines[(L, W)] = _mk(L, W, a0, "", True, "derived_train_geom")

    ctx = {
        "baselines": baselines,
        "test_geoms": test_geoms,
        "train_geoms": train_geoms,
        "manifest_sha": man.get("rows_sha256", ""),
        "holdout_slabs": man.get("holdout_slabs", dict(HOLDOUT_SLABS)),
        "m_range": man.get("m_range"),
    }
    return splits, ctx


def assert_split_counts(splits: Dict[str, List[EvalConfig]]) -> None:
    """Blocking. A wrong count means configs landed in the wrong split, and every
    cross-split comparison in this chunk (S1 vs S2, S2 vs S3) then compares different
    populations -- a failure mode that produces plausible numbers and no error."""
    bad = []
    for name in SPLIT_ORDER:
        got, want = len(splits.get(name, [])), EXPECTED_COUNTS[name]
        if got != want:
            bad.append("{}: expected {}, got {}".format(name, want, got))
    extra = [k for k in splits if k not in EXPECTED_COUNTS]
    if extra:
        bad.append("unexpected split keys: {}".format(sorted(extra)))
    for name in SPLIT_ORDER:
        for c in splits.get(name, []):
            if c.is_baseline:
                bad.append("{} contains a baseline ({}) -- baselines are anchors".format(
                    name, c.label))
            if c.split != name:
                bad.append("{} contains a config labelled {}".format(name, c.split))
    seen = set()
    for name in SPLIT_ORDER:
        for c in splits.get(name, []):
            k = (c.geom_key, c.alphas)
            if k in seen:
                bad.append("config {} assigned to more than one split".format(c.label))
            seen.add(k)
    if bad:
        raise AssertionError("split assignment is wrong:\n  " + "\n  ".join(bad))


def s5_slab_subset(splits: Dict[str, List[EvalConfig]]) -> List[EvalConfig]:
    """Two-wall configs that contain a slab value -- reported separately so the
    S5-before-slab branch order cannot hide them."""
    return [c for c in splits.get(S5, []) if c.touches_slab]


def slab_summary() -> dict:
    """The ``slabs_m`` block of summary.json."""
    out = {"holdout_slabs_m": {w: list(v) for w, v in HOLDOUT_SLABS.items()},
           "m_of_alpha_baseline": M_BASELINE,
           "slab_combos": [[w, a] for w, a in SLAB_COMBOS],
           "preset_m": {"{:.2f}".format(a): m_of_alpha(a) for a in PRESET_ALPHAS},
           "d_m_vs_baseline": {"{:.2f}".format(a): m_of_alpha(a) - M_BASELINE
                               for a in PRESET_ALPHAS}}
    for w, alpha in SLAB_COMBOS:
        out.setdefault("slab_check", {})["{}_{:.2f}".format(w, alpha)] = {
            "m": m_of_alpha(alpha), "in_slab": bool(in_slab(w, alpha)),
            "twin_in_slab": bool(in_slab({"west": "east", "north": "south"}[w], alpha))}
    return out
