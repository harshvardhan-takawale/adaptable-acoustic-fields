"""P3-3-FAST Track 2b: doorway-APERTURE configs (a two-room domain split by one divider).

A room is a shoebox ``L x W`` with a one-node-thick interior slab at ``x = x0``, and the edit
axis is the doorway width ``a``. Three things about that axis are load-bearing:

**The conditioning coordinate is sqrt(a), not a.** FT-B swept a in {0, 0.1, ..., 4.0} at
dx = 0.01 and fitted the inter-room level difference against six candidate coordinates:
``sqrt a`` is the linearizing one, pooled r^2 = 0.9870 (raw ``a`` gives 0.905, ``a^2`` 0.704).
Same role that m = -ln(1-alpha) plays on the absorption axis. See
``outputs/p3_3fast/trackB/aperture_sweep.json``.

**a = 0 is a TOPOLOGICAL discontinuity, not the small-aperture limit.** A sealed one-node
divider disconnects room B *exactly*: H_B is identically zero, so the inter-room level
difference is -inf rather than merely large. No continuous coordinate can contain that point
-- the trainable range is a in (0, W]. Sealed configs are still built and kept (they are the
end-point demonstration, and the dataset gate uses one to prove the divider plumbing works),
but they carry ``sealed = True`` in the manifest, are their own ``kind``, and must be excluded
from every continuous-coordinate fit and from training (see ``configs/sweep_2d_mat/
P3_3FAST_trackB.yaml``, which filters them out via ``config_kinds``).

**The hold-out is a BAND in a, not a set of geometries.** No training config has
a in [0.9, 1.1]; draws landing inside are redrawn, so the hold-out is exact rather than
approximate, and three test configs per test domain sit inside it (0.95, 1.00, 1.05). That
makes the generalization question "did the model learn the aperture law?" rather than "did it
memorize 20 aperture values?".

Geometry: L in [7.0, 9.0], W in [3.5, 4.5], x0 in [0.4L, 0.6L] -- both sub-rooms stay at
least 2.8 m long, and with a <= 2.5 at least 0.5 m of divider survives on each side of the
doorway.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aaf.walls import ALPHA_BASELINE, WALLS_2D

# ------------------------------------------------------------------ sampling box (frozen)
L_RANGE: Tuple[float, float] = (7.0, 9.0)
W_RANGE: Tuple[float, float] = (3.5, 4.5)
X0_FRAC_RANGE: Tuple[float, float] = (0.4, 0.6)

#: Continuous aperture range for TRAINING draws. 0.1 m is 10 cells at dx = 0.01; FT-B dropped
#: 0.05 m as under-resolved (``_apply_slab`` needs >= 3 open nodes, so the staircased tips
#: would occupy 2 of 5 cells).
A_RANGE: Tuple[float, float] = (0.1, 2.5)

#: EXACT hold-out band. Training draws inside are redrawn; test configs deliberately sit here.
A_HOLDOUT: Tuple[float, float] = (0.9, 1.1)
A_HOLDOUT_TEST_VALUES: Tuple[float, ...] = (0.95, 1.0, 1.05)

#: Test apertures outside the band: two below 0.9, five above 1.1, spanning the trained range.
A_TEST_VALUES: Tuple[float, ...] = (0.15, 0.30, 0.50, 0.70, 1.50, 2.00, 2.50)

#: sqrt-normalizer for the conditioning coordinate: sqrt(a) / sqrt(A_NORM). Frozen at the
#: FT-B domain width (W = 4.0) so the coordinate does not move when W does.
A_NORM = 4.0

DIVIDER_ALPHA = ALPHA_BASELINE          # 0.15, same as the four outer walls
A_QUANT_DP = 4
GEOM_QUANT_DP = 2
N_PER_TRAIN_DOMAIN = 20                 # 1 sealed + 1 fully-open + 18 drawn apertures
N_TRAIN_DOMAINS = 20
SEED_DEFAULT = 20260817

#: 6 frozen test domains, all INTERIOR to the training box (L in [7.25, 8.90] vs [7.0, 9.0],
#: W in [3.60, 4.40] vs [3.5, 4.5], x0/L in [0.441, 0.573] vs [0.4, 0.6]), so every test room
#: is an interpolation in geometry and the only extrapolation under test is the aperture.
TEST_DOMAINS: Tuple[Tuple[float, float, float], ...] = (
    (7.25, 3.60, 3.20),
    (7.60, 4.40, 4.10),
    (8.00, 3.85, 3.60),
    (8.35, 4.15, 4.60),
    (8.70, 3.70, 4.00),
    (8.90, 4.30, 5.10),
)


def in_holdout(a: float) -> bool:
    """Is ``a`` inside the exact hold-out band [0.9, 1.1]? Closed on both ends."""
    return A_HOLDOUT[0] - 1e-12 <= float(a) <= A_HOLDOUT[1] + 1e-12


def sqrt_a_hat(a: float) -> float:
    """The linearizing conditioning coordinate, sqrt(a) / sqrt(4.0) (FT-B, r^2 = 0.9870)."""
    return float(np.sqrt(max(0.0, float(a))) / np.sqrt(A_NORM))


def aperture_filename(L: float, W: float, x0: float, a: float) -> str:
    return "L{:.2f}_W{:.2f}_x{:.2f}_a{:.4f}.h5".format(
        round(float(L), GEOM_QUANT_DP), round(float(W), GEOM_QUANT_DP),
        round(float(x0), GEOM_QUANT_DP), round(float(a), A_QUANT_DP))


@dataclass(frozen=True)
class ApertureConfig:
    L: float
    W: float
    x0: float          # divider position along x, metres
    a: float           # doorway clear width, metres; 0 = sealed, >= W = no divider at all
    kind: str
    split: str
    geom_id: int

    # -- flags ---------------------------------------------------------------------------
    @property
    def sealed(self) -> bool:
        """A sealed divider disconnects room B EXACTLY (H_B == 0). Not a continuous point."""
        return float(self.a) <= 0.0

    @property
    def fully_open(self) -> bool:
        """``a >= W`` means no divider at all -- the plain L x W shoebox."""
        return float(self.a) >= float(self.W) - 1e-9

    # -- identity ------------------------------------------------------------------------
    @property
    def filename(self) -> str:
        return aperture_filename(self.L, self.W, self.x0, self.a)

    @property
    def label(self) -> str:
        tag = "sealed" if self.sealed else ("open" if self.fully_open
                                            else "a{:.3f}".format(self.a))
        return "L{:.2f}_W{:.2f}_x{:.2f}_{}".format(self.L, self.W, self.x0, tag)

    @property
    def strata(self) -> str:
        """Coarse validation stratum. Keying on the (continuous) aperture would make every
        config its own singleton group and collapse the val subsample -- the P3-2b lesson."""
        return self.kind

    @property
    def geom_key(self) -> Tuple[float, float, float]:
        return (self.L, self.W, self.x0)

    # -- physics -------------------------------------------------------------------------
    @property
    def alphas(self) -> Tuple[float, ...]:
        """The FOUR outer-wall absorptions, all baseline. The aperture is carried separately
        (``extra_walls``); this property exists so every downstream loader / trainer / eval
        that expects a 4-vector keeps working unchanged."""
        return (ALPHA_BASELINE,) * len(WALLS_2D)

    @property
    def x0_frac(self) -> float:
        return float(self.x0) / float(self.L)

    @property
    def sqrt_a_hat(self) -> float:
        return sqrt_a_hat(self.a)

    @property
    def extra_walls(self) -> List[Dict[str, Any]]:
        """The divider spec handed to ``aaf.sim.fdtd_2d.simulate``.

        * sealed (a == 0): a full-span slab with NO apertures key -- room B disconnects.
        * fully open (a >= W): the empty list, i.e. no interior structure at all. Passing a
          slab with a W-wide aperture instead would leave the two staircased tips in place.
        * otherwise: one slab with one aperture, centred on the divider at y = W/2.
        """
        if self.fully_open:
            return []
        spec: Dict[str, Any] = {"type": "slab", "axis": "x", "pos": float(self.x0),
                                "alpha": DIVIDER_ALPHA}
        if not self.sealed:
            half = 0.5 * float(self.a)
            spec["apertures"] = [(0.5 * float(self.W) - half, 0.5 * float(self.W) + half)]
        return [spec]


def _mk(L, W, x0, a, kind, split, gid) -> ApertureConfig:
    return ApertureConfig(L=round(float(L), GEOM_QUANT_DP), W=round(float(W), GEOM_QUANT_DP),
                          x0=round(float(x0), GEOM_QUANT_DP),
                          a=round(float(a), A_QUANT_DP),
                          kind=kind, split=split, geom_id=int(gid))


# ----------------------------------------------------------------------------- geometries
def sample_train_domains(n: int = N_TRAIN_DOMAINS,
                         seed: int = SEED_DEFAULT) -> List[Tuple[float, float, float]]:
    """``n`` distinct (L, W, x0) triples, reproducible from ``seed``.

    Rejects any triple that duplicates another training domain or one of the six frozen test
    domains -- the filename carries only (L, W, x0, a), so a duplicated geometry would alias
    two configs onto one HDF5 file (the P3-2c collision hazard).
    """
    rng = np.random.default_rng(seed)
    taken = {(round(L, GEOM_QUANT_DP), round(W, GEOM_QUANT_DP), round(x, GEOM_QUANT_DP))
             for (L, W, x) in TEST_DOMAINS}
    out: List[Tuple[float, float, float]] = []
    for _ in range(100000):
        if len(out) == n:
            break
        L = round(float(rng.uniform(*L_RANGE)), GEOM_QUANT_DP)
        W = round(float(rng.uniform(*W_RANGE)), GEOM_QUANT_DP)
        x0 = round(float(L * rng.uniform(*X0_FRAC_RANGE)), GEOM_QUANT_DP)
        if not X0_FRAC_RANGE[0] <= x0 / L <= X0_FRAC_RANGE[1]:
            continue                                   # rounding pushed it out of the box
        key = (L, W, x0)
        if key in taken:
            continue
        taken.add(key)
        out.append(key)
    if len(out) != n:
        raise RuntimeError("sampler produced {} of {} domains".format(len(out), n))
    return out


# -------------------------------------------------------------------------------- configs
def sample_train_configs(domains: Sequence[Tuple[float, float, float]],
                         seed: int = SEED_DEFAULT) -> List[ApertureConfig]:
    """20 configs per domain: 1 sealed, 1 fully open, 18 apertures drawn on [0.1, 2.5].

    Any draw landing in [0.9, 1.1] is redrawn, so the hold-out band contains ZERO training
    apertures exactly (asserted in the dataset gate, not merely hoped for).
    """
    out: List[ApertureConfig] = []
    for gid, (L, W, x0) in enumerate(domains):
        out.append(_mk(L, W, x0, 0.0, "sealed", "train", gid))
        out.append(_mk(L, W, x0, W, "open", "train", gid))
        seen = set()
        rng = np.random.default_rng([seed, gid])
        for j in range(N_PER_TRAIN_DOMAIN - 2):
            for _ in range(1000):
                a = round(float(rng.uniform(*A_RANGE)), A_QUANT_DP)
                if not in_holdout(a) and a not in seen:
                    break
            else:
                raise RuntimeError("could not draw aperture {} for domain {}".format(j, gid))
            seen.add(a)
            out.append(_mk(L, W, x0, a, "aperture", "train", gid))
    return out


def enumerate_test_configs(
        domains: Sequence[Tuple[float, float, float]] = TEST_DOMAINS) -> List[ApertureConfig]:
    """12 per domain: sealed, fully open, 3 INSIDE the held-out band, 7 outside it."""
    out: List[ApertureConfig] = []
    for gid, (L, W, x0) in enumerate(domains):
        out.append(_mk(L, W, x0, 0.0, "t_sealed", "test", gid))
        out.append(_mk(L, W, x0, W, "t_open", "test", gid))
        for a in A_HOLDOUT_TEST_VALUES:
            out.append(_mk(L, W, x0, a, "t_holdout", "test", gid))
        for a in A_TEST_VALUES:
            out.append(_mk(L, W, x0, a, "t_aperture", "test", gid))
    return out


def manifest_rows(train, test) -> List[dict]:
    rows = []
    for i, c in enumerate(list(train) + list(test)):
        rows.append({"i": i, "split": c.split, "kind": c.kind, "geom_id": c.geom_id,
                     "L": c.L, "W": c.W, "x0": c.x0, "a": c.a,
                     "sealed": bool(c.sealed), "fully_open": bool(c.fully_open),
                     "sqrt_a_hat": c.sqrt_a_hat,
                     "alphas": list(c.alphas), "filename": c.filename,
                     "strata": c.strata})
    return rows


def configs_from_rows(rows, split: Optional[str] = None,
                      kinds: Sequence[str] = ()) -> List[ApertureConfig]:
    out = []
    for r in rows:
        if split is not None and r["split"] != split:
            continue
        if kinds and r["kind"] not in kinds:
            continue
        out.append(ApertureConfig(L=float(r["L"]), W=float(r["W"]), x0=float(r["x0"]),
                                  a=float(r["a"]), kind=r["kind"], split=r["split"],
                                  geom_id=int(r["geom_id"])))
    return out
