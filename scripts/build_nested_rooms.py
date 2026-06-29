"""P2-4: build nested training supersets (45 ⊂ 90 ⊂ 150 ⊂ 250) + the frozen
interior test set, with full property assertions. CPU only.

- Regenerates the existing 45 (LHS seed 42) and asserts it matches train_rooms.yaml.
- Maximin-augments to 90/150/250 (each a superset, DECISIONS D39).
- Samples 15 strictly-interior test rooms inside the 45-hull, distinct from the
  250 set (DECISIONS D40); writes a FROZEN yaml + per-room NN-distance JSON.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from aaf.data.sample_rooms_3d import (  # noqa: E402
    DEFAULT_RANGES, Room3D, sample_train_rooms_lhs, sample_nested_supersets,
    sample_interior_test_rooms, write_rooms_yaml, read_rooms_yaml, _normalize_room,
)

CFG = REPO / "configs/sweeps_3d"
OUT = REPO / "outputs/coverage_curve"


def _key(r, nd=4):
    return (round(r.L, nd), round(r.W, nd), round(r.H, nd))


def _raw_nn(test_rooms, train_rooms):
    """Min euclidean distance (raw metres) of each test room to a training set."""
    T = np.array([r.as_tuple() for r in test_rooms])
    X = np.array([r.as_tuple() for r in train_rooms])
    d = np.sqrt(((T[:, None, :] - X[None, :, :]) ** 2).sum(2))
    return d.min(1)  # (n_test,)


def main():
    # 1. Regenerate the existing 45 + assert it matches train_rooms.yaml.
    base45 = sample_train_rooms_lhs(n=45, seed=42)
    existing = read_rooms_yaml(CFG / "train_rooms.yaml")["rooms"]
    existing_rooms = [Room3D(**r) for r in existing]
    assert len(base45) == 45 and len(existing_rooms) == 45
    a = sorted(_key(r) for r in base45); b = sorted(_key(r) for r in existing_rooms)
    assert a == b, "regenerated-45 does NOT match train_rooms.yaml — seed/sampler drift!"
    print("[ok] regenerated 45 == train_rooms.yaml")

    # 2. Nested supersets.
    supersets = sample_nested_supersets(base45, targets=(90, 150, 250), seed=7)
    sets = {45: base45, 90: supersets[90], 150: supersets[150], 250: supersets[250]}
    for n, rooms in sets.items():
        assert len(rooms) == n, f"set {n} has {len(rooms)} rooms"
    # nesting: each smaller is a subset of the next
    for small, big in [(45, 90), (90, 150), (150, 250)]:
        ss, bs = {_key(r) for r in sets[small]}, {_key(r) for r in sets[big]}
        assert ss <= bs, f"{small} is NOT a subset of {big}"
    assert {_key(r) for r in base45} <= {_key(r) for r in sets[250]}
    print("[ok] nesting 45 ⊂ 90 ⊂ 150 ⊂ 250 verified")
    # report maximin spacing (mean NN within each set, normalized)
    for n, rooms in sets.items():
        Xn = np.stack([_normalize_room(r, DEFAULT_RANGES) for r in rooms])
        d = np.sqrt(((Xn[:, None] - Xn[None]) ** 2).sum(2)); np.fill_diagonal(d, np.inf)
        print(f"    set {n}: mean nearest-neighbour (normalized) = {d.min(1).mean():.3f}")

    # 3. Frozen interior test set (inside 45-hull, distinct from the 250 set).
    test = sample_interior_test_rooms(hull_rooms=base45, exclude_rooms=sets[250],
                                      n=15, seed=2024, min_train_dist=0.04)
    assert len(test) == 15
    # verify strictly inside the 45-hull
    from scipy.spatial import Delaunay
    hull = Delaunay(np.stack([_normalize_room(r, DEFAULT_RANGES) for r in base45]))
    tn = np.stack([_normalize_room(r, DEFAULT_RANGES) for r in test])
    assert (hull.find_simplex(tn) >= 0).all(), "some test room outside the 45-hull"
    # distinct from all training (250) sets
    train_keys = {_key(r) for r in sets[250]}
    assert not (train_keys & {_key(r) for r in test}), "test room coincides with training"
    print("[ok] 15 test rooms strictly interior to the 45-hull + distinct from training")

    # 4. NN-distance per test room to each training set (raw metres) — the x-axis.
    nn = {str(n): _raw_nn(test, sets[n]).tolist() for n in (45, 90, 150, 250)}
    means = {n: float(np.mean(nn[str(n)])) for n in (45, 90, 150, 250)}
    seq = [means[n] for n in (45, 90, 150, 250)]
    assert all(seq[i] > seq[i + 1] for i in range(3)), f"mean NN-dist not decreasing: {seq}"
    print(f"[ok] mean test NN-distance (m) decreases 45→250: "
          + " → ".join(f"{means[n]:.3f}" for n in (45, 90, 150, 250)))

    # 5. Write YAMLs + NN JSON.
    meta_common = dict(lhs_seed=42, ranges_L=[3.0, 6.0], ranges_W=[3.0, 5.0],
                       ranges_H=[2.5, 4.0], reject_cubic_tol=0.05,
                       construction="maximin-augmented nested (P2-4 D39); seeded by the existing 45")
    for n in (45, 90, 150, 250):
        write_rooms_yaml(CFG / f"train_rooms_{n}.yaml", sets[n],
                         set_name=f"train_{n}", extra_meta=meta_common)
    write_rooms_yaml(CFG / "test_rooms_interior_frozen.yaml", test,
                     set_name="test_interior_frozen",
                     extra_meta=dict(frozen_note="FROZEN — reused across P2-4 and P3-1; do NOT modify",
                                     construction="maximin interior of the 45-hull, distinct from the 250 set (P2-4 D40)",
                                     seed=2024, min_train_dist_norm=0.04,
                                     nn_distance_m_by_trainset=means))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_nn_distances.json").write_text(json.dumps(
        {"per_room_nn_m": nn, "mean_nn_m": means,
         "test_rooms": [r.as_dict() for r in test]}, indent=2))
    print(f"[ok] wrote train_rooms_{{45,90,150,250}}.yaml + test_rooms_interior_frozen.yaml "
          f"+ {OUT}/test_nn_distances.json")


if __name__ == "__main__":
    main()
