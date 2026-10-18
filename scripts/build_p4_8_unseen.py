"""P4-8 Step 0b (D174(b)): the frozen unseen-family test set -- 8 U rooms + 8 three-corner (C3: NW + NE + SE) rooms,
simulated dense exactly as D111 (`build_p4_5_rooms.build_famdense`: 64x64 lattice clipped to the polygon, wall /
reflex-corner clearance, lattice_keep, dx 0.02, fixed source (0.5, 1.6)).
  U : the FAM test-split U rooms sorted by (d_hat, filename): indices 4, 5 (D111 / D112; reused from
      data/track_p4_5_famdense by symlink) + 3, 6, 2, 7, 1, 8 (D111's rule extended: next nearest to the median,
      alternating).
  C3: the first 8 valid draws of default_rng([20261018, k]) under the family ranges and constraints of D174(b).
  python scripts/build_p4_8_unseen.py --manifest     # freeze configs/sweeps_2d_mat/p4_8_unseen_manifest.json
  python scripts/build_p4_8_unseen.py --idx 0 --chunk 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from aaf.data.multi_notch import (D_FRAC_MAX, EPS, L_RANGE, MIN_PASSAGE_M, MIN_STEM_M, W_FRAC_MAX, W_RANGE, Notch,
                                  _connected, _q, _self_intersects, verts_for_notches)
from aaf.data.shape_configs import SRC

FAM_MANIFEST = "configs/sweeps_2d_mat/p4_4_family_manifest.json"
MANIFEST = Path("configs/sweeps_2d_mat/p4_8_unseen_manifest.json")
OUT = Path("data/track_p4_8_unseen")
FAMDENSE = Path("data/track_p4_5_famdense")
C3_SEED = 20261018
U_PICKS = (4, 5, 3, 6, 2, 7, 1, 8)
SLAB = (0.45, 0.60)


@dataclass(frozen=True)
class C3Config:
    """Three corner notches NW + NE + SE. Exposes what build_famdense / _simulate / the evaluators touch."""
    L: float
    W: float
    notches: Tuple[Notch, ...]
    shape_id: int = 0
    split: str = "unseen"

    @property
    def verts(self):
        return verts_for_notches(self.L, self.W, self.notches)

    @property
    def kind(self):
        return "C3"

    @property
    def d_hat(self):
        return self.notches[0].d / (D_FRAC_MAX * self.W)               # the NW notch, the family convention

    @property
    def label(self):
        return "C3L{:.2f}_W{:.2f}_".format(self.L, self.W) + "_".join(
            "{}d{:.2f}w{:.2f}".format(n.side, n.d, n.w) for n in self.notches)

    @property
    def filename(self):
        return self.label + ".h5"

    @property
    def edge_alphas(self):
        return tuple([0.15] * len(self.verts))


def c3_ok(L, W, ns) -> Tuple[bool, str]:
    by = {n.side: n for n in ns}
    for n in ns:
        if n.d <= 0 or n.w <= 0 or n.d > D_FRAC_MAX * W + EPS or n.w > W_FRAC_MAX * L + EPS:
            return False, "notch size"
        if SLAB[0] <= n.d / (D_FRAC_MAX * W) <= SLAB[1]:
            return False, "d_hat in slab"
    if L - by["NW"].w - by["NE"].w < MIN_STEM_M:
        return False, "U stem"
    if W - by["NE"].d - by["SE"].d < MIN_PASSAGE_M:
        return False, "east-wall passage"
    if by["NW"].w + by["SE"].w > L - MIN_PASSAGE_M:
        return False, "diagonal pinch"
    v = verts_for_notches(L, W, ns)
    xs, ys = [p[0] for p in v], [p[1] for p in v]
    if abs(max(xs) - L) > EPS or abs(max(ys) - W) > EPS or abs(min(xs)) > EPS or abs(min(ys)) > EPS:
        return False, "bbox"
    if len(v) != 10:
        return False, "expected 10 vertices, got {}".format(len(v))
    if not _connected(L, W, ns) or _self_intersects(v):
        return False, "ring"
    from aaf.sim.polygon_geom import polygon_contains
    if not bool(polygon_contains(v, np.asarray([SRC], dtype=float))[0]):
        return False, "source in solid"
    return True, "ok"


def draw_c3(n=8, seed=C3_SEED):
    out, k = [], 0
    while len(out) < n:
        rng = np.random.default_rng([seed, k])
        k += 1
        L, W = _q(rng.uniform(*L_RANGE)), _q(rng.uniform(*W_RANGE))
        ns = tuple(Notch(s, _q(rng.uniform(0, 1) * D_FRAC_MAX * W), _q(rng.uniform(0.15, 1) * W_FRAC_MAX * L))
                   for s in ("NW", "NE", "SE"))
        if c3_ok(L, W, ns)[0]:
            out.append(C3Config(L, W, ns, shape_id=12000 + len(out)))
    return out, k


def u_rooms():
    from aaf.data.multi_notch import configs_from_rows
    us = sorted([c for c in configs_from_rows(json.load(open(FAM_MANIFEST))["configs"], split="test") if c.kind == "U"],
                key=lambda c: (c.d_hat, c.filename))
    return [(i, us[i]) for i in U_PICKS]


def row(c, fam, extra):
    return dict({"family": fam, "kind": c.kind, "filename": c.filename, "label": c.label, "L": c.L, "W": c.W,
                 "verts": [list(map(float, p)) for p in c.verts], "edge_alphas": [0.15] * len(c.verts),
                 "notches": [{"side": n.side, "d": n.d, "w": n.w, "x0": n.x0} for n in c.notches],
                 "d_hat": round(float(c.d_hat), 6), "source": list(SRC), "split": "unseen"}, **extra)


def write_manifest():
    rows = [row(c, "U", {"u_sorted_index": i, "origin": "D111" if i in (4, 5) else "D111-rule extension"})
            for i, c in u_rooms()]
    c3, tried = draw_c3()
    rows += [row(c, "C3", {"draw": j}) for j, c in enumerate(c3)]
    assert len({r["filename"] for r in rows}) == 16
    man = {"schema": "p4_1.poly/1", "set": "P4-8 unseen-family test set (D174(b))", "frozen": "2026-10-18",
           "u_rule": "FAM test-split U sorted by (d_hat, filename); indices {}".format(list(U_PICKS)),
           "c3_rule": "NW+NE+SE, first 8 valid draws of default_rng([{}, k]) ({} tried)".format(C3_SEED, tried),
           "rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(), "configs": rows}
    MANIFEST.write_text(json.dumps(man, indent=1))
    print("wrote", MANIFEST, man["rows_sha256"][:16], [(r["family"], r["label"]) for r in rows])


def work():
    rows = json.loads(MANIFEST.read_text())["configs"]
    us = dict(u_rooms())
    c3 = draw_c3()[0]
    out = []
    for r in rows:
        if r["family"] == "U":
            out.append((r, us[r["u_sorted_index"]]))
        else:
            out.append((r, c3[r["draw"]]))
    for r, c in out:
        assert c.filename == r["filename"], (c.filename, r["filename"])
    return out


def main():
    from scripts.build_p4_5_rooms import build_famdense
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=2)
    a = ap.parse_args()
    if a.manifest:
        write_manifest()
        return
    w = work()
    if a.idx is None:
        print("{} rooms -> --array=0-{} at chunk {}".format(len(w), (len(w) - 1) // a.chunk, a.chunk))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for r, c in w[a.idx * a.chunk:(a.idx + 1) * a.chunk]:
        if r.get("origin") == "D111":                                  # reuse D111's dense simulation
            src = FAMDENSE / c.filename
            link = OUT / c.filename
            if not link.exists():
                os.symlink(os.path.relpath(src, OUT), link)
            print("linked", c.filename, flush=True)
            continue
        print(build_famdense(r["family"], c, OUT), c.filename, flush=True)


if __name__ == "__main__":
    main()
