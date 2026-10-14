"""P4-5: does D84's token-count control survive off the lattice?

D84 (the headline of P4-4 Task 3): re-describing an IDENTICAL room with redundant collinear
tokens costs -0.13 to -0.24 spatial Pearson -- more than two reflex corners. It was measured on
the 800 lattice receivers. D99 found the attention arms memorise that lattice, and the
re-tokenised conditioning vector could disturb the memorised channel without disturbing the field
-- so part of the token cost may be lost memorisation rather than lost physics.

Two of the five dense family rooms (`build_p4_5_rooms.family_test_rooms`, chosen by d_hat before
any score) are also token-control rooms: rectL6.56_W4.12 (4/8/12 tokens) and
LL5.56_W4.52_NWd1.10w2.22x0.00 (6/12). Each refinement is scored on the ORIGINAL 800 receivers
(must reproduce FAMILY_EVAL.json's token_control_rooms exactly) and on the dense lattice, on/off.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from aaf.eval.modal_projection import enumerate_modes
from aaf.eval.p3_2_eval import load_model
from scripts.build_p4_5_rooms import OUT, family_test_rooms
from scripts.p4_4_family_eval import MAX_SEG_POLY, N_MODES, REFINE, _metrics, _render, refine_verts
from scripts.p4_5_family_dense import CKPT, PUB, TOL, load, onlat


def main() -> int:
    dev = torch.device("cuda")
    model, renderer, _cfg, _m, _it = load_model(CKPT, dev)
    model.eval(); renderer.eval()
    pub = {(r["label"], r["refine_k"]): r["spatial_pearson"]
           for r in json.load(open(PUB))["token_control_rooms"]}
    rows, bad = [], []
    for fam, c in family_test_rooms():
        ks = [k for k in REFINE.get(c.kind, ()) if (c.label, k) in pub]
        if not ks:
            continue
        modes = enumerate_modes(c.L, c.W, f_max=200.0)[:N_MODES]
        T8, rx8, los8 = load(Path("data/track_p4_4_family") / c.filename)
        Td, rxd, losd = load(Path(OUT["famdense"]) / c.filename)
        on = onlat(rxd)
        for k in ks:
            v = refine_verts(c.verts, k)
            assert len(v) <= MAX_SEG_POLY
            m8 = _metrics(_render(model, renderer, c, rx8, T8.shape[1], dev, verts=v), T8, los8, modes)["spatial_pearson"]
            if abs(m8 - pub[(c.label, k)]) > TOL:
                bad.append((c.label, k, pub[(c.label, k)], m8))
            Pd = _render(model, renderer, c, rxd, Td.shape[1], dev, verts=v)
            r = {"family": fam, "room": c.label, "refine_k": k, "n_tokens": len(v),
                 "published_800rx": pub[(c.label, k)], "reproduced_800rx": m8,
                 "dense_on": _metrics(Pd[on], Td[on], losd[on], modes)["spatial_pearson"],
                 "dense_off": _metrics(Pd[~on], Td[~on], losd[~on], modes)["spatial_pearson"]}
            rows.append(r)
            print("{:5s} {:34s} {:2d} tok  published {:+.4f} (reproduced {:+.4f}) | dense on {:+.4f} OFF {:+.4f}".format(
                fam, c.label, r["n_tokens"], r["published_800rx"], m8, r["dense_on"], r["dense_off"]), flush=True)
    cost = {}
    for room in sorted({r["room"] for r in rows}):
        g = sorted((r for r in rows if r["room"] == room), key=lambda r: r["n_tokens"])
        base = g[0]
        for r in g[1:]:
            cost["{} {}->{} tok".format(room, base["n_tokens"], r["n_tokens"])] = {
                "published_cost": r["published_800rx"] - base["published_800rx"],
                "dense_on_cost": r["dense_on"] - base["dense_on"],
                "dense_off_cost": r["dense_off"] - base["dense_off"]}
    err = max(abs(r["reproduced_800rx"] - r["published_800rx"]) for r in rows)
    print("\nreproduction of the published token-control values (|dr| <= {:g}): {}, max |dr| {:.1e}".format(
        TOL, "YES" if not bad else "NO {}".format(bad), err))
    for k, v in cost.items():
        print("{:52s} token cost: published {:+.3f} | dense on {:+.3f} | OFF {:+.3f}".format(
            k, v["published_cost"], v["dense_on_cost"], v["dense_off_cost"]))
    json.dump({"checkpoint": str(CKPT), "rows": rows, "token_cost": cost,
               "exact_reproduction_of_published": not bad, "mismatches": bad,
               "max_abs_reproduction_error": err, "reproduction_tolerance": TOL},
              open("outputs/p4_5/gate2_dense/FAMILY_TOKENS_DENSE.json", "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
