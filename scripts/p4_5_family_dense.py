"""P4-5, manager item 3: BOUND the family eval (D83/D84, `p4_4_FAM`) instead of leaving it flagged.

The family corpus puts its receivers on the same absolute lattice as every P4 corpus
(`build_p4_4_family_rooms.py:46,70-71`) and FAM is an attention-pooling arm, so D99 predicts its
published numbers are inflated. One held-out room per family -- the median-d_hat test room
(rectangles: median area), chosen by rule before any score was seen -- re-simulated on a dense
64x64 lattice, re-scored with the family evaluator's OWN metric (`p4_4_family_eval._metrics`:
6 bounding-box modes, dB spatial Pearson), split on / off the lattice.

EXACTNESS FIRST. The same rooms are also scored on their ORIGINAL 800 receivers, and must
reproduce the published per-room values in FAMILY_EVAL.json; only then are the dense numbers
reported. A bound is what is asked for: does the family result move by 0.02 or by 0.22?
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from aaf.eval.modal_projection import enumerate_modes
from aaf.eval.p3_2_eval import load_model
from scripts.build_p4_5_rooms import OUT, family_test_rooms
from scripts.p4_4_family_eval import N_MODES, _metrics, _render

CKPT = Path("outputs/p4_4/p4_4_FAM/ckpt_iter0060000.pt")
PUB = "outputs/p4_4/p4_4_FAM_eval/FAMILY_EVAL.json"
# A re-render on another GPU is not bitwise: tcnn accumulation order differs. Measured |dr| <= 7e-10.
TOL = 1e-6


def onlat(rx):
    k = (np.asarray(rx) - 0.12) / 0.06
    return np.all(np.abs(k - np.round(k)) < 1e-6, axis=1)


def load(path):
    with h5py.File(path) as f:
        return (np.asarray(f["ism/H_complex"])[:, :601], np.asarray(json.loads(f.attrs["receiver_pos"])),
                np.asarray(json.loads(f.attrs["line_of_sight"]), dtype=bool))


def main() -> int:
    dev = torch.device("cuda")
    model, renderer, cfg, _m, it = load_model(CKPT, dev)
    model.eval(); renderer.eval()
    pub = {r["label"]: r for r in json.load(open(PUB))["per_room"]}
    rep = {"checkpoint": str(CKPT), "metric": "p4_4_family_eval._metrics, N_MODES = {}".format(N_MODES),
           "selection": "median-d_hat held-out room per family (rect: median area), by rule", "rooms": []}
    bad = []
    for fam, c in family_test_rooms():
        modes = enumerate_modes(c.L, c.W, f_max=200.0)[:N_MODES]
        # 1. exactness: the ORIGINAL 800 receivers must reproduce the published number
        T8, rx8, los8 = load(Path("data/track_p4_4_family") / c.filename)
        P8 = _render(model, renderer, c, rx8, T8.shape[1], dev)
        m8 = _metrics(P8, T8, los8, modes)["spatial_pearson"]
        p_pub = pub[c.label]["spatial_pearson"]
        if abs(m8 - p_pub) > TOL:
            bad.append((c.label, p_pub, m8))
        # 2. dense, split by lattice membership
        Td, rxd, losd = load(Path(OUT["famdense"]) / c.filename)
        Pd = _render(model, renderer, c, rxd, Td.shape[1], dev)
        on = onlat(rxd)
        row = {"family": fam, "room": c.label, "d_hat": float(c.d_hat), "n_dense": int(len(rxd)),
               "n_on_lattice": int(on.sum()), "published_800rx": p_pub, "reproduced_800rx": m8,
               "dense_all": _metrics(Pd, Td, losd, modes)["spatial_pearson"],
               "dense_on": _metrics(Pd[on], Td[on], losd[on], modes)["spatial_pearson"],
               "dense_off": _metrics(Pd[~on], Td[~on], losd[~on], modes)["spatial_pearson"]}
        row["shift_off_minus_published"] = row["dense_off"] - p_pub
        rep["rooms"].append(row)
        print("{:5s} {:52s} published {:+.4f} (reproduced {:+.4f}) | dense on {:+.4f} all {:+.4f} OFF {:+.4f} | shift {:+.3f}".format(
            fam, c.label, p_pub, m8, row["dense_on"], row["dense_all"], row["dense_off"],
            row["shift_off_minus_published"]), flush=True)
    rep["exact_reproduction_of_published"] = not bad
    rep["mismatches"] = bad
    rep["max_abs_reproduction_error"] = max(abs(r["reproduced_800rx"] - r["published_800rx"]) for r in rep["rooms"])
    rep["reproduction_tolerance"] = TOL
    sh = [r["shift_off_minus_published"] for r in rep["rooms"]]
    rep["bound"] = {"mean_shift": float(np.mean(sh)), "min_shift": float(np.min(sh)), "max_shift": float(np.max(sh))}
    print("\nreproduction of the published 800-receiver values (|dr| <= {:g}): {}, max |dr| {:.1e}".format(
        TOL, "YES, all 5 rooms" if not bad else "NO {}".format(bad), rep["max_abs_reproduction_error"]))
    print("BOUND on the family-eval shift (dense off-lattice minus published): mean {:+.3f}, range [{:+.3f}, {:+.3f}]".format(
        *rep["bound"].values()))
    out = Path("outputs/p4_5/gate2_dense")
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(out / "FAMILY_DENSE_BOUND.json", "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
