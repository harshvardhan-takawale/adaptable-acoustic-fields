"""P4-6 (D160(f)): the D125 / D135 training-room fit for FAMILY-corpus arms (g2), off the lattice.

`p4_5_train_offlat_fit` measures fit on 23 shape-corpus training rooms at B6-corpus receivers. A
family-corpus model never trained on those rooms, so its training-room fit needs family training
rooms with off-lattice receivers: `data/track_p4_6_famtrain_offlat` (20 rooms, 5 each of rect / L /
T / DN, every 12th by filename; `build_p4_6_famtrain_offlat.py`). Receivers on the 0.12 + 0.06 k
lattice are dropped. Metrics and keys are `p4_5_train_offlat_fit`'s, so the two read side by side
(LSD 0-100 is D135's kill signal; per-band r / LSD feed the training-to-test gap).
  python scripts/p4_6_train_fit_family.py --items "G2_4K=outputs/p4_6/arms/G2/ckpt_iter0004000.pt"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.build_p4_6_famtrain_offlat import OUT as DATA, rooms
from scripts.p4_4_family_eval import _render
from scripts.p4_5_gate2_dense import on_lattice

OUT = Path("outputs/p4_6/train_fit")


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def fit(name, ck):
    import torch

    from aaf.eval.p3_2_eval import load_model
    dev = torch.device("cuda")
    model, renderer, cfg, _m, it = load_model(Path(ck), dev)
    model.eval(); renderer.eval()
    rows = []
    for c in rooms():
        with h5py.File(DATA / c.filename) as f:
            T = np.asarray(f["ism/H_complex"])[:, :601]
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
        off = ~on_lattice(rx)
        T, rx = T[off], rx[off]
        P = _render(model, renderer, c, rx, 601, dev)[:, :601]
        dP, dT = db(P), db(T)
        a = dP[:, 50:601] - dP[:, 50:601].mean(0); b = dT[:, 50:601] - dT[:, 50:601].mean(0)
        r_bins = (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))
        rows.append({"room": c.label, "family": c.kind, "n_off": int(len(rx)),
                     "lsd_0_100": float(np.mean(np.abs(dP[:, :200] - dT[:, :200]))),
                     "lsd_0_200": float(np.mean(np.abs(dP[:, :401] - dT[:, :401]))),
                     "lsd_0_300": float(np.mean(np.abs(dP - dT))),
                     "r_25_100": float(np.nanmean(r_bins[:150])), "r_25_300": float(np.nanmean(r_bins)),
                     "r_100_200": float(np.nanmean(r_bins[150:350])), "r_200_300": float(np.nanmean(r_bins[350:551])),
                     "lsd_100_200": float(np.mean(np.abs(dP[:, 200:400] - dT[:, 200:400]))),
                     "lsd_200_300": float(np.mean(np.abs(dP[:, 400:601] - dT[:, 400:601])))})
    keys = [k for k in rows[0] if k not in ("room", "family", "n_off")]
    rep = {"name": name, "checkpoint": str(ck), "iter": int(it), "corpus": "family (D160(f))",
           "n_rooms": len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in keys},
           "per_family": {f: {k: float(np.mean([r[k] for r in rows if r["family"] == f])) for k in keys}
                          for f in sorted({r["family"] for r in rows})},
           "rooms": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "{}.json".format(name)).write_text(json.dumps(rep, indent=1))
    print("{:10s} iter {:6d} family TRAIN fit off-lattice: LSD 0-100 {:.3f}; r 100-200 {:.3f}, 200-300 {:.3f}; "
          "LSD 200-300 {:.3f}".format(name, it, rep["lsd_0_100"], rep["r_100_200"], rep["r_200_300"],
                                      rep["lsd_200_300"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    a = ap.parse_args()
    for it in a.items.replace("+", ",").split(","):
        n, ck = it.split("=", 1)
        fit(n, ck)


if __name__ == "__main__":
    main()
