"""P4-9 2c evaluation (D180 NMF scoring, N1 / N2): an NMF checkpoint on the 31 dense rooms, 25-100 Hz only.

Per room: off-lattice r and LSD at 25-100 Hz (the frozen metric), the eigenfrequency MAE of the first 10 modes (Hz,
Hungarian-matched to m1's), the subspace overlap of the first 10 modes (mean squared canonical correlation on the
quadrature points). Writes outputs/p4_9/nmf/{NAME}.json and a dense-format cache (bins 50-199 only).
N1: NMF or NMF-P reaches unseen r >= 0.60 AND in-family >= 0.90. N2: NMF-P - NMF >= +0.03 on unseen r, paired 95% CI
(10k, seed 20261013) excluding 0.
  python scripts/p4_9_nmf_eval.py --items NMF_30K=outputs/p4_9/arms/NMF/ckpt_iter0030000.pt
  python scripts/p4_9_nmf_eval.py --readings NMF_30K,NMFP_30K
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

OUT = Path("outputs/p4_9/nmf")


def evaluate(name, ck):
    import torch

    from aaf.models.nmf_2d import NMF2D, hungarian_k_mae, subspace_overlap
    from aaf.train.nmf_2d import LO, HI, NMFCfg, Room, predict, room_field
    from scripts.p4_7_m1_modal import scores
    from scripts.p4_9_fdtd_stencils import dense_rooms
    dev = torch.device("cuda")
    st = torch.load(ck, map_location="cpu")
    cfg = NMFCfg(**st["cfg"])
    model = NMF2D(hidden=cfg.hidden).to(dev)
    model.load_state_dict(st["model"])
    model.eval()
    rows, cache = [], {}
    with torch.no_grad():
        for g, fam, p in dense_rooms():
            lab = dict(np.load(Path(cfg.dense_labels_dir) / (p.stem + ".npz")))
            with h5py.File(p) as f:
                T = np.asarray(f["ism/H_complex"])[:, :601]
                verts = [tuple(map(float, v)) for v in json.loads(f.attrs["verts"])]
            room = Room(lab, verts, max(v[0] for v in verts), max(v[1] for v in verts), dev, h_fd=cfg.h_fd)
            P = np.zeros((len(lab["rx"]), 601), np.complex64)
            for s0 in range(0, len(lab["rx"]), 4096):
                P[s0:s0 + 4096, LO:HI] = room_field(model, room, room.rx[s0:s0 + 4096])[0].cpu().numpy()
            k = (np.asarray(lab["rx"]) - 0.12) / 0.06
            off = ~np.all(np.abs(k - np.round(k)) < 1e-6, axis=1)
            sc = scores(P[off], T[off])
            c, kh, nq, phi_q, _ = predict(model, room, room.q_pos)
            rows.append({"room": p.name, "group": g, "family": fam, "r_25_100": sc["25-100"], "lsd_25_100": sc["lsd_25-100"],
                         "k_mae_hz_10": hungarian_k_mae(kh.cpu().numpy()[:10], np.sqrt(lab["k2"][1:])),
                         "overlap_10": subspace_overlap(phi_q.cpu().numpy(), np.asarray(lab["psi_q"])[:, 1:], np.asarray(lab["w_q"]))})
            cache[p.stem] = P
            print(p.name[:40], {k_: round(v, 3) for k_, v in rows[-1].items() if isinstance(v, float)}, flush=True)
    sets = {"in_family15": lambda x: x["group"] == "test", "unseen16": lambda x: x["group"] == "unseen",
            "U8": lambda x: x["family"] == "U", "C3_8": lambda x: x["family"] == "C3"}
    summ = {s: {m: float(np.mean([x[m] for x in rows if f(x)])) for m in ("r_25_100", "lsd_25_100", "k_mae_hz_10", "overlap_10")}
            for s, f in sets.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "{}.json".format(name)).write_text(json.dumps({"checkpoint": str(ck), "summary": summ, "rooms": rows}, indent=1))
    (OUT / "cache").mkdir(exist_ok=True)
    np.savez_compressed(OUT / "cache" / "{}.npz".format(name), **cache)
    print(name, json.dumps(summ), flush=True)


def readings(names):
    nmf, nmfp = names
    a = json.loads((OUT / "{}.json".format(nmf)).read_text())
    b = json.loads((OUT / "{}.json".format(nmfp)).read_text())
    n1 = {n: {"unseen": d["summary"]["unseen16"]["r_25_100"], "in_family": d["summary"]["in_family15"]["r_25_100"]}
          for n, d in ((nmf, a), (nmfp, b))}
    for v in n1.values():
        v["holds"] = bool(v["unseen"] >= 0.60 and v["in_family"] >= 0.90)
    ra = {x["room"]: x["r_25_100"] for x in a["rooms"] if x["group"] == "unseen"}
    rb = {x["room"]: x["r_25_100"] for x in b["rooms"] if x["group"] == "unseen"}
    d = np.array([rb[k] - ra[k] for k in sorted(ra)])
    m = d[np.random.default_rng(20261013).integers(0, len(d), (10000, len(d)))].mean(1)
    ci = [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
    rep = {"N1": {"by_arm": n1, "holds": bool(any(v["holds"] for v in n1.values()))},
           "N2": {"NMFP_minus_NMF_unseen_25_100": ci, "holds": bool(ci[0] >= 0.03 and ci[1] > 0)}}
    (OUT / "readings.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="")
    ap.add_argument("--readings", default="")
    a = ap.parse_args()
    for it in [x for x in a.items.split("+") if x]:
        n, ck = it.split("=")
        evaluate(n, ck)
    if a.readings:
        readings(a.readings.split(","))


if __name__ == "__main__":
    main()
