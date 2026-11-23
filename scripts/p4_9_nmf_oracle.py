"""P4-9 2c NMF oracle decomposition (D187; descriptive): which half of the NMF -- eigenfrequencies or mode shapes --
costs its 25-100 Hz accuracy?

Per dense room (31: 15 in-family, 16 unseen), bins 50-199, off-lattice receivers, the frozen scorer, four fields from
m1's formula (`NMF2D.field`, same zero mode / pulse / constant as `room_field`):
  pred              NMF k, eta, shapes (receivers and source)  -- must reproduce outputs/p4_9/nmf/<NAME>.json
  true_shapes       m1's first 24 non-zero eigenpairs (shapes, eta, source values) with the NMF's k (sort order)
  true_k            the NMF's shapes and eta with m1's first 24 non-zero k
  all_true          m1 truncated to 24 modes
  PYTHONPATH=$PWD python scripts/p4_9_nmf_oracle.py --items NMF_30K=outputs/p4_9/arms/NMF/ckpt_iter0030000.pt
  -> outputs/p4_9/nmf/oracle_<NAME>.json ; --report -> outputs/p4_9/nmf/oracle.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

OUT = Path("outputs/p4_9/nmf")
VARIANTS = ("pred", "true_shapes", "true_k", "all_true")


def evaluate(name, ck):
    import torch

    from aaf.models.nmf_2d import NMF2D
    from aaf.train.nmf_2d import LO, HI, NMFCfg, Room, predict
    from aaf.eval.modal_prior import C_FIT
    from scripts.p4_7_m1_modal import scores
    from scripts.p4_9_fdtd_stencils import dense_rooms
    dev = torch.device("cuda")
    st = torch.load(ck, map_location="cpu")
    cfg = NMFCfg(**st["cfg"])
    model = NMF2D(hidden=cfg.hidden).to(dev)
    model.load_state_dict(st["model"])
    model.eval()
    ref = {r["room"]: r["r_25_100"] for r in json.loads((OUT / "{}.json".format(name)).read_text())["rooms"]}
    rows = []
    bins = torch.arange(LO, HI, device=dev)
    with torch.no_grad():
        for g, fam, p in dense_rooms():
            lab = dict(np.load(Path(cfg.dense_labels_dir) / (p.stem + ".npz")))
            with h5py.File(p) as f:
                T = np.asarray(f["ism/H_complex"])[:, :601]
                verts = [tuple(map(float, v)) for v in json.loads(f.attrs["verts"])]
            room = Room(lab, verts, max(v[0] for v in verts), max(v[1] for v in verts), dev, h_fd=cfg.h_fd)
            c, k_hat, nq, phi_x, eta_hat = predict(model, room, room.rx)
            phi_s = model.phi_raw(room.src, c)[0] / nq
            t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa
            k_true = t(np.sqrt(lab["k2"][1:25]))
            psi_x, psi_s, eta_true = t(lab["psi_r"][:, 1:25]), t(lab["psi_s"][1:25]), t(lab["eta"][1:25])
            zero = (room.psi0 * room.td0[LO:HI] * (C_FIT / room.c_pred))[None].expand(len(room.rx), -1)
            parts = {"pred": (k_hat, eta_hat, phi_x, phi_s), "true_shapes": (k_hat, eta_true, psi_x, psi_s),
                     "true_k": (k_true, eta_hat, phi_x, phi_s), "all_true": (k_true, eta_true, psi_x, psi_s)}
            k = (np.asarray(lab["rx"]) - 0.12) / 0.06
            off = ~np.all(np.abs(k - np.round(k)) < 1e-6, axis=1)
            row = {"room": p.name, "group": g, "family": fam}
            for v, (kk, ee, px, ps) in parts.items():
                P = np.zeros((len(lab["rx"]), 601), np.complex64)
                P[:, LO:HI] = NMF2D.field(kk, ee, px, ps, bins, room.S[LO:HI], C_FIT, zero).cpu().numpy()
                row[v] = float(scores(P[off], T[off])["25-100"])
            assert abs(row["pred"] - ref[p.name]) < 2e-3, (p.name, row["pred"], ref[p.name])
            rows.append(row)
            print(p.name[:40], {v: round(row[v], 3) for v in VARIANTS}, flush=True)
    summ = {}
    for s, f in (("in_family15", lambda x: x["group"] == "test"), ("unseen16", lambda x: x["group"] == "unseen")):
        summ[s] = {v: float(np.mean([x[v] for x in rows if f(x)])) for v in VARIANTS}
        gap = summ[s]["all_true"] - summ[s]["pred"]
        summ[s]["recovered_by_true_shapes"] = (summ[s]["true_shapes"] - summ[s]["pred"]) / gap
        summ[s]["recovered_by_true_k"] = (summ[s]["true_k"] - summ[s]["pred"]) / gap
    (OUT / "oracle_{}.json".format(name)).write_text(json.dumps({"checkpoint": ck, "summary": summ, "rooms": rows},
                                                                indent=1))
    print(name, json.dumps(summ, indent=1), flush=True)


def report():
    L = ["# P4-9 2c: NMF oracle decomposition (D187; descriptive)", "",
         "Off-lattice r at 25-100 Hz (frozen scorer), mean over rooms. Each column substitutes m1's exact eigenpairs for "
         "one half of the NMF (24 modes, paired by sort order). 'recovered' = share of the gap pred -> all-true closed "
         "by that substitution. D187(c): the larger single share names the bottleneck; if both are below 0.5, the errors "
         "interact and neither is named.", "",
         "| arm | rooms | pred | true shapes + pred k | pred shapes + true k | all true (24 modes) | recovered: shapes / k |",
         "|---|---|---|---|---|---|---|"]
    for p in sorted(OUT.glob("oracle_*.json")):
        d = json.loads(p.read_text())
        for s in ("in_family15", "unseen16"):
            e = d["summary"][s]
            L.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} / {:.2f} |".format(
                p.stem.replace("oracle_", ""), s, e["pred"], e["true_shapes"], e["true_k"], e["all_true"],
                e["recovered_by_true_shapes"], e["recovered_by_true_k"]))
    (OUT / "oracle.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    for item in [s for s in a.items.replace("+", ",").split(",") if s]:
        name, ck = item.split("=", 1)
        evaluate(name, ck)
    if a.report:
        report()


if __name__ == "__main__":
    main()
