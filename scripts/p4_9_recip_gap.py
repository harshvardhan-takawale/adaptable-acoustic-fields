"""P4-9 X3 diagnostic (D193): the network's reciprocity gap on the 31 dense rooms.

Per room, off-lattice receivers that are interior FDTD nodes: the normal render (receiver r_i, source s; the standard
report's cache) and the SWAPPED render (receiver s, source r_i), both scored against the same truth with the frozen
scorer (`p4_7_m1_modal.scores`, band r / LSD), plus their mutual band r. Reciprocal model => equal scores.
  PYTHONPATH=$PWD python scripts/p4_9_recip_gap.py --items R0_120K=outputs/p4_8/arms/R0/ckpt_iter0120000.pt
  -> outputs/p4_9/recip_gap/<NAME>.json ; --report -> outputs/p4_9/recip_gap.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path("outputs/p4_9/recip_gap")
B = ("25-100", "100-200", "200-300")
G2 = "outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz"
UN = "outputs/p4_8/cache/unseen_{}.npz"


def evaluate(name, ck):
    import torch

    from aaf.data.augment_2d import is_interior_node
    from aaf.eval.p3_2_eval import load_model
    from aaf.models.conditioning_2d import build_cond_vector_2d
    from scripts.p4_7_m1_modal import scores
    from scripts.p4_9_grounding import eval_rooms, load
    dev = torch.device("cuda")
    model, renderer, *_ = load_model(Path(ck), dev)
    model.eval()
    renderer.eval()
    rows = []
    for rr in eval_rooms():
        r = load(rr["path"])
        cache = np.load((G2 if rr["group"] == "test" else UN).format(name))
        assert np.allclose(cache["rx%d" % rr["cache_i"]], r["rx"])
        P = cache["P%d" % rr["cache_i"]][:, :601].astype(np.complex128)
        src = np.array([0.5, 1.6])
        keep = np.nonzero(r["off"] & is_interior_node(r["verts"], r["rx"], 0.02, 0.02))[0]
        cond = build_cond_vector_2d("geom_token", r["L"], r["W"], [0.15] * len(r["verts"]), verts=r["verts"], device=dev)
        lo = torch.zeros(2, device=dev)
        hi = torch.tensor([r["L"], r["W"]], device=dev, dtype=torch.float32)
        Ps = np.zeros((len(keep), 601), np.complex128)
        S = torch.tensor(r["rx"][keep], dtype=torch.float32, device=dev)
        R = torch.tensor(src, dtype=torch.float32, device=dev).view(1, 2)
        with torch.no_grad():
            for s0 in range(0, len(keep), 8):
                t = S[s0:s0 + 8]
                H = renderer(model, R.expand(len(t), -1), t, lo, hi, z_s=cond.unsqueeze(0).expand(len(t), -1))
                Ps[s0:s0 + 8] = H[:, :601].cpu().numpy()
        T = r["T"][keep]
        row = {"room": r["name"], "group": rr["group"], "n": int(len(keep)), "normal": scores(P[keep], T),
               "swapped": scores(Ps, T), "swapped_vs_normal": scores(Ps, P[keep])}
        rows.append(row)
        print(name, r["name"][:40], " ".join("{} {:.3f}/{:.3f}".format(b, row["normal"][b], row["swapped"][b])
                                             for b in B), flush=True)
    summ = {}
    for g, lab in (("test", "in_family15"), ("unseen", "unseen16")):
        sel = [x for x in rows if x["group"] == g]
        summ[lab] = {k: {b: float(np.mean([x[k][b] for x in sel])) for b in B}
                     for k in ("normal", "swapped", "swapped_vs_normal")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "{}.json".format(name)).write_text(json.dumps({"checkpoint": ck, "summary": summ, "rooms": rows}, indent=1))
    print(name, json.dumps(summ), flush=True)


def report():
    L = ["# P4-9 X3 diagnostic (D193): reciprocity gap", "",
         "Off-lattice interior receivers; r per band vs the SAME truth: normal render (receiver r, source s) and swapped "
         "render (receiver s, source r). A reciprocal model scores the same both ways; last column = r(swapped, normal).",
         "", "| model | set | normal | swapped | swapped vs normal |", "|---|---|---|---|---|"]
    for p in sorted(OUT.glob("*.json")):
        d = json.loads(p.read_text())
        for s, e in d["summary"].items():
            L.append("| {} | {} | {} | {} | {} |".format(p.stem, s, *[" / ".join("{:.3f}".format(e[k][b]) for b in B)
                                                                     for k in ("normal", "swapped", "swapped_vs_normal")]))
    Path("outputs/p4_9/recip_gap.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    for item in [s for s in a.items.replace("+", ",").split(",") if s]:
        n, ck = item.split("=", 1)
        evaluate(n, ck)
    if a.report:
        report()


if __name__ == "__main__":
    main()
