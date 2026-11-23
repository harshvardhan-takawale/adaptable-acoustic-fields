"""P4-10 A14 (D197): forward-only probe of R0 @120K on the MAT corpus's tokens, before any material training.
For each sampled room (slot 0, 160 of its 800 receivers): render R0 with alpha = 0.15 tokens (its training condition) and
with the room's per-edge tokens, and compare both to the material truth, per region: bins 0-40 (the DC head's) and
25-100 / 100-200 / 200-300 Hz. Reports r, LSD, the mean |H| at bin 0 (truth / prediction), and the trainer's weighted L1
terms split at bin 40, so a DC blow-up is visible as a loss ratio. Descriptive; the decision rule is in D197 A14.
  PYTHONPATH=$PWD python scripts/p4_10_a14_probe.py  -> outputs/p4_10/a14_probe.json
"""
import json
from pathlib import Path

import numpy as np
import torch

from aaf.eval.p3_2_eval import load_model
from aaf.models.conditioning_2d import build_cond_vector_2d
from scripts.p4_10_build_ms import rows
from scripts.p4_10_build_variants import MM

CK = "outputs/p4_8/arms/R0/ckpt_iter0120000.pt"
B = {"0-40": (0, 41), "25-100": (50, 200), "100-200": (200, 400), "200-300": (400, 601)}


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-30))


def main():
    dev = torch.device("cuda")
    model, renderer, *_ = load_model(Path(CK), dev)
    model.eval()
    renderer.eval()
    d = MM["mat"]
    idx = json.loads((d / "index.json").read_text())
    mm = np.load(d / "H.npy", mmap_mode="r")
    rx_all = np.load(d / "rx.npy")
    src_all = np.load(d / "src.npy")
    rep = []
    for v in np.linspace(0, len(idx["variants"]) - 1, 24).round().astype(int):
        c = idx["variants"][v]
        verts = [tuple(x) for x in c["verts"]]
        sel = np.linspace(0, 799, 160).round().astype(int)
        T = np.array(mm[v, 0][sel])
        lo, hi = torch.zeros(2, device=dev), torch.tensor([c["L"], c["W"]], device=dev, dtype=torch.float32)
        R = torch.tensor(rx_all[v][sel], dtype=torch.float32, device=dev)
        S = torch.tensor(src_all[v][0:1], dtype=torch.float32, device=dev)
        out = {"room": rows()[v]["filename"], "L": c["L"], "W": c["W"], "alpha_min": min(c["edge_alphas"]), "alpha_max": max(c["edge_alphas"])}
        for tag, al in (("alpha015", [0.15] * len(verts)), ("edges", c["edge_alphas"])):
            z = build_cond_vector_2d("geom_token", c["L"], c["W"], al, verts=verts, device=dev)
            P = []
            with torch.no_grad():
                for s0 in range(0, len(R), 32):
                    r = R[s0:s0 + 32]
                    P.append(renderer(model, r, S.expand(len(r), -1), lo, hi, z_s=z.unsqueeze(0).expand(len(r), -1), n_out=601).cpu().numpy())
            P = np.concatenate(P)
            e = {"h0_truth": float(np.mean(np.abs(T[:, 0]))), "h0_pred": float(np.mean(np.abs(P[:, 0]))),
                 "h0_pred_over_truth": float(np.mean(np.abs(P[:, 0])) / np.mean(np.abs(T[:, 0])))}
            for b, (i, j) in B.items():
                a, t = db(P[:, i:j]), db(T[:, i:j])
                x = a - a.mean(0)
                y = t - t.mean(0)
                with np.errstate(all="ignore"):
                    rr = np.nanmean((x * y).sum(0) / np.sqrt((x * x).sum(0) * (y * y).sum(0)))
                e["r_" + b], e["lsd_" + b] = float(rr), float(np.mean(np.abs(a - t)))
            e["l1_dc"] = float(np.mean(np.abs(P[:, :41].real - T[:, :41].real)) + np.mean(np.abs(P[:, :41].imag - T[:, :41].imag)))
            e["l1_rest"] = float(np.mean(np.abs(P[:, 41:].real - T[:, 41:].real)) + np.mean(np.abs(P[:, 41:].imag - T[:, 41:].imag)))
            out[tag] = e
        rep.append(out)
        print(out["room"][:34], "alpha %.2f-%.2f" % (out["alpha_min"], out["alpha_max"]), "| L1 dc/rest alpha015 %.2f/%.4f edges %.2f/%.4f" % (
            out["alpha015"]["l1_dc"], out["alpha015"]["l1_rest"], out["edges"]["l1_dc"], out["edges"]["l1_rest"]),
            "| h0 pred/truth edges %.3f" % out["edges"]["h0_pred_over_truth"], flush=True)
    Path("outputs/p4_10").mkdir(parents=True, exist_ok=True)
    m = lambda tag, k: float(np.mean([x[tag][k] for x in rep]))  # noqa: E731
    summ = {t: {k: m(t, k) for k in rep[0][t]} for t in ("alpha015", "edges")}
    json.dump({"summary": summ, "rooms": rep}, open("outputs/p4_10/a14_probe.json", "w"), indent=1)
    print("SUMMARY", json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
