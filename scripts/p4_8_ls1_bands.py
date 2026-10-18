"""P4-8 (D176(c), manager): LS1's neighbour-difference term PER BAND at a checkpoint, relative to the no-difference value
1.0, plus each band's share of the total loss. Measured OFFLINE on the arm's own training rooms (the arms were already
running when the log was requested), with the trainer's pair rule:
  rooms   : every 25th config of the GX corpus (40 rooms), training receivers only (val_rx_indices held out);
  pairs   : per room, 25 anchors drawn uniformly among training receivers with a neighbour within 0.30 m, one neighbour
            each (seed 20261018) -> 1000 pairs; rendered in eval mode (no ray jitter), the room's own source;
  rel_b   = mean_pairs |dP_b - dT_b| / mean_pairs |dT_b| per bin b (exactly 1.0 when the predicted differences are 0),
            averaged over each band's bins (25-100 / 100-200 / 200-300 Hz; also 20-25 Hz, which LS1 also sees);
  shares  : the total loss = PB3 terms (rows x bins 0-300 Hz) + lambda * LS1 (S_init x mean_{pairs, bins 20-300} rel);
            per band: LS1's contribution lambda * S * (n_band / n_LS1bins) * rel_band and the PB3 terms' contribution
            over the band's bins, each as a fraction of the total.
  python scripts/p4_8_ls1_bands.py --items "LS1A_10K=outputs/p4_8/arms/LS1A/ckpt_iter0010000.pt+..."
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

MAN = "configs/sweeps_2d_mat/p4_7_c500_cx500_manifest.json"
DATA = Path("data/track_p4_7_mix")
BANDS = {"0-20": (0, 40), "20-25": (40, 50), "25-100": (50, 200), "100-200": (200, 400), "200-300": (400, 601)}
OUT = Path("outputs/p4_8/ls1_bands")


def main():
    import torch

    from aaf.data.poly_configs import configs_from_rows
    from aaf.eval.p3_2_eval import load_model
    from aaf.models.conditioning_2d import build_cond_vector_2d
    from aaf.train.multi_room_2d_mat import _losses, val_rx_indices
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    cfgs = configs_from_rows(json.load(open(MAN))["configs"], split="train")[::25]
    rng = np.random.default_rng(20261018)
    rooms = []
    for c in cfgs:
        with h5py.File(DATA / c.filename) as f:
            H = np.asarray(f["ism/H_complex"])[:, :601]
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]), dtype=np.float64)
            src = np.asarray(json.loads(f.attrs["source_pos"]), dtype=np.float64)
        tr = np.array(sorted(set(range(len(rx))) - set(val_rx_indices(len(rx)))))
        d2 = ((rx[tr][:, None] - rx[tr][None]) ** 2).sum(-1)
        nb = [np.nonzero((d2[i] <= 0.09 + 1e-9) & (np.arange(len(tr)) != i))[0] for i in range(len(tr))]
        ok = [i for i in range(len(tr)) if len(nb[i])]
        anc = rng.choice(ok, size=25, replace=False)
        pairs = [(tr[i], tr[rng.choice(nb[i])]) for i in anc]
        rooms.append((c, H, rx, src, pairs))
    for item in a.items.replace("+", ",").split(","):
        name, ck = item.split("=", 1)
        model, renderer, cfg, _m, it = load_model(Path(ck), dev)
        model.eval(); renderer.eval()
        lam = float(cfg.get("ls1_weight", 0.0))
        meta = json.load(open(Path(ck).parent / "train_meta.json"))
        S = float(meta.get("spatial_loss", {}).get("S_init_spec_terms", float("nan")))
        dP, dT, Pr, Tr = [], [], [], []
        with torch.no_grad():
            for c, H, rx, src, pairs in rooms:
                idx = np.array([k for p in pairs for k in p])
                r = torch.tensor(rx[idx], dtype=torch.float32, device=dev)
                tx = torch.tensor(src, dtype=torch.float32, device=dev).expand(len(idx), -1)
                z = build_cond_vector_2d("geom_token", c.L, c.W, c.edge_alphas, verts=c.verts).float().to(dev)
                P = []
                for s0 in range(0, len(idx), 8):
                    P.append(renderer(model, r[s0:s0 + 8], tx[s0:s0 + 8], torch.zeros(2, device=dev),
                                      torch.tensor([c.L, c.W], device=dev, dtype=torch.float32),
                                      z_s=z.expand(min(8, len(idx) - s0), -1))[:, :601].cpu().numpy())
                P = np.concatenate(P)
                T = H[idx]
                dP.append(P[0::2] - P[1::2]); dT.append(T[0::2] - T[1::2]); Pr.append(P); Tr.append(T)
        dP, dT, Pr, Tr = (np.concatenate(x) for x in (dP, dT, Pr, Tr))
        num = np.abs(dP - dT).mean(0)
        den = np.abs(dT).mean(0)
        rel = num / den                                                  # per bin; 1.0 for zero predicted differences
        terms = _losses(torch.tensor(Pr), torch.tensor(Tr), 0)
        w_r, w_i, w_a, w_p = cfg["weights"] if "weights" in cfg else (1.0, 1.0, 1.0, 0.1)
        pb3 = float(w_r * terms["L_spec_real"] + w_i * terms["L_spec_imag"] + w_a * terms["L_amp"] + w_p * terms["L_phase"])
        ls1 = S * float(rel[40:601].mean())
        total = pb3 + lam * ls1
        rep = {"name": name, "checkpoint": ck, "iter": int(it), "lambda": lam, "S_init": S, "n_pairs": int(len(dP)),
               "rel_per_band": {}, "share_of_total": {}, "PB3_total": pb3, "LS1_total": ls1, "total": total}
        Pt, Tt = torch.tensor(Pr), torch.tensor(Tr)
        for b, (lo, hi) in BANDS.items():
            rep["rel_per_band"][b] = float(rel[lo:hi].mean())
            tb = _losses(Pt[:, lo:hi], Tt[:, lo:hi], 0)
            pb = float(w_r * tb["L_spec_real"] + w_i * tb["L_spec_imag"] + w_a * tb["L_amp"] + w_p * tb["L_phase"]) * (hi - lo) / 601
            lb = lam * S * float(rel[max(lo, 40):hi].sum()) / (601 - 40) if hi > 40 else 0.0   # LS1 sees 20-300 Hz
            rep["share_of_total"][b] = {"LS1": lb / total, "PB3": pb / total}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "{}.json".format(name)).write_text(json.dumps(rep, indent=1))
        print(name, "rel", {k: round(v, 3) for k, v in rep["rel_per_band"].items()},
              "| LS1 share", {k: round(v["LS1"], 3) for k, v in rep["share_of_total"].items() if "LS1" in v},
              "| PB3 {:.3f} LS1 {:.3f} total {:.3f}".format(pb3, ls1, total), flush=True)
        del model, renderer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
