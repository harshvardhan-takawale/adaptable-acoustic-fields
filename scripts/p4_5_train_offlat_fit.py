"""P4-5 (manager item 6, D125): a NON-circular measure of how well a model fits its training rooms.

D117 used the trainer's validation LSD as its "fit" predictor. Validation receivers are held-out
indices of the training lattice, so that number is inflated by the very memorisation it was meant
to predict. This measures fit on TRAINING geometries at positions OFF the shared lattice:

  rooms     : 23 training rooms -- the 92-room corpus (`p4_3_shapes_manifest`, train split, nested in
              the 250 and in every P4-5 arm's corpus), sorted by filename, every 4th. Fixed in D125.
  receivers : B6's corpus (`data/track_p4_5_offlat`) -- the same rooms re-simulated with 800
              receivers drawn from ALL in-room 0.02 m nodes -- keeping only those OFF the
              0.12 + 0.06 k lattice (~89%). No new simulation.
  metrics   : band LSD (mean |dB error|) over 0-100 Hz (bins 0-199; the PRIMARY predictor, D125 --
              the band every arm trains from step 0, B3's curriculum included), 0-200 and 0-300 Hz
              (the validation LSD's band); dB spatial Pearson averaged over 25-100 and 25-300 Hz.

B6 TRAINED on exactly these receivers, so for B6 this is an in-sample number, reported but excluded
from the D117 re-score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from aaf.data.shape_configs import SRC, configs_from_rows

MANIFEST92 = "configs/sweeps_2d_mat/p4_3_shapes_manifest.json"
DATA = Path("data/track_p4_5_offlat")
OUT = Path("outputs/p4_5/train_offlat_fit")


def rooms():
    cs = sorted(configs_from_rows(json.load(open(MANIFEST92))["configs"], split="train"), key=lambda c: c.filename)
    return cs[::4]


def onlat(rx):
    k = (np.asarray(rx) - 0.12) / 0.06
    return np.all(np.abs(k - np.round(k)) < 1e-6, axis=1)


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def render(name, ck):
    import torch

    from aaf.eval.p3_2_eval import load_model
    from aaf.models.conditioning_2d import build_cond_vector_2d
    dev = torch.device("cuda")
    model, renderer, cfg, _m, it = load_model(Path(ck), dev)
    model.eval(); renderer.eval()
    assert getattr(model, "token_pool", None) == cfg.get("token_pool")
    rows = []
    for c in rooms():
        with h5py.File(DATA / c.filename) as f:
            T = np.asarray(f["ism/H_complex"])[:, :601]
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
        off = ~onlat(rx)
        T, rx = T[off], rx[off]
        cond = build_cond_vector_2d("geom_token", c.L, c.W, c.edge_alphas, verts=c.verts, device=dev)
        R = torch.tensor(rx, dtype=torch.float32, device=dev)
        S = torch.tensor(np.asarray(SRC, float), dtype=torch.float32, device=dev).unsqueeze(0)
        P = np.zeros_like(T)
        for s0 in range(0, len(rx), 8):
            r = R[s0:s0 + 8]
            with torch.no_grad():
                H = renderer(model, r, S.expand(r.shape[0], -1), torch.zeros(2, device=dev),
                             torch.tensor([c.L, c.W], device=dev, dtype=torch.float32),
                             z_s=cond.unsqueeze(0).expand(r.shape[0], -1))
            P[s0:s0 + 8] = H[:, :601].cpu().numpy()
        dP, dT = db(P), db(T)
        a = dP[:, 50:601] - dP[:, 50:601].mean(0); b = dT[:, 50:601] - dT[:, 50:601].mean(0)
        r_bins = (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))      # bins 50..600
        rows.append({"room": c.label, "n_off": int(len(rx)),
                     "lsd_0_100": float(np.mean(np.abs(dP[:, :200] - dT[:, :200]))),
                     "lsd_0_300": float(np.mean(np.abs(dP - dT))),
                     "lsd_0_200": float(np.mean(np.abs(dP[:, :401] - dT[:, :401]))),
                     "r_25_100": float(np.nanmean(r_bins[:150])), "r_25_300": float(np.nanmean(r_bins)),
                     # D137 item (1): per band, for the training-vs-test comparison
                     "r_100_200": float(np.nanmean(r_bins[150:350])), "r_200_300": float(np.nanmean(r_bins[350:551])),
                     "lsd_100_200": float(np.mean(np.abs(dP[:, 200:400] - dT[:, 200:400]))),
                     "lsd_200_300": float(np.mean(np.abs(dP[:, 400:601] - dT[:, 400:601])))})
    rep = {"name": name, "checkpoint": str(ck), "iter": int(it), "n_rooms": len(rows),
           "lsd_0_100": float(np.mean([r["lsd_0_100"] for r in rows])),
           "lsd_0_300": float(np.mean([r["lsd_0_300"] for r in rows])),
           "lsd_0_200": float(np.mean([r["lsd_0_200"] for r in rows])),
           "r_25_100": float(np.mean([r["r_25_100"] for r in rows])),
           **{k: float(np.mean([r[k] for r in rows])) for k in ("r_100_200", "r_200_300", "lsd_100_200", "lsd_200_300")},
           "r_25_300": float(np.mean([r["r_25_300"] for r in rows])), "rooms": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(OUT / "{}.json".format(name), "w"), indent=1)
    print("{:14s} iter {:6d}  off-lattice TRAIN fit: LSD 0-100 {:.3f}, 0-200 {:.3f}, 0-300 {:.3f} dB; r 25-100 {:+.3f}, 25-300 {:+.3f}".format(
        name, it, rep["lsd_0_100"], rep["lsd_0_200"], rep["lsd_0_300"], rep["r_25_100"], rep["r_25_300"]), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="NAME=CKPT,NAME=CKPT,...")
    a = ap.parse_args()
    for it in a.items.split(","):
        n, ck = it.split("=", 1)
        if (OUT / "{}.json".format(n)).exists():
            print("skip", n); continue
        render(n, ck)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
