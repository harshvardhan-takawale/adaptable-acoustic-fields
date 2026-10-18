"""P4-8 Step 0c / standard report (D174(a), (d)): score a checkpoint on the FROZEN unseen-family set
(configs/sweeps_2d_mat/p4_8_unseen_manifest.json: 8 U + 8 C3 rooms, dense as D111, data/track_p4_8_unseen).

Exactly `p4_6_family_oof`'s protocol on a manifest-driven room list: per room, OFF the 0.12 + 0.06 k lattice
(Rule 11), raw and after the render-time physical projection (POST-PROCESSING, D160(e)): band-mean r over
25-100 / 100-200 / 200-300 Hz (per-bin column Pearson of dB), r at 100 / 200 Hz, band LSD, D111's 6-mode r.
UF = the mean over all 16 rooms at 25-100 Hz (also 100-200, 200-300); U and C3 are reported separately.
  python scripts/p4_8_unseen_eval.py --items "G1_60K=outputs/p4_6/arms/G1/ckpt_iter0060000.pt+GX_60K=..."
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from aaf.eval.modal_projection import enumerate_modes
from aaf.eval.physical_projection import project_field
from scripts.p4_4_family_eval import N_MODES, _render
from scripts.p4_5_gate2_dense import on_lattice
from scripts.p4_6_family_oof import BANDS, POINTS, scores

MANIFEST = "configs/sweeps_2d_mat/p4_8_unseen_manifest.json"
DATA = Path("data/track_p4_8_unseen")
OUT = Path("outputs/p4_8/unseen")
CACHE = Path("outputs/p4_8/cache")


def rooms():
    m = json.load(open(MANIFEST))
    return [SimpleNamespace(L=float(r["L"]), W=float(r["W"]), verts=[tuple(v) for v in r["verts"]], family=r["family"],
                            filename=r["filename"], label=r["label"]) for r in m["configs"]], m["rows_sha256"]


def main():
    import torch

    from aaf.eval.p3_2_eval import load_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    rs, sha = rooms()
    for item in a.items.replace("+", ",").split(","):
        name, ck = item.split("=", 1)
        model, renderer, cfg, _m, it = load_model(Path(ck), dev)
        model.eval(); renderer.eval()
        rep = {"name": name, "checkpoint": ck, "iter": int(it), "manifest_sha": sha,
               "projection": "POST-PROCESSING (D160(e))", "rooms": []}
        blob = {}
        for j, c in enumerate(rs):
            with h5py.File(DATA / c.filename) as f:
                T = np.asarray(f["ism/H_complex"])[:, :601]
                rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
                los = np.asarray(json.loads(f.attrs["line_of_sight"]), dtype=bool)
                keep = np.asarray(json.loads(f.attrs["lattice_keep"]), bool).reshape(64, 64)
            assert len(rx) == keep.sum(), (c.filename, len(rx), keep.sum())
            P = _render(model, renderer, c, rx, T.shape[1], dev)[:, :601]
            Pp = project_field(P, keep, (c.L - 0.24) / 63, (c.W - 0.24) / 63)
            off = ~on_lattice(rx)
            modes = enumerate_modes(c.L, c.W, f_max=200.0)[:N_MODES]
            row = {"family": c.family, "room": c.label, "n_off": int(off.sum()),
                   "raw": scores(P[off], T[off], los[off], modes), "projected": scores(Pp[off], T[off], los[off], modes)}
            rep["rooms"].append(row)
            blob["P%d" % j], blob["T%d" % j], blob["rx%d" % j] = P.astype(np.complex64), T, rx
            print("{:10s} {:3s} {:56s} raw r {:.3f}/{:.3f}/{:.3f} | proj {:.3f}/{:.3f}/{:.3f}".format(
                name, c.family, c.label, *[row["raw"][b] for b in BANDS], *[row["projected"][b] for b in BANDS]),
                flush=True)
        keys = list(BANDS) + list(POINTS) + ["lsd_" + b for b in BANDS] + ["mode_r_D111"]
        for grp, fs in (("unseen_all", ("U", "C3")), ("U", ("U",)), ("C3", ("C3",))):
            rep[grp] = {v: {k: float(np.mean([r[v][k] for r in rep["rooms"] if r["family"] in fs])) for k in keys}
                        for v in ("raw", "projected")}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "{}.json".format(name)).write_text(json.dumps(rep, indent=1))
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE / "unseen_{}.npz".format(name), **blob)
        print("{} UF (all 16) raw {}".format(name, {b: round(rep["unseen_all"]["raw"][b], 3) for b in BANDS}), flush=True)
        del model, renderer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
