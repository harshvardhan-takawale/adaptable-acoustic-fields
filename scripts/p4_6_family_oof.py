"""P4-6 t2 (D160) and the standard report's OUT-OF-FAMILY component: score a checkpoint on D111's dense
family rooms -- the two pre-registered median test rooms per family (`build_p4_5_rooms.
family_test_rooms`, first and second pick; D111 / D112) for rect, L, T, DN and U, each re-simulated
on a 64x64 lattice clipped to the polygon.

Per room, OFF the 0.12 + 0.06 k training lattice (Rule 11), RAW and after the render-time physical
projection (`aaf.eval.physical_projection.project_field`, POST-PROCESSING, D160(e)):
  * band-mean r over 25-100 / 100-200 / 200-300 Hz (per-bin column Pearson of dB, averaged) and
    r at 100 Hz and 200 Hz (+-2.5 Hz);
  * band LSD (mean |dB error|) over the same bands;
  * D111's own metric: 6 bounding-box modes, dB spatial Pearson (`p4_4_family_eval._metrics`).
For a model trained on the L-notch shape corpus (B5, PB3), T / DN / U are OUT of family and rect / L
are the in-family reference. For g2 (U held out), U is out of family and the rest are in-family
held-out instances.
  python scripts/p4_6_family_oof.py --items "PB3_60K=outputs/p4_5/arms/PB3/ckpt_iter0060000.pt+..."
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from aaf.eval.modal_projection import enumerate_modes
from aaf.eval.physical_projection import project_field
from scripts.build_p4_5_rooms import OUT as ROOT, family_test_rooms
from scripts.p4_4_family_eval import N_MODES, _metrics, _render
from scripts.p4_5_gate2_dense import on_lattice

BANDS = {"25-100": (50, 200), "100-200": (200, 400), "200-300": (400, 601)}
POINTS = {"r_100Hz": (195, 206), "r_200Hz": (395, 406)}
OUT = Path("outputs/p4_6/family_oof")
CACHE = Path("outputs/p4_6/cache")


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def colr(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def scores(P, T, los, modes):
    r = colr(db(P), db(T))
    out = {b: float(np.nanmean(r[lo:hi])) for b, (lo, hi) in BANDS.items()}
    out.update({k: float(np.nanmean(r[lo:hi])) for k, (lo, hi) in POINTS.items()})
    out.update({"lsd_" + b: float(np.mean(np.abs(db(P[:, lo:hi]) - db(T[:, lo:hi])))) for b, (lo, hi) in BANDS.items()})
    out["mode_r_D111"] = float(_metrics(P, T, los, modes)["spatial_pearson"])
    return out


def main():
    import torch

    from aaf.eval.p3_2_eval import load_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    rooms = family_test_rooms() + family_test_rooms(second=True)
    for item in a.items.replace("+", ",").split(","):
        name, ck = item.split("=", 1)
        model, renderer, cfg, _m, it = load_model(Path(ck), dev)
        model.eval(); renderer.eval()
        rep = {"name": name, "checkpoint": ck, "iter": int(it), "projection": "POST-PROCESSING (D160(e))",
               "rooms": []}
        blob = {}
        for j, (fam, c) in enumerate(rooms):
            with h5py.File(Path(ROOT["famdense"]) / c.filename) as f:
                T = np.asarray(f["ism/H_complex"])[:, :601]
                rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
                los = np.asarray(json.loads(f.attrs["line_of_sight"]), dtype=bool)
                keep = np.asarray(json.loads(f.attrs["lattice_keep"]), bool).reshape(64, 64)
            assert len(rx) == keep.sum(), (c.filename, len(rx), keep.sum())
            P = _render(model, renderer, c, rx, T.shape[1], dev)[:, :601]
            Pp = project_field(P, keep, (c.L - 0.24) / 63, (c.W - 0.24) / 63)
            off = ~on_lattice(rx)
            modes = enumerate_modes(c.L, c.W, f_max=200.0)[:N_MODES]
            row = {"family": fam, "room": c.label, "pick": "first" if j < 5 else "second",
                   "n_off": int(off.sum()),
                   "raw": scores(P[off], T[off], los[off], modes),
                   "projected": scores(Pp[off], T[off], los[off], modes)}
            rep["rooms"].append(row)
            blob["P%d" % j], blob["T%d" % j], blob["rx%d" % j] = P.astype(np.complex64), T, rx
            print("{:9s} {:5s} {:50s} raw r {:.3f}/{:.3f}/{:.3f} (100 Hz {:+.3f}) | proj {:.3f}/{:.3f}/{:.3f} | mode r {:+.3f}".format(
                name, fam, c.label, *[row["raw"][b] for b in BANDS], row["raw"]["r_100Hz"],
                *[row["projected"][b] for b in BANDS], row["raw"]["mode_r_D111"]), flush=True)
        fams = ("rect", "L", "T", "DN", "U")
        keys = list(BANDS) + list(POINTS) + ["lsd_" + b for b in BANDS] + ["mode_r_D111"]
        rep["per_family"] = {f: {v: {k: float(np.mean([r[v][k] for r in rep["rooms"] if r["family"] == f]))
                                     for k in keys} for v in ("raw", "projected")} for f in fams}
        for grp, fs in (("shape_corpus_out_of_family_T_DN_U", ("T", "DN", "U")), ("in_family_ref_rect_L", ("rect", "L"))):
            rep[grp] = {v: {k: float(np.mean([r[v][k] for r in rep["rooms"] if r["family"] in fs])) for k in keys}
                        for v in ("raw", "projected")}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "{}.json".format(name)).write_text(json.dumps(rep, indent=1))
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE / "famdense_{}.npz".format(name), **blob)
        del model, renderer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
