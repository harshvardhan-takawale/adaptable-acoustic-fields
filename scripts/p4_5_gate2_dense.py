"""P4-5 / D99: re-score Gate 2 on the 15 frozen test shapes at DENSE receivers, split by whether
each receiver sits on the training-receiver lattice the model has memorised.

THE PROTOCOL IS THE FROZEN ONE, VERBATIM. The render loop and every metric below are copied from
`scripts/p4_2_stage2_eval.py` (lines 98-164): six modes of the bounding box, dB spatial Pearson
per mode averaged per shape, in-slab and out-slab means, LOS per shape, NLOS POOLED per slab,
deficit = LOS_in - NLOS_pooled_in, band LSD, phase, receiver-0 RIR. Its helpers are imported, not
re-written. Only the RECEIVER SET changes -- which is the point.

Three subsets of one render: `all` (what the frozen evaluator reports when pointed at the dense
data), `on_lattice` (points of x, y = 0.12 + 0.06 k -- the positions every 800-receiver number
in the project has been measured at) and `off_lattice` (everywhere else: the model as a
continuous field). Rendering once and subsetting avoids tripling a ~40-minute render.

EXACTNESS IS CHECKED, NOT ASSUMED. `--data-dir data/track_p4_2_shapes` (the original 800
receivers) must reproduce a checkpoint's published GATE2.json to the last digit; and on the dense
data, `all` must match the frozen evaluator itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from aaf.data.shape_configs import SRC, configs_from_rows, in_holdout
from aaf.eval.modal_projection import enumerate_modes
from aaf.eval.p3_2_eval import load_model
from aaf.models.conditioning_2d import build_cond_vector_2d
from scripts.p4_2_stage2_eval import (
    DF_HZ, GATE_MAX_NLOS_DEFICIT, GATE_PEARSON_MIN, N_MODES, _circ, _db, _lsd, _pearson,
)

MANIFEST = "configs/sweeps_2d_mat/p4_2_shapes_manifest.json"


def on_lattice(rx):
    k = (np.asarray(rx) - 0.12) / 0.06
    return np.all(np.abs(k - np.round(k)) < 1e-6, axis=1)


def render_all(ck, data_dir, cache, dev):
    test = configs_from_rows(json.load(open(MANIFEST))["configs"], split="test")
    key = "|".join([str(ck), str(data_dir)])                     # D89: the dir is in the key
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            print("[cache] reused", cache, flush=True)
            return [(c, z["P%d" % i], z["T%d" % i], z["rx%d" % i], z["los%d" % i])
                    for i, c in enumerate(test)]
    model, renderer, cfg, _meta, it = load_model(ck, dev)
    model.eval(); renderer.eval()                                # D49 C3
    out = []
    for c in test:                                               # verbatim: stage2_eval 98-116
        with h5py.File(Path(data_dir) / c.filename) as f:
            T = np.asarray(f["ism/H_complex"])
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
            los = np.asarray(json.loads(f.attrs["line_of_sight"]), dtype=bool)
        cond = build_cond_vector_2d("geom_token", c.L, c.W, c.edge_alphas,
                                    verts=c.verts, device=dev)
        room_min = torch.zeros(2, device=dev)
        room_max = torch.tensor([c.L, c.W], device=dev, dtype=torch.float32)
        R = torch.tensor(rx, dtype=torch.float32, device=dev)
        S = torch.tensor(np.asarray(SRC, float), dtype=torch.float32, device=dev).unsqueeze(0)
        P = np.zeros_like(T)
        for s0 in range(0, len(rx), 8):
            sl = slice(s0, s0 + 8)
            r = R[sl]
            with torch.no_grad():
                H = renderer(model, r, S.expand(r.shape[0], -1), room_min, room_max,
                             z_s=cond.unsqueeze(0).expand(r.shape[0], -1))
            P[sl] = H[:, :T.shape[1]].cpu().numpy()
        out.append((c, P, T, rx, los))
        print("  rendered {:34s} n_rx {:5d}".format(c.label, len(rx)), flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    blob = {"key": key}
    for i, (c, P, T, rx, los) in enumerate(out):
        blob["P%d" % i], blob["T%d" % i], blob["rx%d" % i], blob["los%d" % i] = P, T, rx, los
    np.savez_compressed(cache, **blob)
    return out


def score(rooms, sel_fn):
    """stage2_eval lines 118-164, verbatim, on the receivers `sel_fn` keeps."""
    per_shape = []
    pooled = {"in": {"nlos_p": [], "nlos_t": []}, "out": {"nlos_p": [], "nlos_t": []}}
    for c, P_all, T_all, rx_all, los_all in rooms:
        s = sel_fn(rx_all)
        if s.sum() < 3:
            # e.g. the OFF-lattice subset of the original 800-receiver data, which is empty by
            # construction: every one of those receivers sits on the lattice. A room with no
            # receivers in a subset contributes nothing to it rather than crashing it.
            continue
        P, T, los = P_all[s], T_all[s], los_all[s]
        modes = enumerate_modes(c.L, c.W, f_max=200.0)[:N_MODES]
        sp, sp_los, sp_nlos = [], [], []
        key = "in" if in_holdout(c.d_hat) else "out"
        for m in modes:
            b = int(round(m.f / DF_HZ))
            if b >= T.shape[1]:
                continue
            pm, tm = _db(P[:, b]), _db(T[:, b])
            sp.append(_pearson(pm, tm))
            sp_los.append(_pearson(pm[los], tm[los]))
            if (~los).sum() >= 3:
                sp_nlos.append(_pearson(pm[~los], tm[~los]))
            if (~los).any():
                pooled[key]["nlos_p"].append(pm[~los]); pooled[key]["nlos_t"].append(tm[~los])
        rir_p = np.fft.irfft(P[0], n=2 * (T.shape[1] - 1))
        rir_t = np.fft.irfft(T[0], n=2 * (T.shape[1] - 1))
        per_shape.append({
            "label": c.label, "d_hat": round(c.d_hat, 4), "in_holdout": bool(in_holdout(c.d_hat)),
            "n_rx": int(len(P)), "n_nlos": int((~los).sum()),
            "spatial_pearson": float(np.nanmean(sp)) if sp else float("nan"),
            "spatial_pearson_los": float(np.nanmean(sp_los)) if sp_los else float("nan"),
            "spatial_pearson_nlos": float(np.nanmean(sp_nlos)) if sp_nlos else None,
            "band_lsd_db": _lsd(P, T), "phase_circ_corr": _circ(np.angle(P), np.angle(T)),
            "rir_pearson": _pearson(rir_p, rir_t)})
    if not per_shape:
        return None

    def agg(k, f):
        v = [r[f] for r in per_shape if r["in_holdout"] == (k == "in") and r[f] is not None
             and np.isfinite(r[f])]
        return float(np.mean(v)) if v else float("nan")

    def pooled_nlos(k):
        if not pooled[k]["nlos_p"]:
            return float("nan"), 0
        p = np.concatenate(pooled[k]["nlos_p"]); t = np.concatenate(pooled[k]["nlos_t"])
        return _pearson(p, t), int(len(p))

    r_in, r_out = agg("in", "spatial_pearson"), agg("out", "spatial_pearson")
    los_in = agg("in", "spatial_pearson_los")
    nlos_in, n_in = pooled_nlos("in")
    nlos_out, n_out = pooled_nlos("out")
    deficit = los_in - nlos_in if np.isfinite(nlos_in) else float("nan")
    return {"in_slab": {"spatial_pearson": r_in, "band_lsd_db": agg("in", "band_lsd_db"),
                        "phase": agg("in", "phase_circ_corr"), "rir": agg("in", "rir_pearson"),
                        "los": los_in, "nlos_pooled": nlos_in, "n_pooled_nlos": n_in},
            "out_slab": {"spatial_pearson": r_out, "band_lsd_db": agg("out", "band_lsd_db"),
                         "phase": agg("out", "phase_circ_corr"),
                         "rir": agg("out", "rir_pearson"),
                         "los": agg("out", "spatial_pearson_los"), "nlos_pooled": nlos_out,
                         "n_pooled_nlos": n_out},
            "nlos_deficit": deficit,
            "passed": bool(r_in >= GATE_PEARSON_MIN and np.isfinite(deficit)
                           and deficit <= GATE_MAX_NLOS_DEFICIT),
            "n_rx_total": int(sum(r["n_rx"] for r in per_shape)), "per_shape": per_shape}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data-dir", default="data/track_p4_5_testdense")
    ap.add_argument("--verify-against", default=None,
                    help="a published GATE2.json the 'all' subset must reproduce exactly")
    ap.add_argument("--out", default="outputs/p4_5/gate2_dense")
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = Path(a.out)
    tag = "{}__{}".format(a.name, Path(a.data_dir).name)
    rooms = render_all(Path(a.checkpoint), a.data_dir, out / "cache" / (tag + ".npz"), dev)
    res = {"name": a.name, "checkpoint": a.checkpoint, "data_dir": a.data_dir,
           "protocol": "scripts/p4_2_stage2_eval.py lines 98-164, verbatim",
           "subsets": {"all": score(rooms, lambda rx: np.ones(len(rx), bool)),
                       "on_lattice": score(rooms, on_lattice),
                       "off_lattice": score(rooms, lambda rx: ~on_lattice(rx))}}
    if a.verify_against:
        ref = json.load(open(a.verify_against))
        chk = {}
        for slab in ("in_slab", "out_slab"):
            for q in ("spatial_pearson", "band_lsd_db", "los", "nlos_pooled", "rir"):
                chk["{}.{}".format(slab, q)] = (ref[slab][q], res["subsets"]["all"][slab][q])
        bad = {k: v for k, v in chk.items() if not (v[0] == v[1] or (np.isnan(v[0]) and np.isnan(v[1])))}
        res["verification"] = {"against": a.verify_against, "exact": not bad, "mismatches": bad}
        print("VERIFY vs {}: {}".format(a.verify_against,
                                        "EXACT, every field" if not bad else "MISMATCH {}".format(bad)))
        if bad:
            raise SystemExit(3)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / (tag + ".json"), "w"), indent=1, default=float)
    print("{:12s} {:>9s} {:>9s} {:>9s} {:>8s}  {}".format("subset", "n_rx", "in-slab", "out-slab",
                                                         "deficit", "Gate 2"))
    for k, s in res["subsets"].items():
        if s is None:
            print("{:12s}   (no receivers in this subset)".format(k))
            continue
        print("{:12s} {:9d} {:+9.4f} {:+9.4f} {:+8.3f}  {}".format(
            k, s["n_rx_total"], s["in_slab"]["spatial_pearson"], s["out_slab"]["spatial_pearson"],
            s["nlos_deficit"], "PASS" if s["passed"] else "FAIL"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
