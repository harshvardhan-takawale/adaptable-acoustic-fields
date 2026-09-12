"""P4-2 GATE 2: does one model generalize across SHAPES?

GATE 2: in-slab test shapes reach spatial Pearson >= 0.80 mean with an NLOS deficit <= 0.15.
The verdict is computed and printed BEFORE any figure is drawn (D62a).

All 15 test shapes are unseen geometries, so every one of their 800 stored receivers is unseen
too -- unlike P4-1 Stage 1, whose headline was a FIT metric because 87.5% of its receivers were
supervised. This is a real generalization measurement and needs no caveat about that.

THE NLOS CRITERION IS READ ON THE POOLED IN-SLAB SET, AND HERE IS WHY
---------------------------------------------------------------------
The dataset gate measured the actual NLOS census: in-slab test shapes average 1.2% NLOS -- 59
receivers pooled across all 6 -- and 5 of the 15 test shapes have ZERO NLOS receivers. Shadow
needs BOTH depth and width, and the family samples w_hat down to 0.18, so shallow or narrow
notches cast none. Averaging per-shape NLOS Pearsons would therefore average several undefined
values and a handful computed on 3-12 points. The pooled set (n = 59) is the only estimate the
data supports, and per-shape counts are reported beside it so the thinness is visible rather
than buried.

The sigma probe runs on every test shape that has an NLOS receiver. Its ratio is reported
ALONGSIDE the verdict regardless of outcome: a model that scores well in-distribution while
sigma stays at ~1.0 is memorizing each shape in `signal`, and that predicts Z-corridor transfer
will fail no matter how good Gate 2 looks.
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
from aaf.eval.sigma_probe import probe_sigma_occlusion
from aaf.models.conditioning_2d import build_cond_vector_2d

GATE_PEARSON_MIN = 0.80
GATE_MAX_NLOS_DEFICIT = 0.15
N_MODES = 6
DF_HZ = 0.5


def _db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def _pearson(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _lsd(p, t):
    return float(np.mean(np.abs(_db(p) - _db(t))))


def _circ(pa, pb):
    a = np.asarray(pa, float).ravel() - np.angle(np.mean(np.exp(1j * np.asarray(pa).ravel())))
    b = np.asarray(pb, float).ravel() - np.angle(np.mean(np.exp(1j * np.asarray(pb).ravel())))
    den = np.sqrt(np.sum(np.sin(a) ** 2) * np.sum(np.sin(b) ** 2))
    return float(np.sum(np.sin(a) * np.sin(b)) / den) if den > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-dir", default="data/track_p4_2_shapes")
    ap.add_argument("--manifest", default="configs/sweeps_2d_mat/p4_2_shapes_manifest.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default=None)
    a = ap.parse_args()

    run = Path(a.run_dir)
    out = Path(a.out) if a.out else run.parent / (run.name + "_eval")
    out.mkdir(parents=True, exist_ok=True)
    ck = Path(a.checkpoint) if a.checkpoint else sorted(run.glob("ckpt_iter*.pt"))[-1]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, renderer, cfg, meta, it = load_model(ck, dev)
    model.eval(); renderer.eval()                        # D49 C3
    pool = cfg.get("token_pool") or "masked_mean"
    print("[arm] {} iter {} | pool {} | loss_lo {} Hz | world_scale {}".format(
        run.name, it, pool, cfg.get("loss_band_lo_hz", 0.0), cfg.get("world_scale")), flush=True)

    rows = json.load(open(a.manifest))["configs"]
    test = configs_from_rows(rows, split="test")
    print("[data] {} test shapes ({} in-slab)".format(
        len(test), sum(1 for c in test if in_holdout(c.d_hat))), flush=True)

    per_shape, pooled = [], {"in": {"p": [], "t": [], "nlos_p": [], "nlos_t": []},
                             "out": {"p": [], "t": [], "nlos_p": [], "nlos_t": []}}
    for c in test:
        with h5py.File(Path(a.data_dir) / c.filename) as f:
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
            pooled[key]["p"].append(pm); pooled[key]["t"].append(tm)
            if (~los).sum() >= 3:
                sp_nlos.append(_pearson(pm[~los], tm[~los]))
            if (~los).any():
                pooled[key]["nlos_p"].append(pm[~los]); pooled[key]["nlos_t"].append(tm[~los])
        rir_p = np.fft.irfft(P[0], n=2 * (T.shape[1] - 1))
        rir_t = np.fft.irfft(T[0], n=2 * (T.shape[1] - 1))
        per_shape.append({
            "label": c.label, "d_hat": round(c.d_hat, 4), "w_hat": round(c.w_hat, 4),
            "in_holdout": bool(in_holdout(c.d_hat)), "n_rx": int(len(rx)),
            "n_nlos": int((~los).sum()), "nlos_fraction": float(1 - los.mean()),
            "spatial_pearson": float(np.nanmean(sp)) if sp else float("nan"),
            "spatial_pearson_los": float(np.nanmean(sp_los)) if sp_los else float("nan"),
            "spatial_pearson_nlos": float(np.nanmean(sp_nlos)) if sp_nlos else None,
            "band_lsd_db": _lsd(P, T), "phase_circ_corr": _circ(np.angle(P), np.angle(T)),
            "rir_pearson": _pearson(rir_p, rir_t)})
        print("  {:34s} d_hat {:.3f} {:8s} R {:+.3f} | LSD {:5.2f} | NLOS rx {:3d}".format(
            c.label, c.d_hat, "IN-SLAB" if in_holdout(c.d_hat) else "", per_shape[-1]["spatial_pearson"],
            per_shape[-1]["band_lsd_db"], per_shape[-1]["n_nlos"]), flush=True)

    def agg(k, f):
        v = [r[f] for r in per_shape if r["in_holdout"] == (k == "in") and r[f] is not None
             and np.isfinite(r[f])]
        return float(np.mean(v)) if v else float("nan")

    # POOLED NLOS -- the only estimate the census supports (see the module docstring)
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

    print("\n  {:28s} {:>10s} {:>10s}".format("", "IN-SLAB", "OUT-SLAB"))
    print("  {:28s} {:10.3f} {:10.3f}".format("spatial Pearson (mean)", r_in, r_out))
    print("  {:28s} {:10.3f} {:10.3f}".format("band LSD (dB)", agg("in", "band_lsd_db"),
                                              agg("out", "band_lsd_db")))
    print("  {:28s} {:10.3f} {:10.3f}".format("phase circ corr", agg("in", "phase_circ_corr"),
                                              agg("out", "phase_circ_corr")))
    print("  {:28s} {:10.3f} {:10.3f}".format("RIR Pearson", agg("in", "rir_pearson"),
                                              agg("out", "rir_pearson")))
    print("\n  LOS vs NLOS (NLOS is POOLED across shapes -- per-shape n is too small):")
    print("    in-slab  LOS {:+.3f} | NLOS {:+.3f} over {} pooled receiver-modes | deficit {:+.3f}"
          .format(los_in, nlos_in, n_in, deficit))
    print("    out-slab LOS {:+.3f} | NLOS {:+.3f} over {} pooled receiver-modes"
          .format(agg("out", "spatial_pearson_los"), nlos_out, n_out))
    nz = [r["label"] for r in per_shape if r["n_nlos"] == 0]
    if nz:
        print("    {} test shapes have ZERO NLOS receivers -- no per-shape NLOS estimate exists "
              "for them".format(len(nz)))

    crit = {"in_slab_spatial_pearson": {"value": r_in, "op": ">=",
                                        "threshold": GATE_PEARSON_MIN,
                                        "pass": bool(r_in >= GATE_PEARSON_MIN)},
            "nlos_deficit": {"value": deficit, "op": "<=",
                             "threshold": GATE_MAX_NLOS_DEFICIT,
                             "pass": bool(np.isfinite(deficit)
                                          and deficit <= GATE_MAX_NLOS_DEFICIT),
                             "n_pooled_receiver_modes": n_in,
                             "power_note": ("computed on the POOLED in-slab NLOS set; the "
                                            "census gives 1.2% mean NLOS and 5 of 15 test "
                                            "shapes have none at all")}}
    passed = all(c["pass"] for c in crit.values())
    print("\n  GATE 2: {}".format("PASS" if passed else "FAIL"))
    for k, v in crit.items():
        print("    {:26s} {:+.3f} {} {:.2f} -> {}".format(
            k, v["value"], v["op"], v["threshold"], "pass" if v["pass"] else "FAIL"))

    # --- sigma probe: mechanism, reported regardless of the verdict ------------------------
    probes = []
    for c in test:
        with h5py.File(Path(a.data_dir) / c.filename) as f:
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
        p = probe_sigma_occlusion(model, c.verts, SRC, dev, rx=rx,
                                  edge_alphas=list(c.edge_alphas),
                                  world_scale=cfg.get("world_scale"))
        p["label"] = c.label; p["in_holdout"] = bool(in_holdout(c.d_hat))
        for k in ("t", "pts", "sigma"):
            p.pop(k, None)
        probes.append(p)
    ok = [p for p in probes if p.get("probed")]
    ratios = [p["solid_over_air"] for p in ok]
    tr = [p["transmittance_ratio"] for p in ok]
    print("\n  SIGMA PROBE (mechanism) -- {} of {} shapes probeable".format(len(ok), len(test)))
    if ok:
        print("    solid/air ratio  mean {:.4f}  range {:.4f}-{:.4f}".format(
            float(np.mean(ratios)), min(ratios), max(ratios)))
        print("    transmittance ratio (solid / same path in air) mean {:.4f}".format(
            float(np.mean(tr))))
        print("    {}".format(ok[0]["interpretation"]))

    res = {"gate": "p4_2.stage2/1", "arm": run.name, "checkpoint": str(ck), "iter": int(it),
           "token_pool": pool, "loss_band_lo_hz": cfg.get("loss_band_lo_hz", 0.0),
           "criteria": crit, "passed": passed,
           "in_slab": {"spatial_pearson": r_in, "band_lsd_db": agg("in", "band_lsd_db"),
                       "phase": agg("in", "phase_circ_corr"), "rir": agg("in", "rir_pearson"),
                       "los": los_in, "nlos_pooled": nlos_in, "n_pooled_nlos": n_in},
           "out_slab": {"spatial_pearson": r_out, "band_lsd_db": agg("out", "band_lsd_db"),
                        "phase": agg("out", "phase_circ_corr"), "rir": agg("out", "rir_pearson"),
                        "los": agg("out", "spatial_pearson_los"), "nlos_pooled": nlos_out,
                        "n_pooled_nlos": n_out},
           "per_shape": per_shape, "sigma_probe": probes,
           "sigma_summary": {"n_probeable": len(ok),
                             "mean_solid_over_air": float(np.mean(ratios)) if ok else None,
                             "mean_transmittance_ratio": float(np.mean(tr)) if ok else None},
           "shapes_with_zero_nlos": nz}
    json.dump(res, open(out / "GATE2.json", "w"), indent=1, default=float)
    print("\n-> {}".format(out / "GATE2.json"))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
