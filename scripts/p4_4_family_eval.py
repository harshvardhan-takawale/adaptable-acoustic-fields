"""P4-4 Task 3: does corner count extrapolate, and does TOKEN COUNT cost anything?

A dedicated driver rather than a flag on `p4_2_stage2_eval.py`. That script imports the
notch-family config module unconditionally and partitions everything on the scalar
`in_holdout(c.d_hat)`, which is a notch-family-only coordinate -- for a mixed corpus the pooled
in/out-slab number is not defined, and reporting one would average five families into a figure
that describes none of them.

TWO QUESTIONS, DELIBERATELY SEPARATED
-------------------------------------
1. **Corner count.** Rect (0 reflex corners), L (1), U/T/DN (2). Reported per family, so a
   failure localizes to a topology instead of smearing across the corpus. Z becomes Stage 3 only
   if the 2-corner families hold.

2. **Token count, at FIXED geometry.** Comparing a rectangle (4 tokens) with a U (8) confounds
   token count with shape. The control instead re-tokenizes the SAME room with collinear
   vertices: identical physics, identical `.h5`, identical receivers, only the conditioning
   vector changes. `MAX_SEG_POLY = 12` bounds what is reachable -- rect at 4/8/12, L at 6/12,
   and the 8-token families not at all -- which makes the rectangle the cleanest control in the
   corpus: one room, three token counts, literally fixed geometry.

Numbers are printed before any figure is drawn (D62a).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

from aaf.data.multi_notch import (
    D_HAT_HOLDOUT_FAM, FAMILIES, N_TOKENS, configs_from_rows, refine_verts,
)
from aaf.data.shape_configs import SRC
from aaf.eval.modal_decay import band_limited_rir
from aaf.eval.modal_projection import enumerate_modes
from aaf.eval.p3_2_eval import load_model
from aaf.eval.sigma_probe import probe_sigma_occlusion
from aaf.models.conditioning_2d import MAX_SEG_POLY, build_cond_vector_2d

N_MODES = 6
DF_HZ = 0.5
RIR_FS, RIR_N, MODAL_LO = 600.0, 1200, 20.0
REFINE = {"rect": (1, 2, 3), "L": (1, 2)}          # what MAX_SEG_POLY allows


def _db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def _pearson(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _render(model, renderer, cfg, rx, n_bins, dev, verts=None, chunk=8):
    """`verts` overrides the config's own ring -- that is the token-count control."""
    v = list(cfg.verts if verts is None else verts)
    cond = build_cond_vector_2d("geom_token", cfg.L, cfg.W, [0.15] * len(v),
                                verts=v, device=dev)
    room_min = torch.zeros(2, device=dev)
    room_max = torch.tensor([cfg.L, cfg.W], device=dev, dtype=torch.float32)
    R = torch.tensor(rx, dtype=torch.float32, device=dev)
    S = torch.tensor(np.asarray(SRC, float), dtype=torch.float32, device=dev).unsqueeze(0)
    P = np.zeros((len(rx), n_bins), dtype=np.complex128)
    for s0 in range(0, len(rx), chunk):
        sl = slice(s0, s0 + chunk)
        r = R[sl]
        with torch.no_grad():
            H = renderer(model, r, S.expand(r.shape[0], -1), room_min, room_max,
                         z_s=cond.unsqueeze(0).expand(r.shape[0], -1))
        P[sl] = H[:, :n_bins].cpu().numpy()
    return P


def _metrics(P, T, los, modes):
    sp, sp_nlos = [], []
    for m in modes:
        b = int(round(m.f / DF_HZ))
        if b >= T.shape[1]:
            continue
        pm, tm = _db(P[:, b]), _db(T[:, b])
        sp.append(_pearson(pm, tm))
        if (~los).sum() >= 3:
            sp_nlos.append(_pearson(pm[~los], tm[~los]))
    rp = band_limited_rir(P, RIR_FS, RIR_N, f_lo=MODAL_LO, f_hi=300.0)
    rt = band_limited_rir(T, RIR_FS, RIR_N, f_lo=MODAL_LO, f_hi=300.0)
    return {
        "spatial_pearson": float(np.nanmean(sp)) if sp else float("nan"),
        "spatial_pearson_nlos": float(np.nanmean(sp_nlos)) if sp_nlos else float("nan"),
        "band_lsd_db": float(np.mean(np.abs(_db(P) - _db(T)))),
        "rir_pearson_modal": float(np.mean([_pearson(rp[i], rt[i])
                                            for i in range(len(rp))])),
        "nlos_frac": float(1.0 - los.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--manifest", default="configs/sweeps_2d_mat/p4_4_family_manifest.json")
    ap.add_argument("--data-dir", default="data/track_p4_4_family")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default=None)
    a = ap.parse_args()
    run = Path(a.run_dir)
    out = Path(a.out) if a.out else run.parent / (run.name + "_eval")
    out.mkdir(parents=True, exist_ok=True)
    ck = Path(a.checkpoint) if a.checkpoint else sorted(run.glob("ckpt_iter*.pt"))[-1]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, renderer, cfg_d, meta, it = load_model(ck, dev)
    model.eval(); renderer.eval()                                 # D49 C3
    print("[arm] {} iter {} | pool {} | loss_lo {} Hz".format(
        run.name, it, cfg_d.get("token_pool") or "masked_mean",
        cfg_d.get("loss_band_lo_hz", 0.0)), flush=True)

    rows = json.load(open(a.manifest))["configs"]
    test = configs_from_rows(rows, split="test")
    print("[data] {} test rooms across {} families".format(len(test), len(FAMILIES)), flush=True)

    per_room, control = [], []
    for c in test:
        p = Path(a.data_dir) / c.filename
        if not p.exists():
            raise FileNotFoundError("room not built: {}".format(p))
        with h5py.File(p) as f:
            T = np.asarray(f["ism/H_complex"])
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
            los = np.asarray(json.loads(f.attrs["line_of_sight"]), dtype=bool)
        modes = enumerate_modes(c.L, c.W, f_max=200.0)[:N_MODES]
        P = _render(model, renderer, c, rx, T.shape[1], dev)
        m = _metrics(P, T, los, modes)
        m.update(label=c.label, kind=c.kind, n_tokens=c.n_tokens, d_hat=c.d_hat,
                 in_slab=bool(D_HAT_HOLDOUT_FAM[0] <= c.d_hat <= D_HAT_HOLDOUT_FAM[1]))
        per_room.append(m)

        # ---- the token-count control: SAME room, more tokens ----
        for k in REFINE.get(c.kind, ()):
            v = refine_verts(c.verts, k)
            if len(v) > MAX_SEG_POLY:
                continue
            Pk = _render(model, renderer, c, rx, T.shape[1], dev, verts=v)
            mk = _metrics(Pk, T, los, modes)
            mk.update(label=c.label, kind=c.kind, n_tokens=len(v), refine_k=k)
            control.append(mk)
        print("  {:4s} {:2d}tok  R {:+.4f}  LSD {:5.2f}  RIR {:+.4f}".format(
            c.kind, c.n_tokens, m["spatial_pearson"], m["band_lsd_db"],
            m["rir_pearson_modal"]), flush=True)

    def _agg(rs, key):
        v = [r[key] for r in rs if np.isfinite(r.get(key, np.nan))]
        return float(np.mean(v)) if v else float("nan")

    # ---------------------------------------------------------------- the two questions
    print("\n" + "=" * 78)
    print("Q1 -- DOES CORNER COUNT EXTRAPOLATE?  (per family, held-out rooms)")
    print("=" * 78)
    print("{:6s} {:>7s} {:>7s} {:>9s} {:>9s} {:>9s} {:>9s} {:>8s}".format(
        "family", "reflex", "tokens", "spatial R", "in-slab", "out-slab", "LSD dB", "RIR r"))
    by_fam = defaultdict(list)
    for r in per_room:
        by_fam[r["kind"]].append(r)
    n_reflex = {"rect": 0, "L": 1, "U": 2, "T": 2, "DN": 2}
    fam_rows = {}
    for fam in sorted(by_fam, key=lambda f: (n_reflex[f], f)):
        rs = by_fam[fam]
        ins = [r for r in rs if r["in_slab"]]
        outs = [r for r in rs if not r["in_slab"]]
        fam_rows[fam] = {
            "n": len(rs), "n_reflex": n_reflex[fam], "n_tokens": N_TOKENS[fam],
            "spatial_pearson": _agg(rs, "spatial_pearson"),
            "in_slab": _agg(ins, "spatial_pearson"), "out_slab": _agg(outs, "spatial_pearson"),
            "band_lsd_db": _agg(rs, "band_lsd_db"),
            "rir_pearson_modal": _agg(rs, "rir_pearson_modal"),
            "nlos_frac": _agg(rs, "nlos_frac"),
        }
        f = fam_rows[fam]
        print("{:6s} {:>7d} {:>7d} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.2f} {:>8.4f}".format(
            fam, f["n_reflex"], f["n_tokens"], f["spatial_pearson"], f["in_slab"],
            f["out_slab"], f["band_lsd_db"], f["rir_pearson_modal"]))

    print("\nby REFLEX-CORNER COUNT (the Z question):")
    by_ref = defaultdict(list)
    for r in per_room:
        by_ref[n_reflex[r["kind"]]].append(r)
    for k in sorted(by_ref):
        print("  {} corner(s): spatial R {:+.4f}   LSD {:.2f} dB   n = {}".format(
            k, _agg(by_ref[k], "spatial_pearson"), _agg(by_ref[k], "band_lsd_db"),
            len(by_ref[k])))

    print("\n" + "=" * 78)
    print("Q2 -- DOES TOKEN COUNT COST ANYTHING AT FIXED GEOMETRY?")
    print("=" * 78)
    print("Identical rooms, identical physics, collinear re-tokenization. Any difference here is")
    print("token count ALONE -- unlike the across-family comparison above, which confounds it")
    print("with shape.")
    ctrl = {}
    for fam in sorted(REFINE):
        rs = [r for r in control if r["kind"] == fam]
        if not rs:
            continue
        print("  {}:".format(fam))
        for k in sorted({r["refine_k"] for r in rs}):
            g = [r for r in rs if r["refine_k"] == k]
            ctrl["{}_k{}".format(fam, k)] = {
                "n_tokens": g[0]["n_tokens"], "n": len(g),
                "spatial_pearson": _agg(g, "spatial_pearson"),
                "band_lsd_db": _agg(g, "band_lsd_db"),
            }
            base = fam_rows.get(fam, {}).get("spatial_pearson", float("nan"))
            print("     {:2d} tokens (k={})  spatial R {:+.4f}   delta vs native {:+.4f}   "
                  "LSD {:.2f}".format(g[0]["n_tokens"], k, _agg(g, "spatial_pearson"),
                                      _agg(g, "spatial_pearson") - base,
                                      _agg(g, "band_lsd_db")))

    # ---- sigma, reported because P4-3 made it a standing readout ----
    sig = []
    for c in test:
        if c.is_rect:
            continue
        s = probe_sigma_occlusion(model, c.verts, SRC, dev, cond_source="geom_token",
                                  edge_alphas=list(c.edge_alphas), world_scale=10.0)
        if s.get("probed"):
            sig.append(s["solid_over_air"])
    sigma_mean = float(np.mean(sig)) if sig else float("nan")
    print("\nsigma solid/air (clipped ray probe, D74): {:.4f} over {} probeable rooms".format(
        sigma_mean, len(sig)))

    report = {"arm": run.name, "checkpoint": str(ck), "iter": int(it),
              "n_modes": N_MODES, "per_family": fam_rows,
              "by_reflex_count": {str(k): {"spatial_pearson": _agg(v, "spatial_pearson"),
                                           "band_lsd_db": _agg(v, "band_lsd_db"), "n": len(v)}
                                  for k, v in by_ref.items()},
              "token_control": ctrl, "sigma_solid_over_air": sigma_mean,
              "per_room": per_room, "token_control_rooms": control}
    json.dump(report, open(out / "FAMILY_EVAL.json", "w"), indent=1, default=float)
    print("\n-> {}".format(out / "FAMILY_EVAL.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
