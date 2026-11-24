"""P4-10 E-B (D211, D214): capture then edit on R1-M. Fit M2 on 32 measurements of an unseen room, then edit the room with the fitted offset kept.
  --sanity                       GPU: Rule-8 style checks on room 0 (unfitted + zero offset = plain render = cached render; captured-edited != captured-unedited)
  --fit-render --room I          GPU: 3 seeds for unseen room I -> scratch eb/<room>__s<seed>.npz (static render, per-cell edited renders on D and C)
  --score-room --room I          CPU: scores of room I (all seeds + zero-shot + KRR + ignore) -> scratch eb/score/<room>.json
  --aggregate                    CPU: outputs/p4_10/eb/report.{md,json}
  --fig                          CPU: outputs/p4_10/meeting/EB_capture_then_edit.{png,pdf,json}
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import scripts.p4_9_grounding as G
from scripts.p4_10_scorecard import BAND, PRED, TRUTH, band_lsd, band_r, cells, cid, decay_cell, on_lattice, stored

NAME = "R1M_150K"
CK = "outputs/p4_10/arms/R1M/ckpt_iter0150000.pt"
EB = Path("/fs/nexus-scratch/htakawal/aaf_p410/eb")
OUT = Path("outputs/p4_10/eb")
CELLS = (("StretchU", "1.10"), ("Wall", "L+0.3"), ("Material", "all0.30"), ("Material", "long0.50"), ("Source", "third"))
FIGROOM = "C3L6.54_W4.82_NWd0.80w2.82_NEd1.84w1.88_SEd0.60w1.66"
K, SEEDS = 32, (0, 1, 2)
SEED, NB = 20261013, 10000
SRC0 = (0.5, 1.6)
B3 = list(BAND)
OCT = ("63", "125", "250")


def rooms():
    out = [r for r in G.eval_rooms() if r["group"] == "unseen"]
    assert len(out) == 16
    return out


def room_cells(stem):
    return [c for c in cells() if c["group"] == "unseen" and c["room"] == stem and (c["family"], c["mag"]) in CELLS]


def tag(c):
    return "{}|{}".format(c["family"], c["mag"])


def same_ring(c):
    """the edited ring has the original's vertex count and the same edge directions"""
    a, b = np.array(c["orig_verts"], float), np.array(c["verts"], float)
    if len(a) != len(b):
        return False
    da, db_ = np.roll(a, -1, 0) - a, np.roll(b, -1, 0) - b
    return bool(np.array_equal(np.sign(np.round(da, 6)), np.sign(np.round(db_, 6))))


class EBNet(G.Net):
    """Net with the edit's absorption and source as arguments (the parent's are fixed at 0.15 and (0.5, 1.6))"""

    def __init__(self, ck=CK):
        G.MODELS[NAME] = ck
        super().__init__(NAME)
        assert self.model.dc_beta_gain is True, "R1-M must render with the D200 DC gain on"

    def cond(self, verts, L, W, alphas=None):
        from aaf.models.conditioning_2d import build_cond_vector_2d
        return build_cond_vector_2d("geom_token", L, W, list(alphas if alphas is not None else [0.15] * len(verts)),
                                    verts=[tuple(map(float, v)) for v in verts], device=self.dev)

    def render(self, z, L, W, pts, E=None, chunk=16, src=SRC0):
        torch = self.torch
        R = torch.as_tensor(np.asarray(pts), dtype=torch.float32, device=self.dev)
        S = torch.tensor([list(map(float, src))], device=self.dev, dtype=torch.float32)
        lo, hi = torch.zeros(2, device=self.dev), torch.tensor([L, W], device=self.dev, dtype=torch.float32)
        out = []
        for s0 in range(0, len(R), chunk):
            rr = R[s0:s0 + chunk]
            kw = {"tok_override": E.expand(len(rr), -1, -1)} if E is not None else {}
            out.append(self.renderer(self.model, rr, S.expand(len(rr), -1), lo, hi, z_s=z.unsqueeze(0).expand(len(rr), -1), n_out=601, **kw))
        return torch.cat(out)

    def field(self, z, L, W, pts, E=None, src=SRC0, chunk=None):
        with self.torch.no_grad():
            return self.render(z, L, W, pts, E=E, src=src, chunk=chunk or int(os.environ.get("P410_CHUNK", "16"))).cpu().numpy().astype(np.complex64)

    def emb(self, z):
        with self.torch.no_grad():
            return self.model.token_embeddings(z.unsqueeze(0)).detach()


def cell_points(c, base):
    """(points D, points C): off-lattice edited dense receivers and common receivers (Material / Source: the same points)"""
    z = np.load(TRUTH / (cid(c) + ".npz"))
    if c["family"] in ("Source", "Material"):
        p = base["rx"][~on_lattice(base["rx"])]
        return p, p, z
    d = z["rx"]
    return d[~on_lattice(d)], z["common"], z


def atomic(dst, **kw):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / "{}.{}.tmp.npz".format(dst.stem, os.getpid())
    np.savez(tmp, **kw)
    os.replace(tmp, dst)


def edit_code(net, c, E0orig, Efit):
    """(z, E, src) of the edited room with the fitted offset"""
    if not same_ring(c):
        raise RuntimeError("edited ring differs: " + c["id"])
    z = net.cond(c["verts"], c["L"], c["W"], c["edge_alphas"])
    if c["family"] == "Source":
        return z, Efit, c["src"]                                     # geometry and absorption unchanged: the fitted code itself
    E = net.emb(z) + (Efit - E0orig)
    return z, E, c["src"]


def fit_room(i):
    rr = rooms()[i]
    stem = Path(rr["path"]).stem
    r = G.load(rr["path"])
    sp = G.split(r)
    cal = json.loads(Path("outputs/p4_9/step1/calibration.json").read_text())["chosen"]["M2"]
    cs = room_cells(stem)
    assert len(cs) == len(CELLS), (stem, len(cs))
    if all((EB / "{}__s{}.npz".format(stem, s)).exists() for s in SEEDS):
        print("all seeds done:", stem, flush=True)
        return
    base = stored(cs[0])
    assert np.allclose(base["rx"], r["rx"]) and np.allclose(base["H"], r["T"], atol=1e-6)
    net = EBNet()
    off = ~on_lattice(base["rx"])
    for seed in SEEDS:
        dst = EB / "{}__s{}.npz".format(stem, seed)
        if dst.exists():
            print("skip", dst.name, flush=True)
            continue
        t0 = time.time()
        idx = G.draw(r, sp, K, seed)
        zo = net.cond(r["verts"], r["L"], r["W"])
        E0 = net.emb(zo)
        snap = net.snapshot()
        Efit = net.fit(zo, r["L"], r["W"], r["rx"][idx], r["T"][idx], "M2", cal["lr"], cal["lam"], cal["steps"], seed=seed)
        tf = time.time() - t0
        out = {"measured": idx, "off": np.nonzero(off)[0], "fit_s": tf, "Pstat": net.field(zo, r["L"], r["W"], base["rx"][off], E=Efit)}
        for c in cs:
            D, C, _z = cell_points(c, base)
            z, E, src = edit_code(net, c, E0, Efit)
            if c["family"] in ("Source", "Material"):
                out["D_" + tag(c)] = net.field(z, c["L"], c["W"], D, E=E, src=src)
            else:
                f = net.field(z, c["L"], c["W"], np.concatenate([D, C]), E=E, src=src)
                out["D_" + tag(c)], out["C_" + tag(c)] = f[:len(D)], f[len(D):]
        net.restore(snap)
        atomic(dst, **out)
        print("{} s{} fit {:.0f}s total {:.0f}s".format(stem, seed, tf, time.time() - t0), flush=True)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def sanity():
    """room 0, seed 0: (i) zero offset + unfitted heads = plain zero-shot edit render; (ii) = the cached R1-M prediction; (iii) captured edited render differs from the captured unedited one"""
    rr = rooms()[0]
    stem = Path(rr["path"]).stem
    r, net = G.load(rr["path"]), EBNet()
    sp = G.split(r)
    cs = room_cells(stem)
    base = stored(cs[0])
    zo = net.cond(r["verts"], r["L"], r["W"])
    E0 = net.emb(zo)
    res = {}
    for c in cs:
        D, _C, tz = cell_points(c, base)
        sel = np.linspace(0, len(D) - 1, 200).round().astype(int)
        z = net.cond(c["verts"], c["L"], c["W"], c["edge_alphas"])
        plain = net.field(z, c["L"], c["W"], D[sel], src=c["src"])
        zz, E, src = edit_code(net, c, E0, E0)
        a = net.field(zz, c["L"], c["W"], D[sel], E=E, src=src)
        pz = np.load(PRED / NAME / (cid(c) + ".npz"))
        cached = pz["Pd"][~on_lattice(tz["rx"]) if c["family"] not in ("Source", "Material") else ~on_lattice(base["rx"])][sel]
        res[tag(c)] = {"zero_offset_vs_plain": rel(a, plain), "plain_vs_cached": rel(plain, cached), "plain_vs_cached_band25-300": rel(plain[:, 50:], cached[:, 50:]), "edit_changes_render": rel(plain, net.field(zo, r["L"], r["W"], D[sel], src=SRC0)) if c["family"] != "Source" else None}
        print(tag(c), res[tag(c)], flush=True)
    idx = G.draw(r, sp, K, 0)
    snap = net.snapshot()
    Efit = net.fit(zo, r["L"], r["W"], r["rx"][idx], r["T"][idx], "M2", 3e-3, 1e-2, 40, seed=0)
    off = np.nonzero(~on_lattice(base["rx"]))[0][::25]
    stat = net.field(zo, r["L"], r["W"], base["rx"][off], E=Efit)
    for c in cs:
        z, E, src = edit_code(net, c, E0, Efit)
        e = net.field(z, c["L"], c["W"], base["rx"][off], E=E, src=src)
        res[tag(c)]["captured_edited_vs_captured_unedited"] = rel(e, stat)
        print(tag(c), "captured edited vs unedited", res[tag(c)]["captured_edited_vs_captured_unedited"], flush=True)
    net.restore(snap)
    a2 = net.field(zo, r["L"], r["W"], base["rx"][off])
    cached0 = np.load(PRED / NAME / "orig" / (stem + ".npz"))["P"][off]
    res["restored_static_vs_cached"] = rel(a2, cached0)
    print("restored heads: static vs cached", res["restored_static_vs_cached"])
    ok = all(v["zero_offset_vs_plain"] < 1e-6 and v["plain_vs_cached"] < 1e-3 and v["captured_edited_vs_captured_unedited"] > 1e-3 for k, v in res.items() if "|" in k) and res["restored_static_vs_cached"] < 1e-3
    print("SANITY", "PASS" if ok else "FAIL")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sanity.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------- scoring
def dchange(e):
    return e.get("rate_Te", np.nan) - e.get("rate_T0", np.nan), e.get("rate_Pe", np.nan) - e.get("rate_P0", np.nan)


def score_room(i):
    rr = rooms()[i]
    stem = Path(rr["path"]).stem
    r = G.load(rr["path"])
    cs = room_cells(stem)
    base = stored(cs[0])
    cal = json.loads(Path("outputs/p4_9/step1/calibration.json").read_text())["chosen"]
    off = np.nonzero(~on_lattice(base["rx"]))[0]
    zs0 = np.load(PRED / NAME / "orig" / (stem + ".npz"))["P"]
    have = [s for s in SEEDS if (EB / "{}__s{}.npz".format(stem, s)).exists()]
    sf = EB / "score" / (stem + ".json")
    if not have:
        print("MISSING room (no fits):", stem, flush=True)
        return
    if sf.exists() and len(json.loads(sf.read_text())["seeds"]) >= len(have):
        print("already scored with", len(have), "seeds:", stem, flush=True)
        return
    if len(have) < len(SEEDS):
        print("PARTIAL room:", stem, "seeds", have, flush=True)
    res = {"room": stem, "family": rr["family"], "seeds": [], "n_off": int(len(off))}
    for seed in have:
        z = np.load(EB / "{}__s{}.npz".format(stem, seed))
        idx = z["measured"]
        assert np.array_equal(np.sort(idx), np.sort(G.draw(r, G.split(r), K, seed)))
        Pstat = z["Pstat"]
        nd = np.setdiff1d(off, idx)
        pos = {int(v): k for k, v in enumerate(off)}
        ndi = np.array([pos[int(v)] for v in nd])
        row = {"seed": seed, "fit_s": float(z["fit_s"]), "static": {"cap": band_r(Pstat[ndi], base["H"][nd]), "zero": band_r(zs0[nd], base["H"][nd])}, "cells": {}}
        # KRR once per seed: fitted on the 32 original measurements, evaluated at the original off-lattice receivers and the edited dense ones
        cellD = {}
        qs = [base["rx"][off]]
        for c in cs:
            D, C, _ = cell_points(c, base)
            cellD[tag(c)] = (D, C)
            if c["family"] not in ("Source", "Material"):
                qs.append(D)
        Q = np.concatenate(qs)
        KR = G.krr(base["rx"][idx], base["H"][idx], Q, cal["M5"]["lam"])
        o0, cur = KR[:len(off)], len(off)
        krrD = {}
        for c in cs:
            if c["family"] not in ("Source", "Material"):
                n = len(cellD[tag(c)][0])
                krrD[tag(c)] = KR[cur:cur + n]
                cur += n
        for c in cs:
            t = np.load(TRUTH / (cid(c) + ".npz"))
            pz = np.load(PRED / NAME / (cid(c) + ".npz"))
            ci = t["common_idx"] if c["family"] not in ("Source", "Material") else off
            keep = ~np.isin(ci, idx)
            if c["family"] in ("Source", "Material"):
                Te = t["Hd"][ci][keep]
                zsC = pz["Pd"][ci][keep]
                capC = z["D_" + tag(c)][keep]
            else:
                Te, zsC, capC = t["Hc"][keep], pz["Pc"][keep], z["C_" + tag(c)][keep]
            T0 = base["H"][ci][keep]
            capU = Pstat[np.array([pos[int(v)] for v in ci])][keep]
            zsU = zs0[ci][keep]
            kr = (o0[np.array([pos[int(v)] for v in ci])])[keep]
            e = {"n_C": int(keep.sum()), "C": {"zero_edit": band_r(zsC, Te), "cap_edit": band_r(capC, Te), "cap_unedited": band_r(capU, Te), "krr": band_r(kr, Te), "ignore": band_r(T0, Te), "zero_unedited": band_r(zsU, Te)},
                 "g_delta": {"cap": {b: band_lsd(capU, Te)[b] / max(band_lsd(capC, Te)[b], 1e-9) for b in BAND}, "zero": {b: band_lsd(zsU, Te)[b] / max(band_lsd(zsC, Te)[b], 1e-9) for b in BAND}}}
            if c["family"] in ("Source", "Material"):
                e["D"] = {"zero_edit": e["C"]["zero_edit"], "cap_edit": e["C"]["cap_edit"]}
            else:
                D, _ = cellD[tag(c)]
                sel = ~on_lattice(t["rx"])
                TD = t["Hd"][sel]
                e["D"] = {"zero_edit": band_r(pz["Pd"][sel], TD), "cap_edit": band_r(z["D_" + tag(c)], TD), "krr": band_r(krrD[tag(c)], TD), "n": int(sel.sum())}
            if c["family"] == "Material":
                sub = lambda A: A[ci][keep]  # noqa: E731
                bs = {"rx": base["rx"][ci][keep], "H": base["H"][ci][keep]}
                d_cap = decay_cell(c, bs, {"P": capU}, {"Hd": Te}, {"Pd": capC})
                d_zs = decay_cell(c, bs, {"P": zsU}, {"Hd": Te}, {"Pd": zsC})
                e["decay"] = {m: {b: {"true": dchange(d[b])[0], "pred": dchange(d[b])[1], "edt_err_edit": d[b].get("edt_err_edit", np.nan), "edt_err_orig": d[b].get("edt_err_orig", np.nan), "n_valid": d[b]["n_valid"]} for b in OCT} for m, d in (("cap", d_cap), ("zero", d_zs))}
            row["cells"][tag(c)] = e
        res["seeds"].append(row)
        print(stem, seed, "scored", flush=True)
    (EB / "score").mkdir(parents=True, exist_ok=True)
    tmp = EB / "score" / ("{}.{}.tmp".format(stem, os.getpid()))
    tmp.write_text(json.dumps(res))
    os.replace(tmp, EB / "score" / (stem + ".json"))


def slope(pairs):
    a = np.array([p for p in pairs if np.isfinite(p[0]) and np.isfinite(p[1])], float)
    if len(a) < 3:
        return {"slope": None, "pearson": None, "n": int(len(a))}
    A = np.polyfit(a[:, 0], a[:, 1], 1)
    return {"slope": float(A[0]), "pearson": float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), "n": int(len(a))}


def ratio_ci(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    ix = np.random.default_rng(SEED).integers(0, len(num), (NB, len(num)))
    nm, dm = num[ix].mean(1), den[ix].mean(1)
    d_ci = [float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))]
    if d_ci[0] <= 0 <= d_ci[1]:
        return {"value": None, "ci": None, "denom": float(den.mean()), "denom_ci": d_ci, "num": float(num.mean()), "undefined": True}
    q = nm / dm
    return {"value": float(num.mean() / den.mean()), "ci": [float(np.percentile(q, 2.5)), float(np.percentile(q, 97.5))], "denom": float(den.mean()), "denom_ci": d_ci, "num": float(num.mean()), "undefined": False}


def aggregate():
    stems = [Path(x["path"]).stem for x in rooms()]
    R = [json.loads((EB / "score" / (s + ".json")).read_text()) for s in stems if (EB / "score" / (s + ".json")).exists()]
    miss = [s for s in stems if not (EB / "score" / (s + ".json")).exists()]
    part = {r["room"]: len(r["seeds"]) for r in R if len(r["seeds"]) < len(SEEDS)}
    print("rooms scored", len(R), "of", len(stems), "missing", miss, "partial", part, flush=True)
    if len(R) < 8:
        raise SystemExit("fewer than 8 rooms: no report")
    rep = {"missing_rooms": miss, "partial_rooms": part, "n_rooms": len(R), "K": K, "seeds": list(SEEDS), "families": {}, "static": {}, "decay": {}, "rooms": [r["room"] for r in R]}
    smean = lambda fn: [float(np.mean([fn(s) for s in r["seeds"]])) for r in R]  # noqa: E731
    for b in B3:
        cap, zs = smean(lambda s: s["static"]["cap"][b]), smean(lambda s: s["static"]["zero"][b])
        rep["static"][b] = {"captured": float(np.mean(cap)), "zero_shot": float(np.mean(zs)), "gain": float(np.mean(np.array(cap) - np.array(zs)))}
    boot = lambda v: [float(np.mean(v))] + [float(np.percentile(np.asarray(v)[np.random.default_rng(SEED).integers(0, len(v), (NB, len(v)))].mean(1), q)) for q in (2.5, 97.5)]  # noqa: E731
    L = ["# P4-10 E-B: capture then edit (R1-M, M2, K = 32, 3 seeds, {} of 16 unseen rooms)".format(len(R)), "", "Missing rooms: {}. Rooms with fewer than 3 seeds: {}.".format(miss or "none", part or "none"), "", "Mean over rooms (each room = mean over 3 seeds). C = common receivers, D = the edited room's own dense off-lattice receivers. The 32 measured receivers are excluded from C (and from D when D = C) for every method.", "",
         "Captured static r (original room, non-drawn off-lattice): " + " / ".join("{:.3f}".format(rep["static"][b]["captured"]) for b in B3) + "; zero-shot static: " + " / ".join("{:.3f}".format(rep["static"][b]["zero_shot"]) for b in B3) + " (25-100 / 100-200 / 200-300 Hz).", ""]
    for fam, mag in CELLS:
        k = "{}|{}".format(fam, mag)
        e = {"n_rooms": len(R), "bands": {}, "g_delta": {}}
        for b in B3:
            q = {m: boot(smean(lambda s: s["cells"][k]["C"][m][b])) for m in ("zero_edit", "cap_edit", "cap_unedited", "krr", "ignore", "zero_unedited")}
            dz, dc = smean(lambda s: s["cells"][k]["D"]["zero_edit"][b]), smean(lambda s: s["cells"][k]["D"]["cap_edit"][b])
            sc, sz = smean(lambda s: s["static"]["cap"][b]), smean(lambda s: s["static"]["zero"][b])
            surv = ratio_ci(np.array(dc) - np.array(dz), np.array(sc) - np.array(sz))
            cz, ce, cu = (np.array(smean(lambda s: s["cells"][k]["C"][m][b])) for m in ("zero_edit", "cap_edit", "cap_unedited"))
            den = np.array(sc) - np.array(sz)
            e["bands"][b] = {"C": q, "D_zero_edit": boot(dz), "D_cap_edit": boot(dc), "survival": surv,
                             "survival_on_C": ratio_ci(ce - cz, den), "not_edited_on_C": ratio_ci(cu - cz, den), "cap_edit_minus_cap_unedited_C": boot(ce - cu), "cap_edit_minus_krr_C": boot(ce - np.array(smean(lambda s: s["cells"][k]["C"]["krr"][b])))}
        for m in ("cap", "zero"):
            e["g_delta"][m] = {b: float(np.mean(smean(lambda s: s["cells"][k]["g_delta"][m][b]))) for b in B3}
        rep["families"][k] = e
        L += ["## {} ({} rooms)".format(k, len(R)), "", "| field (C) | " + " | ".join("{} Hz".format(b) for b in B3) + " |", "|---|---|---|---|"]
        for m, lab in (("zero_edit", "zero-shot edit (R1-M, no capture)"), ("cap_edit", "captured and edited"), ("cap_unedited", "captured, not edited"), ("krr", "KRR on the same 32 microphones"), ("ignore", "ignore the edit (original truth)")):
            L.append("| {} | {} |".format(lab, " | ".join("{:.3f} [{:.3f}, {:.3f}]".format(*e["bands"][b]["C"][m]) for b in B3)))
        L.append("| captured and edited, on D | {} |".format(" | ".join("{:.3f}".format(e["bands"][b]["D_cap_edit"][0]) for b in B3)))
        L.append("| zero-shot edit, on D | {} |".format(" | ".join("{:.3f}".format(e["bands"][b]["D_zero_edit"][0]) for b in B3)))

        def sv(b):
            s = e["bands"][b]["survival"]
            return "undefined (denominator CI includes 0)" if s["undefined"] else "{:.2f} [{:.2f}, {:.2f}]".format(s["value"], *s["ci"])
        def sn(b, key):
            s = e["bands"][b][key]
            return "undefined" if s["undefined"] else "{:.2f} [{:.2f}, {:.2f}]".format(s["value"], *s["ci"])
        L += ["", "Survival (D): " + " / ".join(sv(b) for b in B3) + " (25-100 / 100-200 / 200-300 Hz)",
              "Same scale on C: captured and edited " + " / ".join(sn(b, "survival_on_C") for b in B3) + "; captured and NOT edited (post-hoc reading aid, not pre-registered) " + " / ".join(sn(b, "not_edited_on_C") for b in B3),
              "Captured and edited minus captured and not edited (C): " + " / ".join("{:+.3f} [{:+.3f}, {:+.3f}]".format(*e["bands"][b]["cap_edit_minus_cap_unedited_C"]) for b in B3) + "; minus KRR: " + " / ".join("{:+.3f} [{:+.3f}, {:+.3f}]".format(*e["bands"][b]["cap_edit_minus_krr_C"]) for b in B3),
              "g_delta captured {} ; zero-shot {}".format(" / ".join("{:.2f}".format(e["g_delta"]["cap"][b]) for b in B3), " / ".join("{:.2f}".format(e["g_delta"]["zero"][b]) for b in B3)), ""]
    # CE1
    rep["CE1"] = {}
    L += ["## CE1 (survival >= 0.5 at 25-100 Hz, each family on its own)", ""]
    for fam, mag in (("StretchU", "1.10"), ("Wall", "L+0.3"), ("Material", "all0.30")):
        s = rep["families"]["{}|{}".format(fam, mag)]["bands"]["25-100"]["survival"]
        rep["CE1"]["{}|{}".format(fam, mag)] = {"survival": s["value"], "ci": s["ci"], "holds": bool(s["value"] is not None and s["value"] >= 0.5)}
        L.append("- {} {}: {} -> {}".format(fam, mag, "undefined" if s["undefined"] else "{:.2f} [{:.2f}, {:.2f}]".format(s["value"], *s["ci"]), "HOLDS" if rep["CE1"]["{}|{}".format(fam, mag)]["holds"] else "FAILS"))
    # decay
    for tagk in ("cap", "zero"):
        pairs, perb = [], {b: [] for b in OCT}
        errs = {"edit": [], "orig": []}
        for r in R:
            for mag in ("all0.30", "long0.50"):
                k = "Material|" + mag
                for b in OCT:
                    tr = np.mean([s["cells"][k]["decay"][tagk][b]["true"] for s in r["seeds"]])
                    pr = np.mean([s["cells"][k]["decay"][tagk][b]["pred"] for s in r["seeds"]])
                    pairs.append((tr, pr))
                    perb[b].append((tr, pr))
                    errs["edit"].append(np.mean([s["cells"][k]["decay"][tagk][b]["edt_err_edit"] for s in r["seeds"]]))
                    errs["orig"].append(np.mean([s["cells"][k]["decay"][tagk][b]["edt_err_orig"] for s in r["seeds"]]))
        rep["decay"][tagk] = {"pooled": slope(pairs), "per_octave": {b: slope(v) for b, v in perb.items()}, "edt_err_edit_med": float(np.nanmedian(errs["edit"])), "edt_err_orig_med": float(np.nanmedian(errs["orig"]))}
    L += ["", "## Change in 1/EDT, predicted on true (Material all0.30 and long0.50; 3 octaves x 32 cells)", ""]
    for m, lab in (("zero", "zero-shot"), ("cap", "captured")):
        p = rep["decay"][m]["pooled"]
        L.append("- {}: slope {:.2f}, Pearson {:.2f}, n {} cell-bands; edited |EDT error| {:.1%} vs unedited {:.1%}".format(lab, p["slope"], p["pearson"], p["n"], rep["decay"][m]["edt_err_edit_med"], rep["decay"][m]["edt_err_orig_med"]))
    OUT.mkdir(parents=True, exist_ok=True)
    for nm, txt in (("report.json", json.dumps(rep, indent=1)), ("report.md", "\n".join(L) + "\n")):
        tmp = OUT / (nm + ".{}.tmp".format(os.getpid()))
        tmp.write_text(txt)
        os.replace(tmp, OUT / nm)
    print("\n".join(L))


# ------------------------------------------------------------------------------------------- figure
def edc_db(h, fs, gate_s=1.5):
    n = int(gate_s * fs)
    on = np.argmax(np.abs(h) >= 0.1 * np.abs(h[:, : int(0.4 * fs)]).max(1, keepdims=True), axis=1)
    out = np.zeros((len(h), n))
    for r in range(len(h)):
        seg = h[r, on[r]:on[r] + n]
        out[r, : len(seg)] = seg
    e = np.cumsum(out[:, ::-1] ** 2, axis=1)[:, ::-1]
    return 10 * np.log10(np.maximum(e / np.maximum(e[:, :1], 1e-300), 1e-9))


def fig():
    from aaf.eval.octave_decay import FS, octave_ir
    from scripts.p4_10_md_style import COL, new, plt, save
    rep = json.loads((OUT / "report.json").read_text())
    fams = [("StretchU|1.10", "Stretch\n(uniform 1.10)"), ("Wall|L+0.3", "Wall moved\n(+0.3 m)"), ("Material|all0.30", "All walls\nabsorb 0.30"), ("Material|long0.50", "Longest wall\nabsorb 0.50"), ("Source|third", "Source\nmoved")]
    meths = [("zero_edit", "Zero-shot edit (no capture)", COL["R1-M"]), ("cap_edit", "Captured, then edited", "#1baf7a"), ("cap_unedited", "Captured, not edited", "#9a9a9a"), ("krr", "Interpolating the 32 microphones", COL["KRR"])]
    fig_ = new(16, 9)
    gs = fig_.add_gridspec(2, 2, width_ratios=[1.9, 1], wspace=0.24, hspace=0.55, left=0.07, right=0.985, top=0.81, bottom=0.2)
    meta = {"n_rooms": rep["n_rooms"], "K": rep["K"], "source": "outputs/p4_10/eb/report.json", "values": {}, "receivers": "the common receivers C, 32 measured receivers excluded for every method"}
    for j, b in enumerate(("25-100", "100-200")):
        ax = fig_.add_subplot(gs[j, 0])
        ax.grid(True, axis="y", color="#e8e8e8")
        ax.set_axisbelow(True)
        for m_i, (m, lab, col) in enumerate(meths):
            v = [rep["families"][k]["bands"][b]["C"][m][0] for k, _ in fams]
            xs = [i + (m_i - 1.5) * 0.2 for i in range(len(fams))]
            ax.bar(xs, v, 0.185, color=col, label=lab if j == 0 else None)
            meta["values"]["{}|{}".format(b, lab)] = dict(zip([k for k, _ in fams], v))
            for x0, t in zip(xs, v):
                if t > 0.22:
                    ax.text(x0, t - 0.025, "{:.2f}".format(t), ha="center", va="top", rotation=90, fontsize=14, color="white", zorder=7)
                else:
                    ax.text(x0, t + 0.02, "{:.2f}".format(t), ha="center", va="bottom", rotation=90, fontsize=14, color="#333333", zorder=7)
        ig = [rep["families"][k]["bands"][b]["C"]["ignore"][0] for k, _ in fams]
        ax.hlines(ig, [i - 0.43 for i in range(len(fams))], [i + 0.43 for i in range(len(fams))], color="#b00020", lw=3.4, zorder=6)
        meta["values"]["{}|ignore".format(b)] = dict(zip([k for k, _ in fams], ig))
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels([l for _, l in fams], fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.55, len(fams) - 0.45)
        ax.set_ylabel("accuracy after the edit")
        ax.set_title("{} Hz".format(b), loc="left", fontsize=17)
    from matplotlib.lines import Line2D
    h, l = fig_.axes[0].get_legend_handles_labels()
    h.append(Line2D([0], [0], color="#b00020", lw=3.4))
    l.append("Ignoring the edit (the true original room)")
    fig_.legend(h, l, loc="upper left", ncol=3, frameon=False, fontsize=14, bbox_to_anchor=(0.06, 0.935), columnspacing=1.4, handlelength=1.6)
    # decay panel
    R = rooms()
    i = next(k for k, x in enumerate(R) if Path(x["path"]).stem == FIGROOM)
    r = G.load(R[i]["path"])
    c = next(x for x in room_cells(FIGROOM) if x["family"] == "Material" and x["mag"] == "all0.30")
    base, t = stored(c), np.load(TRUTH / (cid(c) + ".npz"))
    zs = np.load(PRED / NAME / (cid(c) + ".npz"))
    zo = np.load(PRED / NAME / "orig" / (FIGROOM + ".npz"))
    off = np.nonzero(~on_lattice(base["rx"]))[0]
    sc = [np.load(EB / "{}__s{}.npz".format(FIGROOM, s)) for s in SEEDS if (EB / "{}__s{}.npz".format(FIGROOM, s)).exists()]
    far = np.hypot(base["rx"][off][:, 0] - 0.5, base["rx"][off][:, 1] - 1.6) >= 1.0
    for s in sc:
        far &= ~np.isin(off, s["measured"])
    pos = off[far]
    dk = np.nonzero(far)[0]
    curves = {"truth, before": (base["H"][pos], "#9a9a9a", (0, (5, 3))), "truth, after": (t["Hd"][pos], "#111111", "-"),
              "zero-shot edit": (zs["Pd"][pos], COL["R1-M"], "-"), "captured, then edited": (np.mean([s["D_Material|all0.30"][dk] for s in sc[:1]], 0), "#1baf7a", "-")}
    ax = fig_.add_subplot(gs[:, 1])
    tt = np.arange(int(1.5 * FS)) / FS
    meta["decay"] = {"room": FIGROOM, "cell": c["id"], "octave_hz": 125, "n_receivers": int(len(pos)), "seed": 0}
    for lab, (H, col, ls) in curves.items():
        y = np.median(edc_db(octave_ir(H, "125"), FS), 0)
        ax.plot(tt, y, color=col, ls=ls, lw=2.8, label=lab)
        meta["decay"][lab] = {"dB at 0.1 / 0.25 / 0.5 / 0.75 s": [float(y[int(x * FS)]) for x in (0.1, 0.25, 0.5, 0.75)]}
    ax.set_ylim(-45, 2)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("time after the sound arrives (s)")
    ax.set_ylabel("energy decay (dB)")
    ax.grid(True, color="#e8e8e8")
    ax.set_title("All walls to absorption 0.30,\n125 Hz octave, one new room", loc="left", fontsize=15)
    ax.legend(frameon=False, fontsize=14, loc="upper right")
    fig_.suptitle("Measure a new room with 32 microphones, then edit it", fontsize=23, x=0.06, ha="left", y=0.985)
    fig_.text(0.06, 0.018, "{} new rooms, {} microphone draws each. Accuracy = match of the sound-level pattern with the true edited room (off-lattice receivers;\nthe 32 microphones removed for every method). Decay panel: {}, median over {} receivers, first draw; the lower of the\ntwo middle rooms by R1's accuracy.".format(rep["n_rooms"], len(SEEDS) if not rep["partial_rooms"] else "up to {}".format(len(SEEDS)), FIGROOM.split("_NW")[0], len(pos)), fontsize=14, color="#444444")
    import tempfile

    import scripts.p4_10_md_style as ST
    final, tmpd = ST.OUT, Path(tempfile.mkdtemp(dir=str(EB)))
    ST.OUT = tmpd
    save(fig_, "EB_capture_then_edit", meta)
    ST.OUT = final
    final.mkdir(parents=True, exist_ok=True)
    import shutil
    for f in tmpd.iterdir():
        t = final / ("." + f.name + ".tmp")
        shutil.copyfile(f, t)
        os.replace(t, final / f.name)
    shutil.rmtree(tmpd)
    print("figure files in", final)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("--sanity", "--fit-render", "--score-room", "--aggregate", "--fig"):
        ap.add_argument(k, action="store_true")
    ap.add_argument("--room", type=int, default=0)
    a = ap.parse_args()
    if a.sanity:
        sanity()
    elif a.fit_render:
        fit_room(a.room)
    elif a.score_room:
        score_room(a.room)
    elif a.aggregate:
        aggregate()
    elif a.fig:
        fig()
