"""P4-9 X1 (D189): the network as a PRIOR for sparse-measurement reconstruction -- the zero-shot field as the mean of
Step 1's geometry-free Helmholtz kernel ridge regression (M5).

  M6   p(x) = m(x) + a G0(x, x0) + sum_i alpha_i J0(k |x - x_i|)       (M5 fitted to the residual y - m(x_i))
  M6s  p(x) = c m(x) + a G0(x, x0) + sum_i alpha_i J0(k |x - x_i|)     (c, a unpenalised by GLS; alpha ridge)
m = the frozen network's zero-shot field (R0 @120K, S2 @120K) from the standard caches, checked against Step 1's M0 fits.
Rooms, strata, draws, K, seeds and the scorer are Step 1's (`scripts/p4_9_grounding.py`); clean condition only.
  python scripts/p4_9_netprior.py --calibrate     lam per method on the 4 T / DN rooms (E7) -> outputs/p4_9/netprior/
  python scripts/p4_9_netprior.py --room I        every M6 / M6s fit of reported room I (0-30)
  python scripts/p4_9_netprior.py --aggregate     -> outputs/p4_9/netprior/netprior.{json,md}
X1b (D192), the prior under PARTIAL geometry (bounding rectangle for both, A1):
  M6-iii  M6s with the prior = R0's zero-shot field given bounding-rectangle tokens (Step 1's M0-iii)
  M7-iii  M6s with the prior = m1 on the bounding rectangle, one beta scale fitted to the K measurements (Step 1's M4-iii)
  python scripts/p4_9_netprior.py --render-iii     GPU: R0's M0-iii field at every dense receiver of the 31 rooms
  python scripts/p4_9_netprior.py --room-iii I     CPU: every M6-iii / M7-iii fit of room I (M6s's calibrated lam)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts.p4_9_grounding import (BANDS, C0, DF, FITS, KS, M5_LAMS, SEEDS, SRC, _boot, calib_rooms, draw,
                                    eval_rooms, load, score_sets, scores_of, split)

OUT = Path("outputs/p4_9/netprior")
NFITS = OUT / "fits"
MODELS = ("R0_120K", "S2_120K")
METHODS = ("M6", "M6s")
G2 = "outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz"
UN = "outputs/p4_8/cache/unseen_{}.npz"
FD = "outputs/p4_6/cache/famdense_{}.npz"


def net_field(name, rr, r):
    """The network's zero-shot field at every dense receiver of room r, from the standard cache (receivers asserted)."""
    if rr["group"] == "calib":
        z = np.load(FD.format(name))
        hits = [int(k[2:]) for k in z.files if k.startswith("rx") and z[k].shape == r["rx"].shape
                and np.allclose(z[k], r["rx"])]
        assert len(hits) == 1, (r["name"], hits)
        i = hits[0]
    else:
        z = np.load((G2 if rr["group"] == "test" else UN).format(name))
        i = rr["cache_i"]
        assert np.allclose(z["rx%d" % i], r["rx"]), r["name"]
    return np.asarray(z["P%d" % i])[:, :601].astype(np.complex128)


def krr_gls(P, Y, Q, lam, Mp=None, Mq=None, src=SRC):
    """Per bin: p = sum_j beta_j g_j + sum_i alpha_i J0(k |x - x_i|), g = [G0] (+ [m] when Mp is given), beta
    unpenalised (generalised least squares under the ridge), alpha ridge-penalised. With g = [G0] this is M5's `krr`."""
    from scipy.special import hankel2, j0
    out = np.zeros((len(Q), Y.shape[1]), np.complex128)
    dPP = np.hypot(*(P[:, None] - P[None]).transpose(2, 0, 1))
    dQP = np.hypot(*(Q[:, None] - P[None]).transpose(2, 0, 1))
    rP = np.maximum(np.hypot(*(P - src).T), 1e-3)
    rQ = np.maximum(np.hypot(*(Q - src).T), 1e-3)
    out[:, 0] = Y[:, 0].mean() if Mp is None else Mq[:, 0] * (Y[:, 0] @ np.conj(Mp[:, 0])) / max(
        float(np.vdot(Mp[:, 0], Mp[:, 0]).real), 1e-30)
    for b in range(1, Y.shape[1]):
        k = 2 * np.pi * b * DF / C0
        A = j0(k * dPP) + lam * np.eye(len(P))
        G = [-0.25j * hankel2(0, k * rP)]
        Gq = [-0.25j * hankel2(0, k * rQ)]
        if Mp is not None:
            G.append(Mp[:, b])
            Gq.append(Mq[:, b])
        G, Gq = np.stack(G, 1), np.stack(Gq, 1)
        y = Y[:, b]
        Ai_y = np.linalg.solve(A, y)
        Ai_G = np.linalg.solve(A, G)
        beta = np.linalg.lstsq(np.conj(G.T) @ Ai_G, np.conj(G.T) @ Ai_y, rcond=None)[0]
        alpha = Ai_y - Ai_G @ beta
        out[:, b] = Gq @ beta + j0(k * dQP) @ alpha
    return out


def fit(method, m, rx, idx, Y, pts, lam):
    P, Q = rx[idx], rx[pts]
    if method == "M6":
        return m[pts] + krr_gls(P, Y - m[idx], Q, lam)
    if method == "M6s":
        return krr_gls(P, Y, Q, lam, Mp=m[idx], Mq=m[pts])
    raise ValueError(method)


def _m0_check(name, r, sp, m):
    """the cached field must reproduce Step 1's M0 fit (pooled strata) within 2e-3 per band."""
    ref = json.loads((FITS / "{}__M0__{}__K32__s0__clean.json".format(name, r["stem"])).read_text())["scores"]
    sets = score_sets(r, sp, draw(r, sp, 32, 0), headline=False)
    got = scores_of(m, r, {"pooled": sets["pooled"]})["pooled"]
    dev = max(abs(got[b] - ref["pooled"][b]) for b in BANDS)
    assert dev < 2e-3, (name, r["name"], dev, got, ref["pooled"])
    return dev


def render_iii():
    """R0's zero-shot field with bounding-rectangle tokens (Step 1's M0-iii), all dense receivers, 31 rooms."""
    import torch

    from scripts.p4_9_grounding import Net
    net = Net("R0_120K")
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    for rr in eval_rooms():
        r = load(rr["path"])
        dst = OUT / "cache" / "R0_120K_iii__{}.npz".format(r["stem"])
        if dst.exists():
            continue
        z = net.cond([(0.0, 0.0), (r["L"], 0.0), (r["L"], r["W"]), (0.0, r["W"])], r["L"], r["W"])
        with torch.no_grad():
            P = net.render(z, r["L"], r["W"], r["rx"]).cpu().numpy()
        sp = split(r)
        ref = json.loads((FITS / "R0_120K__M0-iii__{}__K32__s0__clean.json".format(r["stem"])).read_text())["scores"]
        got = scores_of(P.astype(np.complex128), r, {"pooled": score_sets(r, sp, draw(r, sp, 32, 0), False)["pooled"]})
        dev = max(abs(got["pooled"][b] - ref["pooled"][b]) for b in BANDS)
        assert dev < 2e-3, (r["name"], dev)
        np.savez_compressed(dst, P=P.astype(np.complex64), rx=r["rx"], m0_check_dev=dev)
        print(dst.name, "M0-iii check dev {:.1e}".format(dev), flush=True)


def run_room_iii(ri):
    from scripts.p4_9_grounding import Modal, beta_fit
    lam = json.loads((OUT / "calibration.json").read_text())["chosen"]["M6s"]["lam"]
    rr = eval_rooms()[ri]
    r = load(rr["path"])
    sp = split(r)
    z = np.load(OUT / "cache" / "R0_120K_iii__{}.npz".format(r["stem"]))
    assert np.allclose(z["rx"], r["rx"])
    m_net = z["P"].astype(np.complex128)
    mod = Modal(r["stem"], "bbox")
    NFITS.mkdir(parents=True, exist_ok=True)
    for K in KS:
        for seed in SEEDS:
            idx = draw(r, sp, K, seed)
            sets = score_sets(r, sp, idx, headline=(K == 32))
            pts = np.unique(np.concatenate(list(sets.values())))
            Y = r["T"][idx]
            for meth in ("M6-iii", "M7-iii"):
                name = "R0_120K" if meth == "M6-iii" else "physics"
                dst = NFITS / "{}__{}__{}__K{}__s{}__clean.json".format(name, meth, r["stem"], K, seed)
                if dst.exists():
                    continue
                t0 = time.time()
                extra = {}
                if meth == "M6-iii":
                    m = m_net
                else:
                    s_beta = beta_fit(mod, mod.z["psi_r"][idx], Y)
                    m = np.zeros_like(m_net)
                    rows = np.unique(np.concatenate([idx, pts]))
                    m[rows] = mod.field(mod.z["psi_r"][rows], s_beta)
                    extra = {"beta_scale": s_beta}
                F = np.zeros_like(m_net)
                F[pts] = krr_gls(r["rx"][idx], Y, r["rx"][pts], lam, Mp=m[idx], Mq=m[pts])
                dt = time.time() - t0
                out = {"scores": scores_of(F, r, sets), "fit_s": dt, "model": name, "method": meth,
                       "room": r["name"], "group": rr["group"], "family": rr["family"], "K": K, "seed": seed,
                       "cond": "clean", "lam": lam, **extra}
                dst.write_text(json.dumps(out))
                print(dst.name, round(out["scores"]["pooled"]["25-100"], 3), round(dt, 2), flush=True)


def calibrate():
    res = {"rooms": [], "grid": {}}
    for meth in METHODS:
        res["grid"][meth] = {}
        for lam in M5_LAMS:
            v = []
            for rr in calib_rooms():
                r = load(rr["path"])
                sp = split(r)
                m = net_field("R0_120K", rr, r)
                idx = draw(r, sp, 32, 0)
                sets = score_sets(r, sp, idx, headline=False)
                pts = sets["pooled"]
                F = np.zeros_like(m)
                F[pts] = fit(meth, m, r["rx"], idx, r["T"][idx], pts, lam)
                s = scores_of(F, r, {"pooled": pts})["pooled"]
                v.append(np.mean([s[b] for b in BANDS]))
            res["grid"][meth][str(lam)] = float(np.mean(v))
            print(meth, lam, round(float(np.mean(v)), 4), flush=True)
    # the zero-shot prior and M5 on the same rooms / draws, for reference
    ref = {}
    for rr in calib_rooms():
        r = load(rr["path"])
        sp = split(r)
        m = net_field("R0_120K", rr, r)
        pts = score_sets(r, sp, draw(r, sp, 32, 0), headline=False)["pooled"]
        s = scores_of(m, r, {"pooled": pts})["pooled"]
        ref[r["name"]] = {"M0_mean_r": float(np.mean([s[b] for b in BANDS]))}
        res["rooms"].append(r["name"])
    res["M0_reference"] = ref
    res["chosen"] = {meth: {"lam": float(max(M5_LAMS, key=lambda lam: res["grid"][meth][str(lam)]))}
                     for meth in METHODS}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["chosen"]), flush=True)


def run_room(ri):
    cal = json.loads((OUT / "calibration.json").read_text())["chosen"]
    rr = eval_rooms()[ri]
    r = load(rr["path"])
    sp = split(r)
    NFITS.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        m = net_field(name, rr, r)
        dev = _m0_check(name, r, sp, m)
        for meth in METHODS:
            for K in KS:
                for seed in SEEDS:
                    dst = NFITS / "{}__{}__{}__K{}__s{}__clean.json".format(name, meth, r["stem"], K, seed)
                    if dst.exists():
                        continue
                    idx = draw(r, sp, K, seed)
                    sets = score_sets(r, sp, idx, headline=(K == 32))
                    pts = np.unique(np.concatenate(list(sets.values())))
                    t0 = time.time()
                    F = np.zeros_like(m)
                    F[pts] = fit(meth, m, r["rx"], idx, r["T"][idx], pts, cal[meth]["lam"])
                    dt = time.time() - t0
                    out = {"scores": scores_of(F, r, sets), "fit_s": dt, "model": name, "method": meth,
                           "room": r["name"], "group": rr["group"], "family": rr["family"], "K": K, "seed": seed,
                           "cond": "clean", "lam": cal[meth]["lam"], "m0_check_dev": dev}
                    dst.write_text(json.dumps(out))
                    print(dst.name, round(out["scores"]["pooled"]["25-100"], 3), round(dt, 2), flush=True)


def aggregate():
    fits = [json.loads(p.read_text()) for p in sorted(NFITS.glob("*.json"))]
    fits += [json.loads(p.read_text()) for p in sorted(FITS.glob("*.json"))
             if p.name.split("__")[1] in ("M0", "M2", "M5", "M3", "M0-iii", "M4-iii") and p.name.endswith("__clean.json")]
    acc, groups = {}, {}
    for f in fits:
        groups[f["room"]] = f["group"]
        for st, sc in f["scores"].items():
            for b in BANDS:
                acc.setdefault((f["model"], f["method"], f["K"], st, b), {}).setdefault(f["room"], []).append(sc[b])
    per_room = {k: {r: float(np.mean(v)) for r, v in d.items()} for k, d in acc.items()}
    SETS = {"unseen16": "unseen", "in_family15": "test"}

    def rooms(key, g):
        return {r: x for r, x in per_room.get(key, {}).items() if groups[r] == g}

    def mean(key, g):
        d = rooms(key, g)
        return float(np.mean(list(d.values()))) if d else float("nan")

    def paired(da, db):
        common = sorted(set(da) & set(db))
        if not common:
            return [float("nan")] * 3, 0
        return _boot([da[r] - db[r] for r in common]), len(common)

    def m0(model, g, b, st="pooled"):          # M0 is scored once (K = 32 seed 0 strata): K-independent
        return rooms((model, "M0", 32, st, b), g)

    rep = {"definition": __doc__.split("\n\n")[0].strip(), "n_fits": len(fits), "readings": {}, "curves": {}}
    for model in MODELS:
        for sname, g in SETS.items():
            cur = {}
            for meth, mdl in (("M0", model), ("M5", "physics"), ("M2", model), ("M6", model), ("M6s", model)):
                cur[meth] = {}
                for K in KS:
                    cur[meth][str(K)] = {b: (float(np.mean(list(m0(model, g, b).values()))) if meth == "M0"
                                             else mean((mdl, meth, K, "pooled", b), g)) for b in BANDS}
            rep["curves"]["{}|{}".format(model, sname)] = cur
            # R2: M6s - better of (M0, M5) per band, every K
            r2 = {}
            for K in KS:
                bands = {}
                for b in BANDS:
                    a0, a5 = m0(model, g, b), rooms(("physics", "M5", K, "pooled", b), g)
                    best = a0 if np.mean(list(a0.values())) >= np.mean(list(a5.values())) else a5
                    ci, n = paired(rooms((model, "M6s", K, "pooled", b), g), best)
                    bands[b] = {"bar": "M0" if best is a0 else "M5", "delta": ci, "n": n}
                r2[str(K)] = {"bands": bands, "wins": int(sum(bands[b]["delta"][1] > 0 for b in BANDS))}
            rep["readings"]["R2|{}|{}".format(model, sname)] = {"by_K": r2, "holds": bool(all(
                r2[k]["wins"] >= 2 for k in r2))}
            # K-equivalent of M6s @ K = 8 on M5's log-K curve
            keq = {}
            for b in BANDS:
                ys = [cur["M5"][str(K)][b] for K in KS]
                t = cur["M6s"]["8"][b]
                if t <= ys[0]:
                    keq[b] = "<= 8"
                elif t > max(ys):
                    keq[b] = "> 64"
                else:
                    i = next(j for j in range(1, len(KS)) if ys[j] >= t)
                    lk = np.log(KS[i - 1]) + (t - ys[i - 1]) / (ys[i] - ys[i - 1]) * (np.log(KS[i]) - np.log(KS[i - 1]))
                    keq[b] = round(float(np.exp(lk)), 1)
            rep["readings"]["Keq|{}|{}".format(model, sname)] = keq
        # R1: M6s - M5, unseen, K = 8 and 16
        r1 = {}
        for K in (8, 16):
            bands = {b: paired(rooms((model, "M6s", K, "pooled", b), "unseen"),
                               rooms(("physics", "M5", K, "pooled", b), "unseen"))[0] for b in BANDS}
            r1[str(K)] = {"bands": bands, "wins": int(sum(v[1] > 0 for v in bands.values()))}
        rep["readings"]["R1|{}".format(model)] = {"by_K": r1, "holds": bool(all(r1[k]["wins"] >= 2 for k in r1))}
    # D192 (X1b): bounding rectangle for both priors; physics references (M3 exact, M4-iii box, beta-fitted per draw)
    for sname, g in SETS.items():
        cur = {}
        for meth, mdl in (("M5", "physics"), ("M3", "physics"), ("M4-iii", "physics"), ("M0-iii", "R0_120K"),
                          ("M6-iii", "R0_120K"), ("M7-iii", "physics")):
            cur[meth] = {}
            for K in KS:
                key = (mdl, meth, 32 if meth == "M0-iii" else K, "pooled")
                cur[meth][str(K)] = {b: mean(key + (b,), g) for b in BANDS}
        rep["curves"]["bbox|{}".format(sname)] = cur
        q = {}
        for K in KS:
            bands = {b: paired(rooms(("R0_120K", "M6-iii", K, "pooled", b), g),
                               rooms(("physics", "M7-iii", K, "pooled", b), g)) for b in BANDS}
            q[str(K)] = {"bands": {b: {"delta": v[0], "n": v[1]} for b, v in bands.items()},
                         "wins": int(sum(v[0][1] > 0 for v in bands.values()))}
        rep["readings"]["Q|{}".format(sname)] = {"by_K": q, "holds": bool(all(q[k]["wins"] >= 2 for k in ("8", "16"))
                                                                          and all(q[k]["bands"]["25-100"]["n"] > 0
                                                                                  for k in ("8", "16")))}
    head = {}
    for model in MODELS:
        for meth, mdl in (("M0", model), ("M5", "physics"), ("M2", model), ("M6", model), ("M6s", model)):
            head["{}|{}".format(mdl if meth == "M5" else model, meth)] = {
                b: mean((mdl, meth, 32, "all_nondrawn", b), "unseen") for b in BANDS}
    rep["headline_unseen_K32_all_nondrawn"] = head
    times = {}
    for f in fits:
        times.setdefault("{}|{}".format(f["model"], f["method"]), []).append(f.get("fit_s", 0.0))
    rep["fit_time_median_s"] = {k: float(np.median(v)) for k, v in times.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "netprior.json").write_text(json.dumps(rep, indent=1))

    L = ["# P4-9 X1 (D189): the network as a prior for sparse-measurement reconstruction", "",
         "M6 = zero-shot network field + Step 1's Helmholtz KRR (M5) on the residual; M6s = the same with the network's "
         "scale c(f) fitted (nests M5 at c = 0). Off-lattice, pooled strata (< 0.3 m + >= 0.3 m), mean over rooms of "
         "the mean over seeds 0-2; clean measurements. Primary: M6s with R0.", ""]
    for k, v in rep["readings"].items():
        if k.startswith("R1") or k.startswith("R2") or k.startswith("Q|"):
            L.append("* **{}: {}**".format(k, "HOLDS" if v["holds"] else "does not hold"))
    L.append("")
    for model in MODELS:
        for sname in SETS:
            cur = rep["curves"]["{}|{}".format(model, sname)]
            L += ["## {} -- {}: r vs K (25-100 / 100-200 / 200-300 Hz)".format(model, sname), "",
                  "| method | " + " | ".join("K={}".format(K) for K in KS) + " |", "|---|" + "---|" * len(KS)]
            for meth in ("M0", "M5", "M2", "M6", "M6s"):
                L.append("| {} | ".format(meth) + " | ".join(" / ".join("{:.3f}".format(cur[meth][str(K)][b])
                                                                        for b in BANDS) for K in KS) + " |")
            r2 = rep["readings"]["R2|{}|{}".format(model, sname)]["by_K"]
            L += ["", "M6s minus the better of M0 / M5 (bar named) [95% CI]:", ""]
            for K in KS:
                L.append("* K={}: ".format(K) + "; ".join("{} {:+.3f} [{:+.3f}, {:+.3f}] vs {}".format(
                    b, *r2[str(K)]["bands"][b]["delta"], r2[str(K)]["bands"][b]["bar"]) for b in BANDS))
            L += ["", "Microphones the prior is worth (K at which M5 matches M6s @ K = 8): " + ", ".join(
                "{} Hz {}".format(b, v) for b, v in rep["readings"]["Keq|{}|{}".format(model, sname)].items()), ""]
    for sname in SETS:
        cur = rep["curves"]["bbox|{}".format(sname)]
        L += ["## Partial geometry (D192): bounding rectangle only -- {} (25-100 / 100-200 / 200-300 Hz)".format(sname), "",
              "| method | " + " | ".join("K={}".format(K) for K in KS) + " |", "|---|" + "---|" * len(KS)]
        for meth in ("M3", "M5", "M0-iii", "M4-iii", "M6-iii", "M7-iii"):
            L.append("| {} | ".format(meth) + " | ".join(" / ".join("{:.3f}".format(cur[meth][str(K)][b])
                                                                    for b in BANDS) for K in KS) + " |")
        q = rep["readings"]["Q|{}".format(sname)]["by_K"]
        L += ["", "M6-iii (network prior, box) minus M7-iii (physics prior, box) [95% CI]:", ""]
        for K in KS:
            L.append("* K={}: ".format(K) + "; ".join("{} {:+.3f} [{:+.3f}, {:+.3f}]".format(
                b, *q[str(K)]["bands"][b]["delta"]) for b in BANDS))
        L.append("")
    L += ["Median fit time (s): " + ", ".join("{} {:.2f}".format(k, v) for k, v in sorted(rep["fit_time_median_s"].items())
                                              if "M5" in k or "M6" in k or "M2" in k)]
    (OUT / "netprior.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--room", type=int, default=None)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--render-iii", action="store_true")
    ap.add_argument("--room-iii", type=int, default=None)
    a = ap.parse_args()
    if a.selftest:                              # krr_gls with g = [G0] must equal Step 1's M5 krr
        from scripts.p4_9_grounding import krr
        rng = np.random.default_rng(0)
        P, Q = rng.uniform(0.5, 4, (12, 2)), rng.uniform(0.5, 4, (30, 2))
        Y = (rng.standard_normal((12, 601)) + 1j * rng.standard_normal((12, 601)))
        a5, a6 = krr(P, Y, Q, 1e-2), krr_gls(P, Y, Q, 1e-2)
        print("krr_gls == krr:", float(np.abs(a5[:, 1:] - a6[:, 1:]).max() / np.abs(a5[:, 1:]).max()))
        m = rng.standard_normal((42, 601)) + 1j * rng.standard_normal((42, 601))
        Yp = 2.0 * m[:12]                           # measurements = 2 x prior: M6s must recover them (c = 2)
        F = krr_gls(P, Yp, Q, 1e-2, Mp=m[:12], Mq=m[12:])
        print("M6s recovers c = 2 exactly:", float(np.abs(F[:, 1:] - 2 * m[12:, 1:]).max()))
    if a.calibrate:
        calibrate()
    if a.room is not None:
        run_room(a.room)
    if a.render_iii:
        render_iii()
    if a.room_iii is not None:
        run_room_iii(a.room_iii)
    if a.aggregate:
        aggregate()


if __name__ == "__main__":
    main()
