"""P4-10 P-2 (D217, D219): zero-training physics audit of R0 / R1 / R1-S / R1-M against the truth, 15 in-family rooms.
  --residual --render --model NAME      GPU: 0d's points rendered raw + band-limited for one model -> outputs/p4_10/p2/resid/NAME__room.npz
  --residual --score                    CPU: single-radius residual per band (model, truth) -> outputs/p4_10/p2/residual.json (R0 re-scored from 0d's stored renders, must match its report)
  --recip --render --model NAME --idx I GPU: swapped renders of room I (in-family 0-14, unseen 15-30) -> outputs/p4_10/p2/recip/NAME__<i>.npz;  --recip --parity : R1's stored scores must be reproduced
  --recip --score                       CPU: normal vs swapped r -> outputs/p4_10/p2/recip.json
  --causal                              CPU: early excess per room (truth + models) -> outputs/p4_10/p2/causality.json
  --aggregate | --fig
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import scripts.p4_10_0d_meanvalue as MV

OUT = Path("outputs/p4_10/p2")
CK = {"R0_120K": "outputs/p4_8/arms/R0/ckpt_iter0120000.pt", "R1_150K": "outputs/p4_10/arms/R1/ckpt_iter0150000.pt",
      "R1S_150K": "outputs/p4_10/arms/R1S/ckpt_iter0150000.pt", "R1M_150K": "outputs/p4_10/arms/R1M/ckpt_iter0150000.pt"}
B = ("25-100", "100-200", "200-300")
SEED, NB = 20261013, 10000
SRC = np.array([0.5, 1.6])
G2 = "outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz"
UN = "outputs/p4_8/cache/unseen_{}.npz"


def write_json(path, obj):
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / "{}.{}.tmp".format(path.name, os.getpid())
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def boot(v):
    v = np.asarray(v, float)
    m = v[np.random.default_rng(SEED).integers(0, len(v), (NB, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


# ------------------------------------------------------------------------------------------------- (a) wave equation
def test_rooms():
    from scripts.p4_9_dir_zeroshot import dense_rooms
    return [x for x in dense_rooms() if x["group"] == "test"]


def residual_render(name):
    import h5py
    import torch

    from aaf.eval.p3_2_eval import load_model
    from aaf.models.conditioning_2d import build_cond_vector_2d
    dev = torch.device("cuda")
    model, renderer, *_ = load_model(Path(CK[name]), dev)
    model.eval()
    renderer.eval()
    lat = np.load(G2.format(name))
    (OUT / "resid").mkdir(parents=True, exist_ok=True)
    for r in test_rooms():
        dst = OUT / "resid" / "{}__{}.npz".format(name, r["path"].stem)
        if dst.exists():
            continue
        P = np.load(MV.OUT / "fdtd" / (r["path"].stem + ".npz"))["P"]
        M = P.reshape(-1, 2)
        with h5py.File(r["path"]) as f:
            verts = [tuple(map(float, v)) for v in json.loads(f.attrs["verts"])]
            L, W = float(f.attrs["L"]), float(f.attrs["W"])
            keep = np.asarray(json.loads(f.attrs["lattice_keep"]), bool).reshape(64, 64)
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]), float)
        z = build_cond_vector_2d("geom_token", L, W, [0.15] * len(verts), verts=verts, device=dev)
        lo, hi = torch.zeros(2, device=dev), torch.tensor([L, W], device=dev, dtype=torch.float32)
        R = torch.tensor(M, dtype=torch.float32, device=dev)
        S = torch.tensor([[0.5, 1.6]], device=dev)
        o = []
        with torch.no_grad():
            for s0 in range(0, len(R), 32):
                rr = R[s0:s0 + 32]
                o.append(renderer(model, rr, S.expand(len(rr), -1), lo, hi, z_s=z.unsqueeze(0).expand(len(rr), -1), n_out=601).cpu().numpy())
        raw = np.concatenate(o)
        proj = MV.band_limited_at(lat["P%d" % r["cache_i"]][:, :601], keep, L, W, rx, M)
        tmp = dst.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, raw=raw.reshape(P.shape[:3] + (601,)).astype(np.complex64), proj=proj.reshape(P.shape[:3] + (601,)).astype(np.complex64))
        tmp.rename(dst)
        print("resid", dst.name, flush=True)


def residual_score():
    rows, rep = {}, {}
    ref = json.loads((MV.OUT / "report.json").read_text())
    truth, per, missing = {}, {n: {} for n in CK}, {n: [] for n in CK}
    for r in test_rooms():
        st = r["path"].stem
        t = np.load(MV.OUT / "fdtd" / (st + ".npz"))
        truth[st] = MV.residuals(t["H"], t["sets"])
        for n in CK:
            f = MV.OUT / "r0" / (st + ".npz") if n == "R0_120K" else OUT / "resid" / "{}__{}.npz".format(n, st)
            if not f.exists():
                missing[n].append(st)
                continue
            o = np.load(f)
            per[n][st] = {"raw": MV.residuals(o["raw"], t["sets"]), "proj": MV.residuals(o["proj"], t["sets"])}
        ctr = t["P"][:, 0, 0]
        k = (ctr - 0.12) / 0.06
        rows[st] = {"on_lattice_centres": int(np.all(np.abs(k - np.round(k)) < 1e-6, axis=1).sum())}
    for b in B:                                                                          # R0 must reproduce 0d's report
        for kind in ("raw", "proj"):
            got = np.mean([per["R0_120K"][x][kind][b]["single"] for x in per["R0_120K"]])
            want = ref["form"]["single"][b][kind + "_mean"][0]
            assert abs(got - want) < 1e-6, (b, kind, got, want)
    for b in B:
        tv_all = np.array([truth[x][b]["single"] for x in truth])
        rep[b] = {"truth": boot(tv_all), "truth_max_room": float(tv_all.max())}
        for n in CK:
            ks = [x for x in truth if x in per[n]]
            tv = np.array([truth[x][b]["single"] for x in ks])
            for kind in ("raw", "proj"):
                v = np.array([per[n][x][kind][b]["single"] for x in ks])
                rep[b]["{}|{}".format(n, kind)] = {"n_rooms": len(ks), "level": boot(v), "minus_truth": boot(v - tv), "over_truth": float(v.mean() / tv.mean())}
    out = {"n_rooms": len(truth), "rooms": rows, "missing_rooms": missing, "on_lattice_centres_total": int(sum(x["on_lattice_centres"] for x in rows.values())), "bands": rep}
    write_json(OUT / "residual.json", out)
    for b in B:
        print(b, "truth {:.4f}".format(rep[b]["truth"][0]), {k: "{:.4f} (x{:.2f}, {} rooms)".format(v["level"][0], v["over_truth"], v["n_rooms"]) for k, v in rep[b].items() if "|" in k})


# --------------------------------------------------------------------------------------------------- (b) reciprocity
def recip_rooms():
    from scripts.p4_9_grounding import eval_rooms
    return eval_rooms()


def recip_render(name, idx):
    import torch

    from aaf.data.augment_2d import is_interior_node
    from aaf.eval.p3_2_eval import load_model
    from aaf.models.conditioning_2d import build_cond_vector_2d
    from scripts.p4_7_m1_modal import scores
    from scripts.p4_9_grounding import load
    rr = recip_rooms()[idx]
    dst = OUT / "recip" / "{}__{:02d}.json".format(name, idx)
    if dst.exists():
        return
    dev = torch.device("cuda")
    model, renderer, *_ = load_model(Path(CK[name]), dev)
    model.eval()
    renderer.eval()
    r = load(rr["path"])
    cache = np.load((G2 if rr["group"] == "test" else UN).format(name))
    assert np.allclose(cache["rx%d" % rr["cache_i"]], r["rx"])
    P = cache["P%d" % rr["cache_i"]][:, :601].astype(np.complex128)
    keep = np.nonzero(r["off"] & is_interior_node(r["verts"], r["rx"], 0.02, 0.02))[0]
    cond = build_cond_vector_2d("geom_token", r["L"], r["W"], [0.15] * len(r["verts"]), verts=r["verts"], device=dev)
    lo = torch.zeros(2, device=dev)
    hi = torch.tensor([r["L"], r["W"]], device=dev, dtype=torch.float32)
    Ps = np.zeros((len(keep), 601), np.complex128)
    S = torch.tensor(r["rx"][keep], dtype=torch.float32, device=dev)
    R = torch.tensor(SRC, dtype=torch.float32, device=dev).view(1, 2)
    with torch.no_grad():
        for s0 in range(0, len(keep), 16):
            t = S[s0:s0 + 16]
            H = renderer(model, R.expand(len(t), -1), t, lo, hi, z_s=cond.unsqueeze(0).expand(len(t), -1), n_out=601)
            Ps[s0:s0 + 16] = H.cpu().numpy()
    T = r["T"][keep]
    row = {"room": r["name"], "group": rr["group"], "n": int(len(keep)), "normal": scores(P[keep], T), "swapped": scores(Ps, T), "swapped_vs_normal": scores(Ps, P[keep])}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(json.dumps(row))
    tmp.rename(dst)
    print(dst.name, " ".join("{} {:.3f}/{:.3f}".format(b, row["normal"][b], row["swapped"][b]) for b in B), flush=True)


def recip_parity(tol=1e-3):
    """two in-family rooms re-rendered for R1 by this script's path; stored R1_150K.json (4097-bin path) must be matched"""
    st = json.loads(Path("outputs/p4_9/recip_gap/R1_150K.json").read_text())["rooms"]
    rooms = recip_rooms()
    worst = 0.0
    for i in sorted((i for i, x in enumerate(rooms) if x["group"] == "test"), key=lambda i: next(s["n"] for s in st if s["room"] == _name(rooms[i])))[:2]:
        f = OUT / "recip" / "R1_150K__{:02d}.json".format(i)
        d = json.loads(f.read_text())
        s = next(x for x in st if x["room"] == d["room"])
        for k in ("normal", "swapped"):
            for b in B:
                worst = max(worst, abs(d[k][b] - s[k][b]))
        print(d["room"], "n", d["n"], "vs", s["n"], {b: (round(d["swapped"][b], 4), round(s["swapped"][b], 4)) for b in B})
    print("PARITY worst abs difference", worst, "PASS" if worst <= tol else "FAIL")
    write_json(OUT / "parity.json", {"worst_abs_difference": worst, "tol": tol, "pass": bool(worst <= tol), "rooms": 2})
    return worst


def _name(x):
    return Path(x["path"]).name


def recip_score():
    rep = {}
    for n in CK:
        if n in ("R0_120K", "R1_150K"):
            rows = json.loads(Path("outputs/p4_9/recip_gap/{}.json".format(n)).read_text())["rooms"]
        else:
            rows = [json.loads(f.read_text()) for f in sorted((OUT / "recip").glob("{}__*.json".format(n)))]
        e = {}
        for g, lab in (("test", "in_family15"), ("unseen", "unseen16")):
            sel = [x for x in rows if x["group"] == g]
            if not sel:
                continue
            e[lab] = {"n_rooms": len(sel)}
            for b in B:
                nv, sv = np.array([x["normal"][b] for x in sel]), np.array([x["swapped"][b] for x in sel])
                e[lab][b] = {"normal": boot(nv), "swapped": boot(sv), "swapped_minus_normal": boot(sv - nv), "gap_ok_0.05": bool(sv.mean() >= nv.mean() - 0.05)}
        rep[n] = e
    write_json(OUT / "recip.json", rep)
    for n, e in rep.items():
        print(n, {lab: {b: "{:.3f}->{:.3f}".format(v["normal"][0], v["swapped"][0]) for b, v in x.items() if b in B} for lab, x in e.items()})


# ----------------------------------------------------------------------------------------------------- (c) causality
T0, FS600, NT, UP = 0.04, 600.0, 1200, 16


def ir_bandlimited(H, t0=T0, up=UP):
    """[R, 601] stored bins -> [R, 1200 * up] IR, 25-300 Hz with 10 Hz cosine tapers inside the band, source pulse delay removed (t = 0 is the pulse centre)"""
    f = np.arange(601) * 0.5
    w = np.where((f >= 25) & (f <= 300), 1.0, 0.0)
    r = (f >= 25) & (f < 35)
    w[r] = np.sin(0.5 * np.pi * (f[r] - 25) / 10) ** 2
    d = (f > 290) & (f <= 300)
    w[d] = np.sin(0.5 * np.pi * (300 - f[d]) / 10) ** 2
    X = np.asarray(H) * w * np.exp(2j * np.pi * f * t0)
    n = NT * up
    Z = np.zeros((len(X), n // 2 + 1), np.complex128)
    Z[:, :601] = X
    return np.fft.irfft(Z, n=n, axis=-1) * up


def early_excess(h, dist, c=343.0, fs=FS600 * UP, late=(1.7, 2.0)):
    """per receiver: (early excess, direct energy, window length in s); h = pulse-centred IR of one period"""
    n = h.shape[1]
    dt = 1.0 / fs
    P = h ** 2
    k_late = slice(int(round(late[0] / dt)), min(n, int(round(late[1] / dt))))
    rate = P[:, k_late].sum(1) / (late[1] - late[0])                                       # energy per second of the late floor
    ex, di, ln = np.zeros(len(h)), np.zeros(len(h)), np.zeros(len(h))
    for i in range(len(h)):
        td = dist[i] / c
        w1 = int(np.floor((td - 0.005) / dt))
        a, b = int(round((td - 0.005) / dt)), int(round((td + 0.005) / dt))
        ln[i] = max(w1, 0) * dt
        ex[i] = P[i, :max(w1, 0)].sum() - rate[i] * ln[i]
        di[i] = P[i, a:b].sum()
    return ex, di, ln


def onset_offset(h, dist, fs=FS600 * UP, c=343.0):
    """s: time of the Hilbert-envelope maximum inside [t_direct - 6 ms, t_direct + 6 ms] minus t_direct (verification on the truth)"""
    from scipy.signal import hilbert
    out = np.zeros(len(h))
    for i in range(len(h)):
        td = dist[i] / c
        a, b = int(round((td - 0.006) * fs)), int(round((td + 0.006) * fs))
        lo = max(0, a - int(0.05 * fs))
        e = np.abs(hilbert(h[i, lo:b + int(0.05 * fs)]))[max(a, 0) - lo: b - lo]
        out[i] = (max(a, 0) + int(np.argmax(e))) / fs - td
    return out

def causal():
    from scripts.p4_10_scorecard import PRED, on_lattice
    from scripts.p4_10_truth import stored
    cells = json.loads(Path("configs/p4_10/edit_manifest.json").read_text())["cells"]
    rooms = {c["room"]: c for c in cells if c["group"] == "test"}
    rep = {"rooms": {}, "min_rx": 50}
    for rm, c in rooms.items():
        base = stored(c)
        d = np.hypot(*(base["rx"] - SRC).T)
        sel = np.nonzero(~on_lattice(base["rx"]) & (d >= 3.0))[0]
        e = {"n_rx": int(len(sel))}
        if len(sel) < 50:
            e["excluded"] = True
            rep["rooms"][rm] = e
            continue
        ht = ir_bandlimited(base["H"][sel])
        off = onset_offset(ht, d[sel])
        e["onset_offset_ms_median"] = float(np.median(off) * 1e3)
        for nm in ("truth",) + tuple(CK):
            h = ht if nm == "truth" else ir_bandlimited(np.load(PRED / nm / "orig" / (rm + ".npz"))["P"][sel])
            row = {}
            for tag, late in (("", (1.7, 2.0)), ("_late_1.6_1.9", (1.6, 1.9))):
                ex, di, ln = early_excess(h, d[sel], late=late)
                row["pooled" + tag] = float(ex.sum() / di.sum())
                row["median_rx" + tag] = float(np.median(ex / np.maximum(di, 1e-300)))
            row["direct_energy_mean"] = float(di.mean())
            row["window_ms_min_max"] = [float(ln.min() * 1e3), float(ln.max() * 1e3)]
            e[nm] = row
        rep["rooms"][rm] = e
    OUT.mkdir(parents=True, exist_ok=True)
    used = [k for k, v in rep["rooms"].items() if "excluded" not in v]
    rep["n_rooms_used"], rep["n_rooms_excluded"] = len(used), len(rep["rooms"]) - len(used)
    rep["onset_offset_ms_median_over_rooms"] = float(np.median([rep["rooms"][k]["onset_offset_ms_median"] for k in used]))
    rep["onset_check_ok"] = bool(abs(rep["onset_offset_ms_median_over_rooms"]) <= 3.0)
    for key in ("pooled", "median_rx", "pooled_late_1.6_1.9"):
        tr = np.array([rep["rooms"][k]["truth"][key] for k in used])
        rep[key] = {"truth": boot(tr)}
        for nm in CK:
            m = np.array([rep["rooms"][k][nm][key] for k in used])
            rep[key][nm] = {"level": boot(m), "minus_2x_truth": boot(m - 2 * tr), "minus_truth": boot(m - tr), "ratio_of_means": float(m.mean() / tr.mean())}
            lo = rep[key][nm]["minus_2x_truth"][1]
            rep[key][nm]["condition_met"] = bool(m.mean() >= 2 * tr.mean() and lo > 0)
    write_json(OUT / "causality.json", rep)
    print("rooms", len(used), "excluded", rep["n_rooms_excluded"], "onset offset (ms, median)", rep["onset_offset_ms_median_over_rooms"], rep["onset_check_ok"])
    for key in ("pooled", "median_rx"):
        print(key, "truth", rep[key]["truth"], {n: (rep[key][n]["level"], rep[key][n]["minus_2x_truth"], rep[key][n]["condition_met"]) for n in CK})


# ----------------------------------------------------------------------------------------------------- report + figure
NAMES = {"R0_120K": "R0", "R1_150K": "R1", "R1S_150K": "R1-S", "R1M_150K": "R1-M"}


def aggregate():
    res, rec, cau = (json.loads((OUT / f).read_text()) for f in ("residual.json", "recip.json", "causality.json"))
    L = ["# P4-10 P-2 (D217, D219): zero-training physics audit", "", "15 in-family rooms, R0 @120K, R1 / R1-S / R1-M @150K against the simulated truth. CIs: paired bootstrap over rooms (10k).", "",
         "## Wave equation: single-radius mean-value residual at held-out centres (RMS residual / RMS |p|; lower is more physical)", "",
         "| band | truth (max room) | " + " | ".join("{} raw".format(NAMES[n]) for n in CK) + " | " + " | ".join("{} band-limited".format(NAMES[n]) for n in CK) + " |", "|---|---|" + "---|" * 8]
    for b in B:
        r = res["bands"][b]
        L.append("| {} | {:.4f} ({:.4f}) | {} | {} |".format(b, r["truth"][0], r["truth_max_room"],
                 " | ".join("{:.4f} (x{:.1f})".format(r["{}|raw".format(n)]["level"][0], r["{}|raw".format(n)]["over_truth"]) for n in CK),
                 " | ".join("{:.4f} (x{:.1f})".format(r["{}|proj".format(n)]["level"][0], r["{}|proj".format(n)]["over_truth"]) for n in CK)))
    L += ["", "On-lattice centres among the {} x 8: {} (must be 0).".format(res["n_rooms"], res["on_lattice_centres_total"]), "", "## Reciprocity: normal and swapped renders (r against the same truth)", "",
          "| model | set | normal 25-100 / 100-200 / 200-300 | swapped | swapped - normal |", "|---|---|---|---|---|"]
    for n in CK:
        for lab in ("in_family15", "unseen16"):
            if lab not in rec[n]:
                continue
            e = rec[n][lab]
            L.append("| {} | {} ({} rooms) | {} | {} | {} |".format(NAMES[n], lab, e["n_rooms"], " / ".join("{:.3f}".format(e[b]["normal"][0]) for b in B), " / ".join("{:.3f}".format(e[b]["swapped"][0]) for b in B),
                                                                    " / ".join("{:+.3f} [{:+.3f}, {:+.3f}]".format(*e[b]["swapped_minus_normal"]) for b in B)))
    c = cau
    L += ["", "## Causality: early excess before the direct sound / direct-sound energy (25-300 Hz, receivers >= 3 m)", "",
          "Rooms used {} (excluded {}), receivers per room {} to {}; onset check (median offset of the truth's arrival from t_direct) {:+.2f} ms, {}.".format(
              c["n_rooms_used"], c["n_rooms_excluded"], min(v["n_rx"] for v in c["rooms"].values()), max(v["n_rx"] for v in c["rooms"].values()), c["onset_offset_ms_median_over_rooms"], "OK" if c["onset_check_ok"] else "FAILS: measure not read"), "",
          "| reading | truth | " + " | ".join(NAMES[n] for n in CK) + " |", "|---|---|" + "---|" * 4]
    for key, lab in (("pooled", "pooled over receivers (primary)"), ("median_rx", "median over receivers"), ("pooled_late_1.6_1.9", "pooled, late window [1.6, 1.9) s")):
        L.append("| {} | {:+.4f} [{:+.4f}, {:+.4f}] | {} |".format(lab, *c[key]["truth"], " | ".join("{:+.4f} [{:+.4f}, {:+.4f}]; {:.2f}x truth; minus truth {:+.4f} [{:+.4f}, {:+.4f}]; minus 2x truth {:+.4f} [{:+.4f}, {:+.4f}]; condition {}".format(
            *c[key][n]["level"], c[key][n]["ratio_of_means"], *c[key][n]["minus_truth"], *c[key][n]["minus_2x_truth"], "MET" if c[key][n]["condition_met"] else "not met") for n in CK)))
    OUT.mkdir(parents=True, exist_ok=True)
    par = json.loads((OUT / "parity.json").read_text()) if (OUT / "parity.json").exists() else {"not_run": "no parity.json"}
    L += ["", "Parity of this script's swapped render (601-bin path) against R1's stored scores on 2 rooms: {}".format(json.dumps(par))]
    n_miss = {NAMES[n]: res["missing_rooms"][n] for n in CK if res["missing_rooms"][n]}
    L += ["", "Missing renders (counted): wave equation " + (json.dumps(n_miss) if n_miss else "none") + "; reciprocity rooms scored are in the table (15 in family, 16 unseen when complete)."]
    import os
    write_json(OUT / "report.json", {"wave_equation": res, "reciprocity": rec, "causality": cau})
    tmp = OUT / "report.md.{}.tmp".format(os.getpid())
    tmp.write_text("\n".join(L) + "\n")
    os.replace(tmp, OUT / "report.md")
    print("\n".join(L))


def figure():
    import scripts.p4_10_md_style as S
    plt = S.plt
    res, rec, cau = (json.loads((OUT / f).read_text()) for f in ("residual.json", "recip.json", "causality.json"))
    fig = S.new(16, 9)
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.985, top=0.745, bottom=0.275, wspace=0.26)
    meta = {"figure": "P2", "source": "outputs/p4_10/p2/report.json"}
    names = list(CK)
    ax = fig.add_subplot(gs[0, 0])                                            # wave equation
    for i, b in enumerate(B):
        for k, n in enumerate(names):
            if "{}|raw".format(n) not in res["bands"][b] or not res["bands"][b]["{}|raw".format(n)]["n_rooms"]:
                continue
            v = res["bands"][b]["{}|raw".format(n)]["level"]
            ax.bar(i + (k - 1.5) * 0.2, v[0], 0.18, color=S.COL[NAMES[n]], yerr=[[v[0] - v[1]], [v[2] - v[0]]], error_kw={"lw": 1.2}, label=NAMES[n] if i == 0 else None)
            pj = res["bands"][b]["{}|proj".format(n)]["level"][0]
            ax.plot(i + (k - 1.5) * 0.2, pj, "D", ms=7, color="white", markeredgecolor="black", label="after removing non-wave detail" if i == 0 and k == 0 else None, zorder=6)
        t = res["bands"][b]["truth"][0]
        ax.plot([i - 0.42, i + 0.42], [t, t], color="black", lw=3, label="truth (simulation)" if i == 0 else None, zorder=7)
        meta["wave|" + b] = {"truth": t, **{NAMES[n]: {"raw": res["bands"][b]["{}|raw".format(n)]["level"][0], "band_limited": res["bands"][b]["{}|proj".format(n)]["level"][0]} for n in names}}
    ax.set_xticks(range(3))
    ax.set_xticklabels([b + " Hz" for b in B])
    ax.set_title("Does it obey the wave equation?\n(error, lower is better)", loc="left", fontsize=16)
    ax.legend(frameon=False, fontsize=14, loc="upper left", ncol=1)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax2 = fig.add_subplot(gs[0, 1])                                           # reciprocity
    for i, b in enumerate(B):
        for k, n in enumerate(names):
            if "in_family15" not in rec[n]:
                continue
            e = rec[n]["in_family15"][b]
            v = e["swapped_minus_normal"]
            ax2.bar(i + (k - 1.5) * 0.2, -v[0], 0.18, color=S.COL[NAMES[n]], yerr=[[v[2] - v[0]], [v[0] - v[1]]], error_kw={"lw": 1.2})
        meta["recip|" + b] = {NAMES[n]: {"normal": rec[n]["in_family15"][b]["normal"][0], "swapped": rec[n]["in_family15"][b]["swapped"][0]} for n in names if "in_family15" in rec[n]}
    ax2.axhline(0, color="black", lw=3)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([b + " Hz" for b in B])
    ax2.set_title("Swap source and listener:\naccuracy lost (a real room: 0)", loc="left", fontsize=16)
    ax3 = fig.add_subplot(gs[0, 2])                                           # causality
    c = cau["pooled"]
    vals = [("truth", c["truth"])] + [(NAMES[n], c[n]["level"]) for n in names]
    for i, (lab, v) in enumerate(vals):
        ax3.bar(i, v[0], 0.62, color="#111111" if lab == "truth" else S.COL[lab], yerr=[[v[0] - v[1]], [v[2] - v[0]]], error_kw={"lw": 1.4})
    ax3.axhline(2 * c["truth"][0], color="#b00020", lw=2.5, ls=(0, (5, 3)))
    ax3.text(4.45, 2 * c["truth"][0] * 1.02, "2x truth: the bar for a non-causal model", ha="right", va="bottom", fontsize=14, color="#b00020")
    ax3.set_xticks(range(len(vals)))
    ax3.set_xticklabels([v[0] for v in vals])
    ax3.set_ylim(0, 2 * c["truth"][0] * 1.35)
    ax3.set_title("Does sound arrive before it can?\n(energy before the direct sound)", loc="left", fontsize=16)
    meta["causality"] = {"truth": c["truth"], **{NAMES[n]: {"level": c[n]["level"], "minus_truth": c[n]["minus_truth"], "minus_2x_truth": c[n]["minus_2x_truth"], "condition_met": c[n]["condition_met"]} for n in names}}
    h, l = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=15, bbox_to_anchor=(0.5, 0.895))
    fig.suptitle("How physical are our fields?", fontsize=22, x=0.02, ha="left", y=0.975, fontweight="bold")
    cap = ("Wave equation: compare the average of the field on a small circle with its centre value; a true sound field matches the prediction of the wave equation, so the error is small "
           "(15 known rooms, points away from the training receivers; white diamonds: the model with its fine, non-wave detail removed). "
           "Swap test: loss of accuracy when the model is given the listener's position as the source (0 = reciprocal). "
           "Causality: energy before the direct sound could arrive, over the energy of the direct sound, at receivers 3 m or more away; band-limiting itself rings, so the simulation is not zero.")
    import textwrap
    fig.text(0.02, 0.02, "\n".join(textwrap.wrap(cap, 150)), fontsize=14, color="#444444", ha="left", va="bottom", linespacing=1.25)
    import os
    for ext in ("png", "pdf"):
        tmp = S.OUT / "P2_how_physical.{}.{}".format(os.getpid(), ext)
        fig.savefig(tmp, dpi=200)
        os.replace(tmp, S.OUT / "P2_how_physical.{}".format(ext))
    write_json(S.OUT / "P2_how_physical.json", meta)
    plt.close(fig)
    print("wrote P2_how_physical")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("--residual", "--recip", "--causal", "--aggregate", "--fig", "--render", "--score", "--parity", "--final"):
        ap.add_argument(k, action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--idx", type=int, default=None)
    a = ap.parse_args()
    if a.residual and a.render:
        residual_render(a.model)
    elif a.residual and a.score:
        residual_score()
    elif a.recip and a.render:
        recip_render(a.model, a.idx)
    elif a.recip and a.parity:
        recip_parity()
    elif a.recip and a.score:
        recip_score()
    elif a.final:                                                                        # chained in Slurm after the renders: every piece, tolerant of missing renders
        try:
            recip_parity()
        except Exception as ex:                                                          # parity rooms not rendered: said so, not hidden
            write_json(OUT / "parity.json", {"not_run": repr(ex)})
            print("parity not run:", repr(ex))
        for fn in (residual_score, recip_score, causal, aggregate, figure):
            fn()
    elif a.causal:
        causal()
    elif a.aggregate:
        aggregate()
    elif a.fig:
        figure()
