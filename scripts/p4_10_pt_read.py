"""P4-10 P-1 reads (D217 / D218): R1-P(t) against R1-ctl at the same step. Arms live in outputs/p4_10/arms_pt/{R1PT,R1CTL}; NAME = ARM_stepK (152K 155K 157K 160K).
  --task NAME=CK --kind in|un|tr --idx I   GPU, one room: in = dense lattice P, 0d points raw and band-limited, omni-FD gradient; un = unseen room; tr = 15 training-rule rooms
                          -> scratch pt/{in,un,tr}/; skipped if the file exists; a lock keeps two GPU pools from rendering the same room
  --score STEPK           CPU: the six reads, paired over the 15 rooms (bootstrap 10k, seed 20261013) -> outputs/p4_10/pt/read_STEP.{json,md}
Static HF / UF: the frozen per-bin Pearson of dB levels on off-lattice receivers of the in-family 15 and unseen 16 sets (the standard report's definition, verified in the rehearsal).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

SCR = Path(os.environ.get("P410_PT_SCR", "/fs/nexus-scratch/htakawal/aaf_p410/pt"))
OUT = Path(os.environ.get("P410_PT_OUT", "outputs/p4_10/pt"))
CK = "outputs/p4_10/arms_pt/{}/ckpt_iter{:07d}.pt"
ARMS = ("R1PT", "R1CTL")
STEPS = json.loads(os.environ.get("P410_PT_STEPS", "{\"152K\": 152500, \"155K\": 155000, \"157K\": 157500, \"160K\": 160000}"))      # the rehearsal sets its own label
BAND = {"25-100": (50, 200), "100-200": (200, 400), "200-300": (400, 601)}
B3 = list(BAND)
SEED, NB = 20261013, 10000
N_TRAIN_ROOMS, SEED_TR = 15, 20261127


def boot(v):
    v = np.asarray(v, float)
    m = v[np.random.default_rng(SEED).integers(0, len(v), (NB, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def load_model(ck):
    import torch

    from aaf.eval.p3_2_eval import load_model as lm
    dev = torch.device("cuda")
    m, r, *_ = lm(Path(ck), dev)
    m.eval()
    r.eval()
    return torch, dev, m, r


def test_rooms():
    from scripts.p4_9_dir_zeroshot import dense_rooms
    return [x for x in dense_rooms() if x["group"] == "test"]


def render_pts(torch, dev, model, renderer, verts, L, W, src, pts, chunk=32):
    from aaf.models.conditioning_2d import build_cond_vector_2d
    z = build_cond_vector_2d("geom_token", L, W, [0.15] * len(verts), verts=[tuple(map(float, v)) for v in verts], device=dev)
    lo, hi = torch.zeros(2, device=dev), torch.tensor([L, W], device=dev, dtype=torch.float32)
    S = torch.tensor([list(map(float, src))], device=dev, dtype=torch.float32)
    R = torch.tensor(np.asarray(pts), dtype=torch.float32, device=dev)
    o = []
    with torch.no_grad():
        for s0 in range(0, len(R), chunk):
            rr = R[s0:s0 + chunk]
            o.append(renderer(model, rr, S.expand(len(rr), -1), lo, hi, z_s=z.unsqueeze(0).expand(len(rr), -1), n_out=601).cpu().numpy())
    return np.concatenate(o)


def train_rooms():
    """15 training rooms, fixed draw, the same centre rule (room_points) with each room's OWN source; verts / src from the MS manifest's base corpus"""
    g = json.load(open("configs/sweeps_2d_mat/p4_7_c500_cx500_manifest.json"))["configs"]
    pick = np.random.default_rng(SEED_TR).choice(len(g), N_TRAIN_ROOMS, replace=False)
    return [g[int(i)] for i in sorted(pick)]


def chunk_size():
    return int(os.environ.get("P410_CHUNK", "16"))


def claim(path):
    """an output is written once: skip if it exists; an O_EXCL lock with stale takeover (40 min) keeps two GPU pools from rendering the same room at the same time"""
    if path.exists():
        return False
    lk = SCR / "lock" / (path.parent.name + "__" + path.stem)
    lk.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(lk, os.O_CREAT | os.O_EXCL))
        return True
    except FileExistsError:
        import time
        if time.time() - lk.stat().st_mtime > 2400:
            lk.touch()
            return True
        return False


def release(path):
    lk = SCR / "lock" / (path.parent.name + "__" + path.stem)
    if lk.exists():
        lk.unlink()


def save_npz(path, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / "{}.{}.tmp.npz".format(path.stem, os.getpid())
    np.savez(tmp, **kw)
    os.replace(tmp, path)


def in_family():
    return test_rooms()


PARTIAL = bool(os.environ.get("P410_PT_PARTIAL"))          # rehearsal only: score the rooms whose files exist (never a real read)


def unseen_rooms():
    from scripts.p4_8_unseen_eval import rooms
    return rooms()[0]


def read_h5(path):
    import h5py
    with h5py.File(path) as f:
        return {"T": np.asarray(f["ism/H_complex"])[:, :601], "rx": np.asarray(json.loads(f.attrs["receiver_pos"]), float),
                "verts": [tuple(map(float, v)) for v in json.loads(f.attrs["verts"])], "L": float(f.attrs["L"]), "W": float(f.attrs["W"])}


def render_pts2(L_, verts, L, W, src, pts):
    torch, dev, model, renderer = L_
    return render_pts(torch, dev, model, renderer, verts, L, W, src, pts, chunk=chunk_size())


def task(name, ck, kind, idx):
    """one room's renders -> one npz; kinds: in (0-14), un (0-15), tr (0: the 15 training-rule rooms)"""
    dst = SCR / kind / ("{}__{:02d}.npz".format(name, idx) if kind != "tr" else "{}.npz".format(name))
    if not claim(dst):
        print("skip", dst, flush=True)
        return
    try:
        Lm = load_model(ck)
        if kind == "in":
            from scripts.p4_10_0d_meanvalue import band_limited_at
            r = in_family()[idx]
            h = read_h5(r["path"])
            P = render_pts2(Lm, h["verts"], h["L"], h["W"], (0.5, 1.6), h["rx"])
            st = np.load("outputs/p4_10/r1p_0d/fdtd/{}.npz".format(r["path"].stem))
            X = st["P"]
            M = X.reshape(-1, 2)
            raw = render_pts2(Lm, h["verts"], h["L"], h["W"], (0.5, 1.6), M)
            import h5py
            with h5py.File(r["path"]) as f:
                keep = np.asarray(json.loads(f.attrs["lattice_keep"]), bool).reshape(64, 64)
            proj = band_limited_at(P.astype(np.complex64), keep, h["L"], h["W"], h["rx"], M)
            fs = np.load("outputs/p4_9/fdtd_stencils/dense/{}.npz".format(r["path"].stem))
            hx, hy = 2 * float(fs["dxx"]), 2 * float(fs["dxy"])
            ok = np.nonzero((~on_lattice_(h["rx"])) & (np.hypot(*(h["rx"] - np.array([0.5, 1.6])).T) >= 0.5))[0]
            ok = ok[::max(1, int(os.environ.get("P410_PT_DIRSTRIDE", "4")))]          # every 4th off-lattice receiver (>= 0.5 m from the source): the queue is saturated (D218 addendum); both arms share it
            nb = [render_pts2(Lm, h["verts"], h["L"], h["W"], (0.5, 1.6), h["rx"][ok] + d) for d in ([hx, 0], [-hx, 0], [0, hy], [0, -hy])]
            G = np.stack([(nb[0] - nb[1]) / (2 * hx), (nb[2] - nb[3]) / (2 * hy)], axis=1)
            save_npz(dst, P=P.astype(np.complex64), raw=raw.reshape(X.shape[:3] + (601,)).astype(np.complex64), proj=proj.reshape(X.shape[:3] + (601,)).astype(np.complex64),
                     G=G.astype(np.complex64), ok=ok, sets=st["sets"])
        elif kind == "un":
            c = unseen_rooms()[idx]
            h = read_h5("data/track_p4_8_unseen/" + c.filename)
            P = render_pts2(Lm, h["verts"], h["L"], h["W"], (0.5, 1.6), h["rx"])
            save_npz(dst, P=P.astype(np.complex64))
        else:
            from scripts.p4_10_0d_meanvalue import distinct_sets, room_points
            sets = distinct_sets()
            out = {}
            for i, c in enumerate(train_rooms()):
                verts = [tuple(map(float, v)) for v in c["verts"]]
                L, W = max(v[0] for v in verts), max(v[1] for v in verts)
                X = room_points({"L": L, "W": W, "verts": verts, "src": c["src"]}, 1000 + i, sets)
                out["tr__{:02d}".format(i)] = render_pts2(Lm, verts, L, W, tuple(c["src"]), X.reshape(-1, 2)).reshape(X.shape[:3] + (601,)).astype(np.complex64)
            save_npz(dst, sets=np.array(sets), **out)
        print("done", dst, flush=True)
    finally:
        release(dst)


def on_lattice_(rx):
    k = (np.asarray(rx) - 0.12) / 0.06
    return np.all(np.abs(k - np.round(k)) < 1e-6, axis=1)


def rows_in(name):
    """per in-family room: P (model), T, rx, and the extra arrays"""
    for i, r in enumerate(in_family()):
        if PARTIAL and not (SCR / "in" / "{}__{:02d}.npz".format(name, i)).exists():
            continue
        z = np.load(SCR / "in" / "{}__{:02d}.npz".format(name, i))
        h = read_h5(r["path"])
        yield i, r, z, h


def band_r(P, T, rx):
    s = ~on_lattice_(rx)
    d = lambda x: 20 * np.log10(np.maximum(np.abs(x), 1e-30))  # noqa: E731
    a, b = d(P[s]), d(T[s])
    a, b = a - a.mean(0), b - b.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))
    return {k: float(np.nanmean(c[lo:hi])) for k, (lo, hi) in BAND.items()}


def static_rows(name):
    tr = {b: [] for b in B3}
    uf = {b: [] for b in B3}
    for i, r, z, h in rows_in(name):
        v = band_r(z["P"], h["T"], h["rx"])
        for b in B3:
            tr[b].append(v[b])
    for j, c in enumerate(unseen_rooms()):
        if PARTIAL and not (SCR / "un" / "{}__{:02d}.npz".format(name, j)).exists():
            continue
        z = np.load(SCR / "un" / "{}__{:02d}.npz".format(name, j))
        h = read_h5("data/track_p4_8_unseen/" + c.filename)
        v = band_r(z["P"], h["T"], h["rx"])
        for b in B3:
            uf[b].append(v[b])
    return {"test": tr, "uf": uf}


def e3_d1(name):
    """E3's 63 Hz ratio and D1's floor / EDT per room (in-family), from the per-room files"""
    from aaf.eval.octave_decay import decay_metrics, valid_mask
    from scripts.p4_10_e3_floor import ratio
    e = {"model": [], "truth": []}
    d = {b: {"EDT_err": [], "floor": []} for b in ("63", "125", "250")}
    for i, r, z, h in rows_in(name):
        P, T, rx = np.asarray(z["P"]), h["T"], h["rx"]
        far = np.hypot(rx[:, 0] - 0.5, rx[:, 1] - 1.6) >= 1.0
        s = ~on_lattice_(rx) & far
        e["model"].append(ratio(P[s], "63")[0])
        e["truth"].append(ratio(T[s], "63")[0])
        k = np.nonzero(s)[0][::4]
        for b in d:
            t, m = decay_metrics(T[k], b), decay_metrics(P[k], b)
            ok = valid_mask(t["t35"])
            d[b]["EDT_err"].append(float(np.nanmedian(np.abs(m["EDT"][ok] - t["EDT"][ok]) / np.maximum(t["EDT"][ok], 1e-6))))
            d[b]["floor"].append(float(np.nanmedian(m["floor_db"][ok])))
    return e, d


def res_rows(name):
    from scripts.p4_10_0d_meanvalue import residuals
    out = {"raw": [], "proj": [], "truth": [], "train": []}
    for i, r, z, h in rows_in(name):
        t = np.load("outputs/p4_10/r1p_0d/fdtd/{}.npz".format(r["path"].stem))
        out["truth"].append(residuals(t["H"], t["sets"]))
        out["raw"].append(residuals(z["raw"], z["sets"]))
        out["proj"].append(residuals(z["proj"], z["sets"]))
    z = np.load(SCR / "tr" / (name + ".npz"))
    out["train"] = [residuals(z["tr__{:02d}".format(i)], z["sets"]) for i in range(N_TRAIN_ROOMS)]
    return {k: {b: [x[b]["single"] for x in v] for b in B3} for k, v in out.items()}


def dir_rows(name):
    from scripts.p4_9_dir_zeroshot import rho
    k = 2 * np.pi * np.arange(601) * 0.5 / 343.0
    jk = 1j * k
    jk[0] = 1.0
    out = {b: [] for b in B3}
    for i, r, z, h in rows_in(name):
        st = np.load("outputs/p4_9/fdtd_stencils/dense/{}.npz".format(r["path"].stem))
        T2 = np.stack([st["g2x"], st["g2y"]], axis=1)[z["ok"]] / jk
        rr = rho(z["G"] / jk, T2)
        for b, (lo, hi) in BAND.items():
            out[b].append(float(np.mean(rr.real[lo:hi])))
    return out


def score(stepk):
    step = STEPS[stepk]
    names = {a: "{}_{}".format(a, stepk) for a in ARMS}
    R = {"step": step, "reads": {}}
    L = ["# P-1 read at +{} steps (R1 @150K + {} ; R1-P(t) against R1-ctl at the same step)".format(step - 150000, step - 150000), ""]
    sa, sc = static_rows(names["R1PT"]), static_rows(names["R1CTL"])
    stop = []
    hf = {"pt": float(np.mean(sa["test"]["200-300"])), "ctl": float(np.mean(sc["test"]["200-300"])), "delta": boot(np.array(sa["test"]["200-300"]) - np.array(sc["test"]["200-300"]))}
    uf = {b: {"pt": float(np.mean(sa["uf"][b])), "ctl": float(np.mean(sc["uf"][b])), "delta": boot(np.array(sa["uf"][b]) - np.array(sc["uf"][b]))} for b in B3}
    R["reads"]["1_static"] = {"HF": hf, "UF": uf, "test_r": {b: {"pt": float(np.mean(sa["test"][b])), "ctl": float(np.mean(sc["test"][b]))} for b in B3}}
    L.append("1. Static: HF {:.3f} vs {:.3f}, {:+.4f} [{:+.4f}, {:+.4f}]; ".format(hf["pt"], hf["ctl"], *hf["delta"]) + "; ".join("UF {} {:.3f} vs {:.3f}, {:+.4f} [{:+.4f}, {:+.4f}]".format(b, uf[b]["pt"], uf[b]["ctl"], *uf[b]["delta"]) for b in B3))
    for nm, x in [("HF", hf["delta"])] + [("UF_" + b, uf[b]["delta"]) for b in B3]:
        if x[0] < -0.02 and x[2] < 0.0:
            stop.append(nm)
    R["stop_rule_triggered"] = stop
    P, C = res_rows(names["R1PT"]), res_rows(names["R1CTL"])
    rr = {}
    for b in B3:
        rr[b] = {"truth": float(np.mean(P["truth"][b])), "pt_raw": boot(P["raw"][b]), "ctl_raw": boot(C["raw"][b]), "pt_minus_ctl": boot(np.array(P["raw"][b]) - np.array(C["raw"][b])),
                 "pt_proj": boot(P["proj"][b]), "ctl_proj": boot(C["proj"][b]), "pt_bl_over_truth": float(np.mean(P["proj"][b]) / np.mean(P["truth"][b])), "ctl_bl_over_truth": float(np.mean(C["proj"][b]) / np.mean(C["truth"][b])),
                 "pt_raw_over_truth": float(np.mean(P["raw"][b]) / np.mean(P["truth"][b])), "ctl_raw_over_truth": float(np.mean(C["raw"][b]) / np.mean(C["truth"][b]))}
        L.append("2. Mean-value residual {} Hz (single radius, held-out centres, 15 rooms): truth {:.4f}; raw R1-P(t) {:.4f} vs R1-ctl {:.4f} (difference {:+.4f} [{:+.4f}, {:+.4f}]); raw over truth {:.2f} vs {:.2f}".format(
            b, rr[b]["truth"], rr[b]["pt_raw"][0], rr[b]["ctl_raw"][0], *rr[b]["pt_minus_ctl"], rr[b]["pt_raw_over_truth"], rr[b]["ctl_raw_over_truth"]))
    L.append("3. Band-limited residual over the truth's at 25-100 Hz: R1-P(t) {:.2f}, R1-ctl {:.2f} (100-200: {:.2f} / {:.2f}; 200-300: {:.2f} / {:.2f}; 0d for R0 was 2.78 at 25-100)".format(
        rr["25-100"]["pt_bl_over_truth"], rr["25-100"]["ctl_bl_over_truth"], rr["100-200"]["pt_bl_over_truth"], rr["100-200"]["ctl_bl_over_truth"], rr["200-300"]["pt_bl_over_truth"], rr["200-300"]["ctl_bl_over_truth"]))
    ed = {n: e3_d1(names[n]) for n in ARMS}
    e = {n: ed[n][0] for n in ARMS}
    e4 = {"pt": boot(np.array(e["R1PT"]["model"]) / np.array(e["R1PT"]["truth"])), "ctl": boot(np.array(e["R1CTL"]["model"]) / np.array(e["R1CTL"]["truth"])),
          "pt_minus_ctl_ratio": boot(np.array(e["R1PT"]["model"]) - np.array(e["R1CTL"]["model"])), "pt_level": boot(e["R1PT"]["model"]), "ctl_level": boot(e["R1CTL"]["model"]), "truth_level": boot(e["R1PT"]["truth"]),
          "pt_over_truth_of_means": float(np.mean(e["R1PT"]["model"]) / np.mean(e["R1PT"]["truth"])), "ctl_over_truth_of_means": float(np.mean(e["R1CTL"]["model"]) / np.mean(e["R1CTL"]["truth"]))}
    R["reads"]["2_3_residual"], R["reads"]["4_e3_63hz"] = rr, e4
    L.append("4. Late floor common to the room at 63 Hz (E3): level R1-P(t) {:.4f} vs R1-ctl {:.4f} (truth {:.4f}; difference {:+.4f} [{:+.4f}, {:+.4f}]); model / truth as a ratio of the means (the published E3 form; R1 2.4) {:.2f} vs {:.2f}; mean of room ratios {:.2f} vs {:.2f}".format(
        e4["pt_level"][0], e4["ctl_level"][0], e4["truth_level"][0], *e4["pt_minus_ctl_ratio"], e4["pt_over_truth_of_means"], e4["ctl_over_truth_of_means"], e4["pt"][0], e4["ctl"][0]))
    dd = {n: ed[n][1] for n in ARMS}
    d1 = {}
    for b in ("63", "125", "250"):
        d1[b] = {k: {"pt": float(np.mean(dd["R1PT"][b][k])), "ctl": float(np.mean(dd["R1CTL"][b][k])), "delta": boot(np.array(dd["R1PT"][b][k]) - np.array(dd["R1CTL"][b][k]))} for k in ("EDT_err", "floor")}
        L.append("   D1 {} Hz: EDT error {:.1%} vs {:.1%} (difference {:+.2%} [{:+.2%}, {:+.2%}]); floor {:.1f} vs {:.1f} dB (difference {:+.2f} [{:+.2f}, {:+.2f}])".format(
            b, d1[b]["EDT_err"]["pt"], d1[b]["EDT_err"]["ctl"], *d1[b]["EDT_err"]["delta"], d1[b]["floor"]["pt"], d1[b]["floor"]["ctl"], *d1[b]["floor"]["delta"]))
    R["reads"]["4_d1"] = d1
    dp = {n: dir_rows(names[n]) for n in ARMS}
    r5 = {b: {"pt": boot(dp["R1PT"][b]), "ctl": boot(dp["R1CTL"][b]), "delta": boot(np.array(dp["R1PT"][b]) - np.array(dp["R1CTL"][b]))} for b in B3}
    R["reads"]["5_direction_raw_fd"] = r5
    L.append("5. Raw finite-difference direction (in-family, h = 4 cm, Re rho): " + "; ".join("{} Hz {:.3f} vs {:.3f} ({:+.4f} [{:+.4f}, {:+.4f}])".format(b, r5[b]["pt"][0], r5[b]["ctl"][0], *r5[b]["delta"]) for b in B3))
    by = {}
    for b in B3:
        by[b] = {"pt_ratio_heldout_over_train": float(np.mean(P["raw"][b]) / np.mean(P["train"][b])), "ctl_ratio": float(np.mean(C["raw"][b]) / np.mean(C["train"][b])), "pt_train": float(np.mean(P["train"][b])), "ctl_train": float(np.mean(C["train"][b]))}
    R["reads"]["6_bypass"] = by
    L.append("6. Bypass check (held-out test-room centres / training-rule centres in 15 training rooms; <= 1.5x): " + "; ".join("{} Hz R1-P(t) {:.2f} (R1-ctl {:.2f})".format(b, by[b]["pt_ratio_heldout_over_train"], by[b]["ctl_ratio"]) for b in B3))
    L.append("")
    L.append("Stop rule (HF or any UF band more than 0.02 below R1-ctl with the CI entirely below zero): **{}** {}".format("TRIGGERED" if stop else "not triggered", stop or ""))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "read_{}.json".format(stepk)).write_text(json.dumps(R, indent=1))
    (OUT / "read_{}.md".format(stepk)).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None)
    ap.add_argument("--kind", default="in")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--score", default=None)
    a = ap.parse_args()
    if a.task:
        task(*a.task.split("=", 1), a.kind, a.idx)
    elif a.score:
        score(a.score)
