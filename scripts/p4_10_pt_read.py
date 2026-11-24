"""P4-10 P-1 reads (D217 / D218): R1-P(t) against R1-ctl at the same step. Arms live in outputs/p4_10/arms_pt/{R1PT,R1CTL}; NAME = ARM_stepK (152K 155K 157K 160K).
  --extra NAME=CKPT       GPU, after the dense job: 0d residual (raw; band-limited from the dense lattice), bypass set -> scratch pt/extra/NAME.npz
  --dir NAME=CKPT --idx I GPU: raw omni-FD gradient (h = 2 nodes) of in-family room I -> scratch pt/dir/NAME__I.npz
  --score STEPK           CPU: the six reads, paired over the 15 rooms (bootstrap 10k, seed 20261013) -> outputs/p4_10/pt/read_STEP.{json,md}
Static HF / UF come from the standard report (p4_5_gate2_dense + speckle + p4_8_unseen_eval) through p4_8_primary_table.load / paired.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCR = Path("/fs/nexus-scratch/htakawal/aaf_p410/pt")
OUT = Path("outputs/p4_10/pt")
CK = "outputs/p4_10/arms_pt/{}/ckpt_iter{:07d}.pt"
ARMS = ("R1PT", "R1CTL")
STEPS = {"152K": 152500, "155K": 155000, "157K": 157500, "160K": 160000}
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


def extra(name, ck):
    import h5py

    from scripts.p4_10_0d_meanvalue import band_limited_at, distinct_sets, room_points
    torch, dev, model, renderer = load_model(ck)
    lat = np.load("outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz".format(name))
    out = {}
    for r in test_rooms():
        st = np.load("outputs/p4_10/r1p_0d/fdtd/{}.npz".format(r["path"].stem))
        P = st["P"]
        with h5py.File(r["path"]) as f:
            verts = [tuple(map(float, v)) for v in json.loads(f.attrs["verts"])]
            L, W = float(f.attrs["L"]), float(f.attrs["W"])
            keep = np.asarray(json.loads(f.attrs["lattice_keep"]), bool).reshape(64, 64)
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]), float)
        M = P.reshape(-1, 2)
        raw = render_pts(torch, dev, model, renderer, verts, L, W, (0.5, 1.6), M)
        proj = band_limited_at(lat["P%d" % r["cache_i"]][:, :601], keep, L, W, rx, M)
        out["raw__" + r["path"].stem] = raw.reshape(P.shape[:3] + (601,)).astype(np.complex64)
        out["proj__" + r["path"].stem] = proj.reshape(P.shape[:3] + (601,)).astype(np.complex64)
        out["sets"] = st["sets"]
        print("extra", name, r["path"].stem, flush=True)
    sets = distinct_sets()
    for i, c in enumerate(train_rooms()):
        verts = [tuple(map(float, v)) for v in c["verts"]]
        L, W = max(v[0] for v in verts), max(v[1] for v in verts)
        P = room_points({"L": L, "W": W, "verts": verts, "src": c["src"]}, 1000 + i, sets)
        out["tr__{:02d}".format(i)] = render_pts(torch, dev, model, renderer, verts, L, W, tuple(c["src"]), P.reshape(-1, 2)).reshape(P.shape[:3] + (601,)).astype(np.complex64)
    out["tr_files"] = np.array([c["filename"] for c in train_rooms()])
    SCR.joinpath("extra").mkdir(parents=True, exist_ok=True)
    tmp = SCR / "extra" / (name + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(SCR / "extra" / (name + ".npz"))


def dir_room(name, ck, idx):
    import h5py
    torch, dev, model, renderer = load_model(ck)
    r = test_rooms()[idx]
    st = np.load("outputs/p4_9/fdtd_stencils/dense/{}.npz".format(r["path"].stem))
    with h5py.File(r["path"]) as f:
        rx = np.asarray(json.loads(f.attrs["receiver_pos"]), float)
        verts = [tuple(map(float, v)) for v in json.loads(f.attrs["verts"])]
        L, W = float(f.attrs["L"]), float(f.attrs["W"])
    hx, hy = 2 * float(st["dxx"]), 2 * float(st["dxy"])
    ok = np.nonzero((~on_lattice_(rx)) & (np.hypot(*(rx - np.array([0.5, 1.6])).T) >= 0.5))[0]
    nb = [render_pts(torch, dev, model, renderer, verts, L, W, (0.5, 1.6), rx[ok] + d) for d in ([hx, 0], [-hx, 0], [0, hy], [0, -hy])]
    G = np.stack([(nb[0] - nb[1]) / (2 * hx), (nb[2] - nb[3]) / (2 * hy)], axis=1)
    SCR.joinpath("dir").mkdir(parents=True, exist_ok=True)
    tmp = SCR / "dir" / "{}__{:02d}.tmp.npz".format(name, idx)
    np.savez_compressed(tmp, G=G.astype(np.complex64), ok=ok, hx=hx, hy=hy)
    tmp.replace(SCR / "dir" / "{}__{:02d}.npz".format(name, idx))
    print("dir", name, r["path"].stem, len(ok), flush=True)


def on_lattice_(rx):
    k = (np.asarray(rx) - 0.12) / 0.06
    return np.all(np.abs(k - np.round(k)) < 1e-6, axis=1)


def e3_ratio(cache_name):
    from scripts.p4_10_d1_decay import rooms_of
    from scripts.p4_10_e3_floor import ratio
    out = {"model": [], "truth": []}
    for P, T, rx in rooms_of(cache_name, "test"):
        sel = ~on_lattice_(rx) & (np.hypot(rx[:, 0] - 0.5, rx[:, 1] - 1.6) >= 1.0)
        out["model"].append(ratio(np.asarray(P)[sel], "63")[0])
        out["truth"].append(ratio(np.asarray(T)[sel], "63")[0])
    return out


def d1_rows(cache_name):
    """per room and octave: EDT error (median), floor (dB) -- D1's definition (in-family)"""
    from aaf.eval.octave_decay import decay_metrics, valid_mask
    from scripts.p4_10_d1_decay import rooms_of
    out = {b: {"EDT_err": [], "floor": [], "EDT_bias": []} for b in ("63", "125", "250")}
    for P, T, rx in rooms_of(cache_name, "test"):
        sel = np.nonzero(~on_lattice_(rx) & (np.hypot(rx[:, 0] - 0.5, rx[:, 1] - 1.6) >= 1.0))[0][::4]
        for b in out:
            t, m = decay_metrics(np.asarray(T)[sel], b), decay_metrics(np.asarray(P)[sel], b)
            ok = valid_mask(t["t35"])
            rel = (m["EDT"][ok] - t["EDT"][ok]) / np.maximum(t["EDT"][ok], 1e-6)
            out[b]["EDT_err"].append(float(np.nanmedian(np.abs(rel))))
            out[b]["EDT_bias"].append(float(np.nanmedian(rel)))
            out[b]["floor"].append(float(np.nanmedian(m["floor_db"][ok])))
    return out


def res_rows(name):
    from scripts.p4_10_0d_meanvalue import residuals
    z = np.load(SCR / "extra" / (name + ".npz"))
    sets = z["sets"]
    out = {"raw": [], "proj": [], "truth": [], "train": []}
    for r in test_rooms():
        t = np.load("outputs/p4_10/r1p_0d/fdtd/{}.npz".format(r["path"].stem))
        out["truth"].append(residuals(t["H"], sets))
        out["raw"].append(residuals(z["raw__" + r["path"].stem], sets))
        out["proj"].append(residuals(z["proj__" + r["path"].stem], sets))
    out["train"] = [residuals(z["tr__{:02d}".format(i)], sets) for i in range(N_TRAIN_ROOMS)]
    return {k: {b: [x[b]["single"] for x in v] for b in B3} for k, v in out.items()}


def dir_rows(name):
    from scripts.p4_9_dir_zeroshot import rho
    k = 2 * np.pi * np.arange(601) * 0.5 / 343.0
    jk = 1j * k
    jk[0] = 1.0
    out = {b: [] for b in B3}
    for i, r in enumerate(test_rooms()):
        st = np.load("outputs/p4_9/fdtd_stencils/dense/{}.npz".format(r["path"].stem))
        z = np.load(SCR / "dir" / "{}__{:02d}.npz".format(name, i))
        ok = z["ok"]
        T2 = np.stack([st["g2x"], st["g2y"]], axis=1)[ok] / jk
        rr = rho(z["G"] / jk, T2)
        for b, (lo, hi) in BAND.items():
            out[b].append(float(np.mean(rr.real[lo:hi])))
    return out


def score(stepk):
    from scripts.p4_8_primary_table import load, paired
    step = STEPS[stepk]
    names = {a: "{}_{}".format(a, stepk) for a in ARMS}
    R = {"step": step, "reads": {}}
    L = ["# P-1 read at +{} steps (R1 @150K + {} ; R1-P(t) against R1-ctl at the same step)".format(step - 150000, step - 150000), ""]
    a, c = load(names["R1PT"]), load(names["R1CTL"])
    if a is None or c is None:
        raise SystemExit("static report missing")
    d = paired(a, c)
    stop = []
    hf = {"pt": a["test_r"]["200-300"], "ctl": c["test_r"]["200-300"], "delta": d["HF"]}
    uf = {b: {"pt": a["uf"][b], "ctl": c["uf"][b], "delta": d["UF_" + b]} for b in B3}
    R["reads"]["1_static"] = {"HF": hf, "UF": uf}
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
    e = {n: e3_ratio(names[n]) for n in ARMS}
    e4 = {"pt": boot(np.array(e["R1PT"]["model"]) / np.array(e["R1PT"]["truth"])), "ctl": boot(np.array(e["R1CTL"]["model"]) / np.array(e["R1CTL"]["truth"])),
          "pt_minus_ctl_ratio": boot(np.array(e["R1PT"]["model"]) - np.array(e["R1CTL"]["model"])), "pt_level": boot(e["R1PT"]["model"]), "ctl_level": boot(e["R1CTL"]["model"]), "truth_level": boot(e["R1PT"]["truth"])}
    R["reads"]["2_3_residual"], R["reads"]["4_e3_63hz"] = rr, e4
    L.append("4. Late floor common to the room at 63 Hz (E3): level R1-P(t) {:.4f} vs R1-ctl {:.4f} (truth {:.4f}; difference {:+.4f} [{:+.4f}, {:+.4f}]); mean of room ratios model/truth {:.2f} vs {:.2f}".format(
        e4["pt_level"][0], e4["ctl_level"][0], e4["truth_level"][0], *e4["pt_minus_ctl_ratio"], e4["pt"][0], e4["ctl"][0]))
    dd = {n: d1_rows(names[n]) for n in ARMS}
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
    ap.add_argument("--extra", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--score", default=None)
    a = ap.parse_args()
    if a.extra:
        extra(*a.extra.split("=", 1))
    elif a.dir:
        dir_room(*a.dir.split("=", 1), a.idx)
    elif a.score:
        score(a.score)
