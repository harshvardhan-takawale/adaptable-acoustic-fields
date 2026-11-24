"""P4-10 E1 (D206, D207): decay, fit or transfer? D1 and D2 on (a) R1 in 15 GX training rooms + the 2 S-1 rooms, dense off-lattice truth simulated anew; (b) the S-1
single-room models (R0's recipe, one room, 10K steps), each in its own room. The source is the room's own training source. Never the training receivers.
  --draw           fix the rooms: 15 GX training rooms by default_rng(20261126) over the 998 rows that are not S-1 rooms, sorted by manifest index, plus the 2 S-1 rooms -> configs/p4_10/e1_rooms.json
  --fdtd --idx I   CPU: dense truth of room I (exactness: 50 stored training receivers of the room must reproduce to <= 1e-5 relative) -> outputs/p4_10/e1/truth/<stem>.npz
  --render --idx I GPU: the models for room I (R1_150K everywhere; S1lo / S1med only in their own rooms) -> outputs/p4_10/e1/pred/<model>__<stem>.npz
  --score          CPU: outputs/p4_10/e1/report.{json,md}; the 15 in-family test rooms of R1 (D1 / D2) printed beside
Readings (D206 + D207, recorded before any E1 number): width ratio = median over the room's matched peaks of model / truth half-power width. S-1 at "truth width" = room ratio <= 1.15; R1 training rooms "still
broad" = mean room ratio >= 1.30; between 1.15 and 1.30 = intermediate, reported as measured. If S-1 is at truth width and R1 is broad, the broadening comes from sharing one model (Q25); if S-1 is broad too, from loss or representation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.p4_10_scorecard import Renderer, boot, on_lattice
from scripts.p4_10_truth import DX, dense_points, record, snap, solver_room

GX = "configs/sweeps_2d_mat/p4_7_c500_cx500_manifest.json"
OUT = Path("outputs/p4_10/e1")
ROOMS_F = Path("configs/p4_10/e1_rooms.json")
TRAIN_DATA = "data/track_p4_7_mix"
S1 = {"lo": ("L6.16_W4.56_d0.34_w0.48.h5", "outputs/p4_9/arms/S1lo/ckpt_iter0010000.pt"),
      "med": ("L5.74_W4.78_d0.62_w2.32.h5", "outputs/p4_9/arms/S1med/ckpt_iter0010000.pt")}
MODELS = {"R1_150K": "outputs/p4_10/arms/R1/ckpt_iter0150000.pt"}
SEED, N_DRAW = 20261126, 15
BANDS3 = ("63", "125", "250")


def draw():
    g = json.load(open(GX))["configs"]
    s1 = {v[0] for v in S1.values()}
    pool = [i for i, r in enumerate(g) if r["filename"] not in s1]
    pick = sorted(np.random.default_rng(SEED).choice(pool, N_DRAW, replace=False).tolist())
    rows = [g[i] for i in pick]
    rows += [r for r in g if r["filename"] in s1]
    out = {"seed": SEED, "source": GX, "n_random": N_DRAW, "s1_rooms": {k: v[0] for k, v in S1.items()},
           "rooms": [{"filename": r["filename"], "verts": r["verts"], "src": r["src"], "origin": r.get("origin"), "s1": next((k for k, v in S1.items() if v[0] == r["filename"]), None)} for r in rows]}
    ROOMS_F.write_text(json.dumps(out, indent=1))
    print("rooms", len(out["rooms"]), [x["filename"][:22] for x in out["rooms"]])


def room(i):
    return json.loads(ROOMS_F.read_text())["rooms"][i]


def fdtd(i):
    r = room(i)
    V = [tuple(map(float, v)) for v in r["verts"]]
    L, W = max(v[0] for v in V), max(v[1] for v in V)
    src = tuple(float(x) for x in r["src"])
    extra = solver_room(L, W, V, [0.15] * len(V))
    with h5py.File(Path(TRAIN_DATA) / r["filename"], "r") as f:
        rx0 = np.asarray(json.loads(f.attrs["receiver_pos"]), float)
        s0 = tuple(float(x) for x in json.loads(f.attrs["source_pos"]))
        H0 = np.asarray(f["ism/H_complex"])[:, :601]
    assert np.allclose(s0, src, atol=1e-6), (s0, src)
    chk = np.linspace(0, len(rx0) - 1, 50).round().astype(int)
    Hc = record(L, W, src, rx0[chk], extra)
    err = float(np.abs(Hc - H0[chk]).max() / np.abs(H0[chk]).max())
    assert err < 1e-5, err
    d, _keep = dense_points(L, W, V)
    pts = snap(d)
    H = record(L, W, src, pts, extra)
    OUT.joinpath("truth").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "truth" / (Path(r["filename"]).stem + ".npz"), rx=pts, H=H.astype(np.complex64), src=np.array(src), L=L, W=W, verts=np.array(V), check_err=err)
    print("truth", r["filename"], len(pts), "rx, check err {:.1e}".format(err), flush=True)


def render(i):
    r = room(i)
    t = np.load(OUT / "truth" / (Path(r["filename"]).stem + ".npz"))
    V = [tuple(map(float, v)) for v in t["verts"]]
    (OUT / "pred").mkdir(parents=True, exist_ok=True)
    todo = dict(MODELS)
    if r["s1"]:
        todo["S1" + r["s1"]] = S1[r["s1"]][1]
    for name, ck in todo.items():
        dst = OUT / "pred" / "{}__{}.npz".format(name, Path(r["filename"]).stem)
        if dst.exists():
            continue
        P = Renderer(name, ck).render(V, float(t["L"]), float(t["W"]), [0.15] * len(V), tuple(t["src"]), t["rx"])
        np.savez_compressed(dst, P=P)
        print("rendered", name, r["filename"], len(P), flush=True)


def analyse(P, T, rx, src):
    """D1 (per octave: EDT error / bias, floor) and D2 (widths) of one room; receivers: off-lattice, >= 1 m from the source"""
    from aaf.eval.octave_decay import decay_metrics, floor_mask, valid_mask
    from scripts.p4_10_modes import half_width, hungarian, peaks, spec
    far = np.hypot(rx[:, 0] - src[0], rx[:, 1] - src[1]) >= 1.0
    off = ~on_lattice(rx)
    sel = np.nonzero(off & far)[0]
    out = {"n_rx": int(len(sel)), "D1": {}}
    for b in BANDS3:
        i4 = sel[::4]
        t, m = decay_metrics(T[i4], b), decay_metrics(P[i4], b)
        ok = valid_mask(t["t35"])
        kp = ok & floor_mask(m)
        rel = lambda x, y, k: float(np.nanmedian(np.abs(x[k] - y[k]) / np.maximum(y[k], 1e-6))) if k.any() else np.nan  # noqa: E731
        out["D1"][b] = {"n_valid": int(ok.sum()), "n_t20_kept": int(kp.sum()), "EDT_truth": float(np.nanmedian(t["EDT"][ok])) if ok.any() else np.nan,
                        "EDT_model": float(np.nanmedian(m["EDT"][ok])) if ok.any() else np.nan, "EDT_err": rel(m["EDT"], t["EDT"], ok),
                        "EDT_bias": float(np.nanmedian((m["EDT"][ok] - t["EDT"][ok]) / np.maximum(t["EDT"][ok], 1e-6))) if ok.any() else np.nan,
                        "floor_model": float(np.nanmedian(m["floor_db"][ok])) if ok.any() else np.nan, "floor_truth": float(np.nanmedian(t["floor_db"][ok])) if ok.any() else np.nan}
    St, Sm = spec(T[off]), spec(P[off])
    ft, it = peaks(St, 25.0, 100.0)
    fm, im = peaks(Sm, 20.0, 110.0, n=40)
    rr, wt, wm = [], [], []
    pairs = hungarian(ft, fm, 3.0)
    for a, b in pairs:
        w1, w2 = half_width(St, it[a]), half_width(Sm, im[b])
        if np.isfinite(w1) and np.isfinite(w2):
            rr.append(w2 / w1)
            wt.append(w1)
            wm.append(w2)
    out["D2"] = {"truth_peaks": int(len(ft)), "matched": len(pairs), "used": len(rr), "ratio_median": float(np.median(rr)) if rr else np.nan,
                 "truth_width": float(np.median(wt)) if wt else np.nan, "model_width": float(np.median(wm)) if wm else np.nan}
    return out


def score():
    rr = json.loads(ROOMS_F.read_text())["rooms"]
    rep = {"rooms": [], "groups": {}}
    for r in rr:
        stem = Path(r["filename"]).stem
        tz = np.load(OUT / "truth" / (stem + ".npz"))
        rx, T, src = tz["rx"], tz["H"], tz["src"]
        row = {"room": stem, "s1": r["s1"], "check_err": float(tz["check_err"]), "models": {}}
        for f in sorted((OUT / "pred").glob("*__{}.npz".format(stem))):
            name = f.name.split("__")[0]
            row["models"][name] = analyse(np.load(f)["P"], T, rx, src)
        rep["rooms"].append(row)
    groups = {"R1 in 15 random training rooms": ("R1_150K", lambda r: not r["s1"]), "R1 in the 2 S-1 rooms": ("R1_150K", lambda r: bool(r["s1"])), "S-1 models in their own rooms": (None, lambda r: bool(r["s1"]))}
    L = ["# P4-10 E1: decay, fit or transfer?", "", "Dense off-lattice truth; receivers >= 1 m from the room's own source. Per group: mean over rooms (95% CI over rooms when >= 3 rooms).", "",
         "| group | rooms | width ratio model/truth | peaks matched / truth | EDT error 63 / 125 / 250 Hz (%) | floor model / truth dB (63 / 125 / 250) |", "|---|---|---|---|---|---|"]
    for lab, (name, pick) in groups.items():
        rows = [x for x in rep["rooms"] if pick(x)]
        pairs = [(x, x["models"].get(name) if name else next((v for k, v in x["models"].items() if k.startswith("S1")), None)) for x in rows]
        pairs = [(x, m) for x, m in pairs if m]
        if not pairs:
            continue
        w = [m["D2"]["ratio_median"] for _, m in pairs]
        b3 = {b: [m["D1"][b]["EDT_err"] for _, m in pairs] for b in BANDS3}
        e = {"n_rooms": len(pairs), "width_ratio": boot(w) if len(w) >= 3 else [float(np.mean(w)), float("nan"), float("nan")], "width_per_room": w,
             "matched": int(sum(m["D2"]["matched"] for _, m in pairs)), "truth_peaks": int(sum(m["D2"]["truth_peaks"] for _, m in pairs)), "EDT_err": {b: (boot(v) if len(v) >= 3 else [float(np.mean(v)), float("nan"), float("nan")]) for b, v in b3.items()},
             "floor_model": {b: float(np.mean([m["D1"][b]["floor_model"] for _, m in pairs])) for b in BANDS3}, "floor_truth": {b: float(np.mean([m["D1"][b]["floor_truth"] for _, m in pairs])) for b in BANDS3}}
        rep["groups"][lab] = e
        L.append("| {} | {} | {:.2f} [{:.2f}, {:.2f}] | {} / {} | {} | {} |".format(lab, e["n_rooms"], *e["width_ratio"], e["matched"], e["truth_peaks"],
                                                                                 " / ".join("{:.0%}".format(e["EDT_err"][b][0]) for b in BANDS3), " / ".join("{:.0f}/{:.0f}".format(e["floor_model"][b], e["floor_truth"][b]) for b in BANDS3)))
    ti = json.loads(Path("outputs/p4_10/modes/d2.json").read_text()).get("R1_150K")
    di = json.loads(Path("outputs/p4_10/d1/report.json").read_text())["groups"].get("R1_150K", {}).get("test", {})
    if ti and di:
        L.append("| R1 on the 15 in-family TEST rooms (D1 / D2 above) | {} | {:.2f} | {} / {} | {} | {} |".format(ti["n_rooms"], ti["median_ratio"], ti["matched"], ti["truth_peaks"], " / ".join("{:.0%}".format(di[b]["EDT_err"][0]) for b in BANDS3),
                                                                                                   " / ".join("{:.0f}/{:.0f}".format(di[b]["floor_db_model"][0], di[b]["floor_db_truth"][0]) for b in BANDS3)))
    L += ["", "Per room (S-1): " + "; ".join("{} {}: S-1 width ratio {:.2f}, R1 {:.2f}".format(x["room"], x["s1"], x["models"]["S1" + x["s1"]]["D2"]["ratio_median"], x["models"]["R1_150K"]["D2"]["ratio_median"])
                                         for x in rep["rooms"] if x["s1"] and "S1" + x["s1"] in x["models"] and "R1_150K" in x["models"])]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("--draw", "--fdtd", "--render", "--score"):
        ap.add_argument(k, action="store_true")
    ap.add_argument("--idx", type=int, default=0)
    a = ap.parse_args()
    if a.draw:
        draw()
    elif a.fdtd:
        fdtd(a.idx)
    elif a.render:
        render(a.idx)
    elif a.score:
        score()
