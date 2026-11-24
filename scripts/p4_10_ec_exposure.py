"""P4-10 E-C (D211, D215): how much per-room training fixes the decay. Two S-1 rooms; trajectories (a) shared models, (b) single room from scratch, (c) R1 @150K fine-tuned on the room.
  --fdtd --idx I            CPU: truth of room I at Stretch-U s = 1.10 -> outputs/p4_10/ec/truth/<tag>_s110.npz
  --train --idx I           GPU: 0 = b lo, 1 = b med, 2 = c lo, 3 = c med (drives P32Trainer's own step; the trainer file is not modified)
  --render --idx I          GPU: render job I (see jobs())
  --score                   CPU: outputs/p4_10/ec/report.{md,json}
  --fig                     outputs/p4_10/meeting/EC_exposure.{png,pdf,json}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path("outputs/p4_10/ec")
E1T = Path("outputs/p4_10/e1/truth")
E1P = Path("outputs/p4_10/e1/pred")
ROOMS = {"lo": "L6.16_W4.56_d0.34_w0.48", "med": "L5.74_W4.78_d0.62_w2.32"}
STEPS = (30, 60, 120, 250, 500, 1000, 2000)
LR_C = 5.0e-5
SRC = (0.5, 1.6)
R1 = "outputs/p4_10/arms/R1/ckpt_iter0150000.pt"
FIT_R = 0.80                                  # D215 addendum: a point is a FIT only if static r at 25-100 Hz >= 0.80
ROWS_FORMULA = "rows per room = steps x [grad_accum_steps x (batch_size // rx_per_config) x rx_per_config] / n_rooms, from each run's yaml and manifest"


def rows_per_step(yaml_path):
    import yaml
    y = yaml.safe_load(open(yaml_path))
    return int(y["grad_accum_steps"]) * (int(y["batch_size"]) // int(y["rx_per_config"])) * int(y["rx_per_config"])


def n_rooms(yaml_path):
    import yaml
    y = yaml.safe_load(open(yaml_path))
    return len([r for r in json.load(open(y["configs_manifest"]))["configs"] if r.get("split", "train") == "train"])


CFG = {"R0": "configs/sweep_2d_mat/P4_8_R0.yaml", "CTL": "configs/sweep_2d_mat/P4_9_CTL.yaml", "S2X": "configs/sweep_2d_mat/P4_9_S2X.yaml", "R1": "configs/sweep_2d_mat/P4_10_R1.yaml"}


def rows(model, steps):
    return steps * rows_per_step(CFG[model]) / n_rooms(CFG[model])


def points():
    """every point: dict(traj, label, steps, rows, ck, tag or None, name)"""
    P = []
    for k in (10, 20, 30, 60, 90, 120):
        P.append(dict(traj="a", model="R0", label="R0 @{}K".format(k), steps=k * 1000, rows=rows("R0", k * 1000), ck="outputs/p4_8/arms/R0/ckpt_iter{:07d}.pt".format(k * 1000), tag=None))
    P.append(dict(traj="a", model="CTL", label="CTL @150K", steps=150000, rows=rows("CTL", 150000), ck="outputs/p4_9/arms/CTL/ckpt_iter0150000.pt", tag=None))
    P.append(dict(traj="a", model="S2X", label="S2X @240K (2000 rooms)", steps=240000, rows=rows("S2X", 240000), ck="outputs/p4_9/arms/S2X/ckpt_iter0240000.pt", tag=None))
    r1 = rows("R1", 150000)
    P.append(dict(traj="a", model="R1", label="R1 @150K", steps=150000, rows=r1, ck=R1, tag=None, rows_own_source=rows("R1", 120000) + rows("R1", 30000) / 4.0))
    rps = rows_per_step(CFG["R0"])
    for tag in ROOMS:
        for s in STEPS:
            P.append(dict(traj="b", model="S-1 re-run", label="{} steps".format(s), steps=s, rows=float(s * rps), ck="{}/arms/b_{}/ckpt_iter{:07d}.pt".format(OUT, tag, s), tag=tag))
        for s in (4000, 10000):
            P.append(dict(traj="b", model="S-1", label="{} steps (original S-1)".format(s), steps=s, rows=float(s * rps), ck="outputs/p4_9/arms/S1{}/ckpt_iter{:07d}.pt".format(tag, s), tag=tag))
        P.append(dict(traj="c", model="R1", label="0 steps (R1 @150K)", steps=0, rows=r1, ck=R1, tag=tag, ft_rows=0.0))
        for s in STEPS:
            P.append(dict(traj="c", model="R1 fine-tuned", label="{} steps".format(s), steps=s, rows=r1 + s * rps, ck="{}/arms/c_{}/ckpt_iter{:07d}.pt".format(OUT, tag, s), tag=tag, ft_rows=float(s * rps)))
    return P


def save_npz(dst, **kw):
    """private name then rename: a preempted writer must not leave a truncated file that a later run skips"""
    import os
    tmp = dst.parent / "{}.{}.tmp.npz".format(dst.stem, os.getpid())
    np.savez_compressed(tmp, **kw)
    os.replace(tmp, dst)


def write_text(path, txt):
    import os
    tmp = path.parent / (path.name + ".{}.tmp".format(os.getpid()))
    tmp.write_text(txt)
    os.replace(tmp, path)


def load_P(f):
    """a prediction, or None if it is absent or unreadable (both are logged and counted by --score)"""
    try:
        return np.load(f)["P"]
    except Exception as ex:
        print("UNREADABLE or missing", f.name, repr(ex)[:80], flush=True)
        return None


def ok_npz(f):
    return f.exists() and load_P(f) is not None


def pred_path(p, tag, kind):
    stem = p["ck"].replace("/", "_").replace("outputs_", "").replace(".pt", "")
    return OUT / "pred" / "{}__{}__{}.npz".format(stem, tag, kind)


def edited(tag):
    from aaf.data.cx_rooms import q
    from scripts.p4_10_edits import valid, verts_of
    L, W, d, w = [float(x[1:]) for x in ROOMS[tag].split("_")]
    N0 = [{"side": "NW", "d": d, "w": w, "x0": 0.0}]
    s = 1.10
    Lx, Wx = q(L * s), q(W * s)
    Nx = [dict(n, w=q(n["w"] * s), d=q(n["d"] * s)) for n in N0]
    why = valid(Lx, Wx, Nx, SRC, True, N0)
    assert why is None, why
    r = next(x for x in json.load(open("configs/p4_10/e1_rooms.json"))["rooms"] if x["s1"] == tag)
    assert np.allclose(verts_of(L, W, N0), r["verts"]) and np.allclose(r["src"], SRC)
    return Lx, Wx, [tuple(v) for v in verts_of(Lx, Wx, Nx)], {"sx": Lx / L, "sy": Wx / W}


def truth(tag, kind):
    if kind == "orig":
        z = np.load(E1T / (ROOMS[tag] + ".npz"))
        return {"rx": z["rx"], "H": z["H"], "L": float(z["L"]), "W": float(z["W"]), "verts": [tuple(map(float, v)) for v in z["verts"]]}
    z = np.load(OUT / "truth" / (tag + "_s110.npz"))
    return {"rx": z["rx"], "H": z["H"], "L": float(z["L"]), "W": float(z["W"]), "verts": [tuple(map(float, v)) for v in z["verts"]]}


def fdtd(i):
    from scripts.p4_10_truth import dense_points, record, snap, solver_room
    tag = list(ROOMS)[i]
    L, W, V, sc = edited(tag)
    d, _ = dense_points(L, W, V)
    pts = snap(d)
    H = record(L, W, SRC, pts, solver_room(L, W, V, [0.15] * len(V)))
    (OUT / "truth").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "truth" / (tag + "_s110.npz"), rx=pts, H=H.astype(np.complex64), L=L, W=W, verts=np.array(V), src=np.array(SRC), **sc)
    print("truth", tag, L, W, len(pts), sc, flush=True)


def train(i):
    import torch
    import yaml

    from aaf.train.multi_room_2d_mat import P32TrainCfg, P32Trainer
    mode, tag = [("b", "lo"), ("b", "med"), ("c", "lo"), ("c", "med")][i]
    d = yaml.safe_load(open("configs/sweep_2d_mat/P4_9_S1{}.yaml".format(tag)))
    d.update(run_id="p4_10_ec_{}_{}".format(mode, tag), ckpt_keep_iters=",".join(map(str, STEPS)))
    cfg = P32TrainCfg(**d)
    out = OUT / "arms" / "{}_{}".format(mode, tag)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_meta.json").write_text(json.dumps({"cfg": d, "n_configs": 1, "mode": mode, "note": "E-C (D215): P32Trainer's own step driven by scripts/p4_10_ec_exposure.py"}))
    t = P32Trainer(cfg, out)
    if mode == "c":
        if t.start_iter == 0:
            st = torch.load(R1, map_location="cpu")["model"]
            own = t.model.state_dict()
            bad = {k: (tuple(st[k].shape), tuple(own[k].shape)) for k in own if k in st and st[k].shape != own[k].shape}
            print("state-dict shape differences (R1 vs one-room model):", bad, "missing", set(own) - set(st), "extra", set(st) - set(own), flush=True)
            assert all(k.startswith("z_table") or "embed" in k or "latent" in k or k.startswith("z") for k in bad), bad      # only the (unused, geom_token) per-room latent table may differ
            t.model.load_state_dict({k: (own[k] if k in bad else st[k]) for k in own})
        for g in t.opt.param_groups:
            g["lr"], g["initial_lr"] = LR_C, LR_C
        t.sched = torch.optim.lr_scheduler.LambdaLR(t.opt, lambda e: 1.0)
    print(mode, tag, "start", t.start_iter, "lr", t.opt.param_groups[0]["lr"], flush=True)
    for it in range(t.start_iter + 1, max(STEPS) + 1):
        m = t._step_fn(it)
        if it <= 5 or it % 50 == 0:
            print("[{}] loss {:.4f} amp {:.4f} phase {:.4f} lr {:.2e}".format(it, m["loss"], m["L_amp"], m["L_phase"], t.opt.param_groups[0]["lr"]), flush=True)
        if it in STEPS:
            t.save_ckpt(it)
            print("saved", it, flush=True)
    print("done", mode, tag, flush=True)


def jobs():
    """render jobs grouped: one per (trajectory group)"""
    P = points()
    G = {}
    for p in P:
        tags = [p["tag"]] if p["tag"] else list(ROOMS)
        for tag in tags:
            g = "a" if p["traj"] == "a" else "{}_{}".format(p["traj"], tag)
            kinds = ["orig"] + (["s110"] if p["traj"] == "c" else [])
            for k in kinds:
                G.setdefault(g, []).append((p, tag, k))
    return G


def render(i):
    from scripts.p4_10_scorecard import Renderer
    G = jobs()
    g = sorted(G)[i]
    (OUT / "pred").mkdir(parents=True, exist_ok=True)
    cache = {}
    for p, tag, kind in G[g]:
        dst = pred_path(p, tag, kind)
        if ok_npz(dst):
            continue
        if kind == "orig" and p["ck"] == R1:
            save_npz(dst, P=np.load(E1P / "R1_150K__{}.npz".format(ROOMS[tag]))["P"])
            continue
        if not Path(p["ck"]).exists():
            print("missing", p["ck"], flush=True)
            continue
        if p["ck"] not in cache:
            cache.clear()
            cache[p["ck"]] = Renderer(p["label"], p["ck"])
        t = truth(tag, kind)
        F = cache[p["ck"]].render(t["verts"], t["L"], t["W"], [0.15] * len(t["verts"]), SRC, t["rx"])
        save_npz(dst, P=F)
        print("rendered", dst.name, len(F), flush=True)


def mode_pairs(T0, P0, Te, Pe):
    from scripts.p4_10_modes import hungarian, peaks
    f0t, _ = peaks(T0, 25.0, 150.0)
    f0p, _ = peaks(P0, 25.0, 150.0, n=40)
    fet, _ = peaks(Te, 25.0, 150.0, n=40)
    fep, _ = peaks(Pe, 25.0, 150.0, n=40)
    mt, mp = dict(hungarian(f0t, fet, 8.0)), dict(hungarian(f0p, fep, 8.0))
    return [(float(fet[mt[i]] - f0t[i]), float(fep[mp[j]] - f0p[j])) for i, j in hungarian(f0t, f0p, 3.0) if i in mt and j in mp], int(len(f0t))


def score():
    from scripts.p4_10_e1_decay import analyse
    from scripts.p4_10_modes import spec
    from scripts.p4_10_scorecard import BAND, band_r, on_lattice
    rec = []
    T = {(tag, k): truth(tag, k) for tag in ROOMS for k in ("orig", "s110")}
    for p in points():
        for tag in ([p["tag"]] if p["tag"] else list(ROOMS)):
            f = pred_path(p, tag, "orig")
            P = load_P(f)
            if P is None:
                rec.append(dict(p, room=tag, missing=True))
                continue
            t = T[(tag, "orig")]
            off = ~on_lattice(t["rx"])
            a = analyse(P, t["H"], t["rx"], np.array(SRC))
            e = dict(p, room=tag, D2=a["D2"], EDT={b: a["D1"][b]["EDT_err"] for b in a["D1"]}, EDT_bias={b: a["D1"][b]["EDT_bias"] for b in a["D1"]},
                     floor={b: a["D1"][b]["floor_model"] for b in a["D1"]}, floor_truth={b: a["D1"][b]["floor_truth"] for b in a["D1"]}, static_r=band_r(P[off], t["H"][off]),
                     n_off=int(off.sum()))
            e["fit_ok"] = bool(e["static_r"]["25-100"] >= FIT_R)
            e["readable"] = bool(e["D2"]["matched"] >= 3 and np.isfinite(e["D2"]["ratio_median"]))
            e["EDT_mean"] = float(np.nanmean(list(e["EDT"].values()))) if np.isfinite(list(e["EDT"].values())).any() else float("nan")
            if p["traj"] == "c":
                g = pred_path(p, tag, "s110")
                Pe = load_P(g)
                if Pe is not None:
                    te = T[(tag, "s110")]
                    ofe = ~on_lattice(te["rx"])
                    pr, nt = mode_pairs(spec(t["H"][off]), spec(P[off]), spec(te["H"][ofe]), spec(Pe[ofe]))
                    e["s110"] = {"edited_r": band_r(Pe[ofe], te["H"][ofe]), "pairs": pr, "n_truth_peaks": nt}
            rec.append(e)
    rep = {"rows_formula": ROWS_FORMULA, "rows_per_step": rows_per_step(CFG["R0"]), "lr_c": LR_C, "points": rec}
    # group by trajectory and step
    tab = {}
    for e in rec:
        if e.get("missing"):
            continue
        tab.setdefault((e["traj"], e["label"], e["rows"]), []).append(e)
    L = ["# P4-10 E-C (D215): how much per-room training fixes the decay", "", "Two S-1 rooms (lo = {}, med = {}); truth and metrics as E1. Rows: {}. A width ratio needs >= 3 matched peaks in the room (else 'n.r.'); the mean is 'n.f.' (not a fit) if either room's static r at 25-100 Hz is below {} (D215 addendum: noise spectra match peaks by chance).".format(ROOMS["lo"], ROOMS["med"], ROWS_FORMULA, FIT_R), "",
         "| trajectory | point | rows per room | width ratio lo / med (mean) | matched / truth peaks | EDT error 63 / 125 / 250 Hz (mean of rooms) | floor 63 / 125 / 250 dB | static r 25-100 / 100-200 / 200-300 | s = 1.10 edited r | mode-shift pairs (slope) |", "|---|---|---|---|---|---|---|---|---|---|"]
    out_rows = []
    for (tr, lab, rw), es in tab.items():
        w = {e["room"]: (e["D2"]["ratio_median"] if e["readable"] else float("nan")) for e in es}
        fit = {e["room"]: e["fit_ok"] for e in es}
        wv = [v for v in w.values() if np.isfinite(v)]
        mean_w = float(np.mean(wv)) if len(wv) == 2 else float("nan")
        d = {"traj": tr, "label": lab, "rows": rw, "width": w, "width_mean_both": mean_w, "matched": sum(e["D2"]["matched"] for e in es), "truth_peaks": sum(e["D2"]["truth_peaks"] for e in es),
             "EDT": {b: float(np.nanmean([e["EDT"][b] for e in es])) for b in ("63", "125", "250")}, "EDT_mean": float(np.nanmean([e["EDT_mean"] for e in es])),
             "EDT_mean_room": {e["room"]: e["EDT_mean"] for e in es}, "floor": {b: float(np.nanmean([e["floor"][b] for e in es])) for b in ("63", "125", "250")},
             "static": {b: float(np.mean([e["static_r"][b] for e in es])) for b in BAND}, "n_rooms": len(es), "fit_ok": all(fit.values()), "fit_by_room": fit}
        pr = [q for e in es if "s110" in e for q in e["s110"]["pairs"]]
        if any("s110" in e for e in es):
            d["edited_r"] = {b: float(np.mean([e["s110"]["edited_r"][b] for e in es if "s110" in e])) for b in BAND}
            d["n_pairs"], d["n_truth_peaks_s"] = len(pr), sum(e["s110"]["n_truth_peaks"] for e in es if "s110" in e)
            a_ = np.array(pr, float).reshape(-1, 2)
            d["slope"] = float(np.polyfit(a_[:, 0], a_[:, 1], 1)[0]) if len(a_) >= 4 and np.ptp(a_[:, 0]) > 0 else float("nan")
            d["pairs"] = pr
        out_rows.append(d)
    for d in sorted(out_rows, key=lambda d: (d["traj"], d["rows"], d["label"])):
        f3 = lambda x: " / ".join("{:.3f}".format(x[b]) for b in BAND)  # noqa: E731
        L.append("| {} | {} | {:.0f} | {} ({}) | {} / {} | {} | {} | {} | {} | {} |".format(
            d["traj"], d["label"], d["rows"], " / ".join("n.r." if not np.isfinite(d["width"].get(t, float("nan"))) else "{:.2f}".format(d["width"][t]) for t in ROOMS),
            ("n.f." if not d["fit_ok"] else "n.r." if not np.isfinite(d["width_mean_both"]) else "{:.2f}".format(d["width_mean_both"])), d["matched"], d["truth_peaks"],
            " / ".join("{:.0%}".format(d["EDT"][b]) for b in ("63", "125", "250")), " / ".join("{:.0f}".format(d["floor"][b]) for b in ("63", "125", "250")), f3(d["static"]),
            f3(d["edited_r"]) if "edited_r" in d else "--", "{} of {} ({:.2f})".format(d["n_pairs"], d["n_truth_peaks_s"], d["slope"]) if "n_pairs" in d else "--"))
    rep["table"] = out_rows
    OUT.mkdir(parents=True, exist_ok=True)
    miss = [(e["traj"], e["label"], e["room"]) for e in rec if e.get("missing")]
    miss_s = [(e["label"], e["room"]) for e in rec if e["traj"] == "c" and not e.get("missing") and "s110" not in e]
    rep["missing"], rep["missing_s110"], rep["n_points_expected"] = miss, miss_s, len(rec)
    L += ["", "Points scored {} of {}; missing: {}; stretched renders missing: {}".format(len(rec) - len(miss), len(rec), miss or "none", miss_s or "none")]
    write_text(OUT / "report.json", json.dumps(rep, indent=1))
    write_text(OUT / "report.md", "\n".join(L) + "\n")
    print("\n".join(L))


def fig():
    import matplotlib.pyplot as plt

    from scripts.p4_10_md_style import OUT as FOUT
    from scripts.p4_10_md_style import new
    rep = json.loads((OUT / "report.json").read_text())
    tab = rep["table"]
    pts = [p for p in rep["points"] if not p.get("missing")]
    col = {"a": "#7a7a7a", "b": "#e8743b", "c": "#2a78d6"}
    nm = {"a": "One model shared by 1,000 rooms (R0)", "b": "One room, trained from scratch", "c": "Shared model (R1), fine-tuned on the room"}
    f = new()
    gs = f.add_gridspec(1, 2, left=0.07, right=0.985, top=0.755, bottom=0.24, wspace=0.2)
    meta = {"rows_formula": rep["rows_formula"], "series": {}, "fit_rule": "static off-lattice r at 25-100 Hz >= {} (D215 addendum); points below it are not drawn".format(FIT_R), "not_fit": [(d["traj"], d["label"], d["rows"]) for d in tab if not d["fit_ok"]], "note": "y on the right panel = mean EDT error over 63 / 125 / 250 Hz and the two rooms; light points = each room"}
    R0_120 = next(p["rows"] for p in pts if p["label"] == "R0 @120K")
    for j, (key, lab, yl) in enumerate((("width", "Resonance width, times the true width", (0.9, 1.75)), ("edt", "Error in the early decay time (mean of three octaves)", (0, 0.40)))):
        ax = f.add_subplot(gs[0, j])
        for tr in "abc":
            P_ = [p for p in pts if p["traj"] == tr]
            if key == "width":
                P_ = [p for p in P_ if p["readable"] and p["fit_ok"]]
                yv = lambda p: p["D2"]["ratio_median"]  # noqa: E731
            else:
                P_ = [p for p in P_ if np.isfinite(p["EDT_mean"]) and p["fit_ok"]]
                yv = lambda p: p["EDT_mean"]  # noqa: E731
            ax.scatter([p["rows"] for p in P_], [yv(p) for p in P_], s=52, color=col[tr], alpha=0.35, linewidths=0, zorder=3)
            # the mean line over rooms where both rooms are readable; shared: the same x may hold several models
            xs = sorted({(d["rows"], d["label"]) for d in tab if d["traj"] == tr and d["fit_ok"] and (np.isfinite(d["width_mean_both"]) if key == "width" else np.isfinite(d["EDT_mean"]))})
            by = {(d["rows"], d["label"]): (d["width_mean_both"] if key == "width" else d["EDT_mean"]) for d in tab if d["traj"] == tr}
            line = [(x, by[(x, l)], l) for x, l in xs]
            if tr == "a":
                r0 = [(x, y) for x, y, l in line if l.startswith("R0")]
                ax.plot([x for x, _ in r0], [y for _, y in r0], color=col[tr], lw=3.2, zorder=4, label=nm[tr])
                for x, y, l in line:
                    if not l.startswith("R0"):
                        ax.plot([x], [y], marker={"CTL": "s", "S2X": "D", "R1 ": "^"}[l[:3]], ls="none", ms=11, color=col[tr], markeredgecolor="white", markeredgewidth=1.2, zorder=6, label={"CTL": "shared, 25% longer (CTL)", "S2X": "shared, 2,000 rooms (S2X)", "R1 ": "shared, moved sources (R1)"}[l[:3]] if j == 0 else None)
            else:
                ax.plot([x for x, _, _ in line], [y for _, y, _ in line], color=col[tr], lw=3.2, marker="o", ms=8, markeredgecolor="white", zorder=4, label=nm[tr])
            meta["series"]["{}|{}".format(key, tr)] = {"mean": [{"rows": x, "y": y, "label": l} for x, y, l in line], "points": [{"rows": p["rows"], "y": yv(p), "room": p["room"], "label": p["label"]} for p in P_]}
        ax.axvline(R0_120, color="#444444", lw=1.6, ls=(0, (4, 3)), zorder=1)
        ax.text(R0_120 * 0.9, yl[1] * 0.985, "rows per room in R0\n(120,000 steps)", ha="right", va="top", fontsize=14, color="#444444")
        ax.axhline(1.0 if key == "width" else 0.0, color="#111111", lw=1.4, ls=":", zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(90, 250000)
        ax.set_xticks([100, 1000, 10000, 100000])
        ax.set_xticklabels(["100", "1,000", "10,000", "100,000"])
        ax.set_ylim(*yl)
        ax.set_xlabel("training rows per room")
        ax.set_title(lab, loc="left")
        ax.grid(True, color="#ececec")
        ax.set_axisbelow(True)
        if key == "edt":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "{:.0%}".format(v)))
            ax.text(95, 0.008, "0 = right", fontsize=13, color="#111111", va="bottom")
        else:
            ax.text(95, 1.01, "true width", fontsize=13, color="#111111", va="bottom")
    h, l = f.axes[0].get_legend_handles_labels()
    f.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=14, bbox_to_anchor=(0.5, 0.005))
    f.suptitle("How much training per room does it take to get the decay right?", x=0.07, ha="left", fontsize=22, y=0.955)
    f.text(0.07, 0.865, "Two training rooms (6.16 x 4.56 m and 5.74 x 4.78 m). Measured off the training lattice. Light points: each room; lines: mean of the two.\nModels that do not yet fit the room (accuracy below 0.80 at 25-100 Hz) are not drawn; from scratch that is every point below 1,000 steps.", fontsize=14, color="#444444", va="top")
    import os
    FOUT.mkdir(parents=True, exist_ok=True)
    meta["missing"] = rep.get("missing", [])
    for ext, kw in (("png", {"dpi": 200}), ("pdf", {})):
        tmp = FOUT / "EC_exposure.{}.tmp.{}".format(os.getpid(), ext)
        f.savefig(tmp, **kw)
        os.replace(tmp, FOUT / ("EC_exposure." + ext))
    write_text(FOUT / "EC_exposure.json", json.dumps(meta, indent=1))
    print("wrote EC_exposure", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("--fdtd", "--train", "--render", "--score", "--fig"):
        ap.add_argument(k, action="store_true")
    ap.add_argument("--idx", type=int, default=0)
    a = ap.parse_args()
    if a.fdtd:
        fdtd(a.idx)
    elif a.train:
        train(a.idx)
    elif a.render:
        render(a.idx)
    elif a.score:
        score()
    elif a.fig:
        fig()
