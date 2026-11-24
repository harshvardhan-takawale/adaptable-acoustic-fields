"""P4-10 meeting day, motion figures M1, M2, M3, M5 (D221). Maps: full grid. Numbers: off-lattice dense receivers, frozen scorer. --fig M1 M2 M3 M5"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import scripts.p4_10_md_figs_a as A
import scripts.p4_10_md_style as S
from scripts.p4_10_md import BINS, POS12, ROOT, SHOW, STRIP_P
from scripts.p4_10_md_figs_a import CASE, FB, badge, caption, colorbar, level_map, pred, r_at, r_band, rk, truth
from scripts.p4_10_md_style import COL, new, plt, save
from scripts.p4_10_scorecard import BAND, on_lattice

A.NAME.update({"RCP_150K": "RCP"})
S.NAME.update({"RCP_150K": "RCP"})
COL["RCP"] = "#eda100"
ROWS = (("Truth", "truth"), ("R0", "R0_120K"), ("RCP", "RCP_150K"), ("R1", "R1_150K"))
LONG = {"R0_120K": "R0: one source per room", "RCP_150K": "RCP: one source, plus source-listener swaps", "R1_150K": "R1: four sources per room, plus swaps"}
POS = [("show_s1.000" if i == 0 else "show_p{:02d}".format(i)) for i in STRIP_P]
LAB = ["start" if i == 0 else "third source" if i == 11 else "{:.0f}%".format(100 * POS12[i]) for i in STRIP_P]


def src_family():
    """source-family r over the 15 in-family rooms (25-100 / 100-200 / 200-300), read from the posted scorecards"""
    a = json.loads(Path("outputs/p4_10/scorecard_source_R0_CTL_RCP.json").read_text())["families"]["Source|test"]
    b = json.loads(Path("outputs/p4_10/scorecard_R1_full.json").read_text())["families"]["Source|test"]
    out = {m: [a[m]["edited_r"][x][0] for x in BAND] for m in ("R0_120K", "RCP_150K")}
    out["R1_150K"] = [b["R1_150K"]["edited_r"][x][0] for x in BAND]
    return out


def m1(POS=POS, room=SHOW, tag="", fs=(50, 125), note=None, ranks_key="V2|R1|third source r|"):
    R = A.ranks()
    fam = src_family()
    for f in fs:
        fig = new(16, 9)
        n, m = len(POS), len(ROWS)
        gs = fig.add_gridspec(m, n, left=0.105, right=0.915, top=0.885, bottom=0.185, wspace=0.03, hspace=0.21)
        meta = {"figure": "M1" + (" second example" if tag else ""), "room": room, "frequency_hz": f, "positions": [CASE[k]["src"] for k in POS], "source_family_r_15_in_family_rooms_25-100|100-200|200-300": {LONG[k]: v for k, v in fam.items()}, "panels": {},
                "ranks": R["V2|R1|third source r|" + A.BAND_OF[f]]}
        Lm, Wm = max(CASE[k]["L"] for k in POS), max(CASE[k]["W"] for k in POS)
        im = None
        for j, k in enumerate(POS):
            for i, (lab, who) in enumerate(ROWS):
                ax = fig.add_subplot(gs[i, j])
                im = A.panel(ax, k, who, f)
                ax.set_xlim(-0.1, Lm + 0.1)
                ax.set_ylim(-0.1, Wm + 0.1)
                if i == 0:
                    ax.set_title(LAB[j], fontsize=15)
                if j == 0:
                    ax.text(-0.07, 0.5, {"Truth": "Truth", "R0": "R0\n(one source)", "RCP": "RCP\n(+ swaps)", "R1": "R1\n(4 sources\n+ swaps)"}[lab], transform=ax.transAxes, ha="right", va="center", fontsize=15, fontweight="bold")
                if who != "truth":
                    s = A.scores(who, k, f)
                    badge(ax, "r = {:.2f}".format(s["r_at_{}Hz".format(f)]))
                    meta["panels"]["{}|{}".format(k, LONG[who])] = s
        colorbar(fig, im)
        fig.suptitle(("Second example, the median room: moving the source, three ways to train it, {} Hz" if tag else "Moving the source: three ways to train it, {} Hz").format(f), fontsize=20, x=0.02, ha="left", y=0.975, fontweight="bold")
        sel = note or "{} on R1's third-source accuracy at {} Hz".format(rk(R, ranks_key + A.BAND_OF[f]), A.BAND_OF[f])
        cp = "Maps: full grid (truth FDTD 2 cm, models 4 cm). r = off-lattice accuracy at that frequency, never read from the map. In-family (held-out) room {}; {}.\nSource-family accuracy over the 15 in-family rooms at 25-100 Hz: R0 {:.3f}, RCP {:.3f}, R1 {:.3f}.".format(
            room, sel, fam["R0_120K"][0], fam["RCP_150K"][0], fam["R1_150K"][0])
        caption(fig, cp)
        meta["caption"] = cp
        save(fig, "M1{}_source_three_recipes_{}Hz".format(tag, f), meta)


def m1m():
    """second example at 125 Hz (D211's selection rule: the showcase ranks 3 of 15 there)"""
    from scripts.p4_10_md import MED125
    pos = ["med_p{:02d}".format(i) for i in STRIP_P]
    m1(pos, MED125, "m", (125,), "the median room on R1's third-source accuracy at 100-200 Hz (the showcase ranks 3 of 15 there)")


def swap(m):
    return np.load(ROOT / "pred" / "{}__swap_show_s1.000.npz".format(m))


def recip_rank():
    d = json.loads(Path("outputs/p4_9/recip_gap/R1_150K.json").read_text())["rooms"]
    d = [x for x in d if x["group"] == "test"]
    v = {Path(x["room"]).stem: x["swapped"]["25-100"] - x["normal"]["25-100"] for x in d}
    s = sorted(v, key=lambda r: -v[r])
    return {"rank": s.index(SHOW) + 1, "of": len(s), "swapped_minus_normal_r_25-100": v[SHOW], "median": float(np.median(list(v.values())))}


def m2():
    rr = recip_rank()
    k = "show_s1.000"
    c, t = CASE[k], truth(k)
    s = ~on_lattice(t["rx"])
    cols = [("Truth", None, None)] + [("{} {}".format(S.NAME[m], lab), m, sw) for m in ("R0_120K", "R1_150K") for lab, sw in (("normal", False), ("swapped", True))]
    fig = new(16, 9)
    gs = fig.add_gridspec(2, 5, left=0.075, right=0.915, top=0.86, bottom=0.205, wspace=0.04, hspace=0.24)
    meta = {"figure": "M2", "room": SHOW, "panels": {}, "ranks": rr}
    im = None
    for i, f in enumerate((50, 125)):
        b = FB[f]
        bi = BINS.index(b)
        ref = float(np.percentile(20 * np.log10(np.abs(t["Hg"][:, bi]) + 1e-12), 99))
        for j, (lab, m, sw) in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            if m is None:
                im = level_map(ax, c, t["gij"], t["Hg"][:, bi], 1, ref)
            else:
                p = swap(m) if sw else pred(m, k)
                im = level_map(ax, c, p["gij"], p["Pg"][:, bi], 2, ref)
                r = r_at(p["Pd"], t["Hd"], t["rx"], b)
                badge(ax, "r = {:.2f}".format(r))
                meta["panels"]["{}|{}|{}Hz".format(S.NAME[m], "swapped" if sw else "normal", f)] = {"r_at_{}Hz".format(f): r, "r_band": r_band(p["Pd"], t["Hd"], t["rx"])}
            if i == 0:
                ax.set_title(lab, fontsize=16)
            if j == 0:
                ax.text(-0.06, 0.5, "{} Hz".format(f), transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=17, fontweight="bold")
    colorbar(fig, im)
    fig.suptitle("Swap the source and the listener: a physical field does not change", fontsize=20, x=0.02, ha="left", y=0.965, fontweight="bold")
    cp = ("Normal: the source stays at the star, the map shows the listener's position. Swapped: at each point the source is put there and the listener sits at the star. By reciprocity both maps equal the truth. r = off-lattice accuracy against the truth at that frequency.\n"
          "In-family (held-out) room {}; it ranks {} of {} on R1's swapped-minus-normal accuracy at 25-100 Hz ({:+.2f}; median room {:+.2f}).").format(SHOW, rr["rank"], rr["of"], rr["swapped_minus_normal_r_25-100"], rr["median"])
    caption(fig, cp)
    meta["caption"] = cp
    save(fig, "M2_reciprocity_maps", meta)


def m3():
    R = A.ranks()
    key = "show_p11"
    c, t = CASE[key], truth(key)
    w = np.load(ROOT / "truth" / "walk_p11.npz")
    P = {m: np.load(ROOT / "pred" / "{}__path_{}.npz".format(m, key))["Pp"] for m in ("R1_150K", "R0_120K")}
    lo, hi = 50, 600
    Lv = lambda H: 20 * np.log10(np.abs(H) + 1e-12)  # noqa: E731
    ref = np.percentile(Lv(w["Hp"][:, lo:hi]), 99)
    M = {"truth": Lv(w["Hp"][:, lo:hi]) - ref, "R1": Lv(P["R1_150K"][:, lo:hi]) - ref, "R0": Lv(P["R0_120K"][:, lo:hi]) - ref}
    x = w["path"][:, 0]
    pos = [int(np.argmin(np.abs(x - q))) for q in (0.2 * c["L"], 0.5 * c["L"], 0.8 * c["L"])]
    rb = {m: r_band(pred(m, key)["Pd"], t["Hd"], t["rx"]) for m in ("R1_150K", "R0_120K")}
    fr = np.arange(601) * 0.5
    fig = new(16, 9)
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.94, top=0.79, bottom=0.165, hspace=0.45, wspace=0.22, height_ratios=[1.35, 1])
    for j, nm in enumerate(("truth", "R1", "R0")):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(M[nm].T, origin="lower", aspect="auto", extent=[x[0], x[-1], 25, 300], cmap="viridis", vmin=-35, vmax=0, interpolation="nearest")
        for q in pos:
            ax.axvline(x[q], color="white", lw=1.6, ls=(0, (5, 3)))
        ax.set_title({"truth": "truth", "R1": "R1", "R0": "R0"}[nm], loc="left", fontsize=16)
        ax.set_xlabel("position along the walk, x (m)")
        if j == 0:
            ax.set_ylabel("frequency (Hz)")
        cb = fig.colorbar(im, ax=ax, pad=0.015)
        cb.ax.tick_params(labelsize=14)
        if j == 2:
            cb.set_label("dB", fontsize=14)
    for j, q in enumerate(pos):
        ax = fig.add_subplot(gs[1, j])
        for nm, col in (("truth", COL["truth"]), ("R1", COL["R1"]), ("R0", COL["R0"])):
            ax.plot(fr[lo:hi], M[nm][q], color=col, lw=2.1, label=nm)
        ax.set_ylim(-45, 18)
        ax.set_xlim(25, 300)
        ax.set_title("walk position x = {:.1f} m".format(x[q]), loc="left", fontsize=15)
        ax.set_xlabel("frequency (Hz)")
        ax.grid(True, color="#e6e6e6")
        if j == 0:
            ax.set_ylabel("level (dB)")
        if j == 1:
            ax.legend(frameon=False, loc="upper right", ncol=3, fontsize=14, columnspacing=1.0, handlelength=1.6)
    ins = fig.add_axes([0.862, 0.825, 0.125, 0.155])
    A.panel(ins, key, "truth", 50)
    ins.plot(w["path"][:, 0], w["path"][:, 1], color="white", lw=2.2, ls=(0, (4, 2)), zorder=7)
    for q in pos:
        ins.plot(x[q], 1.0, "o", color="white", ms=6, markeredgecolor="black", zorder=8)
    fig.suptitle("The listener's walk with the source moved to the third position", fontsize=20, x=0.02, ha="left", y=0.97, fontweight="bold")
    t3_ = lambda d: " / ".join("{:.2f}".format(d[b]) for b in BAND)  # noqa: E731
    fig.text(0.02, 0.885, "Accuracy with the source moved (25-100 / 100-200 / 200-300 Hz): R1 {}; R0 {}.".format(t3_(rb["R1_150K"]), t3_(rb["R0_120K"])), fontsize=14, color="#444444")
    cp = ("Accuracy is the off-lattice dense-receiver score at this source position, never read from the pictures. Walk: y = 1.0 m, x from 0.5 m to {:.2f} m (100 points). In-family (held-out) room {}; {} on R1's third-source accuracy at 25-100 Hz.").format(
        c["L"] - 0.5, SHOW, rk(R, "V2|R1|third source r|25-100"))
    caption(fig, cp)
    save(fig, "M3_walk_moved_source", {"figure": "M3", "room": SHOW, "case": key, "source": c["src"], "R1_r_off_lattice_dense": rb["R1_150K"], "R0_r_off_lattice_dense": rb["R0_120K"], "path_xy": w["path"].tolist(),
                                       "marked_x_m": [float(x[q]) for q in pos], "caption": cp, "ranks": R["V2|R1|third source r|25-100"]})


def intensity(p, g):
    return -np.imag(np.conj(p)[:, None] * g)


FLUX = {}


def outward_flux(key, R=0.15):
    """mean outward component of the truth's active intensity on a circle of radius R around the source, 50 Hz"""
    t, c = truth(key), CASE[key]
    src = np.array(c["src"])
    ij = {(int(a), int(b)): k for k, (a, b) in enumerate(t["gij"])}
    H = t["Hg"][:, 0]
    out = []
    for a in np.linspace(0, 2 * np.pi, 72, endpoint=False):
        p = src + R * np.array([np.cos(a), np.sin(a)])
        i, j = int(round(p[0] / 0.02)), int(round(p[1] / 0.02))
        q = [ij.get(v) for v in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1), (i, j))]
        if None in q:
            continue
        g = np.array([(H[q[0]] - H[q[1]]) / 0.04, (H[q[2]] - H[q[3]]) / 0.04])
        out.append(float(-np.imag(np.conj(H[q[4]]) * g) @ np.array([np.cos(a), np.sin(a)])))
    return float(np.mean(out))


def m5():
    import math

    from aaf.eval.physical_projection import C as C0, CUT, DF, k_grid, project_lowpass
    from scripts.p4_9_dir_zeroshot import lattice_grad
    R = A.ranks()
    key, f = "show_p11", 50
    for k_ in ("show_s1.000", key):
        FLUX[k_] = outward_flux(k_)
    b, bi = FB[f], BINS.index(FB[f])
    c, t = CASE[key], truth(key)
    L, W = c["L"], c["W"]
    dx, dy = (L - 0.24) / 63, (W - 0.24) / 63
    from scripts.p4_10_truth import dense_points
    _, keep = dense_points(L, W, [tuple(v) for v in c["verts"]])
    rx = t["rx"]
    ok = ~on_lattice(rx) & (np.hypot(*(rx - np.array(c["src"])).T) >= 0.5)
    # truth gradient: central differences of the 2 cm FDTD field at this bin
    ij = {(int(a), int(d)): k for k, (a, d) in enumerate(t["gij"])}
    Hg = t["Hg"][:, bi]
    gT = np.full((len(rx), 2), np.nan + 0j, np.complex128)
    for n, p in enumerate(rx):
        i, j = int(round(p[0] / 0.02)), int(round(p[1] / 0.02))
        q = [ij.get((i + 1, j)), ij.get((i - 1, j)), ij.get((i, j + 1)), ij.get((i, j - 1))]
        if None in q:
            continue
        gT[n] = [(Hg[q[0]] - Hg[q[1]]) / 0.04, (Hg[q[2]] - Hg[q[3]]) / 0.04]
    IT = intensity(t["Hd"][:, b], gT)
    K = k_grid(64, 64, dx, dy, 4)
    kc = np.array([CUT * 2 * math.pi * b * DF / C0])
    unit = lambda I: I / np.maximum(np.linalg.norm(I, axis=1, keepdims=True), 1e-30)  # noqa: E731
    res, meta = {}, {"figure": "M5", "room": SHOW, "case": key, "frequency_hz": f, "source": c["src"], "numbers": {}, "ranks": R["V2|R1|third source r|25-100"]}
    fig = new(16, 9)
    gs = fig.add_gridspec(1, 3, left=0.045, right=0.905, top=0.80, bottom=0.19, wspace=0.04)
    cols = [("Truth", None), ("R1", "R1_150K"), ("R0", "R0_120K")]
    im = None
    for j, (lab, m) in enumerate(cols):
        ax = fig.add_subplot(gs[0, j])
        if m is None:
            im = A.panel(ax, key, "truth", f)
            good = ok & np.all(np.isfinite(IT), 1)
            I = IT
            tt = "Truth, {} Hz".format(f)
        else:
            p = pred(m, key)
            im = A.panel(ax, key, m, f)
            F = p["Pd"][:, [b]]
            Pp = project_lowpass(F.T, keep, K, kc).T.astype(np.complex64)
            lr, LG = lattice_grad(Pp, keep, dx, dy)
            gM = np.full((len(rx), 2), np.nan + 0j, np.complex128)
            gM[lr] = LG[:, :, 0]
            I = intensity(Pp[:, 0], gM)
            good = ok & np.all(np.isfinite(I), 1) & np.all(np.isfinite(IT), 1)
            cos = (unit(IT[good]) * unit(I[good])).sum(1)
            n = {"energy_flow_direction_agreement_mean_cosine": float(np.mean(cos)), "median_angle_deg": float(np.degrees(np.median(np.arccos(np.clip(cos, -1, 1))))), "n_rx": int(good.sum())}
            meta["numbers"][lab] = n
            tt = "{}   direction agreement {:.2f}".format(lab, n["energy_flow_direction_agreement_mean_cosine"])
        q = np.nonzero(good)[0]
        q = q[::5]
        u = unit(I[q]) * 0.20
        ax.quiver(rx[q, 0], rx[q, 1], u[:, 0], u[:, 1], color="white", scale=1, scale_units="xy", angles="xy", width=0.0036, zorder=7)
        ax.set_title(tt, fontsize=16, loc="left")
    colorbar(fig, im, rect=(0.93, 0.24, 0.012, 0.5))
    fig.suptitle("Where the sound energy flows, source moved to the third position, {} Hz".format(f), fontsize=20, x=0.02, ha="left", y=0.965, fontweight="bold")
    cp = ("Arrows: direction of the steady energy flow (unit length). Truth: FDTD, 2 cm. Model: gradient of the band-limited field. Direction agreement = mean cosine over off-lattice receivers at least 0.5 m from the source. "
          "At this position the source radiates little at 50 Hz (net outward flow 0.15 m around it {:.3f}, against {:.3f} at the original source), so the arrows show mostly circulating energy. In-family (held-out) room {}; {} on R1's third-source accuracy at 25-100 Hz (the direction agreement has no 15-room rank: no per-room truth gradients exist for a moved source).").format(FLUX[key], FLUX["show_s1.000"], SHOW, rk(R, "V2|R1|third source r|25-100"))
    caption(fig, cp)
    meta["caption"] = cp
    meta["net_outward_flow_at_0.15m"] = dict(FLUX)
    save(fig, "M5_energy_flow_moved_source", meta)


# ------------------------------------------------------------------------------------------------ P1F: the P-1 result
def p1f():
    """P1F from outputs/p4_10/pt/report.json as it stands; the common start (+0) is R1 @150K, read from the posted P-2 and R1 reports"""
    rep = json.loads(Path("outputs/p4_10/pt/report.json").read_text())
    steps = {"152K": 2500, "155K": 5000, "157K": 7500, "160K": 10000}
    rd = [(steps[k], rep["reads"][k]) for k in steps if k in rep["reads"]]
    if not rd:
        print("no P-1 read yet")
        return
    stopf = Path("outputs/p4_10/pt/STOPPED")
    stopped = stopf.read_text().strip() if stopf.exists() else rep.get("stopped", "")
    v1 = json.loads(Path("outputs/p4_10/verdict_R1_150K.json").read_text())
    res0 = json.loads(Path("outputs/p4_10/p2/residual.json").read_text())["bands"]["25-100"]
    hf0, uf0 = v1["RC1"]["HF"]["r1"], {b: v1["RC1"]["UF"][b]["r1"] for b in ("25-100", "100-200", "200-300")}
    R0 = {"residual": res0["R1_150K|raw"]["over_truth"], "n_rooms": res0["R1_150K|raw"]["n_rooms"]}
    xs = [0] + [s for s, _ in rd]
    fig = new(16, 9)
    gs = fig.add_gridspec(1, 2, left=0.075, right=0.975, top=0.80, bottom=0.235, wspace=0.22)
    P, C = COL["R1-M"], COL["R1"]
    ax = fig.add_subplot(gs[0, 0])
    ax.axhline(1.0, color="#111111", ls=(0, (2, 2)), lw=2, label="truth = 1")
    y = {"pt": [R0["residual"]] + [r["reads"]["2_3_residual"]["25-100"]["pt_raw_over_truth"] for _, r in rd], "ctl": [R0["residual"]] + [r["reads"]["2_3_residual"]["25-100"]["ctl_raw_over_truth"] for _, r in rd]}
    ax.plot(xs, y["pt"], "-o", color=P, lw=3, ms=10, label="R1-P(t), with the penalty")
    ax.plot(xs, y["ctl"], "--s", color=C, lw=3, ms=9, label="R1-ctl, no penalty")
    ax.set_xlabel("fine-tune steps")
    ax.set_ylabel("residual, times the truth's (25-100 Hz)")
    ax.set_title("How far from the wave equation", loc="left", fontsize=17)
    ax.grid(True, color="#e6e6e6")
    ax.set_ylim(0, max(max(y["pt"]), max(y["ctl"])) * 1.25)
    ax.legend(frameon=False, fontsize=14, loc="center left", bbox_to_anchor=(0.02, 0.42))
    ax = fig.add_subplot(gs[0, 1])
    st = {"HF (in-family, 200-300 Hz)": ([hf0] + [r["reads"]["1_static"]["HF"]["pt"] for _, r in rd], [hf0] + [r["reads"]["1_static"]["HF"]["ctl"] for _, r in rd], "#1baf7a")}
    for b, col in (("25-100", "#e8743b"), ("100-200", "#2a78d6"), ("200-300", "#7b5ea7")):
        st["UF " + b + " Hz (new rooms)"] = ([uf0[b]] + [r["reads"]["1_static"]["UF"][b]["pt"] for _, r in rd], [uf0[b]] + [r["reads"]["1_static"]["UF"][b]["ctl"] for _, r in rd], col)
    for k, (a, c, col) in st.items():
        ax.plot(xs, a, "-o", color=col, lw=2.6, ms=8, label=k)
        ax.plot(xs, c, "--s", color=col, lw=2.2, ms=7)
    ax.set_xlabel("fine-tune steps")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy, solid = penalty, dashed = control", loc="left", fontsize=17)
    ax.grid(True, color="#e6e6e6")
    ax.legend(frameon=False, fontsize=13.9, loc="center right", bbox_to_anchor=(1.0, 0.33))
    last_s, last = rd[-1]
    e = last["reads"]["4_e3_63hz"]
    d = last["reads"]["4_d1"]
    cp = ("P-1, an exploratory re-registration (D205's verdict stands): fine-tuning R1 with a mean-value (wave-equation) penalty, against the same run without it; planned 10,000 steps. Start = R1 @150K (common to both). Reads done: {}. " + ("Stopped automatically after the first read: accuracy at 200-300 Hz fell more than 0.02 below the control with its interval below zero. " if stopped else "") + "Last read +{}: 63 Hz energy common to the room over the truth's {:.2f} with the penalty, {:.2f} without; early-decay error 63 / 125 / 250 Hz {:.0f} / {:.0f} / {:.0f}% with, {:.0f} / {:.0f} / {:.0f}% without. 15 in-family (held-out) rooms, off-lattice; no room rank applies.").format(
        ", ".join("+{:,}".format(s) for s, _ in rd), last_s, e["pt_over_truth_of_means"], e["ctl_over_truth_of_means"], *(100 * d[o]["EDT_err"]["pt"] for o in ("63", "125", "250")), *(100 * d[o]["EDT_err"]["ctl"] for o in ("63", "125", "250")))
    caption(fig, cp)
    fig.suptitle("Adding physics to training: the mean-value penalty (P-1)" + (", stopped by its pre-set rule" if stopped else "" if 10000 in [s for s, _ in rd] else ", PARTIAL"), fontsize=20, x=0.02, ha="left", y=0.965, fontweight="bold")
    save(fig, "P1F_physics_penalty", {"figure": "P1F", "steps": xs, "residual_over_truth_25-100": y, "accuracy": {k: {"penalty": v[0], "control": v[1]} for k, v in st.items()}, "stopped": rep.get("stopped", ""),
                                      "start": {"residual_over_truth": R0, "HF": hf0, "UF": uf0}, "reads_raw": {str(s): r["reads"] for s, r in rd}, "caption": cp})


# ------------------------------------------------------------------------------------- M4: capture then move the source
def m4():
    from scripts.p4_10_md_m4 import B50, PICK, spec
    from scripts.p4_10_scorecard import PRED, TRUTH, cid, stored, db, colr, on_lattice as onl
    pk = json.loads(PICK.read_text())
    ebr = json.loads(Path("outputs/p4_10/eb/report.json").read_text())["families"]["Source|third"]["bands"]["25-100"]["C"]
    room, c = spec()
    base = stored(c)
    d = np.load(ROOT / "pred" / "m4.npz")
    T = {"orig": np.load(ROOT / "truth" / "m4_orig.npz"), "third": np.load(ROOT / "truth" / "m4_third.npz")}
    Tthird = np.load(TRUTH / (cid(c) + ".npz"))
    off = ~onl(base["rx"])
    keep = off.copy()
    keep[d["measured"]] = False                                        # the 32 measured receivers are excluded for every method
    tr = {"orig": base["H"], "third": Tthird["Hd"]}
    rr = {}
    r50 = lambda P, H: float(colr(db(P[keep]), db(H[keep]))[B50])  # noqa: E731
    for tag in ("orig", "third"):
        rr["zero|" + tag] = r50(d["zero_dense_" + tag][:, :][:, :], tr[tag]) if d["zero_dense_" + tag].shape[1] > 100 else None
        rr["cap|" + tag] = r50(d["cap_dense_" + tag], tr[tag])
    rr["krr|third"] = float(colr(db(d["krr_dense"][keep]), db(tr["third"][keep][:, [B50]]))[0])
    CASEM = {"L": c["orig_L"], "W": c["orig_W"], "verts": [list(v) for v in c["orig_verts"]]}
    cells = [("orig", "Truth (dots: the 32 microphones)", None, "truth"), ("orig", "Zero-shot", "zero_grid_orig", "zero|orig"), ("orig", "Captured", "cap_grid_orig", "cap|orig"), ("orig", None, None, None),
             ("third", "Truth, source moved", None, "truth"), ("third", "Zero-shot", "zero_grid_third", "zero|third"), ("third", "Captured, then moved", "cap_grid_third", "cap|third"), ("third", "32 mics interpolated (KRR)", "krr_grid", "krr|third")]
    fig = new(16, 9)
    gs = fig.add_gridspec(2, 4, left=0.075, right=0.915, top=0.86, bottom=0.205, wspace=0.04, hspace=0.26)
    im = None
    meta = {"figure": "M4", "room": room, "frequency_hz": 50, "source_original": [0.5, 1.6], "source_third": list(c["src"]), "r_at_50Hz": rr, "ranks": {"zero_shot_static_rank_from_lowest": pk["rank_from_lowest"], "of": pk["of"], "value": pk["zero_shot_static_r_25-100"]}}
    for n, (tag, lab, key, rk_) in enumerate(cells):
        if lab is None:
            continue
        ax = fig.add_subplot(gs[n // 4, n % 4])
        t = T[tag]
        ref = float(np.percentile(20 * np.log10(np.abs(t["Hg"][:, 0]) + 1e-12), 99))
        cc = dict(CASEM, src=(0.5, 1.6) if tag == "orig" else tuple(c["src"]))
        if key is None:
            im = level_map(ax, cc, t["gij"], t["Hg"][:, 0], 1, ref)
            if tag == "orig":
                m = base["rx"][d["measured"]]
                ax.plot(m[:, 0], m[:, 1], "o", color="white", ms=5.5, markeredgecolor="black", markeredgewidth=0.9, zorder=9)
        else:
            im = level_map(ax, cc, d["gij"], d[key][:, 0], 2, ref)
            badge(ax, "r = {:.2f}".format(rr[rk_]))
        ax.set_title(lab, fontsize=15)
    colorbar(fig, im)
    fig.text(0.045, 0.70, "source at the\noriginal place", rotation=90, ha="center", va="center", fontsize=16, fontweight="bold")
    fig.text(0.045, 0.36, "source at the\nthird place", rotation=90, ha="center", va="center", fontsize=16, fontweight="bold")
    fig.suptitle("Measure a new room with 32 microphones, then move the source, 50 Hz", fontsize=20, x=0.02, ha="left", y=0.965, fontweight="bold")
    cp = ("R1-M, unseen room {} (the {}th lowest of {} by R1-M's zero-shot accuracy, {:.2f} at 25-100 Hz), first draw. r = off-lattice accuracy at 50 Hz, the 32 microphones left out for every method. KRR: the 32 original measurements interpolated (it cannot know the move). Maps: truth 2 cm, models 4 cm. Over all 16 unseen rooms, moving the source after capture scores below the zero-shot move at 25-100 Hz ({:.2f} against {:.2f}, E-B), the same order as here.").format(
        room, pk["rank_from_lowest"], pk["of"], pk["zero_shot_static_r_25-100"], ebr["cap_edit"][0], ebr["zero_edit"][0])
    caption(fig, cp)
    meta["caption"] = cp
    meta["E-B_source_third_25-100"] = {"captured_then_moved": ebr["cap_edit"], "zero_shot": ebr["zero_edit"]}
    save(fig, "M4_capture_then_move", meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", nargs="+", required=True)
    a = ap.parse_args()
    for f in a.fig:
        globals()[f.lower()]()
