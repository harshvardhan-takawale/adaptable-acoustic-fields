"""P3-2 BLOCKING physics gate — run BEFORE generating the dataset.

Verifies, on pyroomacoustics ground truth, that a single absorbent wall produces a
wall-SELECTIVE modal signature. If it does not, the premise of the whole chunk is false
and no amount of training would fix it, so the dataset build is gated on this
(``sbatch --dependency=afterok:<gate_jobid>``).

Exit codes:  0 = PASS   2 = PASS-WITH-AMENDMENT   3 = STOP

Decision rule (D47) — the spec's "order of magnitude" is applied to BANDWIDTH, not level.
Measured level selectivity is only ~4.4:1 while bandwidth selectivity is ~50:1, so a 10:1
gate on level would STOP on correct physics. Level is therefore checked for sign and
magnitude only, and the selectivity test runs on bandwidth.

T2b is deliberately three-way. Two damping laws are in play:

  ism_ray  (what pyroomacoustics computes)  gamma ~ cos(theta) -> selectivity -> infinity
  kuttruff (locally-reacting impedance wall) gamma ~ alpha*eps_n/L -> selectivity = 2.0

A 2:1 world is still a real, monotone, wall-specific, learnable signal, so it amends the
claim rather than blocking it. Only "no directional signal" or "wrong sign" is a STOP.

Usage:
    python scripts/p3_2_physics_gate.py [--out outputs/p3_2] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pyroomacoustics as pra  # noqa: E402

from aaf.eval.modal_bandwidth import caps_from_predicted_bw, measure_modes  # noqa: E402
from aaf.eval.modal_projection import (  # noqa: E402
    TANGENTIAL,
    X_AXIAL,
    Y_AXIAL,
    project_field,
)
from aaf.sim.analytical_modal_2d import (  # noqa: E402
    damping_to_bandwidth_hz,
    modal_damping_2d,
)
from aaf.walls import MATERIALS, WALLS_2D, alphas_for  # noqa: E402

FS, N_TIME, MAX_ORDER = 4096, 8192, 60
SRC = [0.5, 0.5]
N_GRID, MARGIN = 8, 0.3
F_AXIS = np.arange(N_TIME // 2 + 1) * FS / N_TIME

# Thresholds (D47)
T1_DBW_MIN = 2.0          # Hz   -- M3 must broaden its own family
T1_DLVL_MAX = -2.0        # dB   -- ...and reduce its level
T2_SELECTIVITY_MIN = 5.0  # bandwidth selectivity (measured ~30-50)
T2B_WAVE_REGIME = 2.0     # the Kuttruff prediction
T3_M1_DBW_MAX = -0.5      # Hz   -- concrete must SHARPEN (own, ~9x looser thresholds)
T3_M1_DLVL_MIN = 0.5      # dB
SIGMA_BW_FLOOR = 0.15     # Hz   -- 0.3 bins; measured repeatability is ~0.06 Hz
T5_R2_MIN = 0.98


def receiver_grid(L, W):
    xs = np.linspace(MARGIN, L - MARGIN, N_GRID)
    ys = np.linspace(MARGIN, W - MARGIN, N_GRID)
    return np.array([[x, y] for y in ys for x in xs])   # row-major: outer y, inner x


def simulate(L, W, alphas, rx, max_order=MAX_ORDER):
    mats = {w: pra.Material(energy_absorption=float(a)) for w, a in zip(WALLS_2D, alphas)}
    room = pra.ShoeBox(p=[L, W], fs=FS, materials=mats, max_order=max_order,
                       ray_tracing=False)
    room.add_source(SRC)
    room.add_microphone_array(pra.MicrophoneArray(rx.T, FS))
    room.compute_rir()
    out = np.zeros((len(rx), N_TIME))
    for i, r in enumerate(room.rir):
        h = np.asarray(r[0])
        n = min(len(h), N_TIME)
        out[i, :n] = h[:n]
    return np.fft.rfft(out, n=N_TIME, axis=1)


def measure(L, W, alphas, rx, n_per_family=3):
    """Per-family mean bandwidth and level from the modal projection."""
    pr = project_field(simulate(L, W, alphas, rx), rx, L, W, src=SRC, fs=FS)
    bw_pred = [damping_to_bandwidth_hz(
        modal_damping_2d(L, W, alphas, m.n_x, m.n_y, model="ism_ray")) for m in pr.modes]
    peaks = measure_modes(pr.spectra, F_AXIS, pr.modes, caps=caps_from_predicted_bw(bw_pred))
    fam = {}
    for f in (X_AXIAL, Y_AXIAL, TANGENTIAL):
        idx = [i for i in pr.by_family(f) if peaks[i].bw_valid][:n_per_family]
        fam[f] = {
            "bw": float(np.mean([peaks[i].bw_3db_hz for i in idx])) if idx else float("nan"),
            "level": float(np.mean([peaks[i].level_db for i in idx])) if idx else float("nan"),
            "n": len(idx),
            "bw_each": [float(peaks[i].bw_3db_hz) for i in idx],
            "modes": [[peaks[i].n_x, peaks[i].n_y] for i in idx],
        }
    return fam, pr, peaks, bw_pred


# ------------------------------------------------------------------ G0 asserts
def g0_wall_convention():
    """Image-lattice probe: absorb ONLY west; the image at x=-x_src must carry sqrt(1-a)."""
    alpha = 0.75
    L, W = 4.5, 4.0
    mats = {w: pra.Material(energy_absorption=(alpha if w == "west" else 0.0))
            for w in WALLS_2D}
    room = pra.ShoeBox(p=[L, W], fs=FS, materials=mats, max_order=1, ray_tracing=False)
    room.add_source(SRC)
    room.add_microphone_array(pra.MicrophoneArray(np.array([[1.0], [1.0]]), FS))
    room.image_source_model()
    img = np.asarray(room.sources[0].images)
    dmp = np.asarray(room.sources[0].damping).ravel()
    expected = float(np.sqrt(1.0 - alpha))
    iw = np.flatnonzero(np.isclose(img[0], -SRC[0], atol=1e-9))
    ok = iw.size == 1 and abs(dmp[iw[0]] - expected) < 1e-6
    others = []
    for target, axis in ((2 * L - SRC[0], 0), (-SRC[1], 1), (2 * W - SRC[1], 1)):
        j = np.flatnonzero(np.isclose(img[axis], target, atol=1e-9))
        others.append(bool(j.size and abs(dmp[j[0]] - 1.0) < 1e-6))
    return {
        "pass": bool(ok and all(others)),
        "west_image_damping": float(dmp[iw[0]]) if iw.size else None,
        "expected_sqrt_1_minus_alpha": expected,
        "other_walls_undamped": all(others),
        "wall_names": list(room.wall_names),
    }


def g0_mirror_equivariance(rx):
    """west-absorbing with src at x0 must mirror east-absorbing with src at L-x0."""
    L, W = 4.5, 4.0
    global SRC
    keep = SRC
    try:
        SRC = [0.5, 0.5]
        Hw = simulate(L, W, alphas_for("west", "M3"), rx)
        SRC = [L - 0.5, 0.5]
        He = simulate(L, W, alphas_for("east", "M3"), rx)
    finally:
        SRC = keep
    A = np.abs(Hw).reshape(N_GRID, N_GRID, -1)
    B = np.abs(He).reshape(N_GRID, N_GRID, -1)[:, ::-1, :]     # mirror in x
    rel = float(np.linalg.norm(A - B) / max(np.linalg.norm(A), 1e-30))
    # control: comparing against a south edit must NOT match
    SRC2 = SRC
    Hs = simulate(L, W, alphas_for("south", "M3"), rx)
    ctrl = float(np.linalg.norm(np.abs(Hw) - np.abs(Hs)) / max(np.linalg.norm(np.abs(Hw)), 1e-30))
    return {"pass": rel < 1e-5 and ctrl > 1e-2, "rel_err": rel, "control_rel_err": ctrl}


def g0_max_order(rx):
    """max_order=60 must be converged: 60 vs 120 on the least absorptive config."""
    L, W = 4.5, 4.0
    al = alphas_for("west", "M1")             # concrete = longest tail = worst case
    out = {}
    for mo in (60, 120):
        pr = project_field(simulate(L, W, al, rx, max_order=mo), rx, L, W, src=SRC, fs=FS)
        bw_pred = [damping_to_bandwidth_hz(
            modal_damping_2d(L, W, al, m.n_x, m.n_y, model="ism_ray")) for m in pr.modes]
        pk = measure_modes(pr.spectra, F_AXIS, pr.modes, caps=caps_from_predicted_bw(bw_pred))
        idx = [i for i in pr.by_family(X_AXIAL) if pk[i].bw_valid][:3]
        out[mo] = (float(np.mean([pk[i].bw_3db_hz for i in idx])),
                   float(np.mean([pk[i].level_db for i in idx])))
    d_bw = abs(out[120][0] - out[60][0]) / max(out[60][0], 1e-9)
    d_lv = abs(out[120][1] - out[60][1])
    return {"pass": d_bw <= 0.05 and d_lv <= 0.25, "bw_60": out[60][0], "bw_120": out[120][0],
            "rel_bw_change": d_bw, "level_60": out[60][1], "level_120": out[120][1],
            "abs_level_change_db": d_lv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/p3_2")
    ap.add_argument("--quick", action="store_true", help="skip the second room")
    args = ap.parse_args()
    out_dir = Path(args.out)
    (out_dir / "gate").mkdir(parents=True, exist_ok=True)

    report = {"settings": {"fs": FS, "n_time": N_TIME, "max_order": MAX_ORDER,
                           "src": SRC, "grid": N_GRID, "margin": MARGIN,
                           "pra_version": pra.__version__,
                           "materials": MATERIALS, "walls": list(WALLS_2D)}}
    verdict = "PASS"
    reasons = []

    # ---------------- Stage 0: provenance -------------------------------------------
    rx45 = receiver_grid(4.5, 4.0)
    g0 = {"wall_convention": g0_wall_convention(),
          "mirror_equivariance": g0_mirror_equivariance(rx45),
          "max_order": g0_max_order(rx45)}
    report["G0"] = g0
    for k, v in g0.items():
        if not v["pass"]:
            verdict, _ = "STOP", reasons.append(f"G0.{k} failed")

    # ---------------- Stage 1: measurement ------------------------------------------
    rooms = [(4.5, 4.0)] + ([] if args.quick else [(5.6, 3.4)])
    per_room = {}
    for (L, W) in rooms:
        rx = receiver_grid(L, W)
        base, pr, _, _ = measure(L, W, alphas_for(), rx)
        entry = {"cond_phi": pr.cond, "residual_frac": pr.residual_frac,
                 "n_modes": len(pr.modes), "n_excited": int(pr.used.sum()),
                 "baseline": {k: {kk: vv for kk, vv in v.items()} for k, v in base.items()},
                 "edits": {}}
        for wall in WALLS_2D:
            for mat in ("M1", "M2", "M3"):
                fam, _, _, _ = measure(L, W, alphas_for(wall, mat), rx)
                entry["edits"][f"{wall}_{mat}"] = {
                    f: {"bw": fam[f]["bw"], "level": fam[f]["level"],
                        "d_bw": fam[f]["bw"] - base[f]["bw"],
                        "d_level": fam[f]["level"] - base[f]["level"]}
                    for f in (X_AXIAL, Y_AXIAL, TANGENTIAL)
                }
        per_room[f"{L}x{W}"] = entry
    report["rooms"] = per_room

    # ---------------- Stage 2: decision rule ----------------------------------------
    def own_other(wall):
        return (X_AXIAL, Y_AXIAL) if wall in ("west", "east") else (Y_AXIAL, X_AXIAL)

    sel = []
    tests = {}
    for rname, entry in per_room.items():
        for wall in WALLS_2D:
            own, other = own_other(wall)
            e = entry["edits"][f"{wall}_M3"]
            s = abs(e[own]["d_bw"]) / max(abs(e[other]["d_bw"]), SIGMA_BW_FLOOR)
            sel.append(s)
            tests.setdefault("T1_direction", []).append(
                bool(e[own]["d_bw"] >= T1_DBW_MIN and e[own]["d_level"] <= T1_DLVL_MAX))
            tests.setdefault("T2_selectivity", []).append(bool(s >= T2_SELECTIVITY_MIN))
            tests.setdefault("T4_orthogonal_flip", []).append(
                bool(abs(e[own]["d_bw"]) > abs(e[other]["d_bw"])))
            m1 = entry["edits"][f"{wall}_M1"]
            tests.setdefault("T3_bidirectional", []).append(
                bool(m1[own]["d_bw"] <= T3_M1_DBW_MAX and m1[own]["d_level"] >= T3_M1_DLVL_MIN))
            b = entry["baseline"]
            tests.setdefault("T3_monotonic", []).append(bool(
                entry["edits"][f"{wall}_M1"][own]["bw"] < b[own]["bw"]
                < entry["edits"][f"{wall}_M2"][own]["bw"]
                < entry["edits"][f"{wall}_M3"][own]["bw"]))

    sel = np.array(sel, dtype=float)
    boot = np.array([np.mean(np.random.default_rng(s).choice(sel, sel.size, replace=True))
                     for s in range(2000)])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    report["selectivity"] = {"per_wall_room": sel.tolist(), "mean": float(sel.mean()),
                             "ci95": list(ci), "threshold": T2_SELECTIVITY_MIN}
    report["tests"] = {k: {"all_pass": bool(all(v)), "n_pass": int(sum(v)), "n": len(v)}
                       for k, v in tests.items()}

    # ---------------- T5: which damping law? ----------------------------------------
    xs, ys, xs_k = [], [], []
    for rname, entry in per_room.items():
        L, W = [float(t) for t in rname.split("x")]
        for wall in WALLS_2D:
            own, _ = own_other(wall)
            n = (1, 0) if own == X_AXIAL else (0, 1)
            for tag, al in [("M0", alphas_for())] + [(m, alphas_for(wall, m))
                                                     for m in ("M1", "M2", "M3")]:
                meas = (entry["baseline"][own]["bw"] if tag == "M0"
                        else entry["edits"][f"{wall}_{tag}"][own]["bw"])
                if not np.isfinite(meas):
                    continue
                xs.append(damping_to_bandwidth_hz(
                    modal_damping_2d(L, W, al, n[0], n[1], model="ism_ray")))
                xs_k.append(damping_to_bandwidth_hz(
                    modal_damping_2d(L, W, al, n[0], n[1], model="kuttruff")))
                ys.append(meas)
    xs, ys, xs_k = np.array(xs), np.array(ys), np.array(xs_k)

    def fit(x, y):
        A = np.polyfit(x, y, 1)
        pred = np.polyval(A, x)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-30
        return {"slope": float(A[0]), "intercept": float(A[1]),
                "r2": 1.0 - ss_res / ss_tot, "rss": ss_res}

    f_ray, f_kut = fit(xs, ys), fit(xs_k, ys)
    n = len(ys)
    d_aic = n * np.log(f_kut["rss"] / max(f_ray["rss"], 1e-30))
    report["T5_calibration"] = {"ism_ray": f_ray, "kuttruff": f_kut,
                                "delta_aic_kuttruff_minus_ray": float(d_aic),
                                "favours": "ism_ray" if d_aic > 0 else "kuttruff"}
    tests["T5_theory_fit"] = [bool(f_ray["r2"] >= T5_R2_MIN)]
    report["tests"]["T5_theory_fit"] = {"all_pass": bool(f_ray["r2"] >= T5_R2_MIN),
                                        "n_pass": int(f_ray["r2"] >= T5_R2_MIN), "n": 1}

    # ---------------- verdict -------------------------------------------------------
    if verdict != "STOP":
        if not report["tests"]["T1_direction"]["all_pass"]:
            verdict, _ = "STOP", reasons.append("T1: no directional signal / wrong sign")
        elif ci[0] > T2_SELECTIVITY_MIN:
            verdict = "PASS"
        elif ci[0] > 1.0:
            verdict = "PASS-WITH-AMENDMENT"
            reasons.append(
                f"selectivity CI {ci} straddles the ray/wave boundary; the signal is real "
                f"and monotone but weaker than the ray model predicts -- amend the claim "
                f"(see D48), do not stop")
        else:
            verdict, _ = "STOP", reasons.append(f"selectivity CI {ci} includes 1.0")
    report["verdict"] = verdict
    report["reasons"] = reasons

    # ---------------- figure --------------------------------------------------------
    entry = per_room[f"{rooms[0][0]}x{rooms[0][1]}"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    mats_order = ["M1", "M0", "M2", "M3"]
    av = [MATERIALS[m] for m in mats_order]
    for fam, style in ((X_AXIAL, "o-"), (Y_AXIAL, "s--")):
        bw = [entry["baseline"][fam]["bw"] if m == "M0"
              else entry["edits"][f"west_{m}"][fam]["bw"] for m in mats_order]
        lv = [entry["baseline"][fam]["level"] if m == "M0"
              else entry["edits"][f"west_{m}"][fam]["level"] for m in mats_order]
        ax[0].plot(av, bw, style, label=fam)
        ax[1].plot(av, lv, style, label=fam)
    ax[0].set(xlabel=r"$\alpha_{west}$", ylabel="-3 dB bandwidth (Hz)",
              title="Editing the WEST wall broadens\nonly the x-axial family")
    ax[1].set(xlabel=r"$\alpha_{west}$", ylabel="peak level (dB)", title="Peak level")
    for a in ax[:2]:
        a.grid(alpha=.3); a.legend()
    ax[2].scatter(xs, ys, s=26, label="measured")
    xr = np.linspace(0, max(xs.max(), 1e-9), 50)
    ax[2].plot(xr, np.polyval([f_ray["slope"], f_ray["intercept"]], xr), "r-",
               label=f"ISM-ray fit $R^2$={f_ray['r2']:.4f}")
    ax[2].scatter(xs_k, ys, s=18, marker="x", alpha=.6,
                  label=f"vs Kuttruff ($R^2$={f_kut['r2']:.3f})")
    ax[2].set(xlabel="predicted BW = $\\gamma/\\pi$ (Hz)", ylabel="measured BW (Hz)",
              title="Which damping law?  $\\Delta$AIC=%.0f" % d_aic)
    ax[2].grid(alpha=.3); ax[2].legend(fontsize=8)
    fig.suptitle(f"P3-2 physics gate — VERDICT: {verdict}   (selectivity "
                 f"{sel.mean():.1f}, CI [{ci[0]:.1f}, {ci[1]:.1f}])", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "gate" / "fig_A_gate.png", dpi=120)
    plt.close(fig)

    (out_dir / "gate" / "gate.json").write_text(json.dumps(report, indent=2, default=float))
    _write_markdown(out_dir / "SIM_VALIDATION.md", report, per_room, rooms)

    print(json.dumps({"verdict": verdict, "selectivity_mean": float(sel.mean()),
                      "selectivity_ci": ci, "reasons": reasons,
                      "tests": report["tests"]}, indent=2))
    return {"PASS": 0, "PASS-WITH-AMENDMENT": 2, "STOP": 3}[verdict]


def _write_markdown(path, report, per_room, rooms):
    v = report["verdict"]
    L0, W0 = rooms[0]
    e = per_room[f"{L0}x{W0}"]
    lines = [
        f"# P3-2 simulator validation — BLOCKING GATE: **{v}**", "",
        f"pyroomacoustics {report['settings']['pra_version']}, fs={FS}, N={N_TIME}, "
        f"max_order={MAX_ORDER}, source={SRC}, {N_GRID}x{N_GRID} receivers (margin {MARGIN} m).",
        "",
        "Measured on ISM ground truth via the 64-receiver modal projection "
        "(`aaf.eval.modal_projection`), so each measurement is attributable to a single "
        "(n_x, n_y) mode rather than to whatever dominates one receiver's spectrum.", "",
        "## Provenance asserts (G0)", "",
        "| check | result | detail |", "|---|---|---|",
    ]
    g = report["G0"]
    lines += [
        f"| wall convention (image lattice) | {'PASS' if g['wall_convention']['pass'] else 'FAIL'} "
        f"| west image damping {g['wall_convention']['west_image_damping']:.4f} "
        f"= sqrt(1-a) {g['wall_convention']['expected_sqrt_1_minus_alpha']:.4f}; "
        f"other walls undamped |",
        f"| mirror equivariance | {'PASS' if g['mirror_equivariance']['pass'] else 'FAIL'} "
        f"| rel err {g['mirror_equivariance']['rel_err']:.2e} "
        f"(control {g['mirror_equivariance']['control_rel_err']:.2e}) |",
        f"| max_order 60 converged | {'PASS' if g['max_order']['pass'] else 'FAIL'} "
        f"| BW {g['max_order']['bw_60']:.3f} -> {g['max_order']['bw_120']:.3f} Hz "
        f"({100*g['max_order']['rel_bw_change']:.1f}%), level "
        f"{g['max_order']['abs_level_change_db']:.3f} dB |",
        "",
        f"## Selectivity (room {L0}x{W0}, cond(Phi)={e['cond_phi']:.3f})", "",
        "Bandwidth deltas vs that room's own baseline, per mode family:", "",
        "| edit | dBW x-axial | dBW y-axial | dBW tangential | dLevel x | dLevel y |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for wall in WALLS_2D:
        for mat in ("M1", "M2", "M3"):
            d = e["edits"][f"{wall}_{mat}"]
            lines.append(
                f"| {wall} -> {mat} (a={MATERIALS[mat]}) | {d[X_AXIAL]['d_bw']:+.3f} | "
                f"{d[Y_AXIAL]['d_bw']:+.3f} | {d[TANGENTIAL]['d_bw']:+.3f} | "
                f"{d[X_AXIAL]['d_level']:+.2f} | {d[Y_AXIAL]['d_level']:+.2f} |")
    s = report["selectivity"]
    t5 = report["T5_calibration"]
    lines += [
        "", "## Decision", "",
        "| test | result | n |", "|---|---|---|",
    ] + [
        f"| {k} | {'PASS' if val['all_pass'] else 'FAIL'} | {val['n_pass']}/{val['n']} |"
        for k, val in report["tests"].items()
    ] + [
        "",
        f"**Bandwidth selectivity** = {s['mean']:.1f} (95% CI [{s['ci95'][0]:.1f}, "
        f"{s['ci95'][1]:.1f}]), threshold {s['threshold']}.", "",
        f"**Which damping law?** ISM-ray fit R^2 = {t5['ism_ray']['r2']:.4f} "
        f"(BW = {t5['ism_ray']['intercept']:.3f} + {t5['ism_ray']['slope']:.3f}*gamma/pi); "
        f"Kuttruff R^2 = {t5['kuttruff']['r2']:.4f}. dAIC = "
        f"{t5['delta_aic_kuttruff_minus_ray']:.0f} favouring **{t5['favours']}**.", "",
        "> **Scoping (D48).** ISM uses angle-independent reflection and so has no "
        "grazing-incidence absorption: a purely axial mode is damped only by the wall pair "
        "it bounces between. Real locally-reacting walls follow Kuttruff and would give "
        "~2:1 selectivity with no invariant family. The claim this chunk can support is "
        "therefore *the model learns the simulator's per-wall law*.", "",
    ]
    if report["reasons"]:
        lines += ["**Notes:** " + "; ".join(report["reasons"]), ""]
    lines.append("Sources: `outputs/p3_2/gate/gate.json`, `outputs/p3_2/gate/fig_A_gate.png`.")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
