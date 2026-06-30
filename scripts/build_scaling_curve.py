"""P2-4: build the coverage-density scaling curve + SCALING.md from per-density
known-geometry evals on the frozen interior test set. Robust to partial completion
(missing densities are skipped). CPU only; every number re-read from disk."""
from __future__ import annotations
import json, glob, os
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
CC = REPO / "outputs/coverage_curve"
LOO_FULL, LOO_MODAL = 0.894, 0.938   # P2-3.5 LOO ceiling (training density)
DENSITIES = [45, 90, 150, 250]
TRAIN_DIR = {45: "P3_45rooms_4gpu", 90: "density_90", 150: "density_150", 250: "density_250"}


def _eval_means(label):
    """Mean over frozen test rooms of known-geometry (RBF) mag full/modal, phase, RIR."""
    paths = sorted(glob.glob(str(CC / f"eval_{label}/lookup/L*__rbf/metrics.json")))
    if not paths:
        return None
    full, modal, phase, rir = [], [], [], []
    for p in paths:
        m = json.loads(Path(p).read_text())
        pb = m["per_band_mag_corr"]; sm = m.get("signal_metrics", {})
        full.append(pb["mag_corr_full"]); modal.append(pb["mag_corr_0_250"])
        if "phase_corr_mw" in sm: phase.append(sm["phase_corr_mw"])
        if "rir_pearson" in sm: rir.append(sm["rir_pearson"])
    return dict(n=len(paths),
                full=float(np.mean(full)), full_sd=float(np.std(full)),
                modal=float(np.mean(modal)),
                phase=float(np.mean(phase)) if phase else float("nan"),
                rir=float(np.mean(rir)) if rir else float("nan"))


def _frozen_keys():
    """The 15 frozen test rooms as L{L:.2f}_W{W:.2f}_H{H:.2f} dir names."""
    rooms = json.loads((CC / "test_nn_distances.json").read_text())["test_rooms"]
    return [f"L{r['L']:.2f}_W{r['W']:.2f}_H{r['H']:.2f}" for r in rooms]


def _fewshot_means(train_subdir):
    """SECONDARY: few-shot 8-measurement route on the FROZEN rooms only.
    Reads outputs/multi_room_3d/<train_subdir>/zero_shot/<frozen>/metrics.json
    (filters out any legacy non-frozen rooms sharing the dir). Returns mean
    full-band mag_corr (signal_metrics.mag_corr), phase, RIR, held-out LSD."""
    zsd = REPO / f"outputs/multi_room_3d/{train_subdir}/zero_shot"
    if not zsd.exists():
        return None
    mag, phase, rir, lsd = [], [], [], []
    for key in _frozen_keys():
        f = zsd / key / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text()); sm = m.get("signal_metrics", {})
        if "mag_corr" in sm: mag.append(sm["mag_corr"])
        if "phase_corr_mw" in sm: phase.append(sm["phase_corr_mw"])
        if "rir_pearson" in sm: rir.append(sm["rir_pearson"])
        if m.get("held_out_lsd_db") is not None: lsd.append(m["held_out_lsd_db"])
    if not mag:
        return None
    return dict(n=len(mag),
                mag=float(np.mean(mag)), mag_sd=float(np.std(mag)),
                phase=float(np.mean(phase)) if phase else float("nan"),
                rir=float(np.mean(rir)) if rir else float("nan"),
                lsd=float(np.mean(lsd)) if lsd else float("nan"))


def _indist_lsd(train_subdir):
    sc = REPO / f"outputs/multi_room_3d/{train_subdir}/scalars.json"
    if not sc.exists():
        return None, None
    rows = [r for r in json.loads(sc.read_text()) if r.get("phase") == "val" and r.get("lsd_db") is not None]
    if not rows:
        return None, None
    rows.sort(key=lambda r: r.get("iter", 0))
    return float(rows[-1]["lsd_db"]), int(rows[-1].get("iter", 0))


def main():
    nn = json.loads((CC / "test_nn_distances.json").read_text())["mean_nn_m"]
    pts = []
    for d in DENSITIES:
        ev = _eval_means(f"density_{d}")
        fs = _fewshot_means(TRAIN_DIR[d])
        lsd, it = _indist_lsd(TRAIN_DIR[d])
        pts.append(dict(rooms=d, nn=nn[str(d)], ev=ev, fs=fs, indist_lsd=lsd, indist_iter=it))
    done = [p for p in pts if p["ev"]]
    print(f"# densities with eval: {[p['rooms'] for p in done]}")

    # ---- figure ----
    if done:
        fig, axes = plt.subplots(1, 2, figsize=(19.2, 10.8), constrained_layout=True)
        ax = axes[0]
        rooms = [p["rooms"] for p in done]
        full = [p["ev"]["full"] for p in done]; modal = [p["ev"]["modal"] for p in done]
        sd = [p["ev"]["full_sd"] for p in done]
        ax.errorbar(rooms, full, yerr=sd, fmt="-o", color="#1f77b4", lw=2.5, ms=9, capsize=5,
                    label="known-geometry mag corr (full)")
        ax.plot(rooms, modal, "-s", color="#ff7f0e", lw=2.0, ms=8, label="known-geometry mag corr (0-250 Hz modal)")
        # secondary: few-shot 8-measurement route (where available)
        fdone = [p for p in done if p.get("fs")]
        if fdone:
            ax.plot([p["rooms"] for p in fdone], [p["fs"]["mag"] for p in fdone],
                    "--^", color="#9467bd", lw=1.6, ms=7, alpha=0.8,
                    label="few-shot 8-measurement mag corr (secondary)")
        ax.axhline(LOO_FULL, color="#2ca02c", ls=":", lw=1.5, label=f"LOO ceiling full ({LOO_FULL:.2f})")
        ax.axhline(LOO_MODAL, color="#2ca02c", ls="--", lw=1.0, alpha=0.6, label=f"LOO ceiling modal ({LOO_MODAL:.2f})")
        ax.axhline(0.9, color="green", ls="-.", lw=0.8, alpha=0.5)
        for p in done:
            ax.annotate(f"{p['ev']['full']:.2f}", (p["rooms"], p["ev"]["full"]),
                        textcoords="offset points", xytext=(6, 8), fontsize=12, fontweight="bold")
        ax.set_xlabel("training-room count"); ax.set_ylabel("known-geometry zero-shot mag corr (mean over 15 frozen rooms)")
        ax.set_ylim(0, 1.0); ax.set_xticks(rooms); ax.grid(True, alpha=0.3); ax.legend(loc="center right", fontsize=12)
        # twin x: mean NN-distance
        axt = ax.twiny(); axt.set_xlim(ax.get_xlim())
        axt.set_xticks(rooms); axt.set_xticklabels([f"{p['nn']:.2f}" for p in done])
        axt.set_xlabel("mean nearest-trained-room distance (m)")
        # right panel: in-distribution control
        ax2 = axes[1]
        di = [p for p in done if p["indist_lsd"]]
        if di:
            ax2.plot([p["rooms"] for p in di], [p["indist_lsd"] for p in di], "-o", color="#d62728", lw=2.5, ms=9)
            ax2.axhline(2.5, color="green", ls=":", lw=1, label="target 2.5 dB")
            for p in di:
                ax2.annotate(f"{p['indist_lsd']:.2f}", (p["rooms"], p["indist_lsd"]),
                             textcoords="offset points", xytext=(6, 8), fontsize=12)
            ax2.set_xticks([p["rooms"] for p in di])
        ax2.set_xlabel("training-room count"); ax2.set_ylabel("in-distribution val LSD (dB) — the convergence control")
        ax2.grid(True, alpha=0.3); ax2.legend(loc="upper left", fontsize=12)
        ax2.set_title("Control: did in-distribution hold as rooms scaled?")
        fig.suptitle("Coverage-density scaling: known-geometry zero-shot fidelity vs training-room count "
                     "(frozen interior test set)", fontweight="bold")
        fig.text(0.5, 0.005, "Each point: mean over 15 frozen interior test rooms, known-geometry route (predict latent "
                 "from (L,W,H), no measurements). LOO ceiling = P2-3.5 0.89/0.94. Right: in-distribution val LSD per density "
                 "(if it rose, the high-density zero-shot is a lower bound). Source: outputs/coverage_curve/eval_*.",
                 ha="center", fontsize=11, style="italic", color="#444", wrap=True)
        p = CC / "scaling_curve.png"; fig.savefig(p, dpi=100); plt.close(fig)
        from PIL import Image
        print(f"# wrote {p} {Image.open(p).size}")

    # ---- SCALING.md ----
    md = ["# P2-4 — Coverage-density scaling curve\n",
          "Known-geometry zero-shot fidelity (predict latent from (L,W,H), no measurements) on a FROZEN "
          "interior test set (15 rooms, inside the 45-hull, interpolative at every density), as training-room "
          "count scales 45→90→150→250 with the recipe frozen.\n",
          "\n## Scaling table\n",
          "| rooms | mean NN-dist (m) | in-dist val LSD (dB) | mag corr full | mag corr modal (0-250) | phase corr | RIR Pearson |",
          "|---:|---:|---:|---:|---:|---:|---:|"]
    for p in pts:
        e = p["ev"]
        if e:
            md.append(f"| {p['rooms']} | {p['nn']:.3f} | "
                      f"{p['indist_lsd']:.2f}" + (f" @{p['indist_iter']//1000}K" if p['indist_iter'] else "") + " | "
                      f"{e['full']:.3f} | {e['modal']:.3f} | {e['phase']:.3f} | {e['rir']:.3f} |")
        else:
            md.append(f"| {p['rooms']} | {p['nn']:.3f} | (pending) | — | — | — | — |")
    md.append(f"\n**LOO ceiling (P2-3.5, training density):** mag corr {LOO_FULL:.3f} full / {LOO_MODAL:.3f} modal.\n")

    # ---- secondary: few-shot 8-measurement route (for completeness) ----
    if any(p.get("fs") for p in pts):
        md.append("\n## Few-shot 8-measurement route (secondary, for completeness)\n")
        md.append("Same frozen rooms, but the latent is fitted from 8 observed receivers (test-time "
                  "optimization) instead of predicted from (L,W,H). Reported for completeness; the headline "
                  "is the known-geometry route above.\n")
        md.append("| rooms | n | mag corr (full) | phase corr (mw) | RIR Pearson | held-out LSD (dB) |")
        md.append("|---:|---:|---:|---:|---:|---:|")
        for p in pts:
            fs = p.get("fs")
            if fs:
                md.append(f"| {p['rooms']} | {fs['n']} | {fs['mag']:.3f} | {fs['phase']:.3f} | "
                          f"{fs['rir']:.3f} | {fs['lsd']:.2f} |")
            else:
                md.append(f"| {p['rooms']} | 0 | (pending) | — | — | — |")

    md.append("\n![scaling curve](scaling_curve.png)\n")
    # headline verdict (auto, refined by hand in CHUNK doc)
    if len(done) >= 2:
        f0, fN = done[0]["ev"]["full"], done[-1]["ev"]["full"]
        md.append(f"\n## Reading\n\n- Known-geometry mag corr moves **{f0:.2f} → {fN:.2f}** (full) from "
                  f"{done[0]['rooms']} → {done[-1]['rooms']} rooms; the LOO ceiling is {LOO_FULL:.2f}.\n")
        di = [p for p in done if p["indist_lsd"]]
        if di:
            worst = max(di, key=lambda p: p["indist_lsd"])
            md.append(f"- **Convergence control**: in-distribution val LSD ranges "
                      f"{min(p['indist_lsd'] for p in di):.2f}–{worst['indist_lsd']:.2f} dB; "
                      + ("held near target — the zero-shot trend is clean coverage signal."
                         if worst["indist_lsd"] <= 2.6 else
                         f"degraded at {worst['rooms']} rooms — that point's zero-shot is a LOWER BOUND (undertrained).") + "\n")
        md.append("- Saturation / P3-1 setup / recommendation: see tasks/CHUNK_P2_4_RESULTS.md.\n")
    (CC / "SCALING.md").write_text("\n".join(md))
    print(f"# wrote {CC / 'SCALING.md'}")


if __name__ == "__main__":
    main()
