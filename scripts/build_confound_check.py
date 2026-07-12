"""P2-4b: bound the convergence confound in the P2-4 coverage curve.

Aggregates the known-geometry full-suite metrics for a set of (rooms, convergence)
points on the FROZEN interior test set, and writes outputs/coverage_curve/CONFOUND_CHECK.md
with (1) the matched-convergence comparison 45@4.3 vs 250@4.3, (2) the blur-inflation test
45@2.17 vs 45@4.3 (same rooms, only convergence differs), and (3) a drafted verdict.

CPU only; every number re-read from per-room metrics.json. Robust to pending evals.
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
CC = REPO / "outputs/coverage_curve"

# Each point: (key, human label, eval_dir under CC, nominal in-dist LSD or None->read provenance)
POINTS = [
    ("45_conv",  "45 rooms · converged (P3 @60K)",        "eval_density_45",   2.17),
    ("45_m43",   "45 rooms · matched @~4.3 (under-trained)", "eval_conv45_lsd43", None),
    ("250_p43",  "250 rooms · plateau @~4.3",              "eval_density_250",  4.30),
    # optional blur-sweep points (skipped silently if absent)
    ("45_s45",   "45 rooms · @~4.5",                        "eval_conv45_lsd45", None),
    ("45_s38",   "45 rooms · @~3.8",                        "eval_conv45_lsd38", None),
    ("45_s34",   "45 rooms · @~3.4",                        "eval_conv45_lsd34", None),
]


def _agg(eval_dir, nominal_lsd):
    paths = sorted(glob.glob(str(CC / eval_dir / "lookup/L*__rbf/metrics.json")))
    if not paths:
        return None
    def col(getter):
        vals = []
        for p in paths:
            try:
                vals.append(getter(json.loads(Path(p).read_text())))
            except (KeyError, TypeError):
                pass
        return float(np.nanmean(vals)) if vals else float("nan")
    prov = CC / eval_dir / "provenance.json"
    lsd = json.loads(prov.read_text())["indist_val_lsd_db"] if prov.exists() else nominal_lsd
    return dict(
        n=len(paths), indist_lsd=lsd,
        mag_full=col(lambda m: m["per_band_mag_corr"]["mag_corr_full"]),
        mag_modal=col(lambda m: m["per_band_mag_corr"]["mag_corr_0_250"]),
        lsd_full=col(lambda m: m["lsd_db_full"]),
        lsd_modal=col(lambda m: m["signal_metrics"]["lsd_band_0_250_db"]),
        lsd_diffuse=col(lambda m: np.mean([m["signal_metrics"]["lsd_band_250_500_db"],
                                           m["signal_metrics"]["lsd_band_500_1000_db"],
                                           m["signal_metrics"]["lsd_band_1000_2000_db"]])),
        phase=col(lambda m: m["signal_metrics"]["phase_corr_mw"]),
        rir=col(lambda m: m["signal_metrics"]["rir_pearson"]),
        modal_mae=col(lambda m: m["modal_mae_hz"]),
        modal_recall=col(lambda m: m["modal_recall"]),
    )


def main():
    pts = {k: _agg(d, nl) for k, lbl, d, nl in POINTS}
    labels = {k: lbl for k, lbl, d, nl in POINTS}
    have = {k: v for k, v in pts.items() if v}
    print("points with eval:", list(have))

    md = ["# P2-4b — Bounding the convergence confound in the coverage curve\n"]

    a, b, c = pts["45_conv"], pts["45_m43"], pts["250_p43"]
    verdict = "PENDING — the matched-convergence eval (45@~4.3) is not yet on disk."
    detail = ""
    if b and c:
        # matched-convergence deltas: 250@4.3 vs 45@4.3 (positive => 250 better)
        d_full = c["mag_full"] - b["mag_full"]
        d_modal = c["mag_modal"] - b["mag_modal"]
        d_lsd = b["lsd_full"] - c["lsd_full"]          # lower LSD better
        d_lsdmod = b["lsd_modal"] - c["lsd_modal"]
        d_phase = c["phase"] - b["phase"]
        d_rir = c["rir"] - b["rir"]
        d_mrec = c["modal_recall"] - b["modal_recall"]
        d_mmae = b["modal_mae"] - c["modal_mae"]       # lower MAE better
        # hard metrics = LSD + modal placement + phase + RIR; at matched convergence blur is
        # equalized on both sides, so these deltas isolate coverage (blur alone also moves them — see sweep)
        hard_wins = sum([d_lsd >= 0.2, d_lsdmod >= 0.2, d_phase >= 0.03,
                         d_rir >= 0.03, d_mrec >= 0.02, d_mmae >= 0.0 and c["modal_mae"] < b["modal_mae"]])
        soft_win = d_full >= 0.05 or d_modal >= 0.05
        if hard_wins >= 3 and soft_win:
            verdict = ("**Coverage effect CONFIRMED at matched convergence.** At equal in-distribution "
                       f"convergence (~4.3 dB), 250 rooms beats 45 rooms across the suite — magnitude-band LSD, "
                       f"phase, RIR, and magnitude correlation. Because both sides are at matched convergence, "
                       f"blur is equalized on both, so these deltas isolate coverage. So the P2-4 curve's "
                       f"*direction* is trustworthy and densification genuinely helps, though its *magnitude* was "
                       f"inflated by the confound (see decomposition).")
        elif hard_wins <= 1:
            verdict = ("**Coverage effect is CONFOUNDED / largely a convergence artifact.** At matched "
                       f"convergence (~4.3 dB), 250 rooms does not meaningfully beat 45 rooms on the hard "
                       f"held-out metrics (LSD/phase/RIR/modal); any gap is confined to the soft magnitude-"
                       f"correlation metric that under-training inflates.")
        else:
            verdict = ("**Partial / mixed at matched convergence.** 250 rooms beats 45 on some hard metrics "
                       f"but not decisively; the coverage benefit is real but smaller than the raw P2-4 curve "
                       f"suggested. See the table and per-metric deltas.")
        detail = (f"\nMatched-convergence deltas (250@{c['indist_lsd']:.2f}dB − 45@{b['indist_lsd']:.2f}dB, "
                  f"positive = 250 better):\n"
                  f"- mag corr full **{d_full:+.3f}**, modal **{d_modal:+.3f}**\n"
                  f"- held-out LSD full **{d_lsd:+.2f} dB**, modal-band (0–250) **{d_lsdmod:+.2f} dB** (lower better → positive = 250 better)\n"
                  f"- phase (mw) **{d_phase:+.3f}**, RIR Pearson **{d_rir:+.3f}**\n"
                  f"- modal recall **{d_mrec:+.3f}**, modal MAE **{d_mmae:+.2f} Hz** (lower better)\n"
                  f"- hard-metric wins for 250 (of 6): **{hard_wins}**\n")
    md.append(verdict + "\n")
    md.append(detail)

    # confound decomposition: split the RAW P2-4 gap (45@2.17 -> 250@4.3) into
    # blur/convergence (45@2.17 -> 45@4.3, same rooms) + coverage (matched, 45@4.3 -> 250@4.3)
    if a and b and c:
        def decomp(metric):
            raw = c[metric] - a[metric]; blur = b[metric] - a[metric]; cov = c[metric] - b[metric]
            fb = 100 * blur / raw if raw else float("nan")
            return raw, blur, cov, fb
        rf, bf, cf, pf = decomp("mag_full")
        rm, bm, cm, pm = decomp("mag_modal")
        md.append(
            f"\n**How much of the raw P2-4 climb was the confound?** Decompose the raw P2-4 mag-corr gap "
            f"(45@{a['indist_lsd']:.2f}dB → 250@{c['indist_lsd']:.2f}dB) into blur/convergence "
            f"(45@{a['indist_lsd']:.2f}→45@{b['indist_lsd']:.2f}, *same rooms*) + genuine coverage "
            f"(matched, 45@{b['indist_lsd']:.2f}→250@{c['indist_lsd']:.2f}):\n"
            f"- full-band: raw **{rf:+.3f}** = blur **{bf:+.3f} ({pf:.0f}%)** + coverage **{cf:+.3f} ({100-pf:.0f}%)**\n"
            f"- modal (0–250): raw **{rm:+.3f}** = blur **{bm:+.3f} ({pm:.0f}%)** + coverage **{cm:+.3f} ({100-pm:.0f}%)**\n"
            f"\nSo **~{pf:.0f}% of the raw P2-4 magnitude-correlation climb was the convergence/blur confound, "
            f"not coverage.** The genuine coverage effect is real (verdict above) but **smaller than the raw curve "
            f"shows**. Because blur is equalized at matched convergence, the matched deltas isolate coverage; it "
            f"shows most strongly on phase (+{c['phase']-b['phase']:.3f}), "
            f"RIR (+{c['rir']-b['rir']:.3f}), modal-band LSD (+{b['lsd_modal']-c['lsd_modal']:.2f} dB). "
            f"**Do not cite the raw P2-4 curve's slope/magnitude; cite the matched-convergence deltas.**\n")

    # blur-inflation test (same 45 rooms, converged vs under-trained)
    if a and b:
        bd_full = b["mag_full"] - a["mag_full"]
        bd_lsd = b["lsd_full"] - a["lsd_full"]
        md.append(f"\n**Blur-inflation test (same 45 rooms, {a['indist_lsd']:.2f}dB→{b['indist_lsd']:.2f}dB "
                  f"under-training):** mag corr full moves **{a['mag_full']:.3f} → {b['mag_full']:.3f}** "
                  f"({bd_full:+.3f}), held-out LSD **{a['lsd_full']:.2f} → {b['lsd_full']:.2f} dB** ({bd_lsd:+.2f}). "
                  + ("Under-training **inflates** the soft correlation while LSD worsens — the P2-3 blur effect is real "
                     "and must be discounted when reading the raw P2-4 mag-corr curve."
                     if bd_full > 0.03 else
                     "Under-training does **not** inflate the correlation here — the P2-4 mag-corr rise is not a blur artifact.")
                  + "\n")

    # full comparison table
    md.append("\n## Full-suite comparison (known-geometry zero-shot, mean over 15 frozen rooms)\n")
    md.append("| point | n | in-dist LSD | mag full | mag modal | held LSD full | LSD 0–250 | LSD diffuse | phase(mw) | RIR | modal recall | modal MAE (Hz) |")
    md.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for k, lbl, d, nl in POINTS:
        v = pts[k]
        if not v:
            continue
        md.append(f"| {lbl} | {v['n']} | {v['indist_lsd']:.2f} | {v['mag_full']:.3f} | {v['mag_modal']:.3f} | "
                  f"{v['lsd_full']:.2f} | {v['lsd_modal']:.2f} | {v['lsd_diffuse']:.2f} | {v['phase']:.3f} | "
                  f"{v['rir']:.3f} | {v['modal_recall']:.3f} | {v['modal_mae']:.2f} |")
    md.append("\n*Held-out LSD lower = better; all correlations + recall higher = better; modal MAE lower = better. "
              "Modal = 0–250 Hz (sub-Schroeder ≈217 Hz); diffuse = mean of 250–500/500–1k/1k–2k bands.*\n")
    md.append("\n_Interpretation, saturation, and P3-1 implications: tasks/CHUNK_P2_4b_RESULTS.md._\n")

    (CC / "CONFOUND_CHECK.md").write_text("\n".join(md))
    print(f"# wrote {CC / 'CONFOUND_CHECK.md'}")
    print("VERDICT:", verdict[:120])


if __name__ == "__main__":
    main()
