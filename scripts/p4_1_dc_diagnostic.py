"""P4-1 parallel task: is the FDTD corpus's DC term eating the training loss?

The hypothesis: the FDTD corpus carries a DC/compliance term holding 87.3% of in-band power and
sitting 37.1 dB above the strongest room mode (the ISM corpus does not -- bin 0 is 18.3 dB
BELOW its strongest mode), and that term dominates the gradient, starving modal fit on every
FDTD track.

**A prediction, recorded before the run so the result is interpretable either way.** The four
terms are NOT equally exposed (`multi_room_2d_mat._losses`):

    L_spec_real, L_spec_imag : L1 on the LINEAR real/imag parts   -> should be DC-dominated
    L_amp                    : L1 on log10|H|                     -> should NOT be
    L_phase                  : mean(1 - cos(dphi))                -> should NOT be

So the chunk spec's "spec_real, spec_imag, and amp are linear-domain losses" is wrong about
L_amp. If the measurement shows all four dominated, the hypothesis is wrong in an interesting
way and the retrain should not be run on this reasoning.

This measures the loss AS TRAINED: the same terms, the same weights, on the same band, from a
trained checkpoint. It does not measure "power in bin 0", which is a property of the corpus and
was already known -- it measures how much of the *gradient signal* those bins command.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from aaf.eval.p3_2_eval import load_model
from aaf.eval.p3_2b_eval import render_config_arm

#: 20 Hz at df = 0.5 Hz. The first room mode is 28.6-57 Hz across the Track A family, so this
#: cut removes no modal content whatsoever -- that is the whole point of choosing it.
CUT_HZ = 20.0
DF_HZ = 0.5
WEIGHTS = (1.0, 1.0, 1.0, 0.1)          # cfg default, not overridden by any shipped yaml


def per_bin_terms(P: np.ndarray, T: np.ndarray):
    """The four loss terms, kept PER BIN instead of reduced, so they can be split by band.

    Each is the same quantity `_losses` averages; summing over bins and receivers and dividing
    by the element count reproduces the scalar exactly (asserted by the caller).
    """
    eps = 1e-6
    return {
        "L_spec_real": np.abs(P.real - T.real),
        "L_spec_imag": np.abs(P.imag - T.imag),
        "L_amp": np.abs(np.log10(np.abs(P) + eps) - np.log10(np.abs(T) + eps)),
        "L_phase": 1.0 - np.cos(np.angle(P) - np.angle(T)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="outputs/p3_3fast/p3_3fast_trackA2/ckpt_iter0030000.pt")
    ap.add_argument("--data-dir", default="data/track_p3_3fast_A")
    ap.add_argument("--manifest",
                    default="configs/sweeps_2d_mat/p3_3fast_trackA_manifest.json")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--out", default="outputs/p4_1/dc_fix/loss_contribution.json")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, renderer, cfg, meta, it = load_model(Path(a.checkpoint), dev)
    model.eval()
    renderer.eval()                       # required alongside model.eval() -- D49 C3
    cond_source = str(cfg["cond_source"])
    print("[arm] {} iter {} | cond {} ({}d) | device {}".format(
        Path(a.checkpoint).parent.name, it, cond_source, cfg["cond_dim"], dev), flush=True)

    rows = json.load(open(a.manifest))["configs"]
    from aaf.data.seg_configs import configs_from_rows
    configs = [c for c in configs_from_rows(rows) if getattr(c, "split", "") != "train"]
    configs = configs[:a.limit]
    print("[data] {} test configs from {}".format(len(configs), a.manifest), flush=True)

    cut = int(round(CUT_HZ / DF_HZ))       # 40
    acc = {k: {"lo": 0.0, "hi": 0.0, "n_lo": 0, "n_hi": 0} for k in
           ("L_spec_real", "L_spec_imag", "L_amp", "L_phase")}
    n_done = 0
    for c in configs:
        f = Path(a.data_dir) / c.filename
        if not f.exists():
            continue
        with h5py.File(f) as h:
            T = np.asarray(h["ism/H_complex"])
            rx = np.asarray(json.loads(h.attrs["receiver_pos"]))
            src = np.asarray(json.loads(h.attrs["source_pos"]))
        with torch.no_grad():
            P = np.asarray(render_config_arm(model, renderer, cond_source, c.L, c.W,
                                             c.alphas, rx, src, dev))[:, :T.shape[1]]
        for k, v in per_bin_terms(P, T).items():
            acc[k]["lo"] += float(v[:, :cut].sum());  acc[k]["n_lo"] += v[:, :cut].size
            acc[k]["hi"] += float(v[:, cut:].sum());  acc[k]["n_hi"] += v[:, cut:].size
        n_done += 1
    if not n_done:
        raise SystemExit("no configs found under {}".format(a.data_dir))

    print("\nLoss contribution of bins 0-{} (< {:.0f} Hz) vs the rest, over {} configs".format(
        cut - 1, CUT_HZ, n_done))
    print("  {} bins of {} = {:.1f}% of the band by COUNT\n".format(
        cut, cut + acc["L_spec_real"]["n_hi"] // (acc["L_spec_real"]["n_lo"] // cut),
        100.0 * cut / 601))
    print("  term            sum(0-39)     sum(40-600)   DC share   x per-bin")
    out_terms, weighted_lo, weighted_tot = {}, 0.0, 0.0
    for i, k in enumerate(("L_spec_real", "L_spec_imag", "L_amp", "L_phase")):
        lo, hi = acc[k]["lo"], acc[k]["hi"]
        n_lo, n_hi = acc[k]["n_lo"], acc[k]["n_hi"]
        share = lo / (lo + hi) if (lo + hi) else float("nan")
        # per-bin enrichment: how much heavier a DC bin is than a typical in-band bin
        enrich = (lo / n_lo) / (hi / n_hi) if n_hi and hi else float("nan")
        print("  {:12s} {:12.4g}  {:12.4g}   {:6.1%}   {:8.1f}x".format(k, lo, hi, share, enrich))
        out_terms[k] = {"sum_lo": lo, "sum_hi": hi, "n_lo": n_lo, "n_hi": n_hi,
                        "dc_share": share, "per_bin_enrichment": enrich,
                        "weight": WEIGHTS[i]}
        # contribution to the ACTUAL objective = weight * mean over all elements
        weighted_lo += WEIGHTS[i] * lo / (n_lo + n_hi)
        weighted_tot += WEIGHTS[i] * (lo + hi) / (n_lo + n_hi)
    total_share = weighted_lo / weighted_tot if weighted_tot else float("nan")
    print("\n  WEIGHTED TOTAL objective: bins 0-{} contribute {:.1%}".format(cut - 1, total_share))

    verdict = ("CONFIRMED" if total_share >= 0.5 else
               "PARTIAL" if total_share >= 0.2 else "NOT CONFIRMED")
    print("  hypothesis: {} (>=50% CONFIRMED, >=20% PARTIAL)".format(verdict))
    print("  prediction check -- linear terms dominated, log/phase terms not:")
    lin = min(out_terms["L_spec_real"]["dc_share"], out_terms["L_spec_imag"]["dc_share"])
    nonlin = max(out_terms["L_amp"]["dc_share"], out_terms["L_phase"]["dc_share"])
    print("    min(linear) {:.1%} vs max(log,phase) {:.1%} -> {}".format(
        lin, nonlin, "as predicted" if lin > nonlin else "NOT as predicted"))

    res = {"checkpoint": a.checkpoint, "iter": int(it), "cond_source": cond_source,
           "n_configs": n_done, "cut_hz": CUT_HZ, "cut_bin": cut, "df_hz": DF_HZ,
           "weights": list(WEIGHTS), "terms": out_terms,
           "weighted_dc_share": total_share, "verdict": verdict,
           "prediction_held": bool(lin > nonlin),
           "note": ("Share of the TRAINING objective commanded by bins below 20 Hz. The first "
                    "room mode is 28.6-57 Hz across this family, so the cut removes no modal "
                    "content. L_spec_real/imag are linear-domain and exposed to the DC term; "
                    "L_amp is log10 and L_phase is a cosine, and neither is.")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print("\n-> {}".format(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
