"""Track A summary: aggregate band-limited LSDs into a markdown table + figure.

Reads ``outputs/multi_room/sweep/<run>/zero_shot/L<L>/band_limited_metrics.json``
for each known run/L and writes:

  outputs/multi_room/sweep/band_limited_summary.md
  outputs/multi_room/sweep/figures/band_limited_lsd_per_L.png   (4-panel grouped bar)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS = ("R0_central", "R6_tiny_lhead", "R7_medium_hash", "R8_tiny_latent")
DEFAULT_LS = (3.25, 3.75, 4.25, 4.75, 5.25, 5.75)
BANDS = ((0, 250), (250, 500), (500, 2000), (0, 2000))
BAND_LABELS = ("0-250 Hz (modal)", "250-500 Hz (transition)",
               "500-2000 Hz (diffuse)", "0-2000 Hz (full band)")


def _gather(sweep_root: Path, runs, Ls) -> dict[str, dict[float, dict]]:
    out: dict[str, dict[float, dict]] = {}
    for run_id in runs:
        per_L: dict[float, dict] = {}
        for L in Ls:
            path = sweep_root / run_id / "zero_shot" / f"L{L}" / "band_limited_metrics.json"
            if path.exists():
                per_L[float(L)] = json.loads(path.read_text())
        if per_L:
            out[run_id] = per_L
    return out


def _make_table(data: dict[str, dict[float, dict]]) -> str:
    rows: list[str] = []
    rows.append("| Run | Band | mean LSD (dB) | min | max | count ≤ 2 dB | count ≤ 3 dB | n L |")
    rows.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for run_id, per_L in data.items():
        for (lo, hi), label in zip(BANDS, BAND_LABELS):
            key = f"lsd_band_{lo}_{hi}_db"
            vals = []
            for L in sorted(per_L.keys()):
                v = per_L[L].get("held", {}).get(key)
                if v is not None:
                    vals.append(float(v))
            if not vals:
                continue
            arr = np.asarray(vals)
            rows.append(
                f"| {run_id} | {label} | {arr.mean():.2f} | {arr.min():.2f} | "
                f"{arr.max():.2f} | {(arr <= 2.0).sum()}/{len(arr)} | "
                f"{(arr <= 3.0).sum()}/{len(arr)} | {len(arr)} |"
            )
    return "\n".join(rows)


def _make_figure(data: dict[str, dict[float, dict]], out_path: Path):
    runs = list(data.keys())
    if not runs:
        return
    Ls = sorted({L for per_L in data.values() for L in per_L})
    n_runs = len(runs)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axs = axs.flatten()
    bar_w = 0.8 / max(n_runs, 1)
    x_base = np.arange(len(Ls))
    palette = plt.cm.tab10(np.linspace(0, 1, max(n_runs, 1)))
    for i, ((lo, hi), label) in enumerate(zip(BANDS, BAND_LABELS)):
        ax = axs[i]
        key = f"lsd_band_{lo}_{hi}_db"
        for j, run_id in enumerate(runs):
            ys = []
            for L in Ls:
                v = data[run_id].get(L, {}).get("held", {}).get(key)
                ys.append(float(v) if v is not None else np.nan)
            ax.bar(x_base + j * bar_w - 0.4 + bar_w / 2, ys, width=bar_w,
                   label=run_id, color=palette[j])
        ax.axhline(2.0, color="green", lw=1.0, ls="--", alpha=0.6, label="2 dB target" if i == 0 else None)
        ax.set_title(label)
        ax.set_xticks(x_base)
        ax.set_xticklabels([f"L={L:.2f}" for L in Ls], fontsize=8, rotation=30)
        ax.set_ylabel("held-out LSD (dB)")
        ax.grid(True, alpha=0.3, axis="y")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=min(6, n_runs + 1), fontsize=9, frameon=False)
    fig.suptitle("Band-limited zero-shot held-out LSD (Chunk 3.6 Track A)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", type=str,
                    default=str(REPO_ROOT / "outputs/multi_room/sweep"))
    ap.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    ap.add_argument("--Ls", nargs="+", type=float, default=list(DEFAULT_LS))
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    data = _gather(sweep_root, args.runs, args.Ls)
    if not data:
        raise SystemExit(
            "no band_limited_metrics.json files found under "
            f"{sweep_root} — run scripts/band_limited_recompute.py first"
        )

    table = _make_table(data)
    summary = (
        "# Band-limited zero-shot LSD summary (Chunk 3.6 Track A)\n\n"
        "Recomputed from saved z_star.pt for each (run, L) — no inner-loop re-adaptation.\n"
        "LSD is mean ``|20*log10(|H_pred|/|H_target|)|`` over the 56 held-out receivers and "
        "the bins inside each band.\n\n"
        "Bands: modal (0-250 Hz), transition (250-500 Hz), diffuse (500-2000 Hz), full (0-2000 Hz).\n"
        "Target for the meeting deliverable: ≤ 2 dB on ≥ 4/6 unseen L (modal regime).\n\n"
        + table + "\n\n"
        "## Headline figure\n\n"
        "![band-limited LSD per L](figures/band_limited_lsd_per_L.png)\n\n"
        "## Notes\n\n"
        "- These numbers reuse the EXACT z_star produced by the original Chunk-3.5 zero-shot runs.\n"
        "- The full-band column matches the existing `held_out_lsd_db` field in `metrics.json` "
        "modulo numerical noise — sanity check.\n"
        "- Track B variants (different inner-loop strategies) are aggregated in "
        "`outputs/inner_loop_experiments/SUMMARY.md`.\n"
    )
    out_md = sweep_root / "band_limited_summary.md"
    out_md.write_text(summary)
    fig_path = sweep_root / "figures" / "band_limited_lsd_per_L.png"
    _make_figure(data, fig_path)
    print(f"# wrote {out_md}")
    print(f"# wrote {fig_path}")


if __name__ == "__main__":
    main()
