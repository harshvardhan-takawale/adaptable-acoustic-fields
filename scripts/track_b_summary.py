"""Track B summary: aggregate inner-loop variant results and pick the winner.

Walks ``outputs/inner_loop_experiments/<variant>/<run>/L<L>/{metrics.json,
band_limited_metrics.json}`` for each completed (variant, L). Writes:

  outputs/inner_loop_experiments/SUMMARY.md      — comparison table
  outputs/inner_loop_experiments/best_variant.txt — winner ID + JSON kwargs
  outputs/multi_room/sweep/figures/inner_loop_comparison.png

Winner = lowest mean held-out 0-250 Hz LSD across the 6 unseen L; tiebreaker
is count ≤ 2 dB in the modal band. (If 0-250 Hz isn't recorded — i.e., a
variant pre-dating Chunk 3.6's band-limited integration — fall back to
full-band held LSD.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aaf.eval.zero_shot_variants import ALL_VARIANTS, VARIANT_DESCRIPTIONS, variant_kwargs


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LS = (3.25, 3.75, 4.25, 4.75, 5.25, 5.75)


def _gather(out_root: Path, run: str, Ls) -> dict[str, dict[float, dict]]:
    out: dict[str, dict[float, dict]] = {}
    for variant in ALL_VARIANTS:
        per_L: dict[float, dict] = {}
        for L in Ls:
            d = out_root / variant / run / f"L{L}"
            mp = d / "metrics.json"
            if mp.exists():
                m = json.loads(mp.read_text())
                per_L[float(L)] = m
        if per_L:
            out[variant] = per_L
    return out


def _summarize_variant(per_L: dict[float, dict]) -> dict:
    Ls = sorted(per_L.keys())
    full_held = np.asarray([per_L[L].get("held_out_lsd_db", np.nan) for L in Ls])
    obs_lsd = np.asarray([per_L[L].get("obs_lsd_db", np.nan) for L in Ls])
    band_held_modal = []
    for L in Ls:
        bh = per_L[L].get("band_metrics_held", {})
        v = bh.get("lsd_band_0_250_db")
        band_held_modal.append(float(v) if v is not None else np.nan)
    band_held_modal = np.asarray(band_held_modal)

    def _safe(arr, fn):
        a = arr[np.isfinite(arr)]
        return float(fn(a)) if a.size else float("nan")

    return {
        "n_L": len(Ls),
        "Ls": Ls,
        "full_held_mean": _safe(full_held, np.mean),
        "full_held_count_le_2": int(np.sum(np.isfinite(full_held) & (full_held <= 2.0))),
        "obs_mean": _safe(obs_lsd, np.mean),
        "modal_held_mean": _safe(band_held_modal, np.mean),
        "modal_held_count_le_2": int(np.sum(np.isfinite(band_held_modal) & (band_held_modal <= 2.0))),
        "modal_held_count_le_3": int(np.sum(np.isfinite(band_held_modal) & (band_held_modal <= 3.0))),
        "_full_held": full_held.tolist(),
        "_modal_held": band_held_modal.tolist(),
    }


def _pick_winner(summaries: dict[str, dict]) -> tuple[str, str]:
    """Return (variant_id, reason)."""
    have_modal = {v: s for v, s in summaries.items() if np.isfinite(s["modal_held_mean"])}
    pool = have_modal if have_modal else summaries
    sort_key_field = "modal_held_mean" if have_modal else "full_held_mean"
    items = sorted(
        pool.items(),
        key=lambda kv: (
            kv[1][sort_key_field] if np.isfinite(kv[1][sort_key_field]) else float("inf"),
            -kv[1].get("modal_held_count_le_2", 0),
        ),
    )
    winner = items[0][0]
    reason = (
        f"lowest mean {sort_key_field}={items[0][1][sort_key_field]:.3f} "
        f"(modal count ≤ 2 dB: {items[0][1].get('modal_held_count_le_2', 0)})"
    )
    return winner, reason


def _make_figure(summaries: dict[str, dict], out_path: Path):
    if not summaries:
        return
    variants = list(summaries.keys())
    means_modal = [summaries[v]["modal_held_mean"] for v in variants]
    means_full = [summaries[v]["full_held_mean"] for v in variants]
    x = np.arange(len(variants))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, means_modal, width=0.4, label="0-250 Hz (modal)", color="steelblue")
    ax.bar(x + 0.2, means_full, width=0.4, label="0-2000 Hz (full)", color="indianred")
    ax.axhline(2.0, color="green", lw=1.0, ls="--", alpha=0.6, label="2 dB target")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=10)
    ax.set_ylabel("mean held-out LSD (dB) across 6 unseen L")
    ax.set_title("Track B: inner-loop variant comparison on R6")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str,
                    default=str(REPO_ROOT / "outputs/inner_loop_experiments"))
    ap.add_argument("--sweep_root", type=str,
                    default=str(REPO_ROOT / "outputs/multi_room/sweep"))
    ap.add_argument("--run", type=str, default="R6_tiny_lhead",
                    help="model run-id used for the variant sweep")
    ap.add_argument("--Ls", nargs="+", type=float, default=list(DEFAULT_LS))
    args = ap.parse_args()

    out_root = Path(args.out_root)
    data = _gather(out_root, args.run, args.Ls)
    if not data:
        raise SystemExit(f"no variant outputs found under {out_root}")

    summaries = {v: _summarize_variant(per_L) for v, per_L in data.items()}
    winner, reason = _pick_winner(summaries)
    print(f"# winner: {winner} — {reason}")

    # Write SUMMARY.md
    rows = [
        "| Variant | Description | n L | mean obs LSD | mean full held LSD | "
        "count full ≤ 2 dB | mean 0-250 Hz held LSD | count modal ≤ 2 dB | "
        "count modal ≤ 3 dB | winner |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for v in ALL_VARIANTS:
        if v not in summaries:
            continue
        s = summaries[v]
        marker = "✅" if v == winner else ""
        rows.append(
            f"| {v} | {VARIANT_DESCRIPTIONS[v]} | {s['n_L']} | "
            f"{s['obs_mean']:.2f} | {s['full_held_mean']:.2f} | "
            f"{s['full_held_count_le_2']}/{s['n_L']} | "
            f"{s['modal_held_mean']:.2f} | "
            f"{s['modal_held_count_le_2']}/{s['n_L']} | "
            f"{s['modal_held_count_le_3']}/{s['n_L']} | {marker} |"
        )
    summary_md = (
        f"# Track B summary — inner-loop variants on {args.run}\n\n"
        f"Winner: **{winner}** ({VARIANT_DESCRIPTIONS[winner]}). "
        f"Reason: {reason}\n\n"
        + "\n".join(rows) + "\n\n"
        "## Headline figure\n\n"
        f"![inner-loop comparison]({Path('..').joinpath('multi_room/sweep/figures/inner_loop_comparison.png')})\n\n"
        "## Notes\n\n"
        "- All variants run on the same trained R6 checkpoint; only the inner-loop\n"
        "  procedure differs.\n"
        "- 'modal' = LSD restricted to 0-250 Hz (the band where Track A finds\n"
        "  visually-correct tracking).\n"
        "- The winner's kwargs are written to `best_variant.txt` and consumed by\n"
        "  `scripts/zero_shot_with_best_variant.py` for evaluating Track-C trained\n"
        "  models.\n"
    )
    out_md = out_root / "SUMMARY.md"
    out_md.write_text(summary_md)
    print(f"# wrote {out_md}")

    # Winner kwargs to a file consumed by zero_shot_with_best_variant.py
    best_path = out_root / "best_variant.txt"
    best_payload = {
        "variant": winner,
        "kwargs": variant_kwargs(winner),
        "reason": reason,
    }
    best_path.write_text(json.dumps(best_payload, indent=2))
    print(f"# wrote {best_path}")

    # Headline figure goes under the sweep figures dir.
    fig_path = Path(args.sweep_root) / "figures" / "inner_loop_comparison.png"
    _make_figure(summaries, fig_path)
    print(f"# wrote {fig_path}")


if __name__ == "__main__":
    main()
