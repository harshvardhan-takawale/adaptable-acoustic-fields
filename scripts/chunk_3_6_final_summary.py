"""Compose tasks/CHUNK_3_6_RESULTS.md and refresh the sweep-level SUMMARY.

Pulls together results from:
  - Track A: outputs/multi_room/sweep/band_limited_summary.md
  - Track B: outputs/inner_loop_experiments/SUMMARY.md + best_variant.txt
  - Track C: outputs/multi_room/sweep/{C1_film,C2_latent_jitter}/...

Re-runs scripts/sweep_summary.py to refresh SWEEP_SUMMARY.md with C1/C2 rows.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LS = (3.25, 3.75, 4.25, 4.75, 5.25, 5.75)
TRACK_C_RUNS = ("C1_film", "C2_latent_jitter")


def _load_zs_metrics(run_dir: Path, Ls, subdir: str = "zero_shot") -> list[dict]:
    rows = []
    for L in Ls:
        p = run_dir / subdir / f"L{L}" / "metrics.json"
        if p.exists():
            rows.append({"L": float(L), **json.loads(p.read_text())})
    return rows


def _summary_block(label: str, rows: list[dict]) -> str:
    if not rows:
        return f"**{label}**: no metrics found.\n"
    full = np.asarray([r.get("held_out_lsd_db", np.nan) for r in rows])
    obs = np.asarray([r.get("obs_lsd_db", np.nan) for r in rows])
    modal = np.asarray([
        r.get("band_metrics_held", {}).get("lsd_band_0_250_db", np.nan) for r in rows
    ])

    def _fmt(arr):
        a = arr[np.isfinite(arr)]
        return f"{a.mean():.2f}" if a.size else "—"

    def _cnt(arr, thr):
        a = arr[np.isfinite(arr)]
        return f"{int((a <= thr).sum())}/{len(a)}" if a.size else "—"

    return (
        f"**{label}** ({len(rows)} unseen L)\n\n"
        f"- mean obs LSD: {_fmt(obs)} dB\n"
        f"- mean full-band held LSD: {_fmt(full)} dB  (count ≤ 2 dB: {_cnt(full, 2.0)})\n"
        f"- mean 0-250 Hz held LSD: {_fmt(modal)} dB  (count ≤ 2 dB: {_cnt(modal, 2.0)}, "
        f"count ≤ 3 dB: {_cnt(modal, 3.0)})\n"
    )


def _track_c_block(sweep_root: Path, run_id: str, Ls) -> str:
    run_dir = sweep_root / run_id
    if not run_dir.exists():
        return f"### {run_id}\n\nNot found on disk.\n"
    tm_path = run_dir / "train_meta.json"
    train_status = "no train_meta"
    train_lsd = "—"
    if tm_path.exists():
        meta = json.loads(tm_path.read_text())
        train_status = (
            "completed" if int(meta.get("n_iters_actual", 0)) == int(meta.get("n_iters_target", -1))
            else f"early-stopped@{meta.get('stop_iter')}" if meta.get("stopped_early")
            else "incomplete"
        )
    sc_path = run_dir / "scalars.json"
    if sc_path.exists():
        sc = json.loads(sc_path.read_text())
        vals = [r for r in sc if r.get("phase") == "val"]
        if vals:
            train_lsd = f"{vals[-1].get('lsd_db', float('nan')):.2f}"

    # B1 baseline for C1/C2 lives under outputs/inner_loop_experiments/B1/<run>/L*/
    # (the orchestrator submits B1 via zero_shot_variant.sh, not the legacy
    # outputs/multi_room/sweep/<run>/zero_shot/ path). Fall back to legacy path
    # if the variant-style path is empty (preserves backward compat with the
    # auto-generated R-run hierarchy).
    b1_var_dir = sweep_root.parent / "inner_loop_experiments" / "B1" / run_id
    b1_rows = _load_zs_metrics(b1_var_dir, Ls, subdir=".") if b1_var_dir.exists() \
              else _load_zs_metrics(run_dir, Ls, subdir="zero_shot")
    # The Track-B-winner output dir is zero_shot_<variant>/
    best_dirs = sorted(
        [d for d in run_dir.glob("zero_shot_*") if d.is_dir()],
        key=lambda p: p.name,
    )
    best_block = ""
    for bd in best_dirs:
        variant = bd.name.replace("zero_shot_", "")
        rows = _load_zs_metrics(run_dir, Ls, subdir=bd.name)
        best_block += "\n" + _summary_block(
            f"{run_id} — Track B winner ({variant})", rows
        )
    return (
        f"### {run_id}\n\n"
        f"Training status: {train_status}; final val LSD: {train_lsd} dB.\n\n"
        + _summary_block(f"{run_id} — B1 baseline (8 obs, 2K iters, random init)", b1_rows)
        + best_block
    )


def _read_text_or(path: Path, default: str) -> str:
    return path.read_text() if path.exists() else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", type=str,
                    default=str(REPO_ROOT / "outputs/multi_room/sweep"))
    ap.add_argument("--inner_loop_root", type=str,
                    default=str(REPO_ROOT / "outputs/inner_loop_experiments"))
    ap.add_argument("--Ls", nargs="+", type=float, default=list(DEFAULT_LS))
    ap.add_argument("--out_md", type=str,
                    default=str(REPO_ROOT / "tasks/CHUNK_3_6_RESULTS.md"))
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    inner_root = Path(args.inner_loop_root)

    # Refresh SWEEP_SUMMARY.md with all 11 runs (R0-R8, C1, C2).
    print("# refreshing sweep summary…")
    subprocess.run(
        ["python", "-m", "scripts.sweep_summary"],
        cwd=str(REPO_ROOT), check=False,
    )

    # Pull Track A/B summary text (verbatim if present).
    track_a_md = _read_text_or(
        sweep_root / "band_limited_summary.md",
        "_(Track A summary not found.)_\n",
    )
    track_b_md = _read_text_or(
        inner_root / "SUMMARY.md",
        "_(Track B summary not found.)_\n",
    )
    best_payload = {}
    bp = inner_root / "best_variant.txt"
    if bp.exists():
        best_payload = json.loads(bp.read_text())

    # Track C blocks.
    track_c_blocks = "\n".join(
        _track_c_block(sweep_root, run, args.Ls) for run in TRACK_C_RUNS
    )

    # Compose the writeup.
    writeup = []
    writeup.append("# Chunk 3.6 — Band-limited evaluation + inner-loop fixes + smoothness retrains\n")
    writeup.append(
        "Three parallel tracks layered on the Chunk-3.5 sweep (R0-R8). Goal: "
        "find any configuration with band-limited (0-250 Hz) zero-shot LSD ≤ 2 dB "
        "on ≥ 4/6 unseen L.\n"
    )
    writeup.append("## Headline result\n")
    if best_payload:
        writeup.append(
            f"Track B winner: **{best_payload.get('variant', '?')}** — "
            f"{best_payload.get('reason', 'see SUMMARY.md')}.\n"
        )
    else:
        writeup.append("(Track B summary not yet available.)\n")
    writeup.append("See per-track sections below for full tables.\n")

    writeup.append("## Track A — band-limited evaluation on R0/R6/R7/R8\n")
    writeup.append(track_a_md.strip() + "\n")

    writeup.append("## Track B — inner-loop adaptation variants on R6\n")
    writeup.append(track_b_md.strip() + "\n")

    writeup.append("## Track C — FiLM + latent-jitter retrained variants\n")
    writeup.append(track_c_blocks + "\n")

    writeup.append("## Updated sweep summary\n")
    writeup.append(
        "Refreshed `outputs/multi_room/sweep/SWEEP_SUMMARY.md` includes C1 and C2 alongside R0-R8.\n"
    )
    writeup.append("## Recommendations / next iteration\n")
    writeup.append(
        "TODO (manager-facing): prioritise the next-chunk experiment based on which track moves "
        "the modal-LSD needle. Replace this placeholder once the Track C numbers are in.\n"
    )

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(writeup))
    print(f"# wrote {out_md}")


if __name__ == "__main__":
    main()
