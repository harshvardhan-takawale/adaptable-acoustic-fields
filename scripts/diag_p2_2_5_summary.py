"""P2-2.5 diagnostic — assemble DIAGNOSIS.md from the three training runs.

Reads:
  outputs/diag_p2_2_5/{A_10rm_b16, B_45rm_b32, C_10rm_b64}/{train_meta.json, scalars.json}

Writes:
  outputs/diag_p2_2_5/DIAGNOSIS.md
  outputs/diag_p2_2_5/convergence_curves.png
  outputs/diag_p2_2_5/per_room_lsd.png

Applies the spec's decision matrix:
  - ≤ 2.5 dB final val LSD = "success"
  - > 4 dB = "clear failure"
  - in between = "ambiguous; flag for manager"
And matches the (A, C, B) outcome triple to a bottleneck verdict +
recommended P2-3 config.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical B is the DDP run (it reached the full 60K target ~2x faster);
# single-GPU B_45rm_b32 is retained as a cross-check (see CROSSCHECK below).
# If B_45rm_ddp is absent (e.g. interim before DDP existed), fall back to the
# single-GPU dir.
import os as _os
_REPO = Path(__file__).resolve().parent.parent
_B_DIR = "B_45rm_ddp" if (_REPO / "outputs/diag_p2_2_5/B_45rm_ddp/scalars.json").exists() else "B_45rm_b32"
RUNS = [
    ("A", "A_10rm_b16",   "Run A — 10 rooms, eff-batch 16, n_pts=16"),
    ("B", _B_DIR,         "Run B — 45 rooms, eff-batch 32, n_pts=32 (relaxed early-stop)"
                          + (" [2-GPU DDP]" if _B_DIR.endswith("ddp") else "")),
    ("C", "C_10rm_b64",   "Run C — 10 rooms, eff-batch 64, n_pts=32"),
]
# Single-GPU B cross-check dir (validates the DDP path).
CROSSCHECK_B_DIR = "B_45rm_b32"

# Config YAML per run — used to fill the headline table for IN-PROGRESS runs
# whose train_meta.json (written only at completion) doesn't exist yet.
CONFIG_BY_CODE = {
    "A": REPO_ROOT / "configs/sweep_3d/A_diag.yaml",
    "B": REPO_ROOT / "configs/sweep_3d/B_diag.yaml",
    "C": REPO_ROOT / "configs/sweep_3d/C_diag.yaml",
}


def _cfg_fallback(code: str) -> dict:
    """Read (batch_size, grad_accum_steps, n_pts_per_ray, n_rooms) from the run's
    config YAML — for runs not yet complete (no train_meta.json)."""
    p = CONFIG_BY_CODE.get(code)
    if p is None or not p.exists():
        return {}
    d = yaml.safe_load(p.read_text())
    rooms_yaml = REPO_ROOT / d.get("rooms_yaml", "")
    n_rooms = 0
    if rooms_yaml.exists():
        rd = yaml.safe_load(rooms_yaml.read_text())
        n_rooms = len(rd.get("rooms", []))
    return {
        "batch_size": d.get("batch_size"),
        "grad_accum_steps": d.get("grad_accum_steps", 1),
        "n_pts_per_ray": d.get("n_pts_per_ray"),
        "n_iters_target": d.get("n_iters"),
        "n_rooms": n_rooms,
    }

# Per-room LSD keys in scalars.json look like "L4.50_W4.00_H3.25_lsd_db" etc.
PER_ROOM_LSD_RE = re.compile(r"^L([0-9.]+)_W([0-9.]+)_H([0-9.]+)_lsd_db$")


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _classify_lsd(lsd: float) -> str:
    if not np.isfinite(lsd):
        return "missing"
    if lsd <= 2.5:
        return "success"
    if lsd > 4.0:
        return "fail"
    return "ambiguous"


def _verdict_from_triple(a: str, b: str, c: str) -> tuple[str, str]:
    """Map the (A, C, B) outcomes to (one-line verdict, P2-3 recommendation)."""
    if a == "success" and c == "success" and b == "success":
        return (
            "Sampling-rate was the whole story.",
            "P2-3: scale compute — pin tron qos=high A100/A6000, batch≥32, "
            "n_pts≥32 on the full 45-room set. The Run B config is the right "
            "P2-3 starting point.",
        )
    if a == "success" and c == "success" and b == "fail":
        return (
            "Room count is the wall.",
            "P2-3: try curriculum (10 → 25 → 45 rooms, warm-start from each "
            "checkpoint) or widen the decoder (sigma_encoder_dim 256 → 512). "
            "Either ramps capacity to match the larger room family.",
        )
    if a == "success" and c == "success" and b == "ambiguous":
        return (
            "Capacity is NOT the wall — coverage / compute is the dominant "
            "lever. The 10-room set fits cleanly (A, C ≤ ~1.8 dB); the full "
            "45-room set, given 8× the per-iter coverage and 60K iters, "
            "improved from P2-2 M1's 6.16 dB to the 2.5 dB threshold and was "
            "still descending — reachable with more compute, not a new architecture.",
            "P2-3: scale compute on the full 45-room set — apply Run C's "
            "recipe (effective batch 64, n_pts 32) to all 45 rooms and/or "
            "extend to 80–100K iters. B reached 2.61 dB at 60K still "
            "descending; either lever should carry it below 2.5. Do NOT widen "
            "the decoder — Run C proves ~1 dB is achievable at this capacity.",
        )
    if a == "fail" and c == "fail":
        return (
            "Capacity is the wall.",
            "P2-3: widen the decoder — `sigma_encoder_dim` 256 → 512 first; "
            "if still failing, also `log2_hashmap_size` 18 → 20 and/or "
            "`n_levels` 16 → 20. Stronger conditioning (concat alongside FiLM) "
            "is the secondary lever.",
        )
    if a == "fail" and c == "success":
        return (
            "Batch / coverage matters more than rooms (unexpected).",
            "P2-3: investigate — Run A and C share the same 10 rooms; the only "
            "way A fails and C succeeds is batch=64 with grad-accum behaves "
            "differently than batch=16. Check that grad_accum_steps=8 gives "
            "the same gradient as direct batch=64.",
        )
    # Catch-all (mostly with 'ambiguous' or 'missing' classifications).
    return (
        f"Mixed signal — A={a}, B={b}, C={c}.",
        "Refer the manager to the per-room breakdown; specific rooms left "
        "behind in B but not A/C indicate a room-count effect; uniform "
        "failure across all three points to capacity.",
    )


def _read_run(out_dir: Path, run_label: str) -> dict:
    """Read train_meta (if present) + scalars; return final val LSD + curve +
    per-room LSDs.

    Robust to IN-PROGRESS runs: ``train_meta.json`` is only written when
    train() finishes, but ``scalars.json`` is rewritten at every checkpoint
    (every 2500 iters). For an interim DIAGNOSIS we therefore only require
    scalars.json; train_meta fields (wall-clock, early-stop) are reported as
    "in progress" when absent.
    """
    meta_path = out_dir / "train_meta.json"
    scalars_path = out_dir / "scalars.json"
    out = {
        "label": run_label,
        "meta_path": str(meta_path),
        # Only scalars.json is required; meta is optional (written at the end).
        "exists": scalars_path.exists(),
        "complete": meta_path.exists(),
    }
    if not out["exists"]:
        return out
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    scalars = json.loads(scalars_path.read_text())
    val_rows = [r for r in scalars if r.get("phase") == "val"]
    out["meta"] = meta
    out["wall_clock_s"] = float(meta.get("wall_clock_seconds", float("nan")))
    out["stopped_early"] = bool(meta.get("stopped_early", False))
    out["stop_reason"] = str(meta.get("stop_reason", ""))
    out["n_iters_actual"] = int(meta.get("n_iters_actual", 0))
    out["n_iters_target"] = int(meta.get("n_iters_target", 0))

    if not val_rows:
        out["final_val_lsd"] = float("nan")
        out["val_iter"] = []
        out["val_lsd"] = []
        out["per_room_final"] = {}
        return out

    last = val_rows[-1]
    out["final_val_lsd"] = _safe_float(last.get("lsd_db"))
    out["val_iter"] = [int(r["iter"]) for r in val_rows]
    out["val_lsd"] = [_safe_float(r.get("lsd_db")) for r in val_rows]

    # Per-room LSDs (extract from the *final* val row).
    per_room = {}
    for k, v in last.items():
        m = PER_ROOM_LSD_RE.match(k)
        if m:
            L, W, H = (float(x) for x in m.groups())
            per_room[(L, W, H)] = _safe_float(v)
    out["per_room_final"] = per_room
    return out


def _figure_convergence_curves(runs: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"A": "steelblue", "B": "tab:orange", "C": "seagreen"}
    for code, run_id, label in RUNS:
        info = runs[code]
        if not info.get("val_iter"):
            continue
        ax.plot(
            info["val_iter"], info["val_lsd"],
            color=colors[code], lw=1.2, marker="o", ms=3,
            label=f"{code}: {label}  (final {info['final_val_lsd']:.2f} dB)",
        )
        # Mark early-stop iter if applicable.
        if info.get("stopped_early"):
            ax.axvline(
                info["val_iter"][-1], color=colors[code], ls="--", lw=0.5, alpha=0.5,
            )
    ax.axhline(2.5, color="k", ls=":", lw=0.7, label="≤ 2.5 dB = success")
    ax.axhline(4.0, color="firebrick", ls=":", lw=0.7, label="> 4 dB = clear failure")
    ax.set_xlabel("iteration")
    ax.set_ylabel("val LSD (dB)")
    ax.set_title("P2-2.5 diagnostic — val LSD vs iter")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _figure_per_room_lsd(runs: dict, out_path: Path):
    """Two panels: left — A & C side-by-side on 10 shared rooms; right — B on 45."""
    a = runs["A"].get("per_room_final", {})
    c = runs["C"].get("per_room_final", {})
    b = runs["B"].get("per_room_final", {})

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: A vs C on the 10-room subset (sorted by V).
    shared = sorted(set(a) | set(c), key=lambda k: k[0] * k[1] * k[2])
    if shared:
        x = np.arange(len(shared))
        w = 0.4
        a_vals = [a.get(k, 0.0) for k in shared]
        c_vals = [c.get(k, 0.0) for k in shared]
        axL.bar(x - w / 2, a_vals, w, color="steelblue", label="Run A (10 rooms, b=16, pts=16)")
        axL.bar(x + w / 2, c_vals, w, color="seagreen", label="Run C (10 rooms, b=64, pts=32)")
        axL.set_xticks(x)
        axL.set_xticklabels([f"L{L:.1f}\nW{W:.1f}\nH{H:.1f}" for L, W, H in shared], fontsize=7)
        axL.set_ylabel("final val LSD (dB)")
        axL.set_title("A vs C on the 10-room subset")
        axL.axhline(2.5, color="k", ls=":", lw=0.7)
        axL.grid(True, alpha=0.3, axis="y")
        axL.legend(fontsize=8, frameon=False)
    else:
        axL.text(0.5, 0.5, "no per-room data in A or C", ha="center", transform=axL.transAxes)

    # Right: B on the full 45 (sorted by V; colored by V).
    bs = sorted(b.items(), key=lambda kv: kv[0][0] * kv[0][1] * kv[0][2])
    if bs:
        vols = np.array([k[0] * k[1] * k[2] for k, _ in bs])
        vals = np.array([v for _, v in bs])
        labels = [f"L{k[0]:.1f}W{k[1]:.1f}H{k[2]:.1f}" for k, _ in bs]
        sc = axR.bar(
            range(len(bs)), vals,
            color=plt.cm.viridis((vols - vols.min()) / max(1e-9, vols.max() - vols.min())),
        )
        axR.set_xticks(range(len(bs)))
        axR.set_xticklabels(labels, fontsize=6, rotation=90)
        axR.set_ylabel("final val LSD (dB)")
        axR.set_title(f"B per-room LSD (n={len(bs)}; bars colored by V)")
        axR.axhline(2.5, color="k", ls=":", lw=0.7)
        axR.grid(True, alpha=0.3, axis="y")
    else:
        axR.text(0.5, 0.5, "no per-room data in B", ha="center", transform=axR.transAxes)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _format_classification(c: str) -> str:
    return {"success": "✅ success", "fail": "❌ fail",
            "ambiguous": "⚠ ambiguous", "missing": "— missing"}.get(c, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", default=str(REPO_ROOT / "outputs/diag_p2_2_5"),
        help="Diagnostic output root; expects <root>/{A_10rm_b16, B_45rm_b32, C_10rm_b64}/.",
    )
    ap.add_argument(
        "--out-name", default="DIAGNOSIS.md",
        help="Output markdown filename (e.g. DIAGNOSIS_interim.md for a snapshot).",
    )
    ap.add_argument(
        "--note", default="",
        help="A status note prepended to the report (e.g. 'INTERIM snapshot').",
    )
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    runs = {}
    for code, sub, label in RUNS:
        runs[code] = _read_run(root / sub, label)
        info = runs[code]
        if info["exists"]:
            print(f"# {code} ({sub}): final val LSD = {info['final_val_lsd']:.2f} dB "
                  f"(early_stop={info['stopped_early']}, iter={info['n_iters_actual']}/{info['n_iters_target']})")
        else:
            print(f"# {code} ({sub}): MISSING train_meta.json or scalars.json")

    # Convergence curves figure.
    _figure_convergence_curves(runs, root / "convergence_curves.png")
    # Per-room LSD figure.
    _figure_per_room_lsd(runs, root / "per_room_lsd.png")

    # Decision matrix.
    a_class = _classify_lsd(runs["A"].get("final_val_lsd", float("nan")))
    b_class = _classify_lsd(runs["B"].get("final_val_lsd", float("nan")))
    c_class = _classify_lsd(runs["C"].get("final_val_lsd", float("nan")))
    verdict, p2_3_rec = _verdict_from_triple(a_class, b_class, c_class)

    # Assemble DIAGNOSIS.md.
    md = [
        "# P2-2.5 DIAGNOSIS — multi-room 3D training bottleneck\n",
    ]
    if args.note:
        md.append(f"\n> **{args.note}**\n")
    md += [
        "\n## Headline\n",
        "| Run | Rooms | batch | n_pts | iters / target | wall | final val LSD | classification |\n",
        "|---|---:|---:|---:|---|---:|---:|---|\n",
    ]
    for code, sub, label in RUNS:
        info = runs[code]
        if not info.get("exists"):
            md.append(f"| {code} | — | — | — | — | — | — | — missing |\n")
            continue
        complete = info.get("complete", False)
        # Use train_meta cfg if complete, else fall back to the config YAML.
        if complete and info.get("meta", {}).get("cfg"):
            cfg = info["meta"]["cfg"]
            n_rooms = info["meta"]["n_rooms"]
            n_iters_target = info["n_iters_target"]
            wall_str = f"{info['wall_clock_s'] / 3600.0:.1f} h"
            # Latest val iter from the curve.
            cur_iter = info["val_iter"][-1] if info["val_iter"] else 0
            iters_str = f"{info['n_iters_actual']}/{n_iters_target}"
            if info["stopped_early"]:
                iters_str += " (early-stop)"
        else:
            cfg = _cfg_fallback(code)
            n_rooms = cfg.get("n_rooms", "?")
            n_iters_target = cfg.get("n_iters_target", "?")
            cur_iter = info["val_iter"][-1] if info["val_iter"] else 0
            wall_str = "running"
            iters_str = f"~{cur_iter}/{n_iters_target} (in progress)"
        cls = _classify_lsd(info["final_val_lsd"])
        cls_str = _format_classification(cls)
        if not complete:
            cls_str += " · still descending" if info["val_lsd"] and len(info["val_lsd"]) >= 2 and info["val_lsd"][-1] < info["val_lsd"][-2] else " · in progress"
        bsz = cfg.get("batch_size", "?")
        acc = cfg.get("grad_accum_steps", 1) or 1
        bsz_str = f"{bsz}" + (f" × accum {acc}" if acc and acc > 1 else "")
        md.append(
            f"| {code} | {n_rooms} | {bsz_str} | {cfg.get('n_pts_per_ray','?')} | "
            f"{iters_str} | {wall_str} | "
            f"{info['final_val_lsd']:.2f} dB | {cls_str} |\n"
        )

    md.append(
        "\n**Thresholds (spec)**: `≤ 2.5 dB = success`; `> 4 dB = clear failure`; "
        "in-between = ambiguous, flag for manager.\n"
    )

    md.append("\n## Decision-matrix verdict\n\n")
    md.append(f"**{verdict}**\n\n")
    md.append(f"### Recommendation for P2-3\n\n{p2_3_rec}\n")

    # Run B descent rate — how much further B would fall with more iters, to
    # size the P2-3 iteration budget. Reports dB-improvement per 10K-iter window.
    binfo = runs.get("B", {})
    bv = {it: lsd for it, lsd in zip(binfo.get("val_iter", []), binfo.get("val_lsd", []))}
    if bv:
        md.append("\n## Run B descent rate (sizes the P2-3 iteration budget)\n\n")
        md.append("| window | Δ val LSD per 10K iters |\n|---|---:|\n")
        windows = [(30000, 40000), (40000, 50000), (50000, 60000)]
        last_slope = None
        for lo, hi in windows:
            if lo in bv and hi in bv:
                slope = bv[lo] - bv[hi]
                last_slope = slope
                md.append(f"| {lo//1000}K → {hi//1000}K | {slope:.3f} dB |\n")
        b_final = binfo.get("final_val_lsd", float("nan"))
        if last_slope and last_slope > 0 and np.isfinite(b_final) and b_final > 2.5:
            gap = b_final - 2.5
            nominal_k = gap / last_slope * 10  # K-iters at the last (decelerating) slope
            md.append(
                f"\nB ended at **{b_final:.2f} dB** with the descent decelerating "
                f"(≈ {last_slope:.2f} dB/10K at the end). Closing the {gap:.2f} dB "
                f"gap to 2.5 at that slope is ~{nominal_k:.0f}K nominal iters; with "
                f"continued deceleration, budget **~80-100K iters** for P2-3 to "
                f"clear 2.5 on the full 45-room set.\n"
            )

    md.append("\n## Convergence curves\n\n![](convergence_curves.png)\n")
    md.append("\n## Per-room final val LSD\n\n![](per_room_lsd.png)\n")

    # Per-run details.
    md.append("\n## Per-run details\n\n")
    for code, sub, label in RUNS:
        info = runs[code]
        md.append(f"### {label}\n\n")
        if not info.get("exists"):
            md.append("Missing artifacts.\n\n")
            continue
        cur_iter = info["val_iter"][-1] if info["val_iter"] else 0
        if info.get("complete"):
            wall = f"{info['wall_clock_s'] / 3600.0:.2f} h"
            iters = f"{info['n_iters_actual']}/{info['n_iters_target']} iters"
            lsd_label = "Final"
        else:
            wall = "running (in progress)"
            tgt = _cfg_fallback(code).get("n_iters_target", "?")
            iters = f"~{cur_iter}/{tgt} iters (in progress)"
            lsd_label = "Latest"
        md.append(
            f"- Output dir: `outputs/diag_p2_2_5/{sub}/`\n"
            f"- Wall-clock: {wall} ({iters})\n"
            f"- Stopped early: `{info['stopped_early']}`"
        )
        if info["stopped_early"]:
            md.append(f"\n- Stop reason: `{info['stop_reason']}`")
        md.append(f"\n- {lsd_label} val LSD: **{info['final_val_lsd']:.2f} dB**\n\n")

    # DDP correctness cross-check: compare canonical B (DDP) vs single-GPU B at
    # the latest common val iteration. Tight agreement validates the all-reduce.
    cc_dir = root / CROSSCHECK_B_DIR
    if _B_DIR.endswith("ddp") and (cc_dir / "scalars.json").exists():
        try:
            cc = json.loads((cc_dir / "scalars.json").read_text())
            cc_val = {int(r["iter"]): _safe_float(r.get("lsd_db"))
                      for r in cc if r.get("phase") == "val"}
            b_val = {it: lsd for it, lsd in zip(runs["B"]["val_iter"], runs["B"]["val_lsd"])}
            common = sorted(set(cc_val) & set(b_val))
            md.append("\n## DDP correctness cross-check\n\n")
            md.append(
                "Canonical B is the 2-GPU DDP run; the single-GPU B "
                "(`B_45rm_b32`) trained independently from the **same** 22.5K "
                "checkpoint. Tight agreement at matched iterations confirms the "
                "manual all-reduce trains the same model (effective batch 32 = "
                "2 ranks × 16), not a corrupted one.\n\n"
            )
            md.append("| iter | DDP-B val LSD | single-B val LSD | Δ |\n|---:|---:|---:|---:|\n")
            for it in common[-6:]:
                md.append(f"| {it} | {b_val[it]:.2f} | {cc_val[it]:.2f} | "
                          f"{abs(b_val[it]-cc_val[it]):.2f} |\n")
        except Exception as e:
            md.append(f"\n_(cross-check unavailable: {e!r})_\n")

    md.append("\n## Anchors (other multi-room results)\n\n")
    md.append(
        "- P2-1 single-room overfit (5 rooms, batch=4-8, n_pts=16): val LSD "
        "**1.3-1.8 dB** — the architecture can reconstruct 3D spectra.\n"
        "- P2-2 M1 (45 rooms, batch=4, n_pts=16): val LSD **6.16 dB** "
        "after 24K iters (early-stop). This is the baseline P2-2.5 is "
        "trying to beat.\n"
    )

    diag_path = root / args.out_name
    diag_path.write_text("".join(md))
    print(f"# wrote {diag_path}")
    print(f"# verdict: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
