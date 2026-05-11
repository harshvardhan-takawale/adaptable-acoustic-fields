"""Aggregate the 6 Chunk-3.5 sweep runs into SWEEP_SUMMARY.md + 4 headline figures.

Reads each ``outputs/multi_room/sweep/R*/`` and produces:
  - outputs/multi_room/sweep/SWEEP_SUMMARY.md
  - outputs/multi_room/sweep/figures/{best_config_zero_shot_overlay,
        best_config_receiver_grid, best_config_latent_pca,
        zero_shot_lsd_comparison}.png

The "best config" is identified by the priority order from the spec:
  1. count of unseen L with held-out LSD ≤ 2 dB (more is better)
  2. mean held-out LSD across all 6 unseen L (lower is better)
  3. PC1-vs-L R² (higher is better)
  4. train aggregate val LSD (lower is better; tiebreaker)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Order to display in tables / bar charts (R0..R5 from chunk-3.5; R6..R8 from
# chunk-3.5+ addendum). Runs missing from disk are silently skipped.
RUN_ORDER = [
    "R0_central", "R1_smaller_hash", "R2_larger_latent",
    "R3_no_lhead", "R4_strong_lhead", "R5_strong_l2",
    "R6_tiny_lhead", "R7_medium_hash", "R8_tiny_latent",
]


def _load_run(root: Path, run_id: str) -> dict | None:
    """Aggregate one run's results. Returns None if the run dir is missing or
    incomplete (e.g., training failed before scalars or some zero-shot didn't run)."""
    rd = root / run_id
    if not rd.exists():
        return None
    out: dict = {"run_id": run_id, "dir": str(rd)}

    # Hyperparameters from the YAML (for the configuration table).
    cfg_yaml_path = REPO_ROOT / "configs/sweep" / f"{run_id}.yaml"
    if cfg_yaml_path.exists():
        out["cfg"] = yaml.safe_load(open(cfg_yaml_path))

    # Train meta.
    tm_path = rd / "train_meta.json"
    if tm_path.exists():
        meta = json.loads(tm_path.read_text())
        out["train_meta"] = meta
        out["status"] = "completed" if int(meta["n_iters_actual"]) == int(meta["n_iters_target"]) \
                        else f"early-stopped@{meta['stop_iter']}" if meta["stopped_early"] \
                        else "incomplete"
    else:
        out["status"] = "no train_meta"

    # Final val metrics from scalars.
    sc_path = rd / "scalars.json"
    if sc_path.exists():
        scalars = json.loads(sc_path.read_text())
        vals = [r for r in scalars if r.get("phase") == "val"]
        if vals:
            last = vals[-1]
            out["final_val"] = {
                "iter": last["iter"],
                "agg_lsd_db": last.get("lsd_db"),
                "agg_complex_l1": last.get("complex_l1"),
                "agg_phase_l1": last.get("phase_l1"),
                "L_lhead": last.get("L_lhead"),
                "latent_norm_mean": last.get("latent_norm_mean"),
            }
            out["per_room_lsd"] = {
                L: last.get(f"L_{L}_lsd_db")
                for L in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
            }

    # Zero-shot metrics.
    zs_dir = rd / "zero_shot"
    zs_results = []
    if zs_dir.exists():
        for d in sorted(zs_dir.glob("L*/")):
            mp = d / "metrics.json"
            if mp.exists():
                zs_results.append(json.loads(mp.read_text()))
    out["zs_results"] = sorted(zs_results, key=lambda r: r["L"])

    if zs_results:
        held = [r["held_out_lsd_db"] for r in zs_results]
        out["zs_mean_held"] = float(np.mean(held))
        out["zs_count_below_2"] = int(sum(1 for h in held if h <= 2.0))
        out["zs_max_held"] = float(np.max(held))
        out["zs_min_held"] = float(np.min(held))
    else:
        out["zs_mean_held"] = None
        out["zs_count_below_2"] = 0

    # Latent probe.
    lp_path = rd / "latent_probe" / "latent_probe.json"
    if lp_path.exists():
        lp = json.loads(lp_path.read_text())
        out["pc1_vs_L_r2"] = lp.get("pc1_vs_L_r2")
        out["intrinsic_dim"] = lp.get("intrinsic_dim_95pct")
    return out


def _pick_best(runs: list[dict]) -> dict | None:
    """Apply the 4-tier priority order. Skips runs missing essential metrics."""
    valid = [r for r in runs if r.get("zs_results") and r.get("final_val")]
    if not valid:
        return None
    valid.sort(key=lambda r: (
        -r["zs_count_below_2"],
        r["zs_mean_held"] if r["zs_mean_held"] is not None else 1e9,
        -(r.get("pc1_vs_L_r2") or -1e9),
        r["final_val"].get("agg_lsd_db") or 1e9,
    ))
    return valid[0]


def _md_config_table(runs: list[dict]) -> str:
    out = ["\n## 1. Configuration table\n",
           "| Run | log2_hash | n_levels | latent_dim | L-head wt | λ_latent_L2 |\n",
           "|-----|---------:|---------:|-----------:|----------:|------------:|\n"]
    for r in runs:
        c = r.get("cfg", {})
        if not c:
            continue
        out.append(f"| {r['run_id']} | {c['log2_hashmap_size']} | {c['n_levels']} | "
                   f"{c['latent_dim']} | {c['l_head_weight']} | {c['lambda_latent_l2']} |\n")
    return "".join(out)


def _md_results_table(runs: list[dict]) -> str:
    out = ["\n## 2. Per-run results table\n",
           "| Run | Status | Train agg LSD (dB) | ZS mean held LSD (dB) | ZS LSD ≤ 2 dB | "
           "PC1-vs-L R² | intrinsic_dim |\n",
           "|-----|--------|-------------------:|---------------------:|--------------:|"
           "------------:|--------------:|\n"]
    for r in runs:
        train_lsd = r.get("final_val", {}).get("agg_lsd_db")
        train_lsd_str = f"{train_lsd:.3f}" if train_lsd is not None else "—"
        zs_mean = r.get("zs_mean_held")
        zs_mean_str = f"{zs_mean:.3f}" if zs_mean is not None else "—"
        zs_count = r.get("zs_count_below_2", 0)
        r2 = r.get("pc1_vs_L_r2")
        r2_str = f"{r2:.3f}" if r2 is not None else "—"
        idim = r.get("intrinsic_dim")
        idim_str = str(idim) if idim is not None else "—"
        out.append(f"| {r['run_id']} | {r.get('status', '?')} | {train_lsd_str} | "
                   f"{zs_mean_str} | {zs_count} / 6 | {r2_str} | {idim_str} |\n")
    return "".join(out)


def _ablation_md(runs: list[dict]) -> str:
    by_id = {r["run_id"]: r for r in runs}
    R0 = by_id.get("R0_central")
    out = ["\n## 4. Ablation interpretation\n"]
    if R0 is None or R0.get("zs_mean_held") is None:
        out.append("*(R0 missing; cannot run ablation comparisons)*\n")
        return "".join(out)
    r0_zs = R0.get("zs_mean_held", float("nan"))
    r0_n = R0.get("zs_count_below_2", 0)
    r0_r2 = R0.get("pc1_vs_L_r2", float("nan"))

    pairs = [
        ("R1_smaller_hash", "smaller hash (12 vs 14 bits)"),
        ("R2_larger_latent", "larger latent (16 vs 8 dims)"),
        ("R3_no_lhead", "no L-head (0.0 vs 0.1)"),
        ("R4_strong_lhead", "stronger L-head (1.0 vs 0.1)"),
        ("R5_strong_l2", "stronger latent L2 (1e-2 vs 1e-4)"),
        ("R6_tiny_lhead", "linear L-head (vs mlp_32)"),
        ("R7_medium_hash", "linear L-head + medium hash (16 vs 14)"),
        ("R8_tiny_latent", "linear L-head + tiny latent (2 vs 8 dims)"),
    ]
    for rid, label in pairs:
        r = by_id.get(rid)
        if r is None or r.get("zs_mean_held") is None:
            out.append(f"- **{rid}**: {label} — run incomplete or missing\n")
            continue
        out.append(
            f"- **{rid}** vs R0 ({label}): "
            f"ZS mean held {r['zs_mean_held']:.3f} dB (Δ {r['zs_mean_held']-r0_zs:+.3f} dB), "
            f"≤2dB count {r['zs_count_below_2']}/6 (Δ {r['zs_count_below_2']-r0_n:+d}), "
            f"R² {(r.get('pc1_vs_L_r2') if r.get('pc1_vs_L_r2') is not None else float('nan')):.3f} "
            f"(Δ {(r.get('pc1_vs_L_r2') or 0) - r0_r2:+.3f}).\n"
        )
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO_ROOT / "outputs/multi_room/sweep"))
    ap.add_argument("--showcase_L", type=str, default="4.25",
                    help="L value whose figures get copied as the headline visuals")
    args = ap.parse_args()
    root = Path(args.root)
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for rid in RUN_ORDER:
        r = _load_run(root, rid)
        if r is not None:
            runs.append(r)
    if not runs:
        raise RuntimeError(f"no runs found under {root}")

    best = _pick_best(runs)

    # SWEEP_SUMMARY.md.
    md = ["# Chunk-3.5 sweep summary\n",
          "\nGenerated by `scripts/sweep_summary.py` after the full pipeline completes.\n"]
    md.append(_md_config_table(runs))
    md.append(_md_results_table(runs))

    md.append("\n## 3. Best configuration\n")
    if best is None:
        md.append("**No run produced complete results.** All runs missing essential metrics; see "
                  "`tasks/CHUNK_3_5_RESULTS.md` for diagnosis.\n")
    else:
        md.append(f"**{best['run_id']}** wins by the spec priority order "
                  "(count below 2 dB → mean LSD → R² → train LSD).\n\n")
        md.append(f"- Train agg val LSD: **{best['final_val']['agg_lsd_db']:.3f} dB**\n")
        md.append(f"- ZS mean held-out LSD: **{best['zs_mean_held']:.3f} dB**\n")
        md.append(f"- ZS count below 2 dB: **{best['zs_count_below_2']} / 6**\n")
        md.append(f"- PC1 vs L R²: **{best.get('pc1_vs_L_r2', float('nan')):.3f}**\n")
        md.append(f"- intrinsic_dim (95% var): **{best.get('intrinsic_dim', '—')}**\n")
        if best.get("final_val", {}).get("L_lhead") is not None:
            md.append(f"- L-head val MAE: **{best['final_val']['L_lhead']:.3f} m**\n")

        meets = (
            best["zs_count_below_2"] >= 4
            and (best.get("pc1_vs_L_r2") or 0) > 0.7
            and (best.get("intrinsic_dim") or 99) <= 3
        )
        md.append(f"\n**Meets full meeting bar?** {'Yes' if meets else 'No'} "
                  "(targets: ZS ≤ 2 dB on ≥ 4/6 + PC1 R² > 0.7 + intrinsic_dim ≤ 3).\n")

    md.append(_ablation_md(runs))

    md.append("\n## 5. Headline figures\n"
              "- ![best zero-shot overlay](figures/best_config_zero_shot_overlay.png)\n"
              "- ![best receiver grid](figures/best_config_receiver_grid.png)\n"
              "- ![best latent PCA](figures/best_config_latent_pca.png)\n"
              "- ![ZS LSD comparison](figures/zero_shot_lsd_comparison.png)\n")

    md.append("\n## 6. Per-run artifacts\n")
    for r in runs:
        md.append(f"- **{r['run_id']}**: [training_curves]({r['run_id']}/figures/training_curves.png), "
                  f"[latent PCA]({r['run_id']}/latent_probe/figures/latent_pca_1d.png), "
                  f"[ZS L=4.25 overlay]({r['run_id']}/zero_shot/L4.25/figures/zero_shot_overlay.png)\n")

    md.append("\n## 7. Recommendations for next steps\n")
    if best is None:
        md.append("- All runs failed to produce complete results. Investigate logs in "
                  "`logs/slurm/aaf_sweep_train-*` and `outputs/multi_room/sweep/*/train_meta.json`.\n")
    elif best["zs_count_below_2"] >= 4 and (best.get("pc1_vs_L_r2") or 0) > 0.7:
        md.append(f"- **Use {best['run_id']}** as the meeting result. It meets the headline targets.\n")
    else:
        md.append(f"- **{best['run_id']}** is the best of the sweep but does not meet "
                  "the full meeting bar. Likely next experiments: see CHUNK_3_5_RESULTS.md "
                  "ablation interpretation.\n")

    (root / "SWEEP_SUMMARY.md").write_text("".join(md))

    # ---------- Headline figures ----------

    # 1-3: copy from the best config's per-room artifacts.
    if best is not None:
        bid = best["run_id"]
        showcase_L = args.showcase_L
        zs_dir = root / bid / "zero_shot" / f"L{showcase_L}" / "figures"
        for src_name, dst_name in [
            ("zero_shot_overlay.png", "best_config_zero_shot_overlay.png"),
            ("zero_shot_receiver_grid.png", "best_config_receiver_grid.png"),
        ]:
            src = zs_dir / src_name
            if src.exists():
                shutil.copy(src, fig_dir / dst_name)
        pca_src = root / bid / "latent_probe" / "figures" / "latent_pca_1d.png"
        if pca_src.exists():
            shutil.copy(pca_src, fig_dir / "best_config_latent_pca.png")

    # 4: cross-run bar chart of mean held-out LSD with 2 dB threshold.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = []; means = []; counts = []; colors = []
    for r in runs:
        if r.get("zs_mean_held") is None:
            continue
        xs.append(r["run_id"])
        means.append(r["zs_mean_held"])
        counts.append(r["zs_count_below_2"])
        colors.append("steelblue" if r["zs_count_below_2"] >= 4 else "indianred")
    bars = ax.bar(xs, means, color=colors, alpha=0.8)
    ax.axhline(2.0, color="k", linestyle="--", lw=0.8, label="2 dB target")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                f"{c}/6", ha="center", fontsize=8)
    ax.set_ylabel("mean held-out LSD across 6 unseen L (dB)")
    ax.set_title("Chunk-3.5 sweep: zero-shot held-out LSD per run\n"
                 "(blue = ≥4/6 unseen L meet 2 dB; red = misses target; numbers = count below 2 dB)")
    ax.set_xticklabels(xs, rotation=20, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "zero_shot_lsd_comparison.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"# wrote {root/'SWEEP_SUMMARY.md'} and 4 figures under {fig_dir}/")
    if best is not None:
        print(f"# best run: {best['run_id']}")


if __name__ == "__main__":
    main()
