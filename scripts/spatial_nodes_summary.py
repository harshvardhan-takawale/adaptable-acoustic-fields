"""V1 cross-L spatial summary (Chunk 3.7).

Reads ``outputs/spatial_nodes_check/L<L>/nodes_check.json`` for each unseen L,
builds a mode × L correlation matrix, and writes ``SUMMARY.md`` plus a single
overview figure.
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
DEFAULT_LS = (3.25, 3.75, 4.25, 4.75, 5.25, 5.75)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str,
                    default=str(REPO_ROOT / "outputs/spatial_nodes_check"))
    ap.add_argument("--Ls", nargs="+", type=float, default=list(DEFAULT_LS))
    ap.add_argument("--corr_threshold", type=float, default=0.7)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    per_L = {}
    for L in args.Ls:
        p = out_root / f"L{L}" / "nodes_check.json"
        if p.exists():
            per_L[float(L)] = json.loads(p.read_text())

    if not per_L:
        raise SystemExit(f"no per-L reports found under {out_root}")

    # Build a mode×L matrix. Modes are identified by (n_x, n_y); take the union
    # of (n_x, n_y) across L (modes shift slightly in frequency as L changes,
    # so the index identity is the safest invariant).
    all_mode_keys: list[tuple[int, int]] = []
    for L, data in sorted(per_L.items()):
        for m in data["modes"]:
            key = (m["n_x"], m["n_y"])
            if key not in all_mode_keys:
                all_mode_keys.append(key)
    Ls_sorted = sorted(per_L.keys())
    corr_matrix = np.full((len(all_mode_keys), len(Ls_sorted)), np.nan)
    for j, L in enumerate(Ls_sorted):
        for m in per_L[L]["modes"]:
            i = all_mode_keys.index((m["n_x"], m["n_y"]))
            corr_matrix[i, j] = m["corr"]

    # Per-L verdict summary.
    n_green = sum(1 for d in per_L.values() if d["verdict"] == "GREEN")
    n_yellow = sum(1 for d in per_L.values() if d["verdict"] == "YELLOW")
    n_red = sum(1 for d in per_L.values() if d["verdict"] == "RED")

    # Markdown table.
    lines = []
    lines.append(f"# Chunk 3.7 V1 — spatial-node cross-L summary")
    lines.append("")
    lines.append(f"GREEN: {n_green}  |  YELLOW: {n_yellow}  |  RED: {n_red}  "
                 f"(out of {len(per_L)} L values)")
    lines.append("")
    lines.append("## Per-L verdicts")
    lines.append("")
    lines.append("| L (m) | Verdict | modes ≥ {:.1f} corr | mean corr |".format(args.corr_threshold))
    lines.append("|---:|:---:|---:|---:|")
    for L in Ls_sorted:
        d = per_L[L]
        corrs = [m["corr"] for m in d["modes"]]
        lines.append(
            f"| {L:.2f} | **{d['verdict']}** | {d['n_good']}/{d['n_modes']} | "
            f"{np.mean(corrs):.3f} |"
        )
    lines.append("")
    lines.append("## Correlation matrix (modes × L)")
    lines.append("")
    lines.append("| mode \\ L | " + " | ".join(f"{L:.2f}" for L in Ls_sorted) + " |")
    lines.append("|---|" + "---|" * len(Ls_sorted))
    for i, (n_x, n_y) in enumerate(all_mode_keys):
        cells = []
        for j in range(len(Ls_sorted)):
            v = corr_matrix[i, j]
            cells.append("—" if np.isnan(v) else f"{v:.2f}")
        lines.append(f"| ({n_x},{n_y}) | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Figures:")
    lines.append("  - `figures/correlation_matrix.png` (this summary)")
    for L in Ls_sorted:
        lines.append(f"  - per-L grid at `L{L}/figures/all_modes_overview.png`")
    (out_root / "SUMMARY.md").write_text("\n".join(lines))

    # Heatmap figure.
    fig, ax = plt.subplots(figsize=(1.3 * len(Ls_sorted) + 2, 0.35 * len(all_mode_keys) + 1.5))
    masked = np.ma.masked_invalid(corr_matrix)
    im = ax.imshow(masked, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(Ls_sorted)))
    ax.set_xticklabels([f"{L:.2f}" for L in Ls_sorted], rotation=0)
    ax.set_yticks(range(len(all_mode_keys)))
    ax.set_yticklabels([f"({nx},{ny})" for nx, ny in all_mode_keys], fontsize=8)
    ax.set_xlabel("Unseen L (m)")
    ax.set_ylabel("Mode (n_x, n_y)")
    ax.set_title("Spatial Pearson correlation: predicted vs ISM at first 6 modes per L")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    # Annotate cells with values.
    for i in range(len(all_mode_keys)):
        for j in range(len(Ls_sorted)):
            v = corr_matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if 0.3 < v < 0.7 else "white")
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_dir / "correlation_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"# wrote {out_root/'SUMMARY.md'}")
    print(f"# wrote {fig_dir/'correlation_matrix.png'}")


if __name__ == "__main__":
    main()
