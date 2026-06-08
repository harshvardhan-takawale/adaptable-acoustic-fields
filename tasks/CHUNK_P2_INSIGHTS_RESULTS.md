# Chunk P2-INSIGHTS — Phase 2 meeting-deck figures

**Status**: COMPLETE — 2026-06-08. CPU/plotting only; ran in parallel with P3
training (no GPU contention).

## Deliverable

Five polished, meeting-grade figures (all exactly **1920×1080**) in
`outputs/phase2_meeting_assets/`, each backed by data already on disk — no new
experiments. Every number is traceable to a named source file; the manifest
(`outputs/phase2_meeting_assets/FIGURE_MANIFEST.md`) lists the exact values and
sources, and the generators (`scripts/phase2_figs/*.py`) are version-controlled
so any figure is reproducible.

| # | Figure | Headline | Source |
|---|--------|----------|--------|
| 1 | `01_latent_manifold_3d.png` | "The latent space autonomously encodes all three room dimensions" — 45 M1 latents on PC1/PC2, colored by L/W/H, probe R² **0.991 / 0.967 / 0.974** | M1 `latent_probe.json` + checkpoint |
| 2 | `02_modal_density_2d_vs_3d.png` | "~11× higher modal density below Schroeder" — 3D **135** distinct modes ≤250 Hz vs 2D ~12 | recomputed via `eigenfrequencies_3d`; P2-1 §5 |
| 3 | `03_diagnostic_convergence.png` | "Coverage, not capacity, was the bottleneck" — A/B/C val-LSD curves vs the M1 6.16 dB plateau | `diag_p2_2_5` scalars + M1 scalars |
| 4 | `04_representation_vs_rendering.png` | "Representation and rendering are separable" — latent R² high while LSD lagged at 6.16 dB until coverage fixed (→2.61→0.98) | `latent_probe.json` + scalars |
| 5 | `05_phase2_progress.png` | "Rapid progress once the bottleneck was identified" — M1 6.16 → B 2.61 → **P3 PROJECTED** band | M1 + B + P3 scalars |

## How it was produced (multi-agent, verified)

Authored a 6-agent workflow (`.claude/workflows/phase2_figs.js`): one agent per
figure under a **shared style spec** (exact 1920×1080, consistent palette, honest
captions) plus a final **manifest/critic** agent. Each figure agent was required
to *load every number from its named source file at run time* (no hand-entered
values) and to verify the PNG dimensions; the critic then **independently
re-read every number from disk** (not trusting the agents) and confirmed
dimensions. Verdict: **PASS**, 5/5.

## Honesty notes (caught by the traceability mandate)

These are disclosed in `FIGURE_MANIFEST.md` and do not change any headline:

1. **Modal count 135, not 136.** `eigenfrequencies_3d(4.5,4.0,3.25,f_max=250)`
   returns 136 deduplicated entries, but one is the DC (0,0,0) term at f=0. The
   strict 0<f≤250 Hz rule gives **135** distinct non-DC modes (highest 249.85 Hz;
   93 modes ≤ f_Schroeder≈217 Hz). P2-1 §5's "136" counted DC. The ~11× ratio
   (135/12) and the headline are unchanged. *(P2-1's text is historical; not
   retroactively edited.)*
2. **Fig 5 P3 is a PROJECTION, not a result.** P3 is drawn as a hatched,
   translucent band over the 1.8–2.2 dB target with the **live current value
   (4.54 dB @ 10K) labeled separately and shown above the band** — honest that
   training is in progress and not yet at target.
3. **Fig 1 PCA.** Axis-label variance %s come from the canonical stored
   `pca_explained_variance`; the scatter point geometry uses the generator's own
   SVD projection (slightly different EV) — affects point positions only, never
   the labeled numbers.
4. **Run B canonical = DDP variant** (2.61 dB); the single-GPU B finished at
   2.675 dB. Fig 4 pairs B (45 rm) and C (10 rm) as coverage levers, not two
   values for one room set.

## Reproduce

```bash
conda activate aaf && export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
for f in scripts/phase2_figs/*.py; do python "$f"; done
```

Manifest: `outputs/phase2_meeting_assets/FIGURE_MANIFEST.md`.
