# Chunk P2-VIZ — Meeting visualization pack

**Status**: COMPLETE — 2026-06-09. CPU/plotting (one ~30 s GPU render for Fig 2a; the
LOO spectra weren't cached). No retraining. All data from P2-3.5 / P2-3 / P2-2.5 / M1.

## Deliverable
6 polished, traceable figures (1920×1080) in `outputs/phase2_meeting_assets/`, with
`FIGURE_MANIFEST.md` (per-figure source + exact numbers + honest caption). Story arc:

1. **01_latent_manifold_3d** — representation works (R² 0.991/0.967/0.974). *(reused)*
2. **02_known_geometry_works** — the positive: LOO median-room spectrum overlay +
   45-room distribution; **0.89 mag corr at training density, no measurements**.
3. **03_in_distribution_solved** — the engineering win: 6.43 → **2.169 dB** over 60K.
4. **04_modal_density_2d_vs_3d** — why 3D is hard: 3D 135 vs 2D ~12 modes (~11×).
5. **05_coverage_diagnosis** — the rigor: 8-recv / lookup / on-manifold-oracle all ~0.27
   vs LOO 0.89 → coverage, ruled out three ways.
6. **06_the_density_lever** — the path, **honestly two-regime**: within trained
   geometries denser→better (corr −0.74); untrained rooms fail flat → more rooms (P2-4).

## Honesty
- Fig 2 = leave-one-out at **training density**, not a claim about arbitrary new rooms.
- Fig 4 distinct-mode count is **135** (136th is DC); ~11× unchanged.
- Fig 6 is **two regimes**, not a continuous curve — NN-distance alone doesn't separate
  trained from untrained; stated explicitly in the caption + manifest.

## Reproduce
`python scripts/make_p2viz_pack.py` (figs 2–6); Fig 2a render via
`scripts/slurm/render_loo.sh`. Manifest: `outputs/phase2_meeting_assets/FIGURE_MANIFEST.md`.
