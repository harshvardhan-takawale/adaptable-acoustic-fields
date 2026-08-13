# Chunk P3-2 — 2D material-editable acoustic field — RESULTS

**Does the split-(iii) demo claim hold — unseen geometry × never-seen (wall, material) combination? NO, not at usable strength — but the weaker claim it brackets does hold, cleanly.** On unseen geometry with *trained* combinations (split i) the model reproduces the edit at **essentially correct magnitude: slope 1.059, r 0.896, E_BW 1.215 Hz, edit_gain 1.033**. On never-seen combinations (split iii) it gets the *direction* right and identifies the *correct wall* (r 0.516; wall-identity asymmetry **+0.515** / **+0.201** against α_eff-matched twins) but recovers only **~13 % of the edit magnitude (slope 0.126)**, and **edit_gain 0.862 < 1** — i.e. its edited prediction is not yet better than its own baseline. Compositional transfer is therefore *emerging but not achieved*. **Scoping (D48): the ground truth obeys the SIMULATOR's ray absorption law (ΔAIC = 73 vs Kuttruff); real locally-reacting walls would give ~2:1 rather than ~29:1 selectivity, so every number here supports "the model learns the simulator's per-wall law", not "wall-selective editing works in real rooms".**

## 1. The physics premise is established (blocking gate: PASS)

Before any training, on ISM ground truth via a 64-receiver modal projection (`outputs/p3_2/SIM_VALIDATION.md`):

| edit (4.5×4.0) | ΔBW x-axial | ΔBW y-axial | ΔBW tangential | selectivity |
|---|---:|---:|---:|---:|
| west → M3 | **+10.183** | +0.207 | +6.262 | **49 : 1** |
| south → M3 | +0.341 | **+11.464** | +7.943 | 34 : 1 |
| west → M1 (concrete) | **−1.160** | −0.023 | −0.721 | sharpens ✓ |

Block-diagonal on all four walls, strictly monotone M1 < M0 < M2 < M3, bidirectional. Pooled selectivity **29.1 (95 % CI [20.0, 39.3])** over 2 rooms × 4 walls; T1/T2/T3/T4/T5 all 8/8. Verified robust to paired mode intersection (identical numbers). `max_order=60` confirmed converged (60→120: ΔBW 0.0 %, Δlevel 0.05 dB). Wall convention proven by image-lattice probe.

**Which damping law**: ISM-ray R² = 0.998 vs Kuttruff 0.982, **ΔAIC = 73 favouring ISM-ray** (D48).

## 2. Model

One model, 440 training configs (40 geometries × 11), conditioning = 64-D Fourier features of (L, W, α_west, α_east, α_south, α_north) → FiLM, **no latent table**. Band-limited 0–300 Hz protocol.

**In-distribution val LSD (held-out receivers of training configs): 2.748 dB at 40 000 iters — PLATEAUED** (2.749 at 32 K → 2.748 at 40 K, 0.04 % per 8 K). Train `L_amp` 0.060 vs val 0.143. Every zero-shot number below is qualified by this: the model is **still over-damped** — its rendered baseline modes are materially wider than ground truth — and mode broadening is exactly the observable the claim rests on, so this convergence level is the binding constraint, not the conditioning mechanism.

## 3. Zero-shot results (no measurements, no per-config optimization)

All edit metrics are PAIRED deltas (prediction vs its own predicted baseline, GT vs its own GT baseline, identical estimator on both streams — D49).

| split | n | E_BW (Hz) ↓ | slope → 1 | r | edit_gain > 1 | mag corr |
|---|---:|---:|---:|---:|---:|---:|
| (i) unseen geom × **seen** combo | 110 | **1.215** | **1.059** | **0.896** | **1.033** | 0.863 |
| (ii) seen geom × held-out combo | 80 | 5.472 | 0.040 | 0.543 | 0.907 | 0.961 |
| **(iii) unseen geom × held-out combo** | 20 | 5.002 | 0.126 | 0.516 | 0.862 | 0.878 |
| (iv) unseen α = 0.30 | 40 | 2.249 | 0.331 | 0.293 | 0.681 | 0.864 |

- **(i) → (iii) gap: 1.215 → 5.002 Hz (+3.787 Hz)**, ~95× the measured estimator floor (C2 = 0.040 Hz), so the gap is real and not measurement noise.
- **(ii) ≈ (iii)** (5.47 vs 5.00) — the deficit is *combination* novelty, not *geometry* novelty. Geometry generalization is solved; composition is not. This decomposition is why split (ii) was added beyond the spec.
- Held-out combos reported separately: **(west, M2)** slope 0.104→ r 0.406 and **(north, M3)** r 0.114 at 6 K, both improving by 40 K; (west, M2) — material-value transfer onto a *seen* wall — is consistently the stronger of the two, while (north, M3) — transferring a *seen material to a new wall* — is the harder direction.
- **Selectivity index: GT 18.2, predicted 10.6, theory 33.0** (predicted was 4.6 at 6 K → 10.6 at 40 K, i.e. still climbing when the fit plateaued).

## 4. Controls (a claim without these is not supportable)

| control | result | reading |
|---|---|---|
| **C1** null model (`edit_gain`) | (i) **1.033**; (ii) 0.907; (iii) **0.862**; (iv) 0.681 | only split (i) beats predicting the unedited baseline |
| **C2** estimator floor | **0.040 Hz** (x 0.041, y 0.039) | axial families share one damping rate under the ray law, so this spread is pure estimator noise |
| **C3** conditioning identity | **True** | (wall k, M0) is bitwise identical to baseline — required `renderer.eval()`, see D49 |
| **C4** wall identity vs α_eff-matched twin | (west,M2) **+0.515**; (north,M3) **+0.201** | both positive ⇒ the model has NOT collapsed the four absorptions to a scalar |

C4 is the decisive control: each held-out combo's trained opposite-wall twin has *identical* mean absorption and T60 and differs only in *where* the absorber sits, so a scalar-absorption model scores ≈0. Both are positive and both rose with training (+0.297 → +0.515, −0.048 → +0.201), which is the clearest evidence that wall identity is being learned even where magnitude is not.

## 5. Interpretation

Three findings, in decreasing confidence:

1. **The measurement framework and the physics are established.** The gate passes decisively and every primitive is validated against a closed form (calibrated ISM-ray predicts measured bandwidths to <1 %).
2. **Geometry transfer of material edits works.** Split (i) slope 1.059 / r 0.896 / edit_gain 1.033 on 110 unseen-geometry configs is a genuine positive: a single conditioned model renders the right material edit on rooms it never saw.
3. **Compositional (wall × material) transfer is emerging, not achieved.** Direction and wall identity transfer (r ≈ 0.52, C4 > 0), magnitude does not (slope 0.13, edit_gain < 1).

The most probable cause of (3) is **not** the conditioning mechanism but the **fit**: the model plateaued at 2.748 dB while still over-damped, and blur inflates every bandwidth, which compresses exactly the differences split (iii) measures — note the predicted selectivity index was still climbing (4.6 → 10.6) when the loss stopped improving. **Next step is a sharper recipe, not a different conditioning design**: more ray samples (`n_pts_per_ray` 32 → 64, the Phase-1 2D value), a larger hash grid, or a spectral-sharpness term. Only after the in-distribution fit resolves modal widths is split (iii) a fair test of composition.

## 6. Deliverables

`outputs/p3_2/SIM_VALIDATION.md` (gate) · `outputs/p3_2/eval/summary.json` · `scripts/demo_edit_2d.py` (+README, runs in ~3 s) · `scripts/make_p3_2_figures.py` · `configs/sweeps_2d_mat/` (40 train + 10 FROZEN test geometries) · `data/track_c_2d/` (690 configs) · 6 new CPU test files (49 tests). Decisions **D44–D50**; open questions **Q15–Q16**.

**Not run**: multi-wall edits (out of scope); a converged-fit rerun of the eval (the recommended next step above).
