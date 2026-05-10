# Chunk 2 — Results

**Date**: 2026-05-10. **Scope**: 2D model + renderer port + single-room overfit baseline on L ∈ {3.0, 4.5, 6.0} m. Establishes the upper bound on per-room reconstruction quality at our chosen architecture before Chunk 3 wires in the auto-decoder.

---

## 1. Pipeline summary

| Stage | Module | SLURM script | Hardware | Wall-time |
|-------|--------|--------------|---------:|----------:|
| Tests (62) | `tests/test_*.py` | `scripts/slurm/run_pytest.sh` | scavenger CPU+1 GPU | <1 min |
| Memory check | `scripts/single_room_memory_check.py` | `scripts/slurm/memory_check.sh` | scavenger 1 GPU | 7 s |
| Single-room training (×3 parallel) | `aaf.train.single_room` | `scripts/slurm/single_room_train.sh L` | scavenger 1 GPU each, GTX TITAN X | **~1 h 32 min each** |
| Per-room eval (×3 parallel) | `aaf.eval.single_room_eval` | `scripts/slurm/single_room_eval.sh L` | scavenger 1 GPU | 41 s each |
| Cross-room summary | `scripts/single_room_summary.py` | (login, manual) | login | <2 s |

Orchestrator: `scripts/run_chunk2_pipeline.sh` chains everything via `--dependency=afterok`. End-to-end ~1 h 35 min on `legacygpu06` (3 GPUs running in parallel). No preemption events observed.

## 2. Port summary (3D INFER → 2D)

Vendored source: `aaf/_inference_ref/{inference_model,inference_renderer}.py`.

**Renderer** (`aaf/renderers/freq_2d.py:FreqRenderer2D`):
- Ray sampler: `ray_directions(n_azi, n_ele)` (3D spherical with elevation + extra zenith/nadir rays) → `_ray_directions_2d(n_azi)` (2D circular only, with per-iteration jitter; **n_azi=64**).
- Ray-AABB intersection: 6-wall slab → 4-wall slab via `_ray_aabb_intersect_2d`.
- Sample-position computation: `[B, n_azi, n_pts, 3]` → `[B, n_azi, n_pts, 2]`.
- Geometric attenuation: explicitly disabled (`use_geometric_attn=False`); kept as a constructor flag for forward compatibility.
- σ + jβ decomposition + cumulative transmittance + geometric phase: copied verbatim from the INFER reference, only tensor shapes differ.

**Model** (`aaf/models/inr_2d.py:INR2D_Single`):
- Six `tcnn.Encoding(3, ...)` → six `tcnn.Encoding(2, ...)`.
- Sigma encoder/decoder MLP topology unchanged from INFER defaults.
- Signal MLP topology unchanged.
- `tx_view=None` → zero vector (omni source — Phase 1 doesn't model speaker directivity).
- `z_s` accepted in `forward(...)` but ignored. Inline comments mark the two candidate-A injection points (sigma branch concat, signal branch concat).

**Loader** (`aaf/data/loader.py:ShoeboxDataset`): replaces the Chunk-1 stub. Reads HDF5 via `aaf.data.dataset_builder.read_room_h5`; one sample per (room, receiver) pair. `room_filter=[L]` is the single-room mode for Chunk 2.

**Surprises in the port**:
- The GTX TITAN X (compute capability 5.2) on `legacygpu06` doesn't support tcnn's `FullyFusedMLP`; tcnn falls back to `CutlassMLP` automatically (with a warning in stderr). No code changes needed.
- The dummy-loss test for backward gradient flow needed both `attn` and `signal` referenced — the sigma decoder branch only gets gradient if `attn` appears in the loss. (Caught by the first pytest run.)
- `nvidia-smi | head -10` triggers SIGPIPE which `set -euo pipefail` catches as failure. Fixed across SLURM scripts by using `nvidia-smi -L || true` instead.
- Eval initially OOM'd because it used `n_pts_per_ray=64` (the renderer default) while training used 32 from the memory check. Fixed by reading `train_meta.json` in eval to inherit the same renderer params.

## 3. Memory check

GPU: NVIDIA GeForce GTX TITAN X (12.8 GB).

| n_azi | n_pts_per_ray | batch | status | peak GB | fwd+bwd s |
|------:|--------------:|------:|--------|--------:|----------:|
| 64 | 64 | 8 | OOM (tried to allocate 1.0 GiB on 11.9 GiB device) | — | — |
| 64 | 32 | 8 | **PASS** | **8.09** | 0.76 |

**Chosen configuration**: `n_azi=64, n_pts_per_ray=32, batch=8`. Working set 8.09 GB on a 12 GB GPU (~67% utilisation, 4 GB headroom). Both training and eval use these values.

## 4. Single-room results

All three rooms trained the full 10K iters (no early-stop trigger — each window's improvement was 5-15%, well above the 1% threshold).

| L (m) | iters | wall (h) | modal MAE (Hz) | modal recall (modal-regime) | full-band LSD (dB) | envelope LSD (dB) | complex L1 |
|------:|------:|---------:|---------------:|----------------------------:|-------------------:|------------------:|-----------:|
| 3.0 | 10000 | 1.53 | **0.34** | 0.160 (15/94) | **0.42** | **0.14** | 0.058 |
| 4.5 | 10000 | 1.55 | **0.58** | 0.123 (16/130) | **0.39** | **0.11** | 0.040 |
| 6.0 | 10000 | 1.54 | **0.45** | 0.172 (21/122) | **0.36** | **0.10** | 0.033 |

**Reference** (Chunk 1.5 noise floor): ISM-vs-analytical MAE was 0.36 Hz (full-band) at the centre receiver; modal-regime recall ~0.10–0.20 across all 15 rooms. Chunk 2's predicted-vs-analytical MAE (0.34–0.58 Hz) is **at parity with the noise floor** — the model has effectively learned to reproduce ISM's spectrum.

Other observations:
- **Zero spurious peaks** across all 3 rooms (every picked peak in the predicted spectrum has an analytical eigenfrequency within tolerance).
- **Larger L → lower full-band LSD** (0.42 → 0.36 dB). Larger rooms have more receivers far from source → smoother diffuse field → easier to fit. This is mild and probably not actionable.
- **`complex_l1` shrinks ~half from L=3 to L=6** (0.058 → 0.033) — same trend, consistent with the LSD story.

## 5. Visual artifacts (12 PNGs + 1 cross-room)

All written to `outputs/single_room/L{L}/figures/` for L ∈ {3.0, 4.5, 6.0}:

- **`training_curves.png`**: 4 train losses (top, log y) + 3 val acoustic metrics (bottom, linear y). All three rooms show clean exponential decay across the full 10K iters with no plateau or instability. Val LSD drops from ~5 dB → 0.4 dB; val complex L1 from ~0.7 → 0.04; val phase L1 from ~1.0 → 0.07.
- **`modal_tracking.png`**: predicted spectrum 0–200 Hz at the centre receiver. Green markers (predicted picks) and blue triangles (ISM picks) overlap at every visible peak — the model and ISM agree on what counts as a peak in the modal regime. Orange ticks (deduplicated analytical eigenfrequencies) are denser than the picked peaks; that's the same modal-overlap effect documented in Chunk 1.5 SANITY_NOTES, not a model failure.
- **`spectrum_overlay.png`**: predicted (red) vs ISM (blue) vs analytical-modal-sum (green) `|H(f)|` 0–2 kHz, log y. Red almost perfectly overlays blue across the full band, including the deep nulls (the deepest at L=4.5 is ~−45 dB at ~640 Hz, matched within ~1 dB by the predicted spectrum).
- **`receiver_grid.png`**: 8×8 mini-plots, predicted (red) vs ISM (blue) `|H(f)|` per receiver. **All 64 receivers fit similarly well** — no dead spots or under-fit pockets. The model's reconstruction is spatially uniform across the room, not concentrated on the receivers it sampled most often.

Cross-room: `outputs/single_room/SUMMARY.md` (table + per-room artifact links) and `outputs/single_room/lsd_vs_L.png` (full-band & envelope LSD vs L; both decrease monotonically as L grows).

## 6. Capacity diagnosis

The architecture has plenty of capacity for single-room overfit. Three tells:

1. **Loss is still in clean exponential decay at iter 10K** — no plateau anywhere in the training curves. The model could keep improving with more iters. We chose to stop at 10K because the per-room overfit hits the noise floor on modal MAE (≤ 0.6 Hz, vs. Δf = 0.5 Hz bin spacing) — further training would chase numerical noise.
2. **Full-band LSD ≈ 0.4 dB across all rooms** — within 0.05 dB of the analytical-vs-analytical noise floor measured in Chunk 1.5. The architecture saturates at the data's intrinsic noise floor.
3. **Receiver-grid plot is spatially uniform** — every receiver fits, including ones far from the source. Confirms the HashGrid + MLPs encode the room's spatial structure rather than memorising a few favoured positions.

**Implication for Chunk 3** — this is the primary risk: the HashGrid (~44 M params per room, of which ~120 MB is hash entries × 6 encoders) is enormously over-parameterised for a 64-receiver target. With *one* model shared across multiple rooms in Chunk 3, the architecture has enough capacity to memorise every room's spectrum without ever using `z_s`. The auto-decoder may then learn `z_s` ≈ noise.

**Recommended Chunk-3 starting point**: shrink `log2_hashmap_size` from 18 → 14 (256K → 16K hash entries; 16× fewer params per encoder, ~3 M total) and `n_levels` from 20 → 14 before training. If single-room overfit MAE on the smaller grid stays ≤ 1 Hz on the same 3 rooms, we're safe to scale.

## 7. Time and compute

| Step | Wall (per job) | Total wall-clock |
|------|---------------:|-----------------:|
| Memory check | 7 s | 7 s |
| 3× single-room training (parallel) | 1 h 32–33 min | 1 h 33 min |
| 3× eval (parallel) | 41 s | 41 s |
| Summary script | <2 s | <2 s |

**Total compute**: 3 × 1.5 GPU-hours ≈ **4.6 GPU-hours**. Iter rate: ~1.8 iter/s/job (faster than the initial 1.0 iter/s estimate; the cancelled 50K-iter run shared the node with the 50K runs of the other two L values, which slowed each job; the 10K run had cleaner GPU residency).

**Preemption**: zero events (all 3 trainings landed on `legacygpu06` and ran to completion uninterrupted).

## 8. Recommendations for Chunk 3

1. **Downsize HashGrid before scaling**: per §6, start with `log2_hashmap_size=14, n_levels=14` (~16× param reduction). If single-room overfit MAE stays ≤ 1 Hz on the same 3 rooms, then enable the auto-decoder and add room dimensions.
2. **`z_s` injection point**: candidate A (concat at both branches) per `aaf/_inference_ref/inference_model.py` injection-point comments. The sigma-only ablation can be a quick second run.
3. **Latent dim**: keep `latent_dim=32` per the Chunk-1.5 default. With 7 training rooms in `dense.yaml` (dense sweep) the latent table is small.
4. **Loss weights**: keep current `(1.0, 1.0, 1.0, 0.1)`. Phase decays cleanly under this weighting; no instability observed.
5. **Validation cadence**: bump back to `val_every=1000` for multi-room runs (validation will be 7× longer with all rooms).
6. **Training budget**: target 30K-50K iters for multi-room. Multi-room conditioning is harder than single-room overfit; the Chunk-3 architecture needs more iters per room equivalent.
7. **Eval renderer-config inheritance**: now wired via `train_meta.json` — Chunk 3 can rely on this without changes.

## 9. Surprises and risks for Chunk 3

- **Single-room overfit is "too easy"**: the model converges to noise floor in 10K iters across all 3 L values without instability. This makes the Chunk-2 baseline strong, but it means Chunk 3 cannot rely on Chunk-2's loss curves to predict multi-room behaviour. The over-parameterisation flagged in §6 is the largest risk.
- **High-frequency reconstruction is excellent** (envelope LSD 0.10–0.14 dB across all 3 rooms). The model isn't peak-only — it captures the diffuse field statistics too. So Chunk-3 doesn't need band-specific loss weighting.
- **Modal recall against the *deduplicated* analytical mode list is ~12-17%** — this looks low but it's correct: 2D modal density at α=0.15 means ~100-130 distinct freqs in the modal regime, and the picker only finds ~15-21 (the well-isolated subset). The MAE on matched peaks is what matters, and that's at noise-floor parity.
- **Data-loading is not the bottleneck**: each room's 64 receivers fit in memory; the trainer loads everything once at startup (`SingleRoomTrainer.__init__`). For multi-room training in Chunk 3, this needs revisiting — 7 rooms × 64 receivers × 4097 freq bins × complex64 = ~7 MB per room, still fine for memory but needs a proper DataLoader for shuffling across rooms.

## 10. Manager actions requested

- **Decide Chunk-3 starting capacity**: I recommend shrinking the HashGrid (§6, §8 #1). If you want to retain INFER's defaults and test the auto-decoder against the over-parameterised model first, that's also defensible — just expect the latent-rank diagnostic to look suspicious.
- **Decide Chunk-3 training budget**: 30K iters is a reasonable starting point for multi-room conditioning. If we want to run longer, we'll need to switch to the `tron` partition (Q5) — flag if this is the time to commit.
- Q1 closed; Q5 still open but doesn't block Chunk 3 *if* we stay on scavenger.
