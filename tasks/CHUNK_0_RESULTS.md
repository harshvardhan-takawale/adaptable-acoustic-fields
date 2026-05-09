# Chunk 0 — Results

**Date**: 2026-05-09. **Scope**: recon and scaffolding only. No model implementation, no data generation, no training.

This document is the load-bearing artifact of Chunk 0. The manager (a separate Claude session) reads it to write Chunk 1. It captures the AVR / INFER recon, the chosen Phase-1 starting point, and an exhaustive list of 2D adaptation needs.

---

## 1. AVR architecture summary

Source: `/fs/nexus-projects/multimodal_recon/AVR` (NeurIPS'24 Spotlight, Penn Waves Lab). Compact codebase: ~10 Python files.

**Training loop (`avr_runner.py`)**: `AVR_Runner` class (L24-336). Adam, cosine-annealed LR (T_max ≈ 300k–500k iters; lr 2e-4 → 5e-4 → eta_min 5e-5–8e-5). Three DataLoaders (`train_iter`, `test_iter`, `train_iter_show`). Loss aggregation via `Criterion`: 6 weighted L1 terms (spec real/imag, amplitude, angle, time, energy, multi-resolution STFT). Gradient clipping max_norm=1, NaN/Inf masking. `.tar` checkpoints in `ckpts/`. TensorBoard scalars every 20 iters; validation every 10–20k iters renders 15 visualization PNGs and computes 7 acoustic metrics (Angle / Amp / Env / T60 / EDT / C50 / multi-STFT).

**Dataset (`datasets_loader.py:WaveLoader`)**: three flavors: MeshRIR, Simu (synthetic), RAF (real-world). Loads complex64 RFFT of IRs into memory at `__init__` (line 51-55). 3D position vectors throughout. RAF training-only adds N(0, 0.1) position jitter (L175-176). MeshRIR permutes coords `[0,2,1]` (L147, L157). Yield: `(wave_signal, position_rx, position_tx[, rotation_tx])`.

**Model (`model.py`)**: two variants. `AVRModel` (Simu/MeshRIR): tiny-cuda-nn HashGrid encoders for position / direction / tx-position; FullyFusedMLP sigma encoder→decoder (1 scalar attn output) and CutlassMLP signal network (`signal_output_dim` = 1600 or 2400). `AVRModel_complex` (RAF): adds tx-direction encoder. Network outputs are frequency-domain RFFT coefficients (via real/imag split, complex implicit).

**Renderer (`renderer.py:AVRRender`)**: frequency-domain volume rendering. Pipeline:

1. Spherical ray grid (`ray_directions(n_azi, n_ele)`, L107-139) → `[n_azi*n_ele + 2, 3]`. Two extra rays at zenith/nadir.
2. Sample N points along each ray uniformly in `[near, far]` (L46).
3. Normalize points to `[-1, 1]` (L50), forward through model → `attn` (scalar), `signal` (RFFT real-valued).
4. Causality masks: zero IR samples before `pts2rx_idx = fs * d_vals / speed` (L65-69) and after `tx2pts_idx` (L72-76).
5. 1/r path-loss attenuation (L79-83) sliced per ray sample.
6. RFFT → phase shift `exp(-j 2π f d/c)` per ray distance (L86-87).
7. `acoustic_render()` (L141-167): weighted volume integral `Σ α_i T_i signal_i` per ray, then sum over rays.
8. Split complex result into `[Re, Im]` → return `[bs, signal_len, 2]`.

**Utils**: `Criterion` (criterion.py L5-51) wraps the 6 loss terms with auraloss MultiResolutionSTFTLoss [512, 256, 128, 64]. `metric.py` has CPU and torch versions of the 7 metrics. `logger.py` does TensorBoard + 6-panel viz figures. `spatialization.py` has a `wide_cardioid_beam_pattern` function — defined but unused.

**Configs (`config_files/*.yml`)**: hierarchical `path / render / train / model` YAML, hand-parsed. Four variants: `avr_simu.yml`, `avr_meshrir.yml`, `avr_raf_empty.yml`, `avr_raf_furnished.yml`. Every encoder is a tcnn.HashGrid (`base_resolution: 16`, `log2_hashmap_size: 18`, `n_features_per_level: 2`, `n_levels: 20`).

**No tests, no CI.** De-facto verification is visual inspection of `img_train/` and `img_test/` PNGs.

## 2. What we'll reuse from AVR

- **Loss design** (`utils/criterion.py`): the 6-term weighted L1 sum is a fine starting basis. Drop `time_loss` early (we are frequency-native already); keep complex spec + amplitude + angle + energy. Reuse the multi-resolution STFT loss from auraloss as a perceptual fallback when we want it.
- **Metric battery** (`utils/metric.py`): Angle / Amp / Env / T60 / EDT / C50 / multi-STFT — directly reusable. Add **modal-frequency MAE** as a Phase-1-specific eigenfrequency probe (validates the latent manifold against analytical 2D modes `f_{m,n} = (c/2)·sqrt((m/L)² + (n/W)²)`).
- **Volume-rendering equation** for accumulated transmittance — the mathematical core (`acoustic_render`, INFER's σ + jβ formulation) is preserved verbatim in our 2D renderer.
- **TensorBoard scalar/figure cadence** (`avr_runner.py:196-244`) — clean enough to mirror.
- **tinycudann hash-grid encoding** as the spatial encoder — fast, well-tested, drops in with `tcnn.Encoding(2, ...)`.

## 3. What we'll skip or replace

- **Hand-rolled YAML loader + argparse** → Hydra. AVR's parsing is brittle and won't compose to multi-room / multi-ablation needs.
- **Stochastic 3D spherical ray sampling** in 2D → circular azimuth-only sampling. Pyroomacoustics gives ground-truth image-source paths analytically — open research call (Q1) on whether stochastic rays add value over deterministic ISM-aligned sampling.
- **Hard-coded `tcnn.Encoding(3, ...)`** (six call-sites in INFER's main model, lines 769-774) → 2D encodings.
- **Single-file `avr_runner.py`** → split into `aaf/train/loop.py`, `aaf/train/checkpoint.py`, `aaf/eval/run.py`. Each <200 lines.
- **No test infra** → pytest with smoke + per-module unit tests added incrementally.
- **MeshRIR `[0, 2, 1]` coord permutation** is irrelevant — we generate our own data via pyroomacoustics.

## 4. INFER frequency-domain renderer

Source: `project_files/unified_renderers.py` (~1500 lines, many variants/ablations).

**Inventory** (one-line per class with line range):

| Class | Lines | Note |
|-------|------:|------|
| `AVRRender` | 85-207 | Original time-domain AVR (3D) |
| `AVRRenderFD` | 208-357 | Standard freq-domain renderer; complex attenuation |
| `AVRRenderFD_AbsAtt` | 358-400 | Adds atmospheric absorption + amp attenuation |
| `AVRRenderFD_FreqDep` | 401-445 | Per-frequency-bin attenuation |
| `AVRRenderFD_PhaseCorrection` | 446-504 | Single-attn phase correction |
| `AVRRenderFD_FreqDep_PhaseCorrection` | 506-560 | Earlier freq-dep + phase correction |
| `AVRRenderFD_FreqDep_PhaseCorrection_dist` | 561-634 | + distance-invariant sampling |
| `AVRRenderFD_PhaseCorrection_new` | 635-714 | Refined σ + jβ decomposition (single-attn) |
| **`AVRRenderFD_FreqDep_PhaseCorrection_new`** | **716-790** | **Refined freq-dep + phase correction (chosen baseline)** |
| `AVRRenderFD_FreqDep_PhaseCorrection_KK` | 829-1035 | + Kramers-Kronig consistency loss; "INFER" Table 1 row |
| `AVRRenderFD_FreqDepAbsAtt` | 1037-1077 | + absorption |
| `AVRRenderFD_NoAbsorp` | 1080-1125 | No absorption ablation |
| `AVRRenderFD_PhaseCorrection_DistInvar*` | 1126-1287 | Distance-invariant variants |
| `NAFRenderer`, `NAFFrequencyRenderer` | 1288-1421 | NAF baseline (Luo et al. 2022) |
| `INRASRenderer`, `INRASFrequencyRenderer` | 1422-1536 | INRAS baseline (Su et al. 2022) |

**Main renderer for INFER's headline result (with KK)**: `AVRRenderFD_FreqDep_PhaseCorrection_KK`.
**Main renderer for INFER (w/o KK), our Phase-1 starting point**: **`AVRRenderFD_FreqDep_PhaseCorrection_new` (lines 716-790)**.

**Forward pass — `acoustic_render_fd` of the chosen `_new` variant** (`unified_renderers.py:723-790`):

1. `delta_u = d_vals[i+1] - d_vals[i]`, padded with 1e10 at the tail.
2. Decompose model's complex `attn` per frequency bin: `σ = attn.real.clamp(min=0)` (absorption, ≥ 0) and `β = attn.imag` (phase velocity change, signed).
3. Local opacity `α = 1 - exp(-σ · Δu)`.
4. Cumulative amplitude transmittance `T_amp = ∏(1 - α_i)`.
5. Cumulative material phase `P = exp(j · cumsum(β · Δu))`.
6. Combined `T = T_amp · P`.
7. Frequency axis `freq_hz = arange(n_bins) · fs / n_time_samples`. Geometric phase shift `exp(-j · 2π · f · d/c)` per sample point.
8. Optional 1/r geometric amplitude attenuation (off by default).
9. `rendered = sum(signal_fd · geometric_phase · α · T)` over the sample axis → `[bs, n_rays, n_freq_bins]`.
10. Outer code: spherical sum over rays → optional time-domain causality mask → split into `[Re, Im]` → return.

**vs. AVR's renderer**: INFER (a) uses **frequency-dependent complex attenuation σ(f) + jβ(f)** rather than a scalar; (b) does the integration entirely in the frequency domain (multiplicative phase, additive over rays); (c) supervises complex H(f) directly.

**3D-specific bits**:
- `ray_directions(n_azi, n_ele)` (`unified_renderers.py:50-70`) generates spherical samples — needs a 2D circular variant.
- `ray_aabb_intersect(rays_o, dirs, room_min, room_max)` (`unified_renderers.py:793`) is 3D AABB — needs 2D-rectangle slab tests.

## 5. INFER models

Source: `project_files/unified_models.py` (~1547 lines).

**Inventory**:

| Class | Lines | Note |
|-------|------:|------|
| `AVRModel` | 42-103 | Standard 3D time-domain AVR |
| `AVRModel_complex` | 106-187 | + tx direction |
| `AVRModel_FD` | 194-288 | Freq-domain, complex attn |
| `AVRModel_complex_FD` | 291-403 | Complex FD + tx direction |
| `AVRModel_FD_FreqDep` | 410-505 | Per-frequency-bin attn |
| `AVRModel_complex_FD_FreqDep` | 507-619 | Complex + per-freq attn |
| `AVRModel_complex_FD_PhaseCorrection` | 621-749 | Single-bin attn + phase correction |
| **`AVRModel_complex_FD_FreqDep_PhaseCorrection`** | **752-883** | **Per-freq complex attn (σ + jβ) — chosen INFER main model** |
| `AVRModel_FD_RealAttn`, `AVRModel_complex_FD_RealAttn` | 890-1077 | Real-attenuation ablations |
| `NAFModel`, `NAFModelFrequencyDomain` | 1078-1271 | NAF baselines |
| `INRASModel`, `INRASFrequencyModel` | 1278-1547 | **INRAS baselines (Su et al. 2022)** |

**⚠ Note on a recon error to correct**: an earlier exploration sub-agent identified `INRASFrequencyModel` (L1480-1547) as INFER's main model based on the presence of separate magnitude/phase heads and the name "FrequencyModel". This is **wrong**. `INRAS*` are reimplementations of the **INRAS baseline** (Su et al. 2022) cited in INFER's Related Work. The actual INFER main model is `AVRModel_complex_FD_FreqDep_PhaseCorrection` (`unified_models.py:752-883`); its forward signature `(pts, view, tx, tx_view) → (attn_complex, signal_complex)` matches what `AVRRenderFD_FreqDep_PhaseCorrection_new` expects, while `INRASFrequencyModel.forward(source_pos, receiver_pos, frequencies)` does not. Future agents should not repeat this conflation.

**Architecture of `AVRModel_complex_FD_FreqDep_PhaseCorrection`** (forward signature `forward(pts, view, tx, tx_view) → (attn_complex, signal_complex)`):

- Six tcnn encoders (all 3D positional inputs):
  - `_pos_encoding`, `_pos_signal_encoding` (rx position, sigma + signal branches)
  - `_tx_pos_encoding`, `_tx_pos_signal_encoding` (tx position, sigma + signal branches)
  - `_dir_encoding`, `_tx_dir_encoding` (rx and tx direction)
- Sigma branch: concat `[pos_emb, tx_pos_emb]` → `_model_encoder_sigma` (256-out) → `_model_decoder_sigma` (`signal_output_dim` out, complex split into real/imag with softplus on real ≥ 0). Returns `attn_complex` of shape `[bs, n_pts, n_freq_bins]`.
- Signal branch: concat `[sigma_features, dir_emb, tx_dir_emb, pos_signal_emb, tx_pos_signal_emb]` → `_model_signal` (`signal_output_dim` out, complex split). Masks DC and Nyquist imaginary components (RFFT symmetry, L869-876). Returns `signal_complex` of shape `[bs, n_pts, n_freq_bins]`.

Both outputs are per-point complex tensors over the full one-sided RFFT grid.

**3D assumptions** (every `tcnn.Encoding(3, ...)` at L769-774): six positional encoders all assume 3-component inputs. For 2D, drop `Encoding(3, ...)` → `Encoding(2, ...)` for rx/tx position; for direction encode as `(cos θ, sin θ)` → `Encoding(2, ...)`.

## 6. 2D adaptation needs (exhaustive)

| Concern | Where | Change |
|---------|-------|--------|
| Position encoding dim | 4 × `tcnn.Encoding(3, ...)` for rx/tx position (model L769-772) | `(2, ...)` |
| Direction encoding dim | 2 × `tcnn.Encoding(3, ...)` for rx/tx direction (model L773-774) | `(2, ...)` over `(cos θ, sin θ)` |
| Ray direction sampler | `ray_directions(n_azi, n_ele)` (renderer L50-70) | drop elevation; `n_azi` uniformly on `[0, 2π)`; `(cos θ, sin θ)` |
| AABB intersection | `ray_aabb_intersect` (renderer L793) | 2D rectangle slab tests (4 walls, not 6) |
| Geometric path-loss | renderer L777-781 (1/r) | 2D cylindrical 1/√r (see Q8 — research call) |
| Eigenfrequencies (new code) | n/a in INFER | `f_{m,n} = (c/2)·sqrt((m/L)² + (n/W)²)` in `aaf/eval/eigen_modes.py` |
| Pyroomacoustics simulation | `pra.ShoeBox(p=[L, W])` for 2D | sanity-check; library is 3D-first (Q7) |
| Position normalization | `(pts + 1) / 2` of 3D pts (model L821-823) | trivially generalizes; just `xyz_min/xyz_max` becomes 2D |
| `signal_output_dim` | full RFFT length, even or odd | derived from `n_time_samples` (Q2) |
| Tx direction (RAF only) | always passed in INFER's `tx_view` arg | optional in 2D — point sources don't have direction unless modeling speakers |
| Position vector storage | 3-component throughout dataset | 2-component; or keep 3 with z=0 if interop with INFER ref code |
| Causality time mask | applies to full RFFT then re-FFTs | keep as-is, just with our 2D fs |

## 7. Auto-decoder injection candidates (for `AVRModel_complex_FD_FreqDep_PhaseCorrection`)

Three candidates. All defer to Chunk 1 to choose, per user direction. No leading recommendation.

**Candidate A — concat at both branches**:
- Sigma branch concat at `unified_models.py:831`: `torch.cat([pos_embedding, tx_pos_embedding, z_s_broadcast], -1)`. Widen `network_in_dims` (L776) by `latent_dim`.
- Signal branch concat at `unified_models.py:854`: append `z_s_broadcast` to the existing concat. Widen `n_signal_input` (L790) by `latent_dim`.
- Most DeepSDF-faithful — every branch becomes room-aware.
- Code change: 4 lines in `__init__`, 2 lines in `forward`.

**Candidate B — concat at sigma branch only**:
- Inject `z_s` at L831 only.
- Smaller surgery; signal pathway stays room-agnostic.
- Risk: room modes are largely a function of geometry encoded in σ/β, but the directional signal `S(f, n̂)` from INFER paper Eq. 4 is also room-dependent. Likely under-conditioned for our use case; flag if Candidate A's per-branch ablation shows the signal branch dominates.

**Candidate C — FiLM modulation**:
- Small MLP `f_φ(z_s) → (γ_l, β_l)` produces per-layer scale/bias for each sigma-encoder MLP layer (and optionally the signal MLP). Apply via `h_l = (1 + γ_l) ⊙ tcnn_layer(h_{l-1}) + β_l`.
- More expressive — `z_s` modulates intermediate representations rather than just being concatenated at the input.
- Requires writing a wrapper around the tcnn `Network` (FiLM doesn't drop in cleanly to `FullyFusedMLP`'s fused kernel). May be easier with a `nn.Sequential` of `nn.Linear` layers — losing tcnn's speed. Defer until A/B are shown insufficient.

## 8. Risks / surprises

- **tinycudann ABI**: `tinycudann==1.7` was built against `torch 2.0.1+cu118` on a Nexus compute node. Cloning `avr_scavenger` preserves the wheel. Rebuilding requires a compute node (CUDA toolkit not installed on login). Don't rebuild during scaffold; keep Python 3.8.
- **Login-node `libstdc++` ImportError**: `/lib64/libstdc++.so.6` lacks `GLIBCXX_3.4.29` required by scipy native ext. Manifests as scipy import failure — and pyroomacoustics imports scipy. So `pytest` cannot run on the login node. CLAUDE.md and CLUSTER_INFO.md document this; `tests/test_env.py` is run via SLURM.
- **Pyroomacoustics 2D**: pyroomacoustics is 3D-first. `pra.ShoeBox(p=[L, W])` *should* produce 2D rooms, but the API may auto-extrude or silently fall back to 3D. **Sanity-check** in Chunk 1 before generating the dataset sweep — three checks: (a) IR is causal, (b) early image-source delays match analytical 2D ISM, (c) frequency response shows the expected 2D modal spacing.
- **RFFT length parity**: even/odd `n_time_samples` matters for Nyquist masking (`unified_models.py:873-876`). Pick `n_time_samples` divisible by 2 (or maintain the `if n % 2 == 0` branch).
- **Recon error to track**: as noted in §5, `INRASFrequencyModel` is a baseline, not the INFER main model. CHUNK_0_RESULTS.md, CONTEXT_FOR_MANAGER.md, and DECISIONS.md all flag this so future agents do not repeat the conflation.
- **2D Green's function** is not 1/r — see Q8 in OPEN_QUESTIONS.md.

## 9. Verification (this chunk)

- `aaf` env exists at `/fs/nexus-scratch/htakawal/miniconda3/envs/aaf` (clone of `avr_scavenger` + `hydra-core` + `gh`). Frozen in `environment.yml`.
- `scripts/slurm/hello.sh` submitted as **job 6797442**, ran on `legacygpu06.umiacs.umd.edu` (GeForce GTX TITAN X). Output: `logs/slurm/aaf_hello-6797442.out`. Confirms env activates, nvidia-smi works, torch 2.0.1+cu118 imports, `torch.cuda.is_available()` is True.
- `pytest -q` (jobs 6797463 → 6797480, scavenger). First run failed with `ImportError: GLIBCXX_3.4.29 not found` on compute node, traced to `/lib64/libstdc++.so.6` shadowing the env's `lib/libstdc++.so.6.0.33`. Fixed by adding `export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"` after `conda activate aaf`. Re-submitted as **job 6797480**, all 13 tests pass (1 smoke + 10 import + 2 CUDA). Documented in DECISIONS.md and CLUSTER_INFO.md; sbatch templates updated.
- Repo skeleton, gitignore, pyproject.toml, environment.yml, all 6 root .md docs in place.

## 10. What Chunk 1 should produce

(Manager's call — this is just a guess at the shape of the next deliverable, to anchor the questions in OPEN_QUESTIONS.md.)

Likely Chunk 1:
- `aaf/data/shoebox.py` — pyroomacoustics 2D ISM wrapper. Generates one IR per (room L, rx, tx) tuple. Stores complex RFFT to HDF5.
- `aaf/data/dataset.py` — `WaveLoader`-equivalent: loads multiple rooms keyed by L, yields `(H_complex, rx_2d, tx_2d, room_id)`. The room_id is the auto-decoder index.
- `aaf/eval/eigen_modes.py` — analytical 2D mode formula + a function to detect peaks in a predicted H(f) and compute MAE against truth.
- `configs/data/shoebox_Lsweep.yaml` — Hydra config for the L sweep.
- A sanity-check SLURM script that simulates one room, plots its IR and frequency response, and asserts at least two modes are within Δf of the analytical eigenfrequencies.

The model + renderer port is likely **Chunk 2**, separately, so that Chunk 1's data + sanity probes can be tested in isolation.
