# DECISIONS.md

Append-only log of non-trivial design choices. Newest entries at the bottom. Format:

```
## YYYY-MM-DD: <short title>
**Decision:** <what>
**Rationale:** <why>
**Alternatives considered:** <what we rejected and why>
**Revisit if:** <conditions to reopen>
```

---

## 2026-05-09: Repo layout — package directory `aaf/` + Hydra `configs/`, not AVR's flat layout

**Decision:** organize source as an importable Python package (`aaf/`) with subpackages (`data`, `models`, `renderers`, `train`, `eval`, `utils`), separate `configs/` tree, `tests/` directory, `scripts/slurm/` for job templates, `tasks/` for chunk results. Docs (`CLAUDE.md`, `CONTEXT_FOR_MANAGER.md`, `DECISIONS.md`, `OPEN_QUESTIONS.md`, `CLUSTER_INFO.md`) live at root.

**Rationale:** AVR is a single-paper codebase with `model.py / renderer.py / datasets_loader.py / avr_runner.py` at root and a hand-rolled YAML loader. That works for one config; it breaks down for our project, which needs (a) multiple data variants (different L sweeps, different W sweeps), (b) auto-decoder vs vanilla model heads, (c) 2D vs 3D ablations, (d) eigenfrequency probes vs renderer eval. A package layout gives clean test imports and lets configs reference `aaf.models.inr_2d` paths.

**Alternatives considered:** mirror AVR exactly (rejected — won't scale past Phase 1); use src-layout (`src/aaf/`, rejected as unnecessary indirection at this size).

**Revisit if:** the package becomes >50 modules, at which point introducing a `src/` layer or splitting subpackages is reasonable.

---

## 2026-05-09: Conda env — clone `avr_scavenger` to `aaf`, add only `hydra-core` and `gh`

**Decision:** `conda create --name aaf --clone avr_scavenger`. Clone preserves the exact tinycudann + torch + CUDA build. Added `hydra-core` (config compositionality) and `gh` (GitHub CLI). Frozen via `conda env export --no-builds > environment.yml`. Python 3.8.20 retained.

**Rationale:** `avr_scavenger` was hand-built on a Nexus compute node with the right GCC/CUDA toolchain to compile `tinycudann==1.7` against `torch 2.0.1+cu118`. Rebuilding from scratch risks ABI mismatches and burns hours on a compute node. Phase-1 deps were almost entirely already present (pyroomacoustics, h5py, pytest, matplotlib, scipy, librosa, auraloss, tinycudann); only `hydra-core` was new. Cloning + minimal additions matches the "clone exactly" instruction in the task brief.

**Alternatives considered:** fresh Python 3.10+ env (rejected — requires rebuilding tinycudann; defeats the clone instruction; no concrete benefit for Phase 1); pip-only env (rejected — `gh` from conda-forge is the official binary).

**Revisit if:** Python 3.8 EOL (October 2024 already past; we're already past it but the env is locked) becomes a security concern for any production deployment, or if a Phase-2+ dependency requires Python 3.10+.

---

## 2026-05-09: Code style — `ruff` + `black`, line length 100, target Py 3.8

**Decision:** `pyproject.toml` configures both. Lint rules `E,F,I,W,UP,B`. AVR has no enforced style; we adopt one to keep the codebase tractable as it grows.

**Rationale:** picking *some* style up front is cheaper than retrofitting later. `ruff` is fast enough to run pre-commit; `black` settles formatting debates.

**Alternatives considered:** `ruff format` only (rejected — black is more familiar to most contributors); flake8/isort (rejected — superseded by ruff).

**Revisit if:** the team adopts a different convention or if Python 3.10+ becomes the env target.

---

## 2026-05-09: Configs — Hydra over hand-rolled YAML

**Decision:** use [Hydra](https://hydra.cc) for run configuration. AVR's hand-rolled YAML+argparse pattern is fine for 4 configs but breaks down for compositional needs (multiple data variants × multiple model heads × ablations).

**Rationale:** Phase 2's per-room latent codes plus multi-room sweeps imply config compositions like `data=shoebox-Lsweep model=inr_2d_autodecoder train=fast`. Hydra makes that mechanical.

**Alternatives considered:** OmegaConf alone (rejected — no run-tree management); plain dataclasses + argparse (rejected — composition is verbose); Lightning Fabric / PyTorch Lightning (rejected — too much framework for a research codebase at this size).

**Revisit if:** Hydra's overrides become painful to read/debug, or if we adopt a framework that subsumes it.

---

## 2026-05-09: Phase-1 model/renderer baseline — INFER's `AVRModel_complex_FD_FreqDep_PhaseCorrection` + `AVRRenderFD_FreqDep_PhaseCorrection_new` (non-KK)

**Decision:** start the 2D adaptation from INFER's main model (`unified_models.py:752-883`) paired with the non-KK renderer variant (`unified_renderers.py:716-790`). Drop Kramers-Kronig regularization and perceptual weighting from the Phase-1 critic.

**Rationale:** the model carries forward the per-frequency-bin σ + jβ complex attenuation form, which is the load-bearing physics piece for room-acoustic INRs. The non-KK variant matches the INFER paper's "INFER (w/o KK)" reported result and avoids the extra regularizer and the bandwidth-extension machinery — neither is needed when we have full 0–2 kHz training data and synthetic ground truth from pyroomacoustics. Perceptual weighting (A-weighting) is a deployment-time concern, not a Phase-1-science concern.

**Alternatives considered:** start from AVR's simpler `AVRModel` + scalar attenuation (rejected — no per-frequency dispersion modeling, would limit the modal-frequency probe); use INFER's KK variant (rejected — adds a regularizer that's only useful when data is bandlimited, which ours isn't); use INRASFrequencyModel (rejected — that's a baseline reimplementation of Su et al. 2022, not the INFER main model; an earlier explorer subagent confused this).

**Revisit if:** reconstructions show non-causal phase artifacts (suggesting KK may help), or if perceptual evaluation in Phase 4+ surfaces audible magnitude errors that A-weighted loss would suppress, or if the per-frequency complex attenuation turns out to be over-parameterized for the cleaner 2D shoebox setting.

---

## 2026-05-09: Test discipline — pytest with smoke + env imports only for Chunk 0

**Decision:** ship `tests/test_smoke.py` (trivial assert) and `tests/test_env.py` (parametrized imports + CUDA available). Defer per-module unit tests to chunks that introduce code.

**Rationale:** there is no code yet to test. A premature fixture / mocking infrastructure is debt; smoke + env checks are enough to confirm the cluster + env work. Tests cannot run on the login node (libstdc++ libstdc++.so.6 in /lib64 lacks GLIBCXX_3.4.29 required by scipy native extension); CLAUDE.md and CLUSTER_INFO.md document this.

**Alternatives considered:** import all of `aaf.*` as smoke-import test (rejected — every package in `aaf/` is empty; nothing to import meaningfully).

**Revisit if:** Chunk 1+ adds non-trivial source code, at which point unit tests for that module become required.

---

## 2026-05-09: libstdc++ shim — prepend `${CONDA_PREFIX}/lib` to `LD_LIBRARY_PATH` in every job

**Decision:** every SLURM job (and every interactive `srun` session) that imports `scipy` / `pyroomacoustics` / `matplotlib` must run `export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"` after `conda activate aaf`. Templates in `scripts/slurm/` enforce this; CLAUDE.md and CLUSTER_INFO.md document it.

**Rationale:** `/lib64/libstdc++.so.6` on Nexus caps at `GLIBCXX_3.4.25`. The `aaf` env's scipy `_uarray.cpython-38-x86_64-linux-gnu.so` requires `GLIBCXX_3.4.29`, which is present in `${CONDA_PREFIX}/lib/libstdc++.so.6.0.33`. Without the shim, scipy import (and therefore pyroomacoustics) fails on both login and compute nodes. The first `pytest` SLURM submission (job 6797463) confirmed this fails on the compute node too — not just login.

**Alternatives considered:** patch the env to remove scipy's libstdc++ dependency (rejected — fragile, future packages will hit the same wall); ask UMIACS to upgrade `/lib64/libstdc++.so.6` cluster-wide (rejected — out of scope, cluster-OS upgrade is scheduled for Summer 2026); use `LD_PRELOAD` instead of `LD_LIBRARY_PATH` (rejected — same effect, less standard).

**Revisit if:** Nexus completes the Summer 2026 cluster OS upgrade ([Nexus/ClusterOSUpgrade](https://wiki.umiacs.umd.edu/umiacs/index.php/Nexus/ClusterOSUpgrade)) and `/lib64/libstdc++.so.6` ships with `GLIBCXX_3.4.29` or newer.

---

## 2026-05-09: Default cluster partition — `scavenger`

**Decision:** all chunked work runs on `--partition=scavenger --account=scavenger --qos=scavenger` by default. `scripts/slurm/hello.sh` and the sbatch template in `CLUSTER_INFO.md` reflect this.

**Rationale:** scavenger has unlimited per-job CPU/GPU/RAM caps within a 3-day wall, and is plentiful. We absorb preemption with periodic checkpointing once training begins. For Phase 1 development (under-1-day jobs), preemption is rarely a problem.

**Alternatives considered:** `tron` with `qos=default` (rejected — 1 GPU / 32 GB / 3 d cap is fine but tron is in higher contention; switch to it for non-preemptible final eval runs).

**Revisit if:** preemption rate exceeds ~20% during training, or if a Chunk requires non-preemptible long runs (publish those as separate `tron` jobs and document here).

---

## 2026-05-09: Q2 resolved — Phase-1 frequency grid is fs=4096, n_time_samples=2048, Δf=2 Hz

**Decision:** Track A (Phase-1 science) uses `fs=4096 Hz`, `n_time_samples=2048`, `n_freq_bins=1025`, `Δf=2 Hz`. Track B (high-fidelity) at `fs=48000`, `n_time_samples=16384` is reserved for later and is a config flip — the simulator and dataset writer carry no hard-coded sampling-rate assumptions.

**Rationale:** 0–2 kHz coverage with margin (Nyquist 2048 Hz). Δf=2 Hz gives ~half-bin resolution on the lowest analytical modes (e.g., (1,0) at L=4 m → 42.875 Hz, ~21 bins). Smaller n_time_samples = smaller model output dim later; bigger n_time_samples buys finer Δf at the cost of wider per-sample networks. Picked the smallest that resolves modes adequately.

**Alternatives considered:** fs=4096 / n_time=4096 (Δf=1 Hz, 2x bigger output); fs=2048 / n_time=1024 (Δf=2 Hz but truncates above 1 kHz; rejected — 0–2 kHz spec).

**Revisit if:** Chunk-2 modal-frequency MAE plateaus above ~Δf — that's a sign the model is bin-limited. Bump n_time_samples then.

---

## 2026-05-09: Q3 resolved — L sweep is 15 unique values across three sweep configs

**Decision:** Generate 15 rooms at L ∈ {2.5, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.5} m, all with W=4.0, α=0.15. Source fixed at (0.5, 0.5) m; 8×8 receiver grid on `[0.3, L-0.3] × [0.3, W-0.3]`. Three sweep configs (`configs/sweeps/{dense, sparse, extrapolation}.yaml`) slice subsets for train/test.

**Rationale:** dense (0.5 m spacing) tests interpolation between nearby training rooms; sparse (1.5 m gap) tests extrapolation across wide gaps; extrapolation (training on middle, testing at edges) tests true out-of-distribution. 15 unique rooms cover all three at single-room cost (40 s total to generate).

**Alternatives considered:** more rooms (rejected — 15 is enough for a 1-D family); also varying W and α (deferred to Phase 2 per roadmap); irregular L grid (rejected — interpolation/extrapolation analyses get clearer with regular grid).

**Revisit if:** the latent manifold doesn't smoothly interpolate between dense rooms — would suggest more rooms are needed.

---

## 2026-05-09: Q4 resolved — Auto-decoder latent dim default `d=32`

**Decision:** for the Chunk-3 auto-decoder, default `latent_dim = 32` per room.

**Rationale:** for a Phase-1 single-axis (L only) family, 4–8 dims would suffice. d=32 leaves headroom for Phase 2 (multi-axis: L, W, H, doorway) without redesigning the latent table; also matches DeepSDF's "small but compositional" defaults.

**Alternatives considered:** d=8 (minimal; rejected — too tight for Phase 2 expansion); d=128 (rejected — overkill, harder to regularize).

**Revisit if:** Phase 1 latent space is rank-deficient (most dims unused), or Phase 2 needs >32 axes of variation.

---

## 2026-05-09: Q6 resolved — Vendor only the chosen INFER classes into `aaf/_inference_ref/`

**Decision:** Copy `AVRModel_complex_FD_FreqDep_PhaseCorrection` and `AVRRenderFD_FreqDep_PhaseCorrection_new` (plus the renderer's parent class and helpers) into `aaf/_inference_ref/inference_model.py` and `aaf/_inference_ref/inference_renderer.py` with provenance headers and "do not import in production" warnings. The full INFER source stays in `project_files/` (gitignored). Auto-decoder injection-point comments are placed at the candidate concat lines for future reference.

**Rationale:** keeping the relevant source in-repo means side-by-side diff is one click away during the 2D port (Chunks 2-3); copying the *whole* unified files would 4× the repo size and reproduce many ablations we never use. The vendored files are not import-safe (uses `.cuda()` at module init) but they parse cleanly under `ast.parse` — that's enough for diffing.

**Alternatives considered:** vendor everything (rejected — repo bloat); vendor nothing, only refer to file paths (rejected — manager Claude can't read `project_files/` since it's not on GitHub).

**Revisit if:** we need a different INFER variant than the chosen baseline.

---

## 2026-05-09: Q8 resolved — Geometric amplitude attenuation OFF in renderer

**Decision:** when the Chunk-2 renderer is implemented, `geometric_attenuation` defaults to `False`. The network learns whatever spreading the data exhibits; we don't impose 1/r vs 1/√r vs Hankel.

**Rationale:** in 2D the correct free-field decay is 1/√r (cylindrical), not 1/r (spherical). Pyroomacoustics' ISM emits whatever 2D ISM gives — likely a 1/r-with-image-sources approximation, but we don't need to characterize it because the network will absorb the geometric factor as part of its mapping. Pre-multiplying by 1/r (the INFER default for 3D) would inject a wrong physical prior; pre-multiplying by 1/√r is also a guess. The cleanest scientific stance is to leave it off.

**Alternatives considered:** 1/r (rejected — 3D-correct, 2D-wrong); 1/√r (rejected — would assume free-field, but our rooms are bounded); Hankel function (rejected — numerically annoying near r=0, marginal accuracy gain).

**Revisit if:** Chunk-2 reconstructions show a systematic amplitude bias growing with distance (suggesting the network can't learn the geometry on its own) — then re-enable `geometric_attenuation` with the appropriate 2D form.

---

## 2026-05-09: HDF5 complex storage — native `complex64` (deviation from spec wording)

**Decision:** dataset HDF5 files store `H_complex` as native `complex64` datasets, not as a pair of real arrays. Dataset name `/ism/H_complex` per the spec is preserved; only the dtype interpretation differs.

**Rationale:** h5py 3.11 (in our `aaf` env) supports complex dtypes natively, round-tripping `np.complex64` through HDF5 transparently (verified end-to-end via `tests/test_dataset_io.py::test_complex_dtype_preserved`). The spec's parenthetical "complex stored as two real arrays per the agent's earlier convention" is a hangover from older h5py / interop concerns that don't apply here. Native complex is half the dataset count, half the bookkeeping, no information lost.

**Alternatives considered:** split `_real` and `_imag` datasets (rejected — adds complexity for no benefit); h5py compound dtype manually (rejected — h5py already does this internally for complex).

**Revisit if:** a downstream tool (PyTorch DataLoader collate, JAX, etc.) chokes on complex64 HDF5 datasets — at which point we add real/imag views in the loader, not the writer.

---

## 2026-05-09: ISM `max_order` rule — `ceil(c · 4 · T60 / min(L,W))`

**Decision:** when `max_order` is not provided to `simulate_room_2d`, compute it as `ceil(c · 4 · T60_sabine_2d / min(L, W))`, giving image-source coverage for ~4× T60 of decay.

**Rationale:** matches pyroomacoustics' own internal heuristic (`acoustics.py:576`). Across our L sweep (2.5–6.5 m) at α=0.15 this yields max_order in [80, 120], which is more than enough to capture the 0.5 s IR length we truncate to. Per-room time on CPU is ~2–3 s — total dataset gen is 40 s.

**Alternatives considered:** fixed max_order=50 (under-shoot for 6.5 m rooms); max_order=200 (overkill, ~3× more compute, no recovery benefit).

**Revisit if:** Track B (fs=48000, longer IRs) takes too long — switch to per-receiver shrinking max_order.

---

## 2026-05-09: Modal damping in analytical model — uniform Sabine 2D `γ = c·α·P/(4·A)`

**Decision:** the analytical modal model in `aaf/sim/analytical_modal_2d.py` uses mode-independent damping `γ = c · α · P / (4 · A)` with `P = 2(L+W)` and `A = LW`.

**Rationale:** the Sabine 2D damping is the natural diffuse-field limit — every mode dissipates at the same energy rate determined by wall absorption / area ratio. Mode-by-mode wall-projection damping would be more accurate but adds modelling assumptions (which walls each mode "sees", impedance matching) that aren't justified for a diagnostic reference. The noise-floor report (`outputs/noise_floor/REPORT.md`) shows analytical-vs-analytical recall and MAE bound the picker's noise floor under this damping choice.

**Alternatives considered:** per-mode wall-projected damping (rejected — over-specified for a reference); zero damping (rejected — gives infinite peaks at eigenfrequencies, breaks the picker).

**Revisit if:** the analytical reference's peak shapes differ noticeably from ISM beyond what Sabine over-uniformity explains (would suggest modal-specific damping matters).

---

## 2026-05-10: Q9 resolved — Deduplicate analytical eigenfrequencies before matching

**Decision:** `eigenfrequencies_2d(L, W, c, f_max, dedup_tol_hz=0.01)` returns a list of `EigenFreq(f, multiplicity, pairs)` entries. Modes within `dedup_tol_hz` of each other collapse into one entry; the matcher computes `recall = n_matched / n_distinct_freqs`. The internal modal sum in `modal_rir_2d` still iterates over individual `(n_x, n_y)` pairs via a private `_enumerate_pairs()` helper — physics is unchanged.

**Rationale:** the previous one-to-one matcher penalised L=W rooms (and any L/W rational) where multiple `(n_x, n_y)` modes share a frequency: the picker sees one peak, the matcher counts it as one match against many "modes", recall caps below 1. Distinct-frequency dedup is the cleanest fix — degenerate pairs are now correctly represented as a single physical resonance with multiplicity, and the recall metric reflects what the spectrum can actually distinguish. `dedup_tol_hz=0.01` is well below our `Δf=2 Hz` so it never accidentally merges genuinely-distinct freqs.

**Alternatives considered:** many-to-one matching (rejected — would credit one peak to multiple modes, double-counting); leave matcher one-to-one and report recall stratified by L=W vs L≠W (rejected — papers over the convention rather than fixing it).

**Revisit if:** dataset evolves toward generic non-rectangular geometries where the notion of "distinct mode frequency" is less well-defined.

---

## 2026-05-10: Dataset rebuild — `n_time_samples=8192` (2.0 s IR length)

**Decision:** all sweep YAMLs use `n_time_samples=8192` at `fs=4096`, giving 2.0-second IRs. The Chunk-1 0.5-second IRs (n_time_samples=2048) were backed up to `data/track_a_2048/` for regression purposes; new dataset lives at `data/track_a/`.

**Rationale:** at α=0.15, EDC-measured T60 ranges from 515 ms (L=2.5 m) to 826 ms (L=6.5 m). The old 0.5-s IRs covered ≤ 1× T60 in the smallest room and ≤ 0.6× T60 in the largest, biasing any energy-based metric (EDC slope, T60 fit, C50). 2.0-second IRs cover 1.5–2.5× T60 — comfortably above the -35 dB EDC fit window. File size doubles to ~7.85 MB per room (113 MB total dataset); negligible.

**Alternatives considered:** n_time_samples=4096 (1.0 s, just barely covers T60 for the largest room — too tight); n_time_samples=16384 (4.0 s, double the storage with no benefit since the IR is already at noise floor by 1.0 s).

**Revisit if:** Track B (fs=48000, where Sabine T60 should be similar but Δf is finer) requires different scaling; or if a future room family has T60 > 1.5 s.

---

## 2026-05-10: Q1 resolved — Stochastic uniform-azimuth ray sampling, n_azi=64

**Decision:** the 2D renderer (`aaf/renderers/freq_2d.py:FreqRenderer2D`) samples `n_azi=64` ray directions per receiver, drawn from `θ_grid + uniform(0, 2π/n_azi)` so the angle set rotates jitter-uniformly each iteration. No elevation, no zenith/nadir, no extra rays.

**Rationale:** the faithful 2D analogue of AVR/INFER's stochastic spherical sampler. Per-iteration jitter prevents aliasing with the 64-receiver grid and ensures every angular position is hit in expectation across iterations. n_azi=64 matches AVR's azimuth resolution; deterministic ISM-aligned sampling (option b in Q1) would generalize poorly to non-shoebox rooms in later phases. Eval uses `model.eval()` to switch off jitter for repeatability.

**Alternatives considered:** stochastic Halton/Sobol sequences (rejected — added complexity, marginal gain for our regime); ISM-aligned sampling (rejected per the spec — phase-2+ portability); higher n_azi with smaller n_pts_per_ray (deferred — memory check chose 64×32 for our 12 GB GPU).

**Revisit if:** scalability or aliasing artefacts surface in Phase 2/3 multi-room training; consider importance sampling toward image-source paths for shoeboxes, with stochastic fallback for irregular geometries.

---

## 2026-05-10: Geometric attenuation OFF — re-confirmed for Phase 1

**Decision:** `FreqRenderer2D.use_geometric_attn=False` is the Phase-1 default. The constructor accepts the flag for forward compatibility (Phase 2+) but training and eval pipelines never set it to True.

**Rationale:** see DECISIONS.md 2026-05-09 (Q8 resolution). 2D Green's function isn't 1/r; the network learns whatever spreading the data exhibits. Hardcoding 1/√r or a Hankel function would inject a wrong physical prior.

**Revisit if:** Chunk-2 reconstructions show systematic amplitude bias growing with distance (would indicate the network can't learn the geometry on its own).

---

## 2026-05-10: Output dim — `n_freq_bins = n_time_samples / 2 + 1 = 4097` for Track A

**Decision:** `INR2D_Single` constructor takes `n_freq_bins`. For the rebuilt 8192-sample dataset that's **4097** complex values per (receiver, frequency) cell. Internally the model outputs `2 * n_freq_bins = 8194` real values per branch (split into `[real, imag]`).

**Rationale:** matches the dataset's RFFT one-sided length; using a smaller `n_freq_bins` would force frequency-domain downsampling on the target before loss, throwing away the modal-frequency resolution that this Phase-1 sweep is designed to study.

**Revisit if:** Track B (fs=48000, n_time=16384 → n_freq=8193) is enabled — at that point the output dim doubles to 16386, and the wide-output decoder MLPs may need narrower hidden layers to fit.

---

## 2026-05-10: Single-room loss — 4-term L1 on (real, imag, log-amp, phase)

**Decision:** `aaf/train/single_room.py` minimises a weighted sum of:
  - `L_spec_real = L1(H_pred.real, H_target.real)`
  - `L_spec_imag = L1(H_pred.imag, H_target.imag)`
  - `L_amp = L1(log10(|H_pred|+ε), log10(|H_target|+ε))`
  - `L_phase = mean(1 - cos(angle(H_pred) - angle(H_target)))`
  with weights `(1.0, 1.0, 1.0, 0.1)`.

**Rationale:** AVR's six-term criterion has both time-domain and multi-resolution-STFT terms. We're frequency-native already, and Chunk 1.5 SANITY_NOTES showed pra's EDC differs from Sabine T60 by ~60% — energy/EDC-style losses would chase a mis-calibrated target. The four kept terms cover the magnitude (linear via real+imag, log via L_amp) and the phase (separately, with a low weight because phase is noisy at low magnitudes). The `1 - cos(Δ)` form is bounded in `[0, 2]` and naturally down-weights spurious phase noise at near-zero magnitude.

**Alternatives considered:** AVR's full 6-term battery (rejected — time loss is redundant given freq-domain target; multi-STFT is perceptual, deferred to Phase 4); MSE on complex (rejected — sensitive to outliers, less stable than L1).

**Revisit if:** Chunk 3 finds the auto-decoder's per-room differentiation requires perceptual or energy-decay supervision; consider re-introducing AVR's energy/multi-STFT losses then.

---

## 2026-05-10: Checkpoint cadence — every 2,500 iters, with auto-resume + 3-deep retention

**Decision:** `SingleRoomTrainer` checkpoints every `cfg.ckpt_every = 2500` iterations to `output_dir/ckpt_iter{N:07d}.pt`, and on startup auto-resumes from the highest-iter checkpoint that loads cleanly. Old checkpoints beyond the 3 most recent are pruned to limit disk use. Writes use a `.pt.tmp` rename to avoid leaving partial files when scavenger preemption hits mid-write.

**Rationale:** scavenger preemption is real. At ~0.55 s/iter, 2,500 iters ≈ 23 min of wall-clock work; that's the worst-case loss per preemption. 3-deep retention means a corrupt-most-recent skip still has two viable resume points.

**Alternatives considered:** every 10,000 iters (rejected — preemption loss too large); save only `model.state_dict()` to halve disk (rejected — losing optimizer/scheduler state breaks LR schedule resume).

**Revisit if:** disk usage becomes a concern (per-room dir is ~50 MB at 3 retained ckpts at our model size) or preemption rate is so low that the I/O overhead dominates.

---

## 2026-05-10: HashGrid capacity note — `log2_hashmap_size=18` is over-parameterised for 2D shoeboxes

**Decision (note, not a change):** `INR2D_Single` ships with the INFER hash-grid defaults (`n_levels=20, n_features_per_level=2, log2_hashmap_size=18, base_resolution=16, per_level_scale=1.5`), giving ~20 M parameters per encoder × 6 encoders = ~120 M hash params. For Phase 1 single-room overfit this trivially memorises a 64-receiver grid — which is **intentional** for Chunk 2 (we're measuring the upper bound on what the architecture can fit).

**Rationale (Chunk-3 implication):** if the shared MLP has enough capacity to memorise all rooms' fields without using `z_s`, the auto-decoder will fail to learn meaningful per-room latents — `z_s` would just become noise. Recommend Chunk 3 starts with a downsized grid (try `log2_hashmap_size=14-16`, `n_levels=12-16`) before defaulting to the over-parameterised setting. The memory check + train recommendations in `tasks/CHUNK_2_RESULTS.md` document the explicit numbers.

**Revisit when:** Chunk 3 implements `INR2D_AutoDecoder` — measure whether `z_s` improves per-room fidelity vs. ablation `z_s=0`. If not, downsize the grid.

---

## 2026-05-10: GitHub-tracked figures — allow `outputs/single_room/**/*.png` exception

**Decision:** `.gitignore` keeps `/outputs/**/*.png` ignored generally (the noise-floor figures stay local PNGs since they're not the headline) but adds `!/outputs/single_room/**/*.png` so the four mandatory per-room figures and the cross-room LSD plot are tracked. Tensorboard event files and `*.pt` checkpoints under `outputs/single_room/L*/` remain ignored.

**Rationale:** Chunk 2's headline plots (modal_tracking, spectrum_overlay, receiver_grid, training_curves, lsd_vs_L) are small (≤200 KB each, ~3 MB total) and load-bearing for the manager's review of the upper-bound results. PDFs would also work but are larger and less convenient than PNG.

**Revisit if:** the figure count or size grows significantly (e.g., per-receiver-per-iter videos); switch to compressed PDFs or external storage at that point.

---

## 2026-05-10: Chunk 3.5/3.5+ — auxiliary L-head architecture: `mlp_32` default + `linear` addendum

**Decision:** the `INR2D_AutoDecoder` class supports two L-head architectures via `l_head_arch`:
- `"mlp_32"` (default): `Linear(d→32) → ReLU → Linear(32→1)`. Used by Chunk-3.5 R0-R5.
- `"linear"`: `Linear(d→1)`. Used by Chunk-3.5+ addendum R6/R7/R8.

The MLP head is expressive enough to fit `L_predicted` to ANY encoding of `z_s` and so doesn't directly constrain `z_s` to be smooth in L. The linear head forces `z_s` to be linearly readable as L, which IS a structural constraint. Both are exercised in the sweep so we can compare empirically.

**Rationale:** the Chunk-3.5 progress check identified the mlp_32 head as a confound: a sub-millimetre `L_lhead` val MAE doesn't prove `z_s` learned geometry, only that the head can read L from whatever `z_s` ended up being. The linear head closes this loophole.

**Empirical outcome (R0 vs R6 ablation, both on tron RTX 2080 Ti, otherwise identical):** R6 train latents become almost monotonic in PC1 vs L (slope -0.70 vs R0's -0.54); R6 zero-shot mean held LSD is marginally better (5.30 vs 5.42 dB) but **still nowhere near the 2 dB target** — the linear L-head shapes training latents but doesn't unblock zero-shot adaptation. See `tasks/CHUNK_3_5_RESULTS.md` §9 for the full ablation table.

**Revisit if:** future work changes the inner-loop adaptation strategy (the actual bottleneck per Chunk 3.5+ analysis); the L-head architecture choice may matter differently then.

---

## 2026-05-10: Chunk 3.5/3.5+ — 6+3 hyperparameter sweep design

**Decision:** the auto-decoder retrain after Chunk-3's failure was structured as a **9-run hyperparameter sweep**, not a single re-train at one new configuration:
- Chunk-3.5 R0-R5: central bet (R0) + 5 single-axis ablations (smaller hash, larger latent, no L-head, stronger L-head, stronger L2).
- Chunk-3.5+ R6-R8: linear L-head variants (R6 = R0 with linear head; R7 = R6 + medium hash; R8 = R6 + tiny 2-D latent).

R0/R6/R7/R8 ran on tron RTX 2080 Ti (~1:55 each). R1-R5 ran on scavenger TITAN X (~5-6 h each).

**Rationale:** rather than commit to one architectural fix and risk another failure mode hidden in our assumptions, sweep around the central bet so the data tells us which axes matter. The 9-run cost is small (~16 GPU-hours total across tron + scavenger). The diagnostic value of the ablations is high: comparing R0 vs R6 isolates the L-head architecture; R6 vs R7 isolates hash size; R6 vs R8 isolates latent dimensionality.

**Outcome:** the sweep produced a **conclusive negative result**: all 4 complete runs (R0/R6/R7/R8) fail zero-shot at 5.21-5.91 dB held-out LSD. None of the architectural axes we varied moved the needle on zero-shot. The bottleneck is in the inner-loop adaptation, not the latent geometry. Without the sweep, we wouldn't have known this; we'd have re-tested only one new config (R0) and concluded its L-head was insufficient — actually all four head/capacity/dim combinations fail equivalently.

**Revisit if:** future iterations need to sweep around different hyperparameters (e.g., inner-loop adaptation strategies). The sweep infrastructure (orchestrator, summary, per-config YAMLs, headline-figure rendering) is reusable.

---

## 2026-05-10: Chunk 3.5/3.5+ — final-config decision (no usable best config)

**Decision:** none of the 4 fully-complete sweep runs (R0/R6/R7/R8) meet the meeting bar (held-out LSD ≤ 2 dB on ≥ 4/6 unseen L, PC1-vs-L R² > 0.7, intrinsic_dim ≤ 3). The "best" by spec priority order (count below 2 dB → mean LSD → R² → train LSD) is R6_tiny_lhead by tie-break (R6 mean 5.30 dB vs R0 5.42, R7 5.59, R8 5.44 — but all sit at 0/6 below the 2 dB threshold). R6 is therefore the recommendation **for diagnostic purposes only** — its `latent_pca_1d` figure is the cleanest visual of "the L-head shapes train latents but zero-shot collapses anyway".

**Rationale for not picking a config to ship:** all 4 architectures fail the meeting bar uniformly. Picking one as "the result" would misrepresent the chunk; the negative finding is itself the result.

**Revisit when:** a future chunk addresses the inner-loop adaptation bottleneck (more observed receivers, multi-restart inner adaptation, longer inner loop, latent-hull-constrained adaptation per `tasks/CHUNK_3_5_RESULTS.md` §11). At that point a NEW best-config decision is needed using whichever architecture the inner-loop fix is implemented around.

---

## 2026-05-10: Chunk 3 — keep INFER's HashGrid capacity; revisit after meeting the headline target

**Decision:** for the Chunk-3 dense-sweep auto-decoder run, keep INFER's HashGrid defaults (`log2_hashmap_size=18, n_levels=20, n_features_per_level=2`). Mitigations: small L2 reg on z_s (`λ_latent = 1e-4`) + latent-rank PCA diagnostic in eval.

**Rationale:** spec preference for meeting deliverables over latent-quality risk. The Chunk-2 capacity warning (§6 / §8) was acknowledged but de-prioritised.

**Revisit if:** zero-shot at unseen L fails — at which point the latent-probe plot motivates a smaller HashGrid retrain. **(Triggered: Chunk 3 confirmed the failure. Recommended next config: `log2_hashmap_size=14, n_levels=14` — see `tasks/CHUNK_3_RESULTS.md` §10.)**

---

## 2026-05-10: z_s injection — candidate A (concat at both sigma + signal branches)

**Decision:** the `INR2D_AutoDecoder` widens both `sigma_in_dims` and `n_signal_input` by `latent_dim`, broadcasting `z_s` over the per-point batch dimension and concatenating with the existing pos/dir embeddings at both concat points (the two `# CHUNK-3 INJECTION POINT` comments in `aaf/models/inr_2d.py:INR2D_Single.forward`).

**Rationale:** matches `aaf/_inference_ref/inference_model.py` candidate-A guidance from CHUNK_0_RESULTS §7. The sigma branch alone (candidate B) would only let z_s gate absorption/dispersion; including the signal branch lets z_s also modulate the spectral emission, which is where the modal pattern lives.

**Revisit if:** ablation shows the sigma-only or FiLM variants beat candidate A on the smaller-HashGrid retrain.

---

## 2026-05-10: Chunk-3 multi-room training — auto-decoder loss = 4 spec terms + λ‖z_s‖² with λ=1e-4

**Decision:** the multi-room loss adds an L2 reg on z_s with weight `1e-4` to the four spec terms from Chunk 2 (real, imag, log-amp, phase). Two-param-group optimizer: network at lr=2e-4, latents at lr=1e-3 (DeepSDF convention; latents benefit from a higher LR).

**Rationale:** `λ=1e-4` is small enough not to push z_s toward zero against a useful gradient signal but large enough to discourage runaway magnitudes. The 5×-the-network LR for latents matches DeepSDF's standard recipe.

**Verification:** Chunk 3 final-iter latent norms are 0.70-0.95 (init was 0.18) — they grew naturally from initialisation but did not explode. L_latent term hovers at 0.02 throughout training, confirming the reg is not load-bearing on the loss.

**Revisit if:** a smaller-HashGrid retrain shows the L2 reg is too lenient (latents still don't track L) or too aggressive (latents collapse to zero); current value is the conservative middle.

---

## 2026-05-10: Q5 cluster partition closed — tron for training, scavenger for everything else

**Decision:** Chunk 3 used 1 tron slot for the multi-room training (non-preemptible, 12-h time limit) and 6 scavenger slots in parallel for the zero-shot evals. This pattern stays for any future chunk that needs a single long training run plus N parallel evals.

**Rationale:** the multi-room training landed on `tron64` with an RTX 3070 and finished in 2:38 — well under the 12-h limit, with zero preemption. Scavenger evals queued, ran, and completed all 6 within a 17-min window. The 4 banked tron slots remain available for retries / Chunk-4.

**Revisit if:** a future training run exceeds 12 h on tron, or if scavenger preemption rates climb such that even short eval runs miss their wall.

---

## 2026-05-10: Chunk 2 training — 10K iters default with relative-improvement early-stop, replacing initial 50K

**Decision:** `SingleRoomTrainer` defaults to `n_iters=10_000`, `val_every=500`, and a new relative-improvement early-stop:
- `early_stop_warmup = 2_000` (no check before this iter)
- `early_stop_patience = 2_000` (window of "recent" val checkpoints)
- `early_stop_min_rel_improvement = 0.01` (1%)
- At each val checkpoint past warmup, compare `min(val_total_loss in (iter - patience, iter])` against `min(val_total_loss in [0, iter - patience])`. Stop if the recent best is not at least 1% lower than the prior best.
- The training SLURM time is reduced from 24h → 6h (10K iters at observed ~1.0 s/iter ≈ 3h, plus margin).

**Rationale:** the first run (cancelled at iter 3100) was on track for 14h to reach 50K iters, with train loss already 3× lower than at iter 1000 and still in clean exponential decay. 50K is overkill for a single-room overfit baseline whose only purpose is to upper-bound per-room reconstruction quality. 10K iters lets us iterate to Chunk 3 fast; the relative-improvement stop also auto-cuts training when a room converges before 10K. The previous absolute-threshold stop (`L_spec_real + L_spec_imag < 1e-3`) was tuned for 50K iters and never triggers in 10K — replacing it with a plateau detector is more useful at this budget.

**Alternatives considered:** keep 50K with auto-resume across SLURM time limits (rejected — 14h occupies the queue with no scientific gain); fixed 5K iters (rejected — too aggressive, doesn't catch easy convergence as a "stop earlier" signal); patience based on a fixed iter count rather than relative improvement (rejected — relative is more robust to absolute loss scale).

**Verification:** `tests/test_early_stop.py` exercises the helper with 5 controlled scenarios (warmup-gated, plateau, improving, missing-history, first-eligible). The smoke test on L=4.5 confirmed val logging at iter 500/1000/1500/2000 and the early-stop check correctly returned "continue" at iter 2500 (improving).

**Revisit if:** Chunk 3 finds the auto-decoder needs longer training to differentiate per-room latents (likely — multi-room conditioning is harder than single-room overfit) and the warmup/patience need rebalancing.

---

## 2026-05-11: Band-limited LSD is the standard zero-shot eval going forward

**Decision:** every future zero-shot run reports band-limited LSDs (modal 0-250 Hz, transition 250-500 Hz, diffuse 500-2000 Hz, full 0-2000 Hz) alongside the existing full-band held LSD. `aaf.eval.zero_shot.zero_shot_adapt` now writes both `metrics.json` (full + the four band metrics inline under `band_metrics_held` / `band_metrics_obs`) and a sibling `band_limited_metrics.json` (just the bands, for ergonomics). `aaf/eval/band_limited.py` exposes `compute_band_limited_metrics(H_pred, H_target, fs, n_freq_bins, bands) -> dict` and `DEFAULT_BANDS`.

**Rationale:** Chunk-3.5 zero-shot full-band LSDs (5.2-5.9 dB) were uniform across architecturally-diverse runs and visual inspection of overlays suggested the modal regime was tracking correctly. Track A confirmed this quantitatively: modal LSD is ~1.7 dB lower than full across R0/R6/R7/R8 (3.54-3.69 vs 5.27-5.57 dB), demonstrating the failure is *not* uniform across frequency. Reporting only full-band hides the partial win and conflates physically-distinct regimes (sparse modes vs diffuse density). The 0-250 Hz cutoff is conservative — Schroeder frequency for L=4.5, W=4 m at α=0.15 is ~210 Hz; the modal regime is dominated by isolated eigenmodes and is what the analytical 2-D Helmholtz solution covers.

**Alternatives considered:** keep full-band only (rejected — masked the modal-regime success); octave bands (rejected — overkill for the meeting story; can be added later); per-receiver grouping (rejected — receivers are already collapsed in the existing LSD definition).

**Verification:** `tests/test_band_limited_lsd.py` injects a known +6 dB magnitude offset only in the 250-500 Hz band of a synthetic H_pred and asserts (a) the 250-500 Hz LSD comes back at 6.00 ± 0.05 dB, (b) the other bands are < 0.05 dB, (c) the full-band LSD is the expected weighted average (~0.75 dB).

**Revisit if:** future analysis decides we want octave-band breakdowns or perceptually-weighted (A-weighted) LSDs.

---

## 2026-05-11: FiLM conditioning uses input-side γ·feat + β with γ-init=1, β-init=0

**Decision:** when `INR2D_AutoDecoder` is built with `conditioning_type='film'`:
- The sigma branch input becomes `γ_σ(z_s) · concat(pos_emb, tx_pos_emb) + β_σ(z_s)` (z_s is dropped from the cat). The signal branch input is built the same way: `γ_g(z_s) · concat(F.relu(σ_feature), view_emb, tx_view_emb, signal_pos_emb, tx_signal_pos_emb) + β_g(z_s)`.
- The FiLM generators are single `nn.Linear(latent_dim, 2*F)` whose output is split via `chunk(2, dim=-1)` into γ (first F) and β (last F). F is computed dynamically from the encoders' `n_output_dims` and matches the dim that would be concatenated under `'concat'` (without z_s).
- Weights are initialised to 0 and bias to `[1,...,1, 0,...,0]` so γ=1, β=0 at construction → identity modulation (FiLM doesn't perturb the encoded features until the generator weights move).
- The tcnn `n_input_dims` shrinks by `latent_dim` in the FiLM path so the fused MLP sees only the (modulated) encoded feature, not z_s.

**Rationale:** the natural design of "true FiLM" — applying γ/β between every hidden layer of an MLP — is **infeasible** with our `sigma_mlp` and `signal_mlp` because they're tcnn `FullyFusedMLP`/`CutlassMLP` (fused CUDA kernels with no Python access to intermediates). The tractable design is to FiLM at the **encoded-feature input** of each tcnn block. The hypothesis (smoother latent-to-spectrum response surface → easier zero-shot adaptation) still holds: FiLM modulates the high-rank encoded features through a low-rank affine transform, which is more constrained than the concat path's "anything-goes" first layer. Initialising to identity means the model trains from a well-defined neutral starting point that matches the concat baseline's statistical behaviour at iter 0 (modulo the fact that the tcnn MLP itself is a different fresh network).

**Alternatives considered:** output-side FiLM on the tcnn block's outputs (rejected — modulation acts on a representation already produced without knowing about z, even less expressive than input-side); replacing the tcnn MLPs with torch `nn.Sequential` to enable per-layer FiLM (rejected — ~3-5× slower training, breaks the overnight budget for Chunk 3.6); both input-side AND output-side FiLM (rejected for now — adds parameters without a clear reason to expect a 2× gain).

**Verification:** `tests/test_film_conditioning.py` asserts (a) the FiLM generators have output dim `2*F` matching the encoded-feature dim, (b) the tcnn MLPs' `n_input_dims` shrink by `latent_dim`, (c) γ/β at init are exactly 1/0, (d) gradients flow through the FiLM generators on a forward+backward pass, (e) `conditioning_type='bogus'` raises.

**Revisit if:** C1 underperforms R6 zero-shot — in that case we'd test (a) replacing tcnn with torch MLPs to enable per-layer FiLM despite the speed hit, (b) hyper-networks (z → MLP weights) as a more expressive alternative.

---

## 2026-05-11: Latent jitter applies inside `get_latent`, gated on `self.training`

**Decision:** `INR2D_AutoDecoder.get_latent(room_id)` returns `embedding(room_id) + N(0, σ²I)` when `self.training and self.latent_jitter_sigma > 0`, and the unperturbed embedding otherwise. The default `latent_jitter_sigma=0.0` preserves Chunk-3.5 behaviour exactly. C2 sets it to 0.1 (small relative to ‖z_s‖ ≈ 2-3).

**Rationale:** the failure mode identified in Chunk 3.5 is that the model's latent-to-spectrum response surface has steep curvature outside the trained-latent neighbourhood, so the inner-loop optimiser can't navigate it. Training the decoder to be robust to small perturbations of z_s explicitly smooths that surface around each trained z_s — making zero-shot adaptation an interpolation problem rather than an extrapolation one. Putting the noise inside `get_latent` (not in the trainer's `_step`) means: (a) it composes naturally with the L-head loss (which now sees jittered z_s and so learns to be jitter-robust too — desirable); (b) it's automatically OFF in `validate()` and at zero-shot time because both call `model.eval()`, satisfying the "deterministic at test time" requirement without ad-hoc gating.

**Alternatives considered:** add jitter in `_step` after `z_s = model.get_latent(...)` (rejected — splits the jitter logic across two files); use `nn.Dropout` on z_s (rejected — multiplicative noise has different properties than additive Gaussian and isn't what the literature recommends for representation smoothing); learn σ as a parameter (rejected — adds optimisation surface without clear benefit at this scale).

**Verification:** `tests/test_latent_jitter.py` asserts (a) train mode + σ=0.5 produces different `get_latent` outputs across calls, (b) eval mode + σ=0.5 is deterministic, (c) σ=0.0 is deterministic even in train mode, (d) σ<0 raises.

**Revisit if:** C2 destabilises training (σ=0.1 may be too large vs early-training ‖z_s‖) — in that case we'd anneal σ from 0 to 0.1 across the warmup, or scale σ by `‖z_s‖` to make it relative.

---

## 2026-05-11: I2 FiLM + rank-r LoRA hyper-network-style conditioning — output-side adapter with zero-init projection

**Decision:** `INR2D_AutoDecoder.conditioning_type='film_lora'` adds an output-side rank-r additive adapter on top of plain FiLM (input-side γ/β on encoded features). For both the sigma decoder and the signal MLP:

```
output_with_adapter = tcnn_decoder(...) + proj(A(z_s) * B(features))
```

where `A: Linear(latent_dim, r)`, `B: Linear(feat_dim, r, bias=False)`, `proj: Linear(r, output_dim, bias=False)`, and **`proj.weight` is zero-initialised**. Default `lora_rank=8`. At construction the adapter contributes exactly zero, so the model is bit-identical to plain FiLM (`conditioning_type='film'` — Chunk 3.6 C1). Gradient through A·B·proj kicks in as training proceeds.

**Rationale:** spec called for a "FiLM-extended hyper-network" — true per-layer FiLM with the tcnn fused MLPs is infeasible (kernels don't expose intermediates), so the chosen tractable variant is an additive output-side rank-r correction. Rank 8 keeps parameter overhead minimal (8 × signal_output_dim ≈ 65K params per branch — much smaller than the tcnn MLPs themselves) while giving z a meaningful additional handle on the output. Zero-init on the proj guarantees no-worse-than-FiLM at init.

**Alternatives considered:** side-MLP additive adapter (rejected — higher overfitting risk on the small 7-room training set; the same failure mode we saw with the larger HashGrid in Chunk 3); true weight-generating hyper-network (rejected — wouldn't fit the chunk's overnight budget; deferred to a post-meeting chunk if the LoRA design proves promising).

**Verification:** `tests/test_film_lora_conditioning.py` asserts (a) film_lora at construction has `proj_sigma.weight == 0` and `proj_signal.weight == 0` exactly; (b) the A/B/proj modules exist with rank `lora_rank` (matrix shapes verified); (c) the modules are absent for `conditioning_type='film'`; (d) gradient flows through A/B/proj after one backward; (e) `lora_rank=0` raises ValueError.

**Outcome (in-chunk):** D2_filmlora trained val LSD 1.42 dB (identical to C2's 1.43); zero-shot modal 3.35 dB with B6 inner loop (vs C2 + B6's 3.51). Marginal 0.16 dB improvement — the LoRA branch didn't usefully expand expressivity. The hypothesis that "rank-r z-gated output correction = better generalisation than plain FiLM" was not supported on this dataset.

**Revisit if:** a true weight-generating hyper-network (deferred Track I option C) is attempted in a future chunk and works — that would confirm the bottleneck is hyper-net depth, not the additive-vs-multiplicative structure.

---

## 2026-05-11: Chunked-receiver gradient accumulation requires per-chunk `get_z()` (Chunk 3.7 I3)

**Decision:** `aaf/eval/zero_shot.py:zero_shot_adapt` supports `chunk_size: int = 0`, where `0` means full-batch (legacy behaviour) and `> 0` processes receivers in chunks: each chunk does its own forward+`loss.backward()` (with the per-chunk loss scaled by `chunk_n / total_n` so the accumulated gradient equals the full-batch gradient), and `optimizer.step()` runs once per outer iter after all chunks. **CRUCIAL detail**: each chunk must call `get_z()` fresh to obtain a NEW autograd subgraph for that chunk's forward pass. The first implementation reused a single `z_now = get_z()` for all chunks; this worked for random/nearest-train init (z_now is a leaf nn.Parameter so backward has no subgraph to free) but broke for `init_strategy='simplex'` (where z_now = softmax(logits) @ Z_train is a derived tensor — chunk 1's backward freed the softmax subgraph, breaking chunk 2's backward and the final `(λ_latent * l_latent).backward()`).

**Rationale:** the per-chunk gradient accumulation was added so n_obs=32 fits on TITAN X (B7 variant; the full-batch forward needed > 10 GB). The simplex case was a Chunk-3.6 addition (B6); chunked + simplex is a Chunk-3.7 combination that needed this explicit handling. Calling `get_z()` per chunk is cheap (for simplex it re-runs `softmax(logits) @ Z_train` with logits ≈ 7-dim and Z_train 7×8 → 56 multiplications; for random/nearest_train it returns the cached leaf Parameter directly — no extra cost).

**Alternatives considered:** `retain_graph=True` on all chunk backwards (rejected — keeps the full chunk graph in memory, defeating the chunking's memory benefit); detach z_now and re-attach for each chunk (rejected — would lose gradient flow to simplex logits entirely); fold λ_latent into the first chunk (rejected — fragile and didn't fix the chunk-2-breaks-chunk-3 problem).

**Verification:** `tests/test_chunked_inner_loop.py` asserts the chunked-and-accumulated gradient equals the full-batch gradient within 1e-9 on a synthetic torch-only model (no tcnn). The Chunk-3.7 I3 run with B7 (n_obs=32, chunk_size=8) on the C2 model completed all 6 L values successfully after this fix landed (modal LSD 3.53 dB — essentially flat vs C2 + B6's 3.51, ruling out "8 observations is too few" as the bottleneck).

**Revisit if:** a future variant uses an init_strategy whose computational graph differs from simplex in a way we haven't tested (e.g., learned z = MLP(some_features) where the MLP is large enough that the per-chunk recomputation is expensive). In that case fall back to `retain_graph=True` on the first n-1 chunks.

---

## 2026-05-11: Modal-LSD ceiling is data-density-bound, not info-bound or LoRA-capacity-bound (Chunk 3.7 mechanism finding)

**Decision:** Going forward we treat training-set density as the primary lever for zero-shot modal LSD. Concretely: future improvement chunks should denser-sweep first (more training rooms at finer L spacing) before architectural changes.

**Rationale:** the three Chunk 3.7 Track I experiments give a clean partial ordering of where the ~3.5 dB modal ceiling came from in Chunk 3.6:
- **I3** (n_obs=8 → 32 via chunked inner loop on the same trained model): modal stays at 3.53 dB → not info-bound.
- **I2** (FiLM + rank-8 LoRA on decoder output, otherwise C2 architecture): modal 3.35 dB → not capacity-bound in the additive-output-LoRA direction (marginal 0.16 dB improvement is within run-to-run noise).
- **I1** (15 rooms at 0.2 m vs 7 at 0.5 m, otherwise C2 architecture): modal **drops from 3.51 to 2.55 dB**, a 1 dB improvement — by far the largest single-chunk movement of the project. Per-L modal: 2.81, 2.58, 2.69, 2.63, 2.33, 2.28 — at the upper-half L's within 0.3 dB of the 2 dB target.

So: more interpolation anchors smooth the latent-to-spectrum mapping that the decoder learns, without changing the decoder architecture. This is the mechanism-level finding that locks in the next-chunk direction.

**Alternatives considered:** prioritise hyper-network conditioning instead (deferred — I2's null result doesn't rule out true hyper-networks but reduces the prior probability that purely-architectural changes will hit the 2 dB target as cheaply as more data); prioritise wider decoder capacity (the 15-room D1 val LSD 2.37 vs 7-room C2's 1.43 suggests the decoder is now saturated, so widening could help — but it's a secondary lever).

**Verification:** Chunk 3.7 Track I numbers in `tasks/CHUNK_3_7_RESULTS.md`. The D1 D1_dense15 + B1 baseline at modal 2.55 dB is the live evidence; the per-L breakdown shows the gap shrinks consistently as L gets further from the 3.0 m endpoint.

**Revisit if:** the next-chunk experiments (0.1 m spacing or [2.6, 6.0] extension) DON'T continue to lower modal LSD — that would refute the data-density mechanism and force us to revisit architecture (option I2 part 2, or hyper-network).

---

## 2026-06-04: Phase 2 starts — 3D shoebox port; geometry varies in L, W, H; α fixed at 0.15 (Chunk P2-1, D1-D5)

**Decision:** Phase 2's first chunk (P2-1) ports the core pipeline from 2D to 3D shoeboxes with the following fixed-for-all-of-Phase-2 design choices:
- **Room dimension ranges**: L ∈ [3.0, 6.0], W ∈ [3.0, 5.0], H ∈ [2.5, 4.0] m (D1).
- **Receiver layout**: 8×8×8 = 512 receivers per room, regular grid with 0.3 m margin from all walls (D2). Smallest H=2.5 m → usable Z = 1.9 m → 7×0.27 m spacing fits; no adaptive margins needed.
- **Source position**: fixed at (0.5, 0.5, 0.5) m from the corner (D3).
- **Absorption**: α = 0.15 uniform on all 6 walls (D4).
- **Frequency**: fs = 4096 Hz, n_time = 8192, n_freq_bins = 4097, Δf = 0.5 Hz, 0-2 kHz band (D5) — identical to Phase 1 Track A for continuity.

**Rationale:** these are realistic room proportions (rooms get taller as they get wider — H/W ≥ 0.5). The frequency band continuity lets Phase-1 utilities (`band_limited.py`, `modal_verifier.py`) work for 3D without modification.

**Alternatives considered:** wider dim ranges (rejected — manager spec; broader ranges would dilute the LHS sample density needed for P2-2 multi-room conditioning); variable absorption (rejected — disentangle geometry from materials in Phase 2, revisit in Phase 3).

**Revisit if:** Phase 2 results require absorption diversity for downstream applications. The fixed-α design is a clean control for Phase 2's geometry-only conditioning experiments.

---

## 2026-06-04: 3D ISM `max_order` is hard-capped at 17 (Chunk P2-1, D6, D7)

**Decision:** `aaf/sim/ism_3d.py:MAX_ORDER_CAP = 17`. The auto-rule `ceil(c·4·T60/min(L,W,H))` would request ~478 for a 6×5×4 m room at α=0.15 (T60 ≈ 0.87 s), which generates (2·478+1)³ ≈ 8.8×10⁸ image sources — OOM. Capping at 17 keeps the simulation tractable (~42 K image sources, ~5-15 s per room on 4 CPUs) and the IR covers ~108 ms of decay — 4× the 50 ms early-reflection envelope. `analytical_modal_3d` is the deterministic modal ground truth for low-frequency accuracy. We do **NOT** enable `set_ray_tracing` in P2-1 (D7): the stochastic late tail would break array-task idempotency and the analytical modal reference assumes deterministic ISM.

**Rationale:** 3D ISM image-source count scales as the cube of `max_order`, not the square as in 2D — the 2D auto-rule simply doesn't transfer. Tail truncation beyond 108 ms is acceptable for de-risk single-room overfit because (a) low-frequency modes live in the truncation-stable early field and (b) the diffuse late field is incoherent anyway. The ir_truncated warning flag in the meta surfaces this for downstream awareness.

**Alternatives considered:** hybrid ISM + ray-tracing for the late tail (deferred to Phase 3 — non-deterministic in a way that breaks the modal ground-truth comparison); per-room max_order autoselect with an upper cutoff (rejected — fixed cap is simpler and the resulting IRs are still informative for single-room overfit).

**Verification:** `tests/test_ism_3d.py:test_max_order_cap_in_meta` asserts auto-max_order ≤ MAX_ORDER_CAP and T60 falls in the [0.3, 1.5] s range expected for α=0.15. `outputs/budget_check_3d/REPORT.md` records per-room wall-clock + max_order at run time.

**Revisit if:** Phase 2 / Phase 3 work needs the diffuse-tail accuracy (e.g., reverberation-time-based metrics for unseen rooms). Then add ray-tracing with a per-room seed pulled from a hash of (L, W, H) so tasks remain idempotent.

---

## 2026-06-04: 3D ray sampler mirrors vendored INFER pattern: `n_azi × n_ele` + 2 poles (Chunk P2-1, D8, D11)

**Decision:** `aaf/renderers/freq_3d.py:FreqRenderer3D` samples directions on a stratified `n_azi × n_ele` grid with per-iteration azimuth jitter (training mode only), plus two pole rays explicitly added. Defaults `n_azi = 16, n_ele = 16 → 258 rays`. Azimuth uniform on `[0, 2π)`; elevation `arccos(2·u - 1)` for stratified `u ∈ linspace(0, 1)` so the solid-angle weight is uniform. The spec's "n_rays = 256" maps to n_azi = n_ele = 16. `use_geometric_attn=False` carries over from Phase 1.

**Rationale:** mirrors the vendored `aaf/_inference_ref/inference_renderer.py:40-57` pattern that INFER validated, so the 3D forward graph is structurally identical to the validated 2D forward graph. `arccos(2u-1)` stratification gives solid-angle-uniform area weighting that Fibonacci sphere only approximates non-jitterably. Pole rays catch ceiling/floor specular returns most directly — dropping them would cost the first vertical-axis mode.

**Alternatives considered:** Fibonacci sphere (rejected — no clean jitter mechanism); pure equiangular grid (rejected — biases toward poles); larger n_ray budget (rejected without memory-check evidence; the n_pts × n_ray product dominates GPU memory).

**Verification:** `tests/test_renderer_3d.py:test_renderer_3d_ray_directions_unit_norm` and `test_renderer_3d_ray_count` cover the structural invariants. `test_renderer_3d_jitter_in_train_mode_only` verifies that `.eval()` is deterministic and `.train()` jitters.

**Revisit if:** the single-room overfit shows directional bias (e.g., consistent over-reconstruction along one axis) — that would suggest the 16×16 stratification + jitter is under-sampling. Bump to 24×24 or switch to Fibonacci sphere.

---

## 2026-06-04: 3D ray-AABB intersection by 3-axis slab algorithm (Chunk P2-1, D9)

**Decision:** `aaf/renderers/freq_3d.py:_ray_aabb_intersect_3d` — direct port of the 2D slab algorithm over three axes. Returns per-(receiver, ray) `t_far` so `n_pts_per_ray` is packed inside the actual ray-in-room segment.

**Rationale:** the vendored INFER renderer uses fixed `near, far` per ray which wastes ~40% of the sample budget outside the room. AABB slab gives ~2× effective sample density at the same `n_pts_per_ray`. For a 6×5×4 m room, mean ray length ≈ 4.3 m vs fixed-far of 8.8 m → 2× density. Phase 1's 2D version was already this design; D9 just confirms the 3-axis extension.

**Alternatives considered:** fixed `near, far` (rejected — measurable density penalty); per-ray adaptive sampling (rejected — autograd complexity for unclear benefit).

**Verification:** `tests/test_renderer_3d.py:test_renderer_3d_aabb_in_unit_cube` checks per-ray distances on the unit cube fall in `[0.5, √3/2]` (half-side to half-diagonal).

**Revisit if:** ever wanted (no clear trigger; this is correct geometry).

---

## 2026-06-04: HashGrid 3D defaults — log2_hashmap_size=18, n_levels=16, per_level_scale=1.38 (Chunk P2-1, D10; user-approved override of spec's 16/16/1.5)

**Decision:** `aaf/models/inr_3d.py:_default_hash_grid_config_3d()` returns:
- `log2_hashmap_size = 18` (4× Phase-1's 14)
- `n_levels = 16`
- `per_level_scale = 1.38`
- `base_resolution = 16`
- `n_features_per_level = 2`

**Rationale:** the spec asked for 16/16/1.5 as a "middle ground", but design-time numerical analysis (Plan-agent + user discussion) showed this under-provisions in 3D:
- The HashGrid covers a 3D volume; the table size that gives equivalent collision rate as Phase-1's 2D-validated 14/14 is roughly 4× the entries (scales with the room's smaller-dim in meters → 3-4×).
- At `per_level_scale = 1.5`, finest-level resolution at `n_levels = 16` is 16·1.5¹⁵ ≈ 7700/axis (~0.8 mm in a 6 m room) — 10× finer than λ/2 at 2 kHz, pure collision waste.
- At `per_level_scale = 1.38`, finest-level resolution is ~1700/axis ≈ 3.5 mm in a 6 m room → ≈ λ/2 at 2 kHz, matched to the highest-resolved freq.
- Memory cost: 6 encoders × 2¹⁸ × 2 × fp32 ≈ 12 MB total — trivial.

**Alternatives considered:** spec's 16/16/1.5 (rejected after Plan-agent collision-rate analysis; user approved override); INFER reference's exact config (rejected — INFER targets real-world rooms with different sizing).

**Verification:** the model defaults are returned by the helper at construction; `tests/test_model_3d.py` instantiates the model with a smaller HashGrid (n_levels=4, log2=12, scale=1.5) for memory-frugal tests but the production defaults match D10. The empirical test is the single-room overfit quality — if modal MAE > 3 Hz on a clear majority of de-risk rooms, the open question lands in `OPEN_QUESTIONS.md` and we iterate downward.

**Revisit if:** single-room 3D overfit fails at modal MAE > 3 Hz, OR plateaus too quickly suggesting over-provisioning. In either case adjust `per_level_scale` (1.34 for finer / 1.42 for coarser) before changing `log2_hashmap_size`.

---

## 2026-06-04: Memory cascade for 3D single-room training (Chunk P2-1, D12, D13)

**Decision:** `scripts/memory_check_3d.py` tries configurations in this order:
1. (n_azi=16, n_ele=16, n_pts=32, batch=8) — canonical
2. (16, 16, 16, 8) — reduce n_pts first
3. (16, 16, 32, 4) — then reduce batch
4. (16, 16, 16, 4) — last resort

n_rays stays at 258 — the angular discretization governs modal coverage and shouldn't be reduced. The 5 single-room trainings are pinned to **tron RTX 2080 Ti (24 GB)**; TITAN X (12 GB) is too small for 3D activations.

**Rationale:** Plan-agent memory math at the canonical config: 256 rays × 32 pts × 4097 freq × 8 batch × 8 B (complex64) ≈ 2.0 GB for one intermediate. With ~2-3 live intermediates (signal_with_delays, transmittance_amp, alpha) → ~6 GB activations; plus autograd graph (~2×) → ~14 GB at peak. Won't fit on TITAN X (12 GB). On tron 2080 Ti (24 GB): yes with ~6 GB headroom. Reduce n_pts before batch because n_pts controls the optical-depth integration accuracy more directly.

**Alternatives considered:** mixed-precision (rejected — tcnn HashGrid + complex arithmetic in fp16 has known precision issues for σ-near-zero regions; not worth debugging in P2-1); gradient checkpointing on the renderer (deferred to P2-2 if multi-room training hits memory).

**Verification:** `scripts/memory_check_3d.py` writes `outputs/memory_check_3d/result.json` with the chosen config. The training pipeline gates on this result and passes the chosen `n_pts_per_ray, batch_size` via env vars to `single_room_3d_train.sh`.

**Revisit if:** P2-2 multi-room training pushes the activation footprint higher; consider gradient checkpointing or chunked-receiver gradient accumulation (the chunked-inner-loop pattern from Chunk 3.7 I3 generalizes).

---

## 2026-06-04: 3D room sampling — LHS for train, structured maximin for test, spec-prescribed for derisk (Chunk P2-1, D14, D15)

**Decision:** `aaf/data/sample_rooms_3d.py` provides three sampling routines:
- `sample_train_rooms_lhs(n=45, seed=42)` — Latin hypercube via `scipy.stats.qmc.LatinHypercube` over (L, W, H) ∈ [3,6]×[3,5]×[2.5,4]. Oversample factor 4× and reject draws with `|L - W| < 0.05` (mitigates 2-axis modal degeneracy). Deterministic for seed=42.
- `derisk_rooms()` — returns the 5 spec-prescribed coordinates: box center (4.5, 4.0, 3.25) + 4 extreme-corner-ish rooms.
- `sample_test_rooms(n=8, lhs_rooms=...)` — box center + 7 greedy-maximin selections from a Sobol candidate pool, maximizing min-distance to the LHS draws in normalized [0,1]³ space.

**Rationale:** LHS gives space-filling coverage with reproducibility; reject-near-cubic mitigates the degeneracy where (1,0,0), (0,1,0) modes coincide (which would muddy the modal verifier). The greedy-maximin test selection ensures interpolative interior probes that don't overlap LHS draws — important for P2-2's zero-shot eval validity. The 5 de-risk rooms are dictated by spec and cover the (L, W, H) extreme corners + center.

**Alternatives considered:** Sobol or Halton for training (LHS is preferred for stratified per-axis coverage with small `n`); pure-random test rooms (rejected — could overlap LHS samples).

**Verification:** `tests/test_lhs_sampling.py` asserts the 45-room set is within the ranges, has no near-cubic draws, no duplicates, ≥30% spread per axis, and is reproducible across seed=42 invocations. The test rooms test asserts box center is first and no test room is within 1 mm of any LHS draw.

**Revisit if:** P2-2 multi-room training reveals coverage gaps (e.g., a region of (L, W, H) space where the auto-decoder consistently fails); could augment with active-learning-driven additional draws.

---

## 2026-06-04: 3D dataset HDF5 naming + idempotency via sentinel files (Chunk P2-1, D16)

**Decision:** Each 3D room dataset file is `data/track_a_3d/L{L:.2f}_W{W:.2f}_H{H:.2f}.h5`. The build script (`scripts/build_3d_dataset.py`) writes atomically: simulate → write to `<name>.h5.tmp` → fsync → rename to `<name>.h5` → write `<name>.h5.done` sentinel containing `(L, W, H, wall_clock)`. Skip the simulation entirely if `<name>.h5.done` already exists.

**Rationale:** the dataset is generated by SLURM array jobs (5 de-risk on tron, 45 training on scavenger). Scavenger preempts; we need each array task to be both **idempotent** (re-running re-skips completed work) and **atomic** (partial writes never present a half-built HDF5 to the loader). The sentinel pattern is filesystem-level — no manifest lock contention across the 45 parallel scavenger tasks.

**Alternatives considered:** HDF5 file checksum after write (rejected — too slow + h5py doesn't expose a clean checksum); JSON manifest with file lock (rejected — write contention from 45 parallel tasks is non-trivial); skip-if-exists by file size (rejected — partial writes have non-zero size).

**Verification:** `scripts/build_3d_manifest.py` walks the sentinel files and validates each points to an existing HDF5; emits warnings on mismatches. The pipeline orchestrator runs this after each array completes.

**Revisit if:** scavenger preemption rates become so high that the `data/track_a_3d/manifest.json` falls out of sync more than once per run — then upgrade to per-shard JSON manifests merged at the end.

---

## 2026-06-04: Signal-level eval suite API — 3-layer factoring (Chunk P2-1, D17; Dolby-requested)

**Decision:** `aaf/eval/signal_level.py` follows the same 3-layer factoring as `aaf/eval/band_limited.py`:
- **Layer 1**: pure component functions (one metric each, testable in isolation): `magnitude_correlation`, `phase_correlation_mag_weighted` (mag-weighted cos), `per_band_lsd` (reuses `compute_band_limited_metrics`), `rir_pearson`, `edc_db` (Schroeder integration), `edc_error` (max-dB, RMSE, T20, T30 deltas), `early_late_corr` (split at 50 ms), `envelope_corr` (Hilbert magnitude).
- **Layer 2**: `compute_signal_metrics(H_pred, H_target, fs, n_time_samples, ...)` — one-call aggregator returning a flat dict; accepts optional `rir_pred, rir_target` injectables.
- **Layer 3**: `make_signal_plots(...)` — writes 5 PNGs (`magnitude_overlay`, `phase_overlay`, `rir_time_overlay`, `edc_overlay`, `signal_metrics_summary`) to a directory.

The default bands for per-band LSD are `((0,250), (250,500), (500,1000), (1000,2000))` Hz — sub-divides Phase 1's diffuse band to give Dolby finer resolution above the modal regime.

**Rationale:** Dolby explicitly asked for magnitude/phase correlation + time-domain RIR analysis. The 3-layer factoring matches the Phase-1 utility pattern that proved reusable (Chunk 3.6's `band_limited.py`) and treats the API as stable for all of Phase 2 — P2-2's zero-shot eval will reuse `compute_signal_metrics` directly.

**Alternatives considered:** monolithic `compute_signal_metrics` (rejected — component functions wouldn't be reusable inside the trainer's val loop or modal verifier); per-receiver-then-mean vs flatten-then-mean (kept per-receiver-then-mean for correlation metrics so the per-receiver Pearson stays bounded in `[-1, 1]`).

**Verification:** `tests/test_signal_eval.py` asserts:
- Identical signals → all correlations ≈ 1.0 and LSD ≈ 0.0.
- Constant phase shift → magnitude correlation stays ≈ 1.0, phase correlation drops below 0.1.
- Strong additive noise → all correlations drop below 0.99.
- EDC is monotone non-increasing in time.
- `compute_signal_metrics` returns the expected key set.

**Revisit if:** Dolby asks for additional time-domain metrics (e.g., direct-to-reverberant ratio, IACC, clarity C50/C80). Add as Layer-1 components without changing the Layer-2 / Layer-3 contracts.

---

## 2026-06-04: Modal MAE reported only on f < f_Schroeder (Chunk P2-1, D18)

**Decision:** `aaf/eval/single_room_3d_eval.py` reports modal MAE on the band `[0, f_modal_cap]` where `f_modal_cap = clip(f_Schroeder, 100 Hz, 250 Hz)`. Above f_Schroeder, 3D modal density exceeds the RFFT resolution Δf = 0.5 Hz — the modal MAE metric becomes ill-defined (no way to confidently match peaks at >5 modes/Hz when the resolution is 0.5 Hz). Signal-level metrics (LSD, env corr) replace it above.

**Rationale:** For a representative 3D room (4.5, 4.0, 3.25), modal density:
- ≤ 250 Hz: ~30 distinct modes (manageable, MAE meaningful)
- ≤ 2 kHz: ~15,000 modes (peak density > 5 modes/Hz at high frequencies)

Phase 1's 2D version reported MAE up to f_max = 2000 Hz because 2D modal density grows linearly with f, not quadratically. In 3D this no longer works above the Schroeder frequency.

**Alternatives considered:** report MAE up to fixed 200 Hz regardless of f_Schroeder (rejected — different rooms have different f_Schroeder, so a fixed cap is misleading); report MAE on all bands but down-weight (rejected — adds methodological complexity for a metric that's still ill-defined at the top).

**Verification:** `aaf/eval/single_room_3d_eval.py:f_modal_cap_hz` is recorded in each `eval.json`; the per-room SUMMARY.md surfaces both `f_schroeder_hz` and `f_modal_cap_hz` so the modal MAE is interpretable in context.

**Revisit if:** an experiment needs modal-MAE-like accuracy on a specific narrow band above f_Schroeder (e.g., a Helmholtz absorber design tunes one mid-freq mode). Then introduce a narrow-band modal MAE on the requested band, not a global one.


---

## 2026-06-04: P2-1 amendment — vectorize 3D modal sum + lower MAX_ORDER_CAP 17 → 12 (Chunk P2-1, budget-check-driven)

**Decision:** Two interlocked changes after P2-1's first budget check showed per-room wall-clock 34 min on the largest room (way over the 10 min budget):

1. **`aaf/sim/analytical_modal_3d.py:modal_rir_3d`** rewritten as a single complex matmul:
   - Pack all eigenmode triples into numpy arrays once.
   - Compute `amp = phi_src[:, None] * phi_rx` shape `(n_modes, N_rx)`.
   - Compute `denom = k_m**2 - k**2 - 2j γ k_m / c + eps` shape `(n_modes, F_chunk)` per frequency chunk.
   - `H_acc[:, f_lo:f_hi] = amp.T @ (1.0 / denom)` — single complex BLAS matmul.
   - Frequency chunking targets ≤4 GB peak inv_denom allocation (chunk size auto-derived from n_modes).
2. **`aaf/sim/ism_3d.py:MAX_ORDER_CAP` lowered from 17 to 12** — (2N+1)³ image-source count drops from 42 875 to 15 625 (2.7×). IR still covers ~175 ms of decay (3.5× the 50 ms early-reflection window), still well above the analytical modal regime's resolution requirements.

**Rationale (with measurements):**
- The original per-mode Python loop was O(n_modes) iterations each allocating a `(N_rx, N_freq) = (512, 4097)` intermediate (~33 MB complex128). For the 111K-mode 6×5×4 m room this dominated at ~30 min.
- Matmul reformulation collapses the loop into BLAS: 23.6 s on the same room after the change. Smallest room: 5.7 s. Both well within budget.
- ISM at max_order=17 was a Plan-agent estimate; the empirical measurement showed it's also expensive (~10× the analytical when the analytical is vectorized). Lowering to 12 keeps the early-reflection envelope well-covered for de-risk overfit.

**Alternatives considered:**
- Keep max_order=17 + only vectorize analytical (rejected — analytical is now 23s but ISM at max_order=17 would dominate the next time we exercise it; D6's original 17 was set before we had measurements).
- Halve the receiver count (rejected — 8×8×8=512 is the spec-prescribed grid and not the bottleneck).
- Cap `f_max_modes` more aggressively (rejected — physically the right cap is fs/2 = 2 kHz which we already implicitly use).

**Verification:** `outputs/budget_check_3d/REPORT.md` (PASS), with per-step breakdown:
- Smallest (3.0, 3.0, 2.5): ISM 1.2 s, analytical 4.5 s, total 5.7 s, 19 978 modes.
- Largest (6.0, 5.0, 4.0): ISM 1.2 s, analytical 22.4 s, total 23.6 s, 103 611 modes.

`tests/test_eigenfrequencies_3d.py::test_modal_rir_3d_shape_and_rfft_symmetry` continues to pass after the vectorization (correctness preserved by construction — the matmul evaluates the same mathematical sum).

**Revisit if:** P2-2 or beyond needs ISM late-tail accuracy beyond ~175 ms (e.g., for late-corr or T30-from-EDC sub-metrics). At that point either:
- Raise max_order to 15 (2× wall, still tractable post-vectorization).
- Add ray-tracing fallback with a deterministic per-room seed.


---

## 2026-06-04: P2-2 design — `INR3D_AutoDecoder` with FiLM + latent jitter; d=16 (with d=32 hedge); linear geometry head (D19-D31)

**Decision**: P2-2 (multi-room 3D conditioning + zero-shot) uses the following Phase-1-validated recipe, adapted to 3D:

- **D19** Conditioning: FiLM at both sigma + signal branches. γ init=1, β init=0 (identity at construction). No LoRA in P2-2's first run; no concat-vs-FiLM-vs-LoRA sweep.
- **D20** Latent dimension `d=16` (up from Phase-1's 8). The 3D geometry manifold is parameterized by (L, W, H), so 16 gives ~5×-headroom over the intrinsic dim.
- **D21** Latent jitter σ=0.1 during training (off at val / zero-shot). Mirrors Phase-1 C2.
- **D22 / D31** Geometry head: linear `nn.Linear(latent_dim, 3) → (L, W, H)`, weight 0.1. Linear (not MLP-32) forces the latent to be directly readable as 3D geometry.
- **D23** HashGrid inherits P2-1's D10: `log2_hashmap_size=18, n_levels=16, per_level_scale=1.38`.
- **D24** Loss: 4 spectral (1, 1, 1, 0.1) + λ_latent·‖z‖² (1e-4) + 0.1·L1(predict_geom(z), [L, W, H]).
- **D25** Two-param-group Adam: network lr=2e-4, latents lr=1e-3 (DeepSDF convention). CosineAnnealingLR.
- **D26** n_iters=30K default; early-stop on 1% improvement over 2K window after 2K warmup. Checkpoint every 2500.
- **D27** Zero-shot: 8 obs receivers (corners of 8×8×8 grid → `OBS_INDICES_3D=[0,7,56,63,448,455,504,511]`); n_adapt_iters=2000; Adam lr=1e-2; random init; n_restarts=1 — matches Phase-1 variant B1.
- **D28** GPU pinning: training on tron (D13 lesson); zero-shot + probe on scavenger.
- **D29 (user-approved hedge)** Launch `M2_45rooms_d32` (d=32) in parallel with M1 from chunk start. Definitive d=16 vs d=32 comparison at closeout.
- **D30 (user-approved)** Zero-shot held-out evaluation uses the full 504 receivers (signal_level.py is BLAS-bound; ~1 s/room).

**Rationale**: P2-1 §11 recommended this configuration directly; Phase 1's C2 (FiLM + jitter) was the strongest in-distribution recipe with comparable zero-shot to other variants. The 3-output linear geometry head is the natural extension of Phase 1's linear L-head; P2-1's D14 (reject-near-cubic LHS draws) guarantees the 3 outputs are well-posed. The d=32 hedge is insurance against d=16 under-fitting the 3D manifold; given tron has the slots, the head-to-head answers Q12 cleanly at chunk closeout.

**Alternatives considered**: Concat conditioning (rejected — Phase-1 R0-R8's failure mode); MLP geometry head (rejected — would over-fit the latent away from linear readability); fewer iters / smaller batches (rejected — multi-room is harder than single-room, need headroom for early-stop to make a meaningful decision).

**Verification**: `tests/test_autodecoder_3d.py` covers (a) forward shape, (b) requires-z_s, (c) predict_geometry shape, (d) different z → different geom predictions (and after a few training steps, different forward outputs), (e) gradient flows back to latents + FiLM + geometry head, (f) latent jitter on at train and off at eval. `tests/test_zero_shot_3d.py` covers `OBS_INDICES_3D` correctness and grid-corner semantics. `tests/test_latent_probe_3d.py` covers `_r2_full_latent` and `_r2_per_pc` on synthetic ground truth.

**Revisit if**: M1 (d=16) and M2 (d=32) diverge in val LSD — Q12 closes with the empirical winner. If both fail to reach ≤ 2.5-2.7 dB in-distribution, escalate: widen `sigma_encoder_dim` 256 → 512, or bump `log2_hashmap_size` 18 → 20.


---

## 2026-06-06: P2-2.5 diagnostic — bottleneck is coverage/compute, not capacity (D32-D34)

**D32 — 10-room maximin diagnostic subset.** `aaf.data.sample_rooms_3d.select_diag_subset_maximin` greedily picks 10 rooms from the 45 LHS training rooms (greedy maximin in normalized [0,1]³, seeded at the box center). Runs A and C share this exact subset so their only difference is coverage/batch. Persisted to `configs/sweeps_3d/diag_10rooms.yaml`. Rationale: a fixed, auditable subset spanning the (L,W,H) box with comparable per-axis variation to the full 45, at 1/4.5 the cost — lets the capacity-vs-coverage question be answered cheaply.

**D33 — 2-GPU manual-all-reduce DDP in `multi_room_3d.py`.** Added an opt-in `--ddp` path (default off; single-GPU path byte-identical). Manual gradient all-reduce (SUM/world_size after all accum micro-backwards) rather than the `DistributedDataParallel` wrapper, because the trainer calls the model multiple times per step (per-room AABB grouping) before one backward — a DDP-wrapper failure mode. The averaging is exactly the single-GPU mean-batch gradient for both network params and the per-room latent embedding (disjoint random shards across ranks). Rank-0 owns val/ckpt/logging; the early-stop decision is broadcast so ranks break in lockstep. Verified: smoke test (loss 6.93→2.91, clean NCCL teardown) + a 2.0× measured speedup + a correctness cross-check where DDP-B and an independent single-GPU B agree to 0.00-0.03 dB at every matched iter (45K-50K). **Alternatives considered**: grad-accum tuning (rejected — compute-invariant, no speedup); rebuilding tcnn for arch 86 to get FullyFusedMLP on the A6000 (rejected — CLAUDE.md forbids rebuilding tinycudann); DDP wrapper (rejected — multiple-forward-per-step incompatibility). **Revisit if**: P2-3 needs >2× (use 3-4 ranks; the manual all-reduce scales).

**D34 — P2-2.5 verdict: scale compute on the full set, do NOT change the architecture.** A (10rm, eff-batch 16) = 1.84 dB ✅; C (10rm, eff-batch 64) ≈ 1.0 dB ✅ (the architecture's true 3D multi-room ceiling); B (45rm, eff-batch 32, 60K) = 2.61 dB, up from P2-2 M1's 6.16 dB purely from 8× coverage + more iters, at the 2.5 threshold and still descending. Conclusion: capacity/conditioning is not the bottleneck; per-iter coverage + total compute is. **P2-3**: apply C's recipe (eff-batch 64, n_pts 32) to all 45 rooms and/or extend to 80-100K iters on correctly-targeted A6000s (+DDP); expected ≤ 2.5 in-distribution, after which zero-shot becomes meaningful. Do not widen the decoder. **Revisit if**: scaling B's compute does NOT carry it below 2.5 — then capacity re-enters as a hypothesis (widen sigma_encoder_dim 256→512).

---

## 2026-06-07: P2-3 converged 4-GPU training + zero-shot self-diagnosis (D35-D37)

**D35 — 4-GPU DDP at fixed effective batch 64.** P2-3 trains the full 45-room set with Run C's validated recipe (eff-batch 64, n_pts 32) parallelized across 4 ranks: 4 × per-rank `batch_size=16`, `grad_accum_steps=2` → micro=8 (proven-fitting on a 48 GB A6000). LR unchanged (network 2e-4, latent 1e-3). This is data-parallelism **for wall-clock speed at a fixed effective batch**, NOT a larger effective batch — eff-batch stays 64 (Run-C's regime), so no LR re-tuning. **Do not let eff-batch drift to 128.** The manual all-reduce (D33) is generic in `world_size`; 4-rank works by `srun --ntasks=4 --gres=gpu:rtxa6000:4`. **Wall-clock note**: holding eff-batch at 64 means per-iter throughput ≈ the 2-GPU eff-32 run (~0.64 it/s); the 4× speedup is vs Run C's *single-GPU* eff-64 (0.157 it/s), NOT vs Run B. **Alternatives considered**: more ranks at fixed eff-batch (rejected — qos=high caps at 4 GPU); larger micro to cut accum (rejected — compute-invariant, no speedup).

**D36 — iteration budget 60K (user cap), early-stop relaxed to 0.3%/10K.** P2-2.5's slope rule (B dropping 0.16 dB/10K ≥ 0.1 at 60K) pointed to 100K (~43 h, ~2 days). User capped at **60K (~26 h, ~1 day)** for the meeting timeline. At eff-batch 64, 60K covers 2× the data B's eff-32 60K did → expected ~2.2-2.4 dB, likely clearing 2.5 with modest margin. Checkpoint every 5K, val every 2K, auto-resume (the trainer's `_maybe_resume` loads the latest `ckpt_iter*.pt`, so re-submitting the same command continues past the 24 h qos cap). early_stop_warmup=10K, patience=10K, min_rel_improvement=0.003. **Revisit if**: 60K lands just above 2.5 or is still descending steeply → more iters (toward 100K) is the lever, a clean P2-4 follow-up.

**D37 — 3-way zero-shot self-diagnosis rule** (`aaf/eval/zero_shot_diagnosis.py`, the load-bearing P2-3 verdict). Gated on in-distribution fit: zero-shot is interpretable only if the model cleared ≤2.5 dB val LSD. Then per test room, using mag-corr (≥0.9 = good) and geometry-placement (geometry head on z* vs true L,W,H; ≤0.3 m/axis = placed) + manifold distance (‖z* − nearest train latent‖): (1) good fit + good mag-corr → **success** (method works); (2) good fit + poor mag-corr + geometry **misplaced** → **manifold-coverage** problem → P2-4 more training rooms; (3) good fit + poor mag-corr + geometry **placed** → **decoder-at-interpolated-latent** problem → investigate decoder smoothness. Makes P2-3 produce an actionable answer regardless of the zero-shot outcome (mirrors P2-2.5's self-diagnosis design). Unit-tested in `tests/test_zero_shot_diagnosis.py`; retroactively classifies P2-2 M1 (6.16 dB) as "precondition unmet", consistent with the P2-2.5 finding.


---

## 2026-06-09: P2-3 verdict — in-distribution solved; zero-shot is coverage-limited (D38)

**D38 — P2-3 zero-shot fails on a converged 45-room model, and the cause is training-set coverage, not the model or the test-time procedure.** Training succeeded (4-GPU DDP, eff-batch 64, 60K iters → in-distribution val LSD **2.169 dB**, clearing ≤2.5). Zero-shot on the 8 maximin test rooms: **0/8 reach mag corr ≥0.9 (got 0.20-0.28)**. Three-layer diagnosis: (1) the bottleneck is the test-time latent z* — the optimized z* cannot fit even the 8 *observed* receivers (obs_lsd≈held_lsd≈7 dB); (2) the converged P3 zero-shot (0.20-0.28) is **worse** than the unconverged P2-2 M1 (0.52-0.59), because a sharp decoder punishes a wrong z* while a blurry one fakes an "average room" — so representation + rendering are fine, finding the right latent for an unseen room is the problem; (3) a manifold-anchored adaptation sweep (`--z_init mean`, λ ∈ {1e-4,1e-2,1e-1}) leaves mag corr **invariant (~0.2-0.28)** at both 10 (Run C) and 45 (P3) rooms — there is no good latent to find; the model memorizes its training rooms and does not interpolate. Denser coverage (45 vs 10 rooms) improved z* *placement* (geom-err 0.5-2.6 m vs 4.2; z*norm 7-10 vs 11-13) but not enough. **P2-4 levers**: (a) scale the training set (45 → ~150-300 rooms); (b) condition the decoder explicitly on the *known* test (L,W,H) so zero-shot needs no z* search (the current setup never uses known geometry — potentially decisive). **Rejected for P2-4**: test-time-procedure fixes (proven ineffective, D38 sweep) and capacity widening (in-distribution solved, D34). **Revisit if**: scaling rooms does not move the zero-shot mag-corr-vs-room-count curve — then the decoder's interpolation capacity (architecture) re-enters.


---

## 2026-06-30: P2-4 coverage-density scaling — nested sets, frozen test set, frozen recipe (D39-D41)

**D39 — nested training supersets by maximin augmentation, seeded by the existing 45.** The spec requires `existing 45 ⊂ 90 ⊂ 150 ⊂ 250` with room count as the only independent variable. True nested-LHD cannot preserve an arbitrary pre-existing design, so `aaf.data.sample_rooms_3d.sample_nested_supersets(base, targets=(90,150,250), seed=7)` starts from the fixed existing 45 (LHS seed 42, asserted byte-equal to `train_rooms.yaml`) and greedily adds the candidate that maximizes min-distance (normalized [0,1]³) to the current set, drawn from a fixed-seed Sobol pool (16384) with the same `|L−W| ≥ 0.05` near-cubic rejection, snapshotting at 90/150/250. Space-filling, reproducible, strictly nested. **density-45 = the existing P3 model (no retrain).** Within-set mean NN (normalized) is non-monotone across snapshots (45:0.167 → 90:0.207 → 150:0.188 → 250:0.158) — an expected maximin-augmentation artifact (early additions fill the largest isolated gaps); it is irrelevant because the x-axis is the *test→train* NN distance, which IS monotone (below). Verified by `scripts/build_nested_rooms.py` (asserts counts, nesting, base-preserved) + `tests/test_nested_rooms.py` (7/7). **Alternatives considered**: re-draw four independent LHDs (rejected — breaks nesting, confounds count with sampling); nested-LHD from scratch (rejected — cannot contain the already-trained 45). **Revisit if**: a future chunk needs >250 — extend the same greedy augmentation from the 250 snapshot.

**D40 — frozen interior test set (15 rooms), reused by P3-1; built once, never modified.** `sample_interior_test_rooms(hull_rooms=45, exclude_rooms=250, n=15, seed=2024, min_train_dist=0.04)` draws Sobol candidates strictly inside the 45-room convex hull (scipy `Delaunay.find_simplex ≥ 0`, normalized coords) so the comparison is **interpolative at every density**, ≥0.04 (normalized) from all 250 training points so they are genuine test rooms, then maximin-spread (seed = candidate nearest the box centre). Written to `configs/sweeps_3d/test_rooms_interior_frozen.yaml` with `frozen_note: "FROZEN — reused across P2-4 and P3-1; do NOT modify"`; each room's NN-distance to every training set is precomputed in `outputs/coverage_curve/test_nn_distances.json` (the x-axis mechanism). Mean test→train NN distance (raw m) **decreases monotonically 0.260 → 0.236 → 0.200 → 0.178** across 45/90/150/250 — exactly the coverage axis the curve plots. This test set is **load-bearing for P3-1** (geometry conditioning is measured on the same rooms at the same 4 density baselines). **Why interior, not the old maximin test set**: the P2-3/P2-3.5 maximin test rooms sat far outside dense coverage (NN 0.61 m), so they were untestable as a *density* probe; interior rooms get genuinely surrounded as density rises. Consequence (measured): the density-45 baseline on this fair interior set is **0.273 full / 0.409 modal** — interior NEW geometries are still in the "untrained, flat ~0.27" regime at 45 rooms, consistent with P2-3.5's coverage finding; the open question P2-4 answers is whether 90/150/250 climbs toward the 0.89 LOO ceiling. **Revisit if**: never for P2-4/P3-1 (frozen). A different generalization regime (extrapolative, outside the hull) would need its own separate set.

**D41 — recipe frozen; only room count varies; leaner per-density iteration budget (user-approved).** Every P2-4 density reuses the P3 recipe byte-for-byte (`INR3D_AutoDecoder`, latent_dim 16, hashgrid 18/16/1.38, FiLM conditioning, latent_jitter 0.1, linear geometry head true/0.1, eff-batch 64 = 4×16 micro-8 grad_accum 2, 4-GPU rtxa6000 DDP, LR 2e-4/1e-3, n_pts 32, val 2K / ckpt 5K, early-stop warmup 10K / patience 10K / min_rel 0.003). The ONLY change per density is `rooms_yaml` + `n_iters`. **Iteration budget (user chose "leaner iters for a faster curve")**: 45→60K (reuse P3), **90→60K, 150→70K, 250→85K** (~4 days sequential, each density owning the 4 GPUs; afterany continuations past the 24h qos cap). **Convergence caveat (must be reported)**: per-room sample exposure drops as rooms scale (85K iters / 250 rooms ≪ 60K / 45 rooms), so larger models may be less converged in-distribution. The **per-density in-distribution val LSD is the control** — if it holds ~2.2 dB, the zero-shot delta is pure coverage signal; if it degrades materially, that density's zero-shot is reported as a **lower bound (undertrained), NOT re-tuned** (re-tuning would confound the density axis). **Rejected**: equal-exposure budgets (rejected — would make total compute scale with rooms, blowing the timeline; the user explicitly chose the faster curve with the caveat); any architecture/capacity/conditioning change (rejected — that is P3-1, and isolating density is the entire point of P2-4). **Revisit if**: the zero-shot curve is flat/muddy AND in-dist degraded at high density → the conclusion is "needs more iters at high density," a clean re-run, NOT "coverage doesn't help."


---

## 2026-07-06: P2-4b confound check — coverage confirmed at matched convergence, but ~⅔ of the raw curve was confound (D42)

**D42 — P2-4b: hold convergence constant by under-training 45 to 250's capacity plateau (~4.3 dB), not by training 250 up to 2.17 dB (infeasible); verdict CONFIRMED-but-inflated.** The P2-4 curve (D41) confounded coverage with convergence — the fixed lean budget meant in-dist val LSD degraded with density (45→2.17, 250→4.30 dB), and P2-3 showed under-training *inflates* zero-shot magnitude correlation (blur → "safe average room"). **Empirical finding that forced the design**: density_250 is **capacity-plateaued at ~4.3 dB** (constant LR, slope ~0.008 dB/1k, bouncing) — at fixed latent_dim 16 / hashgrid 18/16/1.38 it **cannot** converge to 45's 2.17 dB (would need ~250K+ iters and asymptotes higher). So training 250 *up* to match is infeasible; instead we match at ~4.3 dB by **under-training 45 down** (fresh frozen-recipe retrain `density_45_conv`, evaluated at the ckpt nearest 4.30 dB → 11K @ 4.333 dB) + a blur sweep (45 at 4.55/4.33/3.80/3.52 dB). The 250 endpoint reuses density_250@85K. **User-approved minimal-compute path** (AskUserQuestion, "Match at 4.3, minimal compute"); the literal `density_250_conv` deliverable is superseded. One 4-GPU retrain + 4 scavenger evals ≈ ~40 GPU-h. **Full metric suite mandatory** (mag full/modal, held-out LSD, phase-mw, RIR, modal placement, Schroeder split — the confound implicates the soft metric, so mag corr alone is disallowed).

**Verdict (`outputs/coverage_curve/CONFOUND_CHECK.md`): coverage effect CONFIRMED at matched convergence.** At ~4.3 dB, 250 beats 45 across the suite — held-out LSD (+0.21 full / +0.59 modal dB), phase (+0.131), RIR (+0.133), plus mag (+0.060 full / +0.113 modal); because both sides are at matched convergence, blur is equalized on both, so these deltas isolate coverage. (Modal *peak placement* — recall/MAE — is a wash, weak metric.) **But the raw P2-4 climb was mostly confound**: decomposing the raw full-band gap (45@2.17→250@4.30 = +0.188) at the matched midpoint gives **blur +0.128 (68%) + coverage +0.060 (32%)**; modal (+0.402) = blur 72% + coverage 28%. **So ~⅔ of the raw curve was blur, not coverage.** Blur also improves held-out LSD (45@2.17→4.33: LSD 7.70→6.25), so LSD is not confound-proof either; phase/RIR carry the cleanest signal. **Corrects the P2-4 SCALING.md claim** that the confound made the read "conservative" — it was **inflated**. **Bug caught + fixed**: the trainer pruned to the last 3 ckpts (`ckpts[:-3].unlink`), deleting the 4.3 dB matched checkpoint mid-run and collapsing the first eval set onto ~3.6 dB ckpts; added config `ckpt_keep_last` (default 3, backward-compatible; density_45_conv=30) and redid the run clean. **Implication for P3-1**: densification is a real but modest lever *and* carries a capacity penalty (250 can't converge at fixed capacity) → explicit (L,W,H) conditioning is doubly motivated; P3-1 benchmarks against the **matched-convergence 250** point (0.461/0.811, phase 0.348, RIR 0.421 @4.30 dB) and should itself report at matched convergence. **Revisit if**: P3-1 wants a fully-converged 250 baseline → needs a capacity bump (latent_dim / hashmap), which re-opens the capacity question D34 deferred.


---

## 2026-07-14: P3-1 geometry-conditioned editing — arm definitions, band scoping, G+ injection (D43)

**D43 — P3-1: three conditioning arms (L / G / G+) under one band-limited (0–300 Hz) protocol; backbone + renderer held byte-identical, only the conditioning path varies.** Band scoping: all supervision, validation, and eval are masked to bins 0–600 (`band_indices(fs=4096, n_freq=4097, 0, 300) = (0, 601)`, slice `[lo:hi]`), which contains the complete modal regime for the room family (Schroeder ≈173–298 Hz); out-of-band bins receive exactly zero gradient (`tests/test_band_mask.py`). **Arm L (latent)** = the Phase-2 auto-decoder (`nn.Embedding(45,16)` + latent jitter 0.1 + linear geometry head + `lambda_latent` reg), zero-shot via RBF (L,W,H)→latent lookup. **Arm G (geom_fourier)** = 48-d Fourier features of normalized (L,W,H) → FiLM; no latent table, no jitter, no geometry head. **Arm G+ (eigen)** = 64-d sorted analytic eigenfrequency vector (`/300`) → FiLM, plus a per-bin resonance map R (Gaussian bumps σ=2 Hz at analytic eigenfreqs ≤310 Hz, max-normalized) modulating the signal-branch output as `signal_complex · (1 + w·R)` with a single learnable scalar w, **zero-init** (so at t=0 G+ ≡ G structurally; R is 0 above bin 600, preserving DC/Nyquist realness). Recipe asymmetries (latent table/jitter/geometry head/reg only in L; none in G/G+) are deliberate — geometry is the input for G/G+, and jittering exact analytic conditioning would decouple it from the target. Implementation is additive and defaults to Phase-2 behavior (`cond_source=latent`, `band_max_hz=None` → byte-identical to P3). New module `aaf/models/conditioning.py` (tcnn-free) is the single shared conditioning entry point for trainer and eval. Modal placement reported at both f_max=300 (headline, matches training band) and f_max=250 (apples-to-apples with P2-4b's `f_modal_cap≤250`). **Status at time of writing (data-only, no verdict)**: L trained 40K iters (in-dist val LSD 0.72 dB), G 28K (1.14 dB), G+ still training (iter 10,900, 2.02 dB). Zero-shot recall@250: L 0.104, G 0.101, G+ @2000/6000/11000 = 0.164/0.114/0.089; G+ learned w by ckpt 0.03/0.14/0.35/0.41/0.34 (1K/2K/4K/6K/11K). Numbers in `outputs/p3_1/HEADTOHEAD.md` + `tasks/CHUNK_P3_1_RESULTS.md`. **Not yet run**: G+ to convergence; matched-convergence comparison (dense L checkpoints removed under the disk quota); disentanglement eval. **Revisit / open**: whether a matched-convergence comparison changes the arm ordering (Q14, OPEN_QUESTIONS.md).


---

## 2026-08-12: P3-1 paused; repo consolidated to nexus-projects; artifact-tracking policy tightened (D44)

**D44 — P3-1 paused mid-flight (user decision to sidestep for now); repo working copy consolidated back to `/fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields`; `.gitignore` tightened so only reports + source are tracked.** (a) **Pause**: P3-1 is not concluded — training advanced past the last eval (Arm G → iter 60K/0.59 dB, Arm G+ → iter 16K/1.61 dB; an experimental `arm_Gplus_fast` native-NCCL hedge also exists), but the head-to-head evals were not refreshed on those checkpoints, and the matched-convergence and disentanglement experiments were never run — so the central question stays open (see CONTEXT_FOR_MANAGER.md status block + Q14). No science changed; work simply stopped. (b) **Repo location**: the project had been temporarily split to `/fs/nexus-scratch/htakawal/adaptable-acoustic-fields` under a projects-disk quota crunch; a failed cross-device `mv` left the tree split (git repo stranded in scratch). Reconsolidated by `rsync` (copy → git-fsck + byte-parity verify → delete source), so nexus-projects is once again the single canonical working copy. The conda env stays at `/fs/nexus-scratch/htakawal/miniconda3/envs/aaf` (unchanged, outside the moved dir). (c) **Artifact-tracking policy**: previously several per-room zero-shot figure/metric dumps, per-run `scalars.json`/`train_meta.json`, and all checkpoints under `outputs/` were either tracked-but-stale or untracked-and-noisy. Tightened `.gitignore` to ignore, everywhere under `outputs/`, all `*.pt` / `tb/` / `*.tfevents.*`, plus the per-room lookup/zero-shot dumps of completed chunks (coverage_curve, multi_room_3d) and the superseded zero-shot scratch dirs (runC_zeroshot_*, p3_zeroshot_anchored) and the P3-1 training-run dirs. **Only human-readable deliverables stay tracked**: the `*.md` reports (HEADTOHEAD, CHUNK_*_RESULTS, SUMMARY, SCALING, CONFOUND_CHECK, …), headline `summary.json`, and the small edit-sweep figures. Removed 60 stale tracked `H_pred_all.pt`/`z_star.pt` blobs under `inner_loop_experiments/`. **Rule of thumb going forward**: commit reports + source; never commit checkpoints, tb logs, or per-room dumps (regenerable). **Revisit**: on P3-1 resume, re-eval arms at converged/matched checkpoints (D43 open question).


---

## 2026-08-13: P3-2 2D material editing — materials, wall convention, max_order, conditioning, gate rule, scoping (D44-D48)

**D44 — Material presets, the canonical wall order, and the held-out combinations.** Flat (frequency-independent) energy absorption: **M0 painted brick 0.15 (baseline), M1 concrete 0.05, M2 heavy curtain 0.50, M3 absorber panel 0.70**. M1 sits *below* baseline by design so the edit axis is bidirectional — concrete sharpens a wall's mode family while curtain/absorber damp it — and both directions are evaluated. Exactly one wall differs from baseline in any edited configuration (single-wall-edit scope this chunk; multi-wall is future work). **Wall order is `WALLS_2D = (west, east, south, north) = (x=0, x=L, y=0, y=W)`, defined once in the dependency-free `aaf/walls.py`** and imported by the simulator, the filename builder, the config enumerator, the conditioning encoder, the evaluation and the demo CLI — a wall-order mismatch between any two of them is the single most catastrophic silent bug available in this chunk (it yields a model that is confidently wrong and no aggregate metric detects it), so it is made structurally impossible rather than merely unlikely. The order was **verified against pyroomacoustics 0.9.0**, both by `ShoeBox.wall_names` and by an image-lattice probe: absorbing only the west wall puts damping `sqrt(1-alpha)` on exactly the image mirrored across `x=0` and leaves every other first-order image undamped (gate assert G0.2; `tests/test_wall_convention.py`). That probe simultaneously establishes that pra's per-bounce factor is the **pressure** reflection coefficient, i.e. `alpha` is an ENERGY absorption. **Held-out combinations, excluded from all training geometries: (west, M2) and (north, M3).** Coverage after the holdout: west trains {M1,M3}, north {M1,M2}, east and south all three; M1 appears on 4 walls, M2 on 3, M3 on 3 (so every wall keeps >=2 materials and every material >=3 walls). These two are not arbitrary — **each has a trained opposite-wall twin**: (east,M2) and (south,M3) are trained, and since west/east both span W while south/north both span L, the twin has **identical mean absorption and identical T60** and differs only in *where* the absorber sits. A model that learned a scalar effective absorption therefore cannot transfer, which is what makes split (iii) a genuine test of wall identity rather than of overall absorptivity. Verified empirically: west-M3 and east-M3 have equal `alpha_eff` but different fields. **Revisit if** multi-wall edits are added, in which case the single-edit invariant and the twin argument both need restating.

**D45 — `max_order = 60`, fixed, not the auto rule.** The 2D auto rule (`ceil(c*4*T60/min(L,W))`) gives ~390 for a typical room, which is expensive and unnecessary; the 3D chunks used 12, which is far too few for a 2-s tail. 60 was **measured, not assumed**: on the worst case (west=M1, the least absorptive and therefore longest-tailed configuration), going 60 -> 120 changes the x-axial -3 dB bandwidth by **0.0%** and the peak level by **0.05 dB** (gate assert G0.4). It is retained as a standing gate assert rather than a comment, so a future geometry range that breaks the convergence is caught before a dataset is generated. **Revisit if** the geometry family grows or a material with alpha < 0.05 is added.

**D46 — Conditioning: 6 physical parameters -> 64 Fourier features -> FiLM, no latent table.** `u = [(L-3)/3, (W-3)/2, a_w/0.7, a_e/0.7, a_s/0.7, a_n/0.7]`; geometry dims get octaves k=0..7 (16 features each), absorption dims k=0..3 (8 features each), total **64**, block layout `[0:16] L | [16:32] W | [32:40] a_west | [40:48] a_east | [48:56] a_south | [56:64] a_north` (offsets asserted in tests). This reuses the one P3-1 result that worked — Arm G's physical-parameter->FiLM path matched Arm L's fidelity at 2x the edit-tracking accuracy — while avoiding Arm G+'s failure mode: there, a *redundant* conditioning path let the network bypass the physics, whereas here the conditioning is the **only** source of material information. The absorption dims get half the octaves because alpha responses are smooth and the four trained levels are discrete, so high octaves buy nothing and invite aliasing between materials. The arm is named **`geom_alpha_fourier`**, deliberately not reusing `geom_fourier`, which already means the 48-d (L,W,H) vector in every P3-1 config and checkpoint `train_meta.json`; reusing it would make checkpoint metadata ambiguous across chunks. Implementation lives in a new `aaf/models/conditioning_2d.py` (imports only math/torch/`aaf.walls`) — `aaf/models/conditioning.py` is left byte-for-byte untouched because P3-1 reproducibility depends on it and its `build_cond_vector(cond_source, L, W, H, ...)` is called positionally. `INR2D_AutoDecoder` gained additive `cond_source`/`cond_dim`/`l_head_out_dim`; **back-compat is bit-identical, not merely shape-compatible** — `cond_dim` defaults to `latent_dim`, so every pre-P3-2 config builds Linears with the same `in_features`, drawing the same values from the global RNG (verified on GPU by state-dict equality). The port also moved the `concat` paths onto `cond_dim`, fixing a latent bug the 3D version left behind. **Revisit if** a 5th material or continuous-alpha training is introduced: the `/0.7` normalization then needs restating (identity normalization over [0,1] would be the more future-proof choice, since it never remaps already-trained points).

**D47 — The blocking gate tests selectivity on BANDWIDTH, not on peak level.** The chunk spec asked for a signature in which the non-edited mode family changes "by an order of magnitude less". Measured on ISM ground truth, that is **true for bandwidth (~29-50:1) and false for peak level (~4.4:1)** — the level of every mode drops somewhat whenever any wall is made absorbent, because the broadband background falls with it. Applying a 10:1 rule to level would therefore have **STOPped the chunk on correct physics**. The gate accordingly runs its selectivity test on bandwidth (threshold 5:1 against a measured 29:1), checks level only for sign and magnitude, and gives the M1 (concrete) direction its own ~9x looser thresholds because that effect is genuinely smaller than M3's. The verdict is three-way rather than binary: a confidence interval clearly above 5 is a PASS (ray regime); an interval straddling 2 is **PASS-WITH-AMENDMENT** — a 2:1 world is still a real, monotone, wall-specific, learnable signal, so it amends the claim rather than blocking it; only "no directional signal" or a wrong sign is a STOP. **Gate result: PASS**, selectivity 29.1 (95% CI [20.0, 39.3]), with T1 direction, T2 selectivity, T3 bidirectionality and monotonicity, T4 orthogonal-wall flip and T5 theory-fit all passing 8/8 across 2 rooms x 4 walls. Full numbers in `outputs/p3_2/SIM_VALIDATION.md`.

**D48 — Scoping: the ground truth obeys a RAY absorption law, and every claim is scoped to it.** Two damping laws are in play and they disagree by ~25x in selectivity. **ISM-ray** (what pyroomacoustics computes) has an angle-independent real reflection coefficient, so `gamma(n,m) = c*[cos(theta_x)*kappa_x + cos(theta_y)*kappa_y]`: a purely axial mode has `cos(theta)=0` on the orthogonal pair **exactly**, hence infinite selectivity in the model and no grazing-incidence absorption at all. **Kuttruff** (a locally-reacting impedance wall, i.e. real rooms) gives `gamma = (c/8)*[(a_w+a_e)*eps_n/L + (a_s+a_n)*eps_m/W]` -> exactly **2:1** selectivity with **no invariant family**, because in a rigid rectangle every wall sits at a pressure antinode of every mode. Both are implemented in `analytical_modal_2d.modal_damping_2d(..., model=...)` and validated against their known limits (Kuttruff reduces to Sabine exactly; the isotropic average of ISM-ray equals 2D Eyring exactly). The gate discriminates them and the data is decisive: ISM-ray fits the measured bandwidths with **R^2 = 0.998** (calibrated `BW = 0.302 + 1.661*gamma/pi`) versus 0.982 for Kuttruff, **dAIC = 73** favouring ISM-ray. **Consequence (CORRECTED in P3-2b, see D52):** the block-diagonal STRUCTURE is genuine physics -- an axial family's damping is dominated by the two walls perpendicular to it, and grazing-incidence absorption on a locally-reacting surface tends to zero as theta->90 deg. What is set by the simulator is the MAGNITUDE of the selectivity ratio (~29:1 here); a locally-reacting wall would show the same block-diagonal structure at a smaller ratio. The supportable claim is therefore **"the representation learns whatever per-wall absorption law it is trained on"** -- not the earlier, imprecise phrasing that the structure itself was a simulator artifact. In a real room the same edit would produce roughly a 2:1 family separation with every family affected. Figure B overlays both laws against the measurement so the scoping is visible rather than buried. **Revisit if** the project moves to measured RIRs or to a wave-based solver, where the Kuttruff regime should appear and the selectivity claim must be restated (and would then be a genuinely harder learning problem).


---

## 2026-08-13: P3-2 zero-shot eval — paired edit deltas, four mandatory controls, split-(iii) reporting (D49)

**D49 — `aaf/eval/p3_2_eval.py`: every edit metric is a PAIRED delta, the headline observable is bandwidth, and no edit claim is reported without C1-C4.** (a) **Pairing.** Each edited config is scored as `|delta_pred - delta_gt|` where `delta_pred = measure(pred_edited) - measure(pred_baseline_of_the_same_geometry)` and likewise for GT — never as an absolute reconstruction. The -3 dB estimator carries a bias that depends on the geometry and on the mode's own frequency, and that bias cancels to first order in the difference; the pairing also moves the target from "reconstruct the room" to "reproduce the edit", which is the actual claim. Both streams use the SAME per-mode walk caps (`caps_from_predicted_bw` on the ISM-ray `modal_damping_2d` widths for that config's own alphas), so prediction and ground truth are measured by an identical estimator. Modes are capped at 200 Hz (`F_MAX_PROJECTION_HZ`) and gated on the projection's excitation mask; a cell counts only when all four measurements (pred/GT x edit/baseline) are valid, so a cell cannot be manufactured by one stream failing. (b) **`theory_d_bw` is calibrated.** The gate's T5 fit `BW = 0.302 + 1.661*gamma/pi` is applied as a slope only — the intercept is an estimator offset and cancels in a delta. (c) **Controls, all four mandatory.** *C1 null model*: the model's own BASELINE render scored against the EDITED ground truth, reported as `edit_gain = LSD(null)/LSD(model)`; needed because the M1 edit is only ~0.3 dB of in-band LSD, well under the in-distribution val LSD, so a model that ignored the material channel entirely could still post a respectable absolute number. Its `E_BW` is exactly `mean|delta_gt|`, which doubles as the reported GT effect size. *C2 floor*: within-family mode-to-mode std of the measured baseline bandwidth, **axial families only** — under the ISM-ray law every mode of an axial family shares one damping rate, so the spread is pure estimator noise (measured 0.040 Hz; tangential is 0.27 Hz and is genuinely mode-dependent, reported separately and explicitly not a floor). *C3 conditioning identity*: "wall k set to M0" must be bit-identical to the baseline in both the conditioning vector and the render; this requires **`renderer.eval()`**, not just `model.eval()` — `FreqRenderer2D` jitters ray azimuths while `self.training`, and the trainer never puts the renderer in eval mode, so a naive port of the training render path fails C3 for reasons unrelated to conditioning. *C4 wall identity*: each held-out combo against its trained opposite-wall twin (`WALL_TWIN`), scored as `wall_asymmetry = r(pred, own GT) - r(pred, twin GT)` on per-receiver in-band dB maps relative to baseline; a model that collapsed the four absorptions to a scalar `alpha_eff` renders the twin's field for both and scores ~0. (d) **Reporting.** The two held-out combos are reported **separately as well as pooled** — (west,M2) tests material-value transfer onto a seen wall and (north,M3) tests wall transfer of a seen material, and pooling hides which one works; the first eval already shows them diverging (at iter 6K: west-M2 slope 0.104, C4 asymmetry +0.297; north-M3 slope 0.011, C4 asymmetry -0.048). The (i)-vs-(iii) gap is reported in Hz **and** as a percent of the GT effect size, because a raw-Hz gap is unreadable without knowing how big the effect was. `edit_bw_slope` is reported alongside `edit_bw_pearson` for the same reason: r=0.95 with slope 0.3 means "right direction, 3x under-predicted", and only the slope says so. **Alternatives rejected**: unpaired absolute measurement (rejected — dominated by reconstruction error, and estimator bias does not cancel); `caps_from_mode_spacing` (rejected — would reject exactly the broad absorber modes the chunk is about); level as the headline (rejected per D47, ~4:1 vs ~29:1 on bandwidth); pooling the held-out combos (rejected, see above). **Convention**: `_pearson` scores a flat vector 0.0 rather than nan, because a model that ignores the material channel emits an identically-zero edit map — that is a result, not missing data, and nan would let a dead model read as "not measured". **Scoping (inherits D48)**: the ~29:1 selectivity this eval scores against is a property of the ISM simulator (angle-independent reflection, no grazing-incidence absorption); real locally-reacting walls follow Kuttruff (~2:1, no invariant family), so every number here supports "the model learns the SIMULATOR's per-wall law". **Revisit if** multi-wall edits arrive (the twin argument and the single-baseline pairing both need restating) or if the projection basis is widened past 8x8 receivers (the 200 Hz cap is a conditioning limit, not a physics one).

---

## 2026-08-13: P3-2 live demo — paired mode intersection, shared walk caps, far-corner default receiver (D50)

**D50 — `scripts/demo_edit_2d.py` measures the baseline/edited pair with one identical estimator, and defaults to the far-corner receiver.** (a) **Paired mode intersection.** Per-family bandwidth deltas average only the modes whose -3 dB width is resolvable in **both** the baseline and the edited render. Taking "the first 3 valid modes" independently per config — as the physics gate does, where it is harmless because measured and predicted widths agree — differences two *different* mode sets and manufactures a bandwidth change out of bookkeeping: measured on the iter-6K checkpoint, the unpaired rule reported `dBW_x = +4.91 Hz` where the paired rule reports `+0.36 Hz`, because modes (2,0) and (3,0) were resolvable only in the edited config. The printed table lists the exact `(n_x, n_y)` used, so the pairing is visible rather than implied. (b) **One set of walk caps for both configs.** `caps_from_predicted_bw` is evaluated on the per-mode **maximum** of the two configs' ISM-ray predicted widths, not on each config's own alphas. Config-specific caps make the *cap itself* differ between baseline and edited, so a mode can be rejected as unresolvable in one and accepted in the other, and the paired difference then partly measures the cap rule. A shared cap makes every mode resolvable in both or in neither. Verified neutral on well-behaved data: on ISM ground truth the shared-cap numbers are identical to the gate's (`dBW_x = +5.350`, selectivity 35.7:1 at 4.51x4.00, west->M2). Note this is a different axis from D49's pred/GT cap sharing and does not conflict with it. (c) **Default receiver is the far corner (index 63), not the centre.** On the 8x8 grid the near-centre receiver sits close to a node of the odd axial modes (mode (1,0) is ~14 dB down there) and receiver 0 is dominated by direct sound, so both make the single-receiver spectrum panel misleadingly flat; `--receiver center` and an explicit index remain available and are labelled in the panel title. The *quantitative* summary never depends on this choice — it comes from the 64-receiver modal projection — so the receiver is a presentation choice only. (d) **`renderer.eval()` as well as `model.eval()`**, for the reason recorded in D49 C3: the azimuth jitter lives on the renderer, and an `nn.Module` is constructed in training mode, so rendering without it is not reproducible between runs. **Scoping (inherits D48)**: the demo prints the ISM-ray and Kuttruff predictions side by side and states in both stdout and the figure caption that the ~29:1 selectivity is a property of the simulator, so the claim it supports is "the model learns the simulator's per-wall law". **Revisit if** the demo is pointed at multi-wall edits (the single-baseline pairing needs restating) or at a receiver grid finer than 8x8 (the 200 Hz projection cap is a conditioning limit that would move).


---

## 2026-08-14: P3-2b — kappa-scaled theory, continuous-m sampling, and the attribution result (D51-D52)

**D51 — the acceptance criterion compares against a kappa-SCALED theoretical slope, decided before any P3-2b result existed.** The P3-2b spec specified `a_theory = c/(4 pi D)` (6.066 Hz per unit m at L=4.5). That is the RAW Lorentzian slope. Our estimator measures a **calibrated** -3 dB width carrying the P3-2 gate's T5 fit `BW = 0.302 + 1.6608*(gamma/pi)`; the intercept cancels in a paired delta but **the slope does not**. Correct value: `a_theory = kappa*c/(4 pi D)` = 10.073 Hz per unit m, with kappa = 1.6607564051417665 frozen from the P3-2 gate and hash-pinned. **Confirmed independently after the fact**: fitting the 240 simulated ground-truth m-response points across 12 (geometry, wall) cells gives mean rho = 1.0050, range [0.947, 1.084], every r^2 >= 0.9996; against raw theory the same GT fits read rho = 1.669. The failure mode this averted was NOT a false negative but a **false positive**: on a mid-training checkpoint the model read rho = 0.590 kappa-scaled and **0.980 raw**, i.e. the raw formula would have made a clearly unconverged model look like a near-perfect physics match. Both numbers are reported (`rho_vs_raw_theory_median`). Thresholds live in a frozen mapping whose sha (`a8479c5e1dcc...`) is written into `verdict.json` and pinned by a test, so they cannot be softened after seeing results. **Revisit if** the estimator's calibration changes (a new gate fit) — kappa must be re-derived, not carried over.

**D52 — P3-2's compositional failure was caused by the TRAINING DISTRIBUTION, not by the fit and not mainly by the conditioning parameterization; adopt continuous sampling in m, with the m-coordinate as the recommended encoder.** The 4-arm ablation isolates it: A (P3-2's 440 preset configs) and B (960 continuous configs) share an encoder, a renderer (`n_pts_per_ray=64`) and an eval, and differ ONLY in the sampled absorptions — S2 edit slope goes **0.153 -> 1.147**, Pearson 0.499 -> 0.871, edit_gain 0.868 -> 1.084. P3-2 sampled alpha at 3 preset values on single walls, i.e. **11 distinct alpha-vectors in a 4-D space**, and the model memorized (wall, alpha) pairs. Three further findings: (a) the **m-coordinate is not necessary but is materially better calibrated** — C's slope 0.959 vs B's 1.147 and rho 0.971 vs 1.045, with **rho = 0.99991 outside the held-out slab**, so it recovers the physical law essentially exactly while B overshoots ~15%; (b) **multi-wall training is not necessary** (arm D passes on single-wall configs alone, at a cost in in-distribution fit); (c) **fit quality is ANTI-correlated with edit transfer** — arm A has the best in-dist val LSD (0.931) of all four arms and is the only failure, which is the second independent refutation of P3-2's "a sharper recipe, not a different conditioning design" recommendation (written by this agent; wrong). The sharpest single diagnostic is rho inside vs outside the held-out slab: **A 1.060 -> 0.509**, i.e. arm A learns the law correctly everywhere EXCEPT the region the holdout defines, while B/C/D pass straight through (1.039 / 0.947 / 1.031). Also verified: P3-2's two holdouts were never comparable tests -- `(west,0.50)` interpolated that wall's trained range {0.05,0.15,0.70} while `(north,0.70)` EXTRAPOLATED beyond north's maximum of 0.50; P3-2b's slabs are both strictly interior. **Adopt arm C.** **Revisit / generalize**: the binding constraint was sampling density in the *physical parameter* space, which suggests the same audit for P3-1's 3D geometry conditioning (trained on a comparably sparse grid) before concluding anything about its conditioning mechanism.


---

## 2026-08-15: P3-2c — derivation sampler, designated S2, realized-gap axis, and the control failure (D53)

**D53 — P3-2c's cross-arm density sweep is CONFOUNDED by its own repair mechanism; the reportable result is the within-run extrapolation curve.** (a) **Derivation, not resampling.** Arms must differ only in the west hold-out band, which rules out re-running the sampler: `draw_alpha` rejects in a loop and `WALLS_2D` puts west first, so a west rejection shifts east/south/north in every four-wall config. Each arm's manifest is therefore DERIVED from the frozen W015 rows using an arm-independent, position-keyed repair stream `default_rng([REPAIR_SEED, geom_id, row_i, wall_slot])`: an arm's value is the first entry of `[W015_value] + repair_stream` its own predicate accepts, so two arms differ **iff** one rejects the other's value. Verified: W015 re-derives with **0** changes (so reusing the trained P3-2b arm C is exact, not approximate), deltas are exactly 31/120/236/156, north draws are byte-identical across all five manifests, and 479 new sims replace a naive 5850 (92% reuse). (b) **S2 membership is DESIGNATED, not inferred.** W100's slab `[0.193, 1.193]` contains alpha=0.30 (m=0.3567), so an "is it in THIS arm's slab" rule would migrate 10 `west@0.30` rooms S4->S2 for W100 alone, giving it a 30-room headline against everyone else's 20 — five well-formed slopes, no error, and a curve confounding gap width with population change. Every arm therefore reuses `p3_2b_splits.classify`, which keys on the frozen W015 slabs; arm-specific facts attach as annotations (`d_support_m`, `arm_holdout`). That annotation immediately caught that **XTRAP's `west@0.50` is TRAINED** (m=0.6931 < its threshold 1.10, `d_support` 0.0001) — unflagged it would have read as a spectacular pass at the widest exclusion in the sweep. (c) **The x-axis is the REALIZED gap, never the nominal width**, and for the edge-excluded arm it is beyond-edge distance, never a gap: XTRAP's `realized_gap_west` is 0.0262 (the ordinary sampling gap of a contiguous region), and plotting that beside the interior gaps would place the extrapolation arm at the dense end of the curve. (d) **THE RESULT: the design does not identify what it was built to measure.** The pre-registered north control — identical slab, byte-identical draws in all four arms — moves *perfectly monotonically* with the manipulation (Spearman **1.000**, spread **0.316** vs a 0.15 tolerance) while west does not (Spearman **-0.400**, spread 0.350). The mechanism is the repair stream itself: repairing a rejected west draw pushes it OUT of the slab toward the extremes of m, so a wider slab does not merely remove a band — it re-shapes the marginal distribution of west absorptions and with it the model's global allocation. North's rooms are the same rooms; their west walls are not. The two gated metrics agree by disagreeing: rho never crosses 0.80 (bound > 1.0033), slope crosses at 0.7542, and **49.5%** of paired bootstrap resamples show no crossing. **No west-specific gap effect is identifiable from this design.** (e) **What IS reportable** is XTRAP's within-run curve — one model, immune to cross-arm realization noise: edit slope 0.917 / 0.597 / 0.313 at +0.106 / +0.288 / +0.511 beyond the training edge, monotone, crossing 0.80 at **Δm ≈ 0.173**, and cleared by a selection-bias check (93 of 93-94 modes always-valid; full-pool and always-valid slopes agree to 3 dp) with GT effect size *rising* (6.162 -> 8.204 Hz) while the model's response falls. **Alternatives rejected**: inferring S2 per arm (rejected, (b)); nominal slab width as x (rejected, (c)); reporting either breakpoint metric alone (rejected — the disagreement is evidence, not noise to be resolved by picking). **Revisit**: a future sweep must hold the training marginal fixed BY CONSTRUCTION (resample so the out-of-slab m distribution matches across arms) **and** use replicate seeds; one seed per arm cannot separate the two explanations even in principle.

---

## 2026-08-15: FT-1 — FDTD over per-bin Helmholtz, and the limits of a one-geometry validation (D54)

**D54 — the wave solver is FDTD, it is correct and affordable, and its GO is conditional.** (a) **FDTD, not per-bin Helmholtz**: Helmholtz needs a sparse factorization per frequency (~49 ms x 601 bins ~ 30 s/room real, 60-90 s complex) while node-centred SLF leapfrog measures **0.83 s per 2 s room** on one pinned core and yields all 601 bins, all receivers and the full IR in one run. Projected **0.231 CPU-h per 1000 configs** against a 12 CPU-h budget (52x headroom), and interior structure — divider, aperture, absorber patch — costs **nothing measurable** (2.8% spread) because it runs the same dense code path as an empty box. Frozen: dx=0.05 (divides 4.5/4.0/8.0/3.0 exactly; dx=0.06 does not and its ~1% frequency shift would consume the entire A1a budget), fs_sim=12288, N=24576, T=2.000 s exactly, lambda_CFL=0.5583 (bound 0.7071), giving rfft bins at exactly 0.5 Hz so `H_complex` is a slice with zero resampling. Source must satisfy S(0)=0 — a DC component drives the undamped (0,0) mode of a rigid room without bound. (b) **Gates assert against the WAVE law and merely report ISM.** ISM and a locally-reacting wall disagree by 30-44% on uniform alpha (1.972/2.218/2.968 vs 2.843/2.957/3.867 Hz) because of grazing-incidence absorption — real physics, not solver error — so A2 asserts within 10% of Kuttruff. Measured worst +8.37%. (c) **A3's stated invariance was FALSE and the gate passed for the wrong reason.** The claim "BW(1,0)/BW(0,1) = 2 exactly, independent of L, W, alpha, c" does not hold; the exact ratio is `2*artanh(xi)/xi`, which is alpha-dependent. Measured 2.1267 at alpha=0.7 against an exact 2.0516 — inside the +/-0.20 band, but not for the reason claimed, and at higher alpha it would fail a correct solver. Additionally the two-room same-alpha framing in the 4.5x4.0 reference room gives `2W/L = 1.7778`, not 2.00. **This is the third gate in this project that would have mis-scored correct physics and the first that PASSED on a wrong target; the standing rule is to derive each gate's expected value from the governing law at the exact configuration being run and to check claimed invariances symbolically.** (d) **kappa must be re-derived per simulator but the reported comparison is not the right one.** `kappa_fdtd = 1.0208` vs the frozen `kappa_ism = 1.6608` compares slopes fitted against DIFFERENT regressors — a change of damping law, not an estimator property. The constant downstream actually consumes, `d(BW)/d(m_wall)`, measures ~1.005 on FDTD. Do not carry "a factor of 1.63" forward as a recalibration. (e) **GO-WITH-CHANGES, and FT-B/FT-C were not run.** All ten gates pass and three independent adversarial lenses agree the solver itself holds, but every gate ran L=4.5, W=4.0 — the ONLY on-grid geometry — while **39 of 40 train and 9 of 10 test geometries are not integer multiples of dx**, and `_grid_count` merely warns before snapping (L=3.68 -> 73.6 cells -> 0.5-0.8% dimension error, 10-16x the tolerance the solver was validated to). dx=0.01 divides every 2-dp dimension but costs 125x (28.8 CPU-h/1000), so the affordable direction is to snap the ROOMS and define the wave-track geometry family on the grid. Two further blockers bear on the edit axes directly: the absorber patch's acoustic extent is exactly dx wider than requested while `_apply_patch` reports the nominal span, and **both new edit parameters are quantized to dx and cannot be sampled continuously** — which collides head-on with D52's finding that continuous sampling in the linearizing coordinate is the operative variable. **Revisit**: answer the quantization question (accept a quantized axis and test whether D52 transfers, or restore continuity via sub-cell boundary interpolation) and add a physics gate per edit axis BEFORE either is trained.

---

## 2026-08-17: P3-3-FAST Track 2b — the doorway-aperture dataset, sqrt(a) conditioning, and the sealed room (D57)

**D57 — the aperture axis is conditioned on sqrt(a), the hold-out is an exact BAND in a, and the sealed room is in the dataset but out of the law.** (Numbering note: this file jumps D54 -> D57; D55/D56 are cited in the P3-3-FAST Track A code but were never written up here.) (a) **Coordinate.** FT-B swept the doorway width a in {0, 0.1, ... 4.0} at dx = 0.01 and fitted the inter-room level difference against six candidate coordinates: **sqrt(a)** is the linearizing one (pooled r^2 = **0.9870**; raw a 0.905, a^2 0.704), so `aperture_features_2d` puts an IDENTITY channel on `sqrt(a)/sqrt(4.0)` plus 3 octaves (pi, 2pi, 4pi), exactly the role m = -ln(1-alpha) plays on the absorption axis (D52). Geometry is `(L, W, x0)` at 8 octaves = 48 dims, so the arm is **55-d** with layout `[0:16] L | [16:32] W | [32:48] x0 | [48:55] aperture`. The divider position x0 is a THIRD geometry dimension, not a material one — without it the model cannot tell which sub-room a receiver is in. Registered in BOTH `cond_dim_for` and `INR2D_AutoDecoder`'s own whitelist; those are independent checks and a new arm registered in only one of them fails at model construction time with a message that blames the config. (b) **a = 0 is a TOPOLOGICAL discontinuity, not the small-aperture limit.** A sealed one-node divider disconnects room B *exactly*: H_B is identically zero and the level difference is -inf, not merely large. Its conditioning coordinate sqrt(0) = 0 is also the limit of a vanishing doorway, so the conditioning **cannot** separate the two, and training on it would force FiLM to fit a discontinuity in its own coordinate and smear every narrow aperture nearby. The 26 sealed rooms are therefore built and kept (they are the end-point demonstration, and the dataset gate uses one to prove the divider plumbing reaches the solver), flagged `sealed: true` in the manifest, given their own `kind`, and excluded from training via `config_kinds: ["open", "aperture"]`. They must be excluded from every continuous-coordinate fit. (c) **The hold-out is a band, not a geometry set.** No training config carries a in [0.9, 1.1] — draws landing inside are redrawn, so the hold-out is EXACT rather than approximate (asserted at manifest-freeze time and again in the dataset gate) — while 18 test configs (3 per test domain: 0.95, 1.00, 1.05) sit inside it. The question is then "did the model learn the aperture law?" rather than "did it memorize 20 aperture values?". The six test domains are strictly INTERIOR to the training box in L, W and x0/L, so geometry is interpolation and the aperture is the only thing under test. (d) **dx and fs are coupled and both are forced.** FT-1b A0c measured the aperture observable moving 10.4x the estimator floor between dx = 0.02 and dx = 0.01, so dx = 0.02 is inside the un-converged regime for THIS axis; and at dx = 0.01 a fixed fs = 12288 raises a CFL ValueError from the solver. Frozen: dx = 0.01, fs = 61440, n = 122880 (lambda = 0.55827, T = 2.000 s, df = 0.5 Hz), which keeps the 601-bin 0-300 Hz slice byte-compatible with every earlier 2D loader. Cost is ~200 s/room against Track A's ~9 s, i.e. the aperture axis is ~22x more expensive per room than the absorption axis. (e) **Receivers are nudged off the divider, per DOMAIN not per config.** The 8x8 grid spans the full domain (both sub-rooms) with a 0.3 m margin; any receiver whose x-node lands on or beside the divider column is pushed two nodes clear of it. Without that, `simulate` raises "snaps onto a solid node" for the sealed and narrow-aperture configs while succeeding for the wide ones — a per-config failure on a property that must be per-domain. The nudge depends only on (L, W, x0), so all 20 configs of a domain share one receiver array and are directly comparable. **Revisit if** the aperture axis is combined with per-segment absorption (the two conditioning blocks would need a shared geometry normalization, and this arm's box — L in [7,9] — is deliberately NOT the P3-2 box).
