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
