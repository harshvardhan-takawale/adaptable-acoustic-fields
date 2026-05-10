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
