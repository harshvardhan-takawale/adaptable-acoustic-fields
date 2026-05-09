# CONTEXT_FOR_MANAGER.md

Manager re-orientation doc. Optimized for catching up in 5 minutes after time away. Updated at the end of every chunk.

**Last updated**: Chunk 0 — 2026-05-09

## Project state

- **Phase**: 1 (2D shoebox sweep, 0–2 kHz, Track A — science).
- **Chunk completed**: Chunk 0 (recon + scaffolding only).
- **What exists today**: empty source package skeleton, conda env (`aaf`), canonical docs, working SLURM hello-world. **No models, no data, no training.**
- **What does not exist**: data generation, model code, renderer code, training loop, eval / eigenfrequency probes.
- **Next chunk** (manager will write Chunk 1): expected to cover dataset generation (pyroomacoustics 2D ISM sweep over L) + 2D renderer/model port from INFER. See `tasks/CHUNK_0_RESULTS.md` §"2D adaptation needs" for the exhaustive list of touch-points.

## Codebase map

```
adaptable-acoustic-fields/
├── CLAUDE.md                   standing rules for any agent
├── CONTEXT_FOR_MANAGER.md      this file
├── DECISIONS.md                append-only design log
├── OPEN_QUESTIONS.md           ambiguities (numbered, append-only)
├── CLUSTER_INFO.md             Nexus SLURM how-to, partitions, sbatch template
├── README.md                   public-facing intro
├── environment.yml             frozen conda env (aaf, Py3.8, torch 2.0.1+cu118)
├── pyproject.toml              ruff + black + pytest config
├── .gitignore                  ignores logs/, ckpts/, data/, project_files/, …
├── aaf/                        importable source package
│   ├── data/                   (chunk 1) dataset & pyroomacoustics simulation
│   ├── models/                 (chunk 2) 2D INR + auto-decoder
│   ├── renderers/              (chunk 2) frequency-domain renderers
│   ├── train/                  (chunk 3) training loop, checkpointing
│   ├── eval/                   (chunk 4) eigenfrequency probes, metrics
│   └── utils/                  logging, criteria
├── configs/                    Hydra configs (chunk 1+)
├── scripts/
│   └── slurm/
│       └── hello.sh            verified runnable
├── tests/
│   ├── test_env.py             imports + CUDA assert (run via SLURM, see CLAUDE.md)
│   └── test_smoke.py           trivial assert
└── tasks/
    └── CHUNK_0_RESULTS.md      ← read this for recon details
```

## What we inherited from AVR / INFER

**AVR** (NeurIPS'24, `/fs/nexus-projects/multimodal_recon/AVR`):
- Frequency-domain volume rendering pipeline (`renderer.py:AVRRender`)
- 6-term criterion (`utils/criterion.py`): spec + amplitude + angle + time + energy + multi-STFT
- Acoustic metric battery (`utils/metric.py`): Angle / Amp / Env / T60 / EDT / C50 / multi-STFT
- TensorBoard logging pattern (`avr_runner.py:196-244`)

**INFER** (ICML'26 submission, `project_files/`):
- Main model: `AVRModel_complex_FD_FreqDep_PhaseCorrection` (`unified_models.py:752-883`)
  - Forward signature: `forward(pts, view, tx, tx_view) → (attn_complex, signal_complex)`
  - Per-frequency-bin complex attenuation: `attn = σ + jβ` (real via softplus → absorption, imag → phase velocity change)
- Main renderer (we keep, sans KK): `AVRRenderFD_FreqDep_PhaseCorrection_new` (`unified_renderers.py:716-790`)
  - Renderer's `acoustic_render_fd` decomposes complex `attn` into σ (clamped ≥ 0) and β, builds amplitude transmittance (cumprod of `1-α`) and phase transmittance (`exp(j·cumsum(β·Δu))`), applies geometric phase `exp(-j·2π·f·d/c)`, sums over rays.
- KK ablation variant exists (`AVRRenderFD_FreqDep_PhaseCorrection_KK`, lines 829-1035) — **not** used for Phase 1.

**Note for next agent**: An earlier sub-agent misidentified `INRASFrequencyModel` as the INFER main model. INRAS is a **baseline** (Su et al. 2022 in INFER's Related Work). The actual INFER main model is the `AVRModel_complex_FD_FreqDep_PhaseCorrection` cited above. See `tasks/CHUNK_0_RESULTS.md` for confirming evidence.

## What's working

- **Conda env**: `aaf` exists at `/fs/nexus-scratch/htakawal/miniconda3/envs/aaf` (cloned from `avr_scavenger`, plus `hydra-core` + `gh`). Frozen in `environment.yml`.
- **pytest**: `tests/test_smoke.py` and `tests/test_env.py` exist. `test_env.py` imports torch/numpy/scipy/h5py/hydra/pyroomacoustics/tinycudann/auraloss/librosa and asserts CUDA. Login-node libstdc++ blocks scipy import; tests run fine on compute nodes via SLURM.
- **Hello-world SLURM**: `scripts/slurm/hello.sh` submitted as job 6797442 (scavenger partition); confirms env activation + nvidia-smi + torch.cuda.is_available. Result recorded in `tasks/CHUNK_0_RESULTS.md`.
- **Git + GitHub**: repo public at `https://github.com/harshvardhan-takawale/adaptable-acoustic-fields`. Raw URL pattern: `https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/<file>.md`.

## What's broken or stubbed

- All `aaf/**/*.py` are empty `__init__.py` files. Source code is intentionally absent for Chunk 0.
- `configs/` empty. Hydra config tree comes in Chunk 1.
- No 2D adaptation done yet — recon document only.

## Recent changes (this chunk)

- Created repo skeleton, `environment.yml`, `pyproject.toml`, `.gitignore`.
- Wrote canonical docs: `CLAUDE.md`, `CONTEXT_FOR_MANAGER.md`, `DECISIONS.md`, `OPEN_QUESTIONS.md`, `CLUSTER_INFO.md`, `README.md`.
- Wrote `tasks/CHUNK_0_RESULTS.md` — the recon writeup the manager should read before authoring Chunk 1.
- Wrote and verified `scripts/slurm/hello.sh`.
- Cloned `avr_scavenger` → `aaf`, added `hydra-core`, `gh` (via conda-forge). Exported `environment.yml`.

## Pointers

- Next-chunk-relevant questions: see `OPEN_QUESTIONS.md`. Numbers 1, 2, 3, 4 directly affect Chunk 1 design.
- The recon writeup: `tasks/CHUNK_0_RESULTS.md`. Read this before writing Chunk 1.
- Design log: `DECISIONS.md`. Six initial entries documenting the framework choices.
- Cluster how-to: `CLUSTER_INFO.md`. Default partition is `scavenger`.

## How the manager can read this repo

Public; raw URLs are stable on `main`. Examples:

```
https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/CONTEXT_FOR_MANAGER.md
https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/CHUNK_0_RESULTS.md
https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/OPEN_QUESTIONS.md
```
