# CLAUDE.md — Standing Instructions for Agents

Read this in full before doing anything in this repo.

## Project one-liner

**adaptable-acoustic-fields (aaf)** — an editable spatial-audio implicit neural representation. We condition a frequency-domain INR on a per-room latent `z_s` (DeepSDF-style auto-decoder) so a single shared model renders any room in a family. Phase 1 (current): 2D shoebox rooms varying in length L, 0–2 kHz (Track A, science). Phase 2+: structured edits (W, H, doorways, materials) and adaptation to unseen rooms.

## Current phase

**Chunk 0 complete: scaffolding only.** No models, no data generation, no training yet. The scaffolding sits at:
- `aaf/` — empty source package (will hold data, models, renderers, train, eval, utils)
- `configs/` — empty (Hydra configs added Chunk 1+)
- `scripts/slurm/hello.sh` — runnable hello-world for env verification
- `tests/` — env smoke tests only

Next: see `tasks/CHUNK_0_RESULTS.md` for the recon writeup that will inform Chunk 1.

## How to run

### Activate the env
```bash
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
# Required: prepend conda's libstdc++ (GLIBCXX_3.4.29) so scipy native ext loads.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
```

The `aaf` env is a clone of `avr_scavenger` (Python 3.8.20, torch 2.0.1+cu118, tinycudann 1.7) plus `hydra-core` and `gh`. Frozen in `environment.yml`. **Do not rebuild tinycudann** — the wheel was built against a specific GCC/CUDA on a compute node; clone preserves it.

**libstdc++ shim**: Both Nexus login and compute nodes have `/lib64/libstdc++.so.6` capped at `GLIBCXX_3.4.25`. The env's scipy native extension needs `GLIBCXX_3.4.29`, present in `${CONDA_PREFIX}/lib/libstdc++.so.6.0.33`. Always prepend it via `LD_LIBRARY_PATH` in any script that imports scipy / pyroomacoustics / matplotlib. The SLURM templates in `scripts/slurm/` do this; copy the activation block verbatim.

### Run tests

`tests/test_env.py` imports torch/scipy/pyroomacoustics and asserts CUDA — needs a compute node + LD_LIBRARY_PATH shim. Run via SLURM:

```bash
sbatch scripts/slurm/run_pytest.sh
# logs in logs/slurm/aaf_pytest-<jobid>.out
```

Or interactively:

```bash
srun --pty --partition=scavenger --account=scavenger --qos=scavenger \
     --gres=gpu:1 --cpus-per-task=2 --mem=8G --time=00:30:00 bash
# inside:
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
pytest -q
```

### Submit a job
See `CLUSTER_INFO.md` for partitions, QoS, sbatch template, log conventions, and login-node policy. Default partition for development: `scavenger` (preemptible, plentiful).

## Hard rules (apply to every task)

1. **End every task by updating `CONTEXT_FOR_MANAGER.md` and writing `tasks/CHUNK_N_RESULTS.md`.** No exceptions. The manager (a separate Claude) reads these to plan the next chunk; if they're stale, the manager wastes a turn re-orienting.

2. **Every non-trivial design choice gets a `DECISIONS.md` entry.** Format is in the file. Append-only.

3. **Ambiguity, blockers, or research-direction questions go in `OPEN_QUESTIONS.md`.** Do not guess on research direction. Numbered, append-only; when answered, move the resolution to `DECISIONS.md` and remove the question.

4. **Cluster work (job scripts, env changes) follows `CLUSTER_INFO.md`.** If the cluster docs are wrong, fix them and note in `DECISIONS.md`.

5. **Update `environment.yml` if you change the env.** Use `conda env export --no-builds > environment.yml`.

6. **Don't push generated artifacts.** `logs/`, `ckpts/`, `data/`, `*.h5`, `*.npz`, `*.wav` are gitignored — keep it that way.

7. **Reference material lives in `project_files/`** (gitignored). It includes the AVR/INFER artifacts and PDFs. Do not copy them into the repo.

## Code style

- Python 3.8 (matches the cloned env). When 3.10+ syntax is tempting, don't.
- `ruff` for lint, `black` for format. Configs in `pyproject.toml`. Line length 100.
- AVR has no enforced style — we adopt one because we plan to scale beyond a single-paper codebase.
- Type hints on public-facing functions; not required everywhere.
- Tests under `tests/`, named `test_*.py`. pytest discovers automatically.

## What this project inherits, conceptually

- **AVR** (`/fs/nexus-projects/multimodal_recon/AVR`, NeurIPS'24): frequency-domain volume rendering, criterion battery, spherical ray integration. We adapt to 2D and add latent conditioning.
- **INFER** (`project_files/unified_models.py`, `project_files/unified_renderers.py`): main model/renderer pair `AVRModel_complex_FD_FreqDep_PhaseCorrection` + `AVRRenderFD_FreqDep_PhaseCorrection_new`. We carry forward the σ + jβ complex attenuation form. We deliberately drop Kramers-Kronig and perceptual weighting for Phase 1 (revisit if reconstructions show non-causal phase or audible magnitude errors).

See `tasks/CHUNK_0_RESULTS.md` for the detailed recon, citations, and the explicit list of 2D adaptation needs.

## Manager / agent collaboration model

- The **manager** (a separate Claude session) writes task specs based on `CONTEXT_FOR_MANAGER.md` and the latest `CHUNK_N_RESULTS.md`. They have no direct access to the codebase — they read the public raw GitHub URLs.
- **You** (the task agent) execute the spec, update the docs, and stop. Don't redirect strategy without consulting `OPEN_QUESTIONS.md`.

## Repo URL conventions

- GitHub: `https://github.com/harshvardhan-takawale/adaptable-acoustic-fields`
- Raw URL pattern (for the manager): `https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/<file>.md`
