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
