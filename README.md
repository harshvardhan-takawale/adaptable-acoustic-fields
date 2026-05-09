# adaptable-acoustic-fields (aaf)

Editable spatial-audio implicit neural representation. Conditions a frequency-domain INR on a per-room latent `z_s` (DeepSDF-style auto-decoder) so a single shared model renders any room in a family — and lets us edit room parameters by moving in latent space.

**Phase 1 (in progress)**: 2D shoebox rooms, sweep over length L, 0–2 kHz, validated against analytical eigenfrequencies.

## Setup

```bash
# clone the avr_scavenger conda env to aaf
conda create --name aaf --clone avr_scavenger
conda activate aaf
pip install hydra-core
# gh CLI (one of):
conda install -c conda-forge gh
```

Or restore from the frozen file:

```bash
conda env create -f environment.yml
```

The env requires a build-compatible compute node for tinycudann (already prebuilt in `avr_scavenger`). Do not rebuild on the login node.

## Run tests

```bash
# tests must run on a compute node (login node libstdc++ blocks scipy import):
srun --pty --partition=scavenger --account=scavenger --qos=scavenger \
     --gres=gpu:1 --cpus-per-task=2 --mem=8G --time=00:15:00 bash
# inside the shell:
conda activate aaf
pytest -q
```

## Cluster

This project runs on the [UMIACS Nexus](https://wiki.umiacs.umd.edu/umiacs/index.php/Nexus) cluster. See `CLUSTER_INFO.md` for partitions, sbatch template, and conventions.

## Repo orientation

- `CLAUDE.md` — standing rules for every agent that touches the repo
- `CONTEXT_FOR_MANAGER.md` — current state for the orchestrating agent
- `DECISIONS.md` — append-only design log
- `OPEN_QUESTIONS.md` — open ambiguities (numbered)
- `CLUSTER_INFO.md` — Nexus how-to
- `tasks/CHUNK_N_RESULTS.md` — per-chunk writeups (start with `CHUNK_0_RESULTS.md` for the recon)
- `aaf/` — source package (data, models, renderers, train, eval, utils)
- `configs/` — Hydra configs
- `scripts/slurm/` — job templates
- `tests/` — pytest

## References

This project builds on:

- **AVR** — *Acoustic Volume Rendering for Neural Impulse Response Fields*, NeurIPS'24 ([arXiv](https://arxiv.org/abs/2411.06307), [code](https://github.com/penn-waves-lab/AVR))
- **INFER** — Implicit Neural Frequency Response fields (ICML'26 submission)
- **DeepSDF** — Park et al., CVPR'19 (auto-decoder conditioning)

## Status

Chunk 0 complete: scaffolding only. No models, no data, no training yet. See `tasks/CHUNK_0_RESULTS.md` for the recon writeup that informs Chunk 1.
