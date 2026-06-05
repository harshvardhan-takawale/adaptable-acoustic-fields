# CLUSTER_INFO.md

Canonical reference for running jobs on **Nexus** (UMIACS @ UMD). Every agent submitting jobs must read this first. If something here is wrong, fix it and append to DECISIONS.md.

Source: [UMIACS Nexus wiki](https://wiki.umiacs.umd.edu/umiacs/index.php/Nexus). When the wiki is more current, trust the wiki.

## What is Nexus

Nexus is the unified UMIACS SLURM cluster covering UMD CS / UMIACS / lab-sponsored compute. It schedules a mix of public partitions (`tron`, `scavenger`, `scavenger-aarch64`) and lab-only partitions (`cbcb`, `clip`, `cml`, `gamma`, `mbrc`, `mc2`, `quics`, `vulcan`). Use of any partition requires UMD VPN (GlobalProtect) before SSH.

## Filesystems

| Path | Purpose | Quota | Backup | Use for |
|------|---------|------:|:------:|---------|
| `/nfshomes/<user>` | home | 30 GB | yes | dotfiles, small configs |
| `/fs/nexus-scratch/<user>` | per-user scratch | 200 GB | no | conda envs, intermediate data |
| `/fs/nexus-projects/<proj>` | shared lab storage | project-quota | yes | code, datasets, checkpoints to keep |
| `/scratch0`, `/scratch1` | per-node local scratch | per-node | no | hot temp data; deleted after 90d inactivity |

For this project:
- Repo root: `/fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields`
- Conda env `aaf`: `/fs/nexus-scratch/htakawal/miniconda3/envs/aaf`
- AVR reference: `/fs/nexus-projects/multimodal_recon/AVR`

## Login-node policy

UMIACS reserves the right to kill user processes on submission/login nodes. Do **not**: long compiles, `pytest` (anything that imports `torch.cuda` or scipy native), training, large data conversions. Run those via `sbatch` or `srun --pty`.

## libstdc++ shim (mandatory for any job that imports scipy / pyroomacoustics / matplotlib)

`/lib64/libstdc++.so.6` on Nexus login and compute nodes caps at `GLIBCXX_3.4.25`. The `aaf` env's scipy native extension needs `GLIBCXX_3.4.29`, which is present in `${CONDA_PREFIX}/lib/libstdc++.so.6.0.33`. Without the shim, `import scipy` fails with `ImportError: ... GLIBCXX_3.4.29 not found`.

Add this line **after** `conda activate aaf` in every job script:

```bash
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
```

The templates in `scripts/slurm/hello.sh` and `scripts/slurm/run_pytest.sh` already do this; copy verbatim.

## Partitions and QoS

| Partition | Use | Account | QoS | Preempt? |
|-----------|-----|---------|-----|----------|
| `scavenger` | preemptible (cheap, plentiful) | `scavenger` | `scavenger` | yes — your job may be killed |
| `tron` | general-purpose UMIACS/CSD | `nexus` (or sponsor) | `default` / `medium` / `high` | no |
| `class` | classroom accounts | n/a for us | n/a | n/a |

QoS limits (Job-level):

| QoS | CPUs | GPUs | RAM | Max wall |
|-----|-----:|-----:|----:|---------:|
| `default` | 4 | 1 | 32 GB | 3 d |
| `medium` | 8 | 2 | 64 GB | 2 d |
| `high` | 16 | 4 | 128 GB | 1 d |
| `scavenger` | unlimited | unlimited | unlimited | 3 d |

Per-partition user caps apply (e.g. `tron`: 32 CPU / 4 GPU / 256 GB simultaneously). Submission cap: max 500 jobs (running + pending) per user per partition.

GPU inventory per partition is not summarized here — see [Nexus/GPUs](https://wiki.umiacs.umd.edu/umiacs/index.php/Nexus/GPUs) before requesting `--gres=gpu:<type>:<n>`. For development, `--gres=gpu:1` (any) is fine.

### ⚠ GPU-type targeting — `qos` does NOT pick the GPU (P2-2.5 lesson)

**A bare `--gres=gpu:1` lands on whatever card is free — which is usually an 11 GB RTX 2080 Ti on `tron62/63`, NOT the big Ampere cards.** `--qos=high` raises your QoS *limits* (CPUs/GPUs/RAM/walltime); it does **not** route you to a bigger GPU. For anything that needs >11 GB you **must** name the GPU type in the gres string. This bit both P2-2 (M1/M2 silently ran on a 2080 Ti, forcing batch=4) and P2-2.5's first launch (all 3 runs OOM'd on 2080 Ti).

`tron` GPU types and how to request them (verify live counts with `sinfo -p tron -o "%n %G %t"`):

| Type | VRAM | gres string | Representative nodes |
|---|---|---|---|
| RTX 2080 Ti | 11 GB | `--gres=gpu:rtx2080ti:1` | tron62, tron63 |
| RTX A4000 | 16 GB | `--gres=gpu:rtxa4000:1` | tron06-36 (many) |
| RTX A5000 | 24 GB | `--gres=gpu:rtxa5000:1` | tron46-58 |
| RTX A6000 | 48 GB | `--gres=gpu:rtxa6000:1` | tron00-05 |

Rule of thumb for this project: the 3D auto-decoder at `batch≥16` or `n_pts_per_ray=32` needs ≥ 24 GB (A5000) or ≥ 48 GB (A6000). The 3D single-room / multi-room at `batch=4, n_pts=16` fits the 2080 Ti (P2-1's memory cascade). When in doubt, name an A5000.

## sbatch template

Place templates in `scripts/slurm/`. Logs land in `logs/slurm/<jobname>-<jobid>.out`. Naming: `<chunk>_<short>.sh` (e.g. `chunk1_simulate.sh`).

```bash
#!/bin/bash
#SBATCH --job-name=aaf_<short>
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# your command here
```

Default partition for development is `scavenger` — cheap, plentiful, preemptible. If a job must complete without preemption (e.g., final eval runs), use `tron --qos=default` and budget time accordingly. Document each long-running job's choice in DECISIONS.md if non-default.

## Common commands

```bash
sbatch scripts/slurm/hello.sh                      # submit
squeue --me                                        # my queued/running jobs
squeue --me -o '%.10i %.9P %.20j %.2t %.10M %R'    # cleaner view
scancel <jobid>                                    # cancel
sacct -j <jobid> --format=JobID,JobName%30,State,Elapsed,MaxRSS  # post-mortem
tail -f logs/slurm/<jobname>-<jobid>.out           # live log
srun --pty --partition=scavenger --account=scavenger --qos=scavenger \
     --gres=gpu:1 --cpus-per-task=2 --mem=8G --time=01:00:00 bash  # interactive
```

## Conventions for this project

- **Job scripts**: `scripts/slurm/<chunk>_<short>.sh`. Always run from repo root (`cd $PROJECT_ROOT`).
- **Logs**: `logs/slurm/<jobname>-<jobid>.{out,err}`. `logs/` is gitignored except for `.gitkeep`.
- **Conda activation block** (copy verbatim into each script):
  ```bash
  source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
  conda activate aaf
  ```
- **Hello-world**: `scripts/slurm/hello.sh` — submit at the start of any new chunk to verify env+GPU still work end-to-end.
- **Tests via SLURM**: `pytest` cannot run on the login node (scipy import fails). Submit pytest as a small scavenger job or run inside an `srun --pty` shell.

## Open items (move to OPEN_QUESTIONS.md if blocking)

- Are we eligible for the `nexus` account on `tron` for non-preemptible jobs, or do we have to stay on scavenger? *(Chunk 3 confirmed yes — `--partition=tron --account=nexus --qos=default` works. Used 1 of 4 banked slots for multi-room training; 3 remain.)*
- Per-GPU-type availability — A4000 / A5000 / A6000 / 2080Ti — *(P2-2.5 resolved: see the GPU-type targeting table above. `sinfo -p tron -o "%n %G %t"` shows live counts; A6000 ~20 free, A5000 plentiful, 2080 Ti on tron62/63. The gres string — not qos — selects the type.)*
- Summer 2026 cluster OS upgrade ([Nexus/ClusterOSUpgrade](https://wiki.umiacs.umd.edu/umiacs/index.php/Nexus/ClusterOSUpgrade)): may invalidate cached tinycudann builds.

## Chunk-2 and Chunk-3 pipeline drivers

- `scripts/run_chunk2_pipeline.sh`: memory_check → 3× single-room train (parallel, scavenger) → 3× single-room eval (parallel, scavenger). Wall ~2 h.
- `scripts/run_chunk3_pipeline.sh`: memory_check → 1 multi-room train (tron, non-preemptible) → 6× zero-shot eval (parallel, scavenger) → latent probe (scavenger). Wall ~3 h. The training script defaults to `--partition=tron --account=nexus --qos=default`; if tron is unavailable, edit `scripts/slurm/multi_room_train.sh` to swap those three SBATCH lines for `--partition=scavenger --account=scavenger --qos=scavenger`.
