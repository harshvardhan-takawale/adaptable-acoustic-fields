"""Smoke checks that the aaf conda env has every package Phase 1 requires.

Note: this test imports torch.cuda and scipy native extensions, which fail on
the Nexus login node due to its older libstdc++. Run via SLURM or `srun --pty`.
See CLUSTER_INFO.md.
"""
import importlib

import pytest


REQUIRED = [
    "torch",
    "numpy",
    "scipy",
    "matplotlib",
    "h5py",
    "hydra",
    "pyroomacoustics",
    "tinycudann",
    "auraloss",
    "librosa",
]


@pytest.mark.parametrize("name", REQUIRED)
def test_import(name):
    importlib.import_module(name)


def test_cuda_available():
    import torch
    assert torch.cuda.is_available(), "CUDA not available — running on a CPU-only node?"


def test_cuda_device_name():
    import torch
    assert torch.cuda.device_count() >= 1
    assert isinstance(torch.cuda.get_device_name(0), str)
