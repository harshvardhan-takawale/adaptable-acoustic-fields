"""Forward-compatible PyTorch Dataset interface stub.

The real implementation lives in Chunk 2 once the renderer's batch shape is
fixed. This file defines the API surface so Chunk 2's caller code can be
sketched against it.

Yields one sample = one (room, receiver) pair:

    {
      "H_complex":  (n_freq_bins,) complex64,
      "rx_pos":     (2,) float32,
      "tx_pos":     (2,) float32,
      "L":          float,
      "room_id":    int,            # auto-decoder index
    }
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


class ShoeboxDataset:
    """STUB — implemented in Chunk 2.

    Args:
        sweep_yaml: path to a configs/sweeps/*.yaml file.
        split: 'train' | 'test'. Selects which L list from the YAML.
        data_dir: base dir containing per-room HDF5 files (default: data/track_a/).
    """

    def __init__(
        self,
        sweep_yaml: str | Path,
        split: str,
        data_dir: str | Path = "data/track_a",
    ):
        # TODO(chunk-2): parse YAML, validate every L file exists, build
        #   index of (room, rx) pairs, lazily mmap HDF5 reads.
        raise NotImplementedError(
            "ShoeboxDataset is a Chunk-1 interface stub; implement in Chunk 2 "
            "once the renderer's batch shape is known."
        )

    def __len__(self) -> int:  # pragma: no cover
        raise NotImplementedError

    def __getitem__(self, i: int) -> dict:  # pragma: no cover
        raise NotImplementedError

    def room_ids(self) -> Iterable[int]:  # pragma: no cover
        """Iterate over all unique room indices in this split (for the
        auto-decoder latent table)."""
        raise NotImplementedError
