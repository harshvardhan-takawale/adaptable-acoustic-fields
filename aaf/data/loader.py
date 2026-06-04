"""ShoeboxDataset — loads (room, receiver) samples from the Chunk-1.5 HDF5 dataset.

One sample = one (room, receiver) pair. Iterating one epoch visits all 64
receivers in every room of the requested split.

Yielded dict
------------
{
  "H_complex": Tensor[n_freq_bins] complex64,
  "rir_time":  Tensor[n_time_samples] float32,
  "rx_pos":    Tensor[2] float32,
  "tx_pos":    Tensor[2] float32,
  "L":         float,
  "W":         float,
  "alpha":     float,
  "room_id":   int,             # ordinal index into the room list, used by the
                                # auto-decoder in Chunk 3.
}

For single-room training (Chunk 2), pass `room_filter=[L]` to restrict the
dataset to one room's 64 receivers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from aaf.data.dataset_builder import read_room_h5, room_filename, room_filename_3d


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class ShoeboxDataset(Dataset):
    """Iterate over (room, receiver) pairs across rooms listed in a sweep YAML.

    Args:
        sweep_yaml: path to one of `configs/sweeps/{dense,sparse,extrapolation}.yaml`.
        split: "train" or "test" — picks `train_L` or `test_L` from the YAML.
        track: subdirectory under `data/`. Default "track_a".
        room_filter: optional iterable of L values to restrict to (intersected with the
                     YAML split). Used in Chunk-2 single-room mode (e.g., [3.0]).
        data_dir: override the data root. Defaults to `<repo>/data/<track>/`.
    """

    def __init__(
        self,
        sweep_yaml: str | Path,
        split: str = "train",
        track: str = "track_a",
        room_filter: Optional[list[float]] = None,
        data_dir: Optional[str | Path] = None,
    ):
        sweep_yaml = Path(sweep_yaml)
        with open(sweep_yaml) as f:
            cfg = yaml.safe_load(f)

        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        L_key = "train_L" if split == "train" else "test_L"
        L_list_yaml = [float(L) for L in cfg[L_key]]

        if room_filter is not None:
            keep = {float(L) for L in room_filter}
            L_list = [L for L in L_list_yaml if L in keep]
            if not L_list:
                raise ValueError(
                    f"room_filter={room_filter} has no overlap with sweep '{split}' "
                    f"L list {L_list_yaml}"
                )
        else:
            L_list = L_list_yaml

        self.sweep_yaml = sweep_yaml
        self.split = split
        self.track = track
        self.cfg = cfg
        self.W = float(cfg["W"])
        self.alpha = float(cfg["alpha"])
        self.fs = float(cfg["fs"])
        self.n_time_samples = int(cfg["n_time_samples"])
        self.n_freq_bins = self.n_time_samples // 2 + 1

        self.data_dir = Path(data_dir) if data_dir else REPO_ROOT / "data" / track
        self.L_list: list[float] = sorted(L_list)
        # ordinal room index → L
        self.room_id_to_L: dict[int, float] = {i: L for i, L in enumerate(self.L_list)}

        # Pre-resolve and validate the file paths; eagerly read once to populate the
        # in-memory cache (each file is ~8 MB so 15 rooms ≈ 120 MB — fine).
        self._cache: dict[float, dict[str, Any]] = {}
        for L in self.L_list:
            path = self.data_dir / room_filename(L=L, W=self.W, alpha=self.alpha)
            if not path.exists():
                raise FileNotFoundError(
                    f"missing dataset file for L={L}, W={self.W}, alpha={self.alpha}: {path}"
                )
            self._cache[L] = read_room_h5(path)

        # Each room has the same number of receivers (8x8 = 64).
        first_L = self.L_list[0]
        self.n_rx_per_room = self._cache[first_L]["ism_H"].shape[0]
        for L in self.L_list[1:]:
            assert self._cache[L]["ism_H"].shape[0] == self.n_rx_per_room, (
                f"L={L} has {self._cache[L]['ism_H'].shape[0]} receivers; "
                f"expected {self.n_rx_per_room}"
            )

        # Flat index: (room_id, rx_idx).
        self._index: list[tuple[int, int]] = []
        for room_id, L in self.room_id_to_L.items():
            for rx_idx in range(self.n_rx_per_room):
                self._index.append((room_id, rx_idx))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict:
        room_id, rx_idx = self._index[i]
        L = self.room_id_to_L[room_id]
        room = self._cache[L]
        attrs = room["attrs"]

        # source_pos / receiver_pos round-tripped via JSON in dataset_builder.
        src = np.asarray(attrs["source_pos"], dtype=np.float32)
        rx_all = np.asarray(attrs["receiver_pos"], dtype=np.float32)
        return {
            "H_complex": torch.from_numpy(room["ism_H"][rx_idx]),  # complex64
            "rir_time": torch.from_numpy(room["ism_rir"][rx_idx]),  # float32
            "rx_pos": torch.from_numpy(rx_all[rx_idx]),  # float32 (2,)
            "tx_pos": torch.from_numpy(src),  # float32 (2,)
            "L": float(L),
            "W": float(attrs["W"]),
            "alpha": float(attrs["alpha"]),
            "room_id": int(room_id),
        }

    def room_ids(self):
        """Iterate over unique room indices in this dataset."""
        return list(self.room_id_to_L.keys())

    @property
    def room_id_map(self) -> dict[int, float]:
        """Alias for ``room_id_to_L`` (matches the Chunk-3 manager-spec wording).

        Maps the dataset-internal ordinal room index → the L value (m).
        """
        return self.room_id_to_L

    def get_room_attrs(self, room_id: int) -> dict:
        """Return the HDF5 root attrs for a given room (already JSON-decoded)."""
        L = self.room_id_to_L[room_id]
        return self._cache[L]["attrs"]


# ----------------------------------------------------------------------
# 3D dataset (Phase 2)
# ----------------------------------------------------------------------


class Shoebox3DDataset(Dataset):
    """3D-room (room, receiver) pair iterator.

    Reads a Phase-2 ``configs/sweeps_3d/*.yaml`` config (schema:
    ``set_name, alpha, fs, n_time_samples, source_offset (3,), rooms: [{L, W, H}, ...]``)
    and loads the matching HDF5 files from ``data/<track>/`` (default
    ``track_a_3d``). Each sample is one (room, receiver) pair.

    Yielded dict
    ------------
        {
          "H_complex": Tensor[n_freq_bins] complex64,
          "rir_time":  Tensor[n_time_samples] float32,
          "rx_pos":    Tensor[3] float32,
          "tx_pos":    Tensor[3] float32,
          "L":         float,
          "W":         float,
          "H":         float,
          "alpha":     float,
          "room_id":   int,
        }

    For single-room training (P2-1), pass ``room_filter=[(L, W, H)]`` to
    restrict to one room's receivers.
    """

    def __init__(
        self,
        rooms_yaml: str | Path,
        track: str = "track_a_3d",
        room_filter: Optional[list[tuple[float, float, float]]] = None,
        data_dir: Optional[str | Path] = None,
    ):
        rooms_yaml = Path(rooms_yaml)
        with open(rooms_yaml) as f:
            cfg = yaml.safe_load(f)
        if "rooms" not in cfg:
            raise ValueError(f"{rooms_yaml}: missing 'rooms' key")

        self.rooms_yaml = rooms_yaml
        self.cfg = cfg
        self.alpha = float(cfg["alpha"])
        self.fs = float(cfg["fs"])
        self.n_time_samples = int(cfg["n_time_samples"])
        self.n_freq_bins = self.n_time_samples // 2 + 1
        self.source_offset = tuple(cfg.get("source_offset", (0.5, 0.5, 0.5)))

        rooms_all = [(float(r["L"]), float(r["W"]), float(r["H"])) for r in cfg["rooms"]]
        if room_filter is not None:
            keep = {(round(L, 2), round(W, 2), round(H, 2)) for L, W, H in room_filter}
            rooms = [(L, W, H) for (L, W, H) in rooms_all
                     if (round(L, 2), round(W, 2), round(H, 2)) in keep]
            if not rooms:
                raise ValueError(
                    f"room_filter={room_filter} has no overlap with rooms list."
                )
        else:
            rooms = rooms_all
        self.rooms: list[tuple[float, float, float]] = rooms

        self.data_dir = Path(data_dir) if data_dir else REPO_ROOT / "data" / track
        self.room_id_to_dims: dict[int, tuple[float, float, float]] = {
            i: dims for i, dims in enumerate(self.rooms)
        }

        # Eagerly load each room (3D rooms are ~67 MB each; 50 rooms ≈ 3.4 GB,
        # fits in compute-node RAM during training).
        self._cache: dict[tuple[float, float, float], dict[str, Any]] = {}
        for L, W, H in self.rooms:
            path = self.data_dir / room_filename_3d(L=L, W=W, H=H)
            if not path.exists():
                raise FileNotFoundError(
                    f"missing 3D dataset file for L={L}, W={W}, H={H}: {path}"
                )
            self._cache[(L, W, H)] = read_room_h5(path)

        first = self.rooms[0]
        self.n_rx_per_room = self._cache[first]["ism_H"].shape[0]
        for dims in self.rooms[1:]:
            assert self._cache[dims]["ism_H"].shape[0] == self.n_rx_per_room, (
                f"{dims} has {self._cache[dims]['ism_H'].shape[0]} receivers; "
                f"expected {self.n_rx_per_room}"
            )

        self._index: list[tuple[int, int]] = []
        for room_id, dims in self.room_id_to_dims.items():
            for rx_idx in range(self.n_rx_per_room):
                self._index.append((room_id, rx_idx))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict:
        room_id, rx_idx = self._index[i]
        L, W, H = self.room_id_to_dims[room_id]
        room = self._cache[(L, W, H)]
        attrs = room["attrs"]

        src = np.asarray(attrs["source_pos"], dtype=np.float32)
        rx_all = np.asarray(attrs["receiver_pos"], dtype=np.float32)
        return {
            "H_complex": torch.from_numpy(room["ism_H"][rx_idx]),       # complex64
            "rir_time": torch.from_numpy(room["ism_rir"][rx_idx]),       # float32
            "rx_pos": torch.from_numpy(rx_all[rx_idx]),                  # float32 (3,)
            "tx_pos": torch.from_numpy(src),                              # float32 (3,)
            "L": float(L),
            "W": float(W),
            "H": float(H),
            "alpha": float(attrs["alpha"]),
            "room_id": int(room_id),
        }

    def room_ids(self):
        return list(self.room_id_to_dims.keys())

    @property
    def room_id_map(self) -> dict[int, tuple[float, float, float]]:
        """Maps internal ordinal room index → (L, W, H) tuple."""
        return self.room_id_to_dims

    def get_room_attrs(self, room_id: int) -> dict:
        dims = self.room_id_to_dims[room_id]
        return self._cache[dims]["attrs"]
