"""HDF5 writer for one room of the 2D shoebox dataset.

Layout per file (one file per (L, W, α) tuple):

    /ism/H_complex          complex64    shape (N_rx, n_freq_bins)
    /ism/rir_time           float32      shape (N_rx, n_time_samples)
    /analytical/H_complex   complex64    shape (N_rx, n_freq_bins)
    /analytical/rir_time    float32      shape (N_rx, n_time_samples)

    Root attrs include all simulator metadata + git commit + build timestamp.

Native h5py complex64 storage works in h5py 3.11; we don't split into
real/imag pairs.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _flatten_attr(value: Any) -> Any:
    """h5py's attr writer can't store dicts or lists-of-lists directly; JSON it."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    return value


def write_room_to_h5(
    out_path: os.PathLike | str,
    ism_result: dict,
    analytical_result: dict,
    sweep_meta: dict | None = None,
) -> Path:
    """Write one room (ISM + analytical) to an HDF5 file. Overwrites if exists.

    Args:
        out_path: target HDF5 path. Parent dir is created if missing.
        ism_result: dict from `aaf.sim.ism_2d.simulate_room_2d`.
        analytical_result: dict from `aaf.sim.analytical_modal_2d.modal_rir_2d`.
        sweep_meta: optional extra metadata (e.g. sweep name) to record in attrs.

    Returns: resolved Path to the written file.
    """
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ism_meta = dict(ism_result["meta"])
    ana_meta = dict(analytical_result["meta"])

    # Cross-check shapes match — they must, since both consume the same cfg.
    if ism_result["H_complex"].shape != analytical_result["H_complex"].shape:
        raise ValueError(
            f"H_complex shape mismatch: ISM {ism_result['H_complex'].shape} vs "
            f"analytical {analytical_result['H_complex'].shape}"
        )
    if ism_result["rir_time"].shape != analytical_result["rir_time"].shape:
        raise ValueError(
            f"rir_time shape mismatch: ISM {ism_result['rir_time'].shape} vs "
            f"analytical {analytical_result['rir_time'].shape}"
        )

    with h5py.File(out_path, "w") as f:
        ism_grp = f.create_group("ism")
        ism_grp.create_dataset("H_complex", data=ism_result["H_complex"], compression="gzip")
        ism_grp.create_dataset("rir_time", data=ism_result["rir_time"], compression="gzip")
        for k, v in ism_meta.items():
            ism_grp.attrs[k] = _flatten_attr(v)

        ana_grp = f.create_group("analytical")
        ana_grp.create_dataset(
            "H_complex", data=analytical_result["H_complex"], compression="gzip"
        )
        ana_grp.create_dataset(
            "rir_time", data=analytical_result["rir_time"], compression="gzip"
        )
        for k, v in ana_meta.items():
            ana_grp.attrs[k] = _flatten_attr(v)

        # Root attrs: things that uniquely identify this room.
        root = {
            "L": ism_meta["L"],
            "W": ism_meta["W"],
            "alpha": ism_meta["alpha"],
            "fs": ism_meta["fs"],
            "c": ism_meta["c"],
            "n_time_samples": ism_meta["n_time_samples"],
            "n_freq_bins": ism_meta["n_freq_bins"],
            "n_rx": ism_result["H_complex"].shape[0],
            "source_pos": ism_meta["source_pos"],
            "receiver_pos": ism_meta["receiver_pos"],
            "ism_max_order": ism_meta["max_order"],
            "T60_sabine_2d": ism_meta["T60_sabine_2d"],
            "schroeder_freq_approx_hz": ism_meta["schroeder_freq_approx_hz"],
            "build_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "git_commit": _git_commit(),
            "versions": ism_meta.get("versions", {}),
        }
        if sweep_meta:
            root["sweep_meta"] = sweep_meta
        for k, v in root.items():
            f.attrs[k] = _flatten_attr(v)

    return out_path


def read_room_h5(path: os.PathLike | str) -> dict:
    """Round-trip helper for tests and the noise-floor report.

    Returns dict with keys: ism_H, ism_rir, ana_H, ana_rir, attrs (root attrs as
    a dict, JSON-decoded where applicable).
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        ism_H = f["ism/H_complex"][:]
        ism_rir = f["ism/rir_time"][:]
        ana_H = f["analytical/H_complex"][:]
        ana_rir = f["analytical/rir_time"][:]
        attrs: dict[str, Any] = {}
        for k, v in f.attrs.items():
            if isinstance(v, (bytes, bytearray)):
                v = v.decode()
            if isinstance(v, str):
                try:
                    v = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    pass
            attrs[k] = v
    return {
        "ism_H": ism_H,
        "ism_rir": ism_rir,
        "ana_H": ana_H,
        "ana_rir": ana_rir,
        "attrs": attrs,
    }


def room_filename(L: float, W: float, alpha: float) -> str:
    """Canonical filename for a single-room HDF5 file (2D)."""
    return f"L_{L:.2f}m_W_{W:.2f}m_alpha_{alpha:.2f}.h5"


def room_filename_3d(L: float, W: float, H: float) -> str:
    """Canonical filename for a single-room 3D HDF5 file.

    Format: ``L{L:.2f}_W{W:.2f}_H{H:.2f}.h5``. Alpha is implicit (fixed across
    all of Phase 2 at 0.15); kept out of the name to keep paths short.
    """
    return f"L{L:.2f}_W{W:.2f}_H{H:.2f}.h5"


def write_room_3d_to_h5(
    out_path: os.PathLike | str,
    ism_result: dict,
    analytical_result: dict,
    sweep_meta: dict | None = None,
) -> Path:
    """Write one 3D room (ISM + analytical) to an HDF5 file.

    Same layout as ``write_room_to_h5`` but with 3D-specific root attrs
    (``L, W, H, T60_sabine_3d, schroeder_freq_hz``).

    Returns: resolved Path to the written file.
    """
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ism_meta = dict(ism_result["meta"])
    ana_meta = dict(analytical_result["meta"])

    if ism_result["H_complex"].shape != analytical_result["H_complex"].shape:
        raise ValueError(
            f"H_complex shape mismatch: ISM {ism_result['H_complex'].shape} vs "
            f"analytical {analytical_result['H_complex'].shape}"
        )
    if ism_result["rir_time"].shape != analytical_result["rir_time"].shape:
        raise ValueError(
            f"rir_time shape mismatch: ISM {ism_result['rir_time'].shape} vs "
            f"analytical {analytical_result['rir_time'].shape}"
        )

    with h5py.File(out_path, "w") as f:
        ism_grp = f.create_group("ism")
        ism_grp.create_dataset("H_complex", data=ism_result["H_complex"], compression="gzip")
        ism_grp.create_dataset("rir_time", data=ism_result["rir_time"], compression="gzip")
        for k, v in ism_meta.items():
            ism_grp.attrs[k] = _flatten_attr(v)

        ana_grp = f.create_group("analytical")
        ana_grp.create_dataset(
            "H_complex", data=analytical_result["H_complex"], compression="gzip"
        )
        ana_grp.create_dataset(
            "rir_time", data=analytical_result["rir_time"], compression="gzip"
        )
        for k, v in ana_meta.items():
            ana_grp.attrs[k] = _flatten_attr(v)

        root = {
            "L": ism_meta["L"],
            "W": ism_meta["W"],
            "H": ism_meta["H"],
            "alpha": ism_meta["alpha"],
            "fs": ism_meta["fs"],
            "c": ism_meta["c"],
            "n_time_samples": ism_meta["n_time_samples"],
            "n_freq_bins": ism_meta["n_freq_bins"],
            "n_rx": ism_result["H_complex"].shape[0],
            "source_pos": ism_meta["source_pos"],
            "receiver_pos": ism_meta["receiver_pos"],
            "ism_max_order": ism_meta["max_order"],
            "T60_sabine_3d": ism_meta["T60_sabine_3d"],
            "schroeder_freq_hz": ism_meta["schroeder_freq_hz"],
            "build_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "git_commit": _git_commit(),
            "versions": ism_meta.get("versions", {}),
        }
        if sweep_meta:
            root["sweep_meta"] = sweep_meta
        for k, v in root.items():
            f.attrs[k] = _flatten_attr(v)

    return out_path
