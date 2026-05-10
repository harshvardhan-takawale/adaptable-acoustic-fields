"""ShoeboxDataset basics: room_filter, item shapes, room_id consistency."""
import numpy as np
import pytest
import torch
from pathlib import Path

from aaf.data.loader import ShoeboxDataset


REPO_ROOT = Path(__file__).resolve().parent.parent
DENSE_YAML = REPO_ROOT / "configs" / "sweeps" / "dense.yaml"


def test_single_room_filter_yields_64_samples():
    ds = ShoeboxDataset(sweep_yaml=DENSE_YAML, split="train", room_filter=[3.0])
    assert len(ds) == 64
    assert ds.L_list == [3.0]


def test_item_keys_and_shapes():
    ds = ShoeboxDataset(sweep_yaml=DENSE_YAML, split="train", room_filter=[3.0])
    item = ds[0]
    expected = {"H_complex", "rir_time", "rx_pos", "tx_pos", "L", "W", "alpha", "room_id"}
    assert set(item.keys()) == expected

    assert item["H_complex"].dtype == torch.complex64
    assert item["H_complex"].shape == (ds.n_freq_bins,)
    assert item["rir_time"].dtype == torch.float32
    assert item["rir_time"].shape == (ds.n_time_samples,)
    assert item["rx_pos"].dtype == torch.float32
    assert item["rx_pos"].shape == (2,)
    assert item["tx_pos"].dtype == torch.float32
    assert item["tx_pos"].shape == (2,)
    assert isinstance(item["L"], float)
    assert isinstance(item["W"], float)
    assert isinstance(item["alpha"], float)
    assert isinstance(item["room_id"], int)


def test_room_id_consistent_within_room():
    ds = ShoeboxDataset(sweep_yaml=DENSE_YAML, split="train", room_filter=[3.0])
    room_ids = [ds[i]["room_id"] for i in range(len(ds))]
    assert all(r == room_ids[0] for r in room_ids), (
        f"single-room filter should yield exactly one room_id; got {set(room_ids)}"
    )


def test_multi_room_assigns_distinct_ordinals():
    ds = ShoeboxDataset(sweep_yaml=DENSE_YAML, split="train", room_filter=[3.0, 4.5, 6.0])
    assert len(ds) == 3 * 64
    assert ds.L_list == [3.0, 4.5, 6.0]
    # First 64 → room_id 0; next 64 → 1; next 64 → 2.
    assert ds[0]["room_id"] == 0
    assert ds[63]["room_id"] == 0
    assert ds[64]["room_id"] == 1
    assert ds[127]["room_id"] == 1
    assert ds[128]["room_id"] == 2
    assert ds[191]["room_id"] == 2


def test_room_filter_no_overlap_raises():
    with pytest.raises(ValueError, match="no overlap"):
        ShoeboxDataset(sweep_yaml=DENSE_YAML, split="train", room_filter=[2.5])


def test_invalid_split_raises():
    with pytest.raises(ValueError, match="split"):
        ShoeboxDataset(sweep_yaml=DENSE_YAML, split="other", room_filter=[3.0])
