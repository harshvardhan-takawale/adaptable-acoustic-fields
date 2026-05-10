"""Unit tests for `SingleRoomTrainer._check_early_stop`.

The helper is a pure-Python function on `self.scalars` and `self.cfg` — we can
exercise it via a minimal stub object without instantiating the (CUDA-only)
trainer.
"""
from types import SimpleNamespace

import pytest


def _stub_trainer(scalars):
    from aaf.train.single_room import TrainCfg, SingleRoomTrainer

    self = SimpleNamespace()
    self.cfg = TrainCfg()
    self.scalars = scalars
    self._val_total_loss = SingleRoomTrainer._val_total_loss.__get__(self)
    self._check_early_stop = SingleRoomTrainer._check_early_stop.__get__(self)
    return self


def _val_row(it, total_loss):
    """Construct a fake val scalar row whose weighted sum yields `total_loss`."""
    return {
        "phase": "val",
        "iter": it,
        "L_spec_real": total_loss / 2.0,
        "L_spec_imag": total_loss / 2.0,
        "L_amp": 0.0,
        "L_phase": 0.0,
    }


def test_no_stop_during_warmup():
    rows = [_val_row(500, 1.0), _val_row(1000, 0.5), _val_row(1500, 0.25), _val_row(2000, 0.125)]
    t = _stub_trainer(rows)
    stop, _ = t._check_early_stop(2000)
    assert stop is False, "must not stop at warmup boundary even if improvement is small"


def test_stop_when_plateau():
    rows = []
    for it in [500, 1000, 1500, 2000]:
        rows.append(_val_row(it, 1.0 - 0.1 * (it / 500.0)))   # 0.9, 0.8, 0.7, 0.6
    for it in [2500, 3000, 3500, 4000]:
        rows.append(_val_row(it, 0.598))                       # < 1% better than 0.6
    t = _stub_trainer(rows)
    stop, why = t._check_early_stop(4000)
    assert stop is True, "should stop on near-flat plateau"
    assert "improvement" in why


def test_no_stop_while_improving():
    rows = []
    for it in [500, 1000, 1500, 2000]:
        rows.append(_val_row(it, 1.0 / (it / 500.0)))          # 1.0, 0.5, 0.33, 0.25
    for it in [2500, 3000, 3500, 4000]:
        rows.append(_val_row(it, 0.5 / (it / 500.0)))          # 0.1, 0.083, 0.071, 0.0625
    t = _stub_trainer(rows)
    stop, _ = t._check_early_stop(4000)
    assert stop is False, "decreasing val loss must not trigger stop"


def test_no_stop_when_window_empty():
    """If patience > current_iter, before-window is empty and we can't compare."""
    rows = [_val_row(500, 1.0), _val_row(1000, 0.5)]
    t = _stub_trainer(rows)
    # iter 1500 < warmup (2000); also boundary = 1500-2000 = -500 → ineligible
    stop, _ = t._check_early_stop(1500)
    assert stop is False


def test_first_eligible_check():
    """At iter 2500 (just past warmup), boundary = 500. Before = {500}, window = {1000,1500,2000,2500}.
    With strong improvement (0.1 vs 1.0), we must NOT stop.
    """
    rows = [_val_row(it, v) for it, v in
            [(500, 1.0), (1000, 0.5), (1500, 0.3), (2000, 0.15), (2500, 0.10)]]
    t = _stub_trainer(rows)
    stop, _ = t._check_early_stop(2500)
    assert stop is False
