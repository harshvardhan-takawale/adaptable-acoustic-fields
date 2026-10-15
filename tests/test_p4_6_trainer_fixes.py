"""P4-6 (D160): two trainer fixes, tested on behaviour (Rule 8).

* `validate()` renders with the renderer in EVAL mode (no ray-fan jitter) and restores training
  mode afterwards. Before the fix the renderer stayed in training mode, so two validations at
  fixed weights disagreed (D150(d)). The test asserts the two now agree EXACTLY, that the render
  saw eval mode, and -- so the test has teeth -- that the same render in training mode does NOT
  repeat.
* `_prune_ckpts(keep_last, keep_every)`: keep_every = 0 is the original rule; keep_every = 10000
  with keep_last = 1 keeps every 10K multiple plus the newest.
All tests import the trainer, which imports tinycudann, so they need a GPU node.
"""
from __future__ import annotations

import pytest
import torch

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="tinycudann needs a GPU")
N_BAND = 601


def _stub_trainer():
    from aaf.renderers.freq_2d import FreqRenderer2D
    from aaf.train.multi_room_2d_mat import P32Trainer
    t = P32Trainer.__new__(P32Trainer)
    t.model = torch.nn.Linear(1, 1)
    t.renderer = FreqRenderer2D(n_azi=64)
    t.renderer.train()
    t.val_cfg_ids = [torch.tensor(0)]
    t.val_rx_idx = torch.arange(4)
    t.H = torch.ones(1, 4, N_BAND, dtype=torch.complex64)
    t.loss_lo_bin = 0
    t.scalars = []
    t.n_band = N_BAND
    t.seen_modes = []

    def _render(ci, ri):
        # A stand-in for the field render whose output depends on the fan, as the real one does.
        t.seen_modes.append(t.renderer.training)
        d = t.renderer._ray_directions_2d(torch.device("cpu"))       # [n_azi, 2]
        w = torch.linspace(0.5, 2.0, N_BAND)
        # the FIRST ray's direction: cos(jitter) in training mode, exactly 1 in eval mode (a sum over
        # the whole evenly spaced fan would be jitter-invariant and give the test no teeth)
        amp = 1.0 + 0.3 * d[0, 0] + 0.1 * torch.arange(ci.numel())[:, None]
        return (amp * w).to(torch.complex64)
    t._render = _render
    return t


@GPU
def test_validate_uses_eval_renderer_and_restores_train_mode():
    t = _stub_trainer()
    a = t.validate(1000)["lsd_db"]
    b = t.validate(2000)["lsd_db"]
    assert a == b, "two validations at fixed weights must agree exactly"
    assert t.seen_modes and not any(t.seen_modes), "validation rendered in training mode"
    assert t.renderer.training, "validate() must restore the renderer's training mode"


@GPU
def test_the_stub_render_would_differ_in_training_mode():
    """Teeth: with jitter on, the same render does not repeat, so the test above can fail."""
    t = _stub_trainer()
    torch.manual_seed(0)
    x = t._render(torch.zeros(4, dtype=torch.long), torch.arange(4))
    y = t._render(torch.zeros(4, dtype=torch.long), torch.arange(4))
    assert not torch.equal(x, y)


def _touch(d, iters):
    for it in iters:
        (d / "ckpt_iter{:07d}.pt".format(it)).write_bytes(b"x")


def _left(d):
    return sorted(int(p.stem.replace("ckpt_iter", "")) for p in d.glob("ckpt_iter*.pt"))


@GPU
def test_prune_keep_every_10k_plus_newest(tmp_path):
    from aaf.train.multi_room_2d_mat import _prune_ckpts
    _touch(tmp_path, range(2000, 60001, 2000))
    _prune_ckpts(tmp_path, keep_last=1, keep_every=10000)
    assert _left(tmp_path) == [10000, 20000, 30000, 40000, 50000, 60000]
    _touch(tmp_path, [62000])
    _prune_ckpts(tmp_path, keep_last=1, keep_every=10000)
    assert _left(tmp_path) == [10000, 20000, 30000, 40000, 50000, 60000, 62000]
    _touch(tmp_path, [64000])
    _prune_ckpts(tmp_path, keep_last=1, keep_every=10000)
    assert _left(tmp_path) == [10000, 20000, 30000, 40000, 50000, 60000, 64000]


@GPU
def test_prune_keep_every_off_is_the_original_rule(tmp_path):
    from aaf.train.multi_room_2d_mat import _prune_ckpts
    _touch(tmp_path, range(2000, 20001, 2000))
    _prune_ckpts(tmp_path, keep_last=3, keep_every=0)
    assert _left(tmp_path) == [16000, 18000, 20000]


@GPU
def test_prune_keep_iters_spares_the_4k_kill_signal_checkpoint(tmp_path):
    from aaf.train.multi_room_2d_mat import _prune_ckpts
    _touch(tmp_path, range(2000, 24001, 2000))
    _prune_ckpts(tmp_path, keep_last=1, keep_every=10000, keep_iters={4000})
    assert _left(tmp_path) == [4000, 10000, 20000, 24000]
