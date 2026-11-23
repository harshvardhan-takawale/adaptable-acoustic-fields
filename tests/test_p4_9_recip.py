"""P4-9 X3 (D191): the reciprocity swap on CTL's step (`p49_recip_p`).

Rule 8: the swap adds no parameter, so a checkpoint round-trip cannot see it. The tests assert BEHAVIOUR:
  * p = 0 is CTL's step: no swap generator, the same first-step loss under the same CUDA RNG state.
  * an all-False mask renders exactly CTL's field; an all-True mask renders a DIFFERENT field (on R0 @120K's trained
    weights, so the network is awake), equal to the render with receiver and source exchanged by hand.
  * the swap draws leave CTL's batch generator and the CUDA RNG untouched, so the batch indices are CTL's.
  * the swap generator's state is saved and restored: a resumed run draws the same masks as an uninterrupted one.
  * refusals: p outside [0, 1), or p > 0 outside p49_mode 'ctl'.
"""
from __future__ import annotations

import os

import pytest
import torch

from tests.test_p4_9_dir import _cfg

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="tinycudann needs a GPU")
R0 = "outputs/p4_8/arms/R0/ckpt_iter0120000.pt"


def _pair(tmp_path, p=0.9):
    from aaf.train.multi_room_2d_mat import P32Trainer
    ctl = P32Trainer(_cfg(tmp_path, p49_mode="ctl"), tmp_path / "ctl")
    arm = P32Trainer(_cfg(tmp_path, p49_mode="ctl", p49_recip_p=p), tmp_path / "rcp")
    return ctl, arm


@GPU
def test_refusals(tmp_path):
    from aaf.train.multi_room_2d_mat import P32Trainer
    for kw in ({"p49_mode": "ctl", "p49_recip_p": 1.0}, {"p49_mode": "ctl", "p49_recip_p": -0.1},
               {"p49_mode": "", "p49_recip_p": 0.25}):
        with pytest.raises(ValueError):
            P32Trainer(_cfg(tmp_path, **kw), tmp_path / "bad")


@GPU
def test_zero_is_ctl(tmp_path):
    ctl, arm = _pair(tmp_path, p=0.0)
    assert not hasattr(arm, "_p49_swap_gen") and not hasattr(arm, "_swap_ok")
    s = torch.cuda.get_rng_state()
    a = ctl._step_fn(1)
    torch.cuda.set_rng_state(s)
    b = arm._step_fn(1)
    assert a["loss"] == b["loss"] and "recip_frac" not in b


@GPU
@pytest.mark.skipif(not os.path.exists(R0), reason="needs R0 @120K to wake the model")
def test_swap_changes_the_render_and_nothing_else(tmp_path):
    ctl, arm = _pair(tmp_path)
    sd = torch.load(R0, map_location="cpu")["model"]
    arm.model.load_state_dict(sd)
    assert arm._recip_info["rows_swappable"] > 0.5
    # batch indices: CTL's, and the swap draw touches neither CTL's generator nor the CUDA RNG
    g0, c0 = arm._p49_gen.get_state(), torch.cuda.get_rng_state()
    ci, ri = arm._p49_batch()[0]
    ci_c, ri_c = ctl._p49_batch()[0]
    assert torch.equal(ci, ci_c) and torch.equal(ri, ri_c)
    g1 = arm._p49_gen.get_state()
    sw = arm._p49_swap(ci, ri, {})
    assert torch.equal(arm._p49_gen.get_state(), g1) and torch.equal(torch.cuda.get_rng_state(), c0)
    assert not torch.equal(g0, g1)
    arm.model.eval()
    arm.renderer.eval()
    with torch.no_grad():
        H = arm._render(ci, ri)
        H_f = arm._render(ci, ri, swap=torch.zeros_like(sw))
        H_t = arm._render(ci, ri, swap=torch.ones_like(sw))
        rx, tx = arm.rx[ci, ri], arm.src_per[ci]
        by_hand = torch.zeros_like(H)
        for i in range(ci.numel()):                     # one row per call: receiver and source exchanged by hand
            L, W = arm.geom_dims[int(arm.geom_id[ci[i]])]
            out = arm.renderer(arm.model, tx[i:i + 1], rx[i:i + 1], torch.zeros(2, device=H.device),
                               torch.tensor([L, W], device=H.device, dtype=torch.float32),
                               z_s=arm.cond[ci[i:i + 1]])
            by_hand[i] = out[0, arm.band[0]:arm.band[1]]
    assert torch.equal(H_f, H)
    # compared from 25 Hz up: bins 0-40 carry the DC head's near-constant term (scale ~1e3), which would swamp a max-
    # relative difference (it read 0.4% there while the signal band moves by tens of percent)
    S, St, Sh = H[:, 50:], H_t[:, 50:], by_hand[:, 50:]
    rel = float((St - S).norm() / S.norm())
    assert not torch.equal(H_t, H) and rel > 0.05, rel
    assert float((St - Sh).norm() / Sh.norm()) < 1e-4


@GPU
def test_swap_generator_saved_and_restored(tmp_path):
    from aaf.train.multi_room_2d_mat import P32Trainer
    _, arm = _pair(tmp_path)
    m = arm._step_fn(1)
    assert 0.0 < m["recip_frac"] <= 1.0                                    # a sampled fraction (16 rows), not p
    arm.save_ckpt(1)
    ci, ri = arm._p49_batch()[0]
    nxt = arm._p49_swap(ci, ri, {})
    again = P32Trainer(_cfg(tmp_path, p49_mode="ctl", p49_recip_p=0.9), tmp_path / "rcp")
    assert again.start_iter == 1
    ci2, ri2 = again._p49_batch()[0]
    assert torch.equal(ci2, ci) and torch.equal(ri2, ri)
    assert torch.equal(again._p49_swap(ci2, ri2, {}), nxt)
