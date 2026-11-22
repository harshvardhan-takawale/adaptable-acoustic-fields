"""P4-9 (D181 A8, A11): warm start + closed-form LR restart, and the CTL step's dedicated batch generator.

Behaviours under test (Rule 8 -- outputs and states, not "the checkpoint loads"):
  * `restart_lr` is the stated closed form (eta_min at the restart, the peak after warm-up, eta_min at the horizon).
  * A warm start holds EXACTLY the source checkpoint's weights and Adam moments, starts at its iteration, and its lr
    follows the restart -- NOT the loaded lr (the trap: a plain cosine continues from eta_min and never restarts).
  * A resume of the warm-started run wins over init_ckpt and continues the same schedule.
  * Refusals: the source's own directory, an architecture key that differs, a schedule without init and vice versa.
  * The default path is untouched: the original CosineAnnealingLR, the original `_step`.
  * p49_mode "ctl": batch indices do not depend on the global RNG, and survive a checkpoint round trip.
"""
from __future__ import annotations

import json
import math
import shutil

import pytest
import torch

from aaf.train.multi_room_2d_mat import restart_lr

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="tinycudann needs a GPU")
MIX = "configs/sweeps_2d_mat/p4_7_c500_cx500_manifest.json"
PEAK, LO = 1.25e-4, 5.0e-5


def test_restart_lr_closed_form():
    W, T = 1000, 30000
    assert restart_lr(0, PEAK, LO, W, T) == pytest.approx(LO)
    assert restart_lr(W // 2, PEAK, LO, W, T) == pytest.approx((LO + PEAK) / 2)
    assert restart_lr(W, PEAK, LO, W, T) == pytest.approx(PEAK)
    mid = W + (T - W) // 2
    assert restart_lr(mid, PEAK, LO, W, T) == pytest.approx((LO + PEAK) / 2, rel=1e-3)
    assert restart_lr(T, PEAK, LO, W, T) == pytest.approx(LO)
    assert restart_lr(T + 500, PEAK, LO, W, T) == pytest.approx(LO)          # clamped past the horizon
    seq = [restart_lr(e, PEAK, LO, W, T) for e in range(W, T + 1, 997)]
    assert all(a >= b for a, b in zip(seq, seq[1:]))                          # monotone decay after warm-up
    assert restart_lr(0, PEAK, LO, 0, T) == pytest.approx(PEAK)              # no warm-up: starts at the peak


def _base(tmp_path, **kw):
    import yaml

    from aaf.train.multi_room_2d_mat import P32TrainCfg
    m = json.load(open(MIX))
    rows = [r for r in m["configs"] if r["origin"] == "C500"][:2] + [r for r in m["configs"] if r["origin"] == "CX500"][:2]
    man = tmp_path / "man.json"
    if not man.exists():
        man.write_text(json.dumps(dict(m, configs=rows)))
    base = yaml.safe_load(open("configs/sweep_2d_mat/P4_8_R0.yaml"))
    base = {k: v for k, v in base.items() if k in P32TrainCfg.__dataclass_fields__}
    base.update({"configs_manifest": str(man), "n_iters": 6, "val_max_configs": 2, "early_stop_warmup": 6})
    base.update(kw)
    return P32TrainCfg(**base)


def _source(tmp_path):
    """A default (P4-8 R0 recipe) run with 3 optimizer steps and a FULL checkpoint at iteration 3."""
    from aaf.train.multi_room_2d_mat import P32Trainer
    a = P32Trainer(_base(tmp_path), tmp_path / "src")
    for it in (1, 2, 3):
        a._step_fn(it)
    a.save_ckpt(3)
    return a, tmp_path / "src" / "ckpt_iter0000003.pt"


def _warm(tmp_path, ck, out="warm", **kw):
    from aaf.train.multi_room_2d_mat import P32Trainer
    cfg = _base(tmp_path, n_iters=9, early_stop_warmup=9, init_ckpt=str(ck), lr_restart_peak=PEAK,
                lr_warmup_iters=2, lr_restart_from_iter=3, **kw)
    return P32Trainer(cfg, tmp_path / out)


@GPU
def test_warm_start_loads_weights_and_moments_and_restarts_the_lr(tmp_path):
    a, ck = _source(tmp_path)
    b = _warm(tmp_path, ck)
    assert b.start_iter == 3
    sa, sb = a.model.state_dict(), b.model.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    oa, ob = a.opt.state_dict()["state"], b.opt.state_dict()["state"]
    assert oa.keys() == ob.keys() and len(oa) > 0
    for k in oa:
        for n in oa[k]:
            va, vb = oa[k][n], ob[k][n]
            assert (torch.equal(va, vb.to(va.device)) if torch.is_tensor(va) else va == vb), (k, n)
    # The lr is the restart's, not the loaded one: eta_min at e=0, the peak after the 2-step warm-up, eta_min at 6.
    lrs = [b.opt.param_groups[0]["lr"]]
    for _ in range(6):
        b.opt.step()                                                          # no grads: a no-op step for the counter
        b.sched.step()
        lrs.append(b.opt.param_groups[0]["lr"])
    want = [restart_lr(e, PEAK, LO, 2, 6) for e in range(7)]
    assert lrs == pytest.approx(want, rel=1e-9)
    assert lrs[2] == pytest.approx(PEAK) and lrs[0] == pytest.approx(LO) and lrs[6] == pytest.approx(LO)


@GPU
def test_trap_a_plain_cosine_from_a_finished_run_never_restarts(tmp_path):
    """The failure A8 guards against, demonstrated on torch itself: CosineAnnealingLR recurses from the loaded lr."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([p], lr=2e-4)
    s = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=LO)
    for _ in range(10):
        opt.step()
        s.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(LO)                    # a finished run sits at eta_min
    st = opt.state_dict()
    opt2 = torch.optim.Adam([p], lr=2e-4)
    s2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=20, eta_min=LO)
    opt2.load_state_dict(st)
    for _ in range(5):
        opt2.step()
        s2.step()
    assert opt2.param_groups[0]["lr"] == pytest.approx(LO)                   # ... and never leaves it


@GPU
def test_resume_of_a_warm_run_wins_over_init_and_keeps_the_schedule(tmp_path):
    _, ck = _source(tmp_path)
    b = _warm(tmp_path, ck)
    for it in (4, 5):
        b._step_fn(it)
    b.save_ckpt(5)
    c = _warm(tmp_path, ck)                                                  # same dir: must RESUME, not re-init
    assert c.start_iter == 5
    sb, sc = b.model.state_dict(), c.model.state_dict()
    assert all(torch.equal(sb[k], sc[k]) for k in sb)
    assert c.sched.state_dict()["last_epoch"] == b.sched.state_dict()["last_epoch"] == 2
    assert c.opt.param_groups[0]["lr"] == pytest.approx(b.opt.param_groups[0]["lr"], rel=1e-12)
    c.opt.step()
    c.sched.step()
    assert c.opt.param_groups[0]["lr"] == pytest.approx(restart_lr(3, PEAK, LO, 2, 6), rel=1e-9)


@GPU
def test_warm_start_refusals(tmp_path):
    from aaf.train.multi_room_2d_mat import P32Trainer
    _, ck = _source(tmp_path)
    # Pointing the run at the source's OWN directory is refused by the resume guard (init_ckpt differs) ...
    with pytest.raises(RuntimeError, match="refusing to resume"):
        P32Trainer(_base(tmp_path, n_iters=9, early_stop_warmup=9, init_ckpt=str(ck), lr_restart_peak=PEAK,
                         lr_warmup_iters=2, lr_restart_from_iter=3), tmp_path / "src")
    # ... and, where no resumable checkpoint shadows it, by the warm start itself.
    own = tmp_path / "w1"
    own.mkdir()
    shutil.copy(ck, own / "init.pt")
    with pytest.raises(RuntimeError, match="own output dir"):
        P32Trainer(_base(tmp_path, n_iters=9, early_stop_warmup=9, init_ckpt=str(own / "init.pt"),
                         lr_restart_peak=PEAK, lr_warmup_iters=2, lr_restart_from_iter=3), own)
    with pytest.raises(RuntimeError, match="token_pool"):
        _warm(tmp_path, ck, out="w2", token_pool="masked_mean")
    with pytest.raises(RuntimeError, match="lr_restart_from_iter"):
        P32Trainer(_base(tmp_path, n_iters=9, early_stop_warmup=9, init_ckpt=str(ck), lr_restart_peak=PEAK,
                         lr_warmup_iters=2, lr_restart_from_iter=4), tmp_path / "w3")
    with pytest.raises(ValueError, match="go together"):
        P32Trainer(_base(tmp_path, init_ckpt=str(ck)), tmp_path / "w4")
    with pytest.raises(ValueError, match="go together"):
        P32Trainer(_base(tmp_path, lr_restart_peak=PEAK, lr_restart_from_iter=3), tmp_path / "w5")
    with pytest.raises(ValueError, match="early_stop_warmup"):
        P32Trainer(_base(tmp_path, n_iters=9, early_stop_warmup=5, init_ckpt=str(ck), lr_restart_peak=PEAK,
                         lr_warmup_iters=2, lr_restart_from_iter=3), tmp_path / "w6")


@GPU
def test_default_path_is_the_original_cosine_and_step(tmp_path):
    from aaf.train.multi_room_2d_mat import P32Trainer
    t = P32Trainer(_base(tmp_path), tmp_path / "d")
    assert type(t.sched) is torch.optim.lr_scheduler.CosineAnnealingLR
    assert t.sched.T_max == 6 and t.sched.eta_min == pytest.approx(LO)
    assert t._step_fn == t._step and not hasattr(t, "_p49_gen")


@GPU
def test_ctl_batches_are_independent_of_the_global_rng_and_survive_a_round_trip(tmp_path):
    from aaf.train.multi_room_2d_mat import P32Trainer
    x = P32Trainer(_base(tmp_path, p49_mode="ctl"), tmp_path / "x")
    y = P32Trainer(_base(tmp_path, p49_mode="ctl"), tmp_path / "y")
    assert x._step_fn == x._step_p49
    bx = x._p49_batch()
    torch.rand(1000, device=y.device)                                        # e.g. DIR-G's extra jitter draws
    torch.randint(0, 9, (77,), device=y.device)
    by = y._p49_batch()
    for (cx, rx), (cy, ry) in zip(bx, by):
        assert torch.equal(cx, cy) and torch.equal(rx, ry)
    m = x._step_fn(1)                                                        # a real CTL step runs
    assert math.isfinite(m["loss"])
    x.save_ckpt(1)
    nxt = x._p49_batch()
    z = P32Trainer(_base(tmp_path, p49_mode="ctl"), tmp_path / "x")           # resume restores the generator
    assert z.start_iter == 1
    nz = z._p49_batch()
    for (c1, r1), (c2, r2) in zip(nxt, nz):
        assert torch.equal(c1, c2) and torch.equal(r1, r2)
    assert not torch.equal(bx[0][1], nxt[0][1])                                # the stream advances
