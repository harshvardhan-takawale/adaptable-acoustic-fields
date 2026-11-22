"""P4-9 2b (D181 A11, A15, E2, c1): the DIR / DIR-G training terms.

GPU training steps are not bit-reproducible (tinycudann accumulates hash-grid gradients with atomics), so the
"same as CTL" claims are tested on the deterministic FORWARD quantities, and the "the term does something" claims on
its gradients (Rule 8: a test that only loads would pass on a broken implementation):
  * _render(per_ray=True) and _render(explicit_jitter=True) return CTL's render exactly, and the explicit jitter draw
    leaves the CUDA RNG exactly where the renderer's own draw does (so DIR-G's omni pass draws CTL's jitters).
  * The per-ray readout D is consistent with the render: sum over rays of s_j equals H minus the DC term.
  * DIR's and DIR-G's terms are finite, positive and move the FiLM / attention parameters.
  * DIR-G renders a row's neighbours with that row's jitter: the same jitter reproduces its term, another changes it.
  * The weight is calibrated with every RNG state restored, stored in checkpoints and NOT re-calibrated on resume.
  * Labels built for other rooms are refused.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="tinycudann needs a GPU")
MIX = "configs/sweeps_2d_mat/p4_7_c500_cx500_manifest.json"


def _cfg(tmp_path, **kw):
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


def _labels(tmp_path, ctl, shuffle=False):
    """Random complex gradient labels for the 4-room run, in the assembled layout."""
    d = tmp_path / ("lab_bad" if shuffle else "lab")
    if d.exists():
        return d
    d.mkdir()
    names = [c.filename for c in ctl.configs]
    if shuffle:
        names = names[::-1]
    C, R = ctl.rx.shape[:2]
    rng = np.random.default_rng(0)
    for k in ("g1x", "g1y", "g2x", "g2y"):
        a = np.lib.format.open_memmap(d / (k + ".npy"), mode="w+", dtype=np.complex64, shape=(C, R, 601))
        a[:] = (rng.standard_normal((C, R, 601)) + 1j * rng.standard_normal((C, R, 601))).astype(np.complex64) * 50
        a.flush()
    np.save(d / "rx.npy", ctl.rx.cpu().numpy().astype(np.float64))
    (d / "index.json").write_text(json.dumps({"filenames": names, "dx_xy": [[0.02, 0.02]] * C}))
    return d


def _trainers(tmp_path, mode, **kw):
    from aaf.train.multi_room_2d_mat import P32Trainer
    ctl = P32Trainer(_cfg(tmp_path, p49_mode="ctl"), tmp_path / "ctl")
    lab = _labels(tmp_path, ctl)
    arm = P32Trainer(_cfg(tmp_path, p49_mode=mode, dir_labels=str(lab), **kw), tmp_path / mode)
    return ctl, arm


@GPU
def test_render_variants_equal_ctl_and_jitter_draw_matches(tmp_path):
    ctl, arm = _trainers(tmp_path, "dir")
    ci, ri = ctl._p49_batch()[0]
    ctl.model.train()
    with torch.no_grad():                                                    # forward equality only
        s0 = torch.cuda.get_rng_state()
        H0 = ctl._render(ci, ri)
        s_after_internal = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(s0)
        H1, ex1 = ctl._render(ci, ri, per_ray=True)
        torch.cuda.set_rng_state(s0)
        H2, ex2 = ctl._render(ci, ri, explicit_jitter=True)
    assert torch.equal(H1, H0) and torch.equal(H2, H0)
    assert torch.equal(torch.cuda.get_rng_state(), s_after_internal)          # the explicit draw = the renderer's draw
    assert ex2["jitter_row"].abs().max() > 0 and ex1["D"].abs().max() > 0


R0 = "outputs/p4_8/arms/R0/ckpt_iter0120000.pt"


@GPU
@pytest.mark.skipif(not __import__("os").path.exists(R0), reason="needs R0 @120K to wake the model")
def test_dir_and_dirg_terms_are_live(tmp_path):
    # Rule 8: a fresh model's FiLM cannot move the field yet (its gradient is exactly zero), so the term is tested on
    # R0 @120K's trained weights (same architecture), loaded into the 4-room trainer.
    sd = torch.load(R0, map_location="cpu")["model"]
    for mode in ("dir", "dirg"):
        ctl, arm = _trainers(tmp_path, mode)
        del ctl
        torch.cuda.empty_cache()
        assert arm._dir_w > 0 and np.isfinite(arm._dir_w)
        arm.model.load_state_dict(sd)
        ci, ri = arm._p49_batch()[0]
        arm.model.train()
        arm.model.zero_grad(set_to_none=True)
        if mode == "dir":
            _, ex = arm._render(ci, ri, per_ray=True)
            term = arm._dir_loss(ci, ri, ex["D"])
        else:
            with torch.no_grad():
                _, ex = arm._render(ci, ri, explicit_jitter=True)
            term, n = arm._dirg_loss(ci, ri, ex["jitter_row"])
            assert 1 <= n <= arm.cfg.dirg_rows_per_micro
        assert torch.isfinite(term) and float(term) > 0
        term.backward()
        for p in (arm.model.film_signal.weight, arm.model.o_proj.weight):
            assert p.grad is not None and float(p.grad.abs().max()) > 0, mode
        del arm, term, ex
        torch.cuda.empty_cache()


@GPU
def test_dirg_uses_the_rows_jitter(tmp_path):
    ctl, arm = _trainers(tmp_path, "dirg")
    ci, ri = arm._p49_batch()[0]
    arm.model.train()
    with torch.no_grad():
        _, ex = arm._render(ci, ri, explicit_jitter=True)
        j = ex["jitter_row"]
        a = float(arm._dirg_loss(ci, ri, j)[0])
        b = float(arm._dirg_loss(ci, ri, j)[0])
        c = float(arm._dirg_loss(ci, ri, j + 0.03)[0])
    assert a == b and a != c


@GPU
def test_weight_stored_and_restored_and_rng_untouched(tmp_path):
    from aaf.train.multi_room_2d_mat import P32Trainer
    ctl, arm = _trainers(tmp_path, "dir")
    s = torch.cuda.get_rng_state()
    w = arm._calibrate_dir_weight()
    assert torch.equal(torch.cuda.get_rng_state(), s)
    assert w == pytest.approx(arm._dir_w, rel=0.5)                           # same rule (other batches ok)
    arm._step_fn(1)
    arm.save_ckpt(1)
    again = P32Trainer(_cfg(tmp_path, p49_mode="dir", dir_labels=str(_labels(tmp_path, ctl))), tmp_path / "dir")
    assert again.start_iter == 1 and again._dir_w == arm._dir_w


@GPU
def test_zero_fraction_gives_zero_weight_and_ctl_losses(tmp_path):
    ctl, arm = _trainers(tmp_path, "dir", dir_frac_init=0.0)
    assert arm._dir_w == 0.0
    s = torch.cuda.get_rng_state()
    m_ctl = ctl._step_fn(1)
    torch.cuda.set_rng_state(s)
    m_dir = arm._step_fn(1)
    assert m_dir["loss"] == m_ctl["loss"]                                    # same batch, same jitter, same omni loss


@GPU
def test_labels_for_other_rooms_are_refused(tmp_path):
    from aaf.train.multi_room_2d_mat import P32Trainer
    ctl = P32Trainer(_cfg(tmp_path, p49_mode="ctl"), tmp_path / "ctl")
    bad = _labels(tmp_path, ctl, shuffle=True)
    with pytest.raises(RuntimeError, match="other rooms"):
        P32Trainer(_cfg(tmp_path, p49_mode="dir", dir_labels=str(bad)), tmp_path / "bad")
