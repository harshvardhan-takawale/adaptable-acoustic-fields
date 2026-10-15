"""P4-5 a3 (D150): the floored log-amplitude loss. Rule 8 -- a loss option adds no parameters, so the
tests assert on loss VALUES and GRADIENTS:
  * off (amp_eps=None) is bit-identical to the pre-P4-5 loss;
  * on, L_amp changes and the other three terms do not;
  * near a null -- prediction and target both far below eps_b -- the floored L_amp is ~0 while the
    unfloored one is large: the property the arm exists for;
  * lo_bin / hi_bin slice the floor with the bins (an off-by-one there would silently mis-floor);
  * the resume guard covers both new keys.
All tests need a GPU: importing the trainer imports tinycudann.
"""
from __future__ import annotations

import torch

from tests.test_p4_5_hf_arms import GPU, N_BAND, _pair, _pre_change_losses


@GPU
def test_floor_off_is_bit_identical_to_the_pre_change_loss():
    from aaf.train.multi_room_2d_mat import _losses
    P, T = _pair(3)
    new, old = _losses(P, T, amp_eps=None), _pre_change_losses(P, T)
    for k in old:
        assert torch.equal(new[k], old[k]), k


@GPU
def test_floor_changes_l_amp_only():
    from aaf.train.multi_room_2d_mat import _losses
    P, T = _pair(4)
    eps = torch.full((N_BAND,), 0.5)
    on, off = _losses(P, T, amp_eps=eps), _losses(P, T)
    assert abs(float(on["L_amp"] - off["L_amp"])) > 1e-3
    for k in ("L_spec_real", "L_spec_imag", "L_phase"):
        assert torch.allclose(on[k], off[k], rtol=1e-6), k


@GPU
def test_floor_silences_near_null_disagreement():
    from aaf.train.multi_room_2d_mat import _losses
    rows = 4
    P = torch.full((rows, N_BAND), 1e-4, dtype=torch.complex64)      # -80 dB
    T = torch.full((rows, N_BAND), 1e-2, dtype=torch.complex64)      # -40 dB: 40 dB apart
    eps = torch.full((N_BAND,), 0.05)                                # floor at -26 dB
    floored = float(_losses(P, T, amp_eps=eps)["L_amp"])
    plain = float(_losses(P, T)["L_amp"])
    assert plain > 1.9                                               # log10 ratio of 2 decades
    assert floored < 0.1                                             # log10((0.05+1e-2)/(0.05+1e-4))


@GPU
def test_floor_is_sliced_with_lo_and_hi_bins():
    from aaf.train.multi_room_2d_mat import _losses
    P, T = _pair(5)
    eps = torch.linspace(0.01, 2.0, N_BAND)
    lo, hi = 20, 400
    got = _losses(P, T, lo_bin=lo, hi_bin=hi - lo, amp_eps=eps)["L_amp"]
    p, t, e = P[:, lo:hi], T[:, lo:hi], eps[lo:hi]
    want = (torch.log10(p.abs() + e) - torch.log10(t.abs() + e)).abs().sum() / (P.shape[0] * (N_BAND - lo))
    assert torch.allclose(got, want, rtol=1e-6)


@GPU
def test_resume_guard_covers_the_floor_keys():
    """A cross-resume between a floored and an unfloored arm would splice two objectives."""
    import inspect
    from aaf.train import multi_room_2d_mat as M
    src = inspect.getsource(M)
    guard = src[src.index("def _maybe_resume"):src.index("def _early_stop")]
    for k in ("loss_amp_floor_frac", "loss_amp_floor_ref"):
        assert '"%s"' % k in guard, k
        assert k in M.P32TrainCfg.__dataclass_fields__, k
