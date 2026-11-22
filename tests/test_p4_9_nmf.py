"""P4-9 2c (D180 NMF-LB; D181 A10, E4): the neural modal field.

  * S-a (D180): NMF's field formula with the TRUE eigenpairs (every mode to 600 Hz) reproduces m1's field -- the
    renders agree to 1e-6 and the off-lattice r at 25-100 Hz within 0.002 on dense rooms of each family.
  * The subspace loss is 0 for the true modes, invariant to sign flips and permutations, > 0 for another span, and the
    soft cutoff makes it continuous when two modes cross near the cutoff (A10).
  * GPU: eigenfrequencies are sorted by construction; FiLM is identity at init (the vacuity) and, woken, changes the modes
    (Rule 8); a 2-step training smoke on two labelled rooms gives finite losses.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

MODAL = Path("outputs/p4_9/modal")
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="tinycudann needs a GPU")
ROOMS = ["L5.54_W5.28_d1.42_w1.52", "UL6.36_W5.44_NEd2.18w2.16x0.00_NWd1.72w1.42x0.00",
         "C3L6.74_W4.58_NWd0.08w2.34_NEd1.78w2.44_SEd1.38w0.50"]


def _m1(stem):
    from scripts.p4_9_sensitivity import modal_field
    z = dict(np.load(MODAL / "{}__exact.npz".format(stem)))
    return z, modal_field(z)


@pytest.mark.skipif(not MODAL.exists(), reason="needs the P4-9 modal solves")
@pytest.mark.parametrize("stem", ROOMS)
def test_s_a_true_eigenpairs_reproduce_m1(stem):
    from aaf.eval.modal_prior import C_FIT, C0, DT
    from aaf.models.nmf_2d import NMF2D
    z, H_m1 = _m1(stem)
    meta = json.loads(str(z["meta"]))
    c_pred = meta["dA"] / (C0 * DT) ** 2
    bins = torch.arange(0, 601)
    zero = torch.as_tensor(z["psi_r"][:, :1] * z["td0"][None] * (C_FIT / c_pred))
    H = NMF2D.field(torch.as_tensor(np.sqrt(z["k2"][1:])), torch.as_tensor(z["eta"][1:]),
                    torch.as_tensor(z["psi_r"][:, 1:]), torch.as_tensor(z["psi_s"][1:]), bins,
                    torch.as_tensor(z["S"]), C_FIT, zero).numpy()
    H[:, 0] = H_m1[:, 0]                                                      # bin 0: m1 zeroes the nonzero modes
    assert np.abs(H[:, 50:] - H_m1[:, 50:]).max() / np.abs(H_m1[:, 50:]).max() < 1e-6
    from scripts.p4_7_m1_modal import scores
    rx = z["rx"]
    k = (rx - 0.12) / 0.06
    off = ~np.all(np.abs(k - np.round(k)) < 1e-6, axis=1)
    assert abs(scores(H[off], H_m1[off])["25-100"] - 1.0) < 0.002


def test_subspace_loss_invariances_and_soft_cutoff():
    from aaf.models.nmf_2d import soft_sqrt_weights, subspace_loss
    g = torch.Generator().manual_seed(0)
    n_q, n = 400, 8
    wq = torch.rand(n_q, generator=g) + 0.5
    Q, _ = torch.linalg.qr(torch.randn(n_q, n, generator=g, dtype=torch.float64) * torch.sqrt(wq.double())[:, None])
    psi = (Q / torch.sqrt(wq.double())[:, None]).float()
    w = torch.ones(n)
    assert float(subspace_loss(psi, w, psi, w, wq)) < 1e-5
    flip = psi * torch.tensor([1, -1, 1, -1, 1, 1, -1, 1.0])
    perm = flip[:, torch.randperm(n, generator=g)]
    assert float(subspace_loss(perm, w, psi, w, wq)) < 1e-5
    other = torch.randn(n_q, n, generator=g)
    assert float(subspace_loss(other, w, psi, w, wq)) > 0.1
    k = torch.linspace(1.0, 2.0, n)
    ka, kb = 1.6, 1.9
    k2 = k.clone()
    k2[[5, 6]] = k[[6, 5]] + torch.tensor([1e-4, -1e-4])                   # two modes cross near the cutoff
    l1 = float(subspace_loss(psi, soft_sqrt_weights(k, ka, kb), psi, soft_sqrt_weights(k, ka, kb), wq))
    l2 = float(subspace_loss(psi, soft_sqrt_weights(k2, ka, kb), psi, soft_sqrt_weights(k, ka, kb), wq))
    assert l1 < 1e-5 and l2 < 0.2                                          # a small k change -> a bounded change


@GPU
def test_k_sorted_and_film_live():
    from aaf.models.nmf_2d import NMF2D
    m = NMF2D().cuda()
    c = torch.randn(3, 64, device="cuda")
    k = m.k_modes(c)
    assert (k[:, 1:] > k[:, :-1]).all() and (k > 0).all()
    x = torch.rand(50, 2, device="cuda") * 5
    a, b = m.phi_raw(x, c[0]), m.phi_raw(x, c[1])
    assert torch.equal(a, b)                                                # FiLM zero-init: c has no effect yet
    with torch.no_grad():
        for f in m.decoder.film:
            f.weight.normal_(0, 0.1)
    a, b = m.phi_raw(x, c[0]), m.phi_raw(x, c[1])
    assert float((a - b).abs().max()) > 1e-4                                # woken: the room code moves the modes


@GPU
@pytest.mark.skipif(not Path("data/track_p4_9_eigen/gx/0001.npz").exists(), reason="needs GX eigen-labels")
def test_training_smoke(tmp_path, monkeypatch):
    import aaf.train.nmf_2d as T
    man = json.load(open("configs/sweeps_2d_mat/p4_7_c500_cx500_manifest.json"))
    small = tmp_path / "man.json"
    small.write_text(json.dumps(dict(man, configs=man["configs"][:2])))
    cfg = T.NMFCfg(manifest=str(small), n_iters=2, rooms_per_step=2, rx_per_room=8, val_every=10 ** 9,
                   ckpt_every=10 ** 9, kill_iter=10 ** 9, log_every=1, phys_weight=0.1)
    T.train(cfg, str(tmp_path / "run"))
    sc = json.loads((tmp_path / "run" / "scalars.json").read_text())
    assert all(np.isfinite(v) for r in sc for k, v in r.items() if k != "phase")


def test_kill_rule_treats_nan_as_failing():
    """The first NMF / NMF-P runs went NaN at step ~300 and ran to 30K: `nan < 0.70` is False, so the D180 kill rule
    never fired. The rule must kill anything that is not a finite r >= the bar."""
    import math
    from aaf.train.nmf_2d import kill_fires
    assert kill_fires(float("nan"), 0.70)
    assert kill_fires(-math.inf, 0.70)
    assert kill_fires(0.69, 0.70)
    assert not kill_fires(0.70, 0.70)
    assert not kill_fires(0.95, 0.70)


def test_soft_cutoff_gradient_finite_just_below_kb():
    """D186 regression: the first NMF / NMF-P runs took torch.sqrt of the soft weights. Just below k_b float32 rounds
    cos(pi t) to -1, the weight is exactly 0 while t < 1, sqrt's backward is g / 0 = inf, and one step later every
    parameter upstream of k (k_head, segment_encoder) is NaN -- at step ~265 of the GPU run, from a FINITE loss. The
    sqrt weights are now computed directly; they must square to the soft weights, and the gradient must stay finite
    and bounded at every distance from the cutoff."""
    from aaf.models.nmf_2d import soft_sqrt_weights, soft_weights, subspace_loss
    g = torch.Generator().manual_seed(0)
    k = torch.linspace(0.5, 3.5, 400)
    ka, kb = 2.6, 3.0
    assert torch.allclose(soft_sqrt_weights(k, ka, kb) ** 2, soft_weights(k, ka, kb), atol=1e-6)
    wq = torch.rand(200, generator=g) + 0.5
    psi = torch.randn(200, 24, generator=g)
    kt = torch.linspace(0.5, 3.0, 24)
    for eps in (1e-2, 1e-4, 1e-5, 1e-6, 1e-7, 0.0):
        kk = torch.linspace(0.5, 2.9, 24)
        kk[-1] = kb * (1 - eps)
        kk.requires_grad_(True)
        phi = torch.randn(200, 24, generator=g, requires_grad=True)
        L = subspace_loss(phi, soft_sqrt_weights(kk, ka, kb), psi, soft_sqrt_weights(kt, ka, kb), wq)
        L.backward()
        assert torch.isfinite(L)
        assert torch.isfinite(kk.grad).all() and torch.isfinite(phi.grad).all(), eps
        assert float(kk.grad.abs().max()) < 1e3, (eps, float(kk.grad.abs().max()))
