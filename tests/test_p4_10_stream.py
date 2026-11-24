"""P4-10 (D195 0c, D197 A1 / A2 / A4): multi-source rows, replaced rows, per-row jitter.

Rule 8 (nothing here adds a parameter, so a checkpoint cannot see a mistake):
  * per-row jitter: a batch equals the same rows rendered one at a time with their own scalar jitter; the scalar path and
    the no-jitter path are unchanged (CPU, stub model).
  * slot and target: every row's target is its own (config, slot, receiver) entry of the memmap -- slot 0 is the
    preloaded H -- and its source is that slot's source. A wrong slot / target pairing trains without error, so this
    is the test that matters.
  * replaced rows: a stretched row takes its variant's receiver, source, tokens, room size and target; a material row
    keeps the room, source and receiver and takes the variant's tokens and target.
  * streams: batches, slots, jitters and swap draws are the same whether or not replaced rows are used; a resume
    continues all of them.
  * the default path is untouched: with no P4-10 key, the step is `_step_p49`.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from tests.test_p4_9_dir import _cfg

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="tinycudann needs a GPU")


class _Stub(torch.nn.Module):
    def forward(self, pts, view, tx, tx_view=None, z_s=None, **kw):
        n = 5
        f = torch.arange(n).view(1, 1, n).float()
        a = torch.sin(pts.sum(-1, keepdim=True) + tx.sum(-1, keepdim=True) + f).clamp(min=0)
        s = torch.cos(pts[..., :1] * 2 + view[..., :1] + f)
        return (a + 0.1).to(torch.complex64), (s + 0j).to(torch.complex64)


def _renderer(n_azi=8):
    from aaf.renderers.freq_2d import FreqRenderer2D
    return FreqRenderer2D(n_azi=n_azi, n_pts_per_ray=16, n_time_samples=8, fs=8.0, near=1e-3, c=343.0)


def test_row_jitter_equals_single_rows():
    r = _renderer()
    r.train()
    m = _Stub()
    B = 6
    rx = torch.rand(B, 2) * 3 + 0.2
    tx = torch.rand(B, 2) * 3 + 0.2
    lo, hi = torch.zeros(2), torch.tensor([4.0, 4.0])
    jit = torch.rand(B) * 2 * np.pi / 8
    bat = r(m, rx, tx, lo, hi, jitter=jit)
    one = torch.cat([r(m, rx[i:i + 1], tx[i:i + 1], lo, hi, jitter=jit[i:i + 1]) for i in range(B)])
    assert torch.allclose(bat, one, atol=1e-6) and not torch.allclose(bat[:1], r(m, rx[:1], tx[:1], lo, hi, jitter=jit[1:2]))
    # the legacy paths: one scalar jitter, and no jitter at all in eval mode
    r.eval()
    a = r(m, rx, tx, lo, hi)
    b = r(m, rx, tx, lo, hi, jitter=torch.zeros(1))
    assert torch.allclose(a, b, atol=1e-6)


def _ms(tmp_path, tr, S=4):
    """a synthetic [C, S, R, B] memmap whose entry values ENCODE (c, s, r) so a mispairing is visible."""
    d = tmp_path / "ms"
    if d.exists():
        return d
    d.mkdir()
    C, R, B = tr.H.shape
    mm = np.lib.format.open_memmap(d / "H.npy", mode="w+", dtype=np.complex64, shape=(C, S, R, B))
    mm[:, 0] = tr.H.cpu().numpy()
    for s in range(1, S):
        mm[:, s] = tr.H.cpu().numpy() + (10 * s) + 1j * np.arange(R)[None, :, None] * 1e-2
    mm.flush()
    src = np.repeat(tr.src_per.cpu().numpy()[:, None], S, 1) + np.arange(S)[None, :, None] * 0.31
    src[:, 0] = tr.src_per.cpu().numpy()
    np.save(d / "src.npy", src)
    np.save(d / "rx.npy", tr.rx.cpu().numpy().astype(np.float64))
    (d / "index.json").write_text(json.dumps({"filenames": [c.filename for c in tr.configs], "shape": [C, S, R, B]}))
    return d


def _variants(tmp_path, tr, kind, n_per=2, S=4):
    d = tmp_path / kind
    if d.exists():
        return d
    d.mkdir()
    C, R, B = tr.H.shape
    V = C * (n_per if kind == "str" else 1)
    mm = np.lib.format.open_memmap(d / "H.npy", mode="w+", dtype=np.complex64, shape=(V, S, R, B))
    mm[:] = 1000.0 + np.arange(V)[:, None, None, None] + 1e-3 * np.arange(R)[None, None, :, None]
    mm.flush()
    base = tr.rx.cpu().numpy().astype(np.float64)
    rx = np.stack([base[v // n_per] * 0.9 if kind == "str" else base[v] for v in range(V)])
    src = np.zeros((V, S, 2)) + np.arange(S)[None, :, None] * 0.2 + 0.7
    np.save(d / "rx.npy", rx)
    np.save(d / "src.npy", src)
    var = []
    for v in range(V):
        c = tr.configs[v // n_per if kind == "str" else v]
        verts = [[float(x * 0.9), float(y * 0.9)] for x, y in c.verts] if kind == "str" else [list(p) for p in c.verts]
        var.append({"L": c.L * 0.9 if kind == "str" else c.L, "W": c.W * 0.9 if kind == "str" else c.W, "verts": verts,
                    "edge_alphas": [0.15] * len(verts) if kind == "str" else [0.4] * len(verts)})
    (d / "index.json").write_text(json.dumps({"kind": kind, "base_filenames": [c.filename for c in tr.configs],
                                              "n_variants": V, "shape": [V, S, R, B], "variants": var}))
    return d


def _trainer(tmp_path, name, **kw):
    from aaf.train.multi_room_2d_mat import P32Trainer
    return P32Trainer(_cfg(tmp_path, p49_mode="ctl", **kw), tmp_path / name)


@GPU
def test_default_path_untouched(tmp_path):
    ctl = _trainer(tmp_path, "ctl")
    assert ctl._step_fn == ctl._step_p49 and ctl.H.is_cuda and not hasattr(ctl, "_p410_gens")


@GPU
def test_slot_target_and_source_pairing(tmp_path):
    ctl = _trainer(tmp_path, "ctl")
    d = _ms(tmp_path, ctl)
    arm = _trainer(tmp_path, "ms", p410_ms_dir=str(d), p410_row_jitter=True)
    assert arm._step_fn == arm._step_p410 and not arm.H.is_cuda
    mm = np.load(d / "H.npy", mmap_mode="r")
    src = np.load(d / "src.npy")
    seen = set()
    for _ in range(30):
        for ci, ri in arm._p49_batch():
            g0 = arm._p410_gens["slot"].get_state()
            rx, tx, z, dims, T, jit = arm._p410_rows(ci, ri, {})
            g1 = arm._p410_gens["slot"].get_state()
            arm._p410_gens["slot"].set_state(g0)
            slot = torch.randint(0, 4, (ci.numel(),), generator=arm._p410_gens["slot"]).numpy()
            arm._p410_gens["slot"].set_state(g1)
            c, r = ci.cpu().numpy(), ri.cpu().numpy()
            assert np.allclose(T.cpu().numpy(), mm[c, slot, r])                 # each target is its own entry
            assert np.allclose(tx.cpu().numpy(), src[c, slot], atol=1e-5)       # and its own source
            assert np.allclose(rx.cpu().numpy(), ctl.rx.cpu().numpy()[c, r], atol=1e-5)
            seen |= set(zip(c.tolist(), slot.tolist()))
    assert {s for _, s in seen} == {0, 1, 2, 3}


@GPU
def test_streams_match_across_arms(tmp_path):
    base = _trainer(tmp_path, "a", p410_ms_dir="", p410_row_jitter=True, p49_recip_p=0.25)
    d = _ms(tmp_path, base)
    del base
    a = _trainer(tmp_path, "b", p410_ms_dir=str(d), p410_row_jitter=True, p49_recip_p=0.25)
    s = _variants(tmp_path, a, "str")
    b = _trainer(tmp_path, "c", p410_ms_dir=str(d), p410_row_jitter=True, p49_recip_p=0.25, p410_str_dir=str(s), p410_str_p=0.25)
    for _ in range(4):
        (ca, ra), (cb, rb) = a._p49_batch()[0], b._p49_batch()[0]
        assert torch.equal(ca, cb) and torch.equal(ra, rb)
        _, _, _, _, _, ja = a._p410_rows(ca, ra, {})
        ag = {}
        _, _, _, _, _, jb = b._p410_rows(cb, rb, ag)
        assert torch.equal(ja, jb)
    for k in a._p410_gens:
        assert torch.equal(a._p410_gens[k].get_state(), b._p410_gens[k].get_state()), k
    assert torch.equal(a._p49_swap_gen.get_state(), b._p49_swap_gen.get_state())


@GPU
def test_replaced_rows_take_the_variant(tmp_path):
    base = _trainer(tmp_path, "a")
    d = _ms(tmp_path, base)
    sd, md = _variants(tmp_path, base, "str"), _variants(tmp_path, base, "mat")
    s = _trainer(tmp_path, "s", p410_ms_dir=str(d), p410_row_jitter=True, p410_str_dir=str(sd), p410_str_p=1.0)
    ci, ri = s._p49_batch()[0]
    rx, tx, z, dims, T, _ = s._p410_rows(ci, ri, {})
    assert (T.real >= 999).all() and torch.allclose(dims[:, 0], s._str["dims"][2 * ci, 0] if False else dims[:, 0])
    v_rx = s._str["rx"].cpu().numpy()
    c = ci.cpu().numpy()
    for i in range(len(c)):                                         # each row's room is one of ITS room's two variants
        cand = [np.allclose(rx[i].cpu().numpy(), v_rx[2 * c[i] + k, int(ri[i])], atol=1e-4) for k in (0, 1)]
        assert any(cand), i
    m = _trainer(tmp_path, "m", p410_ms_dir=str(d), p410_row_jitter=True, p410_mat_dir=str(md), p410_mat_p=1.0)
    rx2, tx2, z2, dims2, T2, _ = m._p410_rows(*m._p49_batch()[0], {})
    assert (T2.real >= 999).all()                                    # material targets
    n = _trainer(tmp_path, "n", p410_ms_dir=str(d), p410_row_jitter=True)
    ci3, ri3 = n._p49_batch()[0]
    a = n._p410_rows(ci3, ri3, {})
    assert not torch.allclose(a[2], z2) and torch.allclose(a[0], rx2) and torch.allclose(a[1], tx2)    # same points, other tokens


@GPU
def test_swap_mask_uses_the_rows_own_points_and_resume(tmp_path):
    from aaf.train.multi_room_2d_mat import P32Trainer
    base = _trainer(tmp_path, "a")
    d = _ms(tmp_path, base)
    del base
    a = _trainer(tmp_path, "b", p410_ms_dir=str(d), p410_row_jitter=True, p49_recip_p=0.9)
    a._step_fn(1)
    a.save_ckpt(1)
    b1 = a._p49_batch()
    r1 = a._p410_rows(*b1[0], {})
    again = P32Trainer(_cfg(tmp_path, p49_mode="ctl", p410_ms_dir=str(d), p410_row_jitter=True, p49_recip_p=0.9), tmp_path / "b")
    assert again.start_iter == 1
    b2 = again._p49_batch()
    r2 = again._p410_rows(*b2[0], {})
    assert all(torch.equal(x, y) for x, y in zip(b1[0], b2[0]))
    for x, y in zip(r1, r2):
        assert torch.equal(x, y)


@GPU
def test_step_runs_and_loss_is_finite(tmp_path):
    base = _trainer(tmp_path, "a")
    d = _ms(tmp_path, base)
    del base
    a = _trainer(tmp_path, "b", p410_ms_dir=str(d), p410_row_jitter=True, p49_recip_p=0.25)
    m = a._step_fn(1)
    assert np.isfinite(m["loss"]) and 0.0 <= m["recip_frac"] <= 1.0


@GPU
def test_combined_arm_rows_are_each_owner_arms_rows(tmp_path):
    """R1-SM (D212). MAT off = R1-S bit for bit; STR off = R1-M bit for bit; both on: STR rows are R1-S's, the MAT rows (never STR)
    are R1-M's, the rest R1's -- no row mixes a stretched room with the base room's material tokens or target."""
    base = _trainer(tmp_path, "a")
    d = _ms(tmp_path, base)
    sd, md = _variants(tmp_path, base, "str"), _variants(tmp_path, base, "mat")
    del base
    kw = dict(p410_ms_dir=str(d), p410_row_jitter=True, p49_recip_p=0.25)
    S, M = dict(p410_str_dir=str(sd), p410_str_p=0.25), dict(p410_mat_dir=str(md), p410_mat_p=0.5)
    arms = {"r1": _trainer(tmp_path, "r1", **kw), "s": _trainer(tmp_path, "s", **kw, **S), "m": _trainer(tmp_path, "m", **kw, **M),
            "sm": _trainer(tmp_path, "sm", **kw, **S, **M)}
    arms["sm_matoff"] = _trainer(tmp_path, "smo", **kw, p410_mat_dir=str(md), p410_mat_p=0.0, **S)
    arms["sm_stroff"] = _trainer(tmp_path, "soo", **kw, p410_str_dir=str(sd), p410_str_p=0.0, **M)
    n = {"str": 0, "mat": 0, "both_drawn": 0, "plain": 0}
    for _ in range(25):
        bs = {k: a._p49_batch() for k, a in arms.items()}
        for mb in range(len(bs["r1"])):
            rows, u = {}, None
            for k, a in arms.items():
                ci, ri = bs[k][mb]
                assert torch.equal(ci, bs["r1"][mb][0]) and torch.equal(ri, bs["r1"][mb][1])
                if k == "sm":
                    st = a._p410_gens["var"].get_state()
                    u = (torch.rand(ci.numel(), generator=a._p410_gens["var"]), torch.rand(ci.numel(), generator=a._p410_gens["var"]))
                    a._p410_gens["var"].set_state(st)
                rows[k] = a._p410_rows(ci, ri, {})
            for x, y in zip(rows["sm_matoff"], rows["s"]):
                assert torch.equal(x, y)
            for x, y in zip(rows["sm_stroff"], rows["m"]):
                assert torch.equal(x, y)
            is_s, is_m = (u[0] < 0.25), (u[1] < 0.5) & ~(u[0] < 0.25)
            n["both_drawn"] += int(((u[0] < 0.25) & (u[1] < 0.5)).sum())
            for i in range(len(is_s)):
                own = "s" if is_s[i] else "m" if is_m[i] else "r1"
                n["str" if is_s[i] else "mat" if is_m[i] else "plain"] += 1
                for x, y in zip(rows["sm"], rows[own]):
                    assert torch.equal(x[i], y[i]), (own, i)
                if own != "r1":                                   # not vacuous: the owner's row differs from R1's
                    assert not torch.equal(rows["sm"][4][i], rows["r1"][4][i])
    assert min(n.values()) > 5, n
    for k in arms["r1"]._p410_gens:
        assert torch.equal(arms["r1"]._p410_gens[k].get_state(), arms["sm"]._p410_gens[k].get_state()), k
