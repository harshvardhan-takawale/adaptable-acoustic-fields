"""P4-7 m4 (D169): the trainer's per-room source, tested on BEHAVIOUR (Rule 8).

`per_room_source` adds no parameters -- a checkpoint trained with it loads into a trainer without it and
renders with the wrong source, silently -- so every claim is an OUTPUT claim:
  * off: `_render` equals a FROZEN VERBATIM COPY of the pre-change `_render` (torch.equal), and a corpus
    whose files disagree on the source is REFUSED (it used to train silently against the last file's);
  * on, single-source corpus: the per-room table holds one value and every render is BIT-IDENTICAL to
    the off trainer's (same weights, same rows);
  * on, mixed corpus: every row is rendered with its own file's `source_pos` (captured at the renderer),
    and the output differs from a global-source render (teeth: the tx hash encoding is live at init);
  * augmentation and the sigma term are refused with it on; the resume guard covers the key.
All tests import the trainer (tinycudann), so they need a GPU node.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="tinycudann needs a GPU")
N_BAND = 601
MIX = "configs/sweeps_2d_mat/p4_7_c500_cx500_manifest.json"
MIX_DIR = "data/track_p4_7_mix"


def _frozen_render(self, cfg_ids, rx_ids):
    """VERBATIM copy of P32Trainer._render before P4-7 m4 (main @ 31a5f87)."""
    out = torch.zeros(cfg_ids.numel(), self.n_band,
                      dtype=torch.complex64, device=self.device)
    g = self.geom_id[cfg_ids]
    for gid in torch.unique(g):
        sel = torch.nonzero(g == gid).squeeze(1)
        L, W = self.geom_dims[int(gid)]
        room_min = torch.zeros(2, device=self.device)
        room_max = torch.tensor([L, W], device=self.device, dtype=torch.float32)
        step = max(1, int(self.cfg.max_rows_per_render))
        for s0 in range(0, sel.numel(), step):
            sub = sel[s0:s0 + step]
            rx = self.rx[cfg_ids[sub], rx_ids[sub]]
            tx = self.src.unsqueeze(0).expand(sub.numel(), -1)
            z = self.cond[cfg_ids[sub]]
            H = self.renderer(self.model, rx, tx, room_min, room_max, z_s=z)
            out[sub] = H[:, self.band[0]:self.band[1]]
    return out


class _TxRenderer(torch.nn.Module):
    """Deterministic stand-in whose output depends on rx, tx, the box and z; records tx per call."""
    def __init__(self):
        super().__init__()
        self.tx = []

    def forward(self, model, rx, tx, room_min, room_max, z_s=None):
        self.tx.append(tx.clone())
        d = (rx - tx).norm(dim=-1, keepdim=True)
        f = torch.arange(N_BAND, dtype=torch.float32, device=rx.device)[None]
        return (1.0 + d * 0.01 * f + 0.001 * z_s.sum(-1, keepdim=True) + room_max.prod()).to(torch.complex64)


def _stub(src_per=None):
    from aaf.train.multi_room_2d_mat import P32TrainCfg, P32Trainer
    t = P32Trainer.__new__(P32Trainer)
    t.cfg = P32TrainCfg(max_rows_per_render=3)
    t.device = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    t.rx = (torch.rand(4, 20, 2, generator=g) * 4).to(t.device)
    t.cond = torch.rand(4, 324, generator=g).to(t.device)
    t.geom_id = torch.tensor([0, 1, 0, 1], device=t.device)
    t.geom_dims = [(5.0, 4.5), (6.0, 5.0)]
    t.src = torch.tensor([0.5, 1.6], device=t.device)
    t.band, t.n_band = (0, N_BAND), N_BAND
    t.model, t.renderer = None, _TxRenderer()
    if src_per is not None:
        t.src_per = torch.tensor(src_per, dtype=torch.float32, device=t.device)
    return t


ROWS = (torch.tensor([0, 1, 1, 2, 3, 3, 0, 2]), torch.tensor([1, 4, 7, 2, 9, 3, 0, 19]))


def _rows(t):
    return ROWS[0].to(t.device), ROWS[1].to(t.device)


@GPU
def test_default_is_off():
    from aaf.train.multi_room_2d_mat import P32TrainCfg
    assert P32TrainCfg().per_room_source is False


@GPU
def test_off_render_equals_the_frozen_pre_change_render():
    a = _stub()
    assert getattr(a, "src_per", None) is None
    ref = _frozen_render(_stub(), *_rows(a))
    assert torch.equal(a._render(*_rows(a)), ref)


@GPU
def test_single_source_table_is_bit_identical_to_the_global_source():
    off, on = _stub(), _stub(src_per=[[0.5, 1.6]] * 4)
    assert torch.equal(on._render(*_rows(on)), off._render(*_rows(off)))


@GPU
def test_each_row_is_rendered_with_its_own_source_and_the_output_changes():
    src = [[0.5, 1.6], [2.0, 3.0], [1.1, 0.7], [4.0, 2.2]]
    on = _stub(src_per=src)
    got = on._render(*_rows(on))
    ci = ROWS[0]
    # rows are grouped by geometry then chunked; rebuild the per-row tx in that order
    seen = torch.cat(on.renderer.tx).cpu()
    order = torch.cat([torch.nonzero(on.geom_id.cpu()[ci] == g).squeeze(1) for g in torch.unique(on.geom_id.cpu()[ci])])
    assert torch.equal(seen, torch.tensor(src, dtype=torch.float32)[ci[order]])
    off = _stub()
    assert not torch.equal(got, off._render(*_rows(off)))                      # teeth


def _tiny_manifest(tmp_path, n_c500=3, n_cx=3):
    m = json.load(open(MIX))
    rows = [r for r in m["configs"] if r["origin"] == "C500"][:n_c500]
    cx = [r for r in m["configs"] if r["origin"] == "CX500" and (Path(MIX_DIR) / r["filename"]).exists()][:n_cx]
    if len(cx) < n_cx:
        pytest.skip("CX500 rooms not built yet")
    out = tmp_path / "man.json"
    out.write_text(json.dumps(dict(m, configs=rows + cx)))
    return str(out), rows + cx


def _trainer(tmp_path, man, name, **kw):
    """G1's recipe (the m4 arm's) on a tiny manifest."""
    import yaml

    from aaf.train.multi_room_2d_mat import P32TrainCfg, P32Trainer
    base = yaml.safe_load(open("configs/sweep_2d_mat/P4_6_G1.yaml"))
    base = {k: v for k, v in base.items() if k in P32TrainCfg.__dataclass_fields__}
    base.update({"data_dir": MIX_DIR, "configs_manifest": man, "n_iters": 2, "val_max_configs": 2})
    base.update(kw)
    return P32Trainer(P32TrainCfg(**base), tmp_path / name)


@GPU
def test_integration_mixed_corpus_refused_off_and_read_per_file_on(tmp_path):
    man, rows = _tiny_manifest(tmp_path)
    with pytest.raises(ValueError, match="per_room_source is off"):
        _trainer(tmp_path, man, "off")
    on = _trainer(tmp_path, man, "on", per_room_source=True)
    want = []
    for c in on.configs:
        with h5py.File(Path(MIX_DIR) / c.filename) as f:
            want.append(json.loads(f.attrs["source_pos"]))
    assert torch.equal(on.src_per.cpu(), torch.tensor(want, dtype=torch.float32))
    assert len({tuple(x) for x in want}) == 4                     # three CX sources + the C500 one
    # teeth on the REAL model: a CX row rendered with its own source differs from the global one
    on.model.eval(); on.renderer.eval()
    ci = torch.tensor([3, 4, 5], device=on.device); ri = torch.tensor([0, 1, 2], device=on.device)
    with torch.no_grad():
        mine = on._render(ci, ri)
        keep = on.src_per
        on.src_per = None
        glob = on._render(ci, ri)
        on.src_per = keep
    assert not torch.equal(mine, glob)


@GPU
def test_integration_single_source_corpus_renders_bit_identically(tmp_path):
    man, _ = _tiny_manifest(tmp_path, n_c500=4, n_cx=0)
    off = _trainer(tmp_path, man, "off")
    on = _trainer(tmp_path, man, "on", per_room_source=True)
    assert off.src_per is None and on.src_per.shape == (4, 2)
    assert torch.equal(on.src_per, off.src.unsqueeze(0).expand(4, 2))          # float32 (0.5, 1.6)
    for t in (off, on):
        t.model.eval(); t.renderer.eval()
    assert all(torch.equal(p, q) for p, q in zip(off.model.state_dict().values(), on.model.state_dict().values()))
    ci = torch.tensor([0, 1, 2, 3, 3], device=off.device); ri = torch.tensor([5, 6, 7, 8, 100], device=off.device)
    with torch.no_grad():
        assert torch.equal(off._render(ci, ri), on._render(ci, ri))


@GPU
def test_refusals_and_resume_guard(tmp_path):
    import inspect

    from aaf.train import multi_room_2d_mat as M
    man, _ = _tiny_manifest(tmp_path, n_c500=2, n_cx=1)
    with pytest.raises(ValueError, match="sigma_aux_weight"):
        _trainer(tmp_path, man, "s", per_room_source=True, sigma_aux_weight=0.1)
    with pytest.raises(ValueError, match="per_room_source is not implemented"):
        _trainer(tmp_path, man, "a", per_room_source=True, aug_d4_p=0.5)
    src = inspect.getsource(M)
    guard = src[src.index("def _maybe_resume"):src.index("def _early_stop")]
    assert '"per_room_source"' in guard


def test_c500_rows_condition_identically_as_polygon_rows():
    """The m4 manifest re-expresses C500's training rows as `p4_1.poly` rows; the geom_token conditioning,
    bbox, strata and filename must be EXACTLY what G1 trained on (else GX's C500 half is a different corpus)."""
    from aaf.data.poly_configs import configs_from_rows as poly_rows
    from aaf.data.shape_configs import configs_from_rows as shape_rows
    from aaf.models.conditioning_2d import build_cond_vector_2d
    mix = [r for r in json.load(open(MIX))["configs"] if r["origin"] == "C500"]
    c5 = shape_rows(json.load(open("configs/sweeps_2d_mat/p4_4_shapes_500.json"))["configs"], split="train")
    assert len(mix) == len(c5) == 500
    for a, b in zip(poly_rows(mix), c5):
        assert (a.filename, a.strata, a.L, a.W, list(a.alphas)) == (b.filename, b.strata, b.L, b.W, list(b.alphas))
        ca = build_cond_vector_2d("geom_token", a.L, a.W, a.edge_alphas, x0=getattr(a, "x0", None),
                                  a=getattr(a, "a", None), verts=a.verts)
        cb = build_cond_vector_2d("geom_token", b.L, b.W, b.edge_alphas, x0=getattr(b, "x0", None),
                                  a=getattr(b, "a", None), verts=b.verts)
        assert torch.equal(ca, cb), a.filename
