"""P4-5 PB3 / W3 (D144): per-band signal heads and the parameter-matched wide control.

Rule 8: the alternative pathway changes behaviour, so the tests assert OUTPUTS, not loading.
  * the default build is untouched (no alternative modules; the B5 checkpoint loads strictly and
    renders exactly as its cached dense render -- covered by test_p4_5_hf_arms' pre-P4-5 test too);
  * crossfade weights sum to 1 in every bin and are zero outside each head's support;
  * each head affects ONLY its own bins (perturbing head 3 leaves bins below its support unchanged);
  * each head's own FiLM is live once woken (the vacuous-test trap: FiLM is zero-initialised);
  * bins above the supervised band are exactly zero;
  * a checkpoint round-trips through `load_model` with an identical pathway and identical output.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("tinycudann")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from aaf.models.conditioning_2d import build_cond_vector_2d  # noqa: E402
from aaf.models.inr_2d import INR2D_AutoDecoder  # noqa: E402
from aaf.train.multi_room_2d_mat import signal_kwargs  # noqa: E402

DEV = torch.device("cuda")
HG = dict(otype="HashGrid", n_levels=20, n_features_per_level=2, log2_hashmap_size=18,
          base_resolution=80, per_level_scale=1.5)
VERTS = [[0.0, 0.0], [6.0, 0.0], [6.0, 5.0], [2.0, 5.0], [2.0, 4.0], [0.0, 4.0]]
PB3_CFG = {"signal_heads": 3, "signal_band_edges_hz": "173.0,245.0", "signal_xfade_hz": 10.0,
           "fs": 4096, "n_time_samples": 8192}


def build(**kw):
    torch.manual_seed(0)
    return INR2D_AutoDecoder(n_rooms=1, latent_dim=16, n_freq_bins=4097, hash_grid_config=HG,
                             conditioning_type="film", cond_source="geom_token", cond_dim=324,
                             world_scale=10.0, token_pool="attn_residual_extent", dc_head_bins=41,
                             **kw).to(DEV)


def run(m, n=16):
    torch.manual_seed(1)
    pts = torch.rand(1, n, 2, device=DEV) * 4 + 0.5
    view = torch.nn.functional.normalize(torch.randn(1, n, 2, device=DEV), dim=-1)
    tx = torch.tensor([[[0.5, 1.6]]], device=DEV).expand(1, n, 2)
    cond = build_cond_vector_2d("geom_token", 6.0, 5.0, [0.15] * 6, verts=VERTS, device=DEV).unsqueeze(0)
    with torch.no_grad():
        return m(pts, view, tx, z_s=cond)


def test_default_build_is_untouched():
    m = build()
    assert m._model_signal is not None and not hasattr(m, "_signal_nets")
    assert "signal_xfade_w" not in dict(m.named_buffers())


def test_pb3_weights_partition_unity_and_supports():
    m = build(**signal_kwargs(PB3_CFG, 601))
    w = m.signal_xfade_w.cpu()
    assert w.shape == (3, 601)
    assert torch.allclose(w.sum(0), torch.ones(601), atol=1e-6)
    lohi = m.signal_head_lohi.cpu().tolist()
    assert lohi == [[0, 366], [326, 510], [470, 600]], lohi
    for k, (lo, hi) in enumerate(lohi):
        assert float(w[k, :lo].abs().sum()) == 0.0 and float(w[k, hi + 1:].abs().sum()) == 0.0


def test_pb3_head_locality_and_zero_above_band():
    m = build(**signal_kwargs(PB3_CFG, 601))
    _a, s0 = run(m)
    assert float(s0[..., 601:].abs().max()) == 0.0
    with torch.no_grad():
        m._signal_nets[2].params.add_(0.05 * torch.randn_like(m._signal_nets[2].params))
    _a, s1 = run(m)
    d = (s1 - s0).abs().amax(dim=(0, 1))
    assert float(d[:470].max()) == 0.0, "head 3 changed bins below its support"
    assert float(d[470:601].max()) > 0.0, "head 3 perturbation had no effect in its band"


def test_pb3_per_head_film_is_live():
    m = build(**signal_kwargs(PB3_CFG, 601))
    _a, s0 = run(m)
    with torch.no_grad():
        m._signal_films[0].weight.normal_(0, 0.05)          # head 2's own FiLM, woken
    _a, s1 = run(m)
    d = (s1 - s0).abs().amax(dim=(0, 1))
    assert float(d[326:511].max()) > 0.0, "head 2's FiLM does nothing once woken"
    assert float(d[:326].max()) == 0.0 and float(d[511:].max()) == 0.0


def test_w3_band_only_zero_above(tmp_path):
    m = build(**signal_kwargs({"signal_band_only": True, "signal_n_neurons": 896,
                               "fs": 4096, "n_time_samples": 8192}, 601))
    _a, s = run(m)
    assert float(s[..., 601:].abs().max()) == 0.0 and float(s[..., :601].abs().max()) > 0.0


def test_pb3_roundtrip_through_load_model(tmp_path):
    from pathlib import Path

    from aaf.eval.p3_2_eval import load_model
    st = torch.load("outputs/p4_5/arms/B5/ckpt_iter0060000.pt", map_location="cpu")
    cfg = dict(st["cfg"]); cfg.update(PB3_CFG)
    m = build(**signal_kwargs(cfg, 601))
    with torch.no_grad():
        for n in m._signal_nets:
            n.params.normal_(0, 0.02)
        m._signal_films[1].weight.normal_(0, 0.05)
    p = Path(tmp_path) / "ckpt_iter0000001.pt"
    torch.save({"cfg": cfg, "model": m.state_dict(), "iter": 1}, p)
    m2, _r, _c, _meta, _it = load_model(p, DEV)
    assert m2._model_signal is None and len(m2._signal_nets) == 3
    _a, s0 = run(m.eval()); _a, s1 = run(m2)
    assert torch.equal(s0, s1)
