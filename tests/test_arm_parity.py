"""P3-1 arm parity — the three arms differ ONLY in the conditioning path (CUDA/tcnn).

Run on a GPU node (scripts/slurm/run_pytest.sh). Asserts:
  1. backbone (6 encoders + 3 tcnn MLPs) param counts are identical across L/G/G+;
  2. G/G+ have no latent table / no geometry head; L has both;
  3. G+ has a single zero-init scalar w; L/G have none;
  4. with identical backbone weights, at init (FiLM is identity) the forward is invariant
     to the conditioning vector AND identical across arms;
  5. G+ with w=0 and a resonance map R set == G+ with no R (the injection is a no-op at init).
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="INR3D_AutoDecoder needs tcnn (CUDA)."
)

_BACKBONE = [
    "_pos_encoding", "_pos_signal_encoding", "_tx_pos_encoding",
    "_tx_pos_signal_encoding", "_dir_encoding", "_tx_dir_encoding",
    "_model_encoder_sigma", "_model_decoder_sigma", "_model_signal",
]


def _make(cond_source, cond_dim, l_head, n_freq_bins=129):
    from aaf.models.inr_3d import INR3D_AutoDecoder
    hg = {"otype": "HashGrid", "n_levels": 4, "n_features_per_level": 2,
          "log2_hashmap_size": 12, "base_resolution": 8, "per_level_scale": 1.5}
    return INR3D_AutoDecoder(
        n_rooms=5, latent_dim=16, n_freq_bins=n_freq_bins, hash_grid_config=hg,
        conditioning_type="film", latent_jitter_sigma=0.0, l_head_enabled=l_head,
        cond_source=cond_source, cond_dim=cond_dim,
    ).cuda()


def _backbone_param_count(m):
    return sum(sum(p.numel() for p in getattr(m, name).parameters()) for name in _BACKBONE)


def _copy_backbone(src, dst):
    for name in _BACKBONE:
        getattr(dst, name).load_state_dict(getattr(src, name).state_dict())


def test_backbone_param_counts_equal():
    L = _make("latent", None, True)
    G = _make("geom_fourier", 48, False)
    Gp = _make("eigen", 64, False)
    nL, nG, nGp = map(_backbone_param_count, (L, G, Gp))
    assert nL == nG == nGp, f"backbone param counts differ: L={nL} G={nG} G+={nGp}"


def test_conditioning_members_per_arm():
    L = _make("latent", None, True)
    G = _make("geom_fourier", 48, False)
    Gp = _make("eigen", 64, False)
    assert L.latents is not None and L.l_head is not None
    assert G.latents is None and G.l_head is None
    assert Gp.latents is None and Gp.l_head is None
    assert L.w is None and G.w is None
    assert Gp.w is not None and Gp.w.numel() == 1 and float(Gp.w) == 0.0
    # FiLM input widths track cond_dim; backbone tcnn input dims do not
    assert L.film_sigma.in_features == 16
    assert G.film_sigma.in_features == 48
    assert Gp.film_sigma.in_features == 64


@torch.no_grad()
def test_forward_identical_across_arms_at_init():
    L = _make("latent", None, True).eval()
    G = _make("geom_fourier", 48, False).eval()
    Gp = _make("eigen", 64, False).eval()
    _copy_backbone(L, G)
    _copy_backbone(L, Gp)
    B, N = 1, 16
    pts = torch.rand(B, N, 3, device="cuda") * 2 - 1
    view = torch.rand(B, N, 3, device="cuda") * 2 - 1
    tx = torch.rand(B, N, 3, device="cuda") * 2 - 1
    aL, sL = L(pts, view, tx, z_s=torch.randn(B, 16, device="cuda"))
    aG, sG = G(pts, view, tx, z_s=torch.randn(B, 48, device="cuda"))
    aGp, sGp = Gp(pts, view, tx, z_s=torch.randn(B, 64, device="cuda"))
    # FiLM is identity at init (zero weight, γ=1/β=0) → output independent of conditioning
    assert torch.allclose(aL, aG, atol=1e-4) and torch.allclose(sL, sG, atol=1e-4)
    assert torch.allclose(aL, aGp, atol=1e-4) and torch.allclose(sL, sGp, atol=1e-4)
    # invariance to the conditioning vector within an arm
    aG2, sG2 = G(pts, view, tx, z_s=torch.randn(B, 48, device="cuda"))
    assert torch.allclose(aG, aG2, atol=1e-5) and torch.allclose(sG, sG2, atol=1e-5)


@torch.no_grad()
def test_gplus_w0_resonance_is_noop():
    Gp = _make("eigen", 64, False).eval()
    B, N = 1, 16
    pts = torch.rand(B, N, 3, device="cuda") * 2 - 1
    view = torch.rand(B, N, 3, device="cuda") * 2 - 1
    tx = torch.rand(B, N, 3, device="cuda") * 2 - 1
    z = torch.randn(B, 64, device="cuda")
    Gp.set_resonance(None)
    a0, s0 = Gp(pts, view, tx, z_s=z)
    R = torch.rand(Gp.n_freq_bins, device="cuda")            # arbitrary R
    Gp.set_resonance(R)
    a1, s1 = Gp(pts, view, tx, z_s=z)
    # w=0 ⇒ (1 + w·R) = 1 ⇒ identical output regardless of R
    assert torch.allclose(s0, s1, atol=1e-6) and torch.allclose(a0, a1, atol=1e-6)
