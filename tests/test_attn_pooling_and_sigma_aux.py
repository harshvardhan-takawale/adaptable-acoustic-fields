"""P4-3: position-queried attention pooling (Arm X) and the sigma auxiliary loss (Arm S).

Both changes are cheap to write and expensive to get wrong -- each one fronts a 12-hour
training run, and each fails in a way that produces a plausible-looking mediocre arm rather
than an error. So the properties that make the arms interpretable are pinned here BEFORE any
GPU time is spent:

* **attn_residual must start exactly at masked_mean.** That is the entire reason the residual
  form was chosen over the literal replacement -- it makes iteration 0 the baseline, so any
  later divergence is attributable to position-dependence rather than to a fresh pool relearning
  what mean-pooling already did. If the zero-init were wrong, the arm would still train and the
  comparison would silently stop meaning what it claims.
* **The pooled conditioning must actually vary with position.** Mean and extent-sum hand every
  point the same vector; if attention did too (a broadcasting slip would do it), the arm would
  be a no-op dressed as a hypothesis test.
* **Padded token slots must not influence anything.** A zero token is a legitimate point in
  token space -- an edge at the origin with zero extent -- so leaking attention mass onto one
  conditions the field on a wall that does not exist.
* **The solid/air split must match the polygon**, or Arm S supervises sigma in the wrong place
  and its sigma ratio becomes meaningless.

The model tests import tinycudann and need a GPU; the geometry and layout tests do not.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from aaf.models.conditioning_2d import (
    D_TOK, MAX_SEG_POLY, N_K_TOK_POS, _fourier_block, build_cond_vector_2d,
)
from aaf.sim.polygon_geom import polygon_contains


def _sample_solid_air(*a, **k):
    """Lazy import: `multi_room_2d_mat` imports `inr_2d`, which imports tinycudann and needs a
    GPU. The geometry properties below are CPU-checkable and should stay runnable without one,
    matching how `test_autodecoder_2d.py` defers its model import."""
    from aaf.train.multi_room_2d_mat import sample_solid_air
    return sample_solid_air(*a, **k)

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="tinycudann needs a GPU")

L_VERTS = [(0.0, 0.0), (6.0, 0.0), (6.0, 5.0), (3.0, 5.0), (3.0, 2.5), (0.0, 2.5)]
ALPHAS4 = [0.15] * 4
N_FREQ = 17


# --------------------------------------------------------------------- the batched Fourier map
def test_batched_fourier_matches_the_scalar_featurizer_exactly():
    """The query is compared against keys built from token centres that went through the SCALAR
    `_fourier_block`. A different band order here would still train -- against a silently
    permuted feature space -- and produce a mediocre arm with no error message."""
    from aaf.models.inr_2d import _fourier_pos_batched
    pts = torch.tensor([[0.13, 0.47], [0.0, 0.0], [0.6, 0.5], [-0.2, 0.9]])
    got = _fourier_pos_batched(pts, N_K_TOK_POS)
    assert got.shape == (4, 2 * 2 * N_K_TOK_POS)
    for i in range(pts.shape[0]):
        want = _fourier_block(pts[i], N_K_TOK_POS)
        assert torch.allclose(got[i], want, atol=1e-6), "row {} differs".format(i)


def test_pool_whitelist_registers_both_attention_arms():
    """Read as text so it runs on CPU (importing inr_2d pulls in tinycudann)."""
    src = Path(__file__).resolve().parents[1].joinpath("aaf/models/inr_2d.py").read_text()
    block = src.split("POLY_TOKEN_POOLS = (", 1)[1].split(")", 1)[0]
    for name in ('"masked_mean"', '"extent_sum"', '"attn"', '"attn_residual"'):
        assert name in block, "{} missing from POLY_TOKEN_POOLS".format(name)


# ------------------------------------------------------------------- solid / air stratification
@pytest.mark.parametrize("d,w,L,W", [
    (0.30, 0.60, 6.00, 5.00),      # shallow+narrow: the regime the U-curve says is weakest
    (2.24, 2.00, 6.00, 5.00),      # deep+wide
    (1.08, 2.44, 6.06, 4.52),      # a real P4-2 test shape
])
def test_sample_solid_air_agrees_with_the_polygon(d, w, L, W):
    """The closed form `x < w and y > W - d` must agree with the general point-in-polygon, or
    Arm S supervises sigma in the wrong place and its ratio means nothing."""
    torch.manual_seed(0)
    half = 2000
    P = _sample_solid_air(d, w, L, W, half)
    assert P.shape == (2 * half, 2)
    verts = [(0.0, 0.0), (L, 0.0), (L, W), (w, W), (w, W - d), (0.0, W - d)]
    inside = polygon_contains(verts, P.numpy())
    # first half is SOLID -> outside the room polygon; second half is AIR -> inside it
    assert not inside[:half].any(), "{} solid points landed inside the room".format(
        int(inside[:half].sum()))
    assert inside[half:].all(), "{} air points landed in the notch".format(
        int((~inside[half:]).sum()))
    # and both halves stay inside the bounding box
    assert float(P[:, 0].min()) >= -1e-6 and float(P[:, 0].max()) <= L + 1e-6
    assert float(P[:, 1].min()) >= -1e-6 and float(P[:, 1].max()) <= W + 1e-6


def test_sample_solid_air_is_balanced_even_for_a_sliver_notch():
    """Uniform sampling over the bbox would give ~1% solid here. The whole point of stratifying
    is that the supervision does not thin out exactly where the model is worst."""
    P = _sample_solid_air(0.10, 0.20, 7.0, 5.5, 500)
    assert P.shape == (1000, 2)
    solid_frac = 500 / 1000
    assert solid_frac == 0.5


def test_sample_solid_air_refuses_a_rectangle():
    """A pure rectangle has no solid region; supervising one would be meaningless. The caller
    filters these out, and this makes a slip loud rather than silent."""
    with pytest.raises(ValueError, match="real notch"):
        _sample_solid_air(0.0, 0.0, 6.0, 5.0, 8)


def test_sigma_aux_hinge_is_zero_once_the_ratio_clears_the_target():
    """The term must switch OFF when the model already puts sigma in the wall -- otherwise it
    keeps dragging a model that has discovered occlusion on its own, and Arm S stops being a
    test of 'does injection help' and becomes an unconditional distortion."""
    target = 3.0
    for ratio, want_zero in ((10.0, True), (3.0, True), (1.0, False), (0.5, False)):
        s_solid = torch.full((64,), ratio)
        s_air = torch.ones(64)
        gap = torch.log(s_solid).mean() - torch.log(s_air).mean()
        loss = torch.relu(math.log(target) - gap)
        assert (float(loss) == pytest.approx(0.0, abs=1e-6)) is want_zero, \
            "ratio {} gave loss {}".format(ratio, float(loss))


# --------------------------------------------------------------------------- the model itself
def _model(pool, seed=0, wake_film=True):
    """A model whose FiLM layers actually PROPAGATE the conditioning.

    Both FiLM generators are zero-initialised by design (identity at init: gamma = 1, beta = 0),
    which means that at step 0 the network output does not depend on `z_s` AT ALL -- every
    attenuation bin comes back as softplus(0) = 0.6931 whatever the pooling did. Three of the
    tests below would then pass vacuously: a completely broken attention block, or one wired
    straight to the mean branch, would be indistinguishable from a correct one.

    So the FiLM weights are woken with small noise before any comparison. This is the difference
    between testing the pooling and testing nothing.
    """
    from aaf.models.inr_2d import INR2D_AutoDecoder
    torch.manual_seed(seed)
    m = INR2D_AutoDecoder(
        n_rooms=4, latent_dim=16, n_freq_bins=N_FREQ, cond_source="geom_token", cond_dim=324,
        conditioning_type="film", token_pool=pool, world_scale=10.0,
    ).cuda().eval()
    if wake_film:
        g = torch.Generator(device="cuda").manual_seed(1234)
        for lin in (m.film_sigma, m.film_signal):
            with torch.no_grad():
                lin.weight.copy_(torch.randn(lin.weight.shape, generator=g,
                                             device="cuda") * 0.05)
    return m


def test_film_is_zero_init_so_the_wake_up_above_is_necessary():
    """Pins the premise of `_model`. If FiLM ever stopped being identity-at-init, waking it
    would be unnecessary -- and if it silently STARTED being non-zero, these tests would be
    measuring something other than what they claim."""
    m = _model("masked_mean", seed=0, wake_film=False)
    assert float(m.film_sigma.weight.abs().max()) == 0.0
    assert float(m.film_signal.weight.abs().max()) == 0.0


def _run(m, cond, pts):
    """forward()'s 4th POSITIONAL slot is tx_view, not z_s -- the conditioning must go by
    keyword or it silently becomes a source-direction vector."""
    n = pts.shape[0]
    P = pts.unsqueeze(0).cuda()
    V = torch.zeros(1, n, 2, device="cuda"); V[..., 0] = 1.0
    TX = torch.full((1, n, 2), 0.5, device="cuda")
    with torch.no_grad():
        return m(P, V, TX, tx_view=None, z_s=cond.unsqueeze(0).cuda())


@GPU
def test_attn_residual_is_bit_identical_to_masked_mean_at_init():
    """THE property the residual arm is for. Zero-init o_proj means the delta is exactly zero,
    so the whole network -- not just the pool -- must reproduce the baseline."""
    a = _model("masked_mean", seed=7)
    b = _model("attn_residual", seed=7)
    # copy every shared weight so the ONLY difference is the attention block
    sd = b.state_dict()
    for k, v in a.state_dict().items():
        assert k in sd, "unexpected key {} missing from the attention model".format(k)
        sd[k] = v.clone()
    b.load_state_dict(sd)
    assert float(b.o_proj.weight.abs().max()) == 0.0
    assert float(b.o_proj.bias.abs().max()) == 0.0

    cond = build_cond_vector_2d("geom_token", 6.0, 5.0, [0.15] * 6, verts=L_VERTS)
    pts = torch.tensor([[1.0, 1.0], [5.0, 4.0], [2.9, 2.4], [0.5, 0.5]])
    at, st = _run(a, cond, pts)
    bt, sb = _run(b, cond, pts)
    assert torch.equal(at, bt), "attn_residual is not identical to masked_mean at init"
    assert torch.equal(st, sb)


@GPU
def test_attention_conditioning_actually_varies_with_position():
    """Mean and extent-sum give every point the same shape vector. If attention did too, the
    arm would be a no-op dressed as a hypothesis test. Captured at `film_sigma`, whose input IS
    the pooled conditioning."""
    m = _model("attn", seed=3)
    seen = {}
    h = m.film_sigma.register_forward_hook(lambda _m, i, _o: seen.__setitem__("z", i[0].detach()))
    cond = build_cond_vector_2d("geom_token", 6.0, 5.0, [0.15] * 6, verts=L_VERTS)
    pts = torch.tensor([[0.2, 0.2], [5.8, 4.8], [3.0, 2.5], [1.0, 4.0]])
    _run(m, cond, pts)
    h.remove()
    z = seen["z"]                                        # [N, cond_dim]
    spread = (z - z.mean(dim=0, keepdim=True)).abs().max()
    assert float(spread) > 1e-5, "pooled conditioning is identical at every point"

    # ... and the mean-pool arm must NOT vary, which is what makes the contrast meaningful
    m2 = _model("masked_mean", seed=3)
    seen2 = {}
    h2 = m2.film_sigma.register_forward_hook(
        lambda _m, i, _o: seen2.__setitem__("z", i[0].detach()))
    _run(m2, cond, pts)
    h2.remove()
    z2 = seen2["z"]
    assert float((z2 - z2.mean(dim=0, keepdim=True)).abs().max()) < 1e-6


@GPU
@pytest.mark.parametrize("pool", ["attn", "attn_residual"])
def test_padded_token_slots_cannot_influence_the_output(pool):
    """A rectangle fills 4 of 12 slots. Corrupting the 8 padded ones must change NOTHING --
    tested through the public interface rather than by inspecting attention weights, because
    what matters is the effect, not the intermediate."""
    m = _model(pool, seed=11)
    if pool == "attn_residual":                       # un-zero o_proj or the test is vacuous
        torch.nn.init.normal_(m.o_proj.weight, std=0.1)
    cond = build_cond_vector_2d("geom_token", 6.0, 5.0, ALPHAS4)    # 4 real edges, 8 padded
    n_tok = MAX_SEG_POLY * D_TOK
    mask = cond[n_tok:n_tok + MAX_SEG_POLY]
    assert float(mask[:4].min()) == 1.0 and float(mask[4:].max()) == 0.0, "expected 4 real edges"

    bad = cond.clone()
    pad = bad[:n_tok].reshape(MAX_SEG_POLY, D_TOK)
    pad[4:] = torch.randn_like(pad[4:]) * 5.0                        # garbage in the dead slots
    bad[:n_tok] = pad.reshape(-1)

    pts = torch.tensor([[1.0, 1.0], [4.0, 3.0], [0.3, 4.7]])
    a0, s0 = _run(m, cond, pts)
    a1, s1 = _run(m, bad, pts)
    assert torch.equal(a0, a1), "padded slots leaked into the attenuation output"
    assert torch.equal(s0, s1), "padded slots leaked into the signal output"


@GPU
def test_pure_attention_is_not_identical_to_masked_mean():
    """Guards the opposite slip from the residual test: if `attn` were accidentally wired to the
    mean branch, X2 would silently be a duplicate of the baseline."""
    a = _model("masked_mean", seed=5)
    b = _model("attn", seed=5)
    cond = build_cond_vector_2d("geom_token", 6.0, 5.0, [0.15] * 6, verts=L_VERTS)
    pts = torch.tensor([[1.0, 1.0], [5.0, 4.0]])
    at, _ = _run(a, cond, pts)
    bt, _ = _run(b, cond, pts)
    assert not torch.equal(at, bt)


@GPU
def test_attention_arms_keep_the_reduced_cond_dim_so_film_is_untouched():
    """The whole 'no renderer change, no FiLM change' claim rests on this."""
    for pool in ("masked_mean", "attn", "attn_residual"):
        m = _model(pool, seed=1)
        assert m.cond_dim == 64, "{} changed the reduced cond_dim to {}".format(pool, m.cond_dim)
        assert m.film_sigma.in_features == 64
        assert m.film_signal.in_features == 64


@GPU
def test_an_unknown_pool_is_rejected():
    from aaf.models.inr_2d import INR2D_AutoDecoder
    with pytest.raises(ValueError, match="token_pool"):
        INR2D_AutoDecoder(n_rooms=4, latent_dim=16, n_freq_bins=N_FREQ,
                          cond_source="geom_token", cond_dim=324, conditioning_type="film",
                          token_pool="attn_typo", world_scale=10.0)


# ------------------------------------------------- load_model must honour token_pool (P4-3)
def test_load_model_passes_token_pool_from_the_checkpoint():
    """The bug this test exists for cost P4-2 two arms' worth of conclusions.

    `load_model` rebuilt the model WITHOUT `token_pool`, so every polygon arm was reconstructed
    as masked_mean. For `extent_sum` that is silent: it adds no parameters, so its state_dict is
    key-identical and `load_state_dict` succeeds -- the checkpoint then renders through the WRONG
    POOLING and reports numbers for an arm that was never evaluated. It only surfaced because the
    attention arms add parameters and failed loudly.

    Read as text so it runs without a GPU.
    """
    src = Path(__file__).resolve().parents[1].joinpath("aaf/eval/p3_2_eval.py").read_text()
    body = src.split("model = INR2D_AutoDecoder(", 1)[1].split(").to(device)", 1)[0]
    assert "token_pool" in body, (
        "load_model does not pass token_pool; extent_sum and attention checkpoints would be "
        "rebuilt as masked_mean")


# ------------------------------------------------- P4-4: attn_residual_extent (CLAUDE.md rule 8)
# The attention block adds parameters, but the thing that distinguishes this arm from
# `attn_residual` is its residual BASE -- extent-weighted sum vs masked mean -- and that base
# adds no parameters at all. So the two arms have key-identical state_dicts and a checkpoint
# round-trip cannot tell them apart. These are behavioural tests, per the standing rule.

@GPU
def test_attn_residual_extent_is_bit_identical_to_extent_sum_at_init():
    """The identity-at-init property this arm is built on: zero-init o_proj means the delta is
    exactly zero, so iteration 0 must reproduce pure extent_sum through the WHOLE network."""
    a = _model("extent_sum", seed=13)
    b = _model("attn_residual_extent", seed=13)
    sd = b.state_dict()
    for k, v in a.state_dict().items():
        assert k in sd
        sd[k] = v.clone()
    b.load_state_dict(sd)
    assert float(b.o_proj.weight.abs().max()) == 0.0
    assert float(b.o_proj.bias.abs().max()) == 0.0
    cond = build_cond_vector_2d("geom_token", 6.0, 5.0, [0.15] * 6, verts=L_VERTS)
    pts = torch.tensor([[1.0, 1.0], [5.0, 4.0], [2.9, 2.4]])
    at, st = _run(a, cond, pts)
    bt, sb = _run(b, cond, pts)
    assert torch.equal(at, bt), "attn_residual_extent is not identical to extent_sum at init"
    assert torch.equal(st, sb)


@GPU
def test_attn_residual_extent_differs_from_attn_residual():
    """The two residual arms share every parameter name and shape. If the base were wired to the
    mean in both, this arm would silently BE X1 and the P4-4 comparison would be vacuous."""
    a = _model("attn_residual", seed=17)
    b = _model("attn_residual_extent", seed=17)
    sd = b.state_dict()
    for k, v in a.state_dict().items():
        sd[k] = v.clone()
    b.load_state_dict(sd)
    # wake o_proj on BOTH so the delta is live and only the base differs
    for m in (a, b):
        torch.nn.init.normal_(m.o_proj.weight, std=0.1)
    b.o_proj.load_state_dict(a.o_proj.state_dict())
    cond = build_cond_vector_2d("geom_token", 6.0, 5.0, [0.15] * 6, verts=L_VERTS)
    pts = torch.tensor([[1.0, 1.0], [5.0, 4.0]])
    at, _ = _run(a, cond, pts)
    bt, _ = _run(b, cond, pts)
    assert not torch.equal(at, bt), "the extent base is not being used"


@GPU
def test_attn_residual_extent_ignores_padding_and_varies_with_position():
    """Both properties in one room: a rectangle fills 4 of 12 slots, so corrupting the 8 padded
    ones must change nothing, and the pooled conditioning must still depend on the query point."""
    m = _model("attn_residual_extent", seed=19)
    torch.nn.init.normal_(m.o_proj.weight, std=0.1)
    cond = build_cond_vector_2d("geom_token", 6.0, 5.0, ALPHAS4)
    n_tok = MAX_SEG_POLY * D_TOK
    bad = cond.clone()
    pad = bad[:n_tok].reshape(MAX_SEG_POLY, D_TOK)
    pad[4:] = torch.randn_like(pad[4:]) * 5.0
    bad[:n_tok] = pad.reshape(-1)
    pts = torch.tensor([[1.0, 1.0], [4.0, 3.0], [0.3, 4.7]])
    a0, s0 = _run(m, cond, pts)
    a1, s1 = _run(m, bad, pts)
    assert torch.equal(a0, a1) and torch.equal(s0, s1), "padded slots leaked into the output"

    seen = {}
    h = m.film_sigma.register_forward_hook(
        lambda _m, i, _o: seen.__setitem__("z", i[0].detach()))
    _run(m, cond, torch.tensor([[0.2, 0.2], [5.8, 4.8], [3.0, 2.5]]))
    h.remove()
    z = seen["z"]
    assert float((z - z.mean(dim=0, keepdim=True)).abs().max()) > 1e-5, \
        "conditioning does not vary with position"
