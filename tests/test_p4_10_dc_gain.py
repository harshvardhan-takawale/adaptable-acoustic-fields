"""P4-10 A14 / D200: the DC gain from the geometry tokens -- exactly 1 at alpha 0.15, the physical ratio elsewhere."""
import math
from types import SimpleNamespace

import pytest
import torch

from aaf.models.conditioning_2d import build_cond_vector_2d
from aaf.models.inr_2d import INR2D_AutoDecoder

VERTS = [(0.0, 0.0), (5.0, 0.0), (5.0, 4.2), (1.5, 4.2), (1.5, 2.8), (0.0, 2.8)]


def _beta(a):
    r = math.sqrt(1 - a)
    return (1 - r) / (1 + r)


def _gain(alphas):
    z = build_cond_vector_2d("geom_token", 5.0, 4.2, alphas, verts=VERTS).float().unsqueeze(0)
    return float(INR2D_AutoDecoder._dc_beta_gain(SimpleNamespace(), z)[0])


def test_exactly_one_at_the_base_alpha():
    """bit-identity: every edge at 0.15 must give a gain of exactly 1.0, not 1 + 1e-7"""
    z = build_cond_vector_2d("geom_token", 5.0, 4.2, [0.15] * 6, verts=VERTS).float().unsqueeze(0)
    assert float(INR2D_AutoDecoder._dc_beta_gain(SimpleNamespace(), z)[0]) == 1.0
    assert torch.equal(INR2D_AutoDecoder._dc_beta_gain(SimpleNamespace(), z.repeat(3, 1)), torch.ones(3))


def test_ratio_matches_the_compliance_law():
    al = [0.4, 0.15, 0.3, 0.5, 0.15, 0.2]
    n = 6
    ell = [math.hypot(VERTS[(k + 1) % n][0] - VERTS[k][0], VERTS[(k + 1) % n][1] - VERTS[k][1]) for k in range(n)]
    want = sum(_beta(0.15) * x for x in ell) / sum(_beta(a) * x for a, x in zip(al, ell))
    assert _gain(al) == pytest.approx(want, rel=1e-4)
    assert _gain(al) < 1.0                                   # more absorption, lower DC level


def test_longer_absorbing_edge_lowers_the_gain():
    a = [0.15] * 6
    b = list(a)
    b[2] = 0.6                                               # a 2.7 m wall versus a 1.4 m wall
    c = list(a)
    c[4] = 0.6
    assert _gain(b) < _gain(c) < 1.0
