"""Publication guard (P3-2c audit A1).

P3-2b's flagship results doc printed `slope_fit.aggregate.own_family.all.rho_median`
(a diagnostic that pools slab and non-slab cells) while the acceptance gate consumed
`...slab_local.rho_median`. For arm A those differ by 0.378 -- the doc showed rho = 0.887,
INSIDE the +/-0.25 acceptance band, directly beside a FAIL verdict. No verdict was wrong,
but the document contradicted its own gate.

These tests make that class of error mechanical rather than editorial:
  1. every summary.json on disk agrees that the published rho IS the gated rho;
  2. every rho printed in a human-facing document matches slab_local for that arm.
"""
from __future__ import annotations

import glob
import json
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARIES = sorted(glob.glob(os.path.join(REPO, "outputs", "**", "summary.json"),
                             recursive=True))


def _own(d):
    return ((d.get("slope_fit") or {}).get("aggregate") or {}).get("own_family") or {}


def _slab_local_rho(d):
    return (_own(d).get("slab_local") or {}).get("rho_median")


@pytest.mark.skipif(not SUMMARIES, reason="no eval summaries on disk yet")
def test_published_rho_is_the_gated_rho():
    checked = 0
    for f in SUMMARIES:
        d = json.load(open(f))
        v = d.get("verdict") or {}
        if "rho_used" not in v or not _own(d):
            continue
        used = float(v["rho_used"])
        slab = _slab_local_rho(d)
        assert slab is not None, f"{f}: slab_local.rho_median missing"
        assert abs(used - float(slab)) < 1e-9, (
            f"{f}: verdict.rho_used={used} but slab_local.rho_median={slab}")
        pub = d["slope_fit"].get("rho_published")
        if pub is not None:
            assert abs(float(pub) - float(slab)) < 1e-9, (
                f"{f}: rho_published={pub} != slab_local={slab}")
        checked += 1
    assert checked > 0, "no summary carried both a verdict and a slope_fit aggregate"


@pytest.mark.skipif(not SUMMARIES, reason="no eval summaries on disk yet")
def test_all_aggregate_is_never_the_gate_source():
    """The `all` pool must not be what the gate consumed -- that is the A1 error."""
    for f in SUMMARIES:
        d = json.load(open(f))
        v = d.get("verdict") or {}
        if "rho_used" not in v or not _own(d):
            continue
        pol = (d.get("slope_fit") or {}).get("publication_policy")
        if pol:
            assert pol["gate_source"].endswith("slab_local.rho_median")
            assert "aggregate.own_family.all" in pol["diagnostic_only"]


def test_p3_2b_results_doc_quotes_slab_local():
    """Every rho in the P3-2b arm table must be that arm's slab_local value."""
    doc = os.path.join(REPO, "tasks", "CHUNK_P3_2B_RESULTS.md")
    if not os.path.exists(doc):
        pytest.skip("results doc absent")
    text = open(doc).read()
    expected = {}
    for f in SUMMARIES:
        if "p3_2b" not in f:
            continue
        d = json.load(open(f))
        rho = _slab_local_rho(d)
        if rho is None:
            continue
        arm = os.path.basename(os.path.dirname(f))          # p3_2b_A_preset_fourier
        expected[arm.split("_")[2]] = round(float(rho), 3)  # A / B / C / D
    if not expected:
        pytest.skip("no p3_2b summaries")
    # the arm table rows look like: | **A** | ... | 0.509 | **FAIL** |
    for letter, rho in expected.items():
        row = re.search(r"^\|\s*\*{0,2}" + letter + r"\*{0,2}\s*\|.*$", text, re.M)
        assert row, f"no table row for arm {letter}"
        assert f"{rho:.3f}" in row.group(0), (
            f"arm {letter}: table row does not quote slab_local rho={rho:.3f}. "
            f"Row: {row.group(0)}")


def test_correction_block_records_the_error():
    doc = os.path.join(REPO, "tasks", "CHUNK_P3_2B_RESULTS.md")
    if not os.path.exists(doc):
        pytest.skip("results doc absent")
    text = open(doc).read()
    assert "audit A1" in text, "the A1 correction block must stay in the record"
    assert "0.887" in text and "0.509" in text, (
        "the correction must state both the printed and the gated value")
