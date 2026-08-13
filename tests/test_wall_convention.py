"""The wall convention, pinned three ways.

A west/south mix-up would train a model that is confidently wrong in a way no aggregate
metric detects, so the convention is asserted at every layer it crosses:

1. GEOMETRY -- an image-source probe proves pyroomacoustics puts `west` at x=0 (and that
   the per-bounce factor is the PRESSURE reflection coefficient sqrt(1-alpha)).
2. PHYSICS  -- ISM ground truth with west=M3 damps the x-axial family and west=M1
   sharpens it (the spec's blocking-gate signature #2, automated and bidirectional).
3. PLUMBING -- the conditioning encoder and the demo CLI put alpha_west in the same slot.
"""
from __future__ import annotations

import numpy as np
import pytest

pra = pytest.importorskip("pyroomacoustics")

from aaf.eval.modal_bandwidth import caps_from_predicted_bw, measure_modes  # noqa: E402
from aaf.eval.modal_projection import X_AXIAL, Y_AXIAL, project_field  # noqa: E402
from aaf.sim.analytical_modal_2d import (  # noqa: E402
    damping_to_bandwidth_hz,
    modal_damping_2d,
)
from aaf.walls import WALLS_2D, alphas_for, resolve_material, resolve_wall  # noqa: E402

L, W, FS, N, MAX_ORDER = 4.5, 4.0, 4096, 8192, 60
SRC = [0.5, 0.5]


def _grid():
    xs = np.linspace(0.3, L - 0.3, 8)
    ys = np.linspace(0.3, W - 0.3, 8)
    return np.array([[x, y] for y in ys for x in xs])   # row-major: outer y, inner x


def _simulate(alphas, rx):
    mats = {w: pra.Material(energy_absorption=float(a)) for w, a in zip(WALLS_2D, alphas)}
    room = pra.ShoeBox(p=[L, W], fs=FS, materials=mats, max_order=MAX_ORDER,
                       ray_tracing=False)
    room.add_source(SRC)
    room.add_microphone_array(pra.MicrophoneArray(rx.T, FS))
    room.compute_rir()
    out = np.zeros((len(rx), N))
    for i, r in enumerate(room.rir):
        h = np.asarray(r[0])
        n = min(len(h), N)
        out[i, :n] = h[:n]
    return np.fft.rfft(out, n=N, axis=1)


def test_shoebox_wall_names_match_our_order():
    room = pra.ShoeBox(p=[L, W], fs=FS,
                       materials=pra.Material(energy_absorption=0.15), max_order=0)
    assert room.dim == 2
    assert set(room.wall_names) == set(WALLS_2D)
    assert tuple(room.wall_names) == WALLS_2D, (
        "pyroomacoustics reordered its 2D wall names; aaf.walls.WALLS_2D must follow"
    )


def test_west_is_x0_by_image_lattice_probe():
    """Absorb ONLY the west wall: the first-order image at x = -x_src must carry
    sqrt(1-alpha) and every other first-order image must be undamped."""
    alpha = 0.75
    mats = {w: pra.Material(energy_absorption=(alpha if w == "west" else 0.0))
            for w in WALLS_2D}
    room = pra.ShoeBox(p=[L, W], fs=FS, materials=mats, max_order=1, ray_tracing=False)
    room.add_source(SRC)
    room.add_microphone_array(pra.MicrophoneArray(_grid()[:1].T, FS))
    room.image_source_model()
    src = room.sources[0]
    images, damping = np.asarray(src.images), np.asarray(src.damping).ravel()

    expected = np.sqrt(1.0 - alpha)                       # PRESSURE reflection coefficient
    west_img = np.flatnonzero(np.isclose(images[0], -SRC[0], atol=1e-9))
    assert west_img.size == 1, "expected exactly one image mirrored across x=0"
    assert damping[west_img[0]] == pytest.approx(expected, abs=1e-6)

    for target, axis in ((2 * L - SRC[0], 0), (-SRC[1], 1), (2 * W - SRC[1], 1)):
        idx = np.flatnonzero(np.isclose(images[axis], target, atol=1e-9))
        assert idx.size >= 1
        assert damping[idx[0]] == pytest.approx(1.0, abs=1e-6), (
            "only the west wall was absorbing; image across "
            f"axis {axis} at {target} should be undamped"
        )


@pytest.mark.parametrize(
    "wall,fam_moved,fam_still", [("west", X_AXIAL, Y_AXIAL), ("south", Y_AXIAL, X_AXIAL)]
)
def test_ism_selectivity_signature_and_bidirectionality(wall, fam_moved, fam_still):
    """west/east move the x-family, south/north the y-family; M3 damps, M1 sharpens."""
    rx = _grid()
    f_axis = np.arange(N // 2 + 1) * FS / N

    def family_bw(alphas):
        pr = project_field(_simulate(alphas, rx), rx, L, W, src=SRC, fs=FS)
        bw_pred = [damping_to_bandwidth_hz(
            modal_damping_2d(L, W, alphas, m.n_x, m.n_y, model="ism_ray")) for m in pr.modes]
        peaks = measure_modes(pr.spectra, f_axis, pr.modes,
                              caps=caps_from_predicted_bw(bw_pred))
        out = {}
        for fam in (X_AXIAL, Y_AXIAL):
            idx = [i for i in pr.by_family(fam) if peaks[i].bw_valid][:3]
            assert idx, f"no resolvable {fam} modes"
            out[fam] = float(np.mean([peaks[i].bw_3db_hz for i in idx]))
        return out

    base = family_bw(alphas_for())
    damped = family_bw(alphas_for(wall, "M3"))
    sharpened = family_bw(alphas_for(wall, "M1"))

    d_moved = damped[fam_moved] - base[fam_moved]
    d_still = damped[fam_still] - base[fam_still]
    assert d_moved > 2.0, f"{wall}->M3 should broaden {fam_moved}, got {d_moved:+.3f} Hz"
    # Selectivity on BANDWIDTH (D47): measured ~30-50:1; require a robust 5:1.
    assert abs(d_moved) / max(abs(d_still), 0.15) >= 5.0

    # Bidirectional: concrete (alpha below baseline) must SHARPEN the same family.
    d_m1 = sharpened[fam_moved] - base[fam_moved]
    assert d_m1 < -0.5, f"{wall}->M1 should sharpen {fam_moved}, got {d_m1:+.3f} Hz"


def test_encoder_and_cli_agree_on_the_wall_slot():
    torch = pytest.importorskip("torch")
    from aaf.models.conditioning_2d import normalize_params_2d

    for i, wall in enumerate(WALLS_2D):
        u = normalize_params_2d(L, W, alphas_for(wall, "M3"))
        edited = [j for j in range(4) if float(u[2 + j]) > float(u[2 + (j + 1) % 4]) + 1e-6
                  or float(u[2 + j]) == pytest.approx(0.70 / 0.7)]
        assert i in edited, f"{wall} did not land in conditioning slot {2 + i}"
    # the demo CLI's names resolve to the same canonical wall/material ids
    assert resolve_wall("west") == "west"
    assert resolve_material("curtain") == "M2"
    assert resolve_material("absorber") == "M3"
    assert resolve_material("concrete") == "M1"
    with pytest.raises(ValueError):
        resolve_wall("left")
