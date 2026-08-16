"""Fast CPU unit tests for the 2D FDTD solver (aaf.sim.fdtd_2d).

Everything here uses a SMALL room and few time steps -- these are correctness tests, not the
FT-A validation run. Total runtime is a few seconds.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aaf.sim import fdtd_2d as F
from aaf.walls import WALLS_2D


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _compact_burst(n: int, fs: float, f0: float = 400.0, n_burst: int = 96) -> np.ndarray:
    """Hann-windowed sine burst, mean-removed over its support -> exactly zero-mean AND
    exactly compactly supported, so the field is source-free after ``n_burst`` steps."""
    k = np.arange(n_burst)
    b = np.hanning(n_burst + 2)[1:-1] * np.sin(2 * np.pi * f0 * k / fs)
    b = b - b.mean()
    s = np.zeros(n, dtype=np.float64)
    s[:n_burst] = b
    return s


def _parabolic_peak(freqs: np.ndarray, mag: np.ndarray, lo: float, hi: float) -> float:
    """Sub-bin peak location by parabolic interpolation of the log-magnitude."""
    band = np.where((freqs >= lo) & (freqs <= hi))[0]
    k = band[int(np.argmax(mag[band]))]
    y0, y1, y2 = np.log(mag[k - 1]), np.log(mag[k]), np.log(mag[k + 1])
    delta = 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2)
    df = freqs[1] - freqs[0]
    return float(freqs[k] + delta * df)


def _slf_mode_hz(nx_mode: int, ny_mode: int, L: float, W: float, c: float, fs: float, dx: float):
    """Exact SLF discrete-dispersion mode frequency (module docstring, eq. 2)."""
    dt = 1.0 / fs
    lam = c * dt / dx
    kx = nx_mode * math.pi / L
    ky = ny_mode * math.pi / W
    arg = lam * math.sqrt(math.sin(kx * dx / 2) ** 2 + math.sin(ky * dx / 2) ** 2)
    return math.asin(arg) / (math.pi * dt)


SMALL = dict(L=1.0, W=0.8, dx=0.05, fs=12288.0, c=343.0)


# --------------------------------------------------------------------------------------
# 1. CFL
# --------------------------------------------------------------------------------------
def test_cfl_assertion_fires():
    with pytest.raises(ValueError, match="CFL"):
        F.simulate(
            1.0, 0.8, (0.1,) * 4, (0.2, 0.2), [(0.6, 0.4)], dx=0.05, fs=4000.0, n=64, c=343.0
        )


def test_cfl_bound_value_and_frozen_lambda():
    assert F.CFL_MAX_2D == pytest.approx(0.7071067811865476, abs=1e-15)
    lam = 343.0 * (1.0 / 12288.0) / 0.05
    assert lam == pytest.approx(0.5582682292, abs=1e-9)
    assert lam < F.CFL_MAX_2D
    out = F.simulate(1.0, 0.8, (0.1,) * 4, (0.2, 0.2), [(0.6, 0.4)], n=64, **_no_lwd(SMALL))
    assert out["meta"]["lambda_CFL"] == pytest.approx(lam, abs=1e-12)


def _no_lwd(d):
    """SMALL without the positional room args."""
    return {k: v for k, v in d.items() if k not in ("L", "W")}


# --------------------------------------------------------------------------------------
# 2. source: S(0) = 0
# --------------------------------------------------------------------------------------
def test_band_limited_pulse_has_zero_dc():
    s = F.band_limited_pulse(n=4096, fs=12288.0, f_flat=600.0, f_max=800.0)
    S = np.fft.rfft(s)
    assert abs(S[0]) <= 1e-9 * np.max(np.abs(S))
    assert abs(s.sum()) <= 1e-9 * np.max(np.abs(s))


def test_band_limited_pulse_is_band_limited():
    n, fs = 4096, 12288.0
    s = F.band_limited_pulse(n=n, fs=fs, f_flat=600.0, f_max=800.0)
    S = np.abs(np.fft.rfft(s))
    f = np.arange(n // 2 + 1) * fs / n
    assert np.all(S[f > 810.0] <= 1e-9 * S.max())
    assert S[(f > 100.0) & (f < 500.0)].min() >= 0.9 * S.max()


def test_simulate_reports_zero_dc_source():
    out = F.simulate(1.0, 0.8, (0.1,) * 4, (0.2, 0.2), [(0.6, 0.4)], n=1024, **_no_lwd(SMALL))
    S = out["source_spectrum"]
    assert abs(S[0]) <= 1e-9 * np.max(np.abs(S))
    assert out["meta"]["source"]["dc_rel"] <= 1e-9


def test_user_signal_with_dc_is_rejected():
    n = 512
    bad = np.ones(n)  # pure DC
    with pytest.raises(ValueError, match="zero-mean"):
        F.simulate(
            1.0, 0.8, (0.1,) * 4, (0.2, 0.2), [(0.6, 0.4)], n=n, source={"signal": bad},
            **_no_lwd(SMALL)
        )


# --------------------------------------------------------------------------------------
# 3. energy does not increase in a rigid room
# --------------------------------------------------------------------------------------
def test_rigid_room_energy_is_conserved():
    n, n_burst = 1200, 96
    s = _compact_burst(n, SMALL["fs"], n_burst=n_burst)
    out = F.simulate(
        1.0,
        0.8,
        (0.0, 0.0, 0.0, 0.0),
        (0.25, 0.35),
        [(0.7, 0.55)],
        n=n,
        source={"signal": s},
        record_energy=True,
        **_no_lwd(SMALL)
    )
    e = out["energy"]
    tail = e[n_burst + 5 :]
    assert tail.min() > 0.0
    # exactly conserved by leapfrog with Neumann image BCs; float64 drift only
    assert (tail.max() - tail.min()) / tail.mean() < 1e-9
    # and it never exceeds what the source put in
    assert e.max() <= e[: n_burst + 5].max() * (1 + 1e-9)


def test_absorbing_room_energy_is_non_increasing():
    n, n_burst = 1200, 96
    s = _compact_burst(n, SMALL["fs"], n_burst=n_burst)
    out = F.simulate(
        1.0,
        0.8,
        (0.3, 0.15, 0.5, 0.05),
        (0.25, 0.35),
        [(0.7, 0.55)],
        n=n,
        source={"signal": s},
        record_energy=True,
        **_no_lwd(SMALL)
    )
    e = out["energy"][n_burst + 5 :]
    assert np.all(np.diff(e) <= 1e-12 * e.max())
    assert e[-1] < 0.5 * e[0]  # it really is dissipating


# --------------------------------------------------------------------------------------
# 4. reciprocity
# --------------------------------------------------------------------------------------
def test_reciprocity_interior_nodes():
    a = (0.1, 0.25, 0.4, 0.05)  # asymmetric, so this is not a symmetry artefact
    p, q = (0.25, 0.35), (0.70, 0.55)
    n = 900
    kw = dict(n=n, **_no_lwd(SMALL))
    ir_pq = F.simulate(1.0, 0.8, a, p, [q], **kw)["ir"][0]
    ir_qp = F.simulate(1.0, 0.8, a, q, [p], **kw)["ir"][0]
    scale = np.max(np.abs(ir_pq))
    assert scale > 0.0
    assert np.max(np.abs(ir_pq - ir_qp)) <= 1e-11 * scale


# --------------------------------------------------------------------------------------
# 5. alpha = 0 reduces to the rigid boundary
# --------------------------------------------------------------------------------------
def test_admittance_convention():
    assert F.wall_admittance(0.0) == 0.0  # rigid
    assert F.wall_admittance(1.0) == 1.0  # perfectly matched
    assert math.isinf(F.wall_impedance(0.0))
    assert F.wall_impedance(1.0) == 1.0
    # xi is the ADMITTANCE, so R = (1 - xi)/(1 + xi) = sqrt(1 - alpha)
    for alpha in (0.0, 0.05, 0.15, 0.5, 0.7, 1.0):
        xi = F.wall_admittance(alpha)
        assert (1.0 - xi) / (1.0 + xi) == pytest.approx(math.sqrt(1.0 - alpha), abs=1e-14)


def test_alpha_zero_gives_the_rigid_update_coefficients():
    lam = 343.0 / 12288.0 / 0.05
    lam2 = lam * lam
    geom = F.build_geometry(1.0, 0.8, (0.0, 0.0, 0.0, 0.0), dx=0.05)
    coef = F.build_coefficients(geom, lam, lam)

    assert np.all(coef["B"] == 0.0)
    assert np.allclose(coef["cP"], -1.0, atol=0.0, rtol=0.0)
    assert np.allclose(coef["cC"], 2.0 * (1.0 - 2.0 * lam2))

    nx, ny = geom.nx, geom.ny
    # interior node: plain 5-point stencil
    for key in ("cE", "cW", "cN", "cS"):
        assert coef[key][nx // 2, ny // 2] == pytest.approx(lam2, abs=1e-15)
    # west wall (not a corner): the -x neighbour is mirrored onto +x
    assert coef["cE"][0, ny // 2] == pytest.approx(2 * lam2, abs=1e-15)
    assert coef["cW"][0, ny // 2] == 0.0
    assert coef["cN"][0, ny // 2] == pytest.approx(lam2, abs=1e-15)
    # south-west corner: BOTH neighbours mirrored -- the classic failure mode
    assert coef["cE"][0, 0] == pytest.approx(2 * lam2, abs=1e-15)
    assert coef["cN"][0, 0] == pytest.approx(2 * lam2, abs=1e-15)
    assert coef["cW"][0, 0] == 0.0 and coef["cS"][0, 0] == 0.0
    # energy weights: 1 interior, 1/2 wall, 1/4 corner
    assert coef["mu"][nx // 2, ny // 2] == 1.0
    assert coef["mu"][0, ny // 2] == 0.5
    assert coef["mu"][0, 0] == 0.25


def test_alpha_zero_matches_a_naive_rigid_reference_field():
    """Whole-field comparison against an independent per-node loop with image nodes."""
    _naive_field_matches(alphas=(0.0, 0.0, 0.0, 0.0))


def test_absorbing_walls_match_a_naive_reference_field():
    """Same comparison with four DIFFERENT wall absorptions -> exercises all four corners."""
    _naive_field_matches(alphas=(0.0, 0.35, 0.6, 0.15))


def _naive_field_matches(alphas):
    L, W, dx, fs, c = 0.30, 0.20, 0.05, 12288.0, 343.0
    n = 24
    nx, ny = int(round(L / dx)) + 1, int(round(W / dx)) + 1
    lam = c / fs / dx
    lam2 = lam * lam
    xi = [F.wall_admittance(a) for a in alphas]  # west, east, south, north

    s = np.zeros(n)
    s[2] = 1.0
    s[3] = -1.0  # zero mean, compact

    rx = [(i * dx, j * dx) for i in range(nx) for j in range(ny)]
    out = F.simulate(
        L, W, alphas, (2 * dx, 2 * dx), rx, dx=dx, fs=fs, n=n, c=c, source={"signal": s}
    )
    got = out["ir"].reshape(nx, ny, n)

    p_prev = np.zeros((nx, ny))
    p_cur = np.zeros((nx, ny))
    ref = np.zeros((nx, ny, n))
    for step in range(n):
        p_next = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                pe = p_cur[i + 1, j] if i + 1 < nx else p_cur[i - 1, j]
                pw = p_cur[i - 1, j] if i - 1 >= 0 else p_cur[i + 1, j]
                pn = p_cur[i, j + 1] if j + 1 < ny else p_cur[i, j - 1]
                ps = p_cur[i, j - 1] if j - 1 >= 0 else p_cur[i, j + 1]
                b = 0.0
                if i == 0:
                    b += lam * xi[0]
                if i == nx - 1:
                    b += lam * xi[1]
                if j == 0:
                    b += lam * xi[2]
                if j == ny - 1:
                    b += lam * xi[3]
                p_next[i, j] = (
                    lam2 * (pe + pw + pn + ps)
                    + 2.0 * (1.0 - 2.0 * lam2) * p_cur[i, j]
                    - (1.0 - b) * p_prev[i, j]
                ) / (1.0 + b)
        p_next[2, 2] += s[step]
        ref[:, :, step] = p_next
        p_prev, p_cur = p_cur, p_next

    assert np.max(np.abs(got - ref)) <= 1e-12 * max(np.max(np.abs(ref)), 1e-30)


# --------------------------------------------------------------------------------------
# 6. grid / geometry conventions
# --------------------------------------------------------------------------------------
def test_grid_is_node_centred_not_half_cell():
    geom = F.build_geometry(4.5, 4.0, (0.15,) * 4, dx=0.05)
    assert (geom.nx, geom.ny) == (91, 81)  # L/dx + 1, NOT L/dx
    assert geom.L_grid == pytest.approx(4.5, abs=1e-12)
    assert geom.W_grid == pytest.approx(4.0, abs=1e-12)


def test_wall_order_matches_aaf_walls():
    assert list(WALLS_2D) == ["west", "east", "south", "north"]
    a = (0.11, 0.22, 0.33, 0.44)
    geom = F.build_geometry(1.0, 0.8, a, dx=0.05)
    assert np.allclose(geom.face_alpha[F.XM, 0, :], 0.11)  # west  -> x = 0
    assert np.allclose(geom.face_alpha[F.XP, -1, :], 0.22)  # east  -> x = L
    assert np.allclose(geom.face_alpha[F.YM, :, 0], 0.33)  # south -> y = 0
    assert np.allclose(geom.face_alpha[F.YP, :, -1], 0.44)  # north -> y = W


def test_absorbing_wall_pair_orientation_is_physical():
    """A 2.0 x 0.5 room: the x-normal walls are 4x shorter than the y-normal ones, so
    absorbing the x pair must dissipate far more slowly than absorbing the y pair. Catches a
    west/south transposition, which the face-placement test alone cannot."""
    n, n_burst = 1400, 96
    s = _compact_burst(n, 12288.0, n_burst=n_burst)
    kw = dict(dx=0.05, fs=12288.0, c=343.0, n=n, source={"signal": s}, record_energy=True)
    e_x = F.simulate(2.0, 0.5, (0.5, 0.5, 0.0, 0.0), (0.4, 0.25), [(1.6, 0.25)], **kw)["energy"]
    e_y = F.simulate(2.0, 0.5, (0.0, 0.0, 0.5, 0.5), (0.4, 0.25), [(1.6, 0.25)], **kw)["energy"]
    assert e_x[-1] > 3.0 * e_y[-1]


def test_receiver_snap_offset_is_reported():
    out = F.simulate(
        1.0, 0.8, (0.1,) * 4, (0.2, 0.2), [(0.63, 0.42)], n=64, **_no_lwd(SMALL)
    )
    meta = out["meta"]
    assert meta["rx_nodes"] == [[13, 8]]
    assert meta["rx_pos_snapped"] == [[0.65, 0.4]]
    assert meta["rx_offset_m"][0] == pytest.approx(math.hypot(0.02, 0.02), abs=1e-12)
    assert meta["rx_offset_max_m"] <= 0.05 / math.sqrt(2) + 1e-12


def test_meta_records_the_required_block():
    out = F.simulate(1.0, 0.8, (0.15, 0.0, 0.5, 0.7), (0.2, 0.2), [(0.6, 0.4)], n=128,
                     **_no_lwd(SMALL))
    m = out["meta"]
    for key in ("dx", "fs", "lambda_CFL", "grid_shape", "wall_xi_admittance",
                "wall_impedance_normalized", "throughput"):
        assert key in m
    assert m["grid_shape"] == [21, 17]
    assert m["wall_xi_admittance"]["east"] == 0.0
    assert math.isinf(m["wall_impedance_normalized"]["east"])
    assert m["wall_xi_admittance"]["north"] == pytest.approx(F.wall_admittance(0.7))
    assert m["throughput"]["node_updates_per_s"] > 0.0
    assert out["ir"].shape == (1, 128)
    assert out["H_complex"].shape == (1, 65)
    assert out["freqs"].shape == (65,)


# --------------------------------------------------------------------------------------
# 7. physics smoke test: the lowest axial mode lands on the SLF dispersion relation
# --------------------------------------------------------------------------------------
def test_lowest_axial_mode_frequency():
    L, W, dx, fs, c = 1.0, 0.8, 0.05, 12288.0, 343.0
    n = 4096
    out = F.simulate(L, W, (0.1,) * 4, (0.10, 0.10), [(0.90, 0.70)], dx=dx, fs=fs, n=n, c=c)
    mag = np.abs(out["H_deconv"][0])
    f_est = _parabolic_peak(out["freqs"], mag, 120.0, 200.0)
    f_ref = _slf_mode_hz(1, 0, L, W, c, fs, dx)
    assert f_ref == pytest.approx(171.37, abs=0.5)  # guards the reference itself
    assert abs(f_est - f_ref) / f_ref < 0.01
    # a half-cell grid error would put the mode near f_ref * (1 - dx/L) = -5%
    assert abs(f_est - f_ref * (1 - dx / L)) / f_ref > 0.02


# --------------------------------------------------------------------------------------
# 7b. exact spectrum of the discrete operator -- no FFT, no peak picking
# --------------------------------------------------------------------------------------
def _update_matrix(geom, coef):
    """Dense ``A`` and ``cP`` for ``p_next = A p_cur + diag(cP) p_prev``."""
    nx, ny = geom.nx, geom.ny
    n = nx * ny
    A = np.zeros((n, n))
    cP = np.zeros(n)
    for i in range(nx):
        for j in range(ny):
            r = i * ny + j
            A[r, r] += coef["cC"][i, j]
            cP[r] = coef["cP"][i, j]
            if i + 1 < nx:
                A[r, (i + 1) * ny + j] += coef["cE"][i, j]
            if i - 1 >= 0:
                A[r, (i - 1) * ny + j] += coef["cW"][i, j]
            if j + 1 < ny:
                A[r, i * ny + (j + 1)] += coef["cN"][i, j]
            if j - 1 >= 0:
                A[r, i * ny + (j - 1)] += coef["cS"][i, j]
    return A, cP


def test_discrete_dispersion_relation_is_exact():
    """The rigid operator's eigenfrequencies must equal the SLF dispersion relation to
    machine precision. This is the assumption gate A1b rests on, tested without any
    spectral estimation in the way."""
    L, W, dx, fs, c = 0.30, 0.20, 0.05, 12288.0, 343.0
    dt = 1.0 / fs
    geom = F.build_geometry(L, W, (0.0,) * 4, dx=dx)
    A, cP = _update_matrix(geom, F.build_coefficients(geom, c / fs / dx, c / fs / dx))
    assert np.all(cP == -1.0)
    mu = np.clip(np.linalg.eigvals(A).real / 2.0, -1.0, 1.0)
    got = np.sort(np.arccos(mu) / (2 * np.pi * dt))
    ref = np.sort(
        np.array(
            [
                _slf_mode_hz(m, q, L, W, c, fs, dx)
                for m in range(geom.nx)
                for q in range(geom.ny)
            ]
        )
    )
    assert got.shape == ref.shape
    assert np.max(np.abs(got[1:] - ref[1:]) / ref[1:]) < 1e-10


def test_modal_damping_follows_the_locally_reacting_law():
    """The KW boundary must reproduce the Kuttruff (wave / locally-reacting) damping law,
    NOT the ISM ray law -- that distinction is the whole point of moving to a wave solver.

    Kuttruff linearizes the admittance as xi ~ alpha/4, so the exact reference is the
    Kuttruff rate rescaled by ``4*xi(alpha)/alpha``.
    """
    from aaf.sim.analytical_modal_2d import modal_damping_2d

    L, W, dx, fs, c = 0.50, 0.40, 0.05, 12288.0, 343.0
    alpha = 0.05
    alphas = (alpha,) * 4
    dt = 1.0 / fs
    geom = F.build_geometry(L, W, alphas, dx=dx)
    A, cP = _update_matrix(geom, F.build_coefficients(geom, c / fs / dx, c / fs / dx))
    nn = A.shape[0]
    comp = np.zeros((2 * nn, 2 * nn))
    comp[:nn, :nn] = A
    comp[:nn, nn:] = np.diag(cP)
    comp[nn:, :nn] = np.eye(nn)
    z = np.linalg.eigvals(comp)
    keep = np.angle(z) > 1e-9
    f_mode = np.angle(z[keep]) / (2 * np.pi * dt)
    gamma = -np.log(np.abs(z[keep])) / dt

    corr = 4.0 * F.wall_admittance(alpha) / alpha
    for m, q in [(1, 0), (0, 1), (1, 1)]:
        f_c = (c / 2.0) * math.hypot(m / L, q / W)
        k = int(np.argmin(np.abs(f_mode - f_c)))
        g_kut = modal_damping_2d(L, W, alphas, m, q, c=c, model="kuttruff") * corr
        g_ism = modal_damping_2d(L, W, alphas, m, q, c=c, model="ism_ray")
        assert abs(gamma[k] - g_kut) / g_kut < 5e-3, (m, q, gamma[k], g_kut)
        assert gamma[k] / g_ism > 1.2, (m, q, gamma[k], g_ism)


def test_rigid_walls_are_lossless_in_the_operator_spectrum():
    geom = F.build_geometry(0.30, 0.20, (0.0,) * 4, dx=0.05)
    A, cP = _update_matrix(geom, F.build_coefficients(geom, 343.0 / 12288.0 / 0.05, 343.0 / 12288.0 / 0.05))
    nn = A.shape[0]
    comp = np.zeros((2 * nn, 2 * nn))
    comp[:nn, :nn] = A
    comp[:nn, nn:] = np.diag(cP)
    comp[nn:, :nn] = np.eye(nn)
    z = np.abs(np.linalg.eigvals(comp))
    assert np.max(np.abs(z - 1.0)) < 1e-10  # every mode on the unit circle: no growth, no loss


# --------------------------------------------------------------------------------------
# 8. interior structure (extra_walls)
# --------------------------------------------------------------------------------------
def test_solid_divider_fully_blocks_the_far_side():
    div = {"type": "slab", "axis": "x", "pos": 0.5}
    out = F.simulate(
        1.0, 0.8, (0.1,) * 4, (0.20, 0.40), [(0.80, 0.40)], n=600, extra_walls=[div],
        **_no_lwd(SMALL)
    )
    assert np.max(np.abs(out["ir"][0])) == 0.0
    assert out["meta"]["extra_walls"][0]["n_solid_nodes"] == 17


def test_aperture_lets_sound_through():
    div = {"type": "slab", "axis": "x", "pos": 0.5, "apertures": [(0.30, 0.50)]}
    out = F.simulate(
        1.0, 0.8, (0.1,) * 4, (0.20, 0.40), [(0.80, 0.40)], n=600, extra_walls=[div],
        **_no_lwd(SMALL)
    )
    assert np.max(np.abs(out["ir"][0])) > 0.0
    ap = out["meta"]["extra_walls"][0]["apertures"][0]
    assert ap["clear_width_m"] == pytest.approx(0.20, abs=1e-12)
    assert ap["nodes"] == [6, 10]


def test_absorbing_divider_damps_more_than_a_rigid_one():
    n, n_burst = 1200, 96
    s = _compact_burst(n, SMALL["fs"], n_burst=n_burst)
    kw = dict(n=n, source={"signal": s}, record_energy=True, **_no_lwd(SMALL))
    base = {"type": "slab", "axis": "x", "pos": 0.5, "apertures": [(0.30, 0.50)]}
    rigid = F.simulate(1.0, 0.8, (0.0,) * 4, (0.20, 0.40), [(0.80, 0.40)],
                       extra_walls=[dict(base, alpha=0.0)], **kw)["energy"]
    soft = F.simulate(1.0, 0.8, (0.0,) * 4, (0.20, 0.40), [(0.80, 0.40)],
                      extra_walls=[dict(base, alpha=0.6)], **kw)["energy"]
    assert soft[-1] < 0.3 * rigid[-1]


def test_partial_wall_absorber_patch():
    """The realized ABSORBING extent must equal the requested span.

    Updated for the A0b fix (FT-A blocker B2). The previous expectation encoded the bug: an
    inclusive slice ``j0:j1+1`` painted ``round(a/dx) + 1`` nodes, so a 0.40 m request
    absorbed like 0.45 m -- one full cell too wide, and reported as nominal. The corrected
    rule paints 8 nodes (5..12), realizing 0.40 m exactly.
    """
    patch = {"type": "patch", "wall": "north", "span": (0.20, 0.60), "alpha": 0.8}
    geom = F.build_geometry(1.0, 0.8, (0.1,) * 4, dx=0.05, extra_walls=[patch])
    north = geom.face_alpha[F.YP, :, -1]
    assert np.allclose(north[5:13], 0.8)
    assert np.allclose(north[:5], 0.1)
    assert np.allclose(north[13:], 0.1)
    assert geom.adm[F.YP, 6, -1] == pytest.approx(F.wall_admittance(0.8))

    spec = geom.specs[0]
    assert spec["n_nodes"] == 8
    assert spec["width_requested_m"] == pytest.approx(0.40)
    assert spec["width_realized_m"] == pytest.approx(0.40)
    assert abs(spec["width_error_m"]) < 1e-12


def test_zero_width_channel_is_rejected():
    walls = [
        {"type": "slab", "axis": "x", "pos": 0.50},
        {"type": "slab", "axis": "x", "pos": 0.60},
    ]
    with pytest.raises(ValueError, match="zero-width channel"):
        F.build_geometry(1.0, 0.8, (0.1,) * 4, dx=0.05, extra_walls=walls)


def test_one_node_aperture_is_rejected():
    div = {"type": "slab", "axis": "x", "pos": 0.5, "apertures": [(0.30, 0.30)]}
    with pytest.raises(ValueError, match="aperture"):
        F.build_geometry(1.0, 0.8, (0.1,) * 4, dx=0.05, extra_walls=[div])


def test_receiver_inside_solid_is_rejected():
    div = {"type": "slab", "axis": "x", "pos": 0.5}
    with pytest.raises(ValueError, match="solid"):
        F.simulate(1.0, 0.8, (0.1,) * 4, (0.20, 0.40), [(0.50, 0.40)], n=64,
                   extra_walls=[div], **_no_lwd(SMALL))
