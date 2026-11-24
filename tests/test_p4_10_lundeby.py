"""P4-10 V7 (D216): the Lundeby compensation recovers a known decay under a noise floor (and a no-op would fail), and leaves a noise-free decay alone (Rule 8: no parameters, so only output can catch a mistake)."""
import numpy as np
import pytest

from scripts.p4_10_lundeby import edc_db, lundeby, times

FS = 600.0
N = 900


def carrier(rng, n, lo=88.0, hi=177.0):
    f = np.fft.rfftfreq(n, 1 / FS)
    X = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) * ((f >= lo) & (f <= hi))
    x = np.fft.irfft(X, n=n)
    return x / x.std()


def synth(rng, t60, floor_db=None):
    t = np.arange(N) / FS
    h = carrier(rng, N) * 10 ** (-3.0 * t / t60)
    if floor_db is not None:
        h = h + carrier(rng, N) * 10 ** (floor_db / 20.0)
    return h


def med(t60, floor_db, comp, key, n=120):
    rng = np.random.default_rng(1)
    out = []
    for _ in range(n):
        h = synth(rng, t60, floor_db)
        L, _ = edc_db(h, FS, comp)
        if L is not None:
            out.append(times(L, FS)[key])
    return float(np.nanmedian(out)), len(out)


# Peak-to-noise ranges are those ISO 3382-1 asks for (T20 >= 35 dB, T30 >= 45 dB). Below them the truncation method itself is biased
# (T30 +9 to +18% at 36-44 dB of range on these synthetics), which is why V7 counts receivers below ISO's range.
@pytest.mark.parametrize("t60", [0.6, 0.8, 1.0])
def test_recovers_t20_and_t30_under_a_floor_and_a_no_op_would_fail(t60):
    raw, _ = med(t60, -48.0, False, "T30")
    cmp_, n = med(t60, -48.0, True, "T30")
    assert n >= 110
    assert abs(raw / t60 - 1) > 0.10, raw                    # the floor really corrupts the raw T30 ...
    assert abs(cmp_ / t60 - 1) < 0.07, cmp_                  # ... and the compensation recovers it (a no-op fails one of the two)
    r20, _ = med(t60, -40.0, False, "T20")
    c20, _ = med(t60, -40.0, True, "T20")
    assert abs(r20 / t60 - 1) > 0.10 and abs(c20 / t60 - 1) < 0.08 and abs(c20 / t60 - 1) < abs(r20 / t60 - 1)


def test_noise_free_decay_is_unchanged_and_edt_is_not_touched():
    for t60 in (0.6, 0.8):
        rng = np.random.default_rng(2)
        for _ in range(20):
            h = synth(rng, t60)
            a, b = times(edc_db(h, FS, False)[0], FS), times(edc_db(h, FS, True)[0], FS)
            for k in ("EDT", "T20", "T30"):
                assert abs(a[k] / b[k] - 1) < 0.01, (k, a[k], b[k])


def test_refuses_what_it_cannot_compensate():
    rng = np.random.default_rng(3)
    assert lundeby(carrier(rng, N), FS) is None              # no decay at all: peak-to-noise range < 20 dB
    assert lundeby(synth(rng, 0.8, -8.0), FS) is None        # 8 dB of range


def test_compensated_curve_is_continuous_at_the_truncation_point():
    rng = np.random.default_rng(4)
    L, info = edc_db(synth(rng, 0.8, -32.0), FS, True)
    k = int(round(info["tc"] * FS))
    assert abs(L[min(k, N - 1)] - L[min(k, N - 1) - 1]) < 0.5 and np.all(np.diff(L) <= 1e-9)
