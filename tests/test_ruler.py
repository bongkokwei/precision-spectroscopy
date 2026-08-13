"""Self-check for the MZI frequency ruler, against a synthetic chirp.

The point of the auxiliary interferometer is that it recovers the frequency
axis even when the laser sweeps nonlinearly, so the test chirp is deliberately
quadratic in time. If the ruler only worked for a perfectly linear sweep it
would be useless on real hardware.
"""

import numpy as np
import pytest

from spectro.ruler import phase_rad, relative_freq_hz, resample_uniform

FSR_HZ = 41.0e6  # ~5 m of fibre
FS_HZ = 1.0e6
DURATION_S = 0.1
TRIM = 0.02  # Hilbert transform rings at the record edges

# Chirp coefficients chosen so the record holds ~1200 fringes at 40-140 samples
# per fringe -- the same regime as an 80 nm sweep at 10 nm/s digitised at
# 2.5 MS/s. A test with only a handful of fringes tells you nothing: the
# analytic signal is meaningless when the carrier is comparable to the record
# length, and it will fail for reasons the real instrument never sees.
CHIRP_LINEAR_HZ_S = 3.0e11
CHIRP_QUAD_HZ_S2 = 2.0e12


def synthetic_mzi(noise: float = 0.0, seed: int = 0):
    """Return (t, mzi_trace, true_relative_frequency_hz) for a quadratic chirp."""
    t = np.arange(0.0, DURATION_S, 1.0 / FS_HZ)
    nu = CHIRP_LINEAR_HZ_S * t + CHIRP_QUAD_HZ_S2 * t**2  # Hz traversed since t=0
    signal = np.cos(2.0 * np.pi * nu / FSR_HZ)
    if noise:
        signal = signal + noise * np.random.default_rng(seed).standard_normal(len(t))
    return t, signal, nu


def _interior(a):
    k = int(len(a) * TRIM)
    return a[k:-k]


def _spacing_error(nu_rec, nu_true):
    """Error in the recovered axis after removing a common offset.

    The ruler is only claimed to be accurate in its spacing. Its origin carries
    the analytic signal's edge transient (a few MHz) and, on real hardware, the
    laser's absolute wavelength error (~GHz) -- both constants, and both the gas
    cell's problem rather than the ruler's.
    """
    residual = _interior(nu_rec) - _interior(nu_true)
    return np.abs(residual - residual.mean())


def test_recovers_nonlinear_chirp():
    _, signal, nu_true = synthetic_mzi()
    err = _spacing_error(relative_freq_hz(signal, FSR_HZ, FS_HZ), nu_true)
    assert err.max() < 1.0e3, f"worst error {err.max():.1f} Hz on a {nu_true[-1]:.3g} Hz span"


@pytest.mark.parametrize("noise", [0.005, 0.02, 0.05])
def test_noise_floor_scales_as_white_phase_noise(noise):
    """Per-sample frequency noise must track sigma_nu = (sigma_phi/2pi)*FSR.

    Amplitude noise on the MZI trace turns directly into phase noise and hence
    into jitter on the frequency axis. Checking that the measured jitter scales
    linearly with the injected noise -- rather than checking it against some
    hand-picked threshold -- is what distinguishes "the ruler is noisy because
    the input was noisy" from "the ruler is broken".

    The measured coefficient is ~0.6 rather than 1.0 because the analytic signal
    rejects the noise component in quadrature with the carrier.
    """
    _, signal, nu_true = synthetic_mzi(noise=noise)
    err = _spacing_error(relative_freq_hz(signal, FSR_HZ, FS_HZ), nu_true)
    predicted = noise / (2.0 * np.pi) * FSR_HZ
    assert 0.4 < err.std() / predicted < 0.9, (
        f"jitter {err.std():.3g} Hz vs predicted {predicted:.3g} Hz"
    )


def test_is_monotonic():
    """A non-monotonic axis means a cycle slip; np.interp would silently lie."""
    _, signal, _ = synthetic_mzi()
    nu_rec = _interior(relative_freq_hz(signal, FSR_HZ, FS_HZ))
    assert np.all(np.diff(nu_rec) > 0)


def test_resonance_lands_at_the_right_frequency_end_to_end():
    """A resonance dip must come out of the ruler at the frequency it went in at.

    Uses the *recovered* axis, not the ground-truth one, so this exercises the
    whole chain: Hilbert phase -> frequency axis -> interpolation.
    """
    t, signal, nu_true = synthetic_mzi()
    idx = len(t) // 2
    dut = 1.0 - 0.9 * np.exp(-0.5 * ((np.arange(len(t)) - idx) / 20.0) ** 2)

    nu_rec = relative_freq_hz(signal, FSR_HZ, FS_HZ)
    offset = (_interior(nu_rec) - _interior(nu_true)).mean()  # gas cell's job
    grid, dut_uniform = resample_uniform(nu_rec - offset, dut, trim_frac=TRIM)
    recovered = grid[np.argmin(dut_uniform)]
    assert abs(recovered - nu_true[idx]) < 1.0e6


def test_bandpass_saves_the_phase_from_a_drifting_baseline():
    """Slow intensity drift is what breaks the analytic phase on real traces.

    The 20260813 record does exactly this: unwrapped straight off the raw MZI
    the phase goes non-monotonic, and band-passing it fixes that. The drift here
    is deliberately larger than the fringes themselves, as it is on a trace
    whose photodiode is riding the laser's output-power ripple.
    """
    t, signal, nu_true = synthetic_mzi()
    phi_true = 2.0 * np.pi * nu_true / FSR_HZ
    drifted = signal + 3.0 * np.sin(2.0 * np.pi * 20.0 * t)

    # The chirp runs 7.3 -> 17.1 kHz; leave the usual ~1.5x margin either side.
    phi = phase_rad(drifted, FS_HZ, bandpass_hz=(5.0e3, 25.0e3))
    err = _spacing_error(phi, phi_true)
    assert np.all(np.diff(_interior(phi)) > 0), "band-passed phase must be monotonic"
    assert err.max() < 0.05, f"worst phase error {err.max():.4f} rad"

    # Without the band-pass the same trace is unusable -- if this ever stops
    # being true the band-pass has become optional, not the test wrong.
    assert not np.all(np.diff(_interior(phase_rad(drifted, FS_HZ))) > 0)


def test_resample_rejects_length_mismatch():
    with pytest.raises(ValueError):
        resample_uniform(np.arange(10.0), np.arange(9.0))
