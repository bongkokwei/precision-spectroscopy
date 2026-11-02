"""Self-check for the MZI frequency ruler, against a synthetic chirp.

The point of the auxiliary interferometer is that it recovers the frequency
axis even when the laser sweeps nonlinearly, so the test chirp is deliberately
quadratic in time. If the ruler only worked for a perfectly linear sweep it
would be useless on real hardware.
"""

import logging

import h5py
import numpy as np
import pytest

from spectro import ruler
from spectro.ruler import (
    assign_lines,
    find_dips,
    fit_nu_phi,
    fit_resonances,
    nu_hz,
    phase_rad,
    read_ruler_fit,
    resample_uniform,
)

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


def _relative_freq_hz(mzi, fsr_hz, fs_hz, bandpass_hz=None):
    """Phase scaled to Hz by a single fixed, constant FSR.

    Local test helper, not production code: ruler.py no longer ships this
    conversion (nothing in the package called it -- the gas-anchored fit in
    fit_nu_phi is what actually builds an absolute frequency axis now). Kept
    here because it is still the simplest way to check phase_rad's accuracy
    in frequency units against a synthetic chirp.
    """
    phi = phase_rad(mzi, fs_hz, bandpass_hz)
    return (phi - phi[0]) / (2.0 * np.pi) * float(fsr_hz)


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
    err = _spacing_error(_relative_freq_hz(signal, FSR_HZ, FS_HZ), nu_true)
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
    err = _spacing_error(_relative_freq_hz(signal, FSR_HZ, FS_HZ), nu_true)
    predicted = noise / (2.0 * np.pi) * FSR_HZ
    assert 0.4 < err.std() / predicted < 0.9, (
        f"jitter {err.std():.3g} Hz vs predicted {predicted:.3g} Hz"
    )


def test_is_monotonic():
    """A non-monotonic axis means a cycle slip; np.interp would silently lie."""
    _, signal, _ = synthetic_mzi()
    nu_rec = _interior(_relative_freq_hz(signal, FSR_HZ, FS_HZ))
    assert np.all(np.diff(nu_rec) > 0)


def test_resonance_lands_at_the_right_frequency_end_to_end():
    """A resonance dip must come out of the ruler at the frequency it went in at.

    Uses the *recovered* axis, not the ground-truth one, so this exercises the
    whole chain: Hilbert phase -> frequency axis -> interpolation.
    """
    t, signal, nu_true = synthetic_mzi()
    idx = len(t) // 2
    dut = 1.0 - 0.9 * np.exp(-0.5 * ((np.arange(len(t)) - idx) / 20.0) ** 2)

    nu_rec = _relative_freq_hz(signal, FSR_HZ, FS_HZ)
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


# --- step 3: gas-cell dips ---------------------------------------------------
#
# One decimated sample per pm of sweep, matching gas_anchors' default, so
# "samples" and "pm" are interchangeable throughout these tests.

LINE_FWHM_PM = 16.0
GAS_CFG = {
    "laser": {"start_nm": 1500.0, "stop_nm": 1600.0, "speed_nm_s": 10.0},
    "gascell": {
        "species": {
            "HCN": {"linewidth_pm": LINE_FWHM_PM,
                    "lines_nm": {"P1": 1534.52984, "P2": 1535.23629, "P3": 1535.95333}},
            "CO": {"linewidth_pm": LINE_FWHM_PM, "lines_nm": {"R4": 1570.3314}},
        }
    },
}


def synthetic_gascell(centres, depths=None, noise=0.002, seed=0):
    """Transmission trace with Gaussian absorption lines, on a realistic baseline.

    The baseline matters more than the noise here. A real gas-cell trace rides
    on the laser's output-power ripple, and that ripple is only ~30x below the
    weakest line on the 20260813 record -- so a detector that only thresholds,
    without caring how wide a feature is, finds the ripple too.
    """
    x = np.arange(20_000.0)
    sigma = LINE_FWHM_PM / 2.355
    y = 1.0 + 0.15 * np.sin(2.0 * np.pi * x / 5000.0)  # source power ripple
    for c, d in zip(centres, depths or [0.5] * len(centres)):
        y -= d * np.exp(-0.5 * ((x - c) / sigma) ** 2)
    return y + noise * np.random.default_rng(seed).standard_normal(len(x))


def test_find_dips_locates_lines_to_a_fraction_of_a_linewidth():
    centres = [2000.0, 2500.5, 9000.0, 15_000.0]
    pos, depth = find_dips(synthetic_gascell(centres), LINE_FWHM_PM)
    assert len(pos) == len(centres), f"found {len(pos)}, expected {len(centres)}"
    assert np.allclose(pos, centres, atol=0.1 * LINE_FWHM_PM), pos - centres
    # Depth is prominence, measured against the local baseline, so a line
    # sitting in a trough of the ripple reports the trough as well. It is a
    # diagnostic, not a photometric measurement -- only the position is used.
    assert np.all(depth > 0.45) and np.all(depth < 0.85), depth


def test_find_dips_ignores_features_of_the_wrong_width():
    """Width, not depth, is what separates a line from a glitch or a ripple.

    Both intruders here are deeper than the real line, so a threshold alone
    would take all three.
    """
    y = synthetic_gascell([9000.0], depths=[0.3], noise=0.0)
    y[4000] -= 1.0  # single-sample glitch
    y[13_000:15_000] -= 1.0  # 2 nm dropout, e.g. the laser losing lock
    pos, _ = find_dips(y, LINE_FWHM_PM)
    assert np.allclose(pos, [9000.0], atol=1.0), pos


def test_find_dips_survives_a_saturated_line():
    """A strong line bottoms out flat; its centre is still the middle of it."""
    y = np.clip(synthetic_gascell([9000.0], depths=[3.0], noise=0.0), 0.0, None)
    pos, _ = find_dips(y, LINE_FWHM_PM)
    assert np.allclose(pos, [9000.0], atol=1.0), pos


def test_assign_lines_identifies_dips_within_the_dial_error():
    """A 20 pm systematic offset is what the real laser dial does; it must not
    stop the lines being identified, since they are 400+ pm apart."""
    truth = [("HCN", "P1", 1534.52984), ("HCN", "P3", 1535.95333), ("CO", "R4", 1570.3314)]
    measured = np.array([nm + 0.020 for _, _, nm in truth])
    assert assign_lines(measured, GAS_CFG) == truth


def test_assign_lines_rejects_a_dip_that_is_no_line():
    """A spurious dip 60 pm off a line is inside tol_nm, but it is nowhere near
    the residual cluster the genuine matches form, which is what catches it."""
    measured = np.array([1534.52984, 1535.23629, 1535.95333 + 0.060, 1550.0])
    matched = assign_lines(measured, GAS_CFG)
    assert [m[1] if m else None for m in matched] == ["P1", "P2", None, None]


def test_assign_lines_gives_one_line_to_one_dip():
    """Two dips near the same line would otherwise anchor one frequency twice."""
    matched = assign_lines(np.array([1534.51984, 1534.53084]), GAS_CFG)
    assert [m[1] if m else None for m in matched] == [None, "P1"]


def test_assign_lines_only_offers_lines_inside_the_sweep():
    narrow = {**GAS_CFG, "laser": {**GAS_CFG["laser"], "stop_nm": 1540.0}}
    assert assign_lines(np.array([1570.3314]), narrow) == [None]


# --- step 4: nu(phi) fit ------------------------------------------------------

FIT_CFG = {
    "gascell": {"species": {"H12C14N": {"accuracy_pm": 0.7}, "12CO": {"accuracy_pm": 0.3}}}
}


def _anchors(phi, nu, species="12CO", label="R1"):
    return [
        {"species": species, "label": label, "phi_rad": p, "nu_hz": n}
        for p, n in zip(phi, nu)
    ]


def test_fit_nu_phi_recovers_a_known_quadratic():
    """Anchors sampled exactly off a quadratic must refit to the same law.

    phi here runs to 1e6 rad, the real regime (a 100 nm sweep is ~1e7 rad of
    MZI phase) -- a raw numpy.polyfit/polyval at this scale is badly
    conditioned, which is why fit_nu_phi uses numpy.polynomial.Polynomial
    (domain-scaled) instead. This test would silently pass with a huge but
    unnoticed error if that scaling were ever dropped, so it checks recovered
    values, not just that *some* fit converged.
    """
    true_coeffs = [-1.0e3, 5.0e13, 1.936e14]  # Hz/rad^2, Hz/rad, Hz
    phi = np.linspace(0.0, 2.0e6, 60)
    nu = np.polyval(true_coeffs, phi)
    fit = fit_nu_phi(_anchors(phi, nu), FIT_CFG)
    assert fit["n_anchors"] == 60
    assert fit["dropped_high_j"] == 0
    # Float64 leaves a few 1e4 Hz of noise on a nu ~2e14 Hz scale (relative
    # 1e-10) even with domain scaling -- four orders below the 90 MHz the
    # catalogue itself limits real fits to, so it is exact for any practical
    # purpose. The real assertion is "orders of magnitude better than the
    # catalogue floor", not "zero".
    assert fit["rms_hz"] < 1.0e5
    probe = np.linspace(phi.min(), phi.max(), 200)
    assert np.allclose(nu_hz(probe, fit), np.polyval(true_coeffs, probe), atol=1.0e5)


def test_fit_nu_phi_drops_high_j_h12c14n_but_keeps_low_j():
    anchors = _anchors([0.0, 1.0], [1.0e14, 1.1e14], species="H12C14N", label="P25")
    anchors += _anchors([2.0], [1.2e14], species="H12C14N", label="R5")
    anchors += _anchors([3.0, 4.0], [1.3e14, 1.4e14], species="12CO", label="R1")
    fit = fit_nu_phi(anchors, FIT_CFG, degree=1, max_j=20)
    assert fit["n_anchors"] == 3  # the two P25 anchors (|J|=25) are dropped
    assert fit["dropped_high_j"] == 2


def test_fit_nu_phi_raises_when_too_few_anchors_survive():
    anchors = _anchors([0.0, 1.0, 2.0], [1.0e14, 1.1e14, 1.2e14])
    with pytest.raises(ValueError):
        fit_nu_phi(anchors, FIT_CFG, degree=5)


# --- step 5: reading a fit back out of the file --------------------------------


def test_read_ruler_fit_round_trips_fit_ruler_output(tmp_path):
    """fit_ruler() writes these attrs; read_ruler_fit() must read the same ones."""
    h5_path = tmp_path / "scan.h5"
    coeffs = [1.0, 2.0, 3.0]
    domain = [0.0, 10.0]
    with h5py.File(h5_path, "w") as f:
        grp = f.require_group("ruler/sweep_0")
        grp.attrs["coeffs_hz"] = coeffs
        grp.attrs["domain_rad"] = domain
        grp.attrs["degree"] = 2
        grp.attrs["rms_hz"] = 5.84e7
        grp.attrs["n_anchors"] = 91
        grp.attrs["dropped_high_j"] = 22

    fit = read_ruler_fit(h5_path, sweep=0)
    assert fit["coeffs"] == coeffs
    assert fit["domain"] == domain
    assert fit["n_anchors"] == 91
    # domain midpoint maps to the Polynomial's own origin, where only the
    # constant term survives -- a cheap check that the coeffs/domain pairing
    # (not just their values individually) came back correctly.
    assert abs(nu_hz(5.0, fit) - coeffs[0]) < 1.0e-9


def test_read_ruler_fit_raises_for_missing_sweep(tmp_path):
    h5_path = tmp_path / "scan.h5"
    with h5py.File(h5_path, "w") as f:
        f.require_group("ruler/sweep_0")
    with pytest.raises(ValueError):
        read_ruler_fit(h5_path, sweep=1)


# --- step 6: resonance fitting --------------------------------------------------
#
# FSR and FWHM chosen to match the 20260813 ring (~100 GHz FSR, ~200 MHz
# linewidth, Q ~ 1e6) -- the same reason test_ruler.py's MZI tests use a
# realistic fringe count rather than a handful of cycles.

FSR_TEST_HZ = 1.0e11
FWHM_TEST_HZ = 2.0e8


def synthetic_resonances(n=5, amp=0.12, baseline=-58.1, noise=2.0e-4, seed=0):
    """A DUT-like spectrum: n evenly spaced Lorentzians on a flat baseline."""
    f0s = 190.0e12 + FSR_TEST_HZ * np.arange(n)
    freq_hz = np.arange(f0s[0] - 5 * FWHM_TEST_HZ, f0s[-1] + 5 * FWHM_TEST_HZ, 2.0e6)
    y = np.full_like(freq_hz, baseline)
    for f0 in f0s:
        y = y + amp / (1.0 + ((freq_hz - f0) / (0.5 * FWHM_TEST_HZ)) ** 2)
    if noise:
        y = y + noise * np.random.default_rng(seed).standard_normal(len(freq_hz))
    return freq_hz, y, f0s


def test_fit_resonances_recovers_known_peaks():
    freq_hz, dut, f0s = synthetic_resonances()
    fit = fit_resonances(freq_hz, dut, expected_fwhm_hz=FWHM_TEST_HZ, snr=20.0)
    assert len(fit) == len(f0s)
    recovered = sorted(r["f0_hz"] for r in fit)
    assert np.allclose(recovered, f0s, atol=1.0e6)
    assert all(abs(r["fwhm_hz"] - FWHM_TEST_HZ) / FWHM_TEST_HZ < 0.05 for r in fit)
    assert all(r["amp_hz"] > 0 for r in fit)


def test_fit_resonances_recovers_dips_too():
    """A resonance can be a dip depending on which DUT port is recorded --
    unlike find_dips, fit_resonances must not assume a direction."""
    freq_hz, dut, f0s = synthetic_resonances(amp=-0.12)
    fit = fit_resonances(freq_hz, dut, expected_fwhm_hz=FWHM_TEST_HZ, snr=20.0)
    assert len(fit) == len(f0s)
    assert all(r["amp_hz"] < 0 for r in fit)


def test_fit_resonances_returns_empty_on_pure_noise():
    rng = np.random.default_rng(0)
    freq_hz = np.arange(190.0e12, 191.0e12, 2.0e6)
    dut = -58.1 + 2.0e-4 * rng.standard_normal(len(freq_hz))
    assert fit_resonances(freq_hz, dut, expected_fwhm_hz=FWHM_TEST_HZ, snr=20.0) == []


# --- step 2b: measuring the fringe rate, and deriving the band from it -------
#
# A second chirp, sampled and swept like the real instrument rather than like
# the fast synthetic one above: 2.5 MS/s, 325-433 kHz over a second. The
# numbers matter here in a way they do not elsewhere in this file, because the
# whole point of the auto band is that it lands somewhere specific.

REAL_FS_HZ = 2.5e6
REAL_DURATION_S = 1.0
REAL_F0_HZ, REAL_F1_HZ = 325.0e3, 433.0e3
REAL_KNOBS = {
    "nperseg": 32768,
    "search_hz": [5.0e4, 1.2e6],
    "margin": 0.25,
    "pctile": 1.0,
}
# The fast chirp at the top of this file spans 7.3-17.1 kHz, so it needs its own
# search window -- and, being a 2.3x span, it is also the case where 2f cannot
# be excluded.
WIDE_KNOBS = {**REAL_KNOBS, "nperseg": 4096, "search_hz": [2.0e3, 3.0e4]}


def realistic_chirp(harmonic: float = 0.0, drift: float = 0.0, seed: int = 0):
    """(t, trace, true_fringe_rate_hz) for a linear chirp at instrument scale."""
    t = np.arange(0.0, REAL_DURATION_S, 1.0 / REAL_FS_HZ)
    rate_hz_s = (REAL_F1_HZ - REAL_F0_HZ) / REAL_DURATION_S
    phase = 2.0 * np.pi * (REAL_F0_HZ * t + 0.5 * rate_hz_s * t**2)
    trace = np.cos(phase)
    if harmonic:
        trace = trace + harmonic * np.cos(2.0 * phase)
    if drift:
        trace = trace + drift * np.sin(2.0 * np.pi * 3.0 * t)
    trace = trace + 0.01 * np.random.default_rng(seed).standard_normal(len(t))
    return t, trace, REAL_F0_HZ + rate_hz_s * t


def test_fringe_rate_tracks_a_chirp():
    t, trace, true_rate = realistic_chirp()
    t_seg, rate = ruler.fringe_rate_hz(
        trace, REAL_FS_HZ, REAL_KNOBS["nperseg"], tuple(REAL_KNOBS["search_hz"])
    )
    expected = np.interp(t_seg, t, true_rate)
    assert np.abs(rate - expected).max() < 2.0e3


def test_fringe_rate_ignores_drift_and_the_second_harmonic():
    """The measurement has to survive exactly what it is meant to diagnose.

    Drift near DC is stronger than the fringe on a real trace, and detector
    nonlinearity puts a tone at 2f. Neither may capture the ridge, or the band
    derived from it would be nonsense on precisely the records that need it.
    """
    _, clean, _ = realistic_chirp()
    _, dirty, _ = realistic_chirp(harmonic=0.3, drift=5.0)
    args = (REAL_FS_HZ, REAL_KNOBS["nperseg"], tuple(REAL_KNOBS["search_hz"]))
    _, rate_clean = ruler.fringe_rate_hz(clean, *args)
    _, rate_dirty = ruler.fringe_rate_hz(dirty, *args)
    assert np.array_equal(rate_clean, rate_dirty)


def test_auto_band_brackets_the_fringes_and_excludes_the_harmonic():
    _, trace, _ = realistic_chirp(harmonic=0.3, drift=5.0)
    low, high = ruler.resolve_band_hz(trace, REAL_FS_HZ, REAL_KNOBS)

    assert low < REAL_F0_HZ and high > REAL_F1_HZ, "band must cover the fringes"
    assert high < 2.0 * REAL_F0_HZ, "band must exclude the slowest fringe's 2f"
    # The band this record has been analysed with until now runs to 700 kHz,
    # which is above 2*325 kHz -- the case this whole mechanism exists to stop.
    assert high < 700.0e3


def test_auto_band_keeps_the_fringes_when_the_harmonic_cannot_be_excluded(caplog):
    """A chirp wider than ~1.7:1 overlaps its own second harmonic.

    No band can hold the fundamental and drop 2f, so covering the fringes has
    to win -- silently clipping the top of the sweep would be far worse than
    passing a harmonic.
    """
    _, signal, _ = synthetic_mzi()
    with caplog.at_level(logging.WARNING):
        low, high = ruler.resolve_band_hz(signal, FS_HZ, WIDE_KNOBS)
    assert low < 7.3e3 and high > 17.1e3
    assert "second harmonic" in caplog.text


def test_auto_band_survives_the_drift_that_breaks_the_raw_phase():
    """End to end: measure, derive, filter, and get a monotonic phase out."""
    t, signal, nu_true = synthetic_mzi()
    drifted = signal + 3.0 * np.sin(2.0 * np.pi * 20.0 * t)
    band = ruler.resolve_band_hz(drifted, FS_HZ, WIDE_KNOBS)
    phi = phase_rad(drifted, FS_HZ, band)
    assert np.all(np.diff(_interior(phi)) > 0)
    err = _spacing_error(phi, 2.0 * np.pi * nu_true / FSR_HZ)
    assert err.max() < 0.05, f"worst phase error {err.max():.4f} rad"


def test_band_for_accepts_the_three_config_forms():
    _, trace, _ = realistic_chirp()
    base = {"process": {"fringe_rate": REAL_KNOBS}}

    explicit = {"process": {**base["process"], "bandpass_hz": [2.0e5, 7.0e5]}}
    assert ruler.band_for(explicit, trace, REAL_FS_HZ) == (2.0e5, 7.0e5)

    nofilter = {"process": {**base["process"], "bandpass_hz": None}}
    assert ruler.band_for(nofilter, trace, REAL_FS_HZ) is None

    auto = {"process": {**base["process"], "bandpass_hz": "auto"}}
    band = ruler.band_for(auto, trace, REAL_FS_HZ)
    assert band[0] < REAL_F0_HZ < REAL_F1_HZ < band[1]
    # Resolved back into the config, so whatever reports what it ran with --
    # plot titles, fit_ruler's params_yaml -- writes numbers, not "auto".
    assert auto["process"]["bandpass_hz"] == list(band)


def test_band_for_lets_a_caller_override_auto():
    _, trace, _ = realistic_chirp()
    cfg = {"process": {"bandpass_hz": "auto", "fringe_rate": REAL_KNOBS}}
    assert ruler.band_for(cfg, trace, REAL_FS_HZ, (1.0e5, 6.0e5)) == (1.0e5, 6.0e5)


def test_band_for_rejects_a_word_that_is_not_auto():
    cfg = {"process": {"bandpass_hz": "wide"}}
    with pytest.raises(ValueError, match="auto"):
        ruler.band_for(cfg, np.zeros(1024), REAL_FS_HZ)


def test_divide_baseline_recovers_dips_a_wandering_baseline_hides():
    """Dips on a baseline that wanders more than they are deep, at negative volts
    like the real scope trace: detection on the raw trace misses the ones where
    the baseline sits above the median; after divide_baseline all are found."""
    from spectro.ruler import divide_baseline

    rng = np.random.default_rng(1)
    f = np.arange(0.0, 1.0e12, 5.0e6)                       # 1 THz at 5 MHz
    f0s = np.arange(20e9, 1.0e12 - 20e9, 37e9)
    dips = sum(0.5 / (1 + ((f - c) / 1e8) ** 2) for c in f0s)  # 200 MHz FWHM
    y = -57.0 + 1.0 * np.sin(2 * np.pi * f / 300e9) - dips + 0.02 * rng.standard_normal(len(f))

    def n_found(trace):
        fit = fit_resonances(f, trace, expected_fwhm_hz=2e8, snr=10.0)
        got = np.array([r["f0_hz"] for r in fit])
        return sum(np.min(np.abs(got - c)) < 1e8 for c in f0s) if len(got) else 0

    clean = divide_baseline(f, y, 20e9, 90.0)
    assert n_found(clean) == len(f0s)
    assert n_found(y) < 0.8 * len(f0s)                      # the failure being fixed
    assert np.all([r["amp_hz"] < 0 for r in fit_resonances(f, clean, 2e8, 10.0)])


def test_block_mean_averages_and_drops_the_tail():
    from spectro.ruler import block_mean

    assert np.allclose(block_mean(np.arange(10.0), 3), [1.0, 4.0, 7.0])
    x = np.arange(4.0)
    assert block_mean(x, 1) is x
