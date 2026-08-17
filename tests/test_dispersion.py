"""Synthetic checks for the D2 fit in scripts/calc_d2.py.

Follows the package's own convention (test_ruler.py): synthetic data with a
known answer, not fixed thresholds on real data.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calc_d2  # noqa: E402


def synthetic_resonances(d1_hz, d2_hz, n=61, amx_mu=None, amx_shift_hz=0.0, seed=0):
    """f(mu) = f0 + d1*mu + (d2/2)*mu^2, optionally with one mode displaced."""
    rng = np.random.default_rng(seed)
    mu_true = np.arange(-(n // 2), n // 2 + 1)
    f0 = calc_d2.C_M_S / 1550e-9
    f = f0 + d1_hz * mu_true + 0.5 * d2_hz * mu_true**2
    f = f + rng.normal(0, 1e5, size=mu_true.shape)  # 100 kHz noise, well under D2's scale
    if amx_mu is not None:
        f = f.copy()
        f[mu_true == amx_mu] += amx_shift_hz
    return mu_true, f


def synthetic_resonance_dicts(d1_hz, d2_hz, n=61, amp_hz=0.15, seed=0):
    """Resonance dicts shaped like ruler.find_resonances()'s output."""
    mu_true, f = synthetic_resonances(d1_hz, d2_hz, n=n, seed=seed)
    rng = np.random.default_rng(seed)
    return [
        {
            "f0_hz": float(fi),
            "amp_hz": float(amp_hz * rng.uniform(0.7, 1.3)),
            "fwhm_hz": 2e8,
            "q": float(fi / 2e8),
        }
        for fi in f
    ]


def test_filter_resonances_drops_low_amplitude_fit_failures():
    resonances = synthetic_resonance_dicts(d1_hz=100e9, d2_hz=760e3)
    # A fit failure landing almost exactly between two real modes -- the
    # scenario that made assign_mode_numbers collide on the real ring data.
    bad = dict(resonances[10])
    bad["f0_hz"] += 0.5 * 100e9
    bad["amp_hz"] = 0.0005  # ~300x smaller than the real resonances
    with_failure = resonances[:11] + [bad] + resonances[11:]

    # Left in, mode assignment collides -- this is the bug being fixed.
    f0_with_bad = np.array([r["f0_hz"] for r in with_failure])
    with pytest.raises(ValueError, match="collide"):
        calc_d2.assign_mode_numbers(f0_with_bad, ref_nm=1550.0)

    filtered = calc_d2.filter_resonances(with_failure)
    assert len(filtered) == len(resonances)
    assert bad not in filtered

    f0_filtered = np.array([r["f0_hz"] for r in filtered])
    mu, _ = calc_d2.assign_mode_numbers(f0_filtered, ref_nm=1550.0)
    assert len(mu) == len(filtered)


def test_filter_resonances_keeps_everything_when_no_outliers():
    resonances = synthetic_resonance_dicts(d1_hz=100e9, d2_hz=760e3)
    filtered = calc_d2.filter_resonances(resonances)
    assert len(filtered) == len(resonances)


def test_assign_mode_numbers_recovers_known_spacing():
    mu_true, f = synthetic_resonances(d1_hz=100e9, d2_hz=760e3)
    mu, ref_index = calc_d2.assign_mode_numbers(f, ref_nm=1550.0)
    assert np.array_equal(mu, mu_true)
    assert mu[ref_index] == 0


def test_fit_dispersion_recovers_d1_d2():
    mu_true, f = synthetic_resonances(d1_hz=100e9, d2_hz=760e3)
    mu, _ = calc_d2.assign_mode_numbers(f, ref_nm=1550.0)
    fit = calc_d2.fit_dispersion(mu, f)
    assert fit["d1_hz"] == pytest.approx(100e9, rel=1e-6)
    assert fit["d2_hz"] == pytest.approx(760e3, rel=1e-3)
    assert fit["kept"].all()


def test_fit_dispersion_excludes_amx():
    mu_true, f = synthetic_resonances(
        d1_hz=100e9, d2_hz=760e3, amx_mu=5, amx_shift_hz=50e6
    )
    mu, _ = calc_d2.assign_mode_numbers(f, ref_nm=1550.0)
    fit = calc_d2.fit_dispersion(mu, f)
    amx = calc_d2.amx_candidates(mu, f, fit)

    assert len(amx) == 1
    assert amx[0]["mu"] == 5
    # D2 recovered from the clean points, not dragged by the displaced one.
    assert fit["d2_hz"] == pytest.approx(760e3, rel=1e-2)


def test_assign_mode_numbers_rejects_colliding_modes():
    f = np.array([193.0e12, 193.1e12, 193.1005e12])  # last two land on the same mode
    with pytest.raises(ValueError, match="collide"):
        calc_d2.assign_mode_numbers(f, ref_nm=1550.0)


def test_fit_dispersion_raises_when_too_few_survive():
    mu = np.array([0.0, 1.0])
    f = np.array([193.0e12, 193.1e12])
    with pytest.raises(ValueError, match="survive"):
        calc_d2.fit_dispersion(mu, f, degree=2)


def test_dispersion_of_matches_the_inline_body_it_replaced(monkeypatch):
    """The extraction has to be faithful, not just importable.

    dispersion_of() is the code main() used to carry inline, lifted out so
    refit_all can call it. Both halves are stubbed here -- the file read and
    find_resonances -- so this checks the arithmetic between them, which is the
    only part that moved.
    """
    d1_hz, d2_hz = 100.0e9, 760.0e3
    resonances = synthetic_resonance_dicts(d1_hz=d1_hz, d2_hz=d2_hz, n=125)

    monkeypatch.setattr(
        calc_d2.plot, "load_h5", lambda *a, **k: (None, None, {"process": {}})
    )
    monkeypatch.setattr(
        calc_d2.ruler, "find_resonances", lambda *a, **k: (None, None, resonances)
    )

    d = calc_d2.dispersion_of("nowhere/scan.h5", sweep=0)

    assert abs(d["d1_hz"] - d1_hz) / d1_hz < 1e-4
    assert abs(d["d2_hz"] - d2_hz) / d2_hz < 0.05
    assert d["n_fit"] == 125
    assert d["n_failed"] == 0
    assert d["n_kept"] <= d["n_fit"]
    # The keys main() and refit_all read off the result.
    for key in ("poly", "kept", "rms_hz", "mu", "f0_hz", "ref_index", "amx"):
        assert key in d, key


def test_dispersion_of_raises_when_no_resonances_were_fit():
    """Better a clear error than a D2 fitted through nothing."""
    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(calc_d2.plot, "load_h5", lambda *a, **k: (None, None, {"process": {}}))
        mp.setattr(calc_d2.ruler, "find_resonances", lambda *a, **k: (None, None, []))
        with _pytest.raises(ValueError, match="no resonances"):
            calc_d2.dispersion_of("nowhere/scan.h5")
