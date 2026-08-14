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
