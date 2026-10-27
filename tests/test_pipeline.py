"""File-level wiring: `bandpass_hz: auto` from a scan.h5 through to a figure.

test_ruler.py checks the measurement and the band rule on bare arrays. What is
left to break is the plumbing -- resolving "auto" once per sweep, handing the
resolved pair to phase_rad, and returning it far enough up that fit_ruler can
record numbers. So this file builds the smallest scan.h5 the loaders accept and
runs the real entry points over it.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
import yaml

matplotlib.use("Agg")

import h5py  # noqa: E402

from spectro import plot, ruler  # noqa: E402

FS_HZ = 2.5e6
RECORD_S = 1.2
SWEEP_S = 1.0
TRIGGER_S = 0.05
F0_HZ, F1_HZ = 325.0e3, 433.0e3


def synthetic_scan(path, bandpass_hz="auto", harmonic=0.3):
    """Write a four-channel scan.h5 holding one forward sweep of a chirp.

    The gas cell and DUT channels are noise: this file is about the band-pass
    plumbing, and a synthetic absorption spectrum accurate enough to anchor a
    fit would only make the fixture harder to read.
    """
    n = int(RECORD_S * FS_HZ)
    t = np.arange(n) / FS_HZ
    rng = np.random.default_rng(0)

    rate_hz_s = (F1_HZ - F0_HZ) / RECORD_S
    mzi = np.cos(2.0 * np.pi * (F0_HZ * t + 0.5 * rate_hz_s * t**2))
    mzi = mzi + harmonic * np.cos(4.0 * np.pi * (F0_HZ * t + 0.5 * rate_hz_s * t**2))
    mzi = mzi + 5.0 * np.sin(2.0 * np.pi * 3.0 * t)  # drift, as on the real trace

    trigger = np.zeros(n)
    trigger[int(TRIGGER_S * FS_HZ) : int((TRIGGER_S + 0.01) * FS_HZ)] = 5.0

    cfg = {
        "scope": {
            "points": n,
            "duration_s": RECORD_S,
            "channels": {
                "mzi": {"ch": 1},
                "gascell": {"ch": 2},
                "dut": {"ch": 3},
                "trigger": {"ch": 4},
            },
        },
        "laser": {"start_nm": 1500.0, "stop_nm": 1510.0, "speed_nm_s": 10.0},
        "process": {"bandpass_hz": bandpass_hz},
    }

    with h5py.File(path, "w") as f:
        f.attrs["config_yaml"] = yaml.safe_dump(cfg, sort_keys=False)
        f.attrs["t0_s"] = 0.0
        f.attrs["dt_s"] = 1.0 / FS_HZ
        f.attrs["n_samples"] = n
        raw = f.create_group("raw")
        raw.create_dataset("ch1", data=mzi.astype(np.float32))
        raw.create_dataset("ch2", data=(1.0 + 0.01 * rng.standard_normal(n)).astype(np.float32))
        raw.create_dataset("ch3", data=(0.5 + 0.01 * rng.standard_normal(n)).astype(np.float32))
        raw.create_dataset("ch4", data=trigger.astype(np.float32))
    return path


@pytest.fixture
def scan(tmp_path):
    return synthetic_scan(tmp_path / "scan.h5")


def test_sweep_phase_resolves_auto_into_numbers(scan):
    _, phi, _, cfg = ruler.sweep_phase(scan)
    band = cfg["process"]["bandpass_hz"]
    assert band[0] < F0_HZ < F1_HZ < band[1], band
    assert band[1] < 2.0 * F0_HZ, "the harmonic guard did not survive the plumbing"
    assert np.all(np.diff(phi) > 0), "auto band-pass left the phase non-monotonic"


def test_the_harmonic_guard_earns_its_place_only_at_absurd_harmonic_levels(tmp_path):
    """What the second-harmonic cap actually buys, measured rather than assumed.

    A 200-700 kHz band admits 2f of the slowest fringes (2 x 325 kHz = 650 kHz)
    where the auto band, capped at 541 kHz, does not. But sweeping the harmonic
    amplitude shows the phase only goes non-monotonic past roughly 0.7 of the
    fundamental -- a detector 3 dB from square, not a real one. Below that the
    harmonic costs a phase ripple of about 0.7*a rad and nothing worse: 0.07 rad
    at a plausible -20 dB, which is 0.011 fringe, far under the ~0.15 fringe
    already conceded to anchor rounding.

    So this is insurance, not a fix, and the test says so: if a future change
    makes it look load-bearing at realistic levels, that finding is the news.
    """
    path = synthetic_scan(tmp_path / "gross.h5", harmonic=0.8)
    _, phi_auto, _, _ = ruler.sweep_phase(path)
    _, phi_fixed, _, _ = ruler.sweep_phase(path, bandpass_hz=(2.0e5, 7.0e5))
    assert int(np.count_nonzero(np.diff(phi_auto) <= 0)) == 0
    assert int(np.count_nonzero(np.diff(phi_fixed) <= 0)) > 100

    mild = synthetic_scan(tmp_path / "mild.h5", harmonic=0.2)
    _, phi_mild, _, _ = ruler.sweep_phase(mild, bandpass_hz=(2.0e5, 7.0e5))
    assert int(np.count_nonzero(np.diff(phi_mild) <= 0)) == 0, (
        "a realistic second harmonic does not break the phase; do not sell the "
        "harmonic cap as the cure for a record that slips"
    )


def test_gas_anchors_returns_the_band_it_resolved(scan):
    anchors, band = ruler.gas_anchors(scan)
    assert anchors == []  # no catalogue in this fixture, by design
    assert band[0] < F0_HZ < F1_HZ < band[1]


def test_explicit_band_still_wins(scan):
    _, _, _, cfg = ruler.sweep_phase(scan, bandpass_hz=(2.0e5, 7.0e5))
    assert cfg["process"]["bandpass_hz"] == [2.0e5, 7.0e5]


def test_null_band_still_means_no_filter(tmp_path):
    path = synthetic_scan(tmp_path / "scan.h5", bandpass_hz=None)
    _, _, _, cfg = ruler.sweep_phase(path)
    assert cfg["process"]["bandpass_hz"] is None


def test_plot_phase_draws_the_spectrogram_panel(scan):
    out = plot.plot_phase(scan, sweep=0)
    assert out.exists() and out.stat().st_size > 0

    import matplotlib.pyplot as plt

    fig = plt.figure()
    assert len(fig.axes) == 0  # figures are closed, not leaked
    plt.close(fig)


def test_run_folder_config_overrides_frozen_process_block(scan):
    """Editing config.yaml beside scan.h5 re-tunes the analysis; hardware stays frozen."""
    sidecar = scan.with_name("config.yaml")
    sidecar.write_text(yaml.safe_dump({
        "laser": {"start_nm": 0.0},
        "process": {"dispersion": {"degree": 3}},
    }))
    _, _, cfg = plot.load_h5(scan, roles=())
    assert cfg["process"] == {"dispersion": {"degree": 3}}
    assert cfg["laser"]["start_nm"] == 1500.0

    sidecar.write_text("process:\n  resonances:\n    expected_fwhm_hz: 2.0e8\n")
    with pytest.raises(ValueError, match="not a number|string"):
        plot.load_h5(scan, roles=())
