"""Fibre-MZI frequency ruler.

The MZI trace is I(t) = A + B*cos(phi(t)) with phi(t) = 2*pi*(nu(t) - nu0)/FSR.
Unwrapping the analytic-signal phase therefore counts fringes, and each fringe
is one free spectral range of optical frequency -- independent of how nonlinear
the laser's sweep actually was. That is the whole point of the auxiliary
interferometer.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import butter, filtfilt, hilbert

C_M_S = 299_792_458.0

log = logging.getLogger(__name__)


def phase_rad(
    mzi: np.ndarray,
    fs_hz: float,
    bandpass_hz: tuple[float, float] | None = None,
) -> np.ndarray:
    """Unwrapped analytic-signal phase of the MZI trace, in radians.

    Args:
        mzi: Raw MZI channel samples.
        fs_hz: Sample rate.
        bandpass_hz: Optional (low, high) band-pass applied before the Hilbert
            transform. Use it when the trace carries slow drift -- the analytic
            signal misbehaves badly on a signal with a wandering baseline.

    Returns:
        Monotonically increasing phase. Note the Hilbert transform rings at both
        ends of the record; trim a few per cent before trusting the edges.
    """
    x = np.asarray(mzi, dtype=float)
    x = x - x.mean()

    if bandpass_hz is not None:
        low, high = bandpass_hz
        nyq = 0.5 * fs_hz
        b, a = butter(4, [low / nyq, min(high / nyq, 0.99)], btype="band")
        x = filtfilt(b, a, x)

    # Zero-pad to a fast FFT length: an awkward record length (a large prime,
    # say) turns the transform of a 50 M-point trace pathologically slow.
    n = len(x)
    analytic = hilbert(x, N=next_fast_len(n))[:n]
    return np.unwrap(np.angle(analytic))


def relative_freq_hz(
    mzi: np.ndarray,
    fsr_hz: float,
    fs_hz: float,
    bandpass_hz: tuple[float, float] | None = None,
) -> np.ndarray:
    """Optical frequency traversed since the start of the record, in Hz.

    Always non-negative and increasing: this is distance travelled along the
    frequency axis, not a signed offset. Use frequency_axis_hz() to attach a
    direction and a starting point.
    """
    phi = phase_rad(mzi, fs_hz, bandpass_hz)
    # The origin is deliberately not treated as accurate. phi[0] sits in the
    # analytic signal's edge transient, which puts a few MHz of constant offset
    # on the whole axis -- irrelevant next to the laser's own ~GHz absolute
    # wavelength error, and removed by the gas cell when that step is built.
    # What this function does guarantee is the *spacing*, good to ~100 Hz.
    # ponytail: FSR treated as constant across the sweep. Fibre dispersion makes
    # it drift by ~1e-4 over the C band; add a polynomial FSR(nu) correction here
    # if you need sub-kHz accuracy at the band edges.
    return (phi - phi[0]) / (2.0 * np.pi) * float(fsr_hz)


def frequency_axis_hz(
    mzi: np.ndarray,
    fsr_hz: float,
    fs_hz: float,
    start_nm: float,
    stop_nm: float,
    bandpass_hz: tuple[float, float] | None = None,
) -> np.ndarray:
    """Absolute-ish optical frequency for each sample, in Hz.

    Anchored to c/start_nm, so the axis is accurate in its *spacing* but carries
    the laser's absolute wavelength error as a common offset. Anchoring that
    offset is the gas cell's job -- recorded, not yet used.
    """
    nu_start = C_M_S / (start_nm * 1e-9)
    # Sweeping to longer wavelength means falling optical frequency.
    direction = -1.0 if stop_nm > start_nm else +1.0
    return nu_start + direction * relative_freq_hz(mzi, fsr_hz, fs_hz, bandpass_hz)


def resample_uniform(
    nu_hz: np.ndarray,
    y: np.ndarray,
    n_points: int | None = None,
    trim_frac: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate y from the (non-uniform) frequency axis onto a uniform grid.

    Args:
        nu_hz: Per-sample optical frequency from frequency_axis_hz().
        y: Channel to resample, same length as nu_hz.
        n_points: Output grid size; defaults to the input length.
        trim_frac: Fraction discarded from each end before interpolating, to
            drop the Hilbert transform's edge artefacts.

    Returns:
        (uniform frequency grid, y interpolated onto it), both ascending.
    """
    nu = np.asarray(nu_hz, dtype=float)
    y = np.asarray(y, dtype=float)
    if nu.shape != y.shape:
        raise ValueError(f"length mismatch: nu {nu.shape} vs y {y.shape}")

    k = int(len(nu) * trim_frac)
    if k:
        nu, y = nu[k:-k], y[k:-k]

    if nu[-1] < nu[0]:  # descending frequency; np.interp needs ascending
        nu, y = nu[::-1], y[::-1]

    grid = np.linspace(nu[0], nu[-1], n_points or len(nu))
    return grid, np.interp(grid, nu, y)


def process_file(h5_path: str | Path, n_points: int | None = None) -> Path:
    """Build the frequency axis from the MZI channel and resample the DUT.

    Reads the config back out of the file's attributes, so a scan is
    reprocessable without its original YAML. Results are written into a /proc
    group in the same file; /raw is untouched.
    """
    import h5py
    import yaml

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r+") as f:
        cfg = yaml.safe_load(f.attrs["config_yaml"])
        fs_hz = 1.0 / float(f.attrs["dt_s"])
        roles = cfg["scope"]["channels"]
        bandpass = cfg["process"].get("bandpass_hz")

        mzi = f[f"raw/ch{roles['mzi']['ch']}"][:]
        nu = frequency_axis_hz(
            mzi,
            fsr_hz=float(cfg["process"]["mzi_fsr_hz"]),
            fs_hz=fs_hz,
            start_nm=cfg["laser"]["start_nm"],
            stop_nm=cfg["laser"]["stop_nm"],
            bandpass_hz=bandpass,
        )
        span_thz = abs(nu[-1] - nu[0]) / 1e12
        log.info("ruler spans %.3f THz over %d samples", span_thz, len(nu))

        if "proc" in f:
            del f["proc"]
        proc = f.create_group("proc")
        grid = None
        # The gas cell rides along on the same axis; it is the anchor for
        # absolute frequency once that step gets built.
        for role in ("dut", "gascell"):
            if role not in roles:
                continue
            y = f[f"raw/ch{roles[role]['ch']}"][:]
            grid, y_uniform = resample_uniform(nu, y, n_points=n_points)
            proc.create_dataset(role, data=y_uniform.astype(np.float32), compression="lzf")
        if grid is not None:
            proc.create_dataset("freq_hz", data=grid, compression="lzf")
            proc.attrs["span_thz"] = span_thz

    log.info("processed %s", h5_path)
    return h5_path
