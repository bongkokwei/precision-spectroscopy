"""Fibre-MZI frequency ruler.

The MZI trace is I(t) = A + B*cos(phi(t)), and phi advances by 2*pi for every
free spectral range of optical frequency the laser covers. Unwrapping the
analytic-signal phase therefore gives a monotonic coordinate that grows with
optical frequency however nonlinear the sweep was.

phi is deliberately *not* converted to Hz here. Doing that needs a single FSR,
and fibre dispersion means there is no single FSR across 100 nm. Absolute
frequency comes instead from fitting nu(phi) through gas-cell lines, which
absorbs the dispersion into the fit -- steps 3 to 6.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import butter, hilbert, sosfiltfilt

C_M_S = 299_792_458.0

log = logging.getLogger(__name__)


def phase_rad(
    mzi: np.ndarray,
    fs_hz: float,
    bandpass_hz: tuple[float, float] | None = None,
) -> np.ndarray:
    """Unwrapped analytic-signal phase of the MZI trace, in radians.

    Args:
        mzi: MZI channel samples for one forward sweep.
        fs_hz: Sample rate.
        bandpass_hz: (low, high) band-pass applied before the Hilbert
            transform. Not optional in practice: on the raw trace the analytic
            phase of this data goes non-monotonic, because low-frequency
            intensity drift and broadband noise both leak into the phase.
            Set it wide enough to hold the fringe rate over the whole sweep --
            see process.bandpass_hz in the config.

    Returns:
        Phase in radians, increasing with optical frequency travelled. The
        Hilbert transform rings at both ends of the record; trim a few per cent
        before trusting the edges.
    """
    x = np.asarray(mzi, dtype=float)
    x = x - x.mean()

    if bandpass_hz is not None:
        low, high = bandpass_hz
        nyq = 0.5 * fs_hz
        sos = butter(4, [low / nyq, min(high / nyq, 0.99)], btype="band", output="sos")
        x = sosfiltfilt(sos, x)

    # Zero-pad to a fast FFT length: an awkward record length (a large prime,
    # say) turns the transform of a 25 M-point trace pathologically slow.
    n = len(x)
    analytic = hilbert(x, N=next_fast_len(n))[:n]
    phi = np.unwrap(np.angle(analytic))
    # Compare neighbours rather than np.diff: the boolean array costs 25 MB on a
    # record where the float64 difference would cost 200. Every sample is
    # checked, because a slip is local and a strided check walks straight past it.
    slips = int(np.count_nonzero(phi[1:] <= phi[:-1]))
    if slips:
        log.warning(
            "MZI phase steps backwards at %d of %d samples -- widen or re-centre "
            "process.bandpass_hz, or the frequency axis will fold back on itself",
            slips, n,
        )
    return phi


def sweep_phase(
    h5_path: str | Path,
    sweep: int = 0,
    bandpass_hz: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """MZI phase over one forward sweep of a scan file (steps 1 and 2).

    Args:
        h5_path: Path to scan.h5.
        sweep: Which forward sweep, indexed from the trigger edges.
        bandpass_hz: Overrides the band-pass frozen into the file at
            acquisition time. Needed for records written before that setting
            had a sensible default, and handy for tuning the band on a record
            whose fringe rate you have just measured.

    Returns:
        (t_rel_s, phi_rad, mzi, cfg) for that sweep alone; mzi is returned so a
        caller can check the phase against the fringes it came from. Time is
        re-zeroed to the trigger edge.
    """
    from .plot import load_h5  # ponytail: I/O helper lives with the plotter
    from .sweeps import sweep_slices

    t_s, traces, cfg = load_h5(h5_path, roles=("mzi", "trigger"))
    slices = sweep_slices(t_s, traces.pop("trigger"), cfg)
    if not slices:
        raise ValueError("no complete forward sweep in this record")
    sl = slices[sweep]

    fs_hz = 1.0 / float(t_s[1] - t_s[0])
    mzi = traces["mzi"][sl]
    if bandpass_hz is not None:
        cfg["process"]["bandpass_hz"] = list(bandpass_hz)
    phi = phase_rad(mzi, fs_hz, cfg["process"].get("bandpass_hz"))
    log.info(
        "sweep %d: %.3f Mfringe over %.2f s",
        sweep, (phi[-1] - phi[0]) / (2 * np.pi) / 1e6, len(phi) / fs_hz,
    )
    return t_s[sl] - t_s[sl.start], phi, mzi, cfg


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
