"""Fibre-MZI frequency ruler.

The MZI trace is I(t) = A + B*cos(phi(t)), and phi advances by 2*pi for every
free spectral range of optical frequency the laser covers. Unwrapping the
analytic-signal phase therefore gives a monotonic coordinate that grows with
optical frequency however nonlinear the sweep was.

phi is deliberately *not* converted to Hz here. Doing that needs a single FSR,
and fibre dispersion means there is no single FSR across 100 nm. Absolute
frequency comes instead from fitting nu(phi) through gas-cell lines, which
absorbs the dispersion into the fit. gas_anchors() supplies the (phi, nu) pairs
that fit is made through; the fit itself is step 4 and is not written yet.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.fft import next_fast_len
from scipy.optimize import curve_fit
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt, spectrogram

from . import config

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
        band = (
            "no band-pass"
            if bandpass_hz is None
            else f"band-pass {bandpass_hz[0] / 1e3:.0f}-{bandpass_hz[1] / 1e3:.0f} kHz"
        )
        log.warning(
            "MZI phase steps backwards at %d of %d samples (%s) -- the frequency "
            "axis folds back on itself wherever it does. Check the spectrogram "
            "panel of `spectro plot --phase`: slips only at the record ends are "
            "the Hilbert transform ringing and are trimmed downstream, slips in "
            "the middle are not",
            slips, n, band,
        )
    return phi


def mzi_spectrogram(
    mzi: np.ndarray,
    fs_hz: float,
    nperseg: int = 32768,
    search_hz: tuple[float, float] = (5.0e4, 1.2e6),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spectrogram of the raw MZI trace, restricted to the fringe search window.

    Deliberately takes the trace *before* any band-pass. Measuring the fringe
    rate from the filtered phase is circular: whatever the filter left is what
    you measure, so a band that excludes the true fringe rate still reports a
    rate inside itself. This is the only view that can catch that.

    Args:
        mzi: MZI channel samples for one forward sweep.
        fs_hz: Sample rate.
        nperseg: Samples per segment. The fringe rate chirps, so this trades
            frequency resolution fs_hz/nperseg against the smear a chirp of
            rate r puts across one segment, r*nperseg/fs_hz. At 2.5 MS/s and
            32768 samples that is 76 Hz against ~140 Hz, both negligible
            beside a ~100 kHz excursion.
        search_hz: (low, high) window to look for the fringe in. The low edge
            is not optional: intensity drift near DC is far stronger than the
            fringe, and would win the peak in every segment.

    Returns:
        (f_hz, t_s, S) with f_hz already restricted to search_hz and t_s the
        segment centres. Measured peak on a 25 M-sample sweep is ~300 MB over
        and above the trace, most of it scipy's intermediate complex STFT
        rather than the result, which is ~50 MB.
    """
    # Left at the trace's own dtype -- the scope digitises to 12 bits and the
    # file stores float32, so upcasting a 25 M-sample sweep would cost 200 MB to
    # tell us nothing. detrend does the mean removal per segment, which also
    # suppresses drift better than subtracting one mean from the whole record.
    x = np.asarray(mzi)
    if x.dtype.kind != "f":
        x = x.astype(np.float32)
    nperseg = min(int(nperseg), len(x))
    f_hz, t_s, S = spectrogram(
        x, fs=fs_hz, nperseg=nperseg, noverlap=0, detrend="constant"
    )

    low, high = search_hz
    keep = (f_hz >= low) & (f_hz <= min(high, 0.5 * fs_hz))
    if not keep.any():
        raise ValueError(
            f"no spectrogram bin inside search_hz={search_hz} at fs={fs_hz:.3g} Hz"
        )
    return f_hz[keep], t_s, S[keep]


def fringe_rate_hz(
    mzi: np.ndarray,
    fs_hz: float,
    nperseg: int = 32768,
    search_hz: tuple[float, float] = (5.0e4, 1.2e6),
) -> tuple[np.ndarray, np.ndarray]:
    """Instantaneous fringe rate over one sweep, from the unfiltered MZI trace.

    The rate is the strongest bin in each segment, which is the fundamental as
    long as the fringe is the dominant tone -- true by a wide margin on real
    records, where the second harmonic sits tens of dB down.

    Args:
        mzi: MZI channel samples for one forward sweep.
        fs_hz: Sample rate.
        nperseg: Samples per segment, as in mzi_spectrogram().
        search_hz: Where to look for the fringe, as in mzi_spectrogram().

    Returns:
        (t_s, rate_hz), one point per segment, times at segment centres
        relative to the start of the trace passed in.
    """
    f_hz, t_s, S = mzi_spectrogram(mzi, fs_hz, nperseg, search_hz)
    return t_s, f_hz[S.argmax(axis=0)]


def resolve_band_hz(mzi: np.ndarray, fs_hz: float, knobs: dict) -> tuple[float, float]:
    """Band-pass derived from the measured fringe rate (`bandpass_hz: auto`).

    The band has to bracket the fringe rate over the whole sweep, and the
    sweep chirps, so the only honest way to set it is to measure the chirp.
    Three things bound it from above, and the tightest wins:

    * the measured rate plus a margin;
    * the second harmonic of the *slowest* fringe, 2*f_min. Detector and
      transimpedance nonlinearity put a tone there, and a component at 2*phi
      riding on phi is exactly what makes the unwrapped phase non-monotonic.
      Backed off by 15% so the filter's skirt does some work too;
    * Nyquist.

    The harmonic bound is dropped, with a warning, when the chirp spans more
    than about 1.7:1 -- past that 2*f_min falls inside the fundamental's own
    range and no band can hold one without the other. Covering the fringes
    always wins over excluding the harmonic.

    Args:
        mzi: MZI channel samples for one forward sweep.
        fs_hz: Sample rate.
        knobs: process.fringe_rate from the config -- nperseg, search_hz,
            margin, pctile.

    Returns:
        (low_hz, high_hz) to hand to phase_rad().
    """
    _, rate = fringe_rate_hz(
        mzi, fs_hz, knobs["nperseg"], tuple(knobs["search_hz"])
    )
    # Percentiles, not min/max: one segment latching onto noise where the fringe
    # briefly fades would otherwise set an edge of the band on its own.
    pct = float(knobs["pctile"])
    r_lo, r_hi = (float(v) for v in np.percentile(rate, [pct, 100.0 - pct]))
    margin = float(knobs["margin"])

    low = (1.0 - margin) * r_lo
    high = (1.0 + margin) * r_hi

    harmonic_cap = 0.85 * 2.0 * r_lo
    if harmonic_cap > r_hi:
        high = min(high, harmonic_cap)
    else:
        log.warning(
            "fringe rate spans %.2fx (%.0f-%.0f kHz), so its own second harmonic "
            "overlaps it -- the band cannot exclude 2f. Expect phase slips if the "
            "detector is being driven nonlinearly",
            r_hi / r_lo, r_lo / 1e3, r_hi / 1e3,
        )

    nyquist_cap = 0.95 * 0.5 * fs_hz
    if high > nyquist_cap:
        log.warning(
            "auto band-pass clipped to %.0f kHz by Nyquist; the fringe rate is "
            "close to half the sample rate and the record may be aliased",
            nyquist_cap / 1e3,
        )
        high = nyquist_cap

    if not 0.0 < low < high:
        raise ValueError(
            f"auto band-pass came out as ({low:.0f}, {high:.0f}) Hz from a measured "
            f"fringe rate of {r_lo:.0f}-{r_hi:.0f} Hz; check process.fringe_rate."
            "search_hz actually contains the fringe"
        )

    log.info(
        "auto band-pass %.0f-%.0f kHz, from a measured fringe rate of %.0f-%.0f kHz",
        low / 1e3, high / 1e3, r_lo / 1e3, r_hi / 1e3,
    )
    return low, high


def band_for(
    cfg: dict,
    mzi: np.ndarray,
    fs_hz: float,
    override: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """Settle process.bandpass_hz into the numbers phase_rad() wants.

    Accepts three forms, so a record written before `auto` existed still loads
    and behaves exactly as it did: a [low, high] pair, the string "auto", or
    null for no filtering at all.

    The resolved pair is written back into cfg. That is the point of doing this
    here rather than inside phase_rad: everything downstream that reports what
    it ran with -- the title of plot_phase, fit_ruler's params_yaml -- then
    records real numbers instead of the word "auto".

    Args:
        cfg: Loaded config; mutated in place.
        mzi: MZI samples for the sweep, needed only to resolve "auto".
        fs_hz: Sample rate.
        override: A band supplied by a caller, which wins over the config.

    Returns:
        (low, high), or None for no band-pass.
    """
    spec = override
    if spec is None:
        spec = cfg.get("process", {}).get("bandpass_hz")

    if isinstance(spec, str):
        if spec.strip().lower() != "auto":
            raise ValueError(
                f"process.bandpass_hz: expected [low, high], 'auto' or null, got {spec!r}"
            )
        spec = resolve_band_hz(mzi, fs_hz, config.params(cfg, "fringe_rate"))

    band = (float(spec[0]), float(spec[1])) if spec is not None else None
    cfg.setdefault("process", {})["bandpass_hz"] = list(band) if band else None
    return band


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
    phi = phase_rad(mzi, fs_hz, band_for(cfg, mzi, fs_hz, bandpass_hz))
    log.info(
        "sweep %d: %.3f Mfringe over %.2f s",
        sweep, (phi[-1] - phi[0]) / (2 * np.pi) / 1e6, len(phi) / fs_hz,
    )
    return t_s[sl] - t_s[sl.start], phi, mzi, cfg


def _decimate(y: np.ndarray, k: int) -> np.ndarray:
    """Block-mean a trace by k, discarding the ragged tail.

    Averaged, not strided: an absorption line is hundreds of samples wide, so
    the only thing block-averaging throws away is noise.
    """
    if k <= 1:
        return np.asarray(y, dtype=float)
    return y[: len(y) // k * k].reshape(-1, k).mean(axis=1, dtype=np.float64)


def find_dips(
    y: np.ndarray, width_samples: float, snr: float = 20.0
) -> tuple[np.ndarray, np.ndarray]:
    """Locate absorption dips in a transmission trace.

    Args:
        y: Transmission samples; absorption goes downward.
        width_samples: Expected line FWHM, in samples. Features narrower than a
            third of it or wider than four times it are rejected, which is what
            separates a real line from a single-sample glitch or from a slow
            ripple in the source power. The ripple matters: on the 20260813
            record it is only ~30x below the weakest real line, so a threshold
            alone finds twice as many dips as the cell contains.
        snr: Detection threshold, in multiples of the sample-to-sample noise.

    Returns:
        (positions, depths). Positions are fractional sample indices, taken as
        the midpoint of the half-depth crossings rather than the deepest
        sample: the same thing for a symmetric line, and much better for a
        strong one whose bottom is flat. Depths are peak prominences, measured
        against the local baseline -- a diagnostic for how confidently a line
        was seen, not a photometric absorption depth.
    """
    y = np.asarray(y, dtype=float)
    # Sample-to-sample MAD: robust to the dips themselves, and blind to the slow
    # baseline that a plain standard deviation would end up measuring instead.
    noise = float(np.median(np.abs(np.diff(y))) * 1.4826 / np.sqrt(2.0))
    if noise <= 0.0:  # quiet trace on a coarse ADC: every neighbour identical
        noise = float(np.ptp(y)) * 1e-3
    _, props = find_peaks(
        -y,
        prominence=snr * noise,
        distance=max(int(2.0 * width_samples), 1),
        width=(0.3 * width_samples, 4.0 * width_samples),
    )
    return 0.5 * (props["left_ips"] + props["right_ips"]), props["prominences"]


def assign_lines(
    lambda_nm: np.ndarray,
    cfg: dict,
    tol_nm: float = 0.1,
    n_mad: float = 5.0,
) -> list[tuple[str, str, float] | None]:
    """Match measured dip wavelengths against the gas-cell catalogue.

    Matching is nearest-line-wins on the nominal sweep wavelength. That works
    because the dial is good to ~20 pm while the lines are ~400 pm apart, so
    the assignment is never a close call -- and it is checked afterwards, since
    a spurious dip that happens to fall near a line would otherwise be adopted
    as an anchor.

    Args:
        lambda_nm: Nominal wavelength of each dip.
        cfg: Loaded config, for gascell.species and the sweep range.
        tol_nm: Largest nominal-vs-catalogue disagreement accepted outright.
        n_mad: Second pass: the residuals of the surviving matches should form
            one tight cluster, because they share a single systematic dial
            error. Anything this many robust sigma off that cluster was matched
            by coincidence and is dropped.

    Returns:
        One entry per input dip: (species, branch_label, catalogue_nm) if it was
        identified, None if not.
    """
    catalogue = sorted(
        (
            (species, label, nm)
            for species, lines in config.gas_lines_nm(cfg).items()
            for label, nm in lines.items()
        ),
        key=lambda entry: entry[2],
    )
    lam = np.asarray(lambda_nm, dtype=float)
    if not catalogue or not len(lam):
        return [None] * len(lam)

    known = np.array([entry[2] for entry in catalogue])
    right = np.searchsorted(known, lam).clip(1, len(known) - 1)
    near = np.where(
        np.abs(known[right] - lam) < np.abs(known[right - 1] - lam), right, right - 1
    )
    resid = lam - known[near]

    ok = np.abs(resid) < tol_nm
    if ok.any():
        med = np.median(resid[ok])
        # Floored at a hundredth of the tolerance -- about the catalogue's own
        # accuracy -- so that an unusually consistent set of matches does not
        # collapse the window onto itself and reject everything.
        mad = max(np.median(np.abs(resid[ok] - med)) * 1.4826, 0.01 * tol_nm)
        ok &= np.abs(resid - med) < n_mad * mad

    # Two dips can land on one line if tol_nm is loose. Keep the closer, rather
    # than hand step 4 two contradictory anchors for the same optical frequency.
    best: dict[int, int] = {}
    for i in np.flatnonzero(ok):
        if near[i] not in best or abs(resid[i]) < abs(resid[best[near[i]]]):
            best[near[i]] = i
    chosen = set(best.values())
    return [catalogue[near[i]] if i in chosen else None for i in range(len(lam))]


def gas_anchors(
    h5_path: str | Path,
    sweep: int = 0,
    bandpass_hz: tuple[float, float] | None = None,
) -> list[dict]:
    """Gas-cell lines of one sweep, placed on the MZI phase axis (step 3).

    Detects the absorption dips in the gascell channel, identifies each against
    the catalogue, and reads off the MZI phase where it sits. The result is the
    set of (phi, nu) pairs that step 4 fits nu(phi) through.

    Args:
        h5_path: Path to scan.h5.
        sweep: Which forward sweep, indexed from the trigger edges.
        bandpass_hz: Overrides the file's band-pass, as in sweep_phase().

    Detection and matching are tuned by process.anchors in the config:
    resolution_pm (wavelength per sample after decimation), snr (dip threshold)
    and tol_nm/n_mad (how far a dip may sit from a catalogue line).

    Returns:
        (anchors, band_hz). anchors is one dict per identified line, in sweep
        order, holding: species, label, lambda_nm and nu_hz (both from the
        catalogue -- these are the reference values, not measurements),
        nominal_lambda_nm (the dip's own measured position on the dial's
        wavelength axis -- use this, not lambda_nm, to place a marker on a
        trace plotted against nominal wavelength), phi_rad, depth, and
        residual_pm, the nominal wavelength minus the catalogue one. band_hz is
        the band-pass this call resolved, returned rather than left implicit so
        fit_ruler can record the numbers behind an `auto` without measuring the
        fringe rate a second time.
    """
    from .plot import load_h5
    from .sweeps import sweep_slices

    t_s, traces, cfg = load_h5(h5_path, roles=("mzi", "gascell", "trigger"))
    slices = sweep_slices(t_s, traces["trigger"], cfg)
    if not slices:
        raise ValueError("no complete forward sweep in this record")
    sl = slices[sweep]
    dt_s = float(t_s[1] - t_s[0])

    mzi = traces["mzi"][sl]
    band_hz = band_for(cfg, mzi, 1.0 / dt_s, bandpass_hz)
    phi = phase_rad(mzi, 1.0 / dt_s, band_hz)

    knobs = config.params(cfg, "anchors")
    resolution_pm = knobs["resolution_pm"]

    k = max(int(round(resolution_pm * 1e-3 / cfg["laser"]["speed_nm_s"] / dt_s)), 1)
    widths = [
        s.get("linewidth_pm", 16.0)
        for s in cfg.get("gascell", {}).get("species", {}).values()
    ]
    pos, depth = find_dips(
        _decimate(traces["gascell"][sl], k),
        min(widths, default=16.0) / resolution_pm,
        knobs["snr"],
    )

    index = pos * k  # back to full-rate samples within the sweep
    lam = config.nominal_lambda_nm(index * dt_s, cfg)
    # Rounding rather than interpolating phi costs ~0.15 fringe of frequency,
    # which is some 300x below the catalogue's own accuracy.
    phi_at = phi[np.round(index).astype(int).clip(0, len(phi) - 1)]

    anchors = [
        {
            "species": line[0],
            "label": line[1],
            "lambda_nm": line[2],
            "nu_hz": C_M_S / (line[2] * 1e-9),
            "nominal_lambda_nm": float(measured),
            "phi_rad": float(p),
            "depth": float(d),
            "residual_pm": float((measured - line[2]) * 1e3),
        }
        for line, measured, p, d in zip(
            assign_lines(lam, cfg, tol_nm=knobs["tol_nm"], n_mad=knobs["n_mad"]),
            lam, phi_at, depth,
        )
        if line is not None
    ]

    found = Counter(a["species"] for a in anchors)
    available = {s: len(v) for s, v in config.gas_lines_nm(cfg).items()}
    log.info(
        "sweep %d: %d dip(s) -> %d anchor(s) [%s], dial error %+.0f pm over a "
        "%.0f pm spread",
        sweep,
        len(pos),
        len(anchors),
        ", ".join(f"{s}: {found[s]}/{n}" for s, n in available.items()),
        np.median([a["residual_pm"] for a in anchors]) if anchors else 0.0,
        np.ptp([a["residual_pm"] for a in anchors]) if anchors else 0.0,
    )
    if not anchors:
        log.warning("no gas-cell line identified; nu(phi) cannot be anchored")
    return anchors, band_hz


def fit_nu_phi(
    anchors: list[dict],
    cfg: dict,
    degree: int | None = None,
    max_j: int | None = None,
) -> dict:
    """Fit nu(phi) through gas-cell anchors (step 4).

    A weighted polynomial is the whole model -- degree 3-4 buy nothing over a
    quadratic (measured 88.9-89.6 MHz RMS vs 90.2, which is the catalogue's own
    accuracy). Weights come from each species' accuracy_pm in the config, so
    CO -- roughly twice as tight as HCN -- pulls harder on the fit. H12C14N
    lines above |J|=max_j are dropped outright rather than just downweighted:
    their catalogue error roughly doubles there (SRM 2519a cross-check), and a
    weight alone does not stop a biased line pulling the fit, only a noisy one.

    Args:
        anchors: Output of gas_anchors().
        cfg: Loaded config, for gascell.species.*.accuracy_pm and process.ruler.
        degree: Polynomial degree; None takes process.ruler.degree. The
            pipeline leaves both of these None -- they are here for tests and
            interactive work, not for `spectro fit`, which reads the config so
            that the run folder records what the fit was made with.
        max_j: Lines of process.ruler.max_j_species with |J| (the digits in the
            branch label) above this are excluded; None takes the config's.

    Returns:
        {"coeffs": [...], "domain": [lo, hi], "degree", "rms_hz", "n_anchors",
        "dropped_high_j"}. coeffs are numpy.polynomial.Polynomial coefficients
        over the given domain, not a bare power series -- a raw
        numpy.polyfit/polyval on phi (order 1e6-1e7 rad here) is badly
        conditioned; recover the axis with nu_hz(phi_rad, fit).
    """
    p = config.params(cfg, "ruler")
    degree = p["degree"] if degree is None else degree
    max_j = p["max_j"] if max_j is None else max_j

    kept = [
        a for a in anchors
        if not (a["species"] == p["max_j_species"] and int(a["label"][1:]) > max_j)
    ]
    dropped = len(anchors) - len(kept)
    if len(kept) <= degree:
        raise ValueError(
            f"only {len(kept)} anchor(s) survive filtering; need more than degree={degree}"
        )

    accuracy_pm = {
        name: spec.get("accuracy_pm", 1.0)
        for name, spec in cfg.get("gascell", {}).get("species", {}).items()
    }
    phi = np.array([a["phi_rad"] for a in kept], dtype=float)
    nu = np.array([a["nu_hz"] for a in kept], dtype=float)
    # Polynomial.fit wants 1/sigma, not 1/sigma**2, for Gaussian uncertainties.
    w = np.array([1.0 / accuracy_pm.get(a["species"], 1.0) for a in kept], dtype=float)

    poly = np.polynomial.Polynomial.fit(phi, nu, degree, w=w)
    resid = nu - poly(phi)

    return {
        "coeffs": poly.coef.tolist(),
        "domain": poly.domain.tolist(),
        "degree": degree,
        "rms_hz": float(np.sqrt(np.mean(resid**2))),
        "n_anchors": len(kept),
        "dropped_high_j": dropped,
    }


def nu_hz(phi_rad, fit: dict):
    """Evaluate a fit_nu_phi() result at arbitrary phi -- step 5's entry point."""
    return np.polynomial.Polynomial(fit["coeffs"], domain=fit["domain"])(phi_rad)


def fit_ruler(
    h5_path: str | Path,
    sweep: int = 0,
    bandpass_hz: tuple[float, float] | None = None,
    anchors: list[dict] | None = None,
) -> dict:
    """Fit nu(phi) for one sweep and write it into the file (step 4).

    Stores the polynomial coefficients and domain rather than an evaluated
    axis: a handful of floats reproduce nu(phi) exactly for any phi, where the
    evaluated axis would cost ~200 MB per sweep for no extra accuracy. Written
    to ruler/sweep_<n> so a multi-sweep record keeps one fit per sweep,
    matching how gas_anchors and sweep_phase already work per-sweep.

    The knobs that produced the fit go into the same group, as params_yaml.
    The file's own config_yaml cannot serve: it is frozen at acquisition time,
    while these are chosen here, possibly days later and possibly more than
    once. Read the group, not the config, to know what a fit was made with.

    Args:
        h5_path: Path to scan.h5.
        sweep: Which forward sweep, indexed from the trigger edges.
        bandpass_hz: Overrides the file's band-pass, as in gas_anchors().
        anchors: Result of a gas_anchors() call already made, to skip
            recomputing it.

    The fit itself is tuned by process.ruler in the config.

    Returns:
        The dict from fit_nu_phi().
    """
    import h5py
    import yaml

    from .plot import load_h5

    h5_path = Path(h5_path)
    anchors_supplied = anchors is not None
    band_hz = bandpass_hz
    if anchors is None:
        anchors, band_hz = gas_anchors(h5_path, sweep, bandpass_hz=bandpass_hz)
    _, _, cfg = load_h5(h5_path, roles=())  # attrs only; no channel is read

    fit = fit_nu_phi(anchors, cfg)

    used = {
        # Resolved numbers, never the word "auto" -- unless the anchors came from
        # a caller, in which case this function never saw the trace they were
        # built from and can only report what the file asks for.
        "bandpass_hz": list(band_hz) if band_hz else cfg["process"].get("bandpass_hz"),
        # Only true of the anchors this call made: ones handed in were built by
        # the caller, under knobs this function never sees.
        "anchors": "supplied by caller" if anchors_supplied else config.params(cfg, "anchors"),
        "ruler": config.params(cfg, "ruler"),
    }

    with h5py.File(h5_path, "r+") as f:
        grp = f.require_group(f"ruler/sweep_{sweep}")
        grp.attrs["coeffs_hz"] = fit["coeffs"]
        grp.attrs["domain_rad"] = fit["domain"]
        grp.attrs["degree"] = fit["degree"]
        grp.attrs["rms_hz"] = fit["rms_hz"]
        grp.attrs["n_anchors"] = fit["n_anchors"]
        grp.attrs["dropped_high_j"] = fit["dropped_high_j"]
        grp.attrs["params_yaml"] = yaml.safe_dump(used, sort_keys=False)

    log.info(
        "sweep %d: nu(phi) fit degree %d, %d anchor(s) (%d dropped), %.0f MHz RMS",
        sweep, fit["degree"], fit["n_anchors"], fit["dropped_high_j"], fit["rms_hz"] / 1e6,
    )
    return fit


def read_ruler_fit(h5_path: str | Path, sweep: int = 0) -> dict:
    """Read back a fit_ruler() result written into the file.

    Mirrors fit_ruler's write, so dut_spectrum() can use a fit made earlier
    (by `spectro fit`) without recomputing gas_anchors + fit_nu_phi. "params"
    is what that fit was actually made with, and is {} for a fit written before
    fit_ruler recorded them.
    """
    import h5py
    import yaml

    with h5py.File(h5_path, "r") as f:
        key = f"ruler/sweep_{sweep}"
        if key not in f:
            raise ValueError(f"no ruler fit for sweep {sweep}; run `spectro fit` first")
        grp = f[key]
        return {
            "coeffs": list(grp.attrs["coeffs_hz"]),
            "domain": list(grp.attrs["domain_rad"]),
            "degree": int(grp.attrs["degree"]),
            "rms_hz": float(grp.attrs["rms_hz"]),
            "n_anchors": int(grp.attrs["n_anchors"]),
            "dropped_high_j": int(grp.attrs["dropped_high_j"]),
            "params": yaml.safe_load(grp.attrs.get("params_yaml", "{}")),
        }


def dut_spectrum(
    h5_path: str | Path,
    sweep: int = 0,
    bandpass_hz: tuple[float, float] | None = None,
    fit: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Map the DUT trace onto absolute optical frequency (step 5).

    Uses the sweep's own MZI phase and the nu(phi) fit from fit_ruler() --
    read back from the file if not supplied -- rather than a constant FSR:
    that is the whole reason steps 3-4 exist. resample_uniform then puts the
    DUT on a uniform frequency grid, since nu(phi) is not linear in sample
    index.

    Args:
        h5_path: Path to scan.h5.
        sweep: Which forward sweep (0-based).
        bandpass_hz: Overrides the file's band-pass, as in sweep_phase().
        fit: A fit_nu_phi()/fit_ruler() result already in hand. Read from
            ruler/sweep_<n> in the file if not given -- run `spectro fit`
            first if that group does not exist yet.

    The output grid is set by process.spectrum in the config.

    Returns:
        (freq_hz, dut, fit): ascending frequency grid, DUT transmission on
        it, and the fit used (so a caller can print rms_hz etc. without a
        second file read).
    """
    from .plot import load_h5
    from .sweeps import sweep_slices

    h5_path = Path(h5_path)
    t_s, traces, cfg = load_h5(h5_path, roles=("mzi", "dut", "trigger"))
    slices = sweep_slices(t_s, traces["trigger"], cfg)
    if not slices:
        raise ValueError("no complete forward sweep in this record")
    sl = slices[sweep]
    dt_s = float(t_s[1] - t_s[0])

    mzi = traces["mzi"][sl]
    phi = phase_rad(mzi, 1.0 / dt_s, band_for(cfg, mzi, 1.0 / dt_s, bandpass_hz))

    if fit is None:
        fit = read_ruler_fit(h5_path, sweep)

    p = config.params(cfg, "spectrum")
    nu = nu_hz(phi, fit)
    freq_hz, dut = resample_uniform(
        nu, traces["dut"][sl], n_points=p["n_points"], trim_frac=p["trim_frac"]
    )
    return freq_hz, dut, fit


def _lorentzian(f, f0, fwhm, amp, baseline):
    return baseline + amp / (1.0 + ((f - f0) / (0.5 * fwhm)) ** 2)


def _resonance_candidates(y: np.ndarray, width_samples: float, snr: float) -> np.ndarray:
    """Positions of strong deviations from baseline, peak or dip alike.

    Unlike find_dips (built for the always-absorbing gas cell), a resonance
    can go either way depending on which DUT port is recorded -- on the
    20260813 ring it is a peak, not a dip. Detection therefore runs on
    |y - median(y)| rather than committing to a sign; the fit's amp comes out
    positive for a peak, negative for a dip.
    """
    y = np.asarray(y, dtype=float)
    baseline = float(np.median(y))
    noise = float(np.median(np.abs(np.diff(y))) * 1.4826 / np.sqrt(2.0))
    if noise <= 0.0:  # quiet trace on a coarse ADC: every neighbour identical
        noise = float(np.ptp(y)) * 1e-3
    pos, _ = find_peaks(
        np.abs(y - baseline), prominence=snr * noise, distance=max(int(2.0 * width_samples), 1)
    )
    return pos


def fit_resonances(
    freq_hz: np.ndarray,
    dut: np.ndarray,
    expected_fwhm_hz: float = 2.0e8,
    snr: float = 100.0,
    window_fwhm: float = 4.0,
    window_gap_frac: float = 0.4,
) -> list[dict]:
    """Fit a Lorentzian to each resonance in a DUT spectrum (step 6).

    expected_fwhm_hz only needs to be the right order of magnitude -- the
    real linewidth comes out of the local fit. Each candidate's fit window is
    capped at 40% of its distance to the nearest other candidate, so one
    resonance's fit never eats a neighbour's wing; without that cap a window
    sized off a bad expected_fwhm_hz guess can span several free spectral
    ranges and the fit runs away (seen directly on the 20260813 record: an
    uncapped window returns a FWHM pinned to the window's own edge).

    Args:
        freq_hz: Absolute frequency axis (ascending), from dut_spectrum().
        dut: DUT transmission on the same grid.
        expected_fwhm_hz: Order-of-magnitude linewidth guess, for detection
            and the initial fit guess. 200 MHz matches the 20260813 ring
            (measured 185-266 MHz across 125 resonances, Q ~ 7-11e5, FSR
            ~100 GHz); override for a different DUT.
        snr: Detection threshold, in multiples of the trace noise. Set much
            higher than find_dips' default: this ring's resonances stand
            ~400x the sample noise, so a low threshold just finds every
            noise wiggle inside each resonance's own wings.
        window_fwhm: Fit half-window, in expected FWHM.
        window_gap_frac: Cap on that half-window, as a fraction of the
            distance to the nearest other candidate.

    Returns:
        One dict per resonance that converged: f0_hz, fwhm_hz, q, amp_hz
        (signed: positive for a peak, negative for a dip), baseline, fit_rms.
        A candidate whose fit does not converge (two resonances too close
        together, say) is skipped rather than raising, so one bad candidate
        does not lose the rest.
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    dut = np.asarray(dut, dtype=float)
    df = float(np.median(np.diff(freq_hz)))
    width_samples = expected_fwhm_hz / abs(df)

    pos = _resonance_candidates(dut, width_samples, snr)
    if len(pos) == 0:
        return []
    spacing = np.diff(np.sort(pos))
    typical_gap = float(np.median(spacing)) if len(spacing) else window_fwhm * width_samples
    half_window = max(
        int(min(window_fwhm * width_samples, window_gap_frac * typical_gap)), 3
    )

    baseline0 = float(np.median(dut))
    out = []
    for p in pos:
        i0 = int(round(p))
        lo, hi = max(i0 - half_window, 0), min(i0 + half_window, len(freq_hz))
        f, y = freq_hz[lo:hi], dut[lo:hi]
        if len(f) < 5:
            continue
        amp0 = float(y[np.argmax(np.abs(y - baseline0))] - baseline0)
        p0 = [freq_hz[i0], expected_fwhm_hz, amp0, baseline0]
        try:
            popt, _ = curve_fit(
                _lorentzian, f, y, p0=p0,
                bounds=(
                    [f.min(), 0.0, -np.inf, -np.inf],
                    [f.max(), f.max() - f.min(), np.inf, np.inf],
                ),
                maxfev=3000,
            )
        except RuntimeError:
            continue
        f0, fwhm, amp, baseline = popt
        resid = y - _lorentzian(f, *popt)
        out.append({
            "f0_hz": float(f0),
            "fwhm_hz": float(fwhm),
            "q": float(f0 / fwhm) if fwhm > 0 else float("inf"),
            "amp_hz": float(amp),
            "baseline": float(baseline),
            "fit_rms": float(np.sqrt(np.mean(resid**2))),
        })
    return out


def find_resonances(
    h5_path: str | Path,
    sweep: int = 0,
    fit: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """DUT spectrum plus its fitted resonances, in one call (steps 5+6).

    Detection and fitting are tuned by process.resonances in the config.

    Returns:
        (freq_hz, dut, resonances) -- see dut_spectrum() and fit_resonances().
    """
    from .plot import load_h5

    _, _, cfg = load_h5(h5_path, roles=())  # attrs only; no channel is read
    p = config.params(cfg, "resonances")

    freq_hz, dut, _ = dut_spectrum(h5_path, sweep=sweep, fit=fit)
    resonances = fit_resonances(freq_hz, dut, **p)
    log.info("sweep %d: %d resonance(s) fit", sweep, len(resonances))
    return freq_hz, dut, resonances


def resample_uniform(
    nu_hz: np.ndarray,
    y: np.ndarray,
    n_points: int | None = None,
    trim_frac: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate y from the (non-uniform) frequency axis onto a uniform grid.

    Args:
        nu_hz: Per-sample optical frequency, e.g. from nu_hz() (the
            gas-anchored fit evaluated over a sweep's phase).
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
