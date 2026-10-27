"""Compute D2 (second-order dispersion) from a ring's fitted resonances.

Builds on `spectro.ruler.find_resonances`: drops resonances whose fit looks
like a failure rather than a real feature (near-zero amplitude, typically a
symptom of Q collapsing right at an avoided mode crossing), assigns each
survivor an integer mode number relative to a reference wavelength, fits a
quadratic through frequency vs. mode number with iterative outlier
rejection (so the AMX itself doesn't bias the fit), and reports D1 (FSR)
and D2 the same way Yang et al. 2025 (Turnkey Deterministic Soliton Crystal
Generation) do: D2/2pi is the mu^2 coefficient of a polynomial fit through
the integrated dispersion.

Run `spectro fit <h5file> --sweep <n>` first -- this script reads back that
fit rather than recomputing it, same as `spectro plot --resonances`.

Usage:
    spectro d2 data/<run>/scan.h5
    spectro d2 data/<run>/scan.h5 --sweep 0 --show

The fit knobs come from process.dispersion in the scan's own config, so the
run folder records what produced its D2.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import config, plot, ruler
from .ruler import C_M_S, _robust_sigma

log = logging.getLogger(__name__)


def filter_resonances(resonances: list[dict], n_mad: float = 5.0) -> list[dict]:
    """Drop resonances whose fit is likely a failure, not a real feature.

    Right at an avoided mode crossing, Q drops sharply (Yang et al. 2025)
    and the resonance can broaden and weaken enough that fit_resonances'
    single-Lorentzian model can't constrain it: the fit converges on
    something with amplitude near zero and a linewidth pinned near its
    search window's edge, rather than failing outright (which
    fit_resonances already handles by skipping). These pass through as
    "resonances" that aren't, and land at essentially arbitrary frequencies
    -- exactly what causes assign_mode_numbers to raise a collision.

    Filtering on amplitude rather than frequency or fwhm because it's the
    cleanest signal: on the 20260813 ring, five fit failures sat 50-600x
    below the 120 real resonances' amplitude, a gap wide enough that a
    linear-amplitude threshold would need to be tuned per scan. Rejection
    is on log|amp_hz| instead, since the failures differ multiplicatively
    from the bulk, not additively -- the same MAD-based pattern already
    used in ruler.assign_lines and fit_dispersion, just in log space.

    Args:
        resonances: Output of ruler.find_resonances().
        n_mad: Rejection threshold, in robust sigma of log|amp_hz|.

    Returns:
        resonances with the low-amplitude fit failures removed, in the same
        order. Logs what was dropped and why, since a caller who expects a
        specific resonance count should be able to see where any went.
    """
    if len(resonances) <= 1:
        return list(resonances)

    log_amp = np.log(np.abs([r["amp_hz"] for r in resonances]))
    sigma = _robust_sigma(log_amp) or float(np.std(log_amp))
    # One-sided: only the weak tail is a fit failure.
    keep = log_amp > np.median(log_amp) - n_mad * sigma

    kept = [r for r, k in zip(resonances, keep) if k]
    dropped = [r for r, k in zip(resonances, keep) if not k]
    if dropped:
        log.info(
            "dropped %d likely fit failure(s) (near-zero amplitude, probably an "
            "avoided mode crossing broadening the resonance past this fit's reach): %s",
            len(dropped),
            ", ".join(
                f"{C_M_S / r['f0_hz'] * 1e9:.3f} nm (amp={r['amp_hz']:+.5f})" for r in dropped
            ),
        )
    return kept


def _comb_spacing(f_sorted: np.ndarray) -> float:
    """FSR of the dominant comb, robust to a second family interleaved with it.

    The median consecutive spacing is right for one family but not for two:
    an interleaved family splits each FSR into a pair of spacings that smear
    across 0..FSR (on the 20261026 25 GHz ring the median came out at 19 GHz).
    Spacings with no interloper between them all pile up at the FSR, so the
    histogram's mode finds it where the median does not.
    """
    d = np.diff(f_sorted)
    w = 0.005 * float(np.median(d))  # bin width; the two families' FSRs differ by ~0.4%
    hist, edges = np.histogram(d, bins=np.arange(0.0, d.max() + 2 * w, w))
    peak = edges[np.argmax(hist)] + 0.5 * w
    return float(np.median(d[np.abs(d - peak) < 2 * w]))


def _walk_family(f: np.ndarray, i0: int, fsr: float, tol_fsr: float) -> list[int]:
    """Indices into ascending f of one mode family, followed outward from f[i0].

    Each step predicts the next mode from the median local FSR of the last
    few accepted modes -- so D2 and the seed's error are tracked as it goes --
    and takes the nearest resonance within tol_fsr * FSR of the prediction.
    Nothing that close is a missing mode: skip it and predict one further.
    """
    track = [i0]
    for step in (1, -1):
        pts = [(0, f[i0])]
        mu = 0
        while True:
            mu += step
            recent = np.array(pts[-6:])
            local = fsr if len(pts) == 1 else float(np.median(
                np.diff(recent[:, 1]) / np.diff(recent[:, 0])))
            # Median over the last few anchors, so one wrong pick can't drag the walk.
            pred = float(np.median(recent[:, 1] + (mu - recent[:, 0]) * local))
            if not f[0] - fsr < pred < f[-1] + fsr:
                break
            j = int(np.argmin(np.abs(f - pred)))
            if abs(f[j] - pred) < tol_fsr * local:
                pts.append((mu, f[j]))
                track.append(j)
    return track


def keep_main_family(resonances: list[dict], ref_nm: float = 1550.0,
                     tol_fsr: float = 0.05) -> list[dict]:
    """Keep the resonances of one mode family, drop any other.

    A second family (other polarisation, higher-order transverse mode) sits
    a drifting fraction of an FSR off the main one -- on the 20261026 25 GHz
    ring, ~90 MHz per mode, at depths comparable to the main family's -- and
    makes assign_mode_numbers collide. Depth can't tell the families apart;
    continuity can: walk outward from the resonances nearest ref_nm, one FSR
    at a time, and keep the longest track. The other family is only ever
    nearer the prediction than the main one within a few modes of where the
    two cross, and there the local FSR, learnt from the main family, keeps
    the walk on it.

    ponytail: picks by distance alone, so where the families sit within
    tol_fsr of each other and a main mode is missing, the walk can take the
    other one for a few modes; fit_dispersion's outlier rejection drops
    those. Fit each family's f(mu) jointly if that ever costs real modes.

    Args:
        resonances: Output of filter_resonances().
        ref_nm: Wavelength the walks start from.
        tol_fsr: Acceptance window around each predicted mode, in FSRs.

    Returns:
        The main family's resonances, ascending in frequency. Warns with
        what was dropped, since it means the DUT spectrum is not
        single-family.
    """
    if len(resonances) <= 2:
        return list(resonances)
    res = sorted(resonances, key=lambda r: r["f0_hz"])
    f = np.array([r["f0_hz"] for r in res])
    fsr = _comb_spacing(f)

    i_ref = int(np.argmin(np.abs(f - C_M_S / (ref_nm * 1e-9))))
    # Within one FSR, not half: the main family's own resonance nearest ref_nm
    # may be undetected, leaving only the other family's within half an FSR.
    seeds = np.flatnonzero(np.abs(f - f[i_ref]) < fsr)
    track = set(max((_walk_family(f, int(i), fsr, tol_fsr) for i in seeds), key=len))

    dropped = [r for i, r in enumerate(res) if i not in track]
    if dropped:
        log.warning(
            "%d resonance(s) off the main mode family (FSR %.3f GHz) -- more than one "
            "mode family in this DUT spectrum, or a duplicate fit; dropped: %s",
            len(dropped), fsr / 1e9,
            ", ".join(
                f"{C_M_S / r['f0_hz'] * 1e9:.3f} nm (amp={r['amp_hz']:+.5f})" for r in dropped
            ),
        )
    return [r for i, r in enumerate(res) if i in track]


def assign_mode_numbers(f0_hz: np.ndarray, ref_nm: float = 1550.0) -> tuple[np.ndarray, int]:
    """Assign an integer mode number to each resonance frequency.

    Mode numbers are relative to whichever resonance sits closest to ref_nm
    (mu=0), spaced by a coarse FSR estimate -- the median of consecutive
    spacings, robust to the occasional missing resonance near an avoided
    mode crossing. Rounding to the nearest integer multiple (rather than
    plain enumeration) means a gap in the resonance list does not shift
    every mode number after it.

    Args:
        f0_hz: Resonance centre frequencies, any order.
        ref_nm: Wavelength whose nearest resonance becomes mode 0.

    Returns:
        (mu, ref_index): integer mode number per input resonance (same
        order as f0_hz), and the index of the reference resonance in that
        same order.

    Raises:
        ValueError: two resonances round to the same mode number, which
        means the FSR estimate is off or the DUT spectrum mixes more than
        one mode family -- this function assumes a single family.
    """
    f0_hz = np.asarray(f0_hz, dtype=float)
    order = np.argsort(f0_hz)
    f_sorted = f0_hz[order]
    fsr_est = float(np.median(np.diff(f_sorted)))

    ref_hz = C_M_S / (ref_nm * 1e-9)
    i_ref_sorted = int(np.argmin(np.abs(f_sorted - ref_hz)))
    f_ref = f_sorted[i_ref_sorted]

    mu_sorted = np.round((f_sorted - f_ref) / fsr_est).astype(int)

    values, counts = np.unique(mu_sorted, return_counts=True)
    dupes = values[counts > 1]
    if len(dupes):
        clash_hz = [f_sorted[mu_sorted == m].tolist() for m in dupes[:3]]
        raise ValueError(
            f"mode numbers collide at mu={dupes[:3].tolist()} (FSR estimate "
            f"{fsr_est / 1e9:.3f} GHz may be wrong, or more than one mode "
            f"family is present in this DUT spectrum): frequencies {clash_hz}"
        )

    mu = np.empty_like(mu_sorted)
    mu[order] = mu_sorted
    ref_index = int(order[i_ref_sorted])
    log.info(
        "reference mode mu=0 at %.6f THz (nearest to %.3f nm), coarse FSR estimate %.3f GHz",
        f_ref / 1e12, ref_nm, fsr_est / 1e9,
    )
    return mu, ref_index


def fit_dispersion(
    mu: np.ndarray,
    f0_hz: np.ndarray,
    degree: int = 2,
    n_mad: float = 5.0,
    max_iter: int = 5,
    amx_floor_hz: float = 0.0,
) -> dict:
    """Fit frequency vs. mode number with iterative outlier rejection.

    Mirrors the MAD-based rejection ruler.fit_nu_phi and ruler.assign_lines
    already use elsewhere in this package: fit, reject points more than
    n_mad robust-sigma from the fit, refit on the survivors, repeat. This is
    what keeps an avoided mode crossing from dragging the curve -- without
    it, the AMX would bias D2 by however much it displaces the one or two
    resonances nearest to it.

    Args:
        mu: Integer mode numbers.
        f0_hz: Resonance centre frequencies, same order as mu.
        degree: Polynomial degree. 2 is what recovers D1 and D2; neither the
            paper nor this fit reports D3 or higher.
        n_mad: Rejection threshold, in robust sigma (median absolute
            deviation, scaled to match a Gaussian sigma).
        max_iter: Rejection rounds. Stops early once nothing new is
            rejected.

    Returns:
        {"poly": Polynomial, "kept": bool mask into the input arrays,
        "d1_hz": FSR (D1/2pi), "d2_hz": D2/2pi, "rms_hz": fit RMS on the
        kept points}. d1_hz/d2_hz come from poly.deriv() evaluated at mu=0,
        which sidesteps Polynomial.fit's internal domain scaling rather
        than having to undo it.

    Raises:
        ValueError: fewer than degree+1 points survive rejection.
    """
    mu = np.asarray(mu, dtype=float)
    f0_hz = np.asarray(f0_hz, dtype=float)
    kept = np.ones(len(mu), dtype=bool)

    for _ in range(max_iter):
        if kept.sum() <= degree:
            raise ValueError(
                f"only {int(kept.sum())} resonance(s) survive outlier rejection; "
                f"need more than degree={degree}"
            )
        poly = np.polynomial.Polynomial.fit(mu[kept], f0_hz[kept], degree)
        resid = f0_hz - poly(mu)
        med = np.median(resid[kept])
        sigma = _robust_sigma(resid[kept]) or float(np.std(resid[kept]))
        threshold = max(n_mad * sigma, amx_floor_hz)
        new_kept = np.abs(resid - med) < threshold
        if np.array_equal(new_kept, kept):
            break
        kept = new_kept
    else:
        # Loop exhausted without converging; refit once more on the final
        # `kept` so the returned polynomial always matches what it's fit to.
        poly = np.polynomial.Polynomial.fit(mu[kept], f0_hz[kept], degree)

    resid = f0_hz - poly(mu)
    d1_hz = float(poly.deriv(1)(0.0))
    d2_hz = float(poly.deriv(2)(0.0))
    rms_hz = float(np.sqrt(np.mean(resid[kept] ** 2)))

    log.info(
        "dispersion fit: degree %d, %d/%d resonance(s) kept, D1/2pi=%.4f GHz, "
        "D2/2pi=%.1f kHz/mu^2, RMS %.2f MHz",
        degree, int(kept.sum()), len(mu), d1_hz / 1e9, d2_hz / 1e3, rms_hz / 1e6,
    )
    return {"poly": poly, "kept": kept, "d1_hz": d1_hz, "d2_hz": d2_hz, "rms_hz": rms_hz}


def amx_candidates(mu: np.ndarray, f0_hz: np.ndarray, fit: dict) -> list[dict]:
    """Resonances excluded by fit_dispersion's outlier rejection.

    These are avoided-mode-crossing candidates: whatever fit_dispersion
    rejected as it converged on a clean parabola. Reports each one's mode
    number, wavelength, and deviation from the fit -- the largest deviation
    is the AMX strength, its wavelength the AMX location. This does not
    interpolate between adjacent modes the way the paper's insets do; it's
    the nearest surviving resonance, not a sub-mode crossing point.

    Returns:
        Rejected resonances, largest |deviation| first. Empty if none were
        rejected.
    """
    mu = np.asarray(mu, dtype=float)
    f0_hz = np.asarray(f0_hz, dtype=float)
    poly = fit["poly"]
    resid = f0_hz - poly(mu)
    out = [
        {
            "mu": int(m),
            "lambda_nm": C_M_S / f * 1e9,
            "f0_hz": float(f),
            "deviation_hz": float(r),
        }
        for m, f, r, k in zip(mu, f0_hz, resid, fit["kept"])
        if not k
    ]
    return sorted(out, key=lambda a: -abs(a["deviation_hz"]))


def plot_dispersion(
    mu: np.ndarray,
    f0_hz: np.ndarray,
    fit: dict,
    out_path: str | Path,
    show: bool = False,
) -> Path:
    """D_int(mu) vs mu, with the fit and any AMX candidates marked.

    D_int is the fit's own linear term subtracted off (not a separate
    estimate of FSR), so this is a direct visual check against Figure 1b of
    Yang et al. 2025.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectro.plot import PALETTE

    mu = np.asarray(mu, dtype=float)
    f0_hz = np.asarray(f0_hz, dtype=float)
    poly, d1, kept = fit["poly"], fit["d1_hz"], fit["kept"]
    f_ref = float(poly(0.0))

    dint = f0_hz - f_ref - d1 * mu
    order = np.argsort(mu)
    mu_line = mu[order]
    dint_fit = poly(mu_line) - f_ref - d1 * mu_line

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mu_line, dint_fit / 1e6, lw=1.2, color=PALETTE[1], label="quadratic fit")
    ax.plot(
        mu[kept], dint[kept] / 1e6, ls="none", marker="o", ms=4,
        color=PALETTE[0], label="resonances (fit)",
    )
    if (~kept).any():
        ax.plot(
            mu[~kept], dint[~kept] / 1e6, ls="none", marker="^", ms=8,
            color=PALETTE[5], label="excluded (AMX candidate)",
        )
    ax.axhline(0.0, lw=0.6, color="0.7")
    ax.set_xlabel(r"Mode number $\mu$")
    ax.set_ylabel(r"$D_{\mathrm{int}}$ (MHz)")
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25, lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.suptitle(
        f"Integrated dispersion  --  D2/2π = {fit['d2_hz'] / 1e3:.1f} kHz/μ², "
        f"FSR = {fit['d1_hz'] / 1e9:.3f} GHz",
        fontsize=10,
    )
    fig.tight_layout()

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("wrote %s", out_path)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def dispersion_of(h5_path: str | Path, sweep: int = 0) -> dict:
    """D1, D2 and the mode assignment for one sweep (needs `spectro fit` first).

    The body main() used to carry inline, lifted out so a batch caller can have
    the numbers without going through argparse or re-reading printed output.

    No band-pass argument, deliberately: ruler.dut_spectrum builds phi with the
    band recorded in the ruler fit, so a record refitted under a different band
    is already self-consistent by the time this sees it.

    Args:
        h5_path: Path to scan.h5.
        sweep: Which forward sweep.

    Returns:
        The fit_dispersion() dict -- poly, kept, d1_hz, d2_hz, rms_hz -- plus
        mu, f0_hz, ref_index, amx, and the n_fit/n_failed/n_other_family/n_kept
        counts.
    """
    h5_path = Path(h5_path)
    _, _, cfg = plot.load_h5(h5_path, roles=())  # attrs only; no channel is read
    p = config.params(cfg, "dispersion")

    _, _, resonances = ruler.find_resonances(h5_path, sweep=sweep)
    if not resonances:
        raise ValueError(
            f"no resonances fit for sweep {sweep}; check "
            f"process.resonances.expected_fwhm_hz in the config"
        )

    n_fit = len(resonances)
    resonances = filter_resonances(resonances, n_mad=p["amp_n_mad"])
    n_filtered = len(resonances)
    resonances = keep_main_family(resonances, ref_nm=p["ref_nm"])

    f0_hz = np.array([r["f0_hz"] for r in resonances])
    mu, ref_index = assign_mode_numbers(f0_hz, ref_nm=p["ref_nm"])
    fit = fit_dispersion(
        mu, f0_hz, degree=p["degree"], n_mad=p["n_mad"], max_iter=p["max_iter"],
        amx_floor_hz=p["amx_floor_hz"],
    )
    return {
        **fit,
        "mu": mu,
        "f0_hz": f0_hz,
        "ref_index": ref_index,
        "amx": amx_candidates(mu, f0_hz, fit),
        "n_fit": n_fit,
        "n_failed": n_fit - n_filtered,
        "n_other_family": n_filtered - len(resonances),
        "n_kept": int(fit["kept"].sum()),
    }


def run(h5file: str | Path, sweep: int = 0, out: str | Path | None = None,
        show: bool = False) -> int:
    """Print D1 and D2 for one sweep and write the D_int diagnostic plot.

    The fit knobs -- ref_nm, degree, n_mad, max_iter, amp_n_mad -- live in
    process.dispersion of the run folder's config.yaml (see plot.load_h5), so
    the folder records what produced its numbers. Edit that file, not a flag.
    """
    h5_path = Path(h5file)
    d = dispersion_of(h5_path, sweep)
    f0_ref = d["f0_hz"][d["ref_index"]]

    print(
        f"{d['n_fit']} resonance(s): {d['n_failed']} dropped as low-amplitude fit "
        f"failures, {d['n_other_family']} as another mode family, {d['n_kept']} used in fit, {len(d['amx'])} excluded as AMX "
        "candidate(s)"
    )
    print(
        f"reference mode mu=0 at {f0_ref / 1e12:.6f} THz "
        f"({C_M_S / f0_ref * 1e9:.3f} nm)"
    )
    print(f"D1/2pi (FSR): {d['d1_hz'] / 1e9:.4f} GHz")
    print(f"D2/2pi:       {d['d2_hz'] / 1e3:.1f} kHz/mu^2")
    print(
        f"fit RMS:      {d['rms_hz'] / 1e6:.2f} MHz on {d['n_kept']} kept resonance(s)"
    )
    if d["amx"]:
        print("AMX candidates (mode, wavelength, deviation from fit):")
        for a in d["amx"]:
            print(f"  mu={a['mu']:+4d}  {a['lambda_nm']:10.4f} nm  {a['deviation_hz'] / 1e6:+8.2f} MHz")

    out_path = Path(out) if out else h5_path.with_name(
        f"{h5_path.stem}_dispersion_{sweep}.png"
    )
    plot_dispersion(d["mu"], d["f0_hz"], d, out_path, show=show)
    print(f"wrote {out_path}")
    return 0
