"""Compute D2 (second-order dispersion) from a ring's fitted resonances.

Builds on `spectro.ruler.find_resonances`: assigns each resonance an integer
mode number relative to a reference wavelength, fits a quadratic through
frequency vs. mode number with iterative outlier rejection (so an avoided
mode crossing does not bias the fit), and reports D1 (FSR) and D2 the same
way Yang et al. 2025 (Turnkey Deterministic Soliton Crystal Generation) do:
D2/2pi is the mu^2 coefficient of a polynomial fit through the integrated
dispersion.

Run `spectro fit <h5file> --sweep <n>` first -- this script reads back that
fit rather than recomputing it, same as `spectro plot --resonances`.

Usage:
    python scripts/calc_d2.py data/<run>/scan.h5
    python scripts/calc_d2.py data/<run>/scan.h5 --sweep 0 --ref-nm 1550 --show
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from spectro import ruler

C_M_S = 299_792_458.0

log = logging.getLogger(__name__)


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
        mad = np.median(np.abs(resid[kept] - med))
        sigma = 1.4826 * mad if mad > 0 else np.std(resid[kept])
        new_kept = np.abs(resid - med) < n_mad * sigma
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "h5file", help="path to a scan.h5 (needs `spectro fit` run first)"
    )
    parser.add_argument("--sweep", type=int, default=0, help="which forward sweep")
    parser.add_argument(
        "--ref-nm", type=float, default=1550.0,
        help="wavelength whose nearest resonance becomes mode 0",
    )
    parser.add_argument("--degree", type=int, default=2, help="fit polynomial degree")
    parser.add_argument(
        "--n-mad", type=float, default=5.0,
        help="outlier rejection threshold, in robust sigma",
    )
    parser.add_argument("--max-iter", type=int, default=5, help="rejection rounds")
    parser.add_argument(
        "--expected-fwhm-hz", type=float, default=2.0e8,
        help="order-of-magnitude resonance linewidth guess, passed to find_resonances",
    )
    parser.add_argument(
        "--out", default=None,
        help="diagnostic plot path; defaults to <stem>_dispersion_<sweep>.png",
    )
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    h5_path = Path(args.h5file)
    _, _, resonances = ruler.find_resonances(
        h5_path, sweep=args.sweep, expected_fwhm_hz=args.expected_fwhm_hz
    )
    if not resonances:
        raise ValueError(
            f"no resonances fit for sweep {args.sweep}; check --expected-fwhm-hz"
        )

    f0_hz = np.array([r["f0_hz"] for r in resonances])
    mu, ref_index = assign_mode_numbers(f0_hz, ref_nm=args.ref_nm)
    fit = fit_dispersion(
        mu, f0_hz, degree=args.degree, n_mad=args.n_mad, max_iter=args.max_iter
    )
    amx = amx_candidates(mu, f0_hz, fit)

    n_kept = int(fit["kept"].sum())
    print(f"{len(resonances)} resonance(s): {n_kept} used in fit, {len(amx)} excluded as AMX candidate(s)")
    print(
        f"reference mode mu=0 at {f0_hz[ref_index] / 1e12:.6f} THz "
        f"({C_M_S / f0_hz[ref_index] * 1e9:.3f} nm)"
    )
    print(f"D1/2pi (FSR): {fit['d1_hz'] / 1e9:.4f} GHz")
    print(f"D2/2pi:       {fit['d2_hz'] / 1e3:.1f} kHz/mu^2")
    print(f"fit RMS:      {fit['rms_hz'] / 1e6:.2f} MHz on {n_kept} kept resonance(s)")
    if amx:
        print("AMX candidates (mode, wavelength, deviation from fit):")
        for a in amx:
            print(f"  mu={a['mu']:+4d}  {a['lambda_nm']:10.4f} nm  {a['deviation_hz'] / 1e6:+8.2f} MHz")

    out_path = Path(args.out) if args.out else h5_path.with_name(
        f"{h5_path.stem}_dispersion_{args.sweep}.png"
    )
    plot_dispersion(mu, f0_hz, fit, out_path, show=args.show)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
