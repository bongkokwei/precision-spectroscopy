# D2 (second-order dispersion) from a ring's fitted resonances

Date: 2026-08-14
Status: implemented

## Problem

`spectro.ruler.find_resonances()` already returns each ring resonance's
fitted centre frequency. Nothing in the package turns that list into D1
(FSR) and D2 (second-order dispersion), the quantities Yang et al. 2025
("Turnkey Deterministic Soliton Crystal Generation") report from the same
kind of swept-wavelength measurement.

## Approach

A standalone script, `scripts/calc_d2.py`, rather than a new package
module: it only needs three new pieces of logic on top of what `ruler.py`
already provides, and keeping it out of `src/spectro/` avoids touching
working code or reversing the "no persisted calibrated dataset" choice
noted in the README's Known Simplifications.

1. **Mode numbering** (`assign_mode_numbers`): resonances are sorted by
   frequency, given a coarse FSR estimate (median consecutive spacing), and
   assigned an integer mode number relative to whichever resonance sits
   closest to `--ref-nm` (default 1550 nm, matching the paper). Rounding
   rather than plain enumeration tolerates a missing resonance without
   shifting every mode number after it. Two resonances rounding to the same
   mode number raises, rather than silently overwriting one.

2. **Robust quadratic fit** (`fit_dispersion`): `f(mu) = f0 + D1/2pi * mu +
   (D2/2pi)/2 * mu^2` via `numpy.polynomial.Polynomial.fit`, with iterative
   MAD-based outlier rejection -- the same pattern already used in
   `ruler.assign_lines` and `ruler.fit_nu_phi` -- so an avoided mode
   crossing (AMX) doesn't bias the fit. D1 and D2 are read off as
   `poly.deriv(1)(0)` and `poly.deriv(2)(0)`, which sidesteps
   `Polynomial.fit`'s internal domain scaling instead of having to undo it.

3. **AMX reporting** (`amx_candidates`): whatever the rejection step
   excluded is reported as an AMX candidate (mode, wavelength, deviation).
   This is the nearest surviving resonance, not the paper's sub-mode
   interpolation between adjacent modes -- a documented simplification, not
   an attempt to match Table 1 exactly.

4. **Diagnostic plot** (`plot_dispersion`): D_int(mu) vs mu with the fit
   overlaid and excluded points marked, in the same style (`PALETTE`, spine
   removal) as `plot.py`'s existing diagnostics -- a direct visual check
   against Figure 1b of the paper.

CLI: `python scripts/calc_d2.py <h5file> [--sweep] [--ref-nm] [--degree]
[--n-mad] [--max-iter] [--expected-fwhm-hz] [--out] [--show]`. Requires
`spectro fit` to have already been run on the file (same requirement as
`spectro plot --resonances`); the existing `ValueError` from
`ruler.read_ruler_fit` propagates unchanged rather than silently re-running
it.

## Alternatives considered

- **A `spectro.dispersion` module** integrated into the CLI and persisting
  the fit into `scan.h5` (mirroring `ruler/sweep_<n>`). More consistent
  with the rest of the package, but touches three files and reverses a
  deliberate design choice for a single new calculation -- deferred unless
  D2 needs to be reused programmatically elsewhere.
- **A sandbox one-off** (no CLI, hardcoded path). Fastest, but not reusable
  on the next scan.

## Testing

`tests/test_dispersion.py`, matching the package's synthetic-data
convention (`test_ruler.py`): generates `f(mu)` from known D1/D2, checks
mode assignment and the fit recover them, and checks that one deliberately
displaced mode is both excluded from the fit and correctly flagged as an
AMX candidate. 5/5 pass.

Not verified against the real `20260813-180111_ring01/scan.h5` in the
session that wrote this script: the file is 250 MB and repeatedly timed
out transferring off the device for a from-the-cloud test run, and the
project's own `.venv` is unusable over the same bridge (its interpreter is
a symlink to a macOS-only path). Run it locally to confirm:
`python scripts/calc_d2.py data/20260813-180111_ring01/scan.h5`.

## Known simplifications

- Assumes a single mode family in the DUT spectrum (a mixed-family scan
  will raise from `assign_mode_numbers`'s collision check rather than being
  separated automatically).
- AMX location is the nearest surviving resonance, not interpolated between
  adjacent modes.
- Not persisted into `scan.h5` -- rerun the script to get the numbers
  again, consistent with the rest of the package's calibrated-data
  handling.
