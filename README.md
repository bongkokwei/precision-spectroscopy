# precision-spectroscopy

Swept-wavelength spectroscopy of a DUT, with a fibre-MZI auxiliary
interferometer supplying the frequency axis.

A tunable laser (Agilent/Keysight 8164B) sweeps continuously while a
four-channel scope (Rigol DHO4204) records the MZI fringes, an HCN/CO gas
cell, the DUT transmission and a trigger line. The MZI fringes — not the
laser's own wavelength readout — define the frequency axis, so sweep
nonlinearity does not matter.

## Install

```bash
pip install -e ".[hw,dev]"   # hw pulls the two instrument drivers from GitHub
pip install -e ".[dev]"      # processing only, no VISA stack needed
```

## Use

```bash
spectro check   configs/default.yaml   # validate settings, no hardware touched
spectro run     configs/default.yaml   # sweep and capture; writes raw only

spectro plot data/<run>/scan.h5                  # raw channels
spectro plot data/<run>/scan.h5 --sweep 2        # one forward sweep only, no flyback
spectro plot data/<run>/scan.h5 --phase          # MZI phase diagnostics
spectro plot data/<run>/scan.h5 --gas            # gas-cell line identification + anchor table

spectro fit  data/<run>/scan.h5                  # fit nu(phi) through the gas-cell anchors,
                                                  # write it into the file (run this before --resonances)
spectro plot data/<run>/scan.h5 --resonances     # DUT spectrum + fitted resonances
spectro plot data/<run>/scan.h5 --resonances --expected-fwhm-hz 2e8  # override the linewidth guess
```

`run` acquires and stops. The raw record spans the whole capture window,
including the parked-laser regions at each end where the MZI has no carrier,
and how to trim those is not settled — so building the frequency ruler is a
separate, explicit step (`spectro fit`) rather than something every capture
triggers.

Each run creates `data/<timestamp>_<dut_name>/` containing `scan.h5`,
`run.log`, `screenshot.png` and a copy of the config that produced them.
Nothing is encoded in filenames, so nothing can disagree with the data.

### Running a scan from a script

`spectro run` is a thin wrapper around two calls, so scripting your own
acquisition (batching several DUTs, say) is just:

```python
from spectro import config
from spectro.acquire import run

cfg = config.load("configs/default.yaml")   # also runs the preflight checks
h5_path = run(cfg, config_path="configs/default.yaml")
print(f"wrote {h5_path}")
```

Save that as e.g. `scripts/run_scan.py` and run it with the environment
`pip install -e ".[hw,dev]"` set up (`python scripts/run_scan.py`).

## How the frequency ruler is built

Six steps, one forward sweep at a time:

1. Unwrap the analytic-signal phase for that sweep (`ruler.sweep_phase`).
2. (same call as step 1 — the unwrap and the per-sweep windowing happen together).
3. Detect the gas-cell absorption dips, identify each against the
   H12C14N / 12CO / 13CO catalogue in `configs/default.yaml`, and read off
   the MZI phase at each one (`ruler.gas_anchors`, `plot --gas`).
4. Fit a weighted polynomial nu(phi) through those anchors — degree 2 by
   default, CO weighted roughly 2x HCN by the catalogue's own accuracy — and
   write the coefficients into the file under `ruler/sweep_<n>`
   (`ruler.fit_ruler`, `spectro fit`).
5. Map the DUT trace onto that absolute frequency axis using the stored fit
   (`ruler.dut_spectrum`).
6. Fit a Lorentzian to each resonance in the DUT spectrum
   (`ruler.fit_resonances`, `plot --resonances`).

`plot --resonances` reads back the fit `spectro fit` wrote rather than
recomputing it, so run `fit` first — it errors with a clear message if you
don't.

There is no constant-FSR frequency conversion left in the package —
`plot --phase` (via `sweep_phase`) stops at phase and never converts to
frequency. Steps 3-4 above are the only path to an absolute frequency axis,
by fitting nu(phi) through lines of known absolute frequency instead of
assuming a fixed FSR (fibre dispersion means there is no single FSR
accurate across a 100 nm sweep).

## Configuration

One YAML file, five sections: `scope` (channel roles and vertical setup),
`laser` (sweep range, speed, power), `process` (the band-pass applied before
the Hilbert transform), `run` (output directory, DUT name), and `gascell`
(species, their line catalogue in nm, linewidth and `accuracy_pm` — the
latter sets each species' weight in the step-4 fit). Channels are addressed
by *role* (`mzi`, `gascell`, `dut`, `trigger`), so processing never
hardcodes a channel number.

`process.bandpass_hz` is not optional: it must bracket the MZI fringe rate
over the whole sweep, or the unwrapped phase goes non-monotonic.

The sweep is always continuous and one-way. A two-way sweep gives a
non-monotonic phase ramp, which would need splitting at the turning point
for no benefit here.

## Preflight checks

`spectro check` (and `spectro run`, which loads the config the same way)
refuses to run when the record window is shorter than the sweep, because
that leaves the trace truncated — and that's cheaper to catch here than
after a full sweep and a multi-gigabyte file.

## Data format

`scan.h5`, one file per run:

```
/raw/ch1..ch4     float32, LZF-compressed
/ruler/sweep_<n>  attrs only -- coeffs_hz, domain_rad, degree, rms_hz,
                  n_anchors, dropped_high_j (written by `spectro fit`, one
                  group per fitted sweep)
attrs             t0_s, dt_s, n_samples, timestamp, both *IDN? strings,
                  and the complete config as YAML
```

Samples are `float32` because the scope digitises to 12 bits — `float64`
would double the file for no extra information. The time base is stored as
`t0_s`/`dt_s` rather than an array, saving a few hundred megabytes. Because
the config travels inside the file, any scan is reprocessable without its
original YAML.

There is currently no persisted, ready-to-reload DUT spectrum: `dut_spectrum()`
(behind `plot --resonances`) is computed in memory for the plot and not
written back into the file, so getting it again means rerunning `fit` +
`plot --resonances` (or calling `ruler.dut_spectrum()` yourself).

## Tests

```bash
pytest
```

Three things are checked against synthetic data rather than fixed
thresholds: the ruler against a synthetic *nonlinear* chirp (a linear one
would not distinguish a working ruler from a broken one), with noise
scaling as `sigma_nu = (sigma_phi / 2*pi) * FSR`; gas-cell dip finding, line
assignment and the nu(phi) polynomial fit against a synthetic gas cell and a
known quadratic; and resonance fitting against synthetic Lorentzian peaks
and dips.

## Known simplifications

- No cross-check of resonance linewidth/Q against a second, independent
  method — `fit_resonances` fits one Lorentzian per resonance and that's it.
- No persisted calibrated dataset (see Data format above) — every read of a
  frequency-calibrated DUT spectrum recomputes it.
- Acquisition assumes `DHO4204.capture()` exists with the signature used in
  the driver's example script; it is not in that repo's README.
