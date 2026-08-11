# precision-spectroscopy

Swept-wavelength spectroscopy of a DUT, with a fibre-MZI auxiliary
interferometer supplying the frequency axis.

A tunable laser (Agilent/Keysight 8164B) sweeps continuously while a
four-channel scope (Rigol DHO4204) records the MZI fringes, an HCN gas cell,
the DUT transmission and a trigger line. The MZI fringes — not the laser's
own wavelength readout — define the frequency axis, so sweep nonlinearity
does not matter.

## Install

```bash
pip install -e ".[hw,dev]"   # hw pulls the two instrument drivers from GitHub
pip install -e ".[dev]"      # processing only, no VISA stack needed
```

## Use

```bash
spectro check   configs/default.yaml   # validate settings, no hardware touched
spectro run     configs/default.yaml   # sweep, capture, process
spectro process data/<run>/scan.h5     # reprocess an existing scan
```

Each run creates `data/<timestamp>_<dut_name>/` containing `scan.h5`,
`run.log`, `screenshot.png` and a copy of the config that produced them.
Nothing is encoded in filenames, so nothing can disagree with the data.

## Configuration

One YAML file, four sections: `scope` (channel roles and vertical setup),
`laser` (sweep range, speed, power), `process` (MZI free spectral range),
`run` (output directory, DUT name). Channels are addressed by *role*
(`mzi`, `gascell`, `dut`, `trigger`), so processing never hardcodes a
channel number.

The sweep is always continuous and one-way. A two-way sweep gives a
non-monotonic phase ramp, which would need splitting at the turning point
for no benefit here.

### The one number you must calibrate

`process.mzi_fsr_hz` is the measured free spectral range of your fibre MZI.
It sets the absolute scale of every frequency in the output. A nominal fibre
length is never the real optical delay — measure it, don't compute it from
metres of fibre.

## Preflight checks

`spectro check` refuses to run when either of two conditions fails, because
each otherwise costs a full sweep and a multi-gigabyte file to discover:

1. **Record window shorter than the sweep** — the trace is truncated.
2. **Sample rate below 10x the MZI fringe rate** — the phase unwrap aliases
   and the ruler is silently, plausibly wrong.

The fringe rate follows from the chirp rate and the interferometer delay:

```
dnu/dt   = v * c / lambda0^2          (1.25 THz/s at 10 nm/s near 1550 nm)
f_fringe = (dnu/dt) / FSR             (~30 kHz for a 5 m delay, FSR 41 MHz)
```

The default config runs 82x oversampled.

## How the ruler works

The MZI trace is `I(t) = A + B*cos(phi(t))` with `phi = 2*pi*(nu - nu0)/FSR`.
Unwrapping the analytic-signal phase counts fringes, and each fringe is one
FSR of optical frequency regardless of how the laser actually swept:

```
phi(t)      = unwrap(angle(hilbert(I - mean(I))))
nu(t) - nu0 = (phi(t) - phi(0)) / (2*pi) * FSR
```

The DUT channel is then interpolated onto a uniform frequency grid.

**What is accurate: the spacing.** Measured at ~100 Hz over a 50 GHz
synthetic span.

**What is not: the origin.** It carries the Hilbert transform's edge
transient (a few MHz) plus the laser's absolute wavelength error (~GHz on
real hardware). Both are constants, and both are what the gas cell is for.
The gas cell is recorded and resampled onto the same axis but not yet used —
anchoring it against NIST SRM 2519a is the next step.

## Data format

`scan.h5`, one file per run:

```
/raw/ch1..ch4     float32, LZF-compressed
/proc/freq_hz     uniform optical frequency grid
/proc/dut         DUT transmission on that grid
/proc/gascell     gas cell on that grid
attrs             t0_s, dt_s, n_samples, timestamp, both *IDN? strings,
                  and the complete config as YAML
```

Samples are `float32` because the scope digitises to 12 bits — `float64`
would double the file for no extra information. The time base is stored as
`t0_s`/`dt_s` rather than an array, saving a few hundred megabytes. Because
the config travels inside the file, any scan is reprocessable without its
original YAML.

## Tests

```bash
pytest
```

The ruler is checked against a synthetic *nonlinear* chirp — a linear one
would not distinguish a working ruler from a broken one. The noise test
asserts that jitter scales linearly with injected MZI noise, following
`sigma_nu = (sigma_phi / 2*pi) * FSR`, rather than checking against a
hand-picked threshold.

## Known simplifications

- FSR is treated as constant across the sweep. Fibre dispersion shifts it by
  ~1e-4 across the C band; a polynomial `FSR(nu)` correction goes in
  `ruler.relative_freq_hz` if you need sub-kHz accuracy at the band edges.
- No absolute frequency anchoring (gas cell recorded, not fitted).
- No resonance fitting or integrated dispersion (`D_int`).
- Acquisition assumes `DHO4204.capture()` exists with the signature used in
  the driver's example script; it is not in that repo's README.
