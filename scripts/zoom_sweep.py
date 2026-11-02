"""Zoomed DUT spectra for every step of a heater sweep, to find the aux ring.

The main ring and the broad family barely move with the heater; the aux ring
walks. So besides one zoomed plot per step, this writes a voltage-frequency
map divided by its per-frequency median over all steps: the stationary
families divide out to ~1 and the aux ring is what is left, as tilted lines.

Works on scripts/heater_sweep.py and scripts/aux_track.py folders captured
with save_all_scans: true. Writes everything to <sweep_dir>/zoom/.

Usage:
    python scripts/zoom_sweep.py data/<stamp>_heater_sweep
    python scripts/zoom_sweep.py data/<stamp>_heater_sweep --thz 189.0 190.0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from spectro import ruler  # noqa: E402

log = logging.getLogger(__name__)


def stationary_ratio(stack: np.ndarray) -> np.ndarray:
    """Each step over the median of all steps, per frequency bin."""
    return stack / np.nanmedian(stack, axis=0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--thz", type=float, nargs=2, default=(188.0, 190.0),
                    metavar=("LO", "HI"), help="zoom range (default 188 190)")
    ap.add_argument("--points", type=int, default=20000,
                    help="common grid size; 2 THz / 20000 = 100 MHz bins")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    d = args.sweep_dir
    h5 = next(p for p in (d / "sweep.h5", d / "aux_track.h5") if p.exists())
    with h5py.File(h5, "r") as f:
        voltages = f["voltages"][:]
        n = int(f.attrs["n_completed"])
    scans = sorted(d.glob("*/scan.h5"))[:n]
    if len(scans) < n:
        raise SystemExit(f"{n} steps but {len(scans)} scan.h5; was save_all_scans off?")

    out = d / "zoom"
    out.mkdir(exist_ok=True)
    lo, hi = (x * 1e12 for x in args.thz)
    grid = np.linspace(lo, hi, args.points)
    # ponytail: interp onto 100 MHz bins undersamples the ~200 MHz main-ring lines;
    # fine for spotting the aux ring, raise --points to resolve them.
    stack = np.full((n, len(grid)), np.nan)

    for i, (v, scan) in enumerate(zip(voltages[:n], scans)):
        try:
            freq_hz, dut, _ = ruler.dut_spectrum(scan)
        except Exception as exc:  # noqa: BLE001 -- one bad step keeps its NaN row
            log.error("step %d (%s): %s", i + 1, scan.parent.name, exc)
            continue
        m = (freq_hz >= lo) & (freq_hz <= hi)
        stack[i] = np.interp(grid, freq_hz, dut, left=np.nan, right=np.nan)

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.semilogy(freq_hz[m] / 1e12, np.clip(dut[m], 1e-4, None), lw=0.4, color="0.3")
        ax.set_xlim(*args.thz)
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel("DUT (arb., log)")
        ax.set_title(f"step {i + 1:03d}  {v:.3f} V  ({scan.parent.name})")
        fig.tight_layout()
        fig.savefig(out / f"step_{i + 1:03d}_{v:.3f}V.png", dpi=150)
        plt.close(fig)
        log.info("step %d/%d: %.3f V", i + 1, n, v)

    ext = [args.thz[0], args.thz[1], voltages[0], voltages[n - 1]]
    fig, (a, b) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    a.imshow(np.log10(np.clip(stack, 1e-4, None)), aspect="auto", origin="lower",
             extent=ext, cmap="inferno", interpolation="nearest")
    a.set_title("log10 DUT")
    ratio = np.log10(stationary_ratio(stack))
    lim = np.nanpercentile(np.abs(ratio), 99)
    b.imshow(ratio, aspect="auto", origin="lower", extent=ext, cmap="RdBu_r",
             vmin=-lim, vmax=lim, interpolation="nearest")
    b.set_title("log10(step / median over steps): stationary families cancel, aux ring remains")
    b.set_xlabel("Frequency (THz)")
    for ax in (a, b):
        ax.set_ylabel("Heater voltage (V)")
    fig.tight_layout()
    fig.savefig(out / "map.png", dpi=200)
    log.info("wrote %s", out)
    return 0


def _demo() -> None:
    """A fixed peak cancels; a moving one survives."""
    x = np.arange(100)
    stack = np.array([1 + 10 * (x == 50) + 10 * (x == 10 + 3 * k) for k in range(10)], float)
    r = stationary_ratio(stack)
    assert np.allclose(r[:, 50], 1)            # stationary
    assert all(r[k, 10 + 3 * k] > 5 for k in range(10))  # moving
    print("ok")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--demo"]:
        _demo()
    else:
        raise SystemExit(main())
