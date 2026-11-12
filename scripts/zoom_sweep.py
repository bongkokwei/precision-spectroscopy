"""Zoomed DUT spectra for every step of a heater sweep, to find the aux ring.

For each step this runs the same analysis as `spectro plot --resonances`
(ruler.find_resonances) and plots the DUT trace as returned, zoomed to --thz,
with the fitted resonances ticked. It then writes map.png: every fitted
resonance centre against heater voltage. The main ring and the broad family
barely move and stand as near-vertical columns; the aux ring walks, bending to
lower frequency as the heater power V^2/R rises.

Detection follows process.resonances, read as for any scan: the sweep folder's
config.yaml beats the copy frozen into each scan.h5. To look for a weaker or
broader ring in a sweep already captured, copy
configs/keysight_short_mzi_weak.yaml over <sweep_dir>/config.yaml and rerun.

Works on scripts/heater_sweep.py and scripts/aux_track.py folders captured
with save_all_scans: true. Writes everything to <sweep_dir>/zoom/.

Usage:
    python scripts/zoom_sweep.py data/<stamp>_heater_sweep
    python scripts/zoom_sweep.py data/<stamp>_heater_sweep --thz 189.14 190.34
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectro import ruler  # noqa: E402

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--thz", type=float, nargs=2, default=(188.0, 190.0),
                    metavar=("LO", "HI"), help="zoom range (default 188 190)")
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
    f0_thz, f0_v = [], []

    for i, (v, scan) in enumerate(zip(voltages[:n], scans)):
        try:
            freq_hz, dut, resonances = ruler.find_resonances(scan)
        except Exception as exc:  # noqa: BLE001 -- one bad step is skipped
            log.error("step %d (%s): %s", i + 1, scan.parent.name, exc)
            continue
        m = (freq_hz >= lo) & (freq_hz <= hi)
        if not m.any():
            log.error("step %d: spectrum %.3f-%.3f THz misses the zoom range", i + 1,
                      freq_hz[0] / 1e12, freq_hz[-1] / 1e12)
            continue
        f0 = [r["f0_hz"] / 1e12 for r in resonances if lo <= r["f0_hz"] <= hi]
        f0_thz += f0
        f0_v += [v] * len(f0)

        y = dut[m]
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(freq_hz[m] / 1e12, y, lw=0.4, color="0.3")
        ax.plot(f0, [y.max() + 0.05 * (y.max() - y.min())] * len(f0),
                ls="none", marker="v", ms=4, color="C1")
        ax.set_xlim(*args.thz)
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel("DUT (a.u.)")
        ax.set_title(f"step {i + 1:03d}  {v:.3f} V  ({scan.parent.name})  "
                     f"{len(f0)} resonance(s) in range")
        fig.tight_layout()
        fig.savefig(out / f"step_{i + 1:03d}_{v:.3f}V.png", dpi=150)
        plt.close(fig)
        log.info("step %d/%d: %.3f V, %d resonance(s) in range", i + 1, n, v, len(f0))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(f0_thz, f0_v, ls="none", marker="o", ms=3, color="C0")
    ax.set_xlim(*args.thz)
    ax.set_xlabel("Resonance frequency (THz)")
    ax.set_ylabel("Heater voltage (V)")
    ax.set_title(f"fitted resonances, {len(f0_thz)} over {n} steps")
    ax.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out / "map.png", dpi=200)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
