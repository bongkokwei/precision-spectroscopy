"""Refit every sweep under data/ with the band-pass measured from the trace.

Records taken before `bandpass_hz: auto` existed have an explicit band frozen
into them, so reprocessing one is a per-call override rather than a config edit
(see ruler.band_for). This is that override in a loop:

    ruler.fit_ruler(h5, sweep=n, bandpass_hz="auto")

One line per sweep, showing the band it settled on and the fit RMS before and
after. Note the "before" column is the only copy of the old number: fit_ruler
overwrites ruler/sweep_<n> in place, and the fit it replaces is kept nowhere.

Usage:
    python scripts/refit_all.py                 # everything under data/
    python scripts/refit_all.py data/2026081*   # or just these runs
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from spectro import plot, ruler
from spectro.sweeps import sweep_slices

log = logging.getLogger(__name__)


def n_sweeps(h5_path: Path) -> int:
    """Complete forward sweeps in a record, from the trigger channel alone."""
    t_s, traces, cfg = plot.load_h5(h5_path, roles=("trigger",))
    return len(sweep_slices(t_s, traces["trigger"], cfg))


def rms_before_mhz(h5_path: Path, sweep: int) -> float | None:
    """RMS of the fit already in the file, or None if it was never fitted."""
    try:
        return ruler.read_ruler_fit(h5_path, sweep)["rms_hz"] / 1e6
    except (ValueError, KeyError):
        return None


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    paths = argv if argv else ["data"]
    scans = sorted({s for p in paths for s in Path(p).glob("**/scan.h5")})
    if not scans:
        print(f"no scan.h5 found under {paths}", file=sys.stderr)
        return 2

    failed = 0
    for h5_path in scans:
        for sweep in range(n_sweeps(h5_path)):
            before = rms_before_mhz(h5_path, sweep)
            try:
                fit = ruler.fit_ruler(h5_path, sweep, bandpass_hz="auto")
            except Exception as exc:  # noqa: BLE001 -- one bad record must not
                # abandon the rest of the folder; the reason is printed instead.
                print(f"{h5_path.parent.name} sweep {sweep}: FAILED -- {exc}")
                failed += 1
                continue
            band = ruler.read_ruler_fit(h5_path, sweep)["params"]["bandpass_hz"]
            print(
                f"{h5_path.parent.name} sweep {sweep}: "
                f"band {band[0] / 1e3:.0f}-{band[1] / 1e3:.0f} kHz, "
                f"{fit['n_anchors']} anchors, RMS "
                f"{'--' if before is None else f'{before:.1f}'} -> "
                f"{fit['rms_hz'] / 1e6:.1f} MHz"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
