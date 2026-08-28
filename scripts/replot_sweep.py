"""Re-plot an existing heater sweep, optionally around a different resonance.

Without --target-thz, re-plots from sweep.h5 (fast, no reprocessing).
With --target-thz, re-extracts from per-step scan.h5 files (needs save_all_scans).

Usage:
    python scripts/replot_sweep.py path/to/sweep_dir
    python scripts/replot_sweep.py path/to/sweep_dir --target-thz 189.8 --show
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import yaml

from spectro import ruler

log = logging.getLogger(__name__)


def _load_sweep_h5(sweep_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    h5 = sweep_dir / "sweep.h5"
    with h5py.File(h5, "r") as f:
        voltages = f["voltages"][:]
        detuning_ghz = f["detuning_ghz"][:]
        raw = f["transmission"][:int(f.attrs["n_completed"])]
        cfg = yaml.safe_load(f.attrs.get("config_yaml", "{}"))
    return voltages, detuning_ghz, raw, cfg.get("run", {}).get("dut_name", "")


def _reextract(
    sweep_dir: Path, target_thz: float, window_ghz: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    h5 = sweep_dir / "sweep.h5"
    with h5py.File(h5, "r") as f:
        voltages = f["voltages"][:]
        n = int(f.attrs["n_completed"])
        cfg = yaml.safe_load(f.attrs.get("config_yaml", "{}"))

    scan_dirs = sorted(
        d for d in sweep_dir.iterdir()
        if d.is_dir() and (d / "scan.h5").exists()
    )
    if len(scan_dirs) < n:
        raise FileNotFoundError(
            f"need {n} scan dirs, found {len(scan_dirs)}; was save_all_scans off?"
        )

    half_w = window_ghz * 1e9 / 2
    n_det = 2001
    detuning_grid_hz = np.linspace(-half_w, half_w, n_det)
    f0_hz = None
    slices = []

    for i, d in enumerate(scan_dirs[:n]):
        h5_path = d / "scan.h5"
        try:
            freq_hz, dut, _ = ruler.dut_spectrum(h5_path)
        except Exception as exc:
            log.warning("step %d failed: %s", i + 1, exc)
            slices.append(np.full(n_det, np.nan))
            continue

        if f0_hz is None:
            _, _, resonances = ruler.find_resonances(h5_path)
            if not resonances:
                raise RuntimeError("no resonances in cold scan")
            target_hz = target_thz * 1e12
            pick = min(resonances, key=lambda r: abs(r["f0_hz"] - target_hz))
            f0_hz = pick["f0_hz"]
            log.info(
                "target resonance: %.6f THz, FWHM %.1f MHz",
                f0_hz / 1e12, pick["fwhm_hz"] / 1e6,
            )

        detuning = freq_hz - f0_hz
        mask = np.abs(detuning) <= half_w
        if mask.sum() < 3:
            slices.append(np.full(n_det, np.nan))
            continue
        slices.append(np.interp(
            detuning_grid_hz, detuning[mask], dut[mask],
            left=np.nan, right=np.nan,
        ))

    title = cfg.get("run", {}).get("dut_name", "")
    return voltages[:n], detuning_grid_hz / 1e9, np.array(slices), title


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", help="path to the sweep directory")
    parser.add_argument("--target-thz", type=float, default=None,
                        help="re-extract around this resonance (THz)")
    parser.add_argument("--window-ghz", type=float, default=20.0,
                        help="frequency window width (default 20 GHz)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sweep_dir = Path(args.sweep_dir)

    if args.target_thz is not None:
        voltages, detuning_ghz, transmission, title = _reextract(
            sweep_dir, args.target_thz, args.window_ghz,
        )
    else:
        voltages, detuning_ghz, transmission, title = _load_sweep_h5(sweep_dir)

    baseline = np.nanmedian(transmission[0])
    if baseline != 0:
        transmission = transmission / baseline

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    mesh = ax.pcolormesh(
        detuning_ghz, voltages, transmission,
        vmin=0.0, vmax=1.0, cmap="inferno", shading="nearest",
    )
    ax.set_xlabel("Detuning from cold main resonance (GHz)")
    ax.set_ylabel("Heater voltage (V)")
    fig.colorbar(mesh, ax=ax, label="Transmission")
    if args.target_thz is not None:
        title = f"{title}  ({args.target_thz:.3f} THz)"
    ax.set_title(title)
    fig.tight_layout()

    out = sweep_dir / "heater_sweep.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    if args.show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
