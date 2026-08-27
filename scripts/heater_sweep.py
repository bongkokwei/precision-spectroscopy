"""Heater-voltage sweep: track one resonance across a DC voltage range.

Sets a DC voltage on a Rigol DG952 (driving a heater), runs the existing
spectro pipeline at each step, extracts a frequency window around one
resonance, and plots the 2D transmission colormap (figure a).

Usage:
    python scripts/heater_sweep.py configs/default.yaml
    python scripts/heater_sweep.py configs/default.yaml --show
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import yaml

from spectro import acquire, config, ruler

log = logging.getLogger(__name__)


def _pick_target(resonances: list[dict], target_thz: float | None) -> dict:
    """Nearest resonance to target_thz, or deepest if target_thz is None."""
    if not resonances:
        raise RuntimeError("no resonance found in the cold scan")
    if target_thz is None:
        return min(resonances, key=lambda r: r["amp_hz"])
    target_hz = target_thz * 1e12
    return min(resonances, key=lambda r: abs(r["f0_hz"] - target_hz))


def _extract_window(
    freq_hz: np.ndarray,
    dut: np.ndarray,
    f0_hz: float,
    half_window_hz: float,
    detuning_grid_hz: np.ndarray,
) -> np.ndarray:
    """Interpolate the DUT onto a common detuning grid centred on f0_hz."""
    detuning = freq_hz - f0_hz
    mask = np.abs(detuning) <= half_window_hz
    if mask.sum() < 3:
        log.warning("fewer than 3 points in window; returning NaN")
        return np.full_like(detuning_grid_hz, np.nan)
    return np.interp(detuning_grid_hz, detuning[mask], dut[mask],
                     left=np.nan, right=np.nan)


def _plot(
    voltages: np.ndarray,
    detuning_ghz: np.ndarray,
    transmission: np.ndarray,
    out_path: Path,
    show: bool = False,
    dut_name: str = "",
) -> Path:
    if not show:
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
    ax.set_title(dut_name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    log.info("wrote %s", out_path)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to the YAML config")
    parser.add_argument("--show", action="store_true", help="open interactive plot")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    cfg = config.load(args.config)
    h = cfg["heater"]
    voltages = np.linspace(h["start_v"], h["stop_v"], h["steps"])
    half_window_hz = h["window_ghz"] * 1e9 / 2
    n_detuning = 2001  # ponytail: fixed grid, increase if resolution matters

    sweep_dir = (
        Path(cfg["run"]["dir"])
        / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_heater_sweep"
    )
    sweep_dir.mkdir(parents=True)
    log.info("sweep output: %s", sweep_dir)

    from rigol_dg952 import RigolDG952

    slices = []
    f0_cold_hz = None
    detuning_grid_hz = None
    run_dirs: list[Path] = []

    with RigolDG952(h["resource"]) as dg:
        ch = dg.channel(h["channel"])
        ch.set_load_ohm(h["load_ohm"])
        ch.apply_dc(voltages[0])
        ch.output_on()

        try:
            for i, v in enumerate(voltages):
                log.info("--- step %d/%d: %.3f V ---", i + 1, len(voltages), v)
                ch.apply_dc(v)
                time.sleep(h["settle_s"])

                h5_path = acquire.run(cfg, config_path=args.config,
                                      restart_scope=False)
                run_dirs.append(h5_path.parent)

                try:
                    ruler.fit_ruler(h5_path)
                    freq_hz, dut, _ = ruler.dut_spectrum(h5_path)
                except Exception as exc:
                    log.error("step %d failed: %s", i + 1, exc)
                    if f0_cold_hz is None:
                        raise RuntimeError(
                            "cold scan (step 1) failed; cannot continue"
                        ) from exc
                    slices.append(np.full(n_detuning, np.nan))
                    continue

                if f0_cold_hz is None:
                    _, _, resonances = ruler.find_resonances(h5_path)
                    target = _pick_target(resonances, h.get("target_thz"))
                    f0_cold_hz = target["f0_hz"]
                    detuning_grid_hz = np.linspace(
                        -half_window_hz, half_window_hz, n_detuning
                    )
                    log.info(
                        "cold resonance: %.6f THz (%.4f nm), FWHM %.1f MHz, Q %.0f",
                        f0_cold_hz / 1e12,
                        2.998e8 / f0_cold_hz * 1e9,
                        target["fwhm_hz"] / 1e6,
                        target["q"],
                    )

                slices.append(
                    _extract_window(freq_hz, dut, f0_cold_hz, half_window_hz,
                                    detuning_grid_hz)
                )
        finally:
            ch.apply_dc(0.0)
            ch.output_off()
            dg.local()

    transmission = np.array(slices)
    detuning_ghz = detuning_grid_hz / 1e9

    with h5py.File(sweep_dir / "sweep.h5", "w") as f:
        f.attrs["timestamp"] = datetime.now().isoformat(timespec="seconds")
        f.attrs["config_yaml"] = yaml.safe_dump(cfg, sort_keys=False)
        f.attrs["f0_cold_hz"] = f0_cold_hz
        f.create_dataset("voltages", data=voltages)
        f.create_dataset("detuning_ghz", data=detuning_ghz)
        f.create_dataset("transmission", data=transmission.astype(np.float32))
    log.info("wrote %s", sweep_dir / "sweep.h5")

    _plot(
        voltages, detuning_ghz, transmission,
        sweep_dir / "heater_sweep.png",
        show=args.show,
        dut_name=cfg["run"].get("dut_name", ""),
    )

    if not h.get("save_all_scans", False):
        for d in run_dirs:
            shutil.rmtree(d)
            log.info("removed %s", d)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
