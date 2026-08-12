"""Acquisition: arm the scope, run one laser sweep, store the raw traces.

Synchronisation is software-timed. start_sweep() returns as soon as the SCPI
command lands and the sweep then runs on the instrument, so the capture window
opens a few milliseconds into the sweep. That jitter does not matter: the MZI
ruler defines the frequency axis from the data itself, not from wall-clock time.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import yaml

from .config import new_run_dir, sample_rate_hz, setup_logging, sweep_duration_s

log = logging.getLogger(__name__)


def _configure_laser(laser, cfg: dict) -> None:
    laser_cfg = cfg["laser"]
    laser.set_power_unit(laser_cfg["power_unit"])
    laser.set_power(laser_cfg["power"], unit=laser_cfg["power_unit"])
    laser.set_output_path(laser_cfg["output_path"])
    laser.configure_sweep(
        laser_cfg["start_nm"],
        laser_cfg["stop_nm"],
        speed_nm_s=laser_cfg["speed_nm_s"],
        cycles=laser_cfg["cycles"],
        mode="continuous",
        repeat="oneway",
    )
    status = laser.check_sweep_params()
    if status != "OK":
        raise RuntimeError(f"sweep configuration rejected by the laser: {status}")


def _configure_scope(scope, cfg: dict) -> list[int]:
    scope_cfg = cfg["scope"]
    scope.acquire_memory_depth(scope_cfg["memory_depth"])
    channels = []
    for role, ch_cfg in scope_cfg["channels"].items():
        ch = ch_cfg["ch"]
        scope.channel_enable(ch, True)
        scope.channel_scale(ch, ch_cfg["scale"])
        scope.channel_offset(ch, ch_cfg["offset"])
        channels.append(ch)
        log.info(
            "ch%d = %s (%.3g V/div, offset %.3g V)",
            ch,
            role,
            ch_cfg["scale"],
            ch_cfg["offset"],
        )
    return sorted(channels)


def write_h5(
    path: Path, t: np.ndarray, data: dict[int, np.ndarray], cfg: dict, idn: dict
) -> None:
    """Write raw traces plus everything needed to interpret them later.

    Samples are stored as float32 (the scope digitises to 12 bits, so float64
    doubles the file for no extra information) and the time base as t0/dt
    attributes rather than an array, which saves a few hundred megabytes.
    """
    t = np.asarray(t, dtype=float)
    with h5py.File(path, "w") as f:
        f.attrs["timestamp"] = datetime.now().isoformat(timespec="seconds")
        f.attrs["config_yaml"] = yaml.safe_dump(cfg, sort_keys=False)
        f.attrs["t0_s"] = t[0]
        f.attrs["dt_s"] = float(np.median(np.diff(t)))
        f.attrs["n_samples"] = len(t)
        for name, value in idn.items():
            f.attrs[f"idn_{name}"] = value
        raw = f.create_group("raw")
        for ch, y in data.items():
            raw.create_dataset(
                f"ch{ch}",
                data=np.asarray(y, dtype=np.float32),
                chunks=True,
                compression="lzf",  # fast; scope noise barely compresses anyway
            )
    log.info("wrote %s (%.2f GB)", path, path.stat().st_size / 1e9)


def run(cfg: dict, config_path: str | Path | None = None) -> Path:
    """Execute one sweep and return the path to the written HDF5 file."""
    # Imported here so `spectro process` works without a VISA stack installed.
    from agilent_8164b import Agilent8164B
    from rigol_dho4204 import DHO4204

    run_dir = new_run_dir(cfg, config_path)
    setup_logging(run_dir)
    log.info("run directory: %s", run_dir)

    scope_cfg = cfg["scope"]
    laser_cfg = cfg["laser"]
    log.info(
        "sweep %.1f-%.1f nm at %.1f nm/s (%.1f s), capturing %.1f s at %.2f MS/s",
        laser_cfg["start_nm"],
        laser_cfg["stop_nm"],
        laser_cfg["speed_nm_s"],
        sweep_duration_s(cfg),
        scope_cfg["duration_s"],
        sample_rate_hz(cfg) / 1e6,
    )

    with (
        DHO4204(
            scope_cfg["resource"],
        ) as scope,
        Agilent8164B(
            laser_cfg["resource"],
            slot=laser_cfg["slot"],
            channel=laser_cfg["channel"],
        ) as laser,
    ):
        idn = {"scope": scope.idn(), "laser": laser.identify()}
        log.info("scope: %s", idn["scope"])
        log.info("laser: %s", idn["laser"])

        channels = _configure_scope(scope, cfg)
        _configure_laser(laser, cfg)

        laser.laser_on()
        try:
            laser.start_sweep()
            t, data = scope.capture(
                duration=scope_cfg["duration_s"],
                points=scope_cfg["points"],
                channels=channels,
                wait_for_trigger=False,
            )
        finally:
            laser.stop_sweep()
            laser.laser_off()

        scope.screenshot(str(run_dir / "screenshot.png"))
        # The scope's front end gets unreliable after a multi-gigabyte readout.
        scope.system_restart()

    h5_path = run_dir / "scan.h5"
    write_h5(h5_path, t, data, cfg, idn)
    return h5_path
