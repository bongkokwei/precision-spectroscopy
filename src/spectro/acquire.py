"""Acquisition: arm the scope, run one laser sweep, store the raw traces.

The scope is triggered on the laser's sweep-start pulse rather than
free-running. A free-run capture keeps the last duration_s before the scope
stops, which opens ~1 s after start_sweep() -- after the first sweep's trigger
edge -- so no sweep in the record was both anchored and complete. Instead the
sweep starts first (it runs continuously) and the scope latches on the next
rising edge, with PRETRIGGER_S of record before it.
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

PRETRIGGER_S = 0.5  # record kept before the edge; the rest must hold one sweep


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
    # Trigger on the sweep-start pulse, edge near the left of the record.
    # capture() sets the same scale again; it is set here so the offset,
    # which the scope clamps against the scale, lands after it.
    duration_s = scope_cfg["duration_s"]
    post_s = duration_s - PRETRIGGER_S
    if post_s < sweep_duration_s(cfg):
        raise ValueError(
            f"scope.duration_s = {duration_s} s leaves {post_s:.2f} s after the "
            f"trigger, less than the {sweep_duration_s(cfg):.2f} s sweep"
        )
    # At >= 50 ms/div the scope auto-enters roll mode, where there is no
    # trigger: SINGle never latches and a free-run keeps the last duration_s.
    if not scope.write_verified(
        ":TIMebase:ROLL OFF", ":TIMebase:ROLL?", lambda r: r.upper() in ("0", "OFF")
    ):
        raise RuntimeError(
            "could not turn off the scope's auto roll mode (:TIMebase:ROLL OFF "
            "not confirmed); without it the trigger is ignored at this timebase"
        )
    scope.trigger_edge(scope_cfg["channels"]["trigger"]["ch"], scope_cfg["trigger_level_v"], "POS")
    scope.timebase_scale(duration_s / 10)
    scope.timebase_offset(duration_s / 2 - PRETRIGGER_S)  # +ve = trigger left of centre
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


def run(cfg: dict, config_path: str | Path | None = None, restart_scope: bool = True) -> Path:
    """Execute one sweep and return the path to the written HDF5 file."""
    # Imported here so processing-only installs work without a VISA stack.
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
                wait_for_trigger=True,
                # up to one sweep + flyback for the next edge, then the record
                trigger_timeout=2 * sweep_duration_s(cfg) + scope_cfg["duration_s"] + 5,
            )
        finally:
            laser.stop_sweep()
            laser.laser_off()

        scope.screenshot(str(run_dir / "screenshot.png"))
        if restart_scope:
            # The scope's front end gets unreliable after a multi-gigabyte readout.
            scope.system_restart()

    h5_path = run_dir / "scan.h5"
    write_h5(h5_path, t, data, cfg, idn)
    return h5_path
