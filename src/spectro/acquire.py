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

PRETRIGGER_S = 0.5  # DSOX4024A record kept before the trigger edge


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


def _configure_keysight(scope, cfg: dict) -> list[int]:
    """DSOX4024A counterpart of _configure_scope; same capture window, same timing."""
    scope_cfg = cfg["scope"]
    duration_s = scope_cfg["duration_s"]

    post_s = duration_s - PRETRIGGER_S
    if post_s < sweep_duration_s(cfg):
        raise ValueError(
            f"scope.duration_s = {duration_s} s leaves {post_s:.2f} s after the "
            f"trigger, less than the {sweep_duration_s(cfg):.2f} s sweep"
        )

    scope.write(":TIMebase:MODE MAIN")  # roll mode has no trigger for :SINGle to latch
    # :POINts is total memory, split per channel pair (4 Mpts -> 2 Mpts/ch with all four on).
    # A numeric value enters Digitizer mode: record = points / rate.
    # :TIMebase:SCALe is not sent: 1.1 s/div got -222 in Digitizer mode (cause unknown;
    # the guide says time/div is independent of the record there).
    scope.write(f":ACQuire:POINts {scope_cfg['memory_depth']}")
    scope.write(
        f":ACQuire:SRATe {scope_cfg['points'] / duration_s:g}"
    )  # points = per channel
    scope.check_error()

    channels = []
    for role, ch_cfg in scope_cfg["channels"].items():
        ch = ch_cfg["ch"]
        scope.set_channel(ch, scale_v=ch_cfg["scale"], offset_v=ch_cfg["offset"])
        channels.append(ch)
        log.info(
            "ch%d = %s (%.3g V/div, offset %.3g V)",
            ch,
            role,
            ch_cfg["scale"],
            ch_cfg["offset"],
        )

    # Trigger on the laser's sweep-start TTL, edge PRETRIGGER_S into the record so
    # sweeps.rising_edges sees the low-to-high step and a full sweep after it.
    trig_ch = scope_cfg["channels"]["trigger"]["ch"]
    trig = scope_cfg["trigger"]
    scope.set_trigger_edge(
        f"CHANnel{trig_ch}", trig["level_v"], trig["slope"], sweep="NORMal"
    )
    # ponytail: +position puts the trigger left of centre; flip the sign if the edge lands late in the record
    scope.write(f":TIMebase:POSition {duration_s / 2 - PRETRIGGER_S:g}")
    scope.check_error()  # a refused setting otherwise only shows up as a timeout

    rate = float(
        scope.query(":ACQuire:SRATe?")
    )  # the scope rounds to a rate it supports
    # Not :TIMebase:RANGe?: that reports the run-mode record (capped at 1 Mpts);
    # _capture_keysight uses :SINGle, which gets the full points/ch.
    record_s = scope_cfg["points"] / rate
    log.info("DSOX4024A: %.4g Sa/s, %.3g s record", rate, record_s)
    if record_s < PRETRIGGER_S + sweep_duration_s(cfg):
        raise ValueError(
            f"scope record is {record_s:.3g} s at {rate:.4g} Sa/s, shorter than the "
            f"{PRETRIGGER_S + sweep_duration_s(cfg):.2f} s pretrigger + sweep"
        )
    return sorted(channels)


def _capture_keysight(
    scope, channels: list[int]
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    scope.digitise(
        *channels, single=True
    )  # :SINGle: full memory; waits for the CH trigger edge
    # VIEW is per waveform source. The default MAIN returns only the on-screen part;
    # ALL (Digitizer mode only) returns the whole record. get_waveform leaves VIEW alone.
    for ch in channels:
        scope.write(f":WAVeform:SOURce CHANnel{ch}")
        scope.write(":WAVeform:VIEW ALL")
    scope.check_error()
    traces = {ch: scope.get_waveform(ch) for ch in channels}
    t = traces[channels[0]][0]
    log.info("captured %d points/ch over %.2f s", len(t), t[-1] - t[0])
    return t, {ch: v for ch, (_, v) in traces.items()}


def _capture_keysight(
    scope, channels: list[int]
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    scope.digitise(*channels, single=True)
    # VIEW is per waveform source; ALL (Digitizer mode only) returns the whole record,
    # not just the on-screen part. get_waveform doesn't touch it.
    for ch in channels:
        scope.write(f":WAVeform:SOURce CHANnel{ch}")
        scope.write(":WAVeform:VIEW ALL")
    traces = {ch: scope.get_waveform(ch) for ch in channels}
    t = traces[channels[0]][0]
    log.info("captured %d points/ch over %.2f s", len(t), t[-1] - t[0])
    return t, {ch: v for ch, (_, v) in traces.items()}


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


def run(
    cfg: dict, config_path: str | Path | None = None, restart_scope: bool = True
) -> Path:
    """Execute one sweep and return the path to the written HDF5 file."""
    # Imported here so processing-only installs work without a VISA stack.
    from agilent_8164b import Agilent8164B

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

    driver = scope_cfg.get("driver", "dho4204")
    if driver == "dho4204":
        from rigol_dho4204 import DHO4204

        scope_dev = DHO4204(scope_cfg["resource"])
    elif driver == "dsox4024a":
        from keysight_dsox4024a import DSOX4024A

        # The sweep starts before the scope arms and repeats (cycles: 0), so wait up to
        # one sweep + flyback for the next edge, then the record, plus slack.
        scope_dev = DSOX4024A(
            scope_cfg["resource"],
            timeout=2 * sweep_duration_s(cfg) + scope_cfg["duration_s"] + 30,
        )
    else:
        raise ValueError(
            f"scope.driver must be 'dho4204' or 'dsox4024a', got {driver!r}"
        )
    keysight = driver == "dsox4024a"

    with (
        scope_dev as scope,
        Agilent8164B(
            laser_cfg["resource"],
            slot=laser_cfg["slot"],
            channel=laser_cfg["channel"],
        ) as laser,
    ):
        idn = {
            "scope": scope.identify() if keysight else scope.idn(),
            "laser": laser.identify(),
        }
        log.info("scope: %s", idn["scope"])
        log.info("laser: %s", idn["laser"])

        channels = (_configure_keysight if keysight else _configure_scope)(scope, cfg)
        _configure_laser(laser, cfg)

        laser.laser_on()
        try:
            laser.start_sweep()
            if keysight:
                t, data = _capture_keysight(scope, channels)
            else:
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
        if restart_scope and not keysight:
            # The scope's front end gets unreliable after a multi-gigabyte readout.
            scope.system_restart()

    h5_path = run_dir / "scan.h5"
    write_h5(h5_path, t, data, cfg, idn)
    return h5_path
