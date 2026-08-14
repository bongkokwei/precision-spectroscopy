"""Configuration loading, preflight validation, and run-directory naming."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

log = logging.getLogger(__name__)


def load(path: str | Path, validate: bool = True) -> dict[str, Any]:
    """Load a YAML config and (by default) run the preflight checks."""
    cfg = yaml.safe_load(Path(path).read_text())
    if validate:
        preflight(cfg)
    return cfg


def sample_rate_hz(cfg: dict) -> float:
    """Scope sample rate implied by the requested record length and window."""
    return cfg["scope"]["points"] / cfg["scope"]["duration_s"]


def sweep_duration_s(cfg: dict) -> float:
    laser = cfg["laser"]
    return abs(laser["stop_nm"] - laser["start_nm"]) / laser["speed_nm_s"]


def nominal_lambda_nm(t_s, cfg: dict):
    """Wavelength the laser is nominally at, t seconds after the trigger.

    The dial reading, not a measurement: it assumes the ramp starts exactly at
    start_nm and holds speed_nm_s exactly. On the 20260813 record it lands
    within ~20 pm of the gas-cell lines over the whole 100 nm, which is far
    finer than the ~400 pm line spacing and so plenty to identify lines by --
    but ~20000x coarser than the catalogue, so it is never an anchor itself.
    """
    laser = cfg["laser"]
    direction = 1.0 if laser["stop_nm"] >= laser["start_nm"] else -1.0
    return laser["start_nm"] + direction * laser["speed_nm_s"] * np.asarray(t_s, float)


def gas_lines_nm(cfg: dict) -> dict[str, dict[str, float]]:
    """Gas-cell reference lines that fall inside the laser sweep.

    Returns {species: {branch_label: wavelength_nm}}, sorted by wavelength.
    Species with no line in range come back as an empty dict rather than
    being dropped, so a caller can tell "not in this sweep" from "not fitted".
    """
    lo, hi = sorted((cfg["laser"]["start_nm"], cfg["laser"]["stop_nm"]))
    return {
        name: dict(sorted(
            ((k, v) for k, v in spec["lines_nm"].items() if lo <= v <= hi),
            key=lambda kv: kv[1],
        ))
        for name, spec in cfg.get("gascell", {}).get("species", {}).items()
    }


def preflight(cfg: dict) -> None:
    """Raise before wasting a capture if the settings cannot produce good data.

    One failure mode is worth catching up front, because it costs a full
    sweep plus a multi-gigabyte file to discover the slow way: a record
    window shorter than the sweep, which leaves the trace truncated.
    """
    missing = {"scope", "laser", "process", "run"} - set(cfg)
    if missing:
        raise ValueError(f"config is missing top-level sections: {sorted(missing)}")

    for role in ("mzi", "dut"):
        if role not in cfg["scope"]["channels"]:
            raise ValueError(f"scope.channels must define the '{role}' role")

    window = cfg["scope"]["duration_s"]
    sweep = sweep_duration_s(cfg)
    if window < sweep:
        raise ValueError(
            f"scope.duration_s ({window:.1f} s) is shorter than the sweep "
            f"({sweep:.1f} s); the trace would be truncated"
        )

    log.info(
        "preflight OK: sweep %.1f s in a %.1f s window, %.2f MS/s",
        sweep, window, sample_rate_hz(cfg) / 1e6,
    )

    # Warning, not an error: the ruler still works without an absolute anchor,
    # you just cannot pin its origin afterwards.
    lines = gas_lines_nm(cfg)
    if lines and not any(lines.values()):
        log.warning("no gas-cell line falls inside the sweep; no absolute anchor")
    for name, in_range in lines.items():
        log.info("gas cell: %s contributes %d line(s) in range", name, len(in_range))


def new_run_dir(cfg: dict, config_path: str | Path | None = None) -> Path:
    """Create and return a timestamped run directory: data/<stamp>_<dut>/.

    One directory per run keeps the raw file, log, screenshot and the exact
    config that produced them together, so filenames never need to encode
    parameters.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = Path(cfg["run"]["dir"]) / f"{stamp}_{cfg['run']['dut_name']}"
    d.mkdir(parents=True, exist_ok=False)
    if config_path is not None:
        shutil.copy(config_path, d / "config.yaml")
    return d


def setup_logging(run_dir: Path) -> None:
    """Log to both the console and run.log inside the run directory."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "run.log"),
        ],
        force=True,
    )
