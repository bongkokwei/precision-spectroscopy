"""Heater sweep on the aux ring: record every resonance at every heater step.

No tracking. Each step stores all fitted resonances, plus the detuning of the
nearest one from a fixed reference, heater.target_thz. As the ring heats, that
nearest peak walks off and the next mode takes its place, so the detuning is a
sawtooth with period one FSR: unwrap it afterwards, e.g.

    np.unwrap(detuning_hz, period=fsr_hz)

which is valid as long as each step shifts the mode by less than FSR/2.

Writes <run.dir>/<timestamp>_aux_track/aux_track.h5 after every step, and
aux_track.png at the end (wrapped detuning against heater power).

Usage:
    python scripts/aux_track.py configs/default.yaml
    python scripts/aux_track.py configs/default.yaml --show
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import yaml

from spectro import acquire, config, ruler

log = logging.getLogger(__name__)

FIELDS = ("f0_hz", "fwhm_hz", "q", "amp_hz")


def nearest_detuning(resonances: list[dict], ref_hz: float) -> float:
    """f0 - ref_hz of the resonance nearest ref_hz; NaN if there are none."""
    if not resonances:
        return float("nan")
    f0 = np.array([r["f0_hz"] for r in resonances])
    return float(f0[np.argmin(np.abs(f0 - ref_hz))] - ref_hz)


def _save(path: Path, voltages, load_ohm, per_step: list[list[dict]], ref_hz, cfg) -> None:
    v = voltages[: len(per_step)]
    flat = [(i, r) for i, res in enumerate(per_step) for r in res]
    with h5py.File(path, "w") as f:
        f.attrs["timestamp"] = datetime.now().isoformat(timespec="seconds")
        f.attrs["config_yaml"] = yaml.safe_dump(cfg, sort_keys=False)
        f.attrs["ref_hz"] = ref_hz
        f.attrs["n_completed"] = len(per_step)
        # one row per heater step
        f.create_dataset("voltages", data=v)
        f.create_dataset("power_mw", data=v**2 / load_ohm * 1e3)
        f.create_dataset("detuning_hz", data=[nearest_detuning(r, ref_hz) for r in per_step])
        # one row per resonance; `step` indexes the arrays above
        f.create_dataset("step", data=np.array([i for i, _ in flat], dtype=int))
        for k in FIELDS:
            f.create_dataset(k, data=np.array([r[k] for _, r in flat], dtype=float))


def _plot(power_mw, detuning_hz, out: Path, show: bool, title: str) -> None:
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(power_mw, detuning_hz / 1e9, "o-")
    ax.set_xlabel("Heater power (mW)")
    ax.set_ylabel("Nearest resonance - ref (GHz), wrapped")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    log.info("wrote %s", out)
    if show:
        plt.show()
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to the YAML config")
    parser.add_argument("--show", action="store_true", help="open interactive plot")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s")

    cfg = config.load(args.config)
    h = cfg["heater"]
    if h.get("target_thz") is None:
        raise SystemExit("heater.target_thz must be set: it is the reference frequency")
    ref_hz = h["target_thz"] * 1e12
    voltages = np.linspace(h["start_v"], h["stop_v"], h["steps"])

    out_dir = Path(cfg["run"]["dir"]) / f"{datetime.now():%Y%m%d-%H%M%S}_aux_track"
    out_dir.mkdir(parents=True)
    cfg = {**cfg, "run": {**cfg["run"], "dir": str(out_dir)}}
    out_h5 = out_dir / "aux_track.h5"
    log.info("output: %s", out_dir)

    from rigol_dg952 import RigolDG952

    def set_heater(v: float, init: bool) -> None:
        with RigolDG952(h["resource"]) as dg:
            ch = dg.channel(h["channel"])
            if init:
                ch.set_load_ohm(h["load_ohm"])
            ch.apply_dc(v)
            if init:
                ch.output_on()

    per_step: list[list[dict]] = []
    run_dirs: list[Path] = []

    try:
        for i, v in enumerate(voltages):
            log.info("--- step %d/%d: %.3f V ---", i + 1, len(voltages), v)
            set_heater(v, init=(i == 0))
            time.sleep(h["settle_s"])
            h5_path = acquire.run(cfg, config_path=args.config, restart_scope=False)
            run_dirs.append(h5_path.parent)

            try:
                ruler.fit_ruler(h5_path)
                _, _, res = ruler.find_resonances(h5_path)
            except Exception as exc:
                log.error("step %d failed: %s", i + 1, exc)
                res = []  # step kept, with NaN detuning, so indices stay aligned
            per_step.append(res)
            log.info("%d resonance(s), nearest at %+.3f GHz from ref",
                     len(res), nearest_detuning(res, ref_hz) / 1e9)
            _save(out_h5, voltages, h["load_ohm"], per_step, ref_hz, cfg)
    finally:
        with RigolDG952(h["resource"]) as dg:
            ch = dg.channel(h["channel"])
            ch.apply_dc(0.0)
            ch.output_off()
            dg.local()

    v = voltages[: len(per_step)]
    _plot(v**2 / h["load_ohm"] * 1e3,
          np.array([nearest_detuning(r, ref_hz) for r in per_step]),
          out_dir / "aux_track.png", args.show, cfg["run"].get("dut_name", ""))

    if not h.get("save_all_scans", False):
        for handler in logging.root.handlers[:]:  # release run.log on Windows
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logging.root.removeHandler(handler)
        for d in run_dirs:
            shutil.rmtree(d)
    return 0


def _demo() -> None:
    """Self-check: a 50 GHz-FSR comb redshifting past the ref, then unwrapped."""
    fsr, ref = 50e9, 190e12
    shifts = -np.arange(20) * 7e9                   # 7 GHz/step < FSR/2
    det = [nearest_detuning([{"f0_hz": ref + 3e9 + s + k * fsr}
                             for k in range(-5, 6)], ref) for s in shifts]
    assert max(abs(d) for d in det) <= fsr / 2      # wrapped into one FSR
    assert np.allclose(np.unwrap(det, period=fsr), 3e9 + shifts)
    assert np.isnan(nearest_detuning([], ref))
    print("ok")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--demo"]:
        _demo()
    else:
        raise SystemExit(main())
