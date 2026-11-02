"""Heater sweep on the aux ring: follow the aux resonance step by step.

Unlike heater_sweep.py, which looks at a fixed window around the cold
resonance, this re-picks the resonance at every step: the nearest fitted
resonance to where the aux mode is predicted to be (linear extrapolation from
the last two steps), within half of heater.window_ghz. Starting point is
heater.target_thz in the cold scan.

Writes <run.dir>/<timestamp>_aux_track/aux_track.h5 after every step, and
aux_track.png at the end (aux shift and Q against heater power).

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


def predict(f0s: list[float]) -> float:
    """Where the aux mode should be next: linear step from the last two good f0s."""
    good = [f for f in f0s if np.isfinite(f)]
    if len(good) >= 2:
        return 2 * good[-1] - good[-2]
    return good[-1]


def pick(resonances: list[dict], guess_hz: float, gate_hz: float) -> dict | None:
    """Nearest resonance to guess_hz, or None if nothing lies within gate_hz."""
    # ponytail: nearest-neighbour; near an avoided crossing with a main-ring
    # mode it can hop onto the main mode. Check the plot, shrink window_ghz or
    # add more steps if it does.
    near = [r for r in resonances if abs(r["f0_hz"] - guess_hz) <= gate_hz]
    return min(near, key=lambda r: abs(r["f0_hz"] - guess_hz)) if near else None


def _save(path: Path, voltages, load_ohm, rows: list[dict], f0_cold_hz, cfg) -> None:
    v = voltages[: len(rows)]
    with h5py.File(path, "w") as f:
        f.attrs["timestamp"] = datetime.now().isoformat(timespec="seconds")
        f.attrs["config_yaml"] = yaml.safe_dump(cfg, sort_keys=False)
        f.attrs["f0_cold_hz"] = f0_cold_hz
        f.attrs["n_completed"] = len(rows)
        f.create_dataset("voltages", data=v)
        f.create_dataset("power_mw", data=v**2 / load_ohm * 1e3)
        for k in FIELDS:
            f.create_dataset(k, data=np.array([r[k] for r in rows], dtype=float))


def _plot(power_mw, f0_hz, q, f0_cold_hz, out: Path, show: bool, title: str) -> None:
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    a1.plot(power_mw, (f0_hz - f0_cold_hz) / 1e9, "o-")
    a1.set_ylabel("Aux shift from cold (GHz)")
    a1.set_title(title)
    a2.plot(power_mw, q, "o-")
    a2.set_ylabel("Loaded Q")
    a2.set_xlabel("Heater power (mW)")
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
        raise SystemExit("heater.target_thz must point at the aux resonance")
    voltages = np.linspace(h["start_v"], h["stop_v"], h["steps"])
    gate_hz = h["window_ghz"] * 1e9 / 2

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

    rows: list[dict] = []
    f0s = [h["target_thz"] * 1e12]  # seed; replaced by the fitted cold f0
    f0_cold_hz = float("nan")
    run_dirs: list[Path] = []
    nan_row = dict.fromkeys(FIELDS, float("nan"))

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
                if i == 0:
                    raise RuntimeError("cold scan failed; cannot seed tracking") from exc
                log.error("step %d failed: %s", i + 1, exc)
                res = []

            r = pick(res, predict(f0s), gate_hz)
            if r is None:
                if i == 0:
                    raise RuntimeError(f"no resonance within {gate_hz/1e9:.1f} GHz "
                                       f"of target_thz={h['target_thz']}")
                log.warning("lost aux resonance at %.3f V", v)
                r = nan_row
            else:
                log.info("aux f0 %.6f THz, FWHM %.1f MHz, Q %.0f",
                         r["f0_hz"] / 1e12, r["fwhm_hz"] / 1e6, r["q"])
            if i == 0:
                f0s, f0_cold_hz = [], r["f0_hz"]
            f0s.append(r["f0_hz"])
            rows.append({k: r[k] for k in FIELDS})
            _save(out_h5, voltages, h["load_ohm"], rows, f0_cold_hz, cfg)
    finally:
        with RigolDG952(h["resource"]) as dg:
            ch = dg.channel(h["channel"])
            ch.apply_dc(0.0)
            ch.output_off()
            dg.local()

    v = voltages[: len(rows)]
    _plot(v**2 / h["load_ohm"] * 1e3,
          np.array([r["f0_hz"] for r in rows]), np.array([r["q"] for r in rows]),
          f0_cold_hz, out_dir / "aux_track.png", args.show,
          cfg["run"].get("dut_name", ""))

    if not h.get("save_all_scans", False):
        for handler in logging.root.handlers[:]:  # release run.log on Windows
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logging.root.removeHandler(handler)
        for d in run_dirs:
            shutil.rmtree(d)
    return 0


def _demo() -> None:
    """Self-check: aux mode walking 3 GHz/step past a fixed main-ring comb."""
    main_ring = [{"f0_hz": f} for f in np.arange(-5, 6) * 100e9]
    f0s, gate = [10e9], 10e9
    for step in range(1, 6):
        res = main_ring + [{"f0_hz": 10e9 + 3e9 * step**1.5}]
        r = pick(res, predict(f0s), gate)
        assert r is not None and r["f0_hz"] == res[-1]["f0_hz"], (step, r)
        f0s.append(r["f0_hz"])
    assert pick(main_ring, 50e9, gate) is None          # out of gate -> lost
    assert predict([1.0, float("nan"), 3.0]) == 5.0     # NaN steps skipped
    print("ok")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--demo"]:
        _demo()
    else:
        raise SystemExit(main())
