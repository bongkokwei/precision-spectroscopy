"""Heater sweep on the aux ring: record every resonance at every heater step.

No tracking. Each step stores all fitted resonances, plus the detuning of the
nearest one from a fixed reference, heater.target_thz. As the ring heats, that
nearest peak walks off and the next mode takes its place, so the detuning is a
sawtooth with period one FSR: unwrap it afterwards, e.g.

    np.unwrap(detuning_hz, period=fsr_hz)

which is valid as long as each step shifts the mode by less than FSR/2.

Writes <run.dir>/<timestamp>_aux_track/aux_track.h5 after every step, and
at the end aux_track.png (wrapped detuning against heater power) and
aux_map.png (every step's full DUT spectrum stacked: heater voltage against
optical frequency, block-averaged into N_BINS bins over the laser sweep, in
dB). Each step's full-resolution spectrum, in dB, goes to
spectra/step_<n>_<V>V.png as that step is measured.

The sweep folder holds the one editable config.yaml (step folders get none),
which spectro.plot.load_h5 falls back to for every step. To re-analyse with
new process.* knobs -- cleaning, resonance detection -- or a new
heater.target_thz, edit that file and run --reprocess; it needs
save_all_scans: true at capture time.

Usage:
    python scripts/aux_track.py configs/default.yaml
    python scripts/aux_track.py configs/default.yaml --show
    python scripts/aux_track.py --reprocess data/<stamp>_aux_track
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

FIELDS = ("f0_hz", "fwhm_hz", "q", "amp_hz")
# ponytail: 20k bins over 1500-1600 nm is ~0.6 GHz each, so a ~0.2 GHz dip is
# averaged ~3x shallower but still visible; raise it if dips wash out (the h5
# grows by 4 bytes x steps per bin)
N_BINS = 20_000


def edges_hz(cfg: dict) -> np.ndarray:
    """Bin edges of the stacked map: the laser sweep's span, in optical frequency."""
    c = 299_792_458.0
    lo, hi = c / (cfg["laser"]["stop_nm"] * 1e-9), c / (cfg["laser"]["start_nm"] * 1e-9)
    return np.linspace(lo, hi, N_BINS + 1)


def db(y) -> np.ndarray:
    """10 log10 |y|: the baseline-divided trace may sit at -1 (negative volts)."""
    with np.errstate(divide="ignore"):
        return 10 * np.log10(np.abs(y))


def _plot_step(freq_hz, dut, out: Path, title: str) -> None:
    from matplotlib.figure import Figure  # no pyplot: leaves the --show backend alone

    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()
    ax.plot(freq_hz / 1e12, db(dut), lw=0.5)
    ax.set_xlabel("Optical frequency (THz)")
    ax.set_ylabel("Transmission (dB)")
    ax.set_title(title)
    fig.tight_layout()
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)


def bin_mean(freq_hz: np.ndarray, dut: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean of dut in each frequency bin; NaN where the scan has no samples."""
    total, _ = np.histogram(freq_hz, edges, weights=dut)
    count, _ = np.histogram(freq_hz, edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        return total / count


def nearest_detuning(resonances: list[dict], ref_hz: float) -> float:
    """f0 - ref_hz of the resonance nearest ref_hz; NaN if there are none."""
    if not resonances:
        return float("nan")
    f0 = np.array([r["f0_hz"] for r in resonances])
    return float(f0[np.argmin(np.abs(f0 - ref_hz))] - ref_hz)


def _save(path: Path, voltages, load_ohm, per_step: list[list[dict]], rows, ref_hz, cfg) -> None:
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
        # stacked spectra: transmission[step, k] is the mean over bin k
        f.create_dataset("edges_hz", data=edges_hz(cfg))
        f.create_dataset("transmission", data=np.array(rows, dtype=np.float32))
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


def _plot_map(voltages, edges, rows, out: Path, show: bool, title: str) -> None:
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    centres_thz = (edges[:-1] + edges[1:]) / 2e12
    mesh = ax.pcolormesh(centres_thz, voltages, db(np.asarray(rows)),
                         cmap="inferno", shading="nearest")
    ax.set_xlabel("Optical frequency (THz)")
    ax.set_ylabel("Heater voltage (V)")
    fig.colorbar(mesh, ax=ax, label="Transmission (dB)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    log.info("wrote %s", out)
    if show:
        plt.show()
    plt.close(fig)


def _analyse(h5_path: Path, edges: np.ndarray, out_dir: Path, i: int, v: float,
             title: str) -> tuple[list[dict], np.ndarray]:
    """One step's resonances and binned spectrum, plus its dB plot under
    out_dir/spectra; ([], NaNs) if analysis fails."""
    try:
        freq_hz, dut, resonances = ruler.find_resonances(h5_path)
        _plot_step(freq_hz, dut, out_dir / "spectra" / f"step_{i:03d}_{v:.3f}V.png",
                   f"{title}  step {i}, {v:.3f} V")
        return resonances, bin_mean(freq_hz, dut, edges)
    except Exception as exc:
        log.error("%s: %s", h5_path, exc)
        return [], np.full(len(edges) - 1, np.nan)  # step kept so indices stay aligned


def _finish(out_dir: Path, voltages, per_step, rows, ref_hz, cfg, show: bool) -> None:
    load_ohm = cfg["heater"]["load_ohm"]
    title = cfg["run"].get("dut_name", "")
    _save(out_dir / "aux_track.h5", voltages, load_ohm, per_step, rows, ref_hz, cfg)
    v = voltages[: len(per_step)]
    _plot(v**2 / load_ohm * 1e3,
          np.array([nearest_detuning(r, ref_hz) for r in per_step]),
          out_dir / "aux_track.png", show, title)
    _plot_map(v, edges_hz(cfg), rows, out_dir / "aux_map.png", show, title)


def reprocess(sweep_dir: Path, show: bool = False) -> None:
    """Rebuild aux_track.h5/.png from the saved per-step scans.

    Knobs come from sweep_dir/config.yaml (process.* via load_h5's parent
    lookup, heater.target_thz here); the ruler fit stored in each scan.h5 is
    reused, since cleaning and resonance fitting come after it.
    """
    with h5py.File(sweep_dir / "aux_track.h5", "r") as f:
        voltages = f["voltages"][:]
        cfg = yaml.safe_load(f.attrs["config_yaml"])
        ref_hz = float(f.attrs["ref_hz"])
    sidecar = sweep_dir / "config.yaml"
    if sidecar.exists():
        cfg = yaml.safe_load(sidecar.read_text())
        ref_hz = cfg["heater"]["target_thz"] * 1e12
    scans = sorted(sweep_dir.glob("*/scan.h5"))[: len(voltages)]
    if len(scans) < len(voltages):
        raise SystemExit(f"{len(voltages)} steps but {len(scans)} scan.h5 under "
                         f"{sweep_dir}; was save_all_scans off?")
    per_step, rows, edges = [], [], edges_hz(cfg)
    for i, h5_path in enumerate(scans):
        res, row = _analyse(h5_path, edges, sweep_dir, i, voltages[i],
                            cfg["run"].get("dut_name", ""))
        per_step.append(res)
        rows.append(row)
        log.info("step %d/%d: %d resonance(s), nearest at %+.3f GHz", i + 1,
                 len(scans), len(per_step[-1]), nearest_detuning(per_step[-1], ref_hz) / 1e9)
    _finish(sweep_dir, voltages, per_step, rows, ref_hz, cfg, show)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", help="path to the YAML config")
    parser.add_argument("--reprocess", type=Path, metavar="SWEEP_DIR",
                        help="re-analyse a saved sweep instead of measuring")
    parser.add_argument("--show", action="store_true", help="open interactive plot")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s")

    if args.reprocess:
        reprocess(args.reprocess, args.show)
        return 0
    if not args.config:
        parser.error("give a config to measure, or --reprocess SWEEP_DIR")

    cfg = config.load(args.config)
    h = cfg["heater"]
    if h.get("target_thz") is None:
        raise SystemExit("heater.target_thz must be set: it is the reference frequency")
    ref_hz = h["target_thz"] * 1e12
    voltages = np.linspace(h["start_v"], h["stop_v"], h["steps"])

    out_dir = Path(cfg["run"]["dir"]) / f"{datetime.now():%Y%m%d-%H%M%S}_aux_track"
    out_dir.mkdir(parents=True)
    shutil.copy(args.config, out_dir / "config.yaml")  # the one editable copy
    cfg = {**cfg, "run": {**cfg["run"], "dir": str(out_dir)}}
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
    rows: list[np.ndarray] = []
    edges = edges_hz(cfg)
    run_dirs: list[Path] = []

    try:
        for i, v in enumerate(voltages):
            log.info("--- step %d/%d: %.3f V ---", i + 1, len(voltages), v)
            set_heater(v, init=(i == 0))
            time.sleep(h["settle_s"])
            # config_path=None: no per-step config.yaml, so the sweep folder's wins
            h5_path = acquire.run(cfg, config_path=None, restart_scope=False)
            run_dirs.append(h5_path.parent)
            try:
                ruler.fit_ruler(h5_path)
            except Exception as exc:
                log.error("step %d ruler fit failed: %s", i + 1, exc)
            res, row = _analyse(h5_path, edges, out_dir, i, v, cfg["run"].get("dut_name", ""))
            per_step.append(res)
            rows.append(row)
            log.info("%d resonance(s), nearest at %+.3f GHz from ref",
                     len(per_step[-1]), nearest_detuning(per_step[-1], ref_hz) / 1e9)
            _save(out_dir / "aux_track.h5", voltages, h["load_ohm"], per_step, rows, ref_hz, cfg)
    finally:
        with RigolDG952(h["resource"]) as dg:
            ch = dg.channel(h["channel"])
            ch.apply_dc(0.0)
            ch.output_off()
            dg.local()

    _finish(out_dir, voltages, per_step, rows, ref_hz, cfg, args.show)

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
    row = bin_mean(np.array([1.0, 1.5, 2.5]), np.array([2.0, 4.0, 7.0]),
                   np.array([0.0, 1.0, 2.0, 3.0]))   # first bin empty
    assert np.isnan(row[0]) and np.allclose(row[1:], [3.0, 7.0])
    assert np.allclose(db([-1.0, 0.1, -0.01]), [0, -10, -20])
    print("ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--demo"]:
        _demo()
    else:
        raise SystemExit(main())
