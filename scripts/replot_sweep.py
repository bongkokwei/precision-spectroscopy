"""Re-plot an existing heater sweep from its sweep.h5.

Usage:
    python scripts/replot_sweep.py path/to/sweep.h5
    python scripts/replot_sweep.py path/to/sweep.h5 --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", help="path to sweep.h5")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    h5_path = Path(args.h5)
    with h5py.File(h5_path, "r") as f:
        voltages = f["voltages"][:]
        detuning_ghz = f["detuning_ghz"][:]
        raw = f["transmission"][:]
        n = int(f.attrs["n_completed"])
        dut_name = f.attrs.get("config_yaml", "")

    import yaml
    cfg = yaml.safe_load(dut_name) if dut_name else {}
    title = cfg.get("run", {}).get("dut_name", h5_path.stem)

    transmission = raw[:n]
    baseline = np.nanmedian(transmission[0])
    if baseline != 0:
        transmission = transmission / baseline

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    mesh = ax.pcolormesh(
        detuning_ghz, voltages[:n], transmission,
        vmin=0.0, vmax=1.0, cmap="inferno", shading="nearest",
    )
    ax.set_xlabel("Detuning from cold main resonance (GHz)")
    ax.set_ylabel("Heater voltage (V)")
    fig.colorbar(mesh, ax=ax, label="Transmission")
    ax.set_title(title)
    fig.tight_layout()

    out = h5_path.parent / "heater_sweep.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    if args.show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
