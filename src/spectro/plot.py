"""Load a scan.h5 and plot the raw channels, one stacked panel per role.

A full record is tens of millions of samples per channel, which matplotlib will
not draw in reasonable time. Points are therefore reduced with a min/max
envelope rather than plain striding: striding aliases the MZI fringes into a
moire pattern, whereas the envelope keeps the visible amplitude honest.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import yaml

log = logging.getLogger(__name__)

# Okabe-Ito, colourblind-safe and easier on the eye than the matplotlib default.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]


def load_h5(h5_path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Read a scan file written by `spectro run`.

    Args:
        h5_path: Path to scan.h5.

    Returns:
        (t_s, traces, cfg) where traces maps role name -> samples. Channels with
        no role in the config fall back to the key "chN".
    """
    with h5py.File(h5_path, "r") as f:
        cfg = yaml.safe_load(f.attrs["config_yaml"])
        n = int(f.attrs["n_samples"])
        t_s = f.attrs["t0_s"] + f.attrs["dt_s"] * np.arange(n)
        role_of = {c["ch"]: role for role, c in cfg["scope"]["channels"].items()}
        traces = {
            role_of.get(int(name[2:]), name): f["raw"][name][:] for name in f["raw"]
        }
    return t_s, traces, cfg


def _envelope(t_s: np.ndarray, y: np.ndarray, max_points: int):
    """Reduce to <= max_points by keeping the min and max of each bin."""
    n_bins = max_points // 2
    if len(y) <= max_points:
        return t_s, y
    trimmed = (len(y) // n_bins) * n_bins
    yb = y[:trimmed].reshape(n_bins, -1)
    tb = t_s[:trimmed].reshape(n_bins, -1)
    # Interleave (min, max) per bin so the trace is drawn as one continuous line.
    y_env = np.empty(2 * n_bins)
    y_env[0::2], y_env[1::2] = yb.min(axis=1), yb.max(axis=1)
    return np.repeat(tb.mean(axis=1), 2), y_env


def plot_scan(
    h5_path: str | Path,
    out_path: str | Path | None = None,
    max_points: int = 20_000,
    show: bool = False,
    sweep: int | None = None,
) -> Path:
    """Plot every raw channel of a scan file and save the figure.

    Args:
        h5_path: Path to scan.h5.
        out_path: Where to write the figure; defaults to scan.png beside the data.
        max_points: Points drawn per channel after envelope reduction.
        show: Open an interactive window as well as saving.
        sweep: Plot only this forward sweep, indexed from the trigger edges
            (negative indexes from the end). None plots the whole record,
            flyback included.

    Returns:
        Path to the written figure.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h5_path = Path(h5_path)
    out_path = Path(out_path) if out_path else h5_path.with_suffix(".png")

    t_s, traces, cfg = load_h5(h5_path)
    title = h5_path.parent.name
    if sweep is not None:
        from . import sweeps as _sweeps

        slices = _sweeps.sweep_slices(t_s, traces["trigger"], cfg)
        if not slices:
            raise ValueError("no complete sweep found in this record")
        sl = slices[sweep]
        t_s = t_s[sl] - t_s[sl.start]
        traces = {k: v[sl] for k, v in traces.items()}
        title += f"  sweep {sweep % len(slices) + 1}/{len(slices)}"
    fig, axes = plt.subplots(
        len(traces), 1, sharex=True, figsize=(12, 2.0 * len(traces))
    )
    axes = np.atleast_1d(axes)

    for ax, (role, y), colour in zip(axes, traces.items(), PALETTE):
        ax.plot(*_envelope(t_s, y, max_points), lw=0.7, color=colour)
        ax.set_ylabel(f"{role}\n(V)")
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.margins(x=0.005)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[-1].set_xlabel("Time (s)")
    fig.align_ylabels(axes)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("wrote %s", out_path)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def _demo() -> None:
    """Self-check: the envelope keeps the peak-to-peak of a dense sine."""
    t = np.linspace(0, 1, 1_000_000)
    y = 3.0 * np.sin(2 * np.pi * 5000 * t)
    _, y_env = _envelope(t, y, 20_000)
    assert len(y_env) <= 20_000, len(y_env)
    assert abs(np.ptp(y_env) - 6.0) < 0.01, np.ptp(y_env)
    t_short, y_short = _envelope(t[:100], y[:100], 20_000)
    assert len(y_short) == 100, "short records must pass through untouched"
    print("ok")


if __name__ == "__main__":
    _demo()
