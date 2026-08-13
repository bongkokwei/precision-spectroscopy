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


def load_h5(
    h5_path: str | Path, roles: tuple[str, ...] | None = None
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Read a scan file written by `spectro run`.

    Args:
        h5_path: Path to scan.h5.
        roles: Only read these roles. A full record is ~200 MB per channel, so
            a caller that needs two of them should say so rather than pay for
            four. None reads everything.

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
            role: f["raw"][name][:]
            for name in f["raw"]
            for role in [role_of.get(int(name[2:]), name)]
            if roles is None or role in roles
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
        out_path: Where to write the figure. Defaults to scan.png beside the
            data, or scan_sweep_<n>.png when a single sweep is plotted, so a
            sweep never overwrites the full-record view.
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
    stem = h5_path.stem

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
        n = sweep % len(slices)  # resolve negatives so the filename is stable
        stem = f"{stem}_sweep_{n}"
        title += f"  sweep {n + 1}/{len(slices)}"

    out_path = Path(out_path) if out_path else h5_path.with_name(f"{stem}.png")
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


def plot_phase(
    h5_path: str | Path,
    sweep: int = 0,
    out_path: str | Path | None = None,
    show: bool = False,
    zoom_fringes: float = 8.0,
    bandpass_hz: tuple[float, float] | None = None,
) -> Path:
    """Diagnostic plot of the MZI phase for one forward sweep.

    Four panels, top to bottom: a few fringes of the band-passed trace with
    cos(phi) laid over them, which is the check that the phase actually tracks
    the fringes; total fringes phi/2pi; the departure of phi from a straight
    line, which is the laser's sweep nonlinearity; and the instantaneous fringe
    rate, which is proportional to the tuning rate.

    Args:
        h5_path: Path to scan.h5.
        sweep: Which forward sweep to use.
        out_path: Figure path; defaults to scan_phase_<sweep>.png beside the data.
        show: Open an interactive window as well as saving.
        zoom_fringes: Width of the top panel's window, in fringes.
        bandpass_hz: Passed to ruler.sweep_phase; overrides the file's setting.

    Returns:
        Path to the written figure.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import ruler

    h5_path = Path(h5_path)
    t_s, phi, mzi, cfg = ruler.sweep_phase(h5_path, sweep, bandpass_hz)
    fringes = phi / (2 * np.pi)

    # phi is smooth and monotonic, so plain striding is safe here -- unlike the
    # raw fringes, which alias and need the min/max envelope.
    step = max(len(phi) // 20_000, 1)
    ts, fs_dec = t_s[::step], fringes[::step]
    line = np.polyval(np.polyfit(ts, fs_dec, 1), ts)
    rate = np.diff(fs_dec) / np.diff(ts)  # fringes/s == Hz

    # Top panel: the middle of the sweep, a handful of fringes wide.
    mid = len(phi) // 2
    n_zoom = int(zoom_fringes / (rate[len(rate) // 2] * (t_s[1] - t_s[0])))
    z = slice(mid, mid + n_zoom)

    fig, axes = plt.subplots(4, 1, figsize=(12, 9))
    ax = axes[0]
    x = mzi[z].astype(float)
    x = (x - x.mean()) / (0.5 * np.ptp(x))
    t_us = (t_s[z] - t_s[mid]) * 1e6
    ax.plot(t_us, x, lw=0.9, color=PALETTE[0], label="MZI (normalised)")
    ax.plot(t_us, np.cos(phi[z]), lw=0.9, ls="--", color=PALETTE[5], label=r"$\cos\phi$")
    ax.set_xlabel("Time from mid-sweep (µs)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    axes[1].plot(ts, fs_dec / 1e6, lw=1.0, color=PALETTE[1])
    axes[1].set_ylabel(r"$\phi/2\pi$" + "\n(Mfringe)")

    axes[2].plot(ts, (fs_dec - line) / 1e3, lw=1.0, color=PALETTE[2])
    axes[2].set_ylabel("Departure from\nlinear (kfringe)")

    axes[3].plot(ts[:-1], rate / 1e3, lw=0.5, alpha=0.5, color=PALETTE[3])
    # The raw rate carries a real ~200 Hz tuning ripple, so overlay a smoothed
    # trend rather than let the ripple hide the drift underneath it.
    k = max(len(rate) // 50, 1)
    smooth = np.convolve(rate, np.ones(k) / k, mode="valid")
    axes[3].plot(
        ts[k // 2 : k // 2 + len(smooth)], smooth / 1e3, lw=1.2, color=PALETTE[5]
    )
    axes[3].set_ylabel("Fringe rate\n(kHz)")
    axes[3].set_xlabel("Time from trigger (s)")

    for a in axes:
        a.grid(True, alpha=0.25, lw=0.5)
        a.margins(x=0.005)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

    bp = cfg["process"].get("bandpass_hz")
    fig.suptitle(
        f"{h5_path.parent.name}  sweep {sweep}  "
        f"band-pass {bp[0] / 1e3:.0f}–{bp[1] / 1e3:.0f} kHz"
        if bp
        else f"{h5_path.parent.name}  sweep {sweep}  no band-pass",
        fontsize=10,
    )
    fig.align_ylabels(axes)
    fig.tight_layout()
    out_path = Path(out_path) if out_path else h5_path.with_name(
        f"{h5_path.stem}_phase_{sweep}.png"
    )
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
