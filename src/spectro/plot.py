"""Load a scan.h5 and plot the raw channels, one stacked panel per role.

Traces are plotted at full resolution -- no pre-reduction -- so that with
`show=True` matplotlib's normal interactive zoom/pan on the toolbar reveals
the exact samples rather than a pre-decimated stand-in. A full record is
tens of millions of samples per channel, so the first draw and save can take
a while; if that becomes a problem, plot a single `sweep` instead of the
whole record.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import yaml

from . import config

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
        # The run folder's config.yaml is the editable copy: its process block
        # beats the one frozen at capture, so re-analysing a run is an edit to
        # that file. Hardware sections stay frozen -- they say what was captured.
        sidecar = Path(h5_path).with_name("config.yaml")
        if sidecar.exists():
            process = (yaml.safe_load(sidecar.read_text()) or {}).get("process")
            if process:
                if bad := config._unparsed_numbers(process):
                    raise ValueError(
                        f"{sidecar}: process.{bad[0][0]} is the string {bad[0][1]!r}; "
                        f"write exponents with a sign, e.g. 2.0e+5"
                    )
                cfg["process"] = process
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


def plot_scan(
    h5_path: str | Path,
    out_path: str | Path | None = None,
    show: bool = False,
    sweep: int | None = None,
) -> Path:
    """Plot every raw channel of a scan file and save the figure.

    Args:
        h5_path: Path to scan.h5.
        out_path: Where to write the figure. Defaults to scan.png beside the
            data, or scan_sweep_<n>.png when a single sweep is plotted, so a
            sweep never overwrites the full-record view.
        show: Open an interactive window as well as saving. Use the
            toolbar's zoom/pan to inspect the full-resolution trace.
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
        ax.plot(t_s, y, lw=0.7, color=colour)
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

    Five panels, top to bottom: a few fringes of the band-passed trace with
    cos(phi) laid over them, which is the check that the phase actually tracks
    the fringes; total fringes phi/2pi; the departure of phi from a straight
    line, which is the laser's sweep nonlinearity; the instantaneous fringe
    rate, which is proportional to the tuning rate; and a spectrogram of the
    raw MZI trace with the band-pass edges drawn on it.

    The last two are the pair worth reading together. The fringe-rate panel is
    differentiated from the filtered phase, so it can only ever show something
    inside the band -- it cannot tell you the band is wrong. The spectrogram
    never sees the filter, so it can. Its ridge is repeated on the fringe-rate
    panel for exactly that comparison: where the two part company, the filter
    moved the answer.

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

    from . import config, ruler

    h5_path = Path(h5_path)
    t_s, phi, mzi, cfg = ruler.sweep_phase(h5_path, sweep, bandpass_hz)
    fringes = phi / (2 * np.pi)
    # Resolved by sweep_phase, so this is a pair of numbers even when the config
    # asked for `auto`.
    bp = cfg["process"].get("bandpass_hz")

    # The one measurement here that does not depend on the band-pass: taken from
    # the raw trace, so where it disagrees with dphi/dt below, the filter -- not
    # the laser -- is what moved.
    fs_hz = 1.0 / float(t_s[1] - t_s[0])
    knobs = config.params(cfg, "fringe_rate")
    f_hz, t_seg, spec = ruler.mzi_spectrogram(
        mzi, fs_hz, knobs["nperseg"], tuple(knobs["search_hz"])
    )
    ridge_hz = f_hz[spec.argmax(axis=0)]

    # phi is smooth and monotonic, so plain striding is safe here -- unlike
    # the raw fringes/gascell/DUT traces plotted elsewhere in this module,
    # which are left at full resolution instead and rely on the interactive
    # window's zoom to inspect them, since striding would alias them.
    step = max(len(phi) // 20_000, 1)
    ts, fs_dec = t_s[::step], fringes[::step]
    line = np.polyval(np.polyfit(ts, fs_dec, 1), ts)
    rate = np.diff(fs_dec) / np.diff(ts)  # fringes/s == Hz

    # Top panel: the middle of the sweep, a handful of fringes wide.
    mid = len(phi) // 2
    n_zoom = int(zoom_fringes / (rate[len(rate) // 2] * (t_s[1] - t_s[0])))
    z = slice(mid, mid + n_zoom)

    fig, axes = plt.subplots(5, 1, figsize=(12, 11.5))
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
        ts[k // 2 : k // 2 + len(smooth)], smooth / 1e3, lw=1.2, color=PALETTE[5],
        label=r"$\dot\phi/2\pi$ (post-filter)",
    )
    axes[3].plot(
        t_seg, ridge_hz / 1e3, lw=1.0, ls=":", color=PALETTE[0],
        label="spectrogram ridge (raw)",
    )
    axes[3].set_ylabel("Fringe rate\n(kHz)")
    axes[3].legend(loc="upper left", fontsize=8, frameon=False)

    # Bottom panel: where the fringe energy actually is. The band-pass edges are
    # drawn on it because "does the ridge stay between the dashed lines over the
    # whole sweep" is the entire question the band has to answer.
    ax = axes[4]
    with np.errstate(divide="ignore"):
        db = 10.0 * np.log10(spec / spec.max())
    mesh = ax.pcolormesh(
        t_seg, f_hz / 1e3, db, vmin=-60.0, vmax=0.0, cmap="magma", shading="nearest"
    )
    if bp:
        for edge in bp:
            ax.axhline(edge / 1e3, lw=1.0, ls="--", color=PALETTE[2])
    ax.set_ylabel("Spectrogram\n(kHz)")
    ax.set_xlabel("Time from trigger (s)")
    # Inset rather than fig.colorbar: a colourbar of its own would shrink this
    # panel alone and knock the five time axes out of alignment.
    cax = ax.inset_axes([0.80, 0.16, 0.18, 0.035])
    fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cax.tick_params(labelsize=7, colors="white", length=2, pad=1)
    cax.set_title("dB re peak", fontsize=7, color="white", pad=3)

    for a in axes[:4]:  # not the spectrogram: a grid over a heatmap only hides it
        a.grid(True, alpha=0.25, lw=0.5)
    for a in axes:
        a.margins(x=0.005)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

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


def plot_gas(
    h5_path: str | Path,
    sweep: int = 0,
    out_path: str | Path | None = None,
    show: bool = False,
    anchors: list[dict] | None = None,
    **kwargs,
) -> Path:
    """Diagnostic plot of the gas-cell line identification for one sweep.

    Two panels: the gascell trace against nominal wavelength with every
    identified line ticked and coloured by species, and the residual of each
    against its catalogue wavelength. The residual panel is the one to read --
    it should be a single tight band a few tens of pm wide, sloping gently with
    the laser's speed error. A point off that band is a misassignment.

    Args:
        h5_path: Path to scan.h5.
        sweep: Which forward sweep to use.
        out_path: Figure path; defaults to scan_gas_<sweep>.png beside the data.
        show: Open an interactive window as well as saving. Use the
            toolbar's zoom/pan to inspect the full-resolution trace.
        anchors: Result of a ruler.gas_anchors() call you have already made.
            Worth passing: recomputing them re-runs the Hilbert transform over
            the whole sweep.
        **kwargs: Passed to ruler.gas_anchors (snr, tol_nm, resolution_pm...).

    Returns:
        Path to the written figure.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import config, ruler
    from .sweeps import sweep_slices

    h5_path = Path(h5_path)
    if anchors is None:
        anchors = ruler.gas_anchors(h5_path, sweep, **kwargs)
    if not anchors:
        raise ValueError("no gas-cell line identified in this sweep; nothing to plot")

    t_s, traces, cfg = load_h5(h5_path, roles=("gascell", "trigger"))
    sl = sweep_slices(t_s, traces["trigger"], cfg)[sweep]
    lam = config.nominal_lambda_nm(t_s[sl] - t_s[sl.start], cfg)

    species = sorted({a["species"] for a in anchors})
    colour = dict(zip(species, PALETTE))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(lam, traces["gascell"][sl], lw=0.7, color="0.35")
    span = np.ptp(traces["gascell"][sl])
    for a in anchors:
        axes[0].plot(
            a["nominal_lambda_nm"], np.max(traces["gascell"][sl]) + 0.03 * span,
            marker="v", ms=4, color=colour[a["species"]],
        )
    axes[0].set_ylabel("Gas cell\n(V)")

    for s in species:
        pts = [a for a in anchors if a["species"] == s]
        axes[1].plot(
            [a["nominal_lambda_nm"] for a in pts], [a["residual_pm"] for a in pts],
            ls="none", marker="o", ms=4, color=colour[s], label=f"{s} ({len(pts)})",
        )
    axes[1].axhline(0.0, lw=0.8, color="0.6")
    axes[1].set_ylabel("Nominal − catalogue\n(pm)")
    axes[1].set_xlabel("Nominal wavelength (nm)")
    axes[1].legend(loc="best", fontsize=8, frameon=False, ncol=len(species))

    for ax in axes:
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.margins(x=0.005)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle(
        f"{h5_path.parent.name}  sweep {sweep}  {len(anchors)} anchors", fontsize=10
    )
    fig.align_ylabels(axes)
    fig.tight_layout()
    out_path = Path(out_path) if out_path else h5_path.with_name(
        f"{h5_path.stem}_gas_{sweep}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("wrote %s", out_path)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_resonances(
    h5_path: str | Path,
    sweep: int = 0,
    out_path: str | Path | None = None,
    show: bool = False,
    freq_hz: np.ndarray | None = None,
    dut: np.ndarray | None = None,
    resonances: list[dict] | None = None,
    **kwargs,
) -> Path:
    """Diagnostic plot of the DUT spectrum and its fitted resonances (steps 5-6).

    Two panels: the DUT transmission against absolute frequency with each
    fitted resonance centre ticked, and Q vs frequency below it -- the panel
    to read for whether Q trends with wavelength (bend loss, say) or is flat.

    Args:
        h5_path: Path to scan.h5.
        sweep: Which forward sweep to use.
        out_path: Figure path; defaults to scan_resonances_<sweep>.png beside
            the data.
        show: Open an interactive window as well as saving. Use the
            toolbar's zoom/pan to inspect the full-resolution trace.
        freq_hz, dut, resonances: Result of a ruler.find_resonances() call you
            have already made. Worth passing all three: a full sweep is
            25 M samples per channel, so recomputing dut_spectrum a second
            time roughly doubles peak memory on top of re-running the
            Hilbert transform and every resonance fit.
        **kwargs: Passed to ruler.find_resonances (expected_fwhm_hz, snr...).

    Returns:
        Path to the written figure.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import ruler

    h5_path = Path(h5_path)
    if freq_hz is None or dut is None or resonances is None:
        freq_hz, dut, resonances = ruler.find_resonances(h5_path, sweep, **kwargs)
    if not resonances:
        raise ValueError("no resonance identified in this sweep; nothing to plot")

    freq_thz = freq_hz / 1e12
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(freq_thz, dut, lw=0.7, color=PALETTE[0])
    span = np.ptp(dut)
    for r in resonances:
        axes[0].plot(
            r["f0_hz"] / 1e12, np.max(dut) + 0.05 * span,
            marker="v", ms=4, color=PALETTE[1],
        )
    axes[0].set_ylabel("DUT\n(a.u.)")

    axes[1].plot(
        [r["f0_hz"] / 1e12 for r in resonances], [r["q"] for r in resonances],
        ls="none", marker="o", ms=4, color=PALETTE[2],
    )
    axes[1].set_ylabel("Q")
    axes[1].set_xlabel("Frequency (THz)")

    for ax in axes:
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.margins(x=0.005)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle(
        f"{h5_path.parent.name}  sweep {sweep}  {len(resonances)} resonance(s)",
        fontsize=10,
    )
    fig.align_ylabels(axes)
    fig.tight_layout()
    out_path = Path(out_path) if out_path else h5_path.with_name(
        f"{h5_path.stem}_resonances_{sweep}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("wrote %s", out_path)
    if show:
        plt.show()
    plt.close(fig)
    return out_path
