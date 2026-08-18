"""Command-line entry point for `spectro`."""

from __future__ import annotations

import argparse
import logging
import sys

from . import config, ruler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spectro", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="acquire one sweep")
    p_run.add_argument("config", help="path to the YAML config")

    p_check = sub.add_parser("check", help="validate a config without touching hardware")
    p_check.add_argument("config")

    p_plot = sub.add_parser("plot", help="plot the raw channels of an existing scan")
    p_plot.add_argument("h5file", help="path to a scan.h5 written by `spectro run`")
    p_plot.add_argument("--show", action="store_true", help="open an interactive window")
    p_plot.add_argument(
        "--sweep", type=int, default=None,
        help="plot only the Nth forward sweep (0-based), excluding the flyback",
    )
    p_plot.add_argument(
        "--phase", action="store_true",
        help="plot the MZI phase diagnostics for --sweep instead of the raw channels",
    )
    p_plot.add_argument(
        "--gas", action="store_true",
        help="plot the gas-cell line identification for --sweep and list the anchors",
    )
    p_plot.add_argument(
        "--resonances", action="store_true",
        help="plot the DUT spectrum and fitted resonances for --sweep (steps 5-6); "
        "run `spectro fit` first",
    )

    p_fit = sub.add_parser(
        "fit", help="fit nu(phi) through the gas-cell anchors for one sweep (step 4)"
    )
    p_fit.add_argument("h5file", help="path to a scan.h5 written by `spectro run`")
    p_fit.add_argument("--sweep", type=int, default=0, help="which forward sweep (0-based)")

    p_d2 = sub.add_parser("d2", help="fit the ring's dispersion, D1 and D2 (needs `fit` first)")
    p_d2.add_argument("h5file", help="path to a scan.h5 written by `spectro run`")
    p_d2.add_argument("--sweep", type=int, default=0, help="which forward sweep (0-based)")
    p_d2.add_argument(
        "--out", default=None,
        help="diagnostic plot path; defaults to <stem>_dispersion_<sweep>.png",
    )
    p_d2.add_argument("--show", action="store_true", help="open an interactive window")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    sweep = getattr(args, "sweep", None) or 0  # plot_scan alone wants the None

    if args.cmd == "check":
        cfg = config.load(args.config)
        print(
            f"OK  sweep {config.sweep_duration_s(cfg):.1f} s  |  "
            f"{config.sample_rate_hz(cfg) / 1e6:.2f} MS/s"
        )
        return 0

    if args.cmd == "plot":
        from . import plot

        if args.gas:
            anchors, _ = ruler.gas_anchors(args.h5file, sweep=sweep)
            for a in anchors:
                print(
                    f"{a['species']:>8} {a['label']:<4} {a['lambda_nm']:12.5f} nm  "
                    f"{a['nu_hz'] / 1e12:10.6f} THz  phi {a['phi_rad'] / 1e6:9.4f} Mrad  "
                    f"depth {a['depth']:6.3f}  resid {a['residual_pm']:+7.1f} pm"
                )
            print(
                plot.plot_gas(
                    args.h5file, sweep=sweep, show=args.show, anchors=anchors
                )
            )
        elif args.resonances:
            freq_hz, dut, resonances = ruler.find_resonances(
                args.h5file, sweep=sweep
            )
            for r in resonances:
                print(
                    f"{r['f0_hz'] / 1e12:10.6f} THz  fwhm {r['fwhm_hz'] / 1e6:7.2f} MHz  "
                    f"Q {r['q']:9.0f}  amp {r['amp_hz']:+8.4f}"
                )
            print(
                plot.plot_resonances(
                    args.h5file, sweep=sweep, show=args.show,
                    freq_hz=freq_hz, dut=dut, resonances=resonances,
                )
            )
        elif args.phase:
            print(plot.plot_phase(args.h5file, sweep=sweep, show=args.show))
        else:
            print(plot.plot_scan(args.h5file, show=args.show, sweep=args.sweep))
        return 0

    if args.cmd == "d2":
        from . import dispersion

        return dispersion.run(args.h5file, sweep=sweep, out=args.out, show=args.show)

    if args.cmd == "fit":
        fit = ruler.fit_ruler(args.h5file, sweep=sweep)
        print(
            f"sweep {sweep}: degree {fit['degree']} fit, {fit['n_anchors']} anchor(s) "
            f"({fit['dropped_high_j']} dropped), {fit['rms_hz'] / 1e6:.1f} MHz RMS"
        )
        print(f"coeffs (Hz, highest power first): {fit['coeffs']}")
        return 0

    if args.cmd == "run":
        from . import acquire  # deferred: needs the VISA drivers
        from . import plot

        # Acquire only. The raw record still holds the parked-laser regions at
        # both ends, and how to trim them is not settled -- so building the
        # frequency ruler (`spectro fit`) stays a separate, explicit step
        # rather than running on every capture. A raw plot is cheap and is
        # the first thing you look at, so it runs here.
        cfg = config.load(args.config)
        h5_path = acquire.run(cfg, config_path=args.config)
        print(h5_path)
        print(plot.plot_scan(h5_path))
        return 0

    raise AssertionError(f"unhandled command: {args.cmd!r}")


if __name__ == "__main__":
    sys.exit(main())
