"""Command-line entry point: `spectro run` and `spectro process`."""

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

    p_proc = sub.add_parser("process", help="build the frequency ruler for an existing scan")
    p_proc.add_argument("h5file", help="path to a scan.h5 written by `spectro run`")
    p_proc.add_argument("--points", type=int, default=None, help="output grid size")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    if args.cmd == "check":
        cfg = config.load(args.config)
        print(
            f"OK  sweep {config.sweep_duration_s(cfg):.1f} s  |  "
            f"{config.sample_rate_hz(cfg) / 1e6:.2f} MS/s  |  "
            f"fringe rate {config.fringe_rate_hz(cfg) / 1e3:.1f} kHz"
        )
        return 0

    if args.cmd == "run":
        from . import acquire  # deferred: needs the VISA drivers

        # Acquire only. The raw record still holds the parked-laser regions at
        # both ends, and how to trim them is not settled -- so `spectro process`
        # stays a separate, explicit step rather than running on every capture.
        cfg = config.load(args.config)
        print(acquire.run(cfg, config_path=args.config))
        return 0

    ruler.process_file(args.h5file, n_points=args.points)
    return 0


if __name__ == "__main__":
    sys.exit(main())
