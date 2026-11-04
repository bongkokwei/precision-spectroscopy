"""Heater DC source, picked by heater.driver: a Rigol DG952 or the Arduino voltage_ctrl supply."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator


@contextmanager
def source(h: dict) -> Iterator[Callable[[float], None]]:
    """Yield set_v(volts); the output is zeroed and released on exit, even on error.

    h is the config's heater section. Both drivers open the device per call, so
    the output stays latched between steps and nothing is held open during a scan.
    """
    driver = h.get("driver", "dg952")
    if driver == "dg952":
        from rigol_dg952 import RigolDG952

        d = h["dg952"]
        first = True

        def set_v(v: float) -> None:
            nonlocal first
            with RigolDG952(d["resource"]) as dg:
                ch = dg.channel(d["channel"])
                if first:
                    ch.set_load_ohm(h["load_ohm"])
                ch.apply_dc(v)
                if first:
                    ch.output_on()
            first = False

        try:
            yield set_v
        finally:
            with RigolDG952(d["resource"]) as dg:
                ch = dg.channel(d["channel"])
                ch.apply_dc(0.0)
                ch.output_off()
                dg.local()

    elif driver == "voltage_ctrl":
        from voltage_ctrl import VoltageController

        d = h["voltage_ctrl"]
        # zero_on_exit zeros every channel this controller touched
        with VoltageController(com_port=d["com_port"],
                               baud_rate=d.get("baud_rate", 9600),
                               zero_on_exit=True) as vc:
            # ponytail: set_voltages_safe reopens the port twice per call (~6 s of
            # Arduino resets); hold the port open in voltage_ctrl if steps get slow
            yield lambda v: vc.set_voltages_safe(
                [d["channel"]], [v], resistance=h["load_ohm"], v_max=d["v_max"])

    else:
        raise ValueError(f"heater.driver must be 'dg952' or 'voltage_ctrl', got {driver!r}")
