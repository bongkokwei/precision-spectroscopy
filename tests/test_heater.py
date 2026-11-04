"""heater.source dispatches on heater.driver and always zeroes the output."""

import sys
import types

import pytest

from spectro import heater

H = {"load_ohm": 50, "dg952": {"resource": "R", "channel": 2},
     "voltage_ctrl": {"com_port": "COM9", "channel": 8, "v_max": 5.0}}


def test_dg952_sets_then_zeroes_on_error(monkeypatch):
    calls = []

    class Ch:
        def __getattr__(self, name):
            return lambda *a: calls.append((name, *a))

    class DG:
        def __init__(self, res): assert res == "R"
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def channel(self, n): assert n == 2; return Ch()
        def local(self): calls.append(("local",))

    monkeypatch.setitem(sys.modules, "rigol_dg952", types.SimpleNamespace(RigolDG952=DG))
    with pytest.raises(RuntimeError):
        with heater.source({**H, "driver": "dg952"}) as set_v:
            set_v(1.0)
            set_v(2.0)
            raise RuntimeError
    assert calls == [("set_load_ohm", 50), ("apply_dc", 1.0), ("output_on",),
                     ("apply_dc", 2.0), ("apply_dc", 0.0), ("output_off",), ("local",)]


def test_voltage_ctrl(monkeypatch):
    calls = []

    class VC:
        def __init__(self, **kw): calls.append(kw)
        def __enter__(self): return self
        def __exit__(self, *a): calls.append("exit")
        def set_voltages_safe(self, *a, **kw): calls.append((a, kw))

    monkeypatch.setitem(sys.modules, "voltage_ctrl", types.SimpleNamespace(VoltageController=VC))
    with heater.source({**H, "driver": "voltage_ctrl"}) as set_v:
        set_v(1.5)
    assert calls == [{"com_port": "COM9", "baud_rate": 9600, "zero_on_exit": True},
                     (([8], [1.5]), {"resistance": 50, "v_max": 5.0}), "exit"]


def test_unknown_driver():
    with pytest.raises(ValueError):
        with heater.source({"driver": "nope"}):
            pass
