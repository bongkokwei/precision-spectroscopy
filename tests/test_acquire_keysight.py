"""acquire.run drives the DSOX4024A when scope.driver is dsox4024a."""

import sys
import types
from pathlib import Path

import numpy as np
import yaml

from spectro import acquire

CFG = Path(__file__).parents[1] / "configs" / "keysight.yaml"


def test_keysight_run(monkeypatch, tmp_path):
    calls = []

    class Scope:
        def __init__(self, res, timeout): calls.append(("open", timeout))
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def __getattr__(self, name): return lambda *a, **kw: calls.append((name, a, kw))
        def identify(self): return "KEYSIGHT,DSO-X 4024A"
        def get_waveform(self, ch): return np.arange(5.0), np.full(5, float(ch))
        def query(self, cmd): return "1.0E+05"

    class Laser:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def __getattr__(self, name): return lambda *a, **kw: calls.append((name,))
        def identify(self): return "8164B"
        def check_sweep_params(self): return "OK"

    monkeypatch.setitem(sys.modules, "keysight_dsox4024a", types.SimpleNamespace(DSOX4024A=Scope))
    monkeypatch.setitem(sys.modules, "agilent_8164b", types.SimpleNamespace(Agilent8164B=Laser))
    written = {}
    monkeypatch.setattr(acquire, "new_run_dir", lambda *a: tmp_path)
    monkeypatch.setattr(acquire, "setup_logging", lambda *a: None)
    monkeypatch.setattr(acquire, "write_h5", lambda p, t, d, c, i: written.update(t=t, d=d, idn=i))

    acquire.run(yaml.safe_load(CFG.read_text()))

    names = [c[0] for c in calls]
    assert calls[0] == ("open", 70.0)
    assert names.index("start_sweep") < names.index("digitise") < names.index("stop_sweep")
    assert next(c for c in calls if c[0] == "digitise") == ("digitise", (1, 2, 3, 4), {"single": True})
    assert ("set_timebase", (2.0,), {"position_s": 9.5}) in calls
    assert ("set_trigger_edge", ("CHANnel4", 1.5, "POSitive"), {"sweep": "NORMal"}) in calls
    assert "check_error" in names
    assert ("write", (":ACQuire:SRATe 100000",), {}) in calls
    assert "system_restart" not in names
    assert written["idn"]["scope"].startswith("KEYSIGHT")
    assert sorted(written["d"]) == [1, 2, 3, 4] and written["d"][3][0] == 3.0
