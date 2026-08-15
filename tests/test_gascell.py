"""Checks on the transcribed gas-cell reference table."""

from pathlib import Path

import pytest

from spectro import config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture(scope="module")
def cfg():
    return config.load(CONFIG, validate=False)


def test_line_counts_match_datasheets(cfg):
    """Guards against a line dropped or duplicated in transcription."""
    counts = {n: len(s["lines_nm"]) for n, s in cfg["gascell"]["species"].items()}
    assert counts == {"H12C14N": 81, "12CO": 41, "13CO": 41}


def test_h12c14n_band_is_physically_consistent(cfg):
    """The H12C14N table is calculated from fitted constants, not measured, so
    check it against the physics: R and P branches must converge on a common
    band origin, and the rotational constant must be sane."""
    lines = cfg["gascell"]["species"]["H12C14N"]["lines_nm"]
    wn = {k: 1e7 / v for k, v in lines.items()}
    # R(0) and P(1) share the J''=0 and J''=1 levels; their mean brackets the
    # band origin, which Sasada reports near 6519.6 cm-1.
    assert 6519.0 < 0.5 * (wn["R0"] + wn["P1"]) < 6520.5
    # Adjacent-line spacing gives ~2B; B(HCN) is about 1.478 cm-1.
    b = (wn["R1"] - wn["R0"]) / 2
    assert 1.40 < abs(b) < 1.55


def test_lines_lie_in_the_datasheet_range(cfg):
    ranges = {
        "H12C14N": (1514.0, 1571.0),
        "12CO": (1560.0, 1598.0),
        "13CO": (1595.0, 1633.0),
    }
    for name, (lo, hi) in ranges.items():
        w = list(cfg["gascell"]["species"][name]["lines_nm"].values())
        assert lo <= min(w) and max(w) <= hi, name


def test_gas_lines_nm_excludes_out_of_range(cfg):
    """A narrower sweep must drop the L-band species entirely."""
    cfg = {**cfg, "laser": {**cfg["laser"], "start_nm": 1510.0, "stop_nm": 1590.0}}
    lines = config.gas_lines_nm(cfg)
    assert lines["13CO"] == {}
    assert len(lines["12CO"]) == 35             # P14..P19 fall past 1590 nm
    assert sorted(lines["12CO"].values()) == list(lines["12CO"].values())
