"""Checks on the config as a single source of truth for the analysis knobs.

The two things that can silently rot: DEFAULTS drifting away from
default.yaml (so an old file and a new one process differently for no stated
reason), and a number written in a YAML form that parses as a string.
"""

import pytest
import yaml

from spectro import config

CONFIG_PATH = "configs/default.yaml"


@pytest.fixture
def cfg():
    return config.load(CONFIG_PATH)


def test_defaults_match_the_shipped_config(cfg):
    """Every knob in DEFAULTS is in default.yaml with the same value.

    They are two copies on purpose -- the config is what you edit, DEFAULTS is
    where a scan.h5 written before a block existed lands -- so a change to one
    must be a change to both, or old and new files quietly disagree.
    """
    for step, defaults in config.DEFAULTS.items():
        assert step in cfg["process"], f"process.{step} missing from {CONFIG_PATH}"
        assert cfg["process"][step] == defaults, step


def test_params_config_beats_defaults(cfg):
    assert config.params(cfg, "ruler")["degree"] == 2
    assert config.params({"process": {"ruler": {"degree": 4}}}, "ruler")["degree"] == 4

    old = {"process": {"bandpass_hz": [2.0e5, 7.0e5]}}  # no per-step blocks
    assert config.params(old, "ruler") == config.DEFAULTS["ruler"]
    assert config.params({}, "anchors") == config.DEFAULTS["anchors"]

    with pytest.raises(KeyError):
        config.params(cfg, "no-such-step")


def test_params_is_a_copy(cfg):
    """A caller mutating the result must not edit DEFAULTS for everyone else."""
    config.params(cfg, "ruler")["degree"] = 99
    assert config.DEFAULTS["ruler"]["degree"] == 2


@pytest.mark.parametrize("form", ["2.0e5", "2e+5", "1.0E5"])
def test_preflight_rejects_unparsed_scientific_notation(cfg, form):
    """YAML 1.1 needs both a decimal point and a signed exponent."""
    cfg["process"]["resonances"]["expected_fwhm_hz"] = yaml.safe_load(f"a: {form}")["a"]
    with pytest.raises(ValueError, match="not a number"):
        config.preflight(cfg)


def test_preflight_accepts_real_strings(cfg):
    """max_j_species is legitimately a string; the guard must not trip on it."""
    assert cfg["process"]["ruler"]["max_j_species"] == "H12C14N"
    config.preflight(cfg)  # does not raise


def test_shipped_config_numbers_all_parsed(cfg):
    assert config._unparsed_numbers(cfg["process"]) == []
