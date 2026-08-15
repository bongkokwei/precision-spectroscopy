"""Cut a raw record into individual forward sweeps using the trigger channel.

The trigger fires at the start of each forward ramp, which then runs for
config.sweep_duration_s. Everything after that until the next trigger is the
flyback -- the laser snapping back to start_nm -- where the tuning rate is
large and uncontrolled and the MZI phase is not a usable frequency ruler.
Taking a fixed duration from each rising edge discards the flyback without
having to detect it.
"""

from __future__ import annotations

import logging

import numpy as np

from . import config

log = logging.getLogger(__name__)


def rising_edges(trigger: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Sample indices where the trigger crosses low to high.

    Args:
        trigger: Trigger channel samples.
        threshold: Crossing level; defaults to the midpoint of the trace, which
            clears the baseline noise band as long as the pulse amplitude
            dominates it.

    Returns:
        Indices of the first high sample of each pulse.
    """
    trigger = np.asarray(trigger)
    if threshold is None:
        threshold = 0.5 * (trigger.min() + trigger.max())
    high = trigger > threshold
    return np.flatnonzero(~high[:-1] & high[1:]) + 1


def sweep_slices(t_s: np.ndarray, trigger: np.ndarray, cfg: dict) -> list[slice]:
    """Return one slice per complete forward sweep in the record.

    A trailing sweep that runs past the end of the capture is dropped rather
    than returned short, since a truncated ramp would silently bias any fit
    made on it.

    Args:
        t_s: Time base from plot.load_h5.
        trigger: Trigger channel samples.
        cfg: Loaded config, for the sweep duration and, as rising_edges'
            threshold, process.sweeps.trigger_threshold.

    Returns:
        Slices into the sample axis, in acquisition order.
    """
    dt_s = float(t_s[1] - t_s[0])
    n = int(round(config.sweep_duration_s(cfg) / dt_s))
    starts = rising_edges(trigger, config.params(cfg, "sweeps")["trigger_threshold"])
    out = [slice(int(s), int(s) + n) for s in starts if s + n <= len(trigger)]
    log.info(
        "%d trigger edge(s), %d complete sweep(s) of %d samples (%.1f s)",
        len(starts), len(out), n, n * dt_s,
    )
    if len(starts) and not out:
        log.warning(
            "every sweep runs past the end of the capture; raise "
            "scope.duration_s or check laser.speed_nm_s"
        )
    return out


def iter_sweeps(t_s, traces: dict[str, np.ndarray], cfg, trigger_role="trigger"):
    """Yield (t_rel_s, {role: samples}) for each complete sweep.

    The time base is re-zeroed to the trigger so sweeps overlay directly.
    """
    for sl in sweep_slices(t_s, traces[trigger_role], cfg):
        yield t_s[sl] - t_s[sl.start], {k: v[sl] for k, v in traces.items()}


def _demo() -> None:
    """Self-check against a synthetic record: 4 triggers, 5 sweeps of room."""
    cfg = {"laser": {"start_nm": 1500.0, "stop_nm": 1600.0, "speed_nm_s": 10.0}}
    dt_s, n = 1e-3, 10_000  # 10 s sweep at 1 kS/s
    t_s = np.arange(5 * n) * dt_s
    trig = np.random.default_rng(0).normal(0, 0.2, 5 * n)
    for i in range(4):
        trig[i * n + 10 : i * n + 60] = 4.0

    sl = sweep_slices(t_s, trig, cfg)
    assert len(sl) == 4, len(sl)
    assert all(s.stop - s.start == n for s in sl)
    assert sl[0].start == 10 and sl[1].start - sl[0].start == n

    # A sweep that would overrun the record is dropped, not truncated.
    assert len(sweep_slices(t_s[: 3 * n + 10], trig[: 3 * n + 10], cfg)) == 3
    # No trigger at all -> no sweeps, no exception.
    assert sweep_slices(t_s, np.zeros_like(t_s), cfg) == []

    t_rel, seg = next(iter_sweeps(t_s, {"trigger": trig, "mzi": t_s.copy()}, cfg))
    assert t_rel[0] == 0.0 and len(seg["mzi"]) == n
    assert abs(t_rel[-1] - 9.999) < 1e-6, t_rel[-1]
    print("ok")


if __name__ == "__main__":
    _demo()
