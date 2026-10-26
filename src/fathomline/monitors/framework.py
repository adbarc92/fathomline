"""The per-step hook and the flag semantics.

The loop (`fathomline.core.run_episode`) already calls every `Monitor` once per
step and keeps what they return on the `EpisodeResult`. This module adds the two
things that make a flag mean something:

* `MonitorSuite`, the hook to hand the loop. It runs its monitors in order and
  refuses a flag the monitor could not have raised: one carrying another
  monitor's name, or another step's index. A misattributed flag would send the
  reader looking at the wrong condition or the wrong moment.
* `is_flagged` / `flagged_episodes`, the run-level reading. One flag on one step
  of one episode flags the whole run, whether that episode succeeded or not.
  The gate never passes a run for which `is_flagged` is true, and the report
  surfaces every episode `flagged_episodes` returns.

A monitor that raises is not caught: the episode stops with the error, so there
is no result to read as unflagged.
"""

from __future__ import annotations

from collections.abc import Sequence

from fathomline.core.protocols import Monitor
from fathomline.core.types import EpisodeResult, Flag, RunResult, StepRecord


class MonitorContractError(RuntimeError):
    """A monitor returned a flag that does not describe the step it was shown."""


class MonitorSuite:
    """The monitors for one run, called by the loop as a single `Monitor`."""

    def __init__(self, monitors: Sequence[Monitor]) -> None:
        if not monitors:
            # An unmonitored run has no flags because nothing looked, which
            # would read exactly like a run that was looked at and was clean.
            raise ValueError("a MonitorSuite needs at least one monitor")
        names = [monitor.name for monitor in monitors]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"monitor names must be unique; repeated: {', '.join(duplicates)}")
        self._monitors = tuple(monitors)
        self._name = f"suite[{','.join(names)}]"

    @property
    def name(self) -> str:
        return self._name

    def on_step(self, record: StepRecord) -> Sequence[Flag]:
        flags: list[Flag] = []
        for monitor in self._monitors:
            for flag in monitor.on_step(record):
                if flag.monitor != monitor.name:
                    raise MonitorContractError(
                        f"monitor {monitor.name!r} returned a flag attributed to {flag.monitor!r}"
                    )
                if flag.step != record.index:
                    raise MonitorContractError(
                        f"monitor {monitor.name!r} returned a flag for step {flag.step} "
                        f"while shown step {record.index}"
                    )
                flags.append(flag)
        return flags


def is_flagged(result: RunResult | EpisodeResult) -> bool:
    """True if any step of any episode was flagged."""
    if isinstance(result, EpisodeResult):
        return bool(result.flags)
    return any(episode.flags for episode in result.episodes)


def flagged_episodes(run: RunResult) -> tuple[EpisodeResult, ...]:
    """Every episode carrying at least one flag, in run order."""
    return tuple(episode for episode in run.episodes if episode.flags)
