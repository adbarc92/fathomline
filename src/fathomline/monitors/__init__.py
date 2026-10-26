"""The monitor framework: the per-step hook and what a flag means for a run.

A flagged run is flagged, never silently reported. `MonitorSuite` is the hook
the loop calls each step; `is_flagged` and `flagged_episodes` are what the gate
and the report read. The envelope conditions themselves are Lane G's.
"""

from __future__ import annotations

from fathomline.monitors.framework import (
    MonitorContractError,
    MonitorSuite,
    flagged_episodes,
    is_flagged,
)

__all__ = ["MonitorContractError", "MonitorSuite", "flagged_episodes", "is_flagged"]
