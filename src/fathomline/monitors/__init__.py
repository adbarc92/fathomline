"""The monitor framework, and the envelope conditions it runs.

A flagged run is flagged, never silently reported. `MonitorSuite` is the hook
the loop calls each step; `is_flagged` and `flagged_episodes` are what the gate
and the report read. The conditions are ruled in D10: `Provenance`,
`ActionBound`, `FrameBand` and `InstructionSet`, and `fixture_monitors`
assembles the D1 fixture's declared set from `envelope`.
"""

from __future__ import annotations

from fathomline.monitors.action_bound import ActionBound
from fathomline.monitors.envelope import (
    ARM_ACTION_LIMIT,
    LIBERO_ARM_DIMS,
    LIBERO_FRAME_BANDS,
    LIBERO_INSTRUCTIONS,
    LIBERO_LOADED_PIPELINE,
    fixture_monitors,
)
from fathomline.monitors.framework import (
    MonitorContractError,
    MonitorSuite,
    flagged_episodes,
    is_flagged,
)
from fathomline.monitors.instruction import InstructionSet
from fathomline.monitors.observation_shift import FrameBand
from fathomline.monitors.provenance import Provenance

__all__ = [
    "ARM_ACTION_LIMIT",
    "LIBERO_ARM_DIMS",
    "LIBERO_FRAME_BANDS",
    "LIBERO_INSTRUCTIONS",
    "LIBERO_LOADED_PIPELINE",
    "ActionBound",
    "FrameBand",
    "InstructionSet",
    "MonitorContractError",
    "MonitorSuite",
    "Provenance",
    "fixture_monitors",
    "flagged_episodes",
    "is_flagged",
]
