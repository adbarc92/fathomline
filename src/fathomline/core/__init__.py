"""The harness core: protocols, records, the step loop and result schema v1.

Nothing here imports a simulator or a policy library. LeRobot sits behind one
adapter that implements `Env` and `Policy`; the loop that calls them is ours.
"""

from __future__ import annotations

from fathomline.core.loop import run_episode
from fathomline.core.protocols import Env, Monitor, Policy
from fathomline.core.schema import SCHEMA_VERSION, SchemaError, from_document, to_document
from fathomline.core.types import (
    Action,
    ArtifactRef,
    EpisodeResult,
    Flag,
    Observation,
    RunManifest,
    RunResult,
    StepRecord,
    Transition,
)

__all__ = [
    "SCHEMA_VERSION",
    "Action",
    "ArtifactRef",
    "Env",
    "EpisodeResult",
    "Flag",
    "Monitor",
    "Observation",
    "Policy",
    "RunManifest",
    "RunResult",
    "SchemaError",
    "StepRecord",
    "Transition",
    "from_document",
    "run_episode",
    "to_document",
]
