"""Records the loop produces and the result schema persists.

All of them are frozen: a result that can be edited after the run is not a
measurement.
"""

from __future__ import annotations

import platform as _platform
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fathomline import __version__

if TYPE_CHECKING:
    from fathomline.core.protocols import Env, Policy

type Observation = Mapping[str, object]

#: Plain floats rather than an array type, so the core has no dependencies and
#: the action-trajectory hash is defined over exact values. Adapters convert.
type Action = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """What was run, pinned: a name and the revision that fixes its contents."""

    name: str
    revision: str


@dataclass(frozen=True, slots=True)
class Transition:
    """What an `Env` returns from one step."""

    observation: Observation
    reward: float
    done: bool
    success: bool


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One step as a monitor sees it: what the policy saw, did, and received."""

    index: int
    observation: Observation
    action: Action
    reward: float


@dataclass(frozen=True, slots=True)
class Flag:
    """A monitor's statement that a step fell outside the declared envelope."""

    monitor: str
    step: int
    reason: str


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    seed: int
    steps: int
    success: bool
    sum_reward: float
    max_reward: float
    #: SHA-256 over every action taken, in order. Determinism is asserted on
    #: this and on the metrics above, never on video.
    action_hash: str
    flags: tuple[Flag, ...] = ()


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Provenance: enough to say what a result is a result *of*."""

    fathomline_version: str
    policy: ArtifactRef
    env: ArtifactRef
    seeds: tuple[int, ...]
    max_steps: int
    python_version: str
    platform: str

    @classmethod
    def capture(
        cls, *, env: Env, policy: Policy, seeds: tuple[int, ...], max_steps: int
    ) -> RunManifest:
        return cls(
            fathomline_version=__version__,
            policy=policy.ref,
            env=env.ref,
            seeds=seeds,
            max_steps=max_steps,
            python_version=_platform.python_version(),
            platform=f"{_platform.system()}-{_platform.machine()}",
        )


@dataclass(frozen=True, slots=True)
class RunResult:
    manifest: RunManifest
    episodes: tuple[EpisodeResult, ...]
