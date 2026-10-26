"""The seams. Everything outside the core plugs in through one of these three.

They are deliberately small: this is the surface the monitors are retargeted
across, so every method here is one a second simulator has to implement.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from fathomline.core.types import Action, ArtifactRef, Flag, Observation, StepRecord, Transition


class Env(Protocol):
    @property
    def ref(self) -> ArtifactRef: ...

    def reset(self, seed: int) -> Observation: ...

    def step(self, action: Action) -> Transition: ...


class Policy(Protocol):
    @property
    def ref(self) -> ArtifactRef: ...

    def reset(self, seed: int) -> None: ...

    def act(self, observation: Observation) -> Action: ...


class Monitor(Protocol):
    """Called once per step. Observes only: it cannot alter the trajectory."""

    @property
    def name(self) -> str: ...

    def on_step(self, record: StepRecord) -> Sequence[Flag]: ...
