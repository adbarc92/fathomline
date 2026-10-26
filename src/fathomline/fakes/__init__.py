"""A fake environment and policy, for testing the harness without a simulator.

The real adapter is Linux-only and takes minutes per episode. These run
anywhere in microseconds, and their outcome is a dial: gates, monitors and the
determinism tests need a fixture whose right answer is known in advance.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from fathomline.core.types import Action, ArtifactRef, Observation, Transition

__all__ = ["FakeEnv", "FakePolicy"]


class FakeEnv:
    """A point on a line. Each episode lasts `horizon` steps.

    Whether an episode succeeds is drawn from the seed at reset, with
    probability `success_rate`, so 0.0 and 1.0 are exact and a seed always
    gives the same outcome. `perturb` is applied to every observation handed
    out, which is how a test puts the policy outside its envelope.
    """

    ref = ArtifactRef(name="fake-env", revision="0")

    def __init__(
        self,
        *,
        success_rate: float,
        horizon: int = 5,
        perturb: Callable[[Observation], Observation] | None = None,
    ) -> None:
        if not 0.0 <= success_rate <= 1.0:
            raise ValueError(f"success_rate must be in [0, 1], got {success_rate}")
        self._success_rate = success_rate
        self._horizon = horizon
        self._perturb = perturb
        self._succeeds = False
        self._position = 0.0
        self._steps = 0

    def reset(self, seed: int) -> Observation:
        rng = random.Random(seed)
        self._succeeds = rng.random() < self._success_rate
        self._position = rng.uniform(-1.0, 1.0)
        self._steps = 0
        return self._observe()

    def step(self, action: Action) -> Transition:
        self._position += action[0]
        self._steps += 1
        done = self._steps >= self._horizon
        return Transition(
            observation=self._observe(),
            reward=-abs(self._position),
            done=done,
            success=done and self._succeeds,
        )

    def _observe(self) -> Observation:
        observation: Observation = {"position": self._position}
        return self._perturb(observation) if self._perturb else observation


class FakePolicy:
    """Steers the point toward zero, with seeded noise so the seed moves the actions."""

    ref = ArtifactRef(name="fake-policy", revision="0")

    def __init__(self) -> None:
        self._rng = random.Random(0)

    def reset(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def act(self, observation: Observation) -> Action:
        position = observation["position"]
        if not isinstance(position, float):
            raise TypeError(f"position must be a float, got {type(position).__name__}")
        return (-0.5 * position + self._rng.gauss(0.0, 0.01),)
