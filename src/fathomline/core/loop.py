"""The step loop. Fathomline owns it, so what gets recorded is not up to a library."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence

from fathomline.core.protocols import Env, Monitor, Policy
from fathomline.core.types import EpisodeResult, Flag, StepRecord


def run_episode(
    env: Env,
    policy: Policy,
    *,
    seed: int,
    max_steps: int,
    monitors: Sequence[Monitor] = (),
) -> EpisodeResult:
    """Run one episode. An episode that reaches `max_steps` without finishing fails."""
    if max_steps < 1:
        raise ValueError(f"max_steps must be at least 1, got {max_steps}")

    observation = env.reset(seed)
    policy.reset(seed)

    digest = hashlib.sha256()
    flags: list[Flag] = []
    sum_reward = 0.0
    max_reward = float("-inf")
    success = False
    steps = 0

    for index in range(max_steps):
        action = policy.act(observation)
        transition = env.step(action)
        # Length-prefixed so two different step boundaries cannot hash alike.
        digest.update(struct.pack(f"<I{len(action)}d", len(action), *action))

        record = StepRecord(
            index=index, observation=observation, action=action, reward=transition.reward
        )
        for monitor in monitors:
            flags.extend(monitor.on_step(record))

        sum_reward += transition.reward
        max_reward = max(max_reward, transition.reward)
        steps = index + 1
        if transition.done:
            success = transition.success
            break
        observation = transition.observation

    return EpisodeResult(
        seed=seed,
        steps=steps,
        success=success,
        sum_reward=sum_reward,
        max_reward=max_reward,
        action_hash=digest.hexdigest(),
        flags=tuple(flags),
    )
