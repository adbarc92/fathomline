"""The step loop. Fathomline owns it, so what gets recorded is not up to a library.

`run_episode` runs one seeded episode. `run_sweep` runs a declared seed set,
one episode per seed, and keeps what persistence needs beside the result: each
episode's actions and its wall-clock time. The seed is the only thing an episode
may depend on; nothing carries over from the episode before it.
"""

from __future__ import annotations

import hashlib
import struct
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from fathomline.core.protocols import Env, Monitor, Policy
from fathomline.core.types import Action, EpisodeResult, Flag, RunManifest, RunResult, StepRecord


def _update(digest: hashlib._Hash, action: Action) -> None:
    # Length-prefixed so two different step boundaries cannot hash alike.
    digest.update(struct.pack(f"<I{len(action)}d", len(action), *action))


def action_hash(actions: Iterable[Action]) -> str:
    """The action-trajectory hash `run_episode` records, recomputed from the actions."""
    digest = hashlib.sha256()
    for action in actions:
        _update(digest, action)
    return digest.hexdigest()


def run_episode(
    env: Env,
    policy: Policy,
    *,
    seed: int,
    max_steps: int,
    monitors: Sequence[Monitor] = (),
    actions: list[Action] | None = None,
) -> EpisodeResult:
    """Run one episode. An episode that reaches `max_steps` without finishing fails.

    If `actions` is given, every action taken is appended to it, in order.
    """
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
        _update(digest, action)
        if actions is not None:
            actions.append(action)

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


@dataclass(frozen=True, slots=True)
class Sweep:
    """A run over a seed set, with what persistence keeps beside the result."""

    result: RunResult
    #: Every action of every episode, in run order. Each hashes to its episode's
    #: `action_hash`, so a later mismatch can be traced to the step it began at.
    trajectories: tuple[tuple[Action, ...], ...]
    #: Wall-clock seconds per episode. Never part of the result: it varies run to run.
    durations: tuple[float, ...]


def run_sweep(
    env: Env,
    policy: Policy,
    *,
    seeds: Sequence[int],
    max_steps: int,
    monitors: Sequence[Monitor] = (),
    clock: Callable[[], float] = time.perf_counter,
) -> Sweep:
    """Run one episode per seed, in the order given.

    A repeated seed is refused: it would count one episode twice toward a gate's n.
    """
    if not seeds:
        raise ValueError("a sweep needs at least one seed")
    repeated = sorted({seed for seed in seeds if list(seeds).count(seed) > 1})
    if repeated:
        raise ValueError(f"seeds must be unique; repeated: {', '.join(map(str, repeated))}")

    episodes: list[EpisodeResult] = []
    trajectories: list[tuple[Action, ...]] = []
    durations: list[float] = []
    for seed in seeds:
        actions: list[Action] = []
        start = clock()
        episodes.append(
            run_episode(
                env, policy, seed=seed, max_steps=max_steps, monitors=monitors, actions=actions
            )
        )
        durations.append(clock() - start)
        trajectories.append(tuple(actions))

    manifest = RunManifest.capture(env=env, policy=policy, seeds=tuple(seeds), max_steps=max_steps)
    return Sweep(
        result=RunResult(manifest=manifest, episodes=tuple(episodes)),
        trajectories=tuple(trajectories),
        durations=tuple(durations),
    )
