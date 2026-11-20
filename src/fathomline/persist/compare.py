"""Compare two runs of the same declaration: the determinism check.

Equality is judged on the fixture, the seeds, each episode's metrics and its
action-trajectory hash (D9), never on video. Host facts are not compared: they
say where each run happened, and which hosts are expected to agree is ruled by
D9b (and enforced by `sweep`'s pin check), not this module's call.

Floats are compared bit for bit, as the action hash compares them, so 0.0
differs from -0.0, with one exception: any NaN equals any NaN. Result schema v1
writes a metric as a JSON number, which keeps no NaN sign or payload; the
actions keep theirs, and the action hash covers them.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass

from fathomline.core.types import Action, RunResult

#: The per-episode fields equality is asserted on.
EPISODE_FIELDS = ("steps", "success", "sum_reward", "max_reward", "action_hash", "flags")


def compare_runs(expected: RunResult, actual: RunResult) -> tuple[str, ...]:
    """Every difference between two runs, one line each; empty if they are equal."""
    a, b = expected.manifest, actual.manifest
    differences = [
        f"{name}: {left!r} != {right!r}"
        for name, left, right in (
            ("policy", a.policy, b.policy),
            ("env", a.env, b.env),
            ("max_steps", a.max_steps, b.max_steps),
            ("seeds", a.seeds, b.seeds),
        )
        if left != right
    ]
    left_seeds = tuple(episode.seed for episode in expected.episodes)
    right_seeds = tuple(episode.seed for episode in actual.episodes)
    if left_seeds != right_seeds:
        differences.append(f"episode seeds: {left_seeds!r} != {right_seeds!r}")
    if a.seeds != b.seeds or left_seeds != right_seeds:
        # Episodes are only comparable seed for seed.
        return tuple(differences)
    for left, right in zip(expected.episodes, actual.episodes, strict=True):
        for name in EPISODE_FIELDS:
            left_value, right_value = getattr(left, name), getattr(right, name)
            if not _same(left_value, right_value):
                differences.append(f"seed {left.seed}: {name} {left_value!r} != {right_value!r}")
    return tuple(differences)


def _same(left: object, right: object) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
        return struct.pack("<d", left) == struct.pack("<d", right)
    return left == right


def _packed(action: Action) -> bytes:
    return struct.pack(f"<{len(action)}d", *action)


@dataclass(frozen=True, slots=True)
class Divergence:
    """Where two trajectories of one seed first differ."""

    step: int
    #: The largest absolute difference between the two actions at `step`; None
    #: if one trajectory ended there or the actions differ in length.
    max_abs_diff: float | None


def first_divergence(expected: Sequence[Action], actual: Sequence[Action]) -> Divergence | None:
    """The first step at which two trajectories differ, or None if they are identical."""
    for step, (left, right) in enumerate(zip(expected, actual, strict=False)):
        # Bit for bit, NaN sign and payload included: this explains a hash mismatch.
        if _packed(left) != _packed(right):
            if len(left) != len(right):
                return Divergence(step=step, max_abs_diff=None)
            return Divergence(
                step=step, max_abs_diff=max(abs(x - y) for x, y in zip(left, right, strict=True))
            )
    if len(expected) != len(actual):
        return Divergence(step=min(len(expected), len(actual)), max_abs_diff=None)
    return None
