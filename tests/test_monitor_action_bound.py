"""Lane G: the action-bound monitor (ruling D10, condition 2), on the fake.

D10 rules the condition as: **arm dimensions only, +/-1.1, a fixed declared value**.
The gripper is excluded and the training range is not the bound. The values the
tests pin against are the envelope spike's (`docs/SPIKE-envelope.md`): clean arm
peaks reached 1.01, the gripper 1.12 on every condition, and lost action stats
pushed the arm to 2.27-2.68.

The fault is injected the way it happens: the policy's output is unnormalized
with the wrong statistics, so the arm leaves its range while the gripper, which
is clipped to its command range either way, carries no signal.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from fathomline.core import (
    Action,
    ArtifactRef,
    Observation,
    RunManifest,
    RunResult,
    StepRecord,
    run_episode,
)
from fathomline.fakes import FakeEnv, FakePolicy
from fathomline.gates import GateConfig, evaluate
from fathomline.monitors import (
    ARM_ACTION_LIMIT,
    LIBERO_ARM_DIMS,
    ActionBound,
    MonitorSuite,
    fixture_monitors,
    is_flagged,
)

SEEDS = tuple(range(1000, 1010))
MAX_STEPS = 10
GRIPPER = 6


def _record(action: Sequence[float], index: int = 0) -> StepRecord:
    return StepRecord(index=index, observation={}, action=tuple(action), reward=0.0)


def _arm(value: float, *, dim: int = 0, gripper: float = 1.0) -> Action:
    """A LIBERO-shaped action: six arm dimensions at 0.5 except `dim`, then the gripper."""
    arm = [0.5] * 6
    arm[dim] = value
    return (*arm, gripper)


def _monitor() -> ActionBound:
    return ActionBound(limit=1.1, dims=LIBERO_ARM_DIMS)


# --- The ruled declaration ----------------------------------------------------------------------


def test_the_declared_bound_is_the_ruled_value() -> None:
    # D10: a fixed, declared value. Changing it is a ruling, not a tuning step.
    assert ARM_ACTION_LIMIT == 1.1


def test_the_declared_dimensions_are_the_arm_and_exclude_the_gripper() -> None:
    assert LIBERO_ARM_DIMS == (0, 1, 2, 3, 4, 5)
    assert GRIPPER not in LIBERO_ARM_DIMS


def test_the_fixture_monitors_include_the_action_bound() -> None:
    suite = fixture_monitors()
    assert isinstance(suite, MonitorSuite)
    assert "action-bound" in suite.name


def test_the_fixture_suite_flags_lost_stats_and_passes_a_clean_step() -> None:
    suite = fixture_monitors()
    assert list(suite.on_step(_record(_arm(1.01)))) == []
    (flag,) = suite.on_step(_record(_arm(2.27, dim=1), index=5))
    assert (flag.monitor, flag.step) == ("action-bound", 5)


# --- One step -----------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.0, 0.94, 1.01, -1.01, 1.1, -1.1])
def test_an_arm_value_within_the_bound_is_not_flagged(value: float) -> None:
    # 1.01 is the clean peak the spike measured; the bound itself is inside the envelope.
    for dim in LIBERO_ARM_DIMS:
        assert list(_monitor().on_step(_record(_arm(value, dim=dim)))) == []


@pytest.mark.parametrize("value", [1.1001, -1.1001, 2.27, -2.68])
def test_an_arm_value_beyond_the_bound_is_flagged_on_every_arm_dimension(value: float) -> None:
    for dim in LIBERO_ARM_DIMS:
        (flag,) = _monitor().on_step(_record(_arm(value, dim=dim), index=7))
        assert flag.monitor == "action-bound"
        assert flag.step == 7
        assert f"dim {dim}" in flag.reason


@pytest.mark.parametrize("gripper", [1.12, -1.12, 5.0])
def test_the_gripper_is_not_monitored(gripper: float) -> None:
    # The gripper passed +/-1 on 68% of clean steps and peaked at 1.06-1.12 everywhere.
    assert list(_monitor().on_step(_record(_arm(0.5, gripper=gripper)))) == []


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_non_finite_arm_value_is_flagged(value: float) -> None:
    # A NaN compares false against any bound; a monitor that only asks "is it above
    # the limit?" would wave it through.
    (flag,) = _monitor().on_step(_record(_arm(value, dim=3)))
    assert "dim 3" in flag.reason


def test_one_flag_per_step_names_every_dimension_out_of_bounds() -> None:
    action = (2.3, -2.4, 0.2, 0.1, 0.0, 0.0, 1.1)
    (flag,) = _monitor().on_step(_record(action, index=3))
    assert "dim 0 = 2.3" in flag.reason
    assert "dim 1 = -2.4" in flag.reason
    assert "dim 2" not in flag.reason
    assert "1.1" in flag.reason


def test_the_monitor_name_is_configurable() -> None:
    monitor = ActionBound(limit=1.1, dims=(0,), name="arm")
    assert monitor.name == "arm"
    (flag,) = monitor.on_step(_record((2.0,)))
    assert flag.monitor == "arm"


def test_an_action_too_short_for_the_declared_dimensions_is_refused() -> None:
    # Skipping the missing dimensions would read as "inside the envelope" without
    # anything having been checked. The monitor stops the episode instead.
    with pytest.raises(ValueError, match="dimension 5"):
        _monitor().on_step(_record((0.0, 0.0, 0.0)))


# --- Construction -------------------------------------------------------------------------------


@pytest.mark.parametrize("limit", [0.0, -1.1, math.nan, math.inf])
def test_a_limit_that_is_not_a_positive_finite_number_is_refused(limit: float) -> None:
    with pytest.raises(ValueError, match="limit"):
        ActionBound(limit=limit, dims=(0,))


def test_no_dimensions_is_refused() -> None:
    # A monitor that checks nothing never flags, which reads like a clean run.
    with pytest.raises(ValueError, match="at least one"):
        ActionBound(limit=1.1, dims=())


def test_a_negative_dimension_is_refused() -> None:
    # -1 would silently mean "the last dimension", which here is the gripper.
    with pytest.raises(ValueError, match="-1"):
        ActionBound(limit=1.1, dims=(0, -1))


def test_a_repeated_dimension_is_refused() -> None:
    with pytest.raises(ValueError, match="repeated"):
        ActionBound(limit=1.1, dims=(0, 1, 1))


# --- Through the loop, with the fault injected --------------------------------------------------


class _ArmPolicy:
    """A seven-dimensional policy on the fake: six arm dimensions and a gripper.

    The fake's own action, a normalized command, is unnormalized per arm dimension
    as `z * std + mean`, the way LeRobot's postprocessor does. With the right
    statistics the arm stays well inside the bound. `lost_stats_from` swaps in
    mean 0, std 1 from that step on: the "action metadata lost" fault. The gripper
    is held at a clean-run value beyond +/-1 throughout.
    """

    ref = ArtifactRef(name="fake-arm-policy", revision="0")
    STD = 0.2

    def __init__(self, *, lost_stats_from: int | None = None) -> None:
        self._inner = FakePolicy()
        self._lost_stats_from = lost_stats_from
        self._steps = 0

    def reset(self, seed: int) -> None:
        self._inner.reset(seed)
        self._steps = 0

    def act(self, observation: Observation) -> Action:
        (command,) = self._inner.act(observation)
        lost = self._lost_stats_from is not None and self._steps >= self._lost_stats_from
        self._steps += 1
        # A normalized output a few standard deviations out, as a trained policy's is.
        z = 3.0 if command >= 0 else -3.0
        std = 1.0 if lost else self.STD
        arm = z * std
        return (arm, arm, arm, 0.0, 0.0, 0.0, 1.12)


def _run(policy: _ArmPolicy) -> RunResult:
    env = FakeEnv(success_rate=1.0)
    manifest = RunManifest.capture(env=env, policy=policy, seeds=SEEDS, max_steps=MAX_STEPS)
    episodes = tuple(
        run_episode(env, policy, seed=seed, max_steps=MAX_STEPS, monitors=[fixture_monitors()])
        for seed in SEEDS
    )
    return RunResult(manifest=manifest, episodes=episodes)


def test_a_clean_run_is_not_flagged() -> None:
    run = _run(_ArmPolicy())
    assert is_flagged(run) is False


def test_lost_action_stats_flag_every_episode() -> None:
    run = _run(_ArmPolicy(lost_stats_from=0))
    assert all(episode.flags for episode in run.episodes)
    assert {flag.monitor for episode in run.episodes for flag in episode.flags} == {
        "action-bound"
    }


def test_the_flag_fires_at_the_step_the_stats_are_lost() -> None:
    run = _run(_ArmPolicy(lost_stats_from=5))
    for episode in run.episodes:
        assert episode.flags[0].step == 5


def test_a_run_with_lost_stats_fails_the_gate_even_when_every_episode_succeeds() -> None:
    run = _run(_ArmPolicy(lost_stats_from=0))
    assert all(episode.success for episode in run.episodes)
    verdict = evaluate(run, GateConfig(threshold=0.5, min_episodes=len(SEEDS)))
    assert verdict.passed is False
    assert verdict.flagged_episodes == len(SEEDS)
