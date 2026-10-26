"""Lane F: the monitor framework and its flag semantics, on the fake.

The rule under test: a flagged run is flagged, never silently reported. A flag
is raised per step, is attributed to the monitor and step that raised it,
survives persistence, and marks the whole run, whatever the episodes' outcome.

`PositionBound` is the one example monitor. It exists to exercise the
framework; the real envelope conditions are Lane G's, after ruling D10.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

import pytest

from fathomline.core import (
    EpisodeResult,
    Flag,
    Monitor,
    Observation,
    RunManifest,
    RunResult,
    StepRecord,
    from_document,
    run_episode,
    to_document,
)
from fathomline.fakes import FakeEnv, FakePolicy
from fathomline.monitors import MonitorContractError, MonitorSuite, flagged_episodes, is_flagged

SEEDS = (1000, 2000, 3000)
MAX_STEPS = 10


class PositionBound:
    """Flags any step whose observed position lies outside +/- `limit`.

    The unperturbed fake starts in [-1, 1] and is steered toward zero, so it
    never trips a limit of 2.
    """

    def __init__(self, limit: float = 2.0, name: str = "position-bound") -> None:
        self._limit = limit
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def on_step(self, record: StepRecord) -> Sequence[Flag]:
        position = record.observation["position"]
        assert isinstance(position, float)
        if abs(position) <= self._limit:
            return []
        return [Flag(monitor=self.name, step=record.index, reason=f"|position| > {self._limit}")]


def _shift(observation: Observation) -> Observation:
    position = observation["position"]
    assert isinstance(position, float)
    return {"position": position + 10.0}


def _shift_after(steps: int) -> Callable[[Observation], Observation]:
    """A perturbation that leaves the first `steps` observations alone (reset counts as one)."""
    calls = 0

    def perturb(observation: Observation) -> Observation:
        nonlocal calls
        calls += 1
        return _shift(observation) if calls > steps else observation

    return perturb


def _suite() -> MonitorSuite:
    return MonitorSuite([PositionBound()])


def _run(
    perturb: Callable[[Observation], Observation] | None = None, *, success_rate: float = 1.0
) -> RunResult:
    env, policy = FakeEnv(success_rate=success_rate, perturb=perturb), FakePolicy()
    manifest = RunManifest.capture(env=env, policy=policy, seeds=SEEDS, max_steps=MAX_STEPS)
    episodes = tuple(
        run_episode(env, policy, seed=seed, max_steps=MAX_STEPS, monitors=[_suite()])
        for seed in SEEDS
    )
    return RunResult(manifest=manifest, episodes=episodes)


# --- Flag semantics at the run level ------------------------------------------------------------


def test_an_unperturbed_run_is_not_flagged() -> None:
    run = _run()
    assert is_flagged(run) is False
    assert flagged_episodes(run) == ()


def test_an_injected_perturbation_flags_the_run() -> None:
    run = _run(_shift)
    assert is_flagged(run) is True
    assert flagged_episodes(run) == run.episodes
    for episode in run.episodes:
        assert episode.flags, f"seed {episode.seed} was perturbed but not flagged"
        assert {flag.monitor for flag in episode.flags} == {"position-bound"}
        assert episode.flags[0].step == 0


def test_a_flag_is_raised_at_the_step_the_envelope_is_left() -> None:
    episode = run_episode(
        FakeEnv(success_rate=1.0, perturb=_shift_after(2)),
        FakePolicy(),
        seed=1000,
        max_steps=MAX_STEPS,
        monitors=[_suite()],
    )
    # The reset and the first step's observation are clean; step 2 is the first to see the shift.
    assert episode.flags
    assert episode.flags[0].step == 2


def test_one_flagged_episode_flags_the_whole_run() -> None:
    clean, perturbed = _run(), _run(_shift)
    mixed = RunResult(manifest=clean.manifest, episodes=(*clean.episodes, perturbed.episodes[0]))
    assert is_flagged(mixed) is True
    assert flagged_episodes(mixed) == (perturbed.episodes[0],)


def test_a_successful_episode_outside_the_envelope_is_still_flagged() -> None:
    run = _run(_shift, success_rate=1.0)
    assert all(episode.success for episode in run.episodes)
    assert is_flagged(run) is True


def test_is_flagged_reads_a_single_episode() -> None:
    clean, perturbed = _run().episodes[0], _run(_shift).episodes[0]
    assert is_flagged(clean) is False
    assert is_flagged(perturbed) is True


def test_a_flag_survives_persistence() -> None:
    run = _run(_shift)
    restored = from_document(json.loads(json.dumps(to_document(run))))
    assert is_flagged(restored) is True
    assert flagged_episodes(restored) == run.episodes


# --- The per-step hook: MonitorSuite ------------------------------------------------------------


def test_the_suite_is_a_monitor_the_loop_accepts() -> None:
    suite: Monitor = _suite()
    assert suite.name == "suite[position-bound]"


def test_the_suite_does_not_change_the_trajectory() -> None:
    def episode(monitors: Sequence[Monitor]) -> EpisodeResult:
        env = FakeEnv(success_rate=1.0, perturb=_shift)
        return run_episode(env, FakePolicy(), seed=1000, max_steps=MAX_STEPS, monitors=monitors)

    assert episode([_suite()]).action_hash == episode([]).action_hash


def test_the_suite_runs_every_monitor_in_order() -> None:
    suite = MonitorSuite([PositionBound(name="a"), PositionBound(name="b")])
    episode = run_episode(
        FakeEnv(success_rate=1.0, perturb=_shift),
        FakePolicy(),
        seed=1000,
        max_steps=MAX_STEPS,
        monitors=[suite],
    )
    # The policy steers the observed position back inside the bound, so only the
    # first few steps flag; at each of those, both monitors flag, in order.
    assert episode.flags
    pairs = [episode.flags[i : i + 2] for i in range(0, len(episode.flags), 2)]
    for a, b in pairs:
        assert (a.monitor, b.monitor) == ("a", "b")
        assert a.step == b.step


def test_an_empty_suite_is_refused() -> None:
    # A run with no monitor reads as clean without anything having looked.
    with pytest.raises(ValueError, match="at least one monitor"):
        MonitorSuite([])


def test_duplicate_monitor_names_are_refused() -> None:
    with pytest.raises(ValueError, match="position-bound"):
        MonitorSuite([PositionBound(), PositionBound()])


class _Misattributing:
    def __init__(self, *, monitor: str = "liar", step_offset: int = 0) -> None:
        self._monitor = monitor
        self._step_offset = step_offset

    @property
    def name(self) -> str:
        return "liar"

    def on_step(self, record: StepRecord) -> Sequence[Flag]:
        return [Flag(monitor=self._monitor, step=record.index + self._step_offset, reason="r")]


def test_a_flag_attributed_to_another_monitor_is_refused() -> None:
    suite = MonitorSuite([_Misattributing(monitor="someone-else")])
    with pytest.raises(MonitorContractError, match="liar"):
        run_episode(FakeEnv(success_rate=1.0), FakePolicy(), seed=1, max_steps=1, monitors=[suite])


def test_a_flag_for_another_step_is_refused() -> None:
    suite = MonitorSuite([_Misattributing(step_offset=1)])
    with pytest.raises(MonitorContractError, match="step 1"):
        run_episode(FakeEnv(success_rate=1.0), FakePolicy(), seed=1, max_steps=1, monitors=[suite])


def test_a_monitor_that_raises_stops_the_run() -> None:
    # A crashed monitor must not leave behind a result that reads as unflagged.
    class Boom:
        name = "boom"

        def on_step(self, record: StepRecord) -> Sequence[Flag]:
            raise RuntimeError("monitor crashed")

    with pytest.raises(RuntimeError, match="monitor crashed"):
        run_episode(
            FakeEnv(success_rate=1.0),
            FakePolicy(),
            seed=1,
            max_steps=1,
            monitors=[MonitorSuite([Boom()])],
        )
