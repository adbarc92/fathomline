"""Lane G: the instruction monitor (ruling D10, condition 4), on the fake.

D10 rules the condition as: **the instruction is in the declared set, exact match.
Harmless paraphrases are flagged by design.** In the envelope spike, task 1's
instruction on task 0's scene gave 0/30, and a paraphrase showed no measurable
harm; both are outside what was tested, so both are flagged.

The instruction reaches a monitor the way it reaches the policy: as the
observation's `task` entry, a one-element list for the fixture's one-wide env.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from conftest import SuiteFactory

from fathomline.core import Observation, RunManifest, RunResult, StepRecord, run_episode
from fathomline.fakes import FakeEnv, FakePolicy
from fathomline.gates import GateConfig, evaluate
from fathomline.monitors import (
    LIBERO_INSTRUCTIONS,
    InstructionSet,
    is_flagged,
)

TASK_0 = "pick up the alphabet soup and place it in the basket"
PARAPHRASE = "grab the alphabet soup can and put it into the basket"
TASK_1 = "pick up the cream cheese and place it in the basket"
SEEDS = tuple(range(1000, 1010))
MAX_STEPS = 10


def _record(task: object, index: int = 0) -> StepRecord:
    return StepRecord(index=index, observation={"task": task}, action=(0.0,), reward=0.0)


def _monitor() -> InstructionSet:
    return InstructionSet(declared={TASK_0})


# --- The ruled declaration ----------------------------------------------------------------------


def test_the_declared_set_is_the_fixture_task_instruction_alone() -> None:
    assert frozenset({TASK_0}) == LIBERO_INSTRUCTIONS


def test_the_fixture_monitors_include_the_instruction_monitor(
    fixture_suite: SuiteFactory,
) -> None:
    suite = fixture_suite()
    assert "instruction-set" in suite.name


def _fixture_step(task: object) -> StepRecord:
    # A LIBERO-shaped action inside every bound, so only the instruction can flag.
    return StepRecord(index=0, observation={"task": task}, action=(0.0,) * 7, reward=0.0)


def test_the_fixture_suite_passes_the_task_own_instruction(fixture_suite: SuiteFactory) -> None:
    assert list(fixture_suite().on_step(_fixture_step([TASK_0]))) == []


@pytest.mark.parametrize("instruction", [TASK_1, PARAPHRASE])
def test_the_fixture_suite_flags_any_other_instruction(
    instruction: str, fixture_suite: SuiteFactory
) -> None:
    # The suite the fixture runs with, not only the constant: a widened set would pass these.
    (flag,) = fixture_suite().on_step(_fixture_step([instruction]))
    assert flag.monitor == "instruction-set"


# --- One step -----------------------------------------------------------------------------------


@pytest.mark.parametrize("task", [[TASK_0], (TASK_0,), TASK_0])
def test_a_declared_instruction_is_not_flagged(task: object) -> None:
    assert list(_monitor().on_step(_record(task))) == []


@pytest.mark.parametrize(
    "instruction",
    [
        TASK_1,
        PARAPHRASE,
        TASK_0.capitalize(),
        TASK_0 + " ",
        TASK_0 + ".",
        "",
    ],
)
def test_anything_but_an_exact_match_is_flagged(instruction: str) -> None:
    # No normalization of case, whitespace or punctuation: "declared" means these exact strings.
    (flag,) = _monitor().on_step(_record([instruction], index=4))
    assert (flag.monitor, flag.step) == ("instruction-set", 4)
    assert repr(instruction) in flag.reason


def test_one_undeclared_entry_in_a_batch_flags_the_step() -> None:
    (flag,) = _monitor().on_step(_record([TASK_0, TASK_1]))
    assert repr(TASK_1) in flag.reason
    assert repr(TASK_0) not in flag.reason


def test_the_monitor_name_and_key_are_configurable() -> None:
    monitor = InstructionSet(declared={"go"}, key="language", name="lang")
    record = StepRecord(index=0, observation={"language": ["stop"]}, action=(0.0,), reward=0.0)
    (flag,) = monitor.on_step(record)
    assert flag.monitor == "lang"


def test_a_step_with_no_instruction_is_refused() -> None:
    # Treating a missing instruction as "nothing undeclared" would pass a step nothing checked.
    record = StepRecord(index=0, observation={}, action=(0.0,), reward=0.0)
    with pytest.raises(ValueError, match="'task'"):
        _monitor().on_step(record)


@pytest.mark.parametrize("task", [[], ()])
def test_an_empty_instruction_list_is_refused(task: object) -> None:
    with pytest.raises(ValueError, match="no instruction"):
        _monitor().on_step(_record(task))


@pytest.mark.parametrize("task", [None, 3, [TASK_0, 3], {TASK_0}])
def test_an_instruction_that_is_not_text_is_refused(task: object) -> None:
    with pytest.raises(TypeError, match="instruction"):
        _monitor().on_step(_record(task))


# --- Construction -------------------------------------------------------------------------------


def test_an_empty_declared_set_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        InstructionSet(declared=set())


def test_a_bare_string_as_the_declared_set_is_refused() -> None:
    # A string is a collection of characters: it would declare 'p', 'i', 'c', ...
    with pytest.raises(TypeError, match="not a string"):
        InstructionSet(declared=TASK_0)


def test_a_declared_instruction_that_is_not_text_is_refused() -> None:
    with pytest.raises(TypeError, match="declared"):
        InstructionSet(declared={TASK_0, 3})  # type: ignore[arg-type]


def test_the_declared_set_is_copied() -> None:
    declared = {TASK_0}
    monitor = InstructionSet(declared=declared)
    declared.add(TASK_1)
    assert monitor.on_step(_record([TASK_1]))


# --- Through the loop, with the instruction injected --------------------------------------------


def _with_task(instruction: str) -> Callable[[Observation], Observation]:
    def perturb(observation: Observation) -> Observation:
        return {**observation, "task": [instruction]}

    return perturb


def _run(instruction: str) -> RunResult:
    env, policy = FakeEnv(success_rate=1.0, perturb=_with_task(instruction)), FakePolicy()
    manifest = RunManifest.capture(env=env, policy=policy, seeds=SEEDS, max_steps=MAX_STEPS)
    suite = InstructionSet(declared=LIBERO_INSTRUCTIONS)
    episodes = tuple(
        run_episode(env, policy, seed=seed, max_steps=MAX_STEPS, monitors=[suite]) for seed in SEEDS
    )
    return RunResult(manifest=manifest, episodes=episodes)


def test_a_run_on_the_declared_instruction_is_not_flagged() -> None:
    assert is_flagged(_run(TASK_0)) is False


@pytest.mark.parametrize("instruction", [TASK_1, PARAPHRASE])
def test_a_run_on_another_instruction_is_flagged_from_its_first_step(instruction: str) -> None:
    run = _run(instruction)
    for episode in run.episodes:
        assert episode.flags[0].step == 0
        assert {flag.monitor for flag in episode.flags} == {"instruction-set"}


def test_a_paraphrased_run_fails_the_gate_even_when_every_episode_succeeds() -> None:
    run = _run(PARAPHRASE)
    assert all(episode.success for episode in run.episodes)
    verdict = evaluate(run, GateConfig(threshold=0.5, min_episodes=len(SEEDS)))
    assert verdict.passed is False
    assert verdict.flagged_episodes == len(SEEDS)
