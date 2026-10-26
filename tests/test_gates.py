"""Contract tests for the gate (ruling D6).

The gate statistic is the Clopper-Pearson exact 95% lower bound on the success
rate, compared against a declared threshold, plus a declared minimum number of
episodes. The lane is done when an under-performing fixture fails and an
insufficient sample fails; both are asserted here against `FakeEnv`, whose true
success rate is a dial.

The bound is checked against its definition in exact rational arithmetic, not
only against tabulated values, so a test passing does not depend on the same
floating-point code path as the thing it tests.
"""

from __future__ import annotations

import dataclasses
import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

from fathomline.cli import main
from fathomline.core import EpisodeResult, Flag, RunManifest, RunResult, run_episode, to_document
from fathomline.fakes import FakeEnv, FakePolicy
from fathomline.gates import GateConfig, GateVerdict, clopper_pearson_lower, evaluate
from fathomline.gates.command import EXIT_CANNOT_JUDGE, EXIT_FAILED, EXIT_PASSED

# --- the bound ------------------------------------------------------------------------------


def _exact_upper_tail(successes: int, episodes: int, p: Fraction) -> Fraction:
    """P(X >= successes) for X ~ Binomial(episodes, p), exactly."""
    return sum(
        (
            math.comb(episodes, i) * p**i * (1 - p) ** (episodes - i)
            for i in range(successes, episodes + 1)
        ),
        start=Fraction(0),
    )


def test_zero_successes_has_a_lower_bound_of_zero() -> None:
    assert clopper_pearson_lower(0, 10) == 0.0
    assert clopper_pearson_lower(0, 0) == 0.0


@pytest.mark.parametrize("episodes", [1, 5, 10, 30, 100, 1000])
def test_all_successes_matches_the_closed_form(episodes: int) -> None:
    """With k = n the tail is p**n, so the bound is (alpha/2) ** (1/n)."""
    assert clopper_pearson_lower(episodes, episodes) == pytest.approx(
        0.025 ** (1 / episodes), rel=1e-8
    )


@pytest.mark.parametrize("episodes", [1, 5, 10, 30, 100, 1000])
def test_one_success_matches_the_closed_form(episodes: int) -> None:
    """With k = 1 the tail is 1 - (1-p)**n, so the bound is 1 - (1 - alpha/2) ** (1/n)."""
    assert clopper_pearson_lower(1, episodes) == pytest.approx(
        1 - 0.975 ** (1 / episodes), rel=1e-8
    )


@pytest.mark.parametrize(
    ("successes", "episodes", "expected"),
    [
        # Two-sided 95% Clopper-Pearson interval, lower endpoint; the textbook
        # example (statsmodels `proportion_confint(5, 10, method="beta")`).
        (5, 10, 0.18708602844739),
        (50, 100, 0.39832112950330),
    ],
)
def test_matches_tabulated_values(successes: int, episodes: int, expected: float) -> None:
    assert clopper_pearson_lower(successes, episodes) == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("episodes", range(1, 26))
def test_the_bound_satisfies_its_definition_exactly(episodes: int) -> None:
    """The bound p_L solves P(X >= k | p_L) = alpha/2. Checked in exact arithmetic.

    It must never be overstated: at the returned value the tail is at most
    alpha/2. It must not be needlessly loose either: a relative 1e-8 above it
    (ten times the documented shading margin) the tail already exceeds alpha/2.
    """
    target = Fraction(1, 40)
    for successes in range(1, episodes + 1):
        bound = clopper_pearson_lower(successes, episodes)
        assert _exact_upper_tail(successes, episodes, Fraction(bound)) <= target
        assert _exact_upper_tail(successes, episodes, Fraction(bound * (1 + 1e-8))) > target


def test_the_bound_rises_with_successes_and_with_sample_size_at_a_fixed_rate() -> None:
    by_successes = [clopper_pearson_lower(k, 50) for k in range(51)]
    assert by_successes == sorted(by_successes)
    assert len(set(by_successes)) == 51
    # The same observed 80% rate is stronger evidence on a bigger sample.
    by_size = [clopper_pearson_lower(4 * m, 5 * m) for m in (1, 2, 4, 8, 16, 32)]
    assert by_size == sorted(by_size)


def test_a_wider_confidence_gives_a_lower_bound() -> None:
    assert clopper_pearson_lower(8, 10, confidence=0.99) < clopper_pearson_lower(8, 10)
    assert clopper_pearson_lower(8, 10, confidence=0.80) > clopper_pearson_lower(8, 10)


def test_a_large_sample_does_not_overflow() -> None:
    bound = clopper_pearson_lower(9_000, 10_000)
    assert 0.89 < bound < 0.9


@pytest.mark.parametrize(
    ("successes", "episodes", "confidence"),
    [(-1, 10, 0.95), (11, 10, 0.95), (0, -1, 0.95), (5, 10, 0.0), (5, 10, 1.0)],
)
def test_impossible_inputs_are_rejected(successes: int, episodes: int, confidence: float) -> None:
    with pytest.raises(ValueError):
        clopper_pearson_lower(successes, episodes, confidence=confidence)


# --- the config -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 0.0, "min_episodes": 10},
        {"threshold": 1.0, "min_episodes": 10},
        {"threshold": 0.8, "min_episodes": 0},
        {"threshold": 0.8, "min_episodes": 10, "confidence": 1.0},
        {"threshold": 0.8, "min_episodes": 10, "confidence": 0.0},
    ],
)
def test_a_gate_that_cannot_fail_or_cannot_pass_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        GateConfig(**kwargs)  # type: ignore[arg-type]


def test_the_default_confidence_is_95_percent() -> None:
    assert GateConfig(threshold=0.8, min_episodes=10).confidence == 0.95


@pytest.mark.parametrize("confidence", [0.01, 0.5, 0.90, 0.9499])
def test_a_gate_weaker_than_95_percent_is_rejected(confidence: float) -> None:
    """D6 fixes the statistic at 95%. Weakening it would be an amendment, not a setting.

    Without this, 5/5 at confidence 0.01 bounds at about 0.87 and passes an 0.8 gate.
    """
    with pytest.raises(ValueError, match=r"confidence must be at least 0\.95"):
        GateConfig(threshold=0.8, min_episodes=5, confidence=confidence)


# --- the gate on the fixture ----------------------------------------------------------------


def _run(*, success_rate: float, episodes: int) -> RunResult:
    env, policy = FakeEnv(success_rate=success_rate), FakePolicy()
    seeds = tuple(range(episodes))
    return RunResult(
        manifest=RunManifest.capture(env=env, policy=policy, seeds=seeds, max_steps=10),
        episodes=tuple(run_episode(env, policy, seed=seed, max_steps=10) for seed in seeds),
    )


GATE = GateConfig(threshold=0.8, min_episodes=30)


def test_a_good_fixture_on_a_sufficient_sample_passes() -> None:
    verdict = evaluate(_run(success_rate=1.0, episodes=100), GATE)
    assert verdict.passed is True
    assert verdict.reasons == ()
    assert verdict.successes == 100
    assert verdict.episodes == 100
    assert verdict.lower_bound == pytest.approx(0.025 ** (1 / 100))


def test_an_under_performing_fixture_fails() -> None:
    """Done-when, part one."""
    verdict = evaluate(_run(success_rate=0.5, episodes=100), GATE)
    assert verdict.passed is False
    assert verdict.lower_bound < GATE.threshold
    assert len(verdict.reasons) == 1
    assert "lower bound" in verdict.reasons[0]


def test_an_insufficient_sample_fails_even_when_every_episode_succeeds() -> None:
    """Done-when, part two: a small sample fails by construction."""
    verdict = evaluate(_run(success_rate=1.0, episodes=29), GATE)
    assert verdict.passed is False
    assert verdict.successes == 29
    assert any("insufficient sample" in reason for reason in verdict.reasons)


def test_the_minimum_sample_is_enforced_even_when_the_bound_clears() -> None:
    """The two checks are independent: a low threshold does not waive minimum n."""
    lenient = GateConfig(threshold=0.1, min_episodes=30)
    verdict = evaluate(_run(success_rate=1.0, episodes=10), lenient)
    assert verdict.lower_bound >= lenient.threshold
    assert verdict.passed is False
    assert verdict.reasons == ("insufficient sample: 10 episodes, minimum 30",)


def test_the_bound_alone_fails_a_small_perfect_sample() -> None:
    """Without minimum n, 5/5 still fails an 0.8 gate: its lower bound is about 0.48."""
    verdict = evaluate(_run(success_rate=1.0, episodes=5), GateConfig(0.8, min_episodes=1))
    assert verdict.passed is False
    assert verdict.lower_bound < 0.5


def test_an_empty_run_fails() -> None:
    verdict = evaluate(_run(success_rate=1.0, episodes=0), GATE)
    assert verdict.passed is False
    assert verdict.lower_bound == 0.0
    assert len(verdict.reasons) == 2


def test_a_flagged_run_never_passes() -> None:
    """Ruling F's seam: a flag on any episode fails the gate, whatever the rate.

    Producing flags is Lane F's job. The gate only has to refuse to pass a run
    that carries one, so it is tested with a flag set by hand.
    """
    clean = _run(success_rate=1.0, episodes=100)
    flagged_episode: EpisodeResult = dataclasses.replace(
        clean.episodes[0], flags=(Flag(monitor="test", step=0, reason="outside the envelope"),)
    )
    flagged = RunResult(manifest=clean.manifest, episodes=(flagged_episode, *clean.episodes[1:]))
    verdict = evaluate(flagged, GATE)
    assert verdict.passed is False
    assert verdict.flagged_episodes == 1
    assert verdict.reasons == ("flagged: 1 episode(s) carry monitor flags",)


def test_a_flag_on_a_failed_episode_still_fails_the_run() -> None:
    """A flag counts whether or not its episode succeeded.

    99/100 bounds at about 0.95 and clears an 0.8 gate, so the flag on the one
    failed episode is the only thing that may fail this run.
    """
    clean = _run(success_rate=1.0, episodes=100)
    failed_and_flagged: EpisodeResult = dataclasses.replace(
        clean.episodes[0],
        success=False,
        flags=(Flag(monitor="test", step=0, reason="outside the envelope"),),
    )
    run = RunResult(manifest=clean.manifest, episodes=(failed_and_flagged, *clean.episodes[1:]))
    verdict = evaluate(run, GATE)
    assert verdict.successes == 99
    assert verdict.lower_bound >= GATE.threshold
    assert verdict.passed is False
    assert verdict.flagged_episodes == 1
    assert verdict.reasons == ("flagged: 1 episode(s) carry monitor flags",)


def test_the_verdict_records_what_it_was_judged_against() -> None:
    verdict = evaluate(_run(success_rate=1.0, episodes=100), GATE)
    assert isinstance(verdict, GateVerdict)
    assert verdict.config == GATE
    assert verdict.flagged_episodes == 0


# --- the `gate` subcommand ------------------------------------------------------------------


def _write(tmp_path: Path, run: RunResult) -> str:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(to_document(run)), encoding="utf-8")
    return str(path)


def _gate(path: str, *extra: str) -> int:
    return main(["gate", path, "--threshold", "0.8", "--min-episodes", "30", *extra])


def test_the_gate_command_passes_a_good_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _gate(_write(tmp_path, _run(success_rate=1.0, episodes=100))) == EXIT_PASSED
    out = capsys.readouterr().out
    assert out.startswith("gate PASSED")
    assert "100/100" in out


def test_the_gate_command_fails_an_under_performing_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _gate(_write(tmp_path, _run(success_rate=0.5, episodes=100))) == EXIT_FAILED
    out = capsys.readouterr().out
    assert out.startswith("gate FAILED")
    assert "lower bound" in out


def test_the_gate_command_fails_an_insufficient_sample(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _gate(_write(tmp_path, _run(success_rate=1.0, episodes=10))) == EXIT_FAILED
    assert "insufficient sample: 10 episodes, minimum 30" in capsys.readouterr().out


def test_the_gate_command_judges_at_95_percent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """30/30 with a minimum of 30: exactly at minimum n, and bounds at 0.884."""
    path = _write(tmp_path, _run(success_rate=1.0, episodes=30))
    assert _gate(path) == EXIT_PASSED
    assert "95% confidence" in capsys.readouterr().out


def test_the_gate_command_has_no_confidence_knob(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """D6 fixes the confidence at 95%, so the CLI offers no way to change it."""
    path = _write(tmp_path, _run(success_rate=1.0, episodes=5))
    argv = ["gate", path, "--threshold", "0.8", "--min-episodes", "5", "--confidence", "0.01"]
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert "unrecognized arguments: --confidence" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--threshold", "--min-episodes"])
def test_the_gate_command_requires_a_declared_threshold_and_minimum(
    tmp_path: Path, flag: str
) -> None:
    """No defaults: a gate whose bar is implicit is not a declared gate."""
    path = _write(tmp_path, _run(success_rate=1.0, episodes=100))
    argv = ["gate", path, "--threshold", "0.8", "--min-episodes", "30"]
    del argv[argv.index(flag) : argv.index(flag) + 2]
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2


def test_the_gate_command_rejects_an_impossible_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, _run(success_rate=1.0, episodes=100))
    assert main(["gate", path, "--threshold", "1.5", "--min-episodes", "30"]) == EXIT_CANNOT_JUDGE
    captured = capsys.readouterr()
    assert "threshold must be in (0, 1)" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    "content",
    [
        None,  # the file does not exist
        "not json",
        "[1, 2, 3]",
        '{"schema_version": 999}',
        '{"schema_version": 1}',
    ],
)
def test_the_gate_command_cannot_judge_an_unreadable_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str | None
) -> None:
    """Fail closed, and distinctly: an unreadable result is not a failed gate."""
    path = tmp_path / "result.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    assert _gate(str(path)) == EXIT_CANNOT_JUDGE
    captured = capsys.readouterr()
    assert "cannot read result" in captured.err
    assert captured.out == ""
