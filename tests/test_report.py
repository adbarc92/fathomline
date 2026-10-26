"""Contract tests for Lane D: the report is rendered from persisted JSON, never from a re-run.

Ruling D12: versioned JSON is canonical and Markdown is rendered from it. So the
report reads a run directory and nothing else. The golden fixture below was
written by hand, not produced by the loop: its action hashes could not come out
of a re-run, so a report that shows them can only have read them from disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fathomline.cli import main
from fathomline.core import (
    ArtifactRef,
    EpisodeResult,
    Flag,
    RunManifest,
    RunResult,
    run_episode,
    to_document,
)
from fathomline.fakes import FakeEnv, FakePolicy
from fathomline.report import (
    GATE_FILENAME,
    RESULT_FILENAME,
    GateVerdict,
    ReportError,
    load_gate,
    load_run,
    render_markdown,
    report_run_dir,
)

FIXTURES = Path(__file__).parent / "fixtures" / "report"


def _manifest(*, seeds: tuple[int, ...] = (1000, 1001, 1002)) -> RunManifest:
    return RunManifest(
        fathomline_version="0.0.1",
        policy=ArtifactRef(name="fake-policy", revision="0"),
        env=ArtifactRef(name="fake-env", revision="0"),
        seeds=seeds,
        max_steps=10,
        python_version="3.12.5",
        platform="Linux-x86_64",
    )


def _episode(seed: int, *, success: bool = True, flags: tuple[Flag, ...] = ()) -> EpisodeResult:
    return EpisodeResult(
        seed=seed,
        steps=5,
        success=success,
        sum_reward=-1.25,
        max_reward=-0.125,
        action_hash=(f"{seed:x}" * 22)[:64],
        flags=flags,
    )


def _clean_run() -> RunResult:
    return RunResult(
        manifest=_manifest(),
        episodes=(_episode(1000), _episode(1001), _episode(1002, success=False)),
    )


def _flagged_run() -> RunResult:
    flags = (
        Flag(monitor="action-bounds", step=2, reason="action 1.7 exceeds bound 1.0"),
        Flag(monitor="obs-shift", step=4, reason="position drifted"),
    )
    return RunResult(
        manifest=_manifest(),
        episodes=(_episode(1000), _episode(1001, flags=flags), _episode(1002, success=False)),
    )


def _failed_and_flagged_run() -> RunResult:
    """Lane F's rule: one flag flags the run, whether its episode succeeded or not."""
    flag = Flag(monitor="obs-shift", step=3, reason="position drifted")
    return RunResult(
        manifest=_manifest(),
        episodes=(_episode(1000), _episode(1001), _episode(1002, success=False, flags=(flag,))),
    )


def _gate(
    *,
    passed: bool,
    successes: int = 2,
    n: int = 3,
    min_n: int = 3,
    threshold: float = 0.05,
    lower_bound: float = 0.0943,
) -> GateVerdict:
    """By default a verdict on 2 / 3 that agrees with the clean and flagged runs above.

    The defaults sit on the D6 boundaries (n == min_n) and clear the threshold, so
    a PASS built from them is self-consistent and any contradiction is one a test
    put there on purpose.
    """
    return GateVerdict(
        passed=passed,
        successes=successes,
        n=n,
        min_n=min_n,
        threshold=threshold,
        lower_bound=lower_bound,
    )


def _persist(run_dir: Path, result: RunResult) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RESULT_FILENAME).write_text(json.dumps(to_document(result)), encoding="utf-8")
    return run_dir


def _write_gate(run_dir: Path, document: object) -> None:
    (run_dir / GATE_FILENAME).write_text(json.dumps(document), encoding="utf-8")


def _gate_document(**overrides: object) -> dict[str, object]:
    document: dict[str, object] = {
        "schema_version": 1,
        "passed": False,
        "successes": 2,
        "n": 3,
        "min_n": 30,
        "threshold": 0.8,
        "lower_bound": 0.0943,
    }
    document.update(overrides)
    return document


# --- Rendering ---------------------------------------------------------------


def test_a_clean_run_reports_provenance_summary_and_every_episode() -> None:
    markdown = render_markdown(_clean_run())

    assert markdown.startswith("# Fathomline run report\n")
    for expected in (
        "| Fathomline version | 0.0.1 |",
        "| Policy | fake-policy @ 0 |",
        "| Env | fake-env @ 0 |",
        "| Seeds | 1000, 1001, 1002 |",
        "| Max steps | 10 |",
        "| Python | 3.12.5 |",
        "| Platform | Linux-x86_64 |",
        "| Episodes | 3 |",
        "| Successes | 2 |",
        "| Success rate | 66.7% |",
        "| Flagged episodes | 0 |",
        "| 1000 | yes | 5 | -1.2500 | -0.1250 | 0 | `3e83e83e83e8` |",
        "| 1002 | no | 5 | -1.2500 | -0.1250 | 0 | `3ea3ea3ea3ea` |",
    ):
        assert expected in markdown
    assert "FLAGGED" not in markdown


def test_a_clean_run_says_no_flags_were_raised_rather_than_omitting_the_section() -> None:
    markdown = render_markdown(_clean_run())
    assert "## Monitor flags\n\nNone raised.\n" in markdown


def test_a_flagged_run_is_flagged_before_any_result_is_reported() -> None:
    """Lane F's rule as the report keeps it: a flagged run is never silently reported."""
    markdown = render_markdown(_flagged_run())

    assert markdown.startswith("# Fathomline run report: FLAGGED\n")
    banner = markdown.index("**FLAGGED: 1 of 3 episodes raised 2 monitor flags.**")
    assert banner < markdown.index("## Summary")
    assert "| Flagged episodes | 1 |" in markdown


def test_every_flag_is_listed_with_its_episode_step_monitor_and_reason() -> None:
    markdown = render_markdown(_flagged_run())

    assert "| 1001 | 2 | action-bounds | action 1.7 exceeds bound 1.0 |" in markdown
    assert "| 1001 | 4 | obs-shift | position drifted |" in markdown
    assert "| 1001 | yes | 5 | -1.2500 | -0.1250 | 2 |" in markdown


def test_a_flag_on_a_failed_episode_flags_the_run() -> None:
    markdown = render_markdown(_failed_and_flagged_run())

    assert markdown.startswith("# Fathomline run report: FLAGGED\n")
    banner = markdown.index("**FLAGGED: 1 of 3 episodes raised 1 monitor flag.**")
    assert banner < markdown.index("## Summary")
    assert "| Flagged episodes | 1 |" in markdown


def test_a_flag_on_a_failed_episode_is_listed() -> None:
    markdown = render_markdown(_failed_and_flagged_run())

    assert "| 1002 | 3 | obs-shift | position drifted |" in markdown
    assert "| 1002 | no | 5 | -1.2500 | -0.1250 | 1 |" in markdown
    assert "None raised." not in markdown


def test_without_a_gate_verdict_the_gate_is_reported_as_not_evaluated() -> None:
    markdown = render_markdown(_clean_run())
    assert "## Gate\n\nNot evaluated: this run has no gate verdict.\n" in markdown


@pytest.mark.parametrize(("passed", "word"), [(True, "PASS"), (False, "FAIL")])
def test_a_gate_verdict_is_rendered_with_its_statistic(passed: bool, word: str) -> None:
    markdown = render_markdown(_clean_run(), gate=_gate(passed=passed))

    for expected in (
        f"| Verdict | **{word}** |",
        "| Successes / n | 2 / 3 |",
        "| Minimum n | 3 |",
        "| Clopper-Pearson 95% lower bound | 0.0943 |",
        "| Threshold | 0.0500 |",
    ):
        assert expected in markdown
    assert "Contradiction" not in markdown


def test_a_passing_verdict_on_a_flagged_run_is_called_out_as_invalid() -> None:
    """The gate never passes a flagged run; if a verdict says it did, the report says so."""
    markdown = render_markdown(_flagged_run(), gate=_gate(passed=True))
    assert "**Contradiction:** the gate verdict is PASS but this run is flagged." in markdown
    assert markdown.count("Contradiction") == 1


@pytest.mark.parametrize("passed", [True, False])
@pytest.mark.parametrize(("successes", "n"), [(3, 3), (2, 4), (30, 30)])
def test_a_verdict_whose_counts_disagree_with_the_run_is_called_out(
    passed: bool, successes: int, n: int
) -> None:
    """`gate.json` names no run, so a stale or misplaced verdict is caught by its counts."""
    gate = _gate(passed=passed, successes=successes, n=n, min_n=1, threshold=0.0)

    markdown = render_markdown(_clean_run(), gate=gate)

    assert (
        f"**Contradiction:** the gate verdict counts {successes} / {n} but this run has "
        "2 / 3. The verdict is not about this run;\n> treat it as invalid."
    ) in markdown
    assert markdown.count("Contradiction") == 1


def test_a_passing_verdict_below_the_minimum_sample_is_called_out() -> None:
    """D6: a small sample fails by construction, so a PASS on one is not a real verdict."""
    markdown = render_markdown(_clean_run(), gate=_gate(passed=True, min_n=30))

    assert (
        "**Contradiction:** the gate verdict is PASS but n = 3 is below the minimum 30. "
        "A sample that small\n> fails by construction (D6); treat this verdict as invalid."
    ) in markdown
    assert markdown.count("Contradiction") == 1


def test_a_passing_verdict_whose_lower_bound_misses_the_threshold_is_called_out() -> None:
    markdown = render_markdown(_clean_run(), gate=_gate(passed=True, threshold=0.8))

    assert (
        "**Contradiction:** the gate verdict is PASS but its lower bound 0.0943 is below the "
        "threshold 0.8000.\n> D6 passes only when the bound reaches the threshold; "
        "treat this verdict as invalid."
    ) in markdown
    assert markdown.count("Contradiction") == 1


def test_a_lower_bound_exactly_at_the_threshold_is_not_a_contradiction() -> None:
    """D6 is `lower bound >= threshold`; the boundary passes."""
    gate = _gate(passed=True, threshold=0.0943, lower_bound=0.0943)
    assert "Contradiction" not in render_markdown(_clean_run(), gate=gate)


def test_a_failing_verdict_on_a_small_sample_is_not_a_contradiction() -> None:
    gate = _gate(passed=False, min_n=30, threshold=0.8)
    assert "Contradiction" not in render_markdown(_clean_run(), gate=gate)


def test_every_contradiction_in_a_verdict_is_reported() -> None:
    gate = _gate(passed=True, successes=30, n=30, min_n=40, threshold=0.9, lower_bound=0.88)

    markdown = render_markdown(_flagged_run(), gate=gate)

    assert markdown.count("**Contradiction:**") == 4


def test_a_failing_verdict_on_a_flagged_run_is_not_a_contradiction() -> None:
    markdown = render_markdown(_flagged_run(), gate=_gate(passed=False))
    assert "Contradiction" not in markdown


def test_a_run_with_no_episodes_reports_an_undefined_rate_rather_than_dividing_by_zero() -> None:
    markdown = render_markdown(RunResult(manifest=_manifest(seeds=()), episodes=()))

    assert "| Episodes | 0 |" in markdown
    assert "| Success rate | n/a |" in markdown
    assert "| Seeds | none |" in markdown
    assert "## Episodes\n\nNone.\n" in markdown


def test_text_from_the_document_cannot_break_a_table_row() -> None:
    flag = Flag(monitor="a|b", step=0, reason="line one\nline two \\ end")
    result = RunResult(manifest=_manifest(seeds=(7,)), episodes=(_episode(7, flags=(flag,)),))

    markdown = render_markdown(result)

    assert "| 7 | 0 | a\\|b | line one line two \\\\ end |" in markdown


def test_rendering_is_deterministic_and_ascii() -> None:
    first = render_markdown(_flagged_run(), gate=_gate(passed=False))
    assert first == render_markdown(_flagged_run(), gate=_gate(passed=False))
    assert first.isascii()
    assert first.endswith("\n")


# --- Loading a run directory --------------------------------------------------


def test_a_persisted_run_loads_back_to_the_same_result(tmp_path: Path) -> None:
    result = _flagged_run()
    assert load_run(_persist(tmp_path, result)) == result


def test_a_run_from_the_loop_is_reported_from_disk(tmp_path: Path) -> None:
    episodes = tuple(
        run_episode(FakeEnv(success_rate=1.0), FakePolicy(), seed=seed, max_steps=10)
        for seed in (1000, 1001)
    )
    result = RunResult(manifest=_manifest(seeds=(1000, 1001)), episodes=episodes)

    markdown = report_run_dir(_persist(tmp_path, result))

    assert markdown == render_markdown(result)
    assert f"`{episodes[0].action_hash[:12]}`" in markdown


def test_a_missing_result_is_a_report_error(tmp_path: Path) -> None:
    with pytest.raises(ReportError, match=r"cannot read .*result\.json"):
        load_run(tmp_path / "absent")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "not valid JSON"),
        ("[1, 2]", "not a JSON object"),
        ('{"schema_version": 99}', "unsupported schema_version 99"),
        ('{"schema_version": 1}', "missing field 'manifest'"),
    ],
)
def test_an_unreadable_result_is_a_report_error(tmp_path: Path, content: str, message: str) -> None:
    (tmp_path / RESULT_FILENAME).write_text(content, encoding="utf-8")
    with pytest.raises(ReportError, match=message):
        load_run(tmp_path)


def test_a_result_with_a_mistyped_field_is_a_report_error(tmp_path: Path) -> None:
    document = to_document(_clean_run())
    document["manifest"]["policy"] = {"name": "p"}
    (tmp_path / RESULT_FILENAME).write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ReportError, match="malformed"):
        load_run(tmp_path)


def test_a_run_directory_without_a_gate_verdict_has_none(tmp_path: Path) -> None:
    assert load_gate(_persist(tmp_path, _clean_run())) is None


def test_a_gate_verdict_beside_the_result_is_loaded(tmp_path: Path) -> None:
    _write_gate(tmp_path, _gate_document(passed=True, threshold=1))
    assert load_gate(tmp_path) == GateVerdict(
        passed=True, successes=2, n=3, min_n=30, threshold=1.0, lower_bound=0.0943
    )


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ([1], "not a JSON object"),
        (_gate_document(schema_version=2), "unsupported schema_version 2"),
        (_gate_document(passed="yes"), "'passed' must be a boolean"),
        (_gate_document(n=True), "'n' must be an integer"),
        (_gate_document(n=3.0), "'n' must be an integer"),
        (_gate_document(threshold="high"), "'threshold' must be a number"),
        (_gate_document(lower_bound=False), "'lower_bound' must be a number"),
        ({"schema_version": 1}, "'passed' must be a boolean"),
    ],
)
def test_a_malformed_gate_verdict_is_a_report_error_not_a_silent_omission(
    tmp_path: Path, document: object, message: str
) -> None:
    _write_gate(tmp_path, document)
    with pytest.raises(ReportError, match=message):
        load_gate(tmp_path)


def test_a_gate_verdict_that_is_not_json_is_a_report_error(tmp_path: Path) -> None:
    (tmp_path / GATE_FILENAME).write_text("{", encoding="utf-8")
    with pytest.raises(ReportError, match="not valid JSON"):
        load_gate(tmp_path)


def test_an_unreadable_gate_verdict_is_a_report_error(tmp_path: Path) -> None:
    (tmp_path / GATE_FILENAME).mkdir()
    with pytest.raises(ReportError, match=r"cannot read .*gate\.json"):
        load_gate(tmp_path)


# --- The golden fixture and the CLI -------------------------------------------


def test_the_golden_fixture_renders_exactly_as_recorded() -> None:
    """A hand-written run directory, so nothing in it could have come from a re-run."""
    expected = (FIXTURES / "flagged-run" / "expected.md").read_text(encoding="utf-8")
    assert report_run_dir(FIXTURES / "flagged-run") == expected


def test_the_report_command_prints_the_report_of_a_persisted_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["report", str(FIXTURES / "flagged-run")]) == 0
    captured = capsys.readouterr()
    assert captured.out == report_run_dir(FIXTURES / "flagged-run")
    assert captured.err == ""


def test_the_report_command_regenerates_the_same_report_every_time(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = _persist(tmp_path, _flagged_run())
    _write_gate(run_dir, _gate_document())

    assert main(["report", str(run_dir)]) == 0
    first = capsys.readouterr().out
    assert main(["report", str(run_dir)]) == 0
    assert capsys.readouterr().out == first
    assert "| Verdict | **FAIL** |" in first


def test_the_report_command_fails_loudly_on_a_run_it_cannot_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["report", str(tmp_path / "absent")]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "fathomline report: cannot read" in captured.err


def test_the_report_command_calls_out_a_stale_passing_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The review's probe: a flagged 2 / 3 run beside a stale 30 / 30 PASS verdict."""
    run_dir = _persist(tmp_path, _flagged_run())
    _write_gate(run_dir, _gate_document(passed=True, successes=30, n=30, lower_bound=0.88))

    assert main(["report", str(run_dir)]) == 0
    out = capsys.readouterr().out

    assert "| Verdict | **PASS** |" in out
    assert "the gate verdict counts 30 / 30 but this run has 2 / 3." in out
    assert "the gate verdict is PASS but this run is flagged." in out
    assert out.count("**Contradiction:**") == 2
