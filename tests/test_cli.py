"""CLI contract tests.

At M0 the CLI proves the plumbing and nothing else: the package imports, the
entry point resolves, and `eval` is *honestly* unimplemented rather than
quietly returning something that looks like a result. Week 1 replaces the
`eval` body with a real episode; these tests are what it has to keep true.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

import pytest

from fathomline import __version__, cli
from fathomline.cli import main, run
from fathomline.core import Env, Policy, from_document
from fathomline.fakes import FakeEnv, FakePolicy

EXIT_NOT_IMPLEMENTED = 2
EXIT_FIXTURE_UNAVAILABLE = 3


def test_version_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out


@pytest.mark.parametrize("command", ["sweep", "reproduce"])
def test_a_stub_exits_not_implemented_rather_than_reporting_a_result(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The failure mode this guards: a harness that reports before it measures."""
    assert main([command]) == EXIT_NOT_IMPLEMENTED
    captured = capsys.readouterr()
    assert f"{command} is not implemented" in captured.err
    assert captured.out == ""


def test_no_arguments_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == EXIT_NOT_IMPLEMENTED
    assert "usage" in capsys.readouterr().err.lower()


def test_console_script_turns_the_return_code_into_an_exit_status() -> None:
    with pytest.raises(SystemExit) as exit_info:
        run()
    assert exit_info.value.code == EXIT_NOT_IMPLEMENTED


def test_entry_point_runs_as_a_module() -> None:
    """`python -m fathomline` is the form the container invokes."""
    completed = subprocess.run(
        [sys.executable, "-m", "fathomline", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert __version__ in completed.stdout


def _fake_fixture() -> tuple[Env, Policy, int]:
    print("progress chatter a library prints on stdout")
    return FakeEnv(success_rate=1.0), FakePolicy(), 10


def test_eval_prints_one_schema_v1_document_and_nothing_else_on_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_fixture_installed", lambda: True)
    monkeypatch.setattr(cli, "_load_fixture", _fake_fixture)
    assert main(["eval", "--episodes", "2", "--seed", "7"]) == 0
    captured = capsys.readouterr()
    result = from_document(json.loads(captured.out))
    assert result.manifest.seeds == (7, 8)
    assert result.manifest.max_steps == 10
    assert [episode.seed for episode in result.episodes] == [7, 8]
    assert "progress chatter" in captured.err


def test_eval_defaults_to_one_episode_at_seed_1000(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_fixture_installed", lambda: True)
    monkeypatch.setattr(cli, "_load_fixture", _fake_fixture)
    assert main(["eval"]) == 0
    assert from_document(json.loads(capsys.readouterr().out)).manifest.seeds == (1000,)


def test_eval_rejects_zero_episodes() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["eval", "--episodes", "0"])
    assert exit_info.value.code == 2


@pytest.mark.skipif(
    importlib.util.find_spec("lerobot") is not None, reason="the fixture is installed here"
)
def test_eval_without_the_fixture_says_so_and_reports_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Off the container, eval must fail loudly, not print a result it did not measure."""
    assert main(["eval"]) == EXIT_FIXTURE_UNAVAILABLE
    captured = capsys.readouterr()
    assert "not installed here" in captured.err
    assert captured.out == ""


@pytest.mark.skipif(
    importlib.util.find_spec("lerobot") is not None, reason="the fixture is installed here"
)
def test_without_the_stack_loading_the_fixture_raises_rather_than_returning() -> None:
    with pytest.raises(ModuleNotFoundError):
        cli._load_fixture()


def test_a_module_missing_inside_an_installed_stack_is_not_reported_as_not_installed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A broken image must fail as broken, not say "runs in the container image"."""

    def broken() -> tuple[Env, Policy, int]:
        raise ModuleNotFoundError("No module named 'some_transitive_dependency'")

    monkeypatch.setattr(cli, "_fixture_installed", lambda: True)
    monkeypatch.setattr(cli, "_load_fixture", broken)
    with pytest.raises(ModuleNotFoundError, match="some_transitive_dependency"):
        main(["eval"])
    captured = capsys.readouterr()
    assert "not installed here" not in captured.err
    assert captured.out == ""
