"""`fathomline sweep`: a declared seed set in, a run directory out, nothing else on stdout.

The fixture is replaced by the fakes here, as in the `eval` tests; the container
job runs the real one.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fathomline import cli
from fathomline.cli import main
from fathomline.core import Env, Policy
from fathomline.fakes import FakeEnv, FakePolicy
from fathomline.gates import command as gate_command
from fathomline.persist import command as sweep_command
from fathomline.persist import compare_runs, read_run, seed_digest

EXIT_NOT_RUN = 2
EXIT_FIXTURE_UNAVAILABLE = 3


def _fake_fixture() -> tuple[Env, Policy, int]:
    print("progress chatter a library prints on stdout")
    return FakeEnv(success_rate=1.0), FakePolicy(), 10


@pytest.fixture
def lock(tmp_path: Path) -> Path:
    path = tmp_path / "tests.lock"
    path.write_text(
        "version = 1\n[seed_sets.pair]\n"
        f'description = "two"\nseeds = [7, 8]\ndigest = "{seed_digest((7, 8))}"\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def fake_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_fixture_installed", lambda: True)
    monkeypatch.setattr(cli, "_load_fixture", _fake_fixture)
    # The fakes never import torch, so there is nothing for the pin to check.
    monkeypatch.delitem(sys.modules, "torch", raising=False)


def _torch(monkeypatch: pytest.MonkeyPatch, *, num_threads: int, cpu_capability: str) -> None:
    """Stand in for the torch the real fixture imports, with these settings."""
    torch = types.SimpleNamespace(
        __version__="2.11.0+cpu",
        get_num_threads=lambda: num_threads,
        get_num_interop_threads=lambda: num_threads,
        backends=types.SimpleNamespace(
            cpu=types.SimpleNamespace(get_cpu_capability=lambda: cpu_capability)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)


@pytest.mark.usefixtures("fake_fixture")
def test_sweep_persists_a_run_directory_and_prints_only_its_path(
    tmp_path: Path, lock: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    argv = ["sweep", "--seed-set", "pair", "--lock", str(lock), "--run-dir", str(run_dir)]
    assert main(argv) == 0
    captured = capsys.readouterr()
    assert captured.out == f"{run_dir}\n"
    assert "progress chatter" in captured.err
    assert "pair: 2/2 succeeded" in captured.err

    persisted = read_run(run_dir)
    assert persisted.result.manifest.seeds == (7, 8)
    assert persisted.result.manifest.max_steps == 10
    assert persisted.record.seed_set.name == "pair"
    assert persisted.record.argv == tuple(argv)
    assert persisted.record.started_at.endswith("Z")


@pytest.mark.usefixtures("fake_fixture")
def test_two_sweeps_of_one_seed_set_compare_equal(tmp_path: Path, lock: Path) -> None:
    for name in ("first", "second"):
        argv = ["sweep", "--seed-set", "pair", "--lock", str(lock), "--run-dir"]
        assert main([*argv, str(tmp_path / name)]) == 0
    first, second = read_run(tmp_path / "first"), read_run(tmp_path / "second")
    assert compare_runs(first.result, second.result) == ()


@pytest.mark.usefixtures("fake_fixture")
def test_the_gate_judges_a_swept_run_with_no_re_run(
    tmp_path: Path, lock: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    assert (
        main(["sweep", "--seed-set", "pair", "--lock", str(lock), "--run-dir", str(run_dir)]) == 0
    )
    result = str(run_dir / "result.json")
    # 2/2 is not evidence of anything: the minimum n fails it.
    assert main(["gate", result, "--threshold", "0.5", "--min-episodes", "30"]) == (
        gate_command.EXIT_FAILED
    )
    assert "insufficient sample" in capsys.readouterr().out


@pytest.mark.usefixtures("fake_fixture")
def test_the_default_run_directory_is_named_for_the_set_and_the_time(
    tmp_path: Path, lock: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    args = cli.build_parser().parse_args(["sweep", "--seed-set", "pair", "--lock", str(lock)])
    code = sweep_command.run(
        args,
        fixture_installed=lambda: True,
        load_fixture=_fake_fixture,
        argv=["sweep"],
        now=lambda: datetime(2026, 10, 26, 1, 2, 3, tzinfo=UTC),
    )
    assert code == 0
    expected = Path("runs") / "pair-20261026T010203Z"
    assert capsys.readouterr().out == f"{expected}\n"
    assert read_run(expected).record.started_at == "2026-10-26T01:02:03Z"


@pytest.mark.usefixtures("fake_fixture")
def test_without_argv_the_process_command_line_is_recorded(
    tmp_path: Path, lock: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    argv = ["sweep", "--seed-set", "pair", "--lock", str(lock), "--run-dir", str(run_dir)]
    monkeypatch.setattr("sys.argv", ["fathomline", *argv])
    assert main() == 0
    assert read_run(run_dir).record.argv == tuple(argv)


@pytest.mark.usefixtures("fake_fixture")
def test_a_sweep_under_the_determinism_pin_runs_and_records_it(
    tmp_path: Path, lock: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _torch(monkeypatch, num_threads=2, cpu_capability="AVX2")
    run_dir = tmp_path / "run"
    assert (
        main(["sweep", "--seed-set", "pair", "--lock", str(lock), "--run-dir", str(run_dir)]) == 0
    )
    torch = read_run(run_dir).record.host.torch
    assert torch is not None
    assert (torch["num_threads"], torch["cpu_capability"]) == (2, "AVX2")


@pytest.mark.usefixtures("fake_fixture")
@pytest.mark.parametrize(
    ("num_threads", "cpu_capability", "named"),
    [(1, "AVX2", "num_threads is 1"), (2, "AVX512", "cpu_capability is 'AVX512'")],
)
def test_a_sweep_off_the_determinism_pin_runs_nothing(
    tmp_path: Path,
    lock: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    num_threads: int,
    cpu_capability: str,
    named: str,
) -> None:
    def must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a sweep ran off the determinism pin")

    monkeypatch.setattr(sweep_command, "run_sweep", must_not_run)
    _torch(monkeypatch, num_threads=num_threads, cpu_capability=cpu_capability)
    run_dir = tmp_path / "run"
    argv = ["sweep", "--seed-set", "pair", "--lock", str(lock), "--run-dir", str(run_dir)]
    assert main(argv) == EXIT_NOT_RUN
    captured = capsys.readouterr()
    assert "determinism pin" in captured.err
    assert named in captured.err
    assert captured.out == ""
    assert not run_dir.exists()


def test_an_undeclared_seed_set_runs_nothing(
    tmp_path: Path, lock: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["sweep", "--seed-set", "nope", "--lock", str(lock)]) == EXIT_NOT_RUN
    captured = capsys.readouterr()
    assert "no seed set 'nope'; declared: pair" in captured.err
    assert captured.out == ""


def test_an_existing_run_directory_is_refused_before_anything_runs(
    tmp_path: Path, lock: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def must_not_load() -> tuple[Env, Policy, int]:
        raise AssertionError("the fixture loaded for a sweep that cannot be persisted")

    monkeypatch.setattr(cli, "_fixture_installed", lambda: True)
    monkeypatch.setattr(cli, "_load_fixture", must_not_load)
    argv = ["sweep", "--seed-set", "pair", "--lock", str(lock), "--run-dir", str(tmp_path)]
    assert main(argv) == EXIT_NOT_RUN
    assert "already exists" in capsys.readouterr().err


def test_a_run_directory_that_appears_during_the_sweep_is_not_overwritten(
    tmp_path: Path, lock: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"

    def loads_then_collides() -> tuple[Env, Policy, int]:
        run_dir.mkdir()
        return _fake_fixture()

    args = cli.build_parser().parse_args(
        ["sweep", "--seed-set", "pair", "--lock", str(lock), "--run-dir", str(run_dir)]
    )
    code = sweep_command.run(
        args, fixture_installed=lambda: True, load_fixture=loads_then_collides, argv=["sweep"]
    )
    assert code == sweep_command.EXIT_NOT_PERSISTED
    assert list(run_dir.iterdir()) == []
    err = capsys.readouterr().err
    assert "ran but was not persisted" in err
    assert "never overwritten" in err


@pytest.mark.skipif(
    importlib.util.find_spec("lerobot") is not None, reason="the fixture is installed here"
)
def test_sweep_without_the_fixture_says_so_and_persists_nothing(
    tmp_path: Path, lock: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    argv = ["sweep", "--seed-set", "pair", "--lock", str(lock), "--run-dir", str(run_dir)]
    assert main(argv) == EXIT_FIXTURE_UNAVAILABLE
    captured = capsys.readouterr()
    assert "not installed here" in captured.err
    assert captured.out == ""
    assert not run_dir.exists()


def test_the_seed_set_is_required() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["sweep"])
    assert exit_info.value.code == 2
