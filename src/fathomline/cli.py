"""Command-line entry point.

`eval` runs the pinned LeRobot fixture (D1) through the core's step loop and
prints result schema v1 as JSON on stdout; everything else goes to stderr. The
other subcommands exit non-zero and say they are not implemented, because the
alternative — a command that prints something result-shaped before anything is
measured — is the exact failure this project exists to argue against.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from fathomline import __version__
from fathomline.core import Env, Policy, RunManifest, RunResult, run_episode, to_document
from fathomline.gates import command as gate_command
from fathomline.persist import command as sweep_command
from fathomline.report import RESULT_FILENAME, ReportError, report_run_dir

#: `eval` is declared but unbuilt. Distinct from 1 so a future real failure of
#: the eval loop is never mistaken for "this command does not exist yet".
EXIT_NOT_IMPLEMENTED = 2

#: `eval` cannot load its fixture here: the `lerobot` extra is Linux-only and is
#: installed in the container image, not on a development machine. The same
#: number as `gate`'s "cannot judge": neither command produced a verdict.
EXIT_FIXTURE_UNAVAILABLE = 3

#: LeRobot's default, and the seed every spike measurement used.
DEFAULT_SEED = 1000

#: Every subcommand the harness will have. Each is a stub until its lane lands.
COMMANDS = {
    "eval": "run the pinned fixture for N episodes and print the result as JSON",
    "sweep": "run the pinned fixture over a declared seed set and persist the run",
    "gate": "pass or fail a persisted run against a declared threshold and minimum n",
    "report": "render a Markdown report from a persisted run directory",
    "reproduce": "re-run a published result and compare (not implemented)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fathomline",
        description="Evaluate a learned robot policy: fail-closed gates, out-of-envelope flags.",
    )
    # Deliberately not `action="version"`: that action calls sys.exit() inside the
    # parser, which would make `main` sometimes return a code and sometimes raise.
    # One contract — main returns an exit code — is worth a three-line flag.
    parser.add_argument("--version", action="store_true", help="print the version and exit")

    subcommands = parser.add_subparsers(dest="command")
    subparsers = {name: subcommands.add_parser(name, help=text) for name, text in COMMANDS.items()}
    subparsers["eval"].add_argument(
        "--episodes", type=_positive_int, default=1, help="number of episodes to run"
    )
    subparsers["eval"].add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"seed of the first episode; episode k uses seed + k (default {DEFAULT_SEED})",
    )
    gate_command.add_arguments(subparsers["gate"])
    sweep_command.add_arguments(subparsers["sweep"])
    subparsers["report"].add_argument(
        "run_dir", type=Path, help=f"a run directory holding {RESULT_FILENAME}"
    )
    return parser


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _fixture_installed() -> bool:
    from fathomline.adapters.lerobot import installed

    return installed()


def _load_fixture() -> tuple[Env, Policy, int]:
    """The real fixture. Imported here, not at module level, so the CLI runs anywhere."""
    from fathomline.adapters.lerobot import load_fixture

    return load_fixture()


def _eval(episodes: int, first_seed: int) -> int:
    # Checked up front rather than by catching ModuleNotFoundError: a module missing
    # inside an installed stack is a broken image, and must not read "not installed".
    if not _fixture_installed():
        print(
            "fathomline eval: the LeRobot fixture is not installed here (no `lerobot`). "
            "It is Linux-only and runs in the container image.",
            file=sys.stderr,
        )
        return EXIT_FIXTURE_UNAVAILABLE
    seeds = tuple(range(first_seed, first_seed + episodes))
    # LeRobot and LIBERO print progress on stdout; stdout is reserved for the result.
    with contextlib.redirect_stdout(sys.stderr):
        env, policy, max_steps = _load_fixture()
        results = tuple(run_episode(env, policy, seed=seed, max_steps=max_steps) for seed in seeds)
    manifest = RunManifest.capture(env=env, policy=policy, seeds=seeds, max_steps=max_steps)
    print(json.dumps(to_document(RunResult(manifest=manifest, episodes=results)), indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"fathomline {__version__}")
        return 0

    if args.command is None:
        parser.print_usage(sys.stderr)
        return EXIT_NOT_IMPLEMENTED

    if args.command == "eval":
        return _eval(args.episodes, args.seed)

    if args.command == "gate":
        return gate_command.run(args)

    if args.command == "sweep":
        return sweep_command.run(
            args,
            fixture_installed=_fixture_installed,
            load_fixture=_load_fixture,
            argv=list(sys.argv[1:] if argv is None else argv),
        )

    if args.command == "report":
        return _report(args.run_dir)

    print(
        f"fathomline {args.command} is not implemented yet: "
        "no policy is loaded and nothing is measured.",
        file=sys.stderr,
    )
    return EXIT_NOT_IMPLEMENTED


def _report(run_dir: Path) -> int:
    """Print the report of a persisted run. Nothing is re-run (ruling D12)."""
    try:
        markdown = report_run_dir(run_dir)
    except ReportError as exc:
        print(f"fathomline report: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(markdown)
    return 0


def run() -> None:
    """Console-script shim: turn the return code into an exit status."""
    raise SystemExit(main())
