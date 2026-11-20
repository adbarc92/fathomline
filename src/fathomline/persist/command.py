"""`fathomline sweep`: run a declared seed set and persist it as a run directory.

Kept here rather than in `cli.py` so the shared CLI module only registers the
subcommand and dispatches to it. The fixture loaders are passed in, so the CLI
keeps the one place they are defined (and replaced, in tests).

stdout carries one line, the run directory's path; everything else goes to
stderr. Exit codes:

* 0 — the sweep ran and its run directory is complete;
* 1 — the sweep ran but could not be persisted (the run directory appeared
  while it ran, or the written directory did not read back, in which case the
  `.<name>.partial` staging directory is left for diagnosis); nothing was
  overwritten;
* 2 — nothing was run: the seed set is not declared, the lock is invalid, the
  run directory already exists, or the loaded fixture's torch is not under the
  determinism pin (D9b) (argparse uses 2 for usage errors too);
* 3 — the fixture is not installed here, as for `eval`.

An episode that raises is not caught: the sweep stops with the error, and the
run directory is never renamed into place, so no partial run reads as whole.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from fathomline.core.loop import run_sweep
from fathomline.core.protocols import Env, Policy
from fathomline.persist.host import capture_host, pin_mismatches
from fathomline.persist.lock import DEFAULT_LOCK, LockError, seed_set
from fathomline.persist.rundir import RunDirError, write_run

EXIT_NOT_PERSISTED = 1
EXIT_NOT_RUN = 2
EXIT_FIXTURE_UNAVAILABLE = 3

#: Where run directories go unless `--run-dir` says otherwise. Ignored by git.
DEFAULT_RUNS = Path("runs")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    # Required, with no default: which seeds a result rests on has to be declared.
    parser.add_argument("--seed-set", required=True, help="a seed set declared in the lock")
    parser.add_argument(
        "--lock",
        type=Path,
        default=DEFAULT_LOCK,
        help=f"the seed-set lock (default {DEFAULT_LOCK})",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=f"where to persist the run; must not exist (default {DEFAULT_RUNS}/<set>-<UTC time>)",
    )


def run(
    args: argparse.Namespace,
    *,
    fixture_installed: Callable[[], bool],
    load_fixture: Callable[[], tuple[Env, Policy, int]],
    argv: Sequence[str],
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> int:
    try:
        declared = seed_set(args.lock, args.seed_set)
    except LockError as exc:
        print(f"fathomline sweep: {exc}", file=sys.stderr)
        return EXIT_NOT_RUN

    started = now()
    run_dir: Path = args.run_dir or DEFAULT_RUNS / f"{declared.name}-{started:%Y%m%dT%H%M%SZ}"
    # Checked before the fixture loads, so a clash costs seconds, not a sweep.
    if run_dir.exists():
        print(
            f"fathomline sweep: {run_dir} already exists; choose another --run-dir", file=sys.stderr
        )
        return EXIT_NOT_RUN

    if not fixture_installed():
        print(
            "fathomline sweep: the LeRobot fixture is not installed here (no `lerobot`). "
            "It is Linux-only and runs in the container image.",
            file=sys.stderr,
        )
        return EXIT_FIXTURE_UNAVAILABLE

    # LeRobot and LIBERO print progress on stdout; stdout is reserved for the run directory.
    with contextlib.redirect_stdout(sys.stderr):
        env, policy, max_steps = load_fixture()
    # Read once the fixture has imported torch, and checked before any episode runs:
    # a run off the pin is outside the determinism claim, so it is not made at all.
    host = capture_host()
    if mismatches := pin_mismatches(host):
        print(
            "fathomline sweep: not run: off the determinism pin (D9b): " + "; ".join(mismatches),
            file=sys.stderr,
        )
        return EXIT_NOT_RUN
    with contextlib.redirect_stdout(sys.stderr):
        sweep = run_sweep(env, policy, seeds=declared.seeds, max_steps=max_steps)
    try:
        write_run(
            run_dir,
            sweep,
            seed_set=declared,
            host=host,
            started_at=f"{started:%Y-%m-%dT%H:%M:%SZ}",
            argv=argv,
        )
    except RunDirError as exc:  # It appeared while the sweep ran, or did not read back.
        print(f"fathomline sweep: the sweep ran but was not persisted: {exc}", file=sys.stderr)
        return EXIT_NOT_PERSISTED
    successes = sum(episode.success for episode in sweep.result.episodes)
    print(
        f"fathomline sweep: {declared.name}: {successes}/{len(sweep.result.episodes)} succeeded",
        file=sys.stderr,
    )
    print(run_dir)
    return 0
