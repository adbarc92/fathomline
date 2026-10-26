"""`fathomline gate`: pass or fail a persisted result against a declared gate.

Kept here rather than in `cli.py` so the shared CLI module only registers the
subcommand and dispatches to it.

Exit codes:

* 0 — the run passed the gate;
* 1 — the run failed the gate (the reasons are printed);
* 3 — the gate could not judge: the result is unreadable or the gate is
  impossible. Distinct from 1 so a broken pipeline is never read as a policy
  that missed its bar, and from 2, which the CLI uses for usage errors and
  unbuilt commands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fathomline.core.schema import from_document
from fathomline.core.types import RunResult
from fathomline.gates.gate import GateConfig, evaluate

EXIT_PASSED = 0
EXIT_FAILED = 1
EXIT_CANNOT_JUDGE = 3


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("result", help="path to a result document (schema v1 JSON)")
    # Required, with no defaults: the bar a run is judged against has to be declared.
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="the lower bound on the success rate must be at least this",
    )
    parser.add_argument(
        "--min-episodes",
        type=int,
        required=True,
        help="a run with fewer episodes fails, whatever its success rate",
    )
    # No --confidence: ruling D6 fixes it at 95%, so it is not the caller's to change.


def run(args: argparse.Namespace) -> int:
    try:
        config = GateConfig(threshold=args.threshold, min_episodes=args.min_episodes)
    except ValueError as error:
        print(f"fathomline gate: impossible gate: {error}", file=sys.stderr)
        return EXIT_CANNOT_JUDGE

    try:
        result = _load(Path(args.result))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"fathomline gate: cannot read result {args.result}: {error!r}", file=sys.stderr)
        return EXIT_CANNOT_JUDGE

    verdict = evaluate(result, config)
    print(
        f"gate {'PASSED' if verdict.passed else 'FAILED'}: "
        f"{verdict.successes}/{verdict.episodes} episodes succeeded; "
        f"lower bound {verdict.lower_bound:.4f} at {config.confidence * 100:g}% confidence "
        f"(threshold {config.threshold}, minimum {config.min_episodes} episodes)"
    )
    for reason in verdict.reasons:
        print(f"  - {reason}")
    return EXIT_PASSED if verdict.passed else EXIT_FAILED


def _load(path: Path) -> RunResult:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"expected a JSON object, got {type(document).__name__}")
    return from_document(document)
