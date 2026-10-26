"""Read a run directory: the persisted result, and a gate verdict if one sits beside it.

Nothing here runs an episode. A report is regenerated from what was persisted,
so a document that cannot be read is an error, never a gap the report papers
over: a missing section would read as "nothing to say", which is not the same
claim as "this could not be read".
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fathomline.core.schema import SchemaError, from_document
from fathomline.core.types import RunResult

#: The persisted run, as result schema v1 (`fathomline.core.schema`).
RESULT_FILENAME = "result.json"

#: An optional gate verdict beside the result. Provisional: Lane C owns gates,
#: and this is the smallest shape the report needs from it (ruling D6).
GATE_FILENAME = "gate.json"
GATE_SCHEMA_VERSION = 1


class ReportError(ValueError):
    """The run directory holds nothing this version can report on."""


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """A gate's verdict on a run: the D6 statistic and the numbers behind it."""

    passed: bool
    successes: int
    n: int
    min_n: int
    threshold: float
    lower_bound: float


def load_run(run_dir: Path) -> RunResult:
    path = run_dir / RESULT_FILENAME
    document = _read_object(path)
    try:
        return from_document(document)
    except SchemaError as exc:
        raise ReportError(f"{path}: {exc}") from exc
    except KeyError as exc:
        raise ReportError(f"{path}: missing field {exc}") from exc
    except TypeError as exc:
        raise ReportError(f"{path}: malformed result: {exc}") from exc


def load_gate(run_dir: Path) -> GateVerdict | None:
    path = run_dir / GATE_FILENAME
    if not path.exists():
        return None
    document = _read_object(path)
    version = document.get("schema_version")
    if version != GATE_SCHEMA_VERSION:
        raise ReportError(
            f"{path}: unsupported schema_version {version!r}; expected {GATE_SCHEMA_VERSION}"
        )
    return GateVerdict(
        passed=_bool(path, document, "passed"),
        successes=_int(path, document, "successes"),
        n=_int(path, document, "n"),
        min_n=_int(path, document, "min_n"),
        threshold=_number(path, document, "threshold"),
        lower_bound=_number(path, document, "lower_bound"),
    )


def _read_object(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReportError(f"cannot read {path}: {exc.strerror}") from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReportError(f"{path}: not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ReportError(f"{path}: not a JSON object")
    return document


# `bool` is a subclass of `int` in Python, so each check below rules it in or
# out explicitly: `"n": true` is a malformed verdict, not a sample of one.


def _bool(path: Path, document: Mapping[str, Any], name: str) -> bool:
    value = document.get(name)
    if not isinstance(value, bool):
        raise ReportError(f"{path}: {name!r} must be a boolean, got {value!r}")
    return value


def _int(path: Path, document: Mapping[str, Any], name: str) -> int:
    value = document.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportError(f"{path}: {name!r} must be an integer, got {value!r}")
    return value


def _number(path: Path, document: Mapping[str, Any], name: str) -> float:
    value = document.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ReportError(f"{path}: {name!r} must be a number, got {value!r}")
    return float(value)
