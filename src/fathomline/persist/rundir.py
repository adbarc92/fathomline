"""The run directory: one persisted sweep, readable with no re-run.

    <run-dir>/
      result.json              result schema v1, what `report` and `gate` read
      run.json                 run schema v1: the seed set, the host, timings
      trajectories/<seed>.json every action of one episode

`result.json` is the measurement; the same seed set on the same host writes the
same bytes. `run.json` is what varies run to run, or host to host, and says what
the measurement is a measurement *on*. A trajectory hashes to its episode's
`action_hash`, so a mismatch found by a re-run can be traced to the step where
it began. `read_run` checks all of that before it returns anything.

A directory is written under a temporary name and renamed into place when
complete, so a run directory that exists is a whole run. An existing one is
never overwritten: a persisted result is not edited after the fact.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fathomline.core.loop import Sweep, action_hash
from fathomline.core.schema import SchemaError, from_document, to_document
from fathomline.core.types import Action, EpisodeResult, RunResult
from fathomline.persist.host import Host
from fathomline.persist.lock import SeedSet

RESULT_FILENAME = "result.json"
RUN_FILENAME = "run.json"
TRAJECTORIES_DIRNAME = "trajectories"

#: Version of `run.json`, independent of the result schema it sits beside.
RUN_SCHEMA_VERSION = 1


class RunDirError(ValueError):
    """The run directory cannot be written, or does not hold a whole, consistent run."""


@dataclass(frozen=True, slots=True)
class RunRecord:
    """`run.json`: everything about a run that is not the measurement itself."""

    seed_set: SeedSet
    #: When the sweep started, ISO 8601 UTC.
    started_at: str
    #: Wall-clock seconds per episode, in run order.
    episode_durations: tuple[float, ...]
    host: Host
    #: The command line, as `fathomline <argv>`.
    argv: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PersistedRun:
    result: RunResult
    record: RunRecord
    #: Per episode, in run order; each hashes to that episode's `action_hash`.
    trajectories: tuple[tuple[Action, ...], ...]


def write_run(
    run_dir: Path,
    sweep: Sweep,
    *,
    seed_set: SeedSet,
    host: Host,
    started_at: str,
    argv: Sequence[str],
) -> Path:
    """Persist `sweep` as a new run directory and return its path."""
    if sweep.result.manifest.seeds != seed_set.seeds:
        raise RunDirError(
            f"the sweep ran seeds {sweep.result.manifest.seeds}, "
            f"not seed set {seed_set.name!r} {seed_set.seeds}"
        )
    if run_dir.exists():
        raise RunDirError(f"{run_dir} already exists; a persisted run is never overwritten")

    record = RunRecord(
        seed_set=seed_set,
        started_at=started_at,
        episode_durations=sweep.durations,
        host=host,
        argv=tuple(argv),
    )
    staging = run_dir.with_name(f".{run_dir.name}.partial")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / TRAJECTORIES_DIRNAME).mkdir(parents=True)
    _write_json(staging / RESULT_FILENAME, to_document(sweep.result))
    _write_json(staging / RUN_FILENAME, _record_document(record))
    for episode, actions in zip(sweep.result.episodes, sweep.trajectories, strict=True):
        _write_json(
            staging / TRAJECTORIES_DIRNAME / f"{episode.seed}.json",
            {"seed": episode.seed, "actions": [list(action) for action in actions]},
            indent=None,
        )
    os.replace(staging, run_dir)
    return run_dir


def read_run(run_dir: Path) -> PersistedRun:
    """Read a run directory back, refusing one that is incomplete or inconsistent."""
    result = _read_result(run_dir / RESULT_FILENAME)
    record = _read_record(run_dir / RUN_FILENAME)
    if record.seed_set.seeds != result.manifest.seeds:
        raise RunDirError(
            f"{run_dir}: {RUN_FILENAME} declares seeds {record.seed_set.seeds} "
            f"but {RESULT_FILENAME} ran {result.manifest.seeds}"
        )
    if len(record.episode_durations) != len(result.episodes):
        raise RunDirError(
            f"{run_dir}: {len(record.episode_durations)} episode durations "
            f"for {len(result.episodes)} episodes"
        )
    trajectories = tuple(
        _read_trajectory(run_dir / TRAJECTORIES_DIRNAME / f"{episode.seed}.json", episode)
        for episode in result.episodes
    )
    return PersistedRun(result=result, record=record, trajectories=trajectories)


def _write_json(path: Path, document: object, *, indent: int | None = 2) -> None:
    # Floats are written as `repr` writes them, which reads back to the same bits.
    # `newline`: the same bytes on every OS, not CRLF on Windows.
    path.write_text(json.dumps(document, indent=indent) + "\n", encoding="utf-8", newline="\n")


def _record_document(record: RunRecord) -> dict[str, Any]:
    return {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "seed_set": {
            "name": record.seed_set.name,
            "description": record.seed_set.description,
            "seeds": list(record.seed_set.seeds),
            "digest": record.seed_set.digest,
        },
        "started_at": record.started_at,
        "episode_durations": list(record.episode_durations),
        "host": dataclasses.asdict(record.host),
        "argv": list(record.argv),
    }


def _read_object(path: Path) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RunDirError(f"cannot read {path}: {exc.strerror}") from exc
    except json.JSONDecodeError as exc:
        raise RunDirError(f"{path}: not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise RunDirError(f"{path}: not a JSON object")
    return document


def _read_result(path: Path) -> RunResult:
    try:
        return from_document(_read_object(path))
    except SchemaError as exc:
        raise RunDirError(f"{path}: {exc}") from exc
    except (KeyError, TypeError) as exc:
        raise RunDirError(f"{path}: malformed result: {exc!r}") from exc


def _read_record(path: Path) -> RunRecord:
    document = _read_object(path)
    version = document.get("run_schema_version")
    if version != RUN_SCHEMA_VERSION:
        raise RunDirError(
            f"{path}: unsupported run_schema_version {version!r}; expected {RUN_SCHEMA_VERSION}"
        )
    try:
        declared = document["seed_set"]
        seed_set = SeedSet(
            name=declared["name"],
            seeds=tuple(declared["seeds"]),
            description=declared["description"],
        )
        host = document["host"]
        record = RunRecord(
            seed_set=seed_set,
            started_at=document["started_at"],
            episode_durations=tuple(document["episode_durations"]),
            host=Host(**{**host, "cpu_flags": tuple(host["cpu_flags"])}),
            argv=tuple(document["argv"]),
        )
        digest = declared["digest"]
    except (KeyError, TypeError) as exc:
        raise RunDirError(f"{path}: malformed run record: {exc!r}") from exc
    if digest != seed_set.digest:
        raise RunDirError(
            f"{path}: seed set {seed_set.name!r} hashes to {seed_set.digest}, "
            f"but the record says {digest!r}"
        )
    return record


def _read_trajectory(path: Path, episode: EpisodeResult) -> tuple[Action, ...]:
    document = _read_object(path)
    try:
        actions = tuple(tuple(float(value) for value in action) for action in document["actions"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RunDirError(f"{path}: malformed trajectory: {exc!r}") from exc
    if document.get("seed") != episode.seed or len(actions) != episode.steps:
        raise RunDirError(
            f"{path}: expected {episode.steps} actions of seed {episode.seed}, "
            f"got {len(actions)} of seed {document.get('seed')!r}"
        )

    recomputed = action_hash(actions)
    if recomputed != episode.action_hash:
        raise RunDirError(
            f"{path}: the actions hash to {recomputed}, "
            f"but episode {episode.seed} records {episode.action_hash}"
        )
    return actions
