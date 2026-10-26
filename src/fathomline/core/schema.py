"""Result schema v1: the JSON-shaped document a run is persisted as.

The version is checked on the way in. A reader that guesses at a document it
does not recognise would report numbers nobody measured in that shape.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from fathomline.core.types import ArtifactRef, EpisodeResult, Flag, RunManifest, RunResult

SCHEMA_VERSION = 1


class SchemaError(ValueError):
    """The document is not a result this version can read."""


def to_document(result: RunResult) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, **dataclasses.asdict(result)}


def from_document(document: Mapping[str, Any]) -> RunResult:
    version = document.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SchemaError(f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION}")

    manifest = document["manifest"]
    return RunResult(
        manifest=RunManifest(
            fathomline_version=manifest["fathomline_version"],
            policy=ArtifactRef(**manifest["policy"]),
            env=ArtifactRef(**manifest["env"]),
            seeds=tuple(manifest["seeds"]),
            max_steps=manifest["max_steps"],
            python_version=manifest["python_version"],
            platform=manifest["platform"],
        ),
        episodes=tuple(
            EpisodeResult(
                seed=episode["seed"],
                steps=episode["steps"],
                success=episode["success"],
                sum_reward=episode["sum_reward"],
                max_reward=episode["max_reward"],
                action_hash=episode["action_hash"],
                flags=tuple(Flag(**flag) for flag in episode["flags"]),
            )
            for episode in document["episodes"]
        ),
    )
