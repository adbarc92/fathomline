"""`tests.lock`: the declared seed sets a sweep may run.

A sweep names a seed set; it never makes one up. Each set in the lock carries
the SHA-256 of its seeds, so a seed set cannot change by accident: an edit that
does not also update the digest is refused, with the digest it now needs. A run
directory records the set's name, seeds and digest, so a re-run can show it ran
the same declaration.

    version = 1

    [seed_sets.smoke]
    description = "One episode."
    seeds = [1000]
    digest = "<seed_digest((1000,))>"
"""

from __future__ import annotations

import hashlib
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOCK_VERSION = 1

#: Where `sweep` looks for the lock unless told otherwise: the repository root,
#: which is also the image's working directory.
DEFAULT_LOCK = Path("tests.lock")


class LockError(ValueError):
    """The lock is unreadable, or does not declare what was asked of it."""


def seed_digest(seeds: Sequence[int]) -> str:
    """SHA-256 of the seeds in order. Order is part of the declaration: it is run order."""
    return hashlib.sha256(",".join(str(seed) for seed in seeds).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SeedSet:
    name: str
    seeds: tuple[int, ...]
    description: str

    @property
    def digest(self) -> str:
        return seed_digest(self.seeds)


def load_lock(path: Path) -> dict[str, SeedSet]:
    """Every seed set in the lock, each checked against its digest."""
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LockError(f"cannot read {path}: {exc.strerror}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise LockError(f"{path}: not valid TOML: {exc}") from exc

    version = document.get("version")
    if version != LOCK_VERSION:
        raise LockError(f"{path}: unsupported version {version!r}; expected {LOCK_VERSION}")
    sets = document.get("seed_sets")
    if not isinstance(sets, dict) or not sets:
        raise LockError(f"{path}: declares no [seed_sets]")
    return {name: _seed_set(path, name, entry) for name, entry in sets.items()}


def seed_set(path: Path, name: str) -> SeedSet:
    """The named seed set, or a `LockError` that lists the ones there are."""
    sets = load_lock(path)
    if name not in sets:
        raise LockError(f"{path}: no seed set {name!r}; declared: {', '.join(sorted(sets))}")
    return sets[name]


def _seed_set(path: Path, name: str, entry: object) -> SeedSet:
    where = f"{path}: seed set {name!r}"
    if not isinstance(entry, Mapping):
        raise LockError(f"{where} is not a table")
    seeds = _seeds(where, entry)
    description = entry.get("description")
    if not isinstance(description, str) or not description.strip():
        raise LockError(f"{where} needs a description: why these seeds")
    declared = SeedSet(name=name, seeds=seeds, description=description)
    digest = entry.get("digest")
    if digest != declared.digest:
        raise LockError(
            f"{where}: its seeds hash to {declared.digest}, but the lock records {digest!r}. "
            "If the change is deliberate, record the new digest."
        )
    return declared


def _seeds(where: str, entry: Mapping[str, Any]) -> tuple[int, ...]:
    seeds = entry.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise LockError(f"{where} needs a non-empty list of seeds")
    # `bool` is an `int` in Python; `true` is not a seed.
    if not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
        raise LockError(f"{where}: every seed must be an integer")
    # A repeated seed would count one episode twice toward a gate's n.
    repeated = sorted({seed for seed in seeds if seeds.count(seed) > 1})
    if repeated:
        raise LockError(f"{where}: repeated seeds {repeated}")
    return tuple(seeds)
