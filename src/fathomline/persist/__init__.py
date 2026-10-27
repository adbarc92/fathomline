"""Lane B: declared seed sets, persisted runs, and the run-to-run comparison.

A sweep runs a seed set declared in `tests.lock` (`lock`) and is persisted as a
run directory (`rundir`) that `report` and `gate` read with no re-run, and that
a re-run can be compared against (`compare`). Each run records the host it ran
on (`host`), so a determinism claim can be scoped to what was actually shown.
"""

from __future__ import annotations

from fathomline.persist.compare import Divergence, compare_runs, first_divergence
from fathomline.persist.host import Host, capture_host
from fathomline.persist.lock import (
    DEFAULT_LOCK,
    LOCK_VERSION,
    LockError,
    SeedSet,
    load_lock,
    seed_digest,
    seed_set,
)
from fathomline.persist.rundir import (
    RESULT_FILENAME,
    RUN_FILENAME,
    RUN_SCHEMA_VERSION,
    TRAJECTORIES_DIRNAME,
    PersistedRun,
    RunDirError,
    RunRecord,
    read_run,
    write_run,
)

__all__ = [
    "DEFAULT_LOCK",
    "LOCK_VERSION",
    "RESULT_FILENAME",
    "RUN_FILENAME",
    "RUN_SCHEMA_VERSION",
    "TRAJECTORIES_DIRNAME",
    "Divergence",
    "Host",
    "LockError",
    "PersistedRun",
    "RunDirError",
    "RunRecord",
    "SeedSet",
    "capture_host",
    "compare_runs",
    "first_divergence",
    "load_lock",
    "read_run",
    "seed_digest",
    "seed_set",
    "write_run",
]
