# The run directory

_What `fathomline sweep` persists, and what `report`, `gate` and (lane H) `reproduce` read.
Code: [`src/fathomline/persist/`](../src/fathomline/persist/). The seed sets:
[`tests.lock`](../tests.lock)._

```bash
fathomline sweep --seed-set smoke --run-dir runs/smoke   # prints runs/smoke on stdout
fathomline report runs/smoke                             # no re-run
fathomline gate runs/smoke/result.json --threshold 0.8 --min-episodes 30
```

In the container, mount a directory to keep the run:
`docker run --rm --network none -v "$PWD/runs:/app/runs" fathomline sweep --seed-set smoke --run-dir runs/smoke`.

## Layout

```
<run-dir>/
  result.json                 result schema v1: the measurement
  run.json                    run schema v1: the declaration, the host, timings
  trajectories/<seed>.json    every action of one episode
```

| File | Versioned by | Varies between two runs of one seed set on one host? |
|---|---|---|
| `result.json` | `schema_version` (`fathomline.core.schema`, currently 1) | **Expected not to:** the same bytes. Shown on the fake by the tests and on the smoke set by the CI container job, each on one host |
| `run.json` | `run_schema_version` (`fathomline.persist.rundir`, currently 1) | Yes: `started_at`, `episode_durations`, `argv` |
| `trajectories/<seed>.json` | the `result.json` beside it (each must hash to its episode's `action_hash`) | No |

A directory is written under a temporary name (`.<name>.partial`), read back with `read_run`, and
only then renamed into place, so a run directory that exists is a whole run that reads. An
existing directory is never overwritten. Files are not fsynced: a power loss just after the rename
could still truncate one, and `read_run` would then refuse the directory. The directory does not
depend on its own path: it can be copied or published as is.

## `result.json`

Result schema v1, unchanged by lane B: the manifest (fixture refs, seeds, max steps, Python,
platform) and one entry per episode (`seed`, `steps`, `success`, `sum_reward`, `max_reward`,
`action_hash`, `flags`). This is the only file `report` and `gate` read.

## `run.json`

```json
{
  "run_schema_version": 1,
  "seed_set": {"name": "smoke", "description": "…", "seeds": [1000], "digest": "<sha256>"},
  "started_at": "2026-10-26T00:00:00Z",
  "episode_durations": [71.2],
  "host": {
    "platform": "Linux-…-x86_64-with-glibc2.41",
    "machine": "x86_64",
    "cpu_model": "AMD EPYC 9V74 80-Core Processor",
    "cpu_flags": ["avx", "avx2", "avx512f", "…"],
    "logical_cpus": 4,
    "usable_cpus": 4,
    "python_version": "3.12.14",
    "threading_env": {"OMP_NUM_THREADS": "4"},
    "torch": {"version": "2.11.0+cpu", "num_threads": 4, "num_interop_threads": 4,
              "cpu_capability": "AVX512"}
  },
  "argv": ["sweep", "--seed-set", "smoke", "--run-dir", "runs/smoke"]
}
```

- `seed_set.digest` is `seed_digest(seeds)`, the SHA-256 of the seeds in order, as recorded in
  `tests.lock`. A re-run can check it ran the same declaration.
- `host` is recorded, never acted on. It holds what an open ruling (D9a) would scope a determinism
  claim by: the CPU model and SIMD flags (from `/proc/cpuinfo`; empty off Linux), the CPUs the
  process could use (its affinity mask, i.e. a container's `--cpuset-cpus`), the thread and
  kernel-dispatch environment variables that were set, and torch's thread counts and the
  instruction set its kernels dispatched to (`null` if the fixture did not import torch).

## `trajectories/<seed>.json`

`{"seed": 1000, "actions": ["<hex>", …]}`, one string per step: the action's float64s,
little-endian, as hex, which are exactly the bytes `fathomline.core.loop.action_hash` hashes. A
decimal JSON number would not do: JSON has no NaN or infinity, and Python's `NaN` extension reads
back without its sign bit, so a policy that emits a NaN would leave a trajectory that no longer
matched its own hash. Decode one with `struct.unpack(f"<{len(raw) // 8}d", bytes.fromhex(text))`,
or read the whole directory with `read_run`. When two runs disagree on a hash, comparing their
trajectories (`fathomline.persist.first_divergence`) names the first step that differs and by how
much.

## What `read_run` checks

`fathomline.persist.read_run(run_dir)` returns the result, the record and the trajectories, or
raises `RunDirError`. It refuses a directory if any file is missing or of an unknown version; if
the result's episodes are not its manifest's seeds, in order; if `run.json`'s seeds differ from
the result's or do not hash to its digest; if the durations do not
match the episodes; or if any trajectory has the wrong seed or length, or does not hash to its
episode's `action_hash`.

## For `reproduce` (lane H)

A re-run of a persisted run is the same `sweep` with the recorded `seed_set.name`, then
`compare_runs(published.result, rerun.result)`: an empty tuple is equality on the fixture, the
seeds, every episode's metrics and its action hash (D9), never video. Floats are compared bit for
bit, except that any NaN metric equals any NaN (`result.json` keeps no NaN sign). On a hash
mismatch, `first_divergence` over the two trajectories locates it. Host facts are not compared.
`run.json`'s `argv` is recorded verbatim, so a run launched with absolute `--lock` or `--run-dir`
paths carries them: relativize or strip them before publishing a run. **Whether two
hosts may be expected to agree is the open ruling D9a**; until it is ruled, the equality shown is
run to run on one host.

## Changing the format

A change that an older reader would misread bumps the relevant version (`schema_version` or
`run_schema_version`), and the reader keeps accepting the old one. The committed fixture
[`tests/fixtures/persist/week2-run/`](../tests/fixtures/persist/week2-run/) is the format as first
written; `test_the_week2_fixture_still_parses` must keep passing without editing it.
