# Fathomline

**An evaluation harness for learned robot policies.** A fathom line is the marked line a crew drops
to measure depth before trusting the water. This measures whether a policy actually works, rather
than reporting that it does.

> **Status: early build.** `eval` runs the pinned fixture (SmolVLA on LIBERO `libero_object`
> task 0) for real inside the container image, and prints a schema-v1 result as JSON. Outside the
> image, where the Linux-only LeRobot stack is absent, it says so and exits **3**. `sweep` runs a
> seed set declared in [`tests.lock`](tests.lock) and persists a run directory
> ([format](docs/run-directory.md)); `gate` and `report` work on persisted results. `reproduce`
> is not built yet and exits **2**.
> No result is published yet. Current state: [`docs/STATUS.md`](docs/STATUS.md).

**Every claim below has a command that proves it.** A claim without one does not belong in this
file — that rule is the point of the project, so it applies to the project's own README first.

| Claim | Proving command |
|---|---|
| The suite passes with 100% line and branch coverage of the core. `adapter.py` (Linux-only LeRobot stack) and `__main__.py` are omitted with stated reasons (ruling D11); the container job runs the adapter end to end | `uv run pytest --cov=fathomline --cov-report=term-missing` |
| Lint and strict type checking are clean | `uv run ruff check . && uv run ruff format --check . && uv run mypy` |
| It runs on a clean container | `docker build -t fathomline . && docker run --rm fathomline --version` |
| `eval` runs one real episode in the image, offline, and prints a schema-v1 result | `docker run --rm --network none fathomline eval --episodes 1; echo $?` → JSON on stdout, exits **0** |
| Without the fixture, `eval` refuses rather than printing a result | `uv run fathomline eval; echo $?` (on a machine without `lerobot`) → prints the reason, exits **3** |
| The gate fails closed: too small a sample, a low lower bound, or a monitor flag each fail it | `uv run fathomline gate tests/fixtures/report/flagged-run/result.json --threshold 0.8 --min-episodes 30; echo $?` → lists all three reasons, exits **1** |
| The report is regenerated from a saved result, with no re-run, and surfaces flags | `uv run fathomline report tests/fixtures/report/flagged-run` → a Markdown report headed **FLAGGED** |
| `sweep` runs a declared seed set in the image and persists a run directory that `report` reads with no re-run | `docker run --rm --network none -v "$PWD/runs:/app/runs" fathomline sweep --seed-set smoke --run-dir runs/smoke && uv run fathomline report runs/smoke` |
| Two sweeps of one seed set are equal on per-episode metrics and the action hash, and the check fails when the seed is removed (on the fake fixture, on one host) | `uv run pytest tests/test_determinism.py -v` |
| Unbuilt commands admit it | `uv run fathomline reproduce; echo $?` → prints the reason, exits **2** |

## What it does, and what it will do

Four capabilities. Each is a claim this README only makes once a command proves it.

1. **A fail-closed gate** (built) — the Clopper-Pearson exact lower bound on the success rate,
   compared with a declared threshold, plus a declared minimum *n*, returned as a CI exit code. Too
   small a sample fails; it does not pass. The bound is the lower end of a two-sided 95% interval,
   i.e. a one-sided 97.5% lower confidence bound (ruling D6).
2. **A structured report** (built) — machine-readable first, human-readable second, regenerable
   from saved results without re-running.
3. **Out-of-envelope flagging** (framework built; conditions planned) — the conditions a run was
   tested for are written down first, and a run outside them is **flagged, never silently
   reported**. The gate never passes a flagged run. Which conditions are monitored is open
   (ruling D10).
4. **Determinism** (pinned; cross-host evidence pending) — each episode records an
   action-trajectory hash, and every persisted run records the host it ran on, so a run can be
   compared with a re-run. The hash depends on torch's thread count and kernel instruction set, so
   the image pins both, and `sweep` refuses to run off the pin (ruling D9b). The claim is bit-exact
   reproduction on any x86_64 host with AVX2 under that pin. Run-to-run equality is tested on one
   host; matching hashes on a second host have not yet been shown.

The harness is the product; the policy is a fixture. It evaluates a small pretrained policy from the
[LeRobot](https://github.com/huggingface/lerobot) ecosystem on a benchmark task that ecosystem
already ships. No training runs, no ROS dependency.

## Getting started

```bash
uv sync --group dev          # create the environment
uv run pytest                # run the suite
uv run fathomline --version  # the CLI
```

Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md). Security reports:
[`SECURITY.md`](SECURITY.md).

## Scope and non-goals

- **Not** a policy, a model, or a training run. Nothing here tries to make a robot better at a task.
- **Not** a benchmark leaderboard. The measurement apparatus is the subject.
- **Not** a claim about real hardware. Everything here runs in simulation, and any result says which.

## License

MIT — see [LICENSE](LICENSE).
