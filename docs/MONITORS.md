# Monitors — the declared envelope at run time

_Lane G. One monitor per runtime condition in ruling D10 ([`ROADMAP.md`](ROADMAP.md)). The values
are declared in [`src/fathomline/monitors/envelope.py`](../src/fathomline/monitors/envelope.py);
`fixture_monitors()` there is the suite a run of the D1 fixture is monitored with. A flag on any
step flags the run, and the gate never passes a flagged run (Lane F). The other D10 conditions are
added here as their monitors land._

| D10 | Condition | Monitor | Status |
|---|---|---|---|
| 2 | Action out of bounds | `ActionBound`: arm dimensions 0–5 within ±1.1; gripper excluded | built |
| 5 | Unseen init | none: held by construction (see `envelope.py`) | declared |

## Action out of bounds

**What it checks.** Every step, each of LIBERO's six arm dimensions (three translations, three
rotations) must lie within ±1.1. A value exactly at 1.1 is inside; a NaN or an infinity is
outside. The gripper (dimension 6) is not checked: it passes ±1 on most clean steps. The bound is
not the training range: clean runs leave it.

**Why 1.1, fixed.** The envelope spike ([`SPIKE-envelope.md`](SPIKE-envelope.md)) measured clean
arm peaks up to 1.01 and lost-stats peaks of 2.27–2.68. The margin was picked after seeing those
data, so D10 fixes it as a declared value. It is not re-fitted from any run, including the check
below.

**What it misses.** It is a backstop for faults that reach the actions, not a shift detector. In
the spike, lighting, noise, camera, instruction and init faults all left the actions looking
normal. A subtle metadata swap could also stay inside ±1.1; D10's provenance condition is the
check meant for that.

## Real-episode check

The D1 fixture in the container, through the harness's own loop, with `fixture_monitors()`
attached. RESULTS_PENDING

What these numbers can and cannot show:

- **The lost-stats rows on seeds 1000–1029 replay the spike's episodes.** Their action hashes match
  the spike's, so the spike's own data (arm peaks 2.27–2.68) already implied the result. They show
  that the monitor is wired correctly into the real loop on real seven-dimensional actions. They
  are not out-of-sample evidence. The rows on seeds 1050–1079 are new episodes.
- **"Held-out" here means new rollouts, not new starts.** `libero_object` task 0 has 50 benchmark
  init states, and seed *s* starts from state `s % 50`. Seeds 1050–1099 therefore start where the
  spike's seeds 1000–1049 did. Only the policy's sampling noise differs, so the trajectories after
  the first step differ too.
- **Action hashes are compared on the same CPU** as the spike, because the hash is known to differ
  across CPUs (D9a). Both ran on the spike's host, an AMD Ryzen 7 5800X under Docker Desktop.
  Rows from runs after this check was written also record the CPU model.

## Reproduce the real-episode check

Build the image from this branch: the source is copied in at build time. Then run each condition,
with this check and the spike's fault injection mounted. These are Linux paths; on Windows, use
absolute paths in `-v`. Runs of one condition append to the same file, but a seed already in it is
refused. `--summarize` reports seeds the spike also ran (replays) and new seeds as separate rows.

```bash
docker build -t fathomline .
mkdir -p runs/monitor-checks
docker run --rm --network none --cpuset-cpus 0-3 --memory 12g \
  -v "$PWD/docs/monitor-checks:/check:ro" -v "$PWD/docs/spike-envelope:/spike:ro" \
  -v "$PWD/runs/monitor-checks:/out" --entrypoint python fathomline /check/check_monitors.py \
  --condition unnorm_identity --first-seed 1000 --episodes 30 --out /out
# ...and likewise unnorm_identity from 1050 for 30 episodes, and baseline from 1050 for 50.
uv run python docs/monitor-checks/check_monitors.py --summarize runs/monitor-checks \
  --spike-results docs/spike-envelope/results
```

The per-episode JSONL behind the numbers above is committed in `docs/monitor-checks/results/`;
`--summarize docs/monitor-checks/results` regenerates the table from it.
