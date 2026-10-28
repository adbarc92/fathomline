# Monitors — the declared envelope at run time

_Lane G. One monitor per runtime condition in ruling D10 ([`ROADMAP.md`](ROADMAP.md)). The values
are declared in [`src/fathomline/monitors/envelope.py`](../src/fathomline/monitors/envelope.py);
`fixture_monitors()` there is the suite a run of the D1 fixture is monitored with. A flag on any
step flags the run, and the gate never passes a flagged run (Lane F)._

| D10 | Condition | Monitor | Status |
|---|---|---|---|
| 2 | Action out of bounds | `ActionBound`: arm dimensions 0–5 within ±1.1; gripper excluded | built |

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
normal. A subtle metadata swap could also stay inside ±1.1; the provenance monitor covers that.

**Real-episode check.** RESULTS_PENDING

## Reproduce the real-episode check

Build the image, then run each condition with the source, this check and the spike's fault
injection mounted. These are Linux paths; on Windows, use absolute paths in `-v`.

```bash
docker build -t fathomline .
mkdir -p runs/monitor-checks
docker run --rm --network none --cpuset-cpus 0-3 --memory 12g \
  -v "$PWD/docs/monitor-checks:/check:ro" -v "$PWD/docs/spike-envelope:/spike:ro" \
  -v "$PWD/runs/monitor-checks:/out" --entrypoint python fathomline /check/check_monitors.py \
  --condition unnorm_identity --first-seed 1000 --episodes 30 --out /out
docker run --rm --network none --cpuset-cpus 0-3 --memory 12g \
  -v "$PWD/docs/monitor-checks:/check:ro" -v "$PWD/docs/spike-envelope:/spike:ro" \
  -v "$PWD/runs/monitor-checks:/out" --entrypoint python fathomline /check/check_monitors.py \
  --condition baseline --first-seed 1050 --episodes 50 --out /out
uv run python docs/monitor-checks/check_monitors.py --summarize runs/monitor-checks \
  --spike-results docs/spike-envelope/results
```

The fault is injected by the spike's own `install`, unchanged, so on the seeds both ran the action
hashes must equal the spike's. The per-episode JSONL is in
[`monitor-checks/results/`](monitor-checks/results/).
