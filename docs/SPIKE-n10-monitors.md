# Spike — the D10 monitors at `n_action_steps=10` @ 2026-10-29

_Evidence, not a build. PR #7 makes the fixture replan every 10 actions instead of every 50, and
Alex has ruled to adopt it. D10 says its thresholds must be re-checked when the fixture's
behaviour changes. Lane G measured them at 50 (`docs/MONITORS.md` on #4–#8). This re-runs Lane G's
check at 10, on the `full` seed set, which also gives the clean number D14's gate is set against.
Nothing in `src/` changes here._

## TL;DR

- **D14: 47/50 clean successes at 10** (94%; two-sided 95% Clopper-Pearson [0.83, 0.99]). The
  gate's lower bound is **0.8345**, so a threshold of 0.834 or below passes and 0.835 or above
  fails. D14's proposed 0.5 gate would pass. At 50 the same seeds scored 27/50; 21 seeds gained,
  1 lost (McNemar p = 1.1e-5).
- **The ±1.1 arm bound holds unchanged.** Clean arm peaks are 0.92–1.01 (0/50 flagged), the same
  range as Lane G's held-out clean episodes at 50 (0.91–1.01). Lost unnormalization statistics are
  still caught on 10/10 episodes, by the bound at steps 3–6 and by provenance at step 0.
- **The frame band holds unchanged.** Over every frame of the 50 clean episodes, the third-person
  camera's brightness stays in [101.44, 105.22] and its noise in [5.81, 6.44]: about 10% inside
  each end of the declared band, and 0/50 flagged. Lane G's rule applied to these episodes would
  give a band within 0.1 of the declared one.
- **No monitor flags a clean episode at 10:** 0/50 under all four.

## What was run

Lane G's real-episode check, [`check_monitors.py`](https://github.com/adbarc92/fathomline/blob/lane/g-observation-shift/docs/monitor-checks/check_monitors.py),
unchanged. Each episode runs through the core's `run_episode` with `fixture_monitors(...)` (all
four D10 monitors), plus the script's arm-peak and frame-statistics recorders, which never flag.

| Condition | Seeds | Episode length | Episodes |
|---|---|---|---|
| `baseline` (clean) | 1000–1049, the `full` set in `tests.lock` (#1) | full (280 steps max) | 50, two containers of 25 |
| `unnorm_identity` (lost statistics, lane E's fault) | 1000–1009 | cut to 20 steps (`--max-steps 20`) | 10 |

**Source.** The top of Lane G's stack (#8's head, `35bda3b`) with #7's three commits
cherry-picked on top. The cherry-pick applied without conflicts, and the unit suite passes on
the result. That source was mounted over the image's `src`, in the image lane E and the baseline
spike used; the Python stack and the baked snapshots are the same as a build from main.

**Same actions as #7 alone.** On seeds 1000–1019 the action hashes equal those of the baseline
spike's `harness-n10` run (#10's `docs/spike-baseline/results/harness-n10.jsonl`), which ran #7's
source with no monitors on the same image and host: 20/20. So the monitors do not change the
actions, and stacking #7 on Lane G changes nothing #7 does. The hashes predate D9a's pin
(`ATEN_CPU_CAPABILITY=avx2` and a fixed thread count), so they are not reference hashes.

**Host.** One x86_64 CPU (AMD Ryzen 7 5800X), 8 logical CPUs and 10 GB per container, two
containers at a time.

## Results

From `check_monitors.py --summarize` on [`spike-n10-monitors/results/`](spike-n10-monitors/results/):

| Condition | Seeds (n) | Flagged | First flag step (min / median / max) | Arm peak range |
|---|---|---|---|---|
| `baseline` | 1000-1049 (50) | 0/50 | - | 0.92 to 1.01 |
| `unnorm_identity` | 1000-1009 (10) | 10/10 | 0 / 0 / 0 | 1.99 to 2.39 |

`unnorm_identity`: `provenance` flagged 10/10, first at step 0. `action-bound` flagged 10/10, first
at step 3 / 4 / 6 (min / median / max).

### D14: the clean rate at 10

| Run | Seeds | Successes | 95% CP interval | Paired with 50 on the same seeds |
|---|---|---|---|---|
| This run, 10 | 1000–1049 | **47/50** (94%) | [0.83, 0.99] | 21 gained, 1 lost; McNemar p = 1.1e-5 |
| Lane E's baseline, 50 | 1000–1049 | 27/50 (54%) | [0.39, 0.68] | — |

- **Failures:** seeds 1016, 1025 and 1026, each a 280-step timeout. 1016 is the one seed that
  succeeds at 50 and fails at 10, as in the baseline spike.
- **The gate.** D6's gate passes when the interval's lower end (0.8345 here) is at or above the
  threshold, at least `min_episodes` ran, and no episode carries a flag. This run meets the flag
  condition (0/50). On the threshold, it passes at 0.834 or below and fails at 0.835 or above.
  A deliberate FAIL needs a threshold above 0.8345; a published PASS needs one at or below it.
- **Successful episodes take 128–252 steps** (median 140).
- **Wall-clock** on 8 CPUs: 250–965 s for a success (median 352 s), and 700–1114 s for a
  280-step failure, with two containers sharing the host.

### D10 condition 2: the arm bound (±1.1, dims 0–5)

- **Clean:** arm peaks 0.92–1.01 (median 0.97) over 50 episodes, 0 flagged. The largest is
  1.0094 (seed 1024); three episodes exceed 1.0. Lane G's held-out clean episodes at 50 peaked
  at 0.91–1.01, so the margin to 1.1 is the same, about 0.09.
- **Fault:** lost unnormalization statistics push the arm to 1.99–2.39, and the bound flags all
  10 episodes at steps 3–6 (Lane G at 50: 60/60 at steps 0–7).
- **Verdict:** ±1.1 still separates clean from the fault with room on both sides. No change.

### D10 condition 3: the frame band (third-person camera)

| Statistic | Declared band (from 50) | Clean range at 10, every frame of 50 episodes | Headroom, low / high |
|---|---|---|---|
| `image.brightness` | [91.19, 115.82] | [101.444, 105.217] | 10.1% / 10.1% |
| `image.noise` | [5.22, 7.09] | [5.8121, 6.4402] | 10.2% / 10.1% |

- **0/50 clean episodes flagged.** The clean range at 10 is the held-out range at 50 to within
  0.12 brightness and 0.01 noise (Lane G at 50: [101.32, 105.29] and [5.80, 6.44]).
- **Lane G's rule (`--calibrate`, held-out range ± 10%) on these episodes gives** (91.3, 115.74)
  and (5.23, 7.08), within 0.1 of the declared band. The declared band is not re-fitted from
  these episodes: they are scored against it, and they pass.
- **Verdict:** the band holds. No change.

### The faults Lane G caught at step 0

Lane G's dim, bright and noise faults (band), wrong-task and paraphrase (instruction), and rename
swap (provenance) were all flagged at step 0 at 50. Step 0 is judged on the first observation,
which is rendered from the init state before any action runs, so the replan horizon cannot change
it. Frame 0 is bit-identical between these episodes and Lane G's clean episodes at 50 on the 30
seeds both ran (1000–1029; brightness differs by 0.0 on both cameras). Those detections therefore
carry over to 10 without a re-run. The one detection that depends on the policy's actions, the
arm bound on lost statistics, is re-run above.

## Not tested

- **Lane E's perturbation table at 10** (`SPIKE-envelope.md`): the success rates under each fault
  were not re-measured, only whether the monitors flag.
- **Seeds 1050–1099 at 10.** Lane G calibrated the band on them at 50. Nothing was calibrated on
  1000–1049 at 10, so they serve as clean episodes here.
- **The camera swap's late flags** (steps 9–27 at 50, documented as incidental). They depend on
  the trajectory and may move at 10.
- **Provenance does not see the replan horizon.** `n_action_steps` is not in the digested loaded
  pipeline, and the four digests are unchanged by #7. The run manifest records the horizon (#7).

## Reproduce

From a checkout with #7 and #8 fetched (`git fetch origin pull/7/head pull/8/head`); Linux
paths. The image supplies only the Python stack and the snapshots, since the mounted source
replaces its `src`. Built after D9a's pin lands, it will give different action hashes.

```bash
git worktree add -b n10-stack ../n10-stack 35bda3b
git -C ../n10-stack cherry-pick bda6432 3618421 dd0cf1e
docker build -t fathomline .
S=$(realpath ../n10-stack) OUT=$PWD/runs/n10
docker run --rm --network none --cpuset-cpus 0-7 --memory 10g \
  -v "$S/src:/app/src:ro" -v "$S/docs/monitor-checks:/check:ro" \
  -v "$S/docs/spike-envelope:/spike:ro" -v "$OUT:/out" \
  --entrypoint python fathomline /check/check_monitors.py \
  --condition baseline --first-seed 1000 --episodes 25 --out /out/clean-a
# ...the same with --first-seed 1025 --out /out/clean-b on --cpuset-cpus 8-15, and with
# --condition unnorm_identity --first-seed 1000 --episodes 10 --max-steps 20 --out /out
cat "$OUT"/clean-a/baseline.jsonl "$OUT"/clean-b/baseline.jsonl > "$OUT"/baseline.jsonl
uv run --project "$S" python "$S/docs/monitor-checks/check_monitors.py" --summarize "$OUT"
uv run --project "$S" python "$S/docs/monitor-checks/check_monitors.py" \
  --calibrate "$OUT"/baseline.jsonl
```

The committed `results/` hold `baseline.jsonl` (both halves concatenated) and
`unnorm_identity.jsonl`.
