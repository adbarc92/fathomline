# Spike — envelope evidence for D10 @ 2026-10-27

_Lane E. A spike, not a build: it measures, and it ships no monitor. The scripts and the raw
per-episode data are in [`spike-envelope/`](spike-envelope/). D10 is Alex's to rule. The last
sections are a recommendation, not a ruling._

## TL;DR

- **Small perturbations break the fixture.** Dim lights, camera noise and 5 cm object shifts
  each cut success from about 50% to 7–10%. A camera mix-up, lost action stats and another
  task's instruction each cut it to 0/30.
- **The fixture is a 54% policy, not the ~95% that LIBERO-Plus reports for VLAs on standard
  LIBERO.** The baseline is 27/50, with an interval of [0.39, 0.68]. LeRobot's own eval loop,
  run on the same loaded objects and seeds 1000–1009, got 6/10 against the harness's 7/10. That
  cross-check script is not committed.
- **Only one fault leaves a trace in the actions.** Lost action stats push the arm to 2.3–2.7
  on every episode. In the one episode traced, the arm passed ±1 at step 5. In every other
  condition the policy fails while its actions look normal. An action-bound monitor is a
  backstop, not a shift detector.
- **The obvious bounds are wrong.** The gripper passes ±1 on 68% of clean steps, and the
  training range is exceeded in 38 of 50 clean episodes. Either bound would flag a healthy run.
- **Recommended D10 set:** provenance mismatch, action out of bounds (arm only, ±1.1),
  observation shift (an image-statistics band), and instruction outside the declared set.
  Unseen init is held by construction for now. Details at the end.

## What was run

The D1 fixture is SmolVLA (`lerobot/smolvla_libero` @ `31d453f`) on LIBERO `libero_object`
task 0. It ran in the repo's own image, through the harness's step loop (`run_episode`) and the
adapter's own env and policy. Each condition patches exactly one seam, where a real fault would
enter.

| Condition | Seam | What changes | D10 candidate it stands for |
|---|---|---|---|
| `baseline` | — | nothing | — |
| `light_dim` | scene | every MuJoCo light and the headlight × 0.35 | observation shift |
| `light_bright` | scene | every light × 2.5, clipped at 1.0 | observation shift |
| `noise` | camera | Gaussian pixel noise, σ = 20 of 255, both cameras | observation shift |
| `camera_swap` | wiring | the third-person and wrist images trade places | provenance mismatch (config) |
| `unnorm_identity` | action metadata | the action unnormalizer's stats set to mean 0, std 1, as if lost | provenance mismatch (metadata) |
| `instruction_paraphrase` | language | "grab the alphabet soup can and put it into the basket" | unseen instruction |
| `instruction_other` | language | task 1's instruction ("pick up the cream cheese and place it in the basket"), scored on task 0's goal | unseen instruction |
| `init_shift` | initial state | every free object moved up to 5 cm in x and y after the benchmark init state | unseen init |

**Episodes.** The baseline ran 50 episodes: seeds 1000–1049, which are the task's 50 benchmark
init states, once each. Each perturbation ran 30 episodes on the same first 30 starts, so every
comparison is paired. Thirty was fixed before the full run, after a one-episode smoke test of
each hook. It is enough to separate a
collapse from no effect: 0/30 has an upper bound of 0.12. It also kept the run to about 13
episode-hours (290 episodes) instead of about 20.

**Statistics.** Each rate carries its two-sided 95% Clopper-Pearson interval (D6a). The lower end
comes from the repo's `clopper_pearson_lower`. The shift is tested with an exact McNemar test on
the paired starts. On those 30 starts the baseline is 15/30, so the shift column is measured
against 50%, in percentage points.

**Host.** A Ryzen 7 5800X (8 cores, 16 threads) under Docker Desktop. Up to four containers ran
at once, each pinned to 4 logical CPUs. Those CPUs are hyperthreads, so the times here are not
CI runner times. The median episode took 92 s when it succeeded and 187 s when it failed.

## Success rate per condition

| Condition | Successes | Rate | 95% CP interval | Shift vs baseline, same starts | Lost / gained (paired) | McNemar p |
|---|---|---|---|---|---|---|
| `baseline` | 27/50 | 54% | [0.39, 0.68] | — | — | — |
| `camera_swap` | 0/30 | 0% | [0.00, 0.12] | -50 pp | 15 / 0 | 6.1e-05 |
| `unnorm_identity` | 0/30 | 0% | [0.00, 0.12] | -50 pp | 15 / 0 | 6.1e-05 |
| `instruction_other` | 0/30 | 0% | [0.00, 0.12] | -50 pp | 15 / 0 | 6.1e-05 |
| `noise` | 2/30 | 7% | [0.01, 0.22] | -43 pp | 13 / 0 | 0.00024 |
| `light_dim` | 3/30 | 10% | [0.02, 0.27] | -40 pp | 13 / 1 | 0.0018 |
| `init_shift` | 3/30 | 10% | [0.02, 0.27] | -40 pp | 14 / 2 | 0.0042 |
| `light_bright` | 8/30 | 27% | [0.12, 0.46] | -23 pp | 10 / 3 | 0.092 |
| `instruction_paraphrase` | 18/30 | 60% | [0.41, 0.77] | +10 pp | 2 / 5 | 0.45 |

Seven of eight perturbations hurt. Six are clear collapses. `light_bright` probably hurts, but
30 episodes cannot confirm it (p = 0.09). The paraphrase shows no detectable change. Under task
1's instruction the policy never completes task 0, which says the instruction steers it. That
differs from LIBERO-Plus's finding that VLAs largely ignore language.

## Action-bound excursions

The action is 6 arm deltas and a gripper command. LIBERO's action space is [-1, 1] on each, and
the robot controller clips to it. The training range is each dimension's [min, max] over the
fine-tune's training actions, from the checkpoint's own stats: translation ±0.94 on x, y and
z; rotation about [-0.26, 0.36], [-0.38, 0.38] and [-0.37, 0.38]; gripper ±1.

| Condition | Gripper beyond ±1: episodes (steps) | Arm beyond train range | Arm beyond ±1 | Arm beyond ±1.1 | Arm peak, per-episode range | Gripper peak |
|---|---|---|---|---|---|---|
| `baseline` | 50/50 (68%) | 38/50 | 3/50 | 0/50 | 0.89 to 1.01 | 1.10 |
| `camera_swap` | 30/30 (17%) | 2/30 | 0/30 | 0/30 | 0.78 to 0.97 | 1.06 |
| `init_shift` | 30/30 (63%) | 25/30 | 6/30 | 0/30 | 0.90 to 1.01 | 1.10 |
| `instruction_other` | 30/30 (62%) | 26/30 | 6/30 | 0/30 | 0.92 to 1.04 | 1.09 |
| `instruction_paraphrase` | 30/30 (38%) | 15/30 | 0/30 | 0/30 | 0.89 to 0.99 | 1.11 |
| `light_bright` | 30/30 (70%) | 22/30 | 1/30 | 0/30 | 0.90 to 1.00 | 1.11 |
| `light_dim` | 30/30 (59%) | 26/30 | 1/30 | 0/30 | 0.90 to 1.00 | 1.10 |
| `noise` | 30/30 (58%) | 18/30 | 2/30 | 0/30 | 0.88 to 1.01 | 1.10 |
| `unnorm_identity` | 30/30 (46%) | 30/30 | 30/30 | 30/30 | 2.27 to 2.68 | 1.12 |

- **The gripper carries no signal.** It overshoots ±1 on most steps of every condition, clean or
  not, and peaks at 1.06–1.12 everywhere. A gripper bound would flag every healthy run.
- **The training range is too tight.** Clean runs leave it in 38 of 50 episodes, always on the
  z translation, by up to 0.07 (1.01 against 0.94).
- **A strict ±1 on the arm is almost right.** It flags 3 of 50 clean episodes (peak 1.01).
- **±1.1 on the arm separates cleanly here:** 30/30 for lost stats, 0 of 260 other episodes.
  That margin was picked after seeing these data. A ruling should fix it before Lane G measures.
- In a rerun of one lost-stats episode, the arm first left ±1 at **step 5**, a quarter-second in.
  The rerun reproduced the recorded action hash exactly.

## Observation statistics

Two numbers per episode come from the first third-person frame the policy sees. One is mean
brightness. The other is a high-frequency energy, the mean absolute difference between neighbouring
pixels, which noise raises. "Flagged" means outside the baseline's range, with each end moved out by
10% of its value (brightness band about 92 to 113). The band is calibrated on the same baseline it
is scored on, so zero false alarms is optimistic.

| Condition | Frame-0 brightness | Frame-0 noise | Flagged by the frame band |
|---|---|---|---|
| `baseline` | 102 to 103 | 5.8 to 5.9 | 0/50 |
| `light_dim` | 37 to 37 | 2.2 to 2.2 | 30/30 |
| `light_bright` | 140 to 140 | 6.7 to 6.8 | 30/30 |
| `noise` | 102 to 102 | 15.4 to 15.5 | 30/30 |
| `camera_swap` | 97 to 101 | 6.0 to 6.6 | 2/30 |
| `init_shift`, `instruction_*`, `unnorm_identity` | 102 to 103 | 5.6 to 5.9 | 0/30 each |

Two cheap statistics catch every lighting and noise episode. They miss the camera swap, because
both views have similar statistics.

## Cost to monitor each candidate at runtime

For scale, one step of the policy and simulator took a median of 672 ms here.

| Candidate | What the check reads | When | Cost | Caught here | Clean runs flagged |
|---|---|---|---|---|---|
| Provenance mismatch | a hash of the stats, rename map and camera mapping the loaded pipeline actually uses, against the pinned values | once, at load | a hash over a few kB; negligible (not timed) | lost stats: yes, by construction. Camera swap: only if the swap is in config, not in the pixel stream as injected here | 0 by construction |
| Action out of bounds | each arm dimension within ±1.1 | every step | six comparisons; negligible | lost stats 30/30 (past ±1 at step 5 in the one traced) | 0/50 |
| Observation shift | frame-0 brightness and noise inside a declared band | once per episode (or per frame) | 2.9 ms per frame in NumPy, about 0.4% of a step | dim 30/30, bright 30/30, noise 30/30; camera swap 2/30 | 0/50 (in-sample) |
| Unseen instruction | the instruction is in the declared set, exact match | once per episode | a string compare | other task 30/30. The paraphrase is flagged too (30/30), though no harm was measured | 0/50 |
| Unseen init | object poses near a declared init state | once per episode | needs simulator-privileged poses; not measured | not evaluated | not evaluated |

## What others have measured

- **LIBERO-Plus** (Fei et al., arXiv 2510.13626) tests seven perturbation axes, including light,
  sensor noise, language, camera viewpoint and robot init state. VLA success falls "from 95% to
  below 30% under modest perturbations". Camera viewpoint and robot init state hurt most.
- **LIBERO-PRO** (Zhou et al., arXiv 2510.03827): models above 90% on standard LIBERO fall to
  0.0% under its generalized setting.
- **Same Weights, Different Robot** (Tai, arXiv 2606.03724): swapping the action-unnormalization
  metadata for a plausible sibling key drops LIBERO-Goal from 28/28 to 2/28. The paper concludes
  that action-space metadata should be checked before rollout.

These numbers agree in direction and size. The one difference is language: this fine-tune
follows its instruction.

## Recommended D10 set — for Alex to rule

1. **Provenance mismatch — in.** It is the cheapest check and the costliest failure: lost stats
   and swapped cameras each gave 0/30, and the published metadata swap gave 2/28. It must check
   what the *loaded pipeline* uses, not only the on-disk digests that `pins.verify` already
   checks. This experiment's fault lived in memory, after load.
2. **Action out of bounds — in, arm only, at a declared margin.** It costs nothing and caught the
   gross metadata fault on every episode, by step 5 in the one traced. The gripper must be
   excluded, and the training range must not be the bound. It will **not** catch observation
   shift, so it backs up item 1 and does not replace item 3. A subtle metadata swap, like the published sibling-key case, may stay inside
   ±1.1. That is another reason item 1 is needed.
3. **Observation shift — in, narrowly defined.** Declare an image-statistics band, measured on
   the declared conditions. It caught every lighting and noise episode for about 0.4% overhead.
   It is a coarse detector: it misses a camera swap and anything that leaves the statistics
   alone. Lane G should calibrate it on held-out episodes.
4. **Instruction outside the declared set — in.** It is free, and a foreign instruction gave
   0/30. The cost is conservatism: a harmless paraphrase is flagged too. That fits "flag what was
   not tested for".
5. **Unseen init — declared, enforced by construction, no runtime monitor yet.** The effect is
   large (3/30). But the harness picks the init state itself (`seed % 50` over the benchmark's
   file), so its own runs stay in the envelope by construction. A geometric monitor needs
   simulator-only object poses and was not evaluated. It becomes necessary once the harness runs
   inits it did not choose.

**Also for Alex, outside D10:**
- The fixture's 54% baseline bounds any gate. At n = 50, a threshold above about 0.39 fails the
  fixture as shipped, with no perturbation at all.
- Same-host determinism held twice here. A full-run episode reproduced its smoke-run action hash.
  A rerun on a different CPU count reproduced its recorded hash. Both were on one host, so this
  says nothing about the cross-host question in D9a.

## Not measured

- **Axes:** camera viewpoint (LIBERO-Plus's largest effect), background textures, robot init
  state, and object layout beyond 5 cm.
- **Severity:** one level per axis, so no dose-response curves.
- **Metadata:** a subtle in-bounds swap, like the published sibling-key case. Only the gross
  case was tested.
- **Scope:** one policy, one task (task 0), 30 episodes per perturbation.
- **Calibration:** the frame band was not checked on held-out episodes. No learned OOD detector
  was tried.
- **Traces:** per-step actions were stored only as per-episode summaries, except in the one
  rerun.
- **CI:** nothing ran on the CI runner shape.

## Reproduce

Build the image, then run the conditions. These are Linux paths; on Windows, use absolute paths
in `-v`.

```bash
docker build -t fathomline .
mkdir -p runs/envelope
docker run --rm --network none --cpuset-cpus 0-3 --memory 12g \
  -v "$PWD/docs/spike-envelope:/spike:ro" -v "$PWD/runs/envelope:/out" \
  --entrypoint python fathomline /spike/run_envelope.py \
  --condition baseline --episodes 50 --out /out
docker run --rm --network none --cpuset-cpus 0-3 --memory 12g \
  -v "$PWD/docs/spike-envelope:/spike:ro" -v "$PWD/runs/envelope:/out" \
  --entrypoint python fathomline /spike/run_envelope.py \
  --condition light_dim --condition noise --episodes 30 --out /out   # and so on
uv run python docs/spike-envelope/summarize.py runs/envelope
```

`--first-seed` and `--episodes` split a condition across containers. The run behind this primer
did that, and its data was merged with no seed duplicated. The JSONL is in
[`spike-envelope/results/`](spike-envelope/results/). `summarize.py` regenerates every table
above from it, except the frame-band row that groups four conditions.
