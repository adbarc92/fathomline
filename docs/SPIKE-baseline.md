# Spike — why the fixture's baseline is 54% @ 2026-10-27

_A spike, not a build. It asks one question: is the D1 fixture's 27/50 a harness bug, or what
this checkpoint does? The scripts are in [`spike-baseline/`](spike-baseline/). The fix it points
to is a separate PR (#7); Alex has ruled to adopt it._

## TL;DR

- **Not a harness bug. Stock `lerobot-eval` gets the same rate.** With no Fathomline code in the
  path, on seeds 1000–1019, it got 14/20 against the harness's 14/20. The two disagree on 2 of
  the 20 episodes, one each way: no detectable difference, though 20 pairs bound it only loosely.
- **The cause is one setting in the checkpoint's config.** `lerobot/smolvla_libero` @ `31d453f`
  ships `n_action_steps: 50`, so the policy plays a whole 50-action chunk before it looks at the
  scene again. Both the harness and stock `lerobot-eval` use it, because both read the config.
- **That setting is the worst of the four the SmolVLA paper tests on LIBERO.** Table 13 gives
  LIBERO-Object 70% at 50 actions per observation, and 94% at 1 or 10. The 96% headline (Table 2)
  replans after every action. LeRobot's tracker reports the same gap for this checkpoint
  (huggingface/lerobot#4614, open).
- **At `n_action_steps=10`, the fixture is a ~95% policy.** Harness: 19/20 on seeds 1000–1019
  (95% CI [0.75, 1.00]). Stock `lerobot-eval`: 10/10 on seeds 1000–1009 ([0.69, 1.00]), level
  with the harness on every one of those ten.
- **Launch-thread sentence:** "Fathomline reproduces stock `lerobot-eval` on this checkpoint
  (14/20 each on the same seeds); the 54% is the checkpoint's shipped `n_action_steps=50`, which
  the SmolVLA paper itself shows costs about 25 points on LIBERO-Object."

## Reference numbers

| Source | Checkpoint | LIBERO-Object | Settings that matter |
|---|---|---|---|
| SmolVLA paper, Table 2 ([arXiv 2506.01844](https://arxiv.org/abs/2506.01844)) | the paper's 0.45B fine-tune | **96%** (Spatial 90, Goal 92, Long 71; avg 87.3) | 10 trials per task; "predicting a new action after each executed action", i.e. 1 action per observation; trained 100k steps, batch 64, on `physical-intelligence/libero` |
| SmolVLA paper, Table 13 (action execution steps) | the paper's ablation model | **94 / 94 / 91 / 70%** at 1 / 10 / 30 / 50 actions per observation | avg over the four suites 80.3 / 82.8 / 70.8 / 51.8% |
| huggingface/lerobot#4614 (open, 2026-09) | `lerobot/smolvla_libero` (this one) | **42/50 at 10, 32/50 at 50**, tasks 0–9 × 5 episodes, paired | lerobot 0.6.1, Apple MPS; a CUDA and a CPU reproduction in the same thread agree at 10 |
| same thread, CPU (x86, WSL2) | `lerobot/smolvla_libero` | **89/100 at 10**; task 0: **10/10** | lerobot 0.6.1, MuJoCo 3.8.1, init states 0–9 per task |

- The paper reports no per-task numbers, and no SmolVLA number on LIBERO at all for
  `lerobot/smolvla_libero`. Its model card has no eval section.
- **This checkpoint is not the paper's.** Its `train_config.json` records 25k steps at batch 32
  on `lerobot/libero`, an eighth of the paper's samples. So 96% is not its target either. The
  third-party 89% at 10 is the closest like-for-like number.
- The other settings agree between the checkpoint, the harness and stock `lerobot-eval`: two
  cameras (`agentview_image` → `camera1`, `robot0_eye_in_hand_image` → `camera2`), rendered at
  360×360 and flipped 180° by LeRobot's `LiberoProcessorStep`, padded to 512×512 by SmolVLA;
  an 8-dim state (end-effector position, axis-angle, gripper qpos); relative control at 20 Hz;
  280 steps for `libero_object`; LIBERO's own instruction ("pick up the alphabet soup and place it
  in the basket"). The training images were 256×256; nothing here tested 256.

## What was run

All on libero_object task 0, on the same starts as the fixture: seed 1000+i meets benchmark init
state i.

| Run | Loop | Env and policy built by | Checkpoint files | `n_action_steps` | Episodes |
|---|---|---|---|---|---|
| `harness-n50` | Fathomline `run_episode` | the adapter (`load_fixture`) | baked into the image | 50 (shipped) | 50 (lane E's baseline, reused) |
| `stock-n50` | `lerobot-eval` | `lerobot-eval` | fetched from the Hub at `31d453f` | 50 (shipped) | 20 |
| `stock-n10` | `lerobot-eval` | `lerobot-eval` | fetched from the Hub at `31d453f` | 10 (`--policy.n_action_steps=10`) | 10 |
| `harness-n10` | Fathomline `run_episode` | the adapter, with the fix branch's override | baked into the image | 10 | 20 (four containers × 5) |

**The stock runs are independent of the harness.** [`run_stock_eval.sh`](spike-baseline/run_stock_eval.sh)
takes the image only for its locked Python stack and the LIBERO meshes. It drops the harness from
`PYTHONPATH` and checks that `import fathomline` fails, turns the Hub back on, and downloads the
checkpoint, the VLM and the tokenizer itself. The flags are the ones LeRobot's own CI passes for
this checkpoint (`.github/workflows/benchmark_tests.yml`: the camera mapping and one empty
camera), plus CPU. So the stock runs share nothing with the harness but the installed packages
and the LIBERO meshes baked into the image (pinned by digest). Lane E's earlier
cross-check (6/10 against 7/10) shared the loaded env and policy objects too.

**Episode counts.** The host was shared with another lane's four containers, so an episode took
10 to 55 minutes. The arms were cut before they started: `stock-n50` and `harness-n10` to 20
episodes (seeds 1000–1019), `stock-n10` to 10 (seeds 1000–1009). `lerobot-eval` cannot start at
an init state other than 0 in a single env, so it could not be split across containers the way
the harness can.

Lane E's `run_envelope.py` does not record the policy's provenance, so the `harness-n10` rows
carry no `n_action_steps` field. They ran with PR #7's source mounted over the image's (its
override, `3618421`, or later).

**Statistics.** Each rate carries its two-sided 95% Clopper-Pearson interval (D6a). Runs on the
same seeds are paired and tested with an exact McNemar test.
[`summarize.py`](spike-baseline/summarize.py) regenerates the table from the files in
[`spike-baseline/results/`](spike-baseline/results/).

## Results

From `summarize.py`, first label `harness-n50`:

| Run | Task | Seeds | Successes | Rate | 95% CP interval | First run, same seeds | Lost / gained vs first run | McNemar p |
|---|---|---|---|---|---|---|---|---|
| harness-n50 | 0 | 1000-1049 | 27/50 | 54% | [0.39, 0.68] | — | — | — |
| stock-n50 | 0 | 1000-1019 | 14/20 | 70% | [0.46, 0.88] | 14/20 | 1 / 1 | 1 |
| harness-n10 | 0 | 1000-1019 | 19/20 | 95% | [0.75, 1.00] | 14/20 | 1 / 6 | 0.12 |
| stock-n10 | 0 | 1000-1009 | 10/10 | 100% | [0.69, 1.00] | 7/10 | 0 / 3 | 0.25 |

Paired at the new setting (first label `harness-n10`): `stock-n10` scores 10/10 against the
harness's 10/10 on seeds 1000–1009, with no discordant episode.

- **No difference between the loops is detectable at either setting.** At 50 they disagree on
  two of twenty seeds, in opposite directions (1015: stock succeeds, the harness fails; 1017: the
  reverse). At 10 they agree on all ten shared seeds. Why 1015 and 1017 differ was not
  investigated; the two loops need not draw the same random numbers for the policy.
- **The 70% vs 54% is the seed range, not the loop.** The harness's own first 20 seeds are 14/20
  too; seeds 1020–1049 bring its 50-seed rate down to 27/50 (13/30 on those; 14/20 vs 13/30 is
  within sampling noise, Fisher p = 0.09).
- **At 10, every failure the harness had at 50 on these seeds becomes a success.** Seeds 1003,
  1005, 1009, 1013, 1015 and 1019 fail at 50 and succeed at 10. One seed goes the other way:
  1016 succeeds at 50 and times out at 10 (280 steps). The paired test on 20 seeds is not
  significant (p = 0.12).
- **A successful episode at 10 takes 130–167 steps** (median 140), against 280 for a failure.
- **Wall-clock, as measured here, is not a benchmark.** A harness episode at 10 took 581–3209 s
  on four CPUs of a shared host, and ~590 s once the host was quiet. The policy runs a forward
  pass every 10 steps instead of every 50, so an episode at 10 makes about 5× the policy calls.

## Conclusion

The fixture's 54% is what this checkpoint does with its shipped config, not something the harness
adds: stock `lerobot-eval`, sharing nothing with the harness but the installed packages and the
meshes, scores the same on the same seeds (within what 20 and 10 pairs can detect), at the
shipped setting and at 10. Most of the gap to the published numbers is the replan horizon: on task
0 at 10 the fixture matches them. Replanning every 10 actions moves the harness from 27/50 (54%)
to 19/20 (95%), Fisher exact p < 0.001; on the same 20 seeds, from 14/20 to 19/20 (6 gained,
1 lost; paired p = 0.12). That is in line with the paper's ablation (70% → 94% on LIBERO-Object)
and the third-party CPU number for this checkpoint (task 0: 10/10 at 10). The fix is PR #7. Its
consequences for D1, D9a, D10 and D14 are listed there.

Twenty and ten episodes are small samples. The 95% intervals at 10 start at 0.75 and 0.69, so
they cannot tell 90% from 99%.

## Not tested

- **GPU numerics and a GPU renderer.** Everything here ran on CPU with Mesa's llvmpipe. The
  third-party CPU, CUDA and MPS numbers in #4614 agree with each other at `n_action_steps=10`,
  which suggests the hardware is not the gap.
- **Other libero_object tasks.** Not re-run here; #4614's per-task table covers tasks 0–9, and
  task 0 is among the easiest there (10/10 at 10).
- **`n_action_steps=1`**, the paper's headline setting. Ten times the policy calls of 10.
  The paper and a CUDA sweep in #4614 put it within a few points of 10.
- **256×256 rendering**, to match the training images. LeRobot's default is 360.
- **The checkpoint's current Hub revision** (`a24f1f3`, 2026-10-12). It only rewrites the
  feature metadata; its PR reports bitwise-identical preprocessing and actions.
- **Lane E's perturbations at the new setting.** Every number in `SPIKE-envelope.md` was measured
  at 50, and its perturbation table was not re-run here. The D10 thresholds that rest on it (the
  ±1.1 arm bound and the frame band) are re-checked at 10 separately, against PR #7.

## Reproduce

These are Linux paths; on Windows, use absolute paths in `-v`. The stock runs need the network.

```bash
docker build -t fathomline .
mkdir -p runs/baseline
# Stock lerobot-eval, shipped setting, then n_action_steps=10.
docker run --rm --cpuset-cpus 0-3 --memory 7g \
  -v "$PWD/docs/spike-baseline:/spike:ro" -v "$PWD/runs/baseline:/out" \
  --entrypoint bash fathomline /spike/run_stock_eval.sh /out/stock-n50 \
  --eval.batch_size=1 --eval.use_async_envs=false --eval.n_episodes=20
docker run --rm --cpuset-cpus 4-7 --memory 7g \
  -v "$PWD/docs/spike-baseline:/spike:ro" -v "$PWD/runs/baseline:/out" \
  --entrypoint bash fathomline /spike/run_stock_eval.sh /out/stock-n10 \
  --eval.batch_size=1 --eval.use_async_envs=false --eval.n_episodes=10 --policy.n_action_steps=10
# The harness at 10: lane E's baseline condition, with PR #7's source over the image's,
# as four containers of five seeds (run them two at a time on a small host).
git fetch origin pull/7/head && git worktree add ../fix-n10 dd0cf1e
for part in a:1000 b:1005 c:1010 d:1015; do
  docker run --rm --network none --memory 7g \
    -v "$PWD/docs/spike-envelope:/spike:ro" -v "$PWD/../fix-n10/src:/app/src:ro" \
    -v "$PWD/runs/baseline:/out" --entrypoint python fathomline \
    /spike/run_envelope.py --condition baseline --first-seed "${part#*:}" --episodes 5 \
    --out "/out/harness-n10-${part%%:*}"
done
uv run python docs/spike-baseline/summarize.py \
  harness-n50=docs/spike-envelope/results/baseline.jsonl \
  stock-n50=runs/baseline/stock-n50/eval_info.json \
  harness-n10=runs/baseline/harness-n10-a/baseline.jsonl \
  harness-n10=runs/baseline/harness-n10-b/baseline.jsonl \
  harness-n10=runs/baseline/harness-n10-c/baseline.jsonl \
  harness-n10=runs/baseline/harness-n10-d/baseline.jsonl \
  stock-n10=runs/baseline/stock-n10/eval_info.json
```

The image is built from main before PR #7. Once #7 merges, main's image already replans every 10
actions, so the `harness-n10` runs need no mounted source, and `harness-n50` needs the image built
from the commit before the merge. The committed [`results/`](spike-baseline/results/) hold the
four `harness-n10` files concatenated (`harness-n10.jsonl`) and each stock run's
`eval_info.json`; the videos are not committed.
