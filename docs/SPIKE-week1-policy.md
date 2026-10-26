# Spike — week-1 policy + task @ 2026-09-29

## TL;DR

Every candidate ran headless on CPU inside a container. Every one was **same-seed deterministic** on
its metrics. The audit's premise, a 2D PushT task for CPU reasons, **did not hold up**: the
VLA-class policy (SmolVLA, 450M params) is *faster* per episode on CPU than `diffusion_pusht`.
SmolVLA needs one forward pass per 50-action chunk with 10 flow steps; Diffusion needs one per 8
actions with 100 DDPM steps. The real constraints are elsewhere:

1. **LeRobot >= 0.5.0 requires Python >= 3.12.** The last 3.11-compatible release is **0.4.4**
   (2026-02-27). The repo pins `>=3.11` and `python:3.11-slim`.
2. **Image size**: 4.2–7.1 GB, against a 14 GB runner disk.
3. **Checkpoint/version coupling**: the same SmolVLA checkpoint, task and seed **succeeds on 0.4.4 and
   fails on 0.6.1**, because the env wrapper changed.

**Recommendation: SmolVLA + LIBERO (candidate C) on LeRobot 0.6.1 / Python 3.12.** It is the only
option that meets the plan's settled parameter without a recorded scope change. Two decisions
belong to Alex: the Python bump, and VLA-class vs PushT.

## Setup

- Host: Ryzen 7 5800X (8C/16T), Docker Desktop 28.3.3 (WSL2). Every image built `FROM python:3.x-slim`
  with CPU-only torch and weights baked in at build time. Every eval ran with `HF_HUB_OFFLINE=1`,
  so an eval needs no network.
- CI shape. `--cpuset-cpus` was used, **not** `--cpus`: `--cpus 4` is a CFS quota, so torch still
  sees 16 cores and spawned 8 threads (measured). A cpuset makes torch see what a runner has.
  - **Public repo** (ubuntu-latest): 4 vCPU / 16 GB / 14 GB SSD → `--cpuset-cpus 0-3 --memory 16g`.
  - **Private repo**: **2 vCPU / 8 GB** / 14 GB SSD → `--cpuset-cpus 0-1 --memory 8g`. This repo
    was private when the spike ran, so CI ran on this shape then. It has since gone public (D5a).
  - Source: [GitHub-hosted runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
- Command in every run:
  `lerobot-eval --eval.n_episodes=1 --eval.batch_size=1 --policy.device=cpu --policy.use_amp=false --seed=1000`.
- Timing is wall-clock, measured two ways: `/usr/bin/time -v` for the whole process, and
  LeRobot's `eval_s` for the rollout only. Peak RAM is `time -v` max RSS.
- Scratch Dockerfiles, logs and `eval_info.json` files live in the session scratchpad, not the repo.

## LeRobot facts (verified 2026-09-28/29)

| Fact | Value | Source |
|---|---|---|
| Latest release | **0.6.1** (2026-08-03) | [PyPI JSON](https://pypi.org/pypi/lerobot/json) |
| Python | 0.6.1, 0.6.0, 0.5.1, 0.5.0: **`>=3.12`**. 0.4.4 and earlier: `>=3.10` | `https://pypi.org/pypi/lerobot/<v>/json` |
| torch range | 0.6.1: `>=2.7,<2.12` (got 2.11.0+cpu). 0.4.4: `>=2.2.1,<2.11` (got 2.10.0+cpu) | PyPI `requires_dist` |
| Upstream torch default | LeRobot's own pyproject routes torch to **cu128** on Linux via `[tool.uv.sources]`. That applies to source checkouts only, not the PyPI wheel | [pyproject @ v0.6.1](https://github.com/huggingface/lerobot/blob/v0.6.1/pyproject.toml) |
| Env extras | `pusht` (gym-pusht 0.1.6, pymunk), `metaworld` (metaworld==3.0.0), `libero` (hf-libero 0.1.4, **`sys_platform == "linux"` only**), `aloha` (gym-aloha, MuJoCo) | PyPI `requires_dist` |
| Policy extras (0.6.1) | `smolvla` (transformers 5.4–5.6, num2words, accelerate), `diffusion` (**diffusers moved here in 0.6.x**), `evaluation` (av) | PyPI `requires_dist` |
| Eval entry point | `lerobot-eval` → `lerobot.scripts.lerobot_eval:main` | pyproject `[project.scripts]` |
| Eval flags | `--policy.path`, `--env.type`, `--env.task`, `--env.task_ids`, `--eval.n_episodes`, `--eval.batch_size`, `--policy.device`, `--policy.use_amp`, `--seed` (default 1000), `--rename_map`, `--output_dir` | [`configs/eval.py` @ v0.6.1](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/configs/eval.py) |
| Result artifact | `<output_dir>/eval_info.json`: per-episode `sum_rewards`, `max_rewards`, `successes`, plus `eval_s`. One mp4 per episode | observed |
| Legacy checkpoints | Checkpoints from before PR #1452 (e.g. `lerobot/diffusion_pusht`, 2024) have no `policy_preprocessor.json`. LeRobot >= 0.4 refuses them until they are migrated with `lerobot.processor.migrate_policy_normalization` | [backwardcomp.mdx](https://github.com/huggingface/lerobot/blob/v0.6.1/docs/source/backwardcomp.mdx) |
| SmolVLA sim fine-tunes on the Hub | `lerobot/smolvla_libero` (16.1k downloads), `smolvla_metaworld` (1.0k), `smolvla_robotwin`, `smolvla_robocasa`, `smolvla_vlabench`, `smolvla_libero_plus` … | [HF API, author=lerobot](https://huggingface.co/api/models?author=lerobot) |
| LIBERO / Meta-World eval docs | `MUJOCO_GL=egl` for headless use; recommended protocols: 10 episodes per task | [libero.mdx](https://github.com/huggingface/lerobot/blob/v0.6.1/docs/source/libero.mdx), [metaworld.mdx](https://github.com/huggingface/lerobot/blob/v0.6.1/docs/source/metaworld.mdx) |

## Candidates

| | Policy | Task | LeRobot / Python |
|---|---|---|---|
| **A1** | `lerobot/diffusion_pusht` @ `84a7c23`, migrated at build | PushT (gym-pusht, 2D pymunk) | 0.4.4 / 3.11.15 |
| **A2** | same | same | 0.6.1 / 3.12.14 |
| **B** | `lerobot/smolvla_metaworld` @ `cd6778d` (SmolVLA, 450,046,176 params) | Meta-World `push-v3` (MuJoCo) | 0.6.1 / 3.12.14 |
| **B311** | same as B | same | 0.4.4 / 3.11.15 |
| **C** | `lerobot/smolvla_libero` @ `31d453f` (SmolVLA, 450,046,176 params) | LIBERO `libero_object`, task 0 (robosuite/MuJoCo) | 0.6.1 / 3.12.14 |

No ACT/VQ-BeT candidate was tested. `act_aloha_sim_*` and `vqbet_pusht` are the same legacy
format as `diffusion_pusht` and not VLA-class, so they would add nothing to the decision.

## Results

Per-episode time is the rollout time (`eval_s`); process wall-clock is in brackets. Determinism
means two runs with seed 1000 produced bit-identical `sum_rewards` / `max_rewards` / `successes`.

| | A1 | A2 | B | B311 | C |
|---|---|---|---|---|---|
| Image size (`docker image ls`) | 4.79 GB | 4.15 GB | 5.16 GB | 5.84 GB | **7.06 GB** |
| Key versions | torch 2.10.0+cpu, gym-pusht 0.1.6, diffusers 0.35.2 | torch 2.11.0+cpu, gym-pusht 0.1.6 | torch 2.11.0+cpu, transformers 5.5.4, metaworld 3.0.0, mujoco 3.14.0 | torch 2.10.0+cpu, transformers 4.57.6, metaworld 3.0.0, mujoco 3.14.0 | torch 2.11.0+cpu, transformers 5.5.4, hf-libero 0.1.4, robosuite 1.4.0, mujoco 3.8.1 |
| Apt packages | `gcc libc6-dev linux-libc-dev` (0.4.4 core dep `pynput`→`evdev` builds from source) | none | `libegl1 libgl1 libegl-mesa0 libgl1-mesa-dri` | A1's + B's | B's + `gcc g++ cmake make` (added pre-emptively; **not tested whether required**) |
| Rendering | pygame surface, no GL | same | MuJoCo EGL → Mesa llvmpipe, `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` | same | same |
| Episode result, seed 1000 | success @ step 137/300, sum_reward 48.4437 | **identical to A1** | fail, 500/500 steps, max_reward 0.7317 | success @ step 61/500, max_reward 10.0 | success @ step 127/280 |
| 4 vCPU / 16 GB | 610 s [623] contended ⚠, 524 s [534] | 520 s [531], 517 s [525] | 147 s [159], 149 s [159] | 26 s [38], 26 s [40] | 69 s [82], 72 s [84] |
| **2 vCPU / 8 GB** (private-repo CI) | not run | **287 s** [295] | **226 s** [238] | not run | **99 s** [113] |
| 8 vCPU | not run | not run | 118 s [130] | not run | 62 s [75] |
| Peak RSS | 2.63 GB | 2.61 GB | 3.60 GB | 3.76 GB | 4.23 GB |
| Same-seed determinism | ✅ (metrics, and mp4 byte-identical) | ✅ (same) | ✅ (same) | ✅ metrics | ✅ (same) |
| Same metrics across 2/4/8 vCPU | — | ✅ 2 vs 4 | ✅ 2/4/8 | — | ✅ 2/4/8 |
| Seed changes the episode | not tested | not tested | ✅ seed 2000: max_reward 0.8864 | ⚠ see finding 4 | ✅ seed 2000: success @ step 123 |
| CI fit | ⚠ py3.11 works, but not VLA-class and on a frozen LeRobot | ⚠ not VLA-class; ~5 min per successful episode | ✅ VLA; ~4 min per full-length episode at 2 vCPU | ❌ py3.11, but the seed does not move the env and LeRobot is frozen | ✅ VLA; fastest; largest image |

⚠ A1 run 1 overlapped a concurrent image build; run 2 is the clean timing. An earlier A1 attempt
under `--cpus 4` (quota, 8 torch threads) managed 95 steps in 9:38 before it was aborted; that
number shows why the shape matters. It is not a result.

**Extra B data point**: one episode each on `reach-v3`, `button-press-topdown-v3` and `drawer-open-v3`
gave **2/3 successes**, in 196 s total at 4 vCPU. The policy works. `push-v3` is simply a task it
fails at seed 1000.

**GPU timings: not measured.** CPU is the constraint.

## Findings that shape week 1

1. **The Python pin is the first decision.** Any LeRobot still receiving releases means 3.12. Staying
   on 3.11 means pinning 0.4.4, which is frozen since February: no fixes, and older transformers and
   env wrappers.
2. **`diffusion_pusht` does not load as published on LeRobot >= 0.4.** It needs a build-time
   migration (`python -m lerobot.processor.migrate_policy_normalization --pretrained-path
   lerobot/diffusion_pusht --revision 84a7c23… --output-dir …`). On 0.6.1 that script also
   **crashes on an upstream bug**: `DiffusionConfig.crop_shape` arrives as a JSON list, the field is
   typed `tuple[int, int]`, and draccus fails to encode it on save. The workaround is a 6-line
   wrapper that turns the list into a tuple. The fixture is therefore *our derivative* of the Hub
   checkpoint, not the checkpoint itself. That weakens the claim "reproduces a published eval
   result".
3. **SmolVLA needs `--rename_map` on the CLI**, even though each checkpoint's
   `policy_preprocessor.json` already contains the same map. Feature validation runs before the
   preprocessor does. Without the flag, eval fails with `Feature mismatch … Missing features:
   observation.images.camera1…`.
   - B: `{"observation.image": "observation.images.camera1"}`
   - C: `{"observation.images.image": "…camera1", "observation.images.image2": "…camera2"}`
4. **Results are bound to the LeRobot version.** B and B311 use the same checkpoint, task and seed,
   yet one succeeds (max_reward 10.0) and the other fails (max_reward 0.73). The cause is in the env
   wrapper: 0.6.1 adds `env.seeded_rand_vec = True`, so `--seed` now controls object placement. In
   0.4.4, every Meta-World episode started from the same `MT1(seed=42)` layout. A week-2 test of
   "fails when the seed is removed" would be weaker on 0.4.4. **Pin LeRobot exactly (`==`)**, not
   with `>=`.
5. **PushT reproduces across versions.** A1 (0.4.4/torch 2.10) and A2 (0.6.1/torch 2.11) gave
   bit-identical rewards and the same step count.
6. **Videos are not a determinism oracle across machines.** Same machine, same shape: the mp4 bytes
   are identical. Across 2/4/8 vCPU the metrics are bit-identical, but decoded frames differ by
   mean-abs ~1.5 **from frame 0**, and the difference stays flat instead of growing. That points to
   encoder or llvmpipe threading, not trajectory divergence. The week-2 determinism test should
   assert on `eval_info.json` metrics (and on actions, if exposed), never on video.
7. **CPU scaling is not monotonic for Diffusion.** A2 took 287 s on 2 vCPU and 520 s on 4 vCPU.
   SmolVLA scales normally: 99 s → 69 s → 62 s for C. Do not extrapolate CI time from local numbers;
   measure it in CI.
8. **Episode length drives CI time.** PushT runs ~2–3.6 s per env step on CPU. A *failed* PushT
   episode runs all 300 steps, **estimated** at ~10–18 min (not measured). SmolVLA's worst case, a
   full 500-step Meta-World failure, measured 226 s at 2 vCPU.
9. **LIBERO downloads assets at first use** from `lerobot/libero-assets`, via hf-libero's
   `download_assets_from_huggingface()`, which **takes no revision**. The build must call it
   explicitly (as the snippet below does), or the eval hits the network. The asset revision is
   currently unpinned, which is a reproducibility gap. LIBERO is also **Linux-only**
   (`hf-libero; sys_platform == "linux"`), so a native Windows `uv sync` will not install it; eval
   would run only in the container.
10. **The SmolVLA constructor fetches `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`** (config and
    tokenizer), about 1.9 GB of HF cache in the image. It must be warmed at build time (`warm.py`,
    below).

## The decision for Alex

The settled parameter reads *"Smallest pretrained VLA-class policy that runs on local hardware"*
and *"a task the ecosystem already ships"*. The audit's CPU-in-CI concern is **met by the
VLA-class option**, so the tension is no longer "VLA vs CPU". It is now "VLA vs Python 3.11 /
image size".

| Option | What changes | Consequences |
|---|---|---|
| **1. SmolVLA + LIBERO (C), LeRobot 0.6.1, Python 3.12** | `requires-python >=3.12`; base `python:3.12-slim`; ~7 GB image | Meets the settled parameter as written. The most-used SmolVLA sim fine-tune, on a benchmark with a published protocol. Fastest per episode (99 s at 2 vCPU). Costs: largest image (headroom on a 14 GB runner is unverified), Linux-only env, unpinned asset revision, sparse 0/1 reward |
| **2. SmolVLA + Meta-World (B), 0.6.1, Python 3.12** | same Python bump; ~5.2 GB image | Meets the parameter. Cross-platform env, dense reward, 50 tasks. Weaker fixture: this checkpoint fails `push-v3` and passed 2/3 easy tasks. Slower worst case (226 s) |
| **3. diffusion_pusht + PushT (A1), 0.4.4, Python 3.11** | no Python change; `gcc` in the image | Keeps the pin, but **is not VLA-class**. Under plan §0 that is a scope change to record, not drift into. Ships a migrated derivative of the checkpoint on a frozen LeRobot |
| **4. diffusion_pusht + PushT (A2), 0.6.1, Python 3.12** | Python bump plus a migration workaround | Takes the costs of both: not VLA-class, *and* the Python bump |
| 5. SmolVLA + Meta-World on 0.4.4 / 3.11 (B311) | none | Runs, but the seed does not move the env (finding 4), and it is frozen. Not recommended |

Also Alex's call: whether the CI eval job runs on every PR at the 2-vCPU private shape (~2 min
per LIBERO episode, plus image build) or is gated until the repo goes public.

## Recommendation

**Option 1**, with option 2 as the fallback if the 7 GB image does not fit the runner.

- Pin `lerobot[smolvla,libero,evaluation]==0.6.1` exactly, and pin every Hub artifact by commit SHA.
- Make the week-1 test `eval --episodes 1` → exit 0, on `libero_object` task 0, seed 1000.
- Week 2 asserts on `eval_info.json` metrics, per finding 6.
- Before merging week 1: confirm in CI that the image builds within the runner's 14 GB. This is
  **not verified**; every build here ran locally.

### Proposed pins (not applied; untested with `uv lock` / `uv sync`)

```toml
requires-python = ">=3.12"
dependencies = ["lerobot[smolvla,libero,evaluation]==0.6.1"]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
torchvision = [{ index = "pytorch-cpu" }]
```

For option 3 instead: `requires-python = ">=3.11"` (unchanged),
`dependencies = ["lerobot[pusht]==0.4.4"]`, and the same CPU index block. The spike installed with
`uv pip install --system --torch-backend=cpu …` rather than `uv sync`. The `[tool.uv.sources]` form
above is the `uv sync` equivalent and has not been tried.

## Working snippets (each ran end to end, exit 0)

`warm.py` (B and C). This prefetches the checkpoint, the SmolVLM2 base and the processors, so eval
runs offline:

```python
import sys
from huggingface_hub import snapshot_download
repo, rev = sys.argv[1], sys.argv[2]
path = snapshot_download(repo, revision=rev, local_dir=f"/opt/policies/{repo.split('/')[1]}")
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
p = SmolVLAPolicy.from_pretrained(path)
from lerobot.policies.factory import make_pre_post_processors
make_pre_post_processors(p.config, pretrained_path=path,
    preprocessor_overrides={"device_processor": {"device": "cpu"}},
    postprocessor_overrides={"device_processor": {"device": "cpu"}})
```

The CPU overrides are needed: the checkpoints ship `device: cuda` in their processor configs.

**C — SmolVLA + LIBERO (0.6.1 / 3.12)**

```dockerfile
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
ENV PIP_NO_CACHE_DIR=1 UV_NO_CACHE=1 HF_HOME=/opt/hf MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
RUN apt-get update && apt-get install -y --no-install-recommends time libegl1 libgl1 libegl-mesa0 libgl1-mesa-dri gcc g++ cmake make && rm -rf /var/lib/apt/lists/*
RUN uv pip install --system --torch-backend=cpu "lerobot[smolvla,libero,evaluation]==0.6.1"
RUN echo N | python -c "from libero.libero import get_libero_path; print(get_libero_path('bddl_files'))"
RUN python -c "from libero.libero.utils.download_utils import download_assets_from_huggingface as d; print(d())"
COPY warm.py /opt/warm.py
RUN python /opt/warm.py lerobot/smolvla_libero 31d453f7edd78c839a8bbc39744a292686daf0de
```

```bash
docker run --rm --cpuset-cpus 0-1 --memory 8g -e HF_HUB_OFFLINE=1 spike-c \
  lerobot-eval --policy.path=/opt/policies/smolvla_libero --env.type=libero --env.task=libero_object \
  '--env.task_ids=[0]' --eval.n_episodes=1 --eval.batch_size=1 --policy.device=cpu --policy.use_amp=false \
  --seed=1000 --output_dir=/out/eval \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
```

**B — SmolVLA + Meta-World (0.6.1 / 3.12).** Same as C, with these changes:

- apt: `time libegl1 libgl1 libegl-mesa0 libgl1-mesa-dri` only.
- install: `"lerobot[smolvla,metaworld,evaluation]==0.6.1"`.
- no LIBERO lines.
- warm: `lerobot/smolvla_metaworld cd6778d2cfa724c1bf5fc637490548e54d81dc4c`.
- eval: `--env.type=metaworld --env.task=push-v3 '--rename_map={"observation.image": "observation.images.camera1"}'`.

B311 is the same as B, but on `python:3.11-slim`, adds `gcc libc6-dev linux-libc-dev`, and
installs `"lerobot[smolvla,metaworld]==0.4.4"`.

**A1 — diffusion_pusht + PushT (0.4.4 / 3.11)**

```dockerfile
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
ENV PIP_NO_CACHE_DIR=1 UV_NO_CACHE=1 HF_HOME=/opt/hf
RUN apt-get update && apt-get install -y --no-install-recommends time gcc libc6-dev linux-libc-dev && rm -rf /var/lib/apt/lists/*
RUN uv pip install --system --torch-backend=cpu "lerobot[pusht]==0.4.4"
RUN python -m lerobot.processor.migrate_policy_normalization --pretrained-path lerobot/diffusion_pusht \
    --revision 84a7c23178445c6bbf7e1a884ff497017910f653 --output-dir /opt/policies/diffusion_pusht
```

```bash
docker run --rm --cpuset-cpus 0-3 --memory 16g -e HF_HUB_OFFLINE=1 spike-a1 \
  lerobot-eval --policy.path=/opt/policies/diffusion_pusht --env.type=pusht --eval.n_episodes=1 \
  --eval.batch_size=1 --policy.device=cpu --policy.use_amp=false --seed=1000 --output_dir=/out/eval
```

The image also keeps the un-migrated checkpoint in the HF cache (about 1 GB). Deleting it after
migration should shrink the image; that has not been measured.

**A2 — diffusion_pusht + PushT (0.6.1 / 3.12).** Same as A1, with these changes:

- base `python:3.12-slim`, no `gcc` packages.
- install: `"lerobot[pusht,evaluation,diffusion]==0.6.1"`.
- migrate through this wrapper (upstream `crop_shape` bug, finding 2):

```python
import sys
import lerobot.processor.migrate_policy_normalization as m
_orig = m.make_policy_config
m.make_policy_config = lambda t, **kw: _orig(t, **{k: tuple(v) if isinstance(v, list) and k == "crop_shape" else v for k, v in kw.items()})
sys.argv = ["migrate"] + sys.argv[1:]
m.main()
```

## Not done

- No `uv lock` / `uv sync` with the proposed pins. No CI run: **disk headroom on a 14 GB runner is
  unverified.**
- No build-time measurement from a cold cache: builds reused cached layers.
- No GPU timing. No multi-episode success rates, beyond B's 3-task check.
- LIBERO: not isolated whether `--seed` changes the initial state or only the policy's sampling
  noise. With `init_states=true`, the initial states come from fixed benchmark files.
- Not tested: whether C's `gcc g++ cmake make` are required, and whether Meta-World runs natively
  on Windows.
