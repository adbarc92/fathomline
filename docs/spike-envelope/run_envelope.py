"""Lane E spike: run the D1 fixture under one perturbation; record what a monitor would see.

Runs inside the image, with this directory and an output directory mounted:

    docker run --rm --network none --cpuset-cpus 0-3 --memory 12g \
      -v "$PWD/docs/spike-envelope:/spike:ro" -v "$OUT:/out" \
      --entrypoint python fathomline /spike/run_envelope.py \
      --condition light_dim [--condition ...] --first-seed 1000 --episodes 50 --out /out [--videos]

One JSON line per episode goes to `/out/<condition>.jsonl`; `summarize.py` reads them. Seeds
1000..1049 map to `libero_object` init states 0..49 (`seed % 50`), so every condition sees the
same 50 starts and the comparison with the baseline is paired.

The loop is the core's own `run_episode` with the adapter's own env and policy. A condition
only patches the one seam it perturbs:

* the camera (pixels, before any processor): `noise`, `camera_swap`;
* the scene (MuJoCo lights, re-applied after every hard reset): `light_dim`, `light_bright`;
* the instruction string: `instruction_paraphrase`, `instruction_other`;
* the initial state (object positions after the benchmark init state): `init_shift`;
* the action metadata (the policy's unnormalizer stats): `unnorm_identity`.

This is spike code recorded as it ran, not product code: it reaches into adapter and LeRobot
privates on purpose, to put a perturbation exactly where a real fault would sit.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fathomline.adapters.lerobot.adapter import LeRobotEnv, LeRobotPolicy, load_fixture
from fathomline.core import run_episode
from fathomline.core.types import Flag, StepRecord

#: LIBERO's action space (lerobot/envs/libero.py: ACTION_LOW / ACTION_HIGH).
ACTION_LOW, ACTION_HIGH = -1.0, 1.0

INSTRUCTIONS = {
    # Same meaning, words the fine-tune never saw for this task.
    "instruction_paraphrase": "grab the alphabet soup can and put it into the basket",
    # libero_object task 1's instruction, on task 0's scene. Scored against task 0's goal.
    "instruction_other": "pick up the cream cheese and place it in the basket",
}

CONDITIONS = (
    "baseline",
    "light_dim",
    "light_bright",
    "noise",
    "camera_swap",
    "instruction_paraphrase",
    "instruction_other",
    "init_shift",
    "unnorm_identity",
)

LIGHT_SCALE = {"light_dim": 0.35, "light_bright": 2.5}
NOISE_SIGMA = 20.0  # uint8 units, ~8% of full scale
INIT_SHIFT_M = 0.05  # metres, uniform in [-s, s] on x and y, per movable object


class Recorder:
    """A `Monitor` that never flags: it keeps the per-step facts a monitor would read."""

    name = "spike-recorder"

    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        self._low, self._high = low, high
        self.actions: list[tuple[float, ...]] = []

    def on_step(self, record: StepRecord) -> Sequence[Flag]:
        self.actions.append(record.action)
        return ()

    def summary(self) -> dict[str, Any]:
        acts = np.asarray(self.actions, dtype=np.float64)
        env_out = (acts < ACTION_LOW) | (acts > ACTION_HIGH)
        data_out = (acts < self._low) | (acts > self._high)
        return {
            "steps_outside_env_bounds": int(env_out.any(axis=1).sum()),
            "steps_outside_train_range": int(data_out.any(axis=1).sum()),
            "steps_outside_env_bounds_per_dim": env_out.sum(axis=0).tolist(),
            "steps_outside_train_range_per_dim": data_out.sum(axis=0).tolist(),
            "max_abs_action": float(np.abs(acts).max()),
            "max_abs_action_per_dim": np.abs(acts).max(axis=0).round(4).tolist(),
            "first_env_excursion_step": int(np.argmax(env_out.any(axis=1)))
            if env_out.any()
            else None,
        }


def frame_stats(image: np.ndarray) -> tuple[float, float]:
    """Mean brightness and a high-frequency energy (mean |dx|) of one uint8 HxWx3 frame."""
    gray = image.astype(np.float32).mean(axis=2)
    return float(gray.mean()), float(np.abs(np.diff(gray, axis=1)).mean())


def _unnormalizer(policy: LeRobotPolicy) -> Any:
    for step in policy._postprocessor.steps:
        if type(step).__name__ == "UnnormalizerProcessorStep":
            return step
    raise RuntimeError("no UnnormalizerProcessorStep in the policy postprocessor")


def train_action_range(policy: LeRobotPolicy) -> tuple[np.ndarray, np.ndarray]:
    stats = _unnormalizer(policy).stats["action"]
    return np.asarray(stats["min"], dtype=np.float64), np.asarray(stats["max"], dtype=np.float64)


def apply_unnorm_identity(policy: LeRobotPolicy) -> None:
    """Action unnormalization as if the stats were lost: mean 0, std 1 (a skipped postprocessor)."""
    from lerobot.processor.converters import to_tensor

    step = _unnormalizer(policy)
    action = {k: np.asarray(v) for k, v in step.stats["action"].items()}
    action["mean"] = np.zeros_like(action["mean"])
    action["std"] = np.ones_like(action["std"])
    step.stats = {**step.stats, "action": action}
    step._tensor_stats = to_tensor(step.stats, device=step.device, dtype=step.dtype)


def _scene_hook(sub_env: Any, after_init: Callable[[Any, np.random.Generator], None]) -> None:
    """Run `after_init(inner_env, rng)` after every benchmark init state is set."""
    sub_env._ensure_env()
    inner = sub_env._env
    original = inner.set_init_state

    def set_init_state(state: Any) -> Any:
        original(state)
        after_init(inner, np.random.default_rng(int(sub_env.init_state_id)))
        inner.env.sim.forward()
        return inner.env._get_observations()

    inner.set_init_state = set_init_state


def scale_lights(scale: float) -> Callable[[Any, np.random.Generator], None]:
    # Scales the live model in place. That is right only because LIBERO's hard reset rebuilds
    # the model before every init state; the recorded brightness is constant across episodes.
    def apply(inner: Any, _rng: np.random.Generator) -> None:
        model = inner.env.sim.model
        model.light_diffuse[:] = np.clip(model.light_diffuse * scale, 0.0, 1.0)
        model.light_ambient[:] = np.clip(model.light_ambient * scale, 0.0, 1.0)
        model.light_specular[:] = np.clip(model.light_specular * scale, 0.0, 1.0)
        model.vis.headlight.diffuse[:] = np.clip(model.vis.headlight.diffuse * scale, 0.0, 1.0)
        model.vis.headlight.ambient[:] = np.clip(model.vis.headlight.ambient * scale, 0.0, 1.0)

    return apply


def shift_objects(inner: Any, rng: np.random.Generator) -> None:
    """Move every free-jointed object (not the robot) by up to INIT_SHIFT_M on x and y."""
    import mujoco

    sim = inner.env.sim
    raw = sim.model._model
    for joint in range(raw.njnt):
        if raw.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        address = raw.jnt_qposadr[joint]
        sim.data.qpos[address : address + 2] += rng.uniform(-INIT_SHIFT_M, INIT_SHIFT_M, 2)


def install(
    condition: str, env: LeRobotEnv, policy: LeRobotPolicy, frames: list[Any], noise: list[Any]
) -> None:
    """Patch the one seam `condition` perturbs, and record the frames the policy sees.

    `noise` holds one `np.random.Generator`, replaced per episode so the noise is a function of
    the seed and not of episode order.
    """
    sub_env = env._vec_env.envs[0]
    if condition in LIGHT_SCALE:
        _scene_hook(sub_env, scale_lights(LIGHT_SCALE[condition]))
    elif condition == "init_shift":
        _scene_hook(sub_env, shift_objects)
    elif condition in INSTRUCTIONS:
        env._task_description = INSTRUCTIONS[condition]
    elif condition == "unnorm_identity":
        apply_unnorm_identity(policy)

    original_observe = env._observe

    def observe(raw: Any) -> Any:
        pixels = raw["pixels"]
        if condition == "noise":
            for key in pixels:
                noisy = pixels[key].astype(np.float32) + noise[0].normal(
                    0.0, NOISE_SIGMA, pixels[key].shape
                )
                pixels[key] = np.clip(noisy, 0, 255).astype(np.uint8)
        elif condition == "camera_swap":
            pixels["image"], pixels["image2"] = pixels["image2"], pixels["image"]
        frames.append((pixels["image"][0].copy(), pixels["image2"][0].copy()))
        return original_observe(raw)

    env._observe = observe  # type: ignore[method-assign]


def write_video(path: Path, frames: list[Any]) -> None:
    from lerobot.utils.io_utils import write_video as _write

    # LIBERO renders upside down; LeRobot's processor flips H and W. Flip for people too.
    tiles = [np.concatenate([a[::-1, ::-1], b[::-1, ::-1]], axis=1) for a, b in frames]
    _write(str(path), np.stack(tiles), fps=20)


def run_condition(condition: str, seeds: range, out: Path, videos: bool) -> None:
    env, policy, max_steps = load_fixture()
    frames: list[Any] = []
    noise: list[Any] = [np.random.default_rng(0)]
    install(condition, env, policy, frames, noise)
    low, high = train_action_range(policy)
    sink = out / f"{condition}.jsonl"

    for seed in seeds:
        frames.clear()
        noise[0] = np.random.default_rng(seed)
        recorder = Recorder(low, high)
        start = time.perf_counter()
        result = run_episode(env, policy, seed=seed, max_steps=max_steps, monitors=(recorder,))
        elapsed = time.perf_counter() - start

        t0 = time.perf_counter()
        brightness, sharpness = frame_stats(frames[0][0])
        stat_us = (time.perf_counter() - t0) * 1e6
        row = {
            "condition": condition,
            "seed": seed,
            "init_state": seed % 50,
            "success": result.success,
            "steps": result.steps,
            "max_reward": result.max_reward,
            "action_hash": result.action_hash,
            "instruction": env._task_description,
            "frame0_brightness": round(brightness, 2),
            "frame0_noise": round(sharpness, 3),
            "frame_stat_us": round(stat_us, 1),
            "episode_s": round(elapsed, 1),
            **recorder.summary(),
        }
        with sink.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        brief = {k: row[k] for k in ("condition", "seed", "success", "steps")}
        print(json.dumps(brief), flush=True)
        if videos:
            outcome = "success" if result.success else "fail"
            write_video(out / f"{condition}-{seed}-{outcome}.mp4", frames)
    env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, action="append", required=True)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--videos", action="store_true", help="write an mp4 per episode")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    seeds = range(args.first_seed, args.first_seed + args.episodes)
    # A fresh fixture per condition, so no patch carries over into the next.
    for condition in args.condition:
        run_condition(condition, seeds, args.out, args.videos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
