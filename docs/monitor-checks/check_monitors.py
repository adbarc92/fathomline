"""Lane G: run the D1 fixture's declared monitors on real episodes, clean and with a fault injected.

Runs inside the image, with this directory, the envelope spike's directory and an output
directory mounted:

    docker run --rm --network none --cpuset-cpus 0-3 --memory 12g \
      -v "$PWD/docs/monitor-checks:/check:ro" -v "$PWD/docs/spike-envelope:/spike:ro" \
      -v "$OUT:/out" --entrypoint python fathomline /check/check_monitors.py \
      --condition unnorm_identity --first-seed 1000 --episodes 30 --out /out

Each episode runs through the core's `run_episode` with `fixture_monitors()`, the suite the
envelope declares, so every monitor that lands in it is checked here without editing this file.
A fault is injected by the envelope spike's own `install` (`docs/spike-envelope/run_envelope.py`),
unchanged, so a condition here is the same fault the spike measured: on the seeds both ran, the
action hashes must match the spike's.

One JSON line per episode goes to `/out/<condition>.jsonl`. `--summarize DIR` prints a table
per condition from those files, on any machine.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class _ArmPeak:
    """Never flags. Keeps the largest arm magnitude, to say how far inside or outside it was."""

    name = "check-arm-peak"

    def __init__(self, dims: Sequence[int]) -> None:
        self._dims = dims
        self.peak = 0.0

    def on_step(self, record: Any) -> Sequence[Any]:
        self.peak = max(self.peak, *(abs(record.action[dim]) for dim in self._dims))
        return ()


class _FrameStats:
    """Never flags. Per camera: frame-0 brightness and noise, and their range over every frame.

    The statistics are the spike's (`frame_stats` in `run_envelope.py`: the channel-mean gray
    level and the mean absolute difference between horizontal neighbours), read from the
    observation the policy is handed, scaled back to 0-255. Recorded for the observation-shift
    band, which is calibrated on held-out clean episodes.
    """

    name = "check-frame-stats"
    CAMERAS = ("observation.images.image", "observation.images.image2")

    def __init__(self) -> None:
        self.frames: dict[str, list[tuple[float, float]]] = {key: [] for key in self.CAMERAS}

    def on_step(self, record: Any) -> Sequence[Any]:
        for key in self.CAMERAS:
            gray = record.observation[key][0].float().mean(dim=0) * 255.0
            noise = (gray[:, 1:] - gray[:, :-1]).abs().mean()
            self.frames[key].append((float(gray.mean()), float(noise)))
        return ()

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, values in self.frames.items():
            camera = key.rsplit(".", 1)[-1]
            brightness = [b for b, _ in values]
            noise = [n for _, n in values]
            out[camera] = {
                "frame0_brightness": round(brightness[0], 3),
                "frame0_noise": round(noise[0], 4),
                "brightness_range": [round(min(brightness), 3), round(max(brightness), 3)],
                "noise_range": [round(min(noise), 4), round(max(noise), 4)],
            }
        return out


def run(condition: str, seeds: range, out: Path, spike: Path) -> None:
    sys.path.insert(0, str(spike))
    from run_envelope import install  # the spike's fault injection, unchanged

    from fathomline.adapters.lerobot.adapter import load_fixture
    from fathomline.core import run_episode
    from fathomline.monitors import LIBERO_ARM_DIMS, fixture_monitors

    env, policy, max_steps = load_fixture()
    frames: list[Any] = []
    noise: list[Any] = [None]
    install(condition, env, policy, frames, noise)
    sink = out / f"{condition}.jsonl"

    import numpy as np

    for seed in seeds:
        frames.clear()
        noise[0] = np.random.default_rng(seed)
        peak, frame_stats = _ArmPeak(LIBERO_ARM_DIMS), _FrameStats()
        start = time.perf_counter()
        result = run_episode(
            env,
            policy,
            seed=seed,
            max_steps=max_steps,
            monitors=(fixture_monitors(), peak, frame_stats),
        )
        flags = result.flags
        row = {
            "condition": condition,
            "seed": seed,
            "success": result.success,
            "steps": result.steps,
            "action_hash": result.action_hash,
            "flagged": bool(flags),
            "flags": len(flags),
            "flag_monitors": sorted({flag.monitor for flag in flags}),
            "first_flag_step": flags[0].step if flags else None,
            "first_flag_reason": flags[0].reason if flags else None,
            "arm_peak": round(peak.peak, 4),
            "frames": frame_stats.summary(),
            "episode_s": round(time.perf_counter() - start, 1),
        }
        with sink.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps({k: row[k] for k in ("condition", "seed", "flagged")}), flush=True)
    env.close()


def summarize(directory: Path, spike_results: Path | None) -> None:
    print("| Condition | Seeds | Flagged | First flag step (min / median / max) | Arm peak range |")
    print("|---|---|---|---|---|")
    notes: list[str] = []
    for path in sorted(directory.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        seeds = sorted(row["seed"] for row in rows)
        steps = [row["first_flag_step"] for row in rows if row["flagged"]]
        step_text = f"{min(steps)} / {statistics.median(steps):g} / {max(steps)}" if steps else "-"
        peaks = [row["arm_peak"] for row in rows]
        print(
            f"| `{path.stem}` | {seeds[0]}-{seeds[-1]} | {len(steps)}/{len(rows)} | {step_text} "
            f"| {min(peaks):.2f} to {max(peaks):.2f} |"
        )
        if spike_results is not None and (spike_results / path.name).exists():
            known = {
                row["seed"]: row["action_hash"]
                for row in map(json.loads, (spike_results / path.name).read_text().splitlines())
            }
            paired = [row for row in rows if row["seed"] in known]
            same = sum(row["action_hash"] == known[row["seed"]] for row in paired)
            notes.append(f"`{path.stem}`: action hash equals the spike's on {same}/{len(paired)}")
    for note in notes:
        print(f"\n{note} seeds run by both.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("/out"))
    parser.add_argument("--spike", type=Path, default=Path("/spike"))
    parser.add_argument("--summarize", type=Path, help="print the tables for a results directory")
    parser.add_argument("--spike-results", type=Path, help="compare action hashes with these")
    args = parser.parse_args()

    if args.summarize:
        summarize(args.summarize, args.spike_results)
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = range(args.first_seed, args.first_seed + args.episodes)
    for condition in args.condition:  # a fresh fixture per condition, so no patch carries over
        run(condition, seeds, args.out, args.spike)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
