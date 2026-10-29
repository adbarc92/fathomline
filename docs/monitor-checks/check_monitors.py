"""Lane G: run the D1 fixture's declared monitors on real episodes, clean and with a fault injected.

Runs inside an image built from this branch, with this directory, the envelope spike's
directory and an output directory mounted:

    docker run --rm --network none --cpuset-cpus 0-3 --memory 12g \
      -v "$PWD/docs/monitor-checks:/check:ro" -v "$PWD/docs/spike-envelope:/spike:ro" \
      -v "$OUT:/out" --entrypoint python fathomline /check/check_monitors.py \
      --condition unnorm_identity --first-seed 1000 --episodes 30 --out /out

Each episode runs through the core's `run_episode` with `fixture_monitors(...)`, the suite the
envelope declares, given the adapter's hooks. A monitor that lands in the suite is checked here;
a monitor that needs a new hook needs it passed in below.
A fault is injected by the envelope spike's own `install` (`docs/spike-envelope/run_envelope.py`),
unchanged, so a condition here is the same fault the spike measured. On seeds both ran, the
action hashes should equal the spike's when run on the same CPU (the hash is known to differ
across CPUs, D9a); each row records the CPU model.

One condition is this check's own, because the spike injected none like it:
`rename_swap` exchanges the two cameras in the loaded policy's rename map, a camera mix-up
in configuration rather than in the pixel stream (the spike's `camera_swap`).

One JSON line per episode goes to `<out>/<condition>.jsonl`; a seed already in that file is
refused rather than appended twice. `--summarize DIR` prints a table per condition from those
files, on any machine, and fails on a duplicated seed.
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

    The statistics are meant to be the spike's (`frame_stats` in `run_envelope.py`: the
    channel-mean gray level and the mean absolute difference between horizontal neighbours),
    read here from the observation the policy is handed and scaled back to 0-255. The spike read
    the raw frame; `--summarize` compares frame 0 with the spike's on the same init state.
    Recorded for the observation-shift band, which is calibrated on held-out clean episodes.
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


#: Conditions this check injects itself; every other name is the spike's.
OWN_CONDITIONS = ("rename_swap",)


def _swap_cameras_in_rename_map(policy: Any) -> None:
    for step in policy._preprocessor.steps:
        if type(step).__name__ == "RenameObservationsProcessorStep":
            image, image2 = "observation.images.image", "observation.images.image2"
            mapping = dict(step.rename_map)
            mapping[image], mapping[image2] = mapping[image2], mapping[image]
            step.rename_map = mapping
            return
    raise RuntimeError("no RenameObservationsProcessorStep in the policy preprocessor")


def _cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def run(condition: str, seeds: range, out: Path, spike: Path, step_limit: int | None) -> None:
    sys.path.insert(0, str(spike))
    from run_envelope import CONDITIONS, install  # the spike's fault injection, unchanged

    if condition not in (*CONDITIONS, *OWN_CONDITIONS):
        # `install` does nothing for an unknown name: a typo would run clean under a fault's label.
        raise SystemExit(
            f"unknown condition {condition!r}; known: {', '.join((*CONDITIONS, *OWN_CONDITIONS))}"
        )
    sink = out / f"{condition}.jsonl"
    done = {row["seed"] for row in _rows(sink)} & set(seeds)
    if done:
        raise SystemExit(f"{sink} already has seeds {sorted(done)}; refusing to append them twice")

    from fathomline.adapters.lerobot.adapter import load_fixture
    from fathomline.core import run_episode
    from fathomline.monitors import LIBERO_ARM_DIMS, fixture_monitors

    env, policy, max_steps = load_fixture()
    if step_limit is not None:
        # Fault runs only: a flag is a first-steps fact, and a failing episode runs all 280.
        max_steps = min(max_steps, step_limit)
    frames: list[Any] = []
    noise: list[Any] = [None]
    install(condition, env, policy, frames, noise)  # a no-op for this check's own conditions
    if condition == "rename_swap":
        _swap_cameras_in_rename_map(policy)

    def loaded_pipeline() -> dict[str, str]:
        return {**env.loaded_pipeline(), **policy.loaded_pipeline()}

    cpu = _cpu_model()

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
            monitors=(fixture_monitors(loaded_pipeline=loaded_pipeline), peak, frame_stats),
        )
        flags = result.flags
        row = {
            "condition": condition,
            "seed": seed,
            "success": result.success,
            "steps": result.steps,
            "max_steps": max_steps,
            "action_hash": result.action_hash,
            "flagged": bool(flags),
            "flags": len(flags),
            "flag_monitors": sorted({flag.monitor for flag in flags}),
            "first_flag_step": flags[0].step if flags else None,
            "first_flag_reason": flags[0].reason if flags else None,
            "first_flag_step_by_monitor": {
                monitor: min(flag.step for flag in flags if flag.monitor == monitor)
                for monitor in {flag.monitor for flag in flags}
            },
            "arm_peak": round(peak.peak, 4),
            "frames": frame_stats.summary(),
            "cpu": cpu,
            "episode_s": round(time.perf_counter() - start, 1),
        }
        with sink.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps({k: row[k] for k in ("condition", "seed", "flagged")}), flush=True)
    env.close()


def _ranges(seeds: list[int]) -> str:
    spans: list[str] = []
    start = previous = seeds[0]
    for seed in [*seeds[1:], None]:
        if seed is not None and seed == previous + 1:
            previous = seed
            continue
        spans.append(f"{start}-{previous}" if start != previous else str(start))
        if seed is not None:
            start = previous = seed
    return ", ".join(spans)


def summarize(directory: Path, spike_results: Path | None) -> int:
    """One table row per condition and kind of episode: a replay of a seed the spike also ran
    (its outcome was already in the spike's data), or a new episode. They are never merged."""
    print(
        "| Condition | Episodes | Seeds (n) | Flagged | First flag step (min / median / max) "
        "| Arm peak range |"
    )
    print("|---|---|---|---|---|---|")
    notes: list[str] = []
    for path in sorted(directory.glob("*.jsonl")):
        rows = _rows(path)
        seeds = sorted(row["seed"] for row in rows)
        if len(set(seeds)) != len(seeds):
            repeated = sorted({seed for seed in seeds if seeds.count(seed) > 1})
            print(f"{path}: seeds {repeated} appear more than once", file=sys.stderr)
            return 1
        reference = _rows(spike_results / path.name) if spike_results is not None else []
        known = {row["seed"]: row["action_hash"] for row in reference}
        paired = [row for row in rows if row["seed"] in known]
        new = [row for row in rows if row["seed"] not in known]
        new_kind = "new" if spike_results is not None else "not compared with the spike"
        for kind, group in (("replay of the spike's", paired), (new_kind, new)):
            if not group:
                continue
            group_seeds = sorted(row["seed"] for row in group)
            steps = [row["first_flag_step"] for row in group if row["flagged"]]
            step_text = (
                f"{min(steps)} / {statistics.median(steps):g} / {max(steps)}" if steps else "-"
            )
            peaks = [row["arm_peak"] for row in group]
            print(
                f"| `{path.stem}` | {kind} | {_ranges(group_seeds)} ({len(group)}) "
                f"| {len(steps)}/{len(group)} | {step_text} "
                f"| {min(peaks):.2f} to {max(peaks):.2f} |"
            )
        by_monitor: dict[str, list[int]] = {}
        # Rows written before per-monitor steps were recorded cannot say which monitor flagged.
        attributed = [row for row in rows if "first_flag_step_by_monitor" in row]
        for row in attributed:
            for monitor, step in row.get("first_flag_step_by_monitor", {}).items():
                by_monitor.setdefault(monitor, []).append(step)
        for monitor, firsts in sorted(by_monitor.items()):
            notes.append(
                f"`{path.stem}`: `{monitor}` flagged {len(firsts)}/{len(attributed)}, "
                f"first at step {min(firsts)} / {statistics.median(firsts):g} / {max(firsts)} "
                "(min / median / max)."
            )
        cpus = sorted({row.get("cpu", "not recorded") for row in rows})
        notes.append(f"`{path.stem}`: CPU {'; '.join(cpus)}.")
        # A run cut short hashes fewer actions than the spike's episode did; it cannot match.
        spike_steps = {row["seed"]: row["steps"] for row in reference}
        whole = [row for row in paired if row.get("max_steps", 10**9) >= spike_steps[row["seed"]]]
        if whole:
            same = sum(row["action_hash"] == known[row["seed"]] for row in whole)
            notes.append(
                f"`{path.stem}`: action hash equals the spike's on {same}/{len(whole)} "
                "seeds run by both at full length."
            )
        if len(whole) < len(paired):
            notes.append(
                f"`{path.stem}`: {len(paired) - len(whole)} replayed seeds ran cut short; "
                "their hashes are not compared."
            )
        by_init = {row["init_state"]: row for row in reference}
        framed = [row for row in rows if "frames" in row and row["seed"] % 50 in by_init]
        if framed:
            gaps = [
                max(
                    abs(row["frames"]["image"]["frame0_brightness"] - spike["frame0_brightness"]),
                    abs(row["frames"]["image"]["frame0_noise"] - spike["frame0_noise"]),
                )
                for row in framed
                for spike in [by_init[row["seed"] % 50]]
            ]
            notes.append(
                f"`{path.stem}`: frame-0 brightness and noise differ from the spike's on the same "
                f"init state by at most {max(gaps):.3f} ({len(framed)} episodes)."
            )
    for note in notes:
        print(f"\n{note}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--max-steps", type=int, help="cut each episode short (fault runs; never clean ones)"
    )
    parser.add_argument("--out", type=Path, default=Path("/out"))
    parser.add_argument("--spike", type=Path, default=Path("/spike"))
    parser.add_argument("--summarize", type=Path, help="print the tables for a results directory")
    parser.add_argument("--spike-results", type=Path, help="compare with the spike's JSONL here")
    args = parser.parse_args()

    if args.summarize:
        return summarize(args.summarize, args.spike_results)
    if not args.condition:
        parser.error("at least one --condition is required")
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = range(args.first_seed, args.first_seed + args.episodes)
    for condition in args.condition:  # a fresh fixture per condition, so no patch carries over
        run(condition, seeds, args.out, args.spike, args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
