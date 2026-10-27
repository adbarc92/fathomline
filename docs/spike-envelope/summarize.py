"""Lane E spike: turn per-episode JSONL from `run_envelope.py` into the primer's tables.

Runs on any host, against the repo's own gate code (no LeRobot needed):

    uv run python docs/spike-envelope/summarize.py docs/spike-envelope/results

The interval is the two-sided 95% Clopper-Pearson interval (D6a). Its lower end
is `fathomline.gates.clopper_pearson_lower`; its upper end is the same function
by symmetry, upper(k, n) = 1 - lower(n - k, n). The shift against the baseline
is tested with a two-sided Fisher exact test, computed from exact binomials.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from fathomline.gates import clopper_pearson_lower

BASELINE = "baseline"

#: LIBERO's action is 6 arm deltas then the gripper command (index 6).
GRIPPER = 6

#: A candidate arm bound: LIBERO's ±1 plus a 10% margin, since a clean run touches 1.01.
ARM_BOUND = 1.1


def arm_excursion(row: dict[str, Any], key: str) -> bool:
    """Whether any arm dimension (not the gripper) left the bound on any step."""
    return any(count > 0 for count in row[key][:GRIPPER])


def interval(successes: int, episodes: int) -> tuple[float, float]:
    """Two-sided 95% Clopper-Pearson interval."""
    low = clopper_pearson_lower(successes, episodes)
    high = 1.0 - clopper_pearson_lower(episodes - successes, episodes)
    return low, high


def mcnemar_exact(lost: int, gained: int) -> float:
    """Two-sided exact McNemar p from the discordant pairs (baseline-only vs condition-only)."""
    n = lost + gained
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(lost, gained) + 1)) / (1 << n)
    return min(1.0, 2 * tail)


def paired(base: list[dict[str, Any]], group: list[dict[str, Any]]) -> tuple[int, int]:
    """(successes lost, successes gained) against the baseline, matched on seed."""
    by_seed = {r["seed"]: r["success"] for r in base}
    lost = sum(1 for r in group if by_seed.get(r["seed"]) and not r["success"])
    gained = sum(
        1 for r in group if r["seed"] in by_seed and not by_seed[r["seed"]] and r["success"]
    )
    return lost, gained


def load(results: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(results.rglob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["condition"]].append(row)
    return rows


def main(argv: list[str]) -> int:
    rows = load(Path(argv[1]))
    if BASELINE not in rows:
        print("no baseline rows", file=sys.stderr)
        return 1
    base = rows[BASELINE]
    base_k, base_n = sum(r["success"] for r in base), len(base)

    print(
        "| Condition | Successes | Rate | 95% CP interval | Shift vs baseline "
        "| Lost / gained (paired) | McNemar p |"
    )
    print("|---|---|---|---|---|---|---|")
    for name, group in rows.items():
        k, n = sum(r["success"] for r in group), len(group)
        low, high = interval(k, n)
        if name == BASELINE:
            shift = pairs = p = "—"
        else:
            lost, gained = paired(base, group)
            shift = f"{k / n - base_k / base_n:+.0%}"
            pairs = f"{lost} / {gained}"
            p = f"{mcnemar_exact(lost, gained):.2g}"
        print(
            f"| `{name}` | {k}/{n} | {k / n:.0%} | [{low:.2f}, {high:.2f}] | {shift} "
            f"| {pairs} | {p} |"
        )

    print()
    print(
        "| Condition | Gripper beyond ±1: episodes (steps) | Arm beyond train range "
        "| Arm beyond ±1 | Arm beyond ±1.1 | Arm peak, per-episode range | Gripper peak |"
    )
    print("|---|---|---|---|---|---|---|")
    for name, group in rows.items():
        n, steps = len(group), sum(r["steps"] for r in group)
        grip_steps = sum(r["steps_outside_env_bounds_per_dim"][GRIPPER] for r in group)
        grip_eps = sum(r["steps_outside_env_bounds_per_dim"][GRIPPER] > 0 for r in group)
        arm_env = sum(arm_excursion(r, "steps_outside_env_bounds_per_dim") for r in group)
        arm_data = sum(arm_excursion(r, "steps_outside_train_range_per_dim") for r in group)
        arm_peaks = [max(r["max_abs_action_per_dim"][:GRIPPER]) for r in group]
        arm_wide = sum(peak > ARM_BOUND for peak in arm_peaks)
        grip_peak = max(r["max_abs_action_per_dim"][GRIPPER] for r in group)
        print(
            f"| `{name}` | {grip_eps}/{n} ({grip_steps / steps:.0%}) | {arm_data}/{n} "
            f"| {arm_env}/{n} | {arm_wide}/{n} | {min(arm_peaks):.2f} to {max(arm_peaks):.2f} "
            f"| {grip_peak:.2f} |"
        )

    # A frame-statistics monitor: flag an episode whose first frame's brightness or
    # high-frequency energy falls outside the baseline's observed band, widened by 10%.
    # Calibrated on the same baseline episodes it is scored on, so its false-positive
    # rate here is optimistic by construction.
    bands = {
        key: (min(r[key] for r in base) * 0.9, max(r[key] for r in base) * 1.1)
        for key in ("frame0_brightness", "frame0_noise")
    }
    print()
    print("| Condition | Frame-0 brightness | Frame-0 noise | Flagged by the frame band |")
    print("|---|---|---|---|")
    for name, group in rows.items():
        flagged = sum(
            any(not low <= r[key] <= high for key, (low, high) in bands.items()) for r in group
        )
        bright = [r["frame0_brightness"] for r in group]
        noise = [r["frame0_noise"] for r in group]
        print(
            f"| `{name}` | {min(bright):.0f} to {max(bright):.0f} "
            f"| {min(noise):.1f} to {max(noise):.1f} | {flagged}/{len(group)} |"
        )
    cost = [r["frame_stat_us"] for group in rows.values() for r in group]
    episode_s = [r["episode_s"] / r["steps"] for group in rows.values() for r in group]
    print()
    print(
        f"Frame statistics: median {sorted(cost)[len(cost) // 2]:.0f} us per frame; "
        f"median step time {sorted(episode_s)[len(episode_s) // 2] * 1000:.0f} ms."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
