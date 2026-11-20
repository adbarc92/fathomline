"""Baseline spike: success rates, intervals and paired shifts for the runs in SPIKE-baseline.md.

Runs on any host, against the repo's own gate code (no LeRobot needed):

    uv run python docs/spike-baseline/summarize.py \
        harness=docs/spike-envelope/results/baseline.jsonl \
        stock=docs/spike-baseline/results/stock-n50.json ...

Each argument is `label=path`. A `.jsonl` path is per-episode harness output (`run_envelope.py`);
a `.json` path is a stock `lerobot-eval` `eval_info.json`. In the latter, episode `i` of a task
ran from seed `1000 + i` and init state `i` (see `run_stock_eval.sh`). Repeat a label to merge a
run that was split across containers.

The interval is the two-sided 95% Clopper-Pearson interval (D6a). The first label is the
reference: every later run on the same task is paired with it on seed and tested with a two-sided
exact McNemar test.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from fathomline.gates import clopper_pearson_lower

FIRST_SEED = 1000

#: (label, task id) -> {seed: success}
Runs = dict[tuple[str, int], dict[int, bool]]


def interval(successes: int, episodes: int) -> tuple[float, float]:
    """Two-sided 95% Clopper-Pearson interval; the upper end by symmetry."""
    low = clopper_pearson_lower(successes, episodes)
    high = 1.0 - clopper_pearson_lower(episodes - successes, episodes)
    return low, high


def mcnemar_exact(lost: int, gained: int) -> float:
    n = lost + gained
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(lost, gained) + 1)) / (1 << n)
    return min(1.0, 2 * tail)


def seed_ranges(seeds: list[int]) -> str:
    """`[1000, 1001, 1002, 1010]` -> `1000-1002, 1010`."""
    spans: list[list[int]] = []
    for seed in seeds:
        if spans and seed == spans[-1][1] + 1:
            spans[-1][1] = seed
        else:
            spans.append([seed, seed])
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in spans)


def load(label: str, path: Path, runs: Runs) -> None:
    """Add one run. A label given twice (a run split across containers) is merged."""
    if path.suffix == ".jsonl":
        episodes = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        merged = runs.setdefault((label, 0), {})
        for row in episodes:
            if row["seed"] in merged:
                raise ValueError(f"seed {row['seed']} appears twice in {label}")
            merged[row["seed"]] = bool(row["success"])
        return
    info = json.loads(path.read_text())
    for task in info["per_task"]:
        successes = task["metrics"]["successes"]
        runs[(label, task["task_id"])] = {
            FIRST_SEED + i: bool(success) for i, success in enumerate(successes)
        }


def main(argv: list[str]) -> int:
    runs: Runs = {}
    labels: list[str] = []
    for argument in argv:
        label, _, path = argument.partition("=")
        if label not in labels:
            labels.append(label)
        load(label, Path(path), runs)
    reference = labels[0]

    print(
        "| Run | Task | Seeds | Successes | Rate | 95% CP interval "
        "| First run, same seeds | Lost / gained vs first run | McNemar p |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for (label, task), episodes in runs.items():
        k, n = sum(episodes.values()), len(episodes)
        low, high = interval(k, n)
        seeds = seed_ranges(sorted(episodes))
        base = runs.get((reference, task))
        if label == reference or base is None:
            same = shift = p = "—"
        else:
            common = episodes.keys() & base.keys()
            same = f"{sum(base[s] for s in common)}/{len(common)}"
            lost = sum(1 for s in common if base[s] and not episodes[s])
            gained = sum(1 for s in common if episodes[s] and not base[s])
            shift = f"{lost} / {gained}"
            p = f"{mcnemar_exact(lost, gained):.2g}"
        print(
            f"| {label} | {task} | {seeds} | {k}/{n} | {k / n:.0%} | [{low:.2f}, {high:.2f}] "
            f"| {same} | {shift} | {p} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
