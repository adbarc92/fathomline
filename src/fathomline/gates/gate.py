"""Pass or fail a run against a declared gate (ruling D6).

A run passes only if every check passes:

* **minimum sample** — at least `min_episodes` episodes were run. Declared, not
  inferred, so a small sample fails by construction;
* **lower bound** — the Clopper-Pearson lower bound on the success rate is at
  least `threshold`. The observed rate is never compared directly: 5/5 is not
  evidence of a 100% policy;
* **no flags** — no episode carries a monitor flag. A flagged run is never
  passed (the ruling that Lane F builds on). This is the gate's whole side of
  that seam: Lane F produces `EpisodeResult.flags`, and the gate refuses them.

Every failed check is reported, not just the first, so one reading of the
verdict says everything that has to change.
"""

from __future__ import annotations

from dataclasses import dataclass

from fathomline.core.types import RunResult
from fathomline.gates.clopper_pearson import clopper_pearson_lower


@dataclass(frozen=True, slots=True)
class GateConfig:
    """What a run is judged against. Threshold and minimum n have no defaults."""

    threshold: float
    min_episodes: int
    confidence: float = 0.95

    def __post_init__(self) -> None:
        # 0 would be a gate the bound can never fail, and 1 one it can never pass.
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {self.threshold}")
        if self.min_episodes < 1:
            raise ValueError(f"min_episodes must be at least 1, got {self.min_episodes}")
        # D6 fixes the statistic at 95%. A weaker gate would be a D6 amendment, not a setting.
        if not 0.95 <= self.confidence < 1.0:
            raise ValueError(
                f"confidence must be at least 0.95 (ruling D6) and below 1, got {self.confidence}"
            )


@dataclass(frozen=True, slots=True)
class GateVerdict:
    config: GateConfig
    passed: bool
    successes: int
    episodes: int
    lower_bound: float
    flagged_episodes: int
    #: Why the run failed, one entry per failed check; empty when it passed.
    reasons: tuple[str, ...]


def evaluate(run: RunResult, config: GateConfig) -> GateVerdict:
    episodes = len(run.episodes)
    successes = sum(episode.success for episode in run.episodes)
    flagged = sum(1 for episode in run.episodes if episode.flags)
    lower_bound = clopper_pearson_lower(successes, episodes, confidence=config.confidence)

    reasons: list[str] = []
    if episodes < config.min_episodes:
        reasons.append(f"insufficient sample: {episodes} episodes, minimum {config.min_episodes}")
    if lower_bound < config.threshold:
        reasons.append(
            f"lower bound {lower_bound:.4f} is below the threshold {config.threshold} "
            f"({successes}/{episodes} at {config.confidence * 100:g}% confidence)"
        )
    if flagged:
        reasons.append(f"flagged: {flagged} episode(s) carry monitor flags")

    return GateVerdict(
        config=config,
        passed=not reasons,
        successes=successes,
        episodes=episodes,
        lower_bound=lower_bound,
        flagged_episodes=flagged,
        reasons=tuple(reasons),
    )
