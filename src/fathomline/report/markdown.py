"""Render a run as Markdown. Ruling D12: the JSON is canonical; this is a view of it.

The output is a pure function of its inputs, so the same persisted run always
regenerates the same bytes; the text this module adds is ASCII. A flagged run
is marked in the title and in a banner above every number, because a flagged
run is flagged, never silently reported.
"""

from __future__ import annotations

from collections.abc import Sequence

from fathomline.core.schema import SCHEMA_VERSION
from fathomline.core.types import EpisodeResult, RunResult
from fathomline.monitors import flagged_episodes
from fathomline.report.load import GateVerdict

#: Characters of the action hash shown per episode. The full hash is in the JSON.
HASH_PREFIX = 12


def render_markdown(result: RunResult, gate: GateVerdict | None = None) -> str:
    flagged = flagged_episodes(result)
    sections = [
        _title(result, flagged),
        f"Rendered from result schema v{SCHEMA_VERSION}. "
        "The JSON is canonical; numbers here are rounded for display.",
        _provenance(result),
        _summary(result, flagged),
        _gate(gate, result, flagged=bool(flagged)),
        _flags(result),
        _episodes(result),
    ]
    return "\n\n".join(sections) + "\n"


def _title(result: RunResult, flagged: Sequence[EpisodeResult]) -> str:
    if not flagged:
        return "# Fathomline run report"
    count = sum(len(episode.flags) for episode in flagged)
    noun = "flag" if count == 1 else "flags"
    return (
        "# Fathomline run report: FLAGGED\n\n"
        f"> **FLAGGED: {len(flagged)} of {len(result.episodes)} episodes raised {count} "
        f"monitor {noun}.** A flagged episode fell outside the declared\n"
        "> envelope, so this run is not a clean measurement. "
        'Every flag is listed under "Monitor flags".'
    )


def _provenance(result: RunResult) -> str:
    manifest = result.manifest
    seeds = ", ".join(str(seed) for seed in manifest.seeds) or "none"
    return "## Provenance\n\n" + _table(
        ("Field", "Value"),
        [
            ("Fathomline version", manifest.fathomline_version),
            ("Policy", f"{manifest.policy.name} @ {manifest.policy.revision}"),
            ("Env", f"{manifest.env.name} @ {manifest.env.revision}"),
            ("Seeds", seeds),
            ("Max steps", str(manifest.max_steps)),
            ("Python", manifest.python_version),
            ("Platform", manifest.platform),
        ],
    )


def _summary(result: RunResult, flagged: Sequence[EpisodeResult]) -> str:
    n = len(result.episodes)
    successes = sum(episode.success for episode in result.episodes)
    rate = f"{successes / n:.1%}" if n else "n/a"
    return "## Summary\n\n" + _table(
        ("Metric", "Value"),
        [
            ("Episodes", str(n)),
            ("Successes", str(successes)),
            ("Success rate", rate),
            ("Flagged episodes", str(len(flagged))),
        ],
    )


def _gate(gate: GateVerdict | None, result: RunResult, *, flagged: bool) -> str:
    if gate is None:
        return "## Gate\n\nNot evaluated: this run has no gate verdict."
    verdict = "PASS" if gate.passed else "FAIL"
    text = "## Gate\n\n" + _table(
        ("Field", "Value"),
        [
            ("Verdict", f"**{verdict}**"),
            ("Successes / n", f"{gate.successes} / {gate.n}"),
            ("Minimum n", str(gate.min_n)),
            ("Clopper-Pearson 95% lower bound", f"{gate.lower_bound:.4f}"),
            ("Threshold", f"{gate.threshold:.4f}"),
        ],
    )
    for contradiction in _contradictions(gate, result, flagged=flagged):
        text += f"\n\n> **Contradiction:** {contradiction}"
    return text


def _contradictions(gate: GateVerdict, result: RunResult, *, flagged: bool) -> list[str]:
    """Every way the verdict disagrees with the run it sits beside, or with ruling D6.

    The report does not judge the run; the gate does. But it never shows a verdict
    the data beside it contradicts without saying so.
    """
    n = len(result.episodes)
    successes = sum(episode.success for episode in result.episodes)
    found = []
    if (gate.successes, gate.n) != (successes, n):
        found.append(
            f"the gate verdict counts {gate.successes} / {gate.n} but this run has "
            f"{successes} / {n}. The verdict is not about this run;\n> treat it as invalid."
        )
    if gate.passed and flagged:
        found.append(
            "the gate verdict is PASS but this run is flagged. "
            "A flagged run never passes;\n> treat this verdict as invalid."
        )
    if gate.passed and gate.n < gate.min_n:
        found.append(
            f"the gate verdict is PASS but n = {gate.n} is below the minimum {gate.min_n}. "
            "A sample that small\n> fails by construction (D6); treat this verdict as invalid."
        )
    if gate.passed and gate.lower_bound < gate.threshold:
        found.append(
            f"the gate verdict is PASS but its lower bound {gate.lower_bound:.4f} is below the "
            f"threshold {gate.threshold:.4f}.\n> D6 passes only when the bound reaches the "
            "threshold; treat this verdict as invalid."
        )
    return found


def _flags(result: RunResult) -> str:
    rows = [
        (str(episode.seed), str(flag.step), flag.monitor, flag.reason)
        for episode in result.episodes
        for flag in episode.flags
    ]
    if not rows:
        return "## Monitor flags\n\nNone raised."
    return "## Monitor flags\n\n" + _table(("Episode seed", "Step", "Monitor", "Reason"), rows)


def _episodes(result: RunResult) -> str:
    if not result.episodes:
        return "## Episodes\n\nNone."
    rows = [
        (
            str(episode.seed),
            "yes" if episode.success else "no",
            str(episode.steps),
            f"{episode.sum_reward:.4f}",
            f"{episode.max_reward:.4f}",
            str(len(episode.flags)),
            f"`{episode.action_hash[:HASH_PREFIX]}`",
        )
        for episode in result.episodes
    ]
    header = ("Seed", "Success", "Steps", "Sum reward", "Max reward", "Flags", "Action hash")
    return "## Episodes\n\n" + _table(header, rows)


def _table(header: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> str:
    lines = [_row(header), "|" + "---|" * len(header)]
    lines.extend(_row(row) for row in rows)
    return "\n".join(lines)


def _row(cells: tuple[str, ...]) -> str:
    return "| " + " | ".join(_cell(cell) for cell in cells) + " |"


def _cell(text: str) -> str:
    """Keep document text inside its cell: a pipe or a newline would end the row."""
    return " ".join(text.replace("\\", "\\\\").replace("|", "\\|").splitlines())
