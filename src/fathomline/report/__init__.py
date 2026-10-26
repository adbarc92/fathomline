"""Lane D: render a persisted run as a Markdown report, with no re-run.

`fathomline report <run-dir>` reads `result.json` (result schema v1) and, when
present, `gate.json` beside it, then prints the report. Ruling D12: the JSON is
canonical and the Markdown is rendered from it.
"""

from __future__ import annotations

from pathlib import Path

from fathomline.report.load import (
    GATE_FILENAME,
    GATE_SCHEMA_VERSION,
    RESULT_FILENAME,
    GateVerdict,
    ReportError,
    load_gate,
    load_run,
)
from fathomline.report.markdown import render_markdown

__all__ = [
    "GATE_FILENAME",
    "GATE_SCHEMA_VERSION",
    "RESULT_FILENAME",
    "GateVerdict",
    "ReportError",
    "load_gate",
    "load_run",
    "render_markdown",
    "report_run_dir",
]


def report_run_dir(run_dir: Path) -> str:
    """The report for a run directory, read from disk and nothing else."""
    return render_markdown(load_run(run_dir), gate=load_gate(run_dir))
