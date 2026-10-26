"""Gates: pass or fail a run on evidence, not on its observed rate (ruling D6).

The statistic is the Clopper-Pearson exact lower bound on the success rate,
held against a declared threshold, plus a declared minimum number of episodes.
"""

from __future__ import annotations

from fathomline.gates.clopper_pearson import clopper_pearson_lower
from fathomline.gates.gate import GateConfig, GateVerdict, evaluate

__all__ = ["GateConfig", "GateVerdict", "clopper_pearson_lower", "evaluate"]
