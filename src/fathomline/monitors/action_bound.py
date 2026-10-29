"""Action out of bounds (ruling D10, condition 2): a fixed bound on declared dimensions.

The monitor reads only the action the policy returned, so it costs a handful of
comparisons per step. It is a backstop for gross faults that reach the actions,
such as lost unnormalization statistics. It is not a shift detector: in the
envelope spike every other fault left the actions looking normal.

The bound and the dimensions are declared, never fitted. Which dimensions and
which value hold for the D1 fixture is `fathomline.monitors.envelope`'s to say.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from fathomline.core.types import Flag, StepRecord


class ActionBound:
    """Flags any step on which a declared action dimension leaves +/-`limit`.

    A value exactly at the limit is inside. A NaN or an infinity is outside: a
    NaN compares false against every bound, so "is it above the limit?" alone
    would pass it. One flag per step, naming every dimension out of bounds with its exact
    value, so a value just past the limit never reads as inside it.
    """

    def __init__(self, *, limit: float, dims: Sequence[int], name: str = "action-bound") -> None:
        if not (math.isfinite(limit) and limit > 0.0):
            raise ValueError(f"limit must be a positive finite number, got {limit}")
        if not dims:
            # A monitor that checks nothing never flags, which reads like a clean run.
            raise ValueError("an ActionBound needs at least one dimension")
        negative = [dim for dim in dims if dim < 0]
        if negative:
            # Python would read -1 as "the last dimension", silently.
            raise ValueError(f"dimensions must be non-negative, got {negative}")
        repeated = sorted({dim for dim in dims if list(dims).count(dim) > 1})
        if repeated:
            raise ValueError(f"dimensions must be unique; repeated: {repeated}")
        self._limit = limit
        self._dims = tuple(dims)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def on_step(self, record: StepRecord) -> Sequence[Flag]:
        action = record.action
        highest = max(self._dims)
        if len(action) <= highest:
            # Skipping what is missing would report "inside" for an unchecked step.
            raise ValueError(
                f"monitor {self._name!r} checks action dimension {highest}, "
                f"but step {record.index} has an action of length {len(action)}"
            )
        outside = [(dim, action[dim]) for dim in self._dims if not abs(action[dim]) <= self._limit]
        if not outside:
            return []
        values = ", ".join(f"dim {dim} = {value!r}" for dim, value in outside)
        return [
            Flag(
                monitor=self._name,
                step=record.index,
                reason=f"action outside +/-{self._limit:g}: {values}",
            )
        ]
