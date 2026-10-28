"""Observation shift (ruling D10, condition 3): frame statistics held to declared bands.

A coarse detector by design. It catches what moves cheap image statistics, such
as lighting and sensor noise, for a fraction of a percent of a step. It misses
anything that leaves them alone: in the envelope spike, exchanging the two
cameras moved neither statistic out of the clean range, so a camera swap is not
caught here.

Which statistics, computed how, is the adapter's to say: it owns the image
tensors. This module only holds each value to its band.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

from fathomline.core.types import Flag, Observation, StepRecord


class FrameBand:
    """Flags any step whose frame statistics leave their declared bands.

    `extract` turns the observation the policy was shown into named statistics.
    Each declared statistic must be in its closed band `[low, high]`; NaN and
    infinity are outside. A declared statistic the extractor did not return is
    refused rather than passed.
    """

    def __init__(
        self,
        *,
        bands: Mapping[str, tuple[float, float]],
        extract: Callable[[Observation], Mapping[str, float]],
        name: str = "observation-shift",
    ) -> None:
        if not bands:
            raise ValueError("a FrameBand needs at least one band")
        for stat, (low, high) in bands.items():
            if not (math.isfinite(low) and math.isfinite(high) and low <= high):
                raise ValueError(
                    f"band for {stat} must be finite with low <= high, got {(low, high)}"
                )
        self._bands = dict(bands)
        self._extract = extract
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def on_step(self, record: StepRecord) -> Sequence[Flag]:
        stats = self._extract(record.observation)
        outside: list[str] = []
        for stat, (low, high) in self._bands.items():
            if stat not in stats:
                raise ValueError(
                    f"monitor {self._name!r} holds {stat} to a band, but the extractor "
                    f"returned no {stat} for step {record.index}"
                )
            value = stats[stat]
            if not low <= value <= high:
                # Exact values: rounded, a value just outside would read as on the edge.
                outside.append(f"{stat} = {value!r} not in [{low!r}, {high!r}]")
        if not outside:
            return []
        return [
            Flag(
                monitor=self._name,
                step=record.index,
                reason="frame statistics outside the declared band: " + "; ".join(outside),
            )
        ]
