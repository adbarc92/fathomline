"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest

from fathomline.core import Observation
from fathomline.monitors import (
    LIBERO_FRAME_BANDS,
    LIBERO_LOADED_PIPELINE,
    MonitorSuite,
    fixture_monitors,
)

type SuiteFactory = Callable[..., MonitorSuite]


def _in_band(_observation: Observation) -> Mapping[str, float]:
    return {stat: (low + high) / 2 for stat, (low, high) in LIBERO_FRAME_BANDS.items()}


@pytest.fixture
def fixture_suite() -> SuiteFactory:
    """`fixture_monitors`, with the adapter's hooks faked as a clean, in-envelope fixture.

    Pass `loaded_pipeline=` or `frame_statistics=` to fake a fault in one of them.
    """

    def build(
        *,
        loaded_pipeline: Callable[[], Mapping[str, str]] = lambda: LIBERO_LOADED_PIPELINE,
        frame_statistics: Callable[[Observation], Mapping[str, float]] = _in_band,
    ) -> MonitorSuite:
        return fixture_monitors(loaded_pipeline=loaded_pipeline, frame_statistics=frame_statistics)

    return build
