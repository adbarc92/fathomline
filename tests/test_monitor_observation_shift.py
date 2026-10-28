"""Lane G: the observation-shift monitor (ruling D10, condition 3), on the fake.

D10 rules the condition as: **a frame brightness/noise band, calibrated on
held-out episodes; it is known to miss a camera swap.** In the envelope spike,
dim and bright lights moved frame brightness from about 102 to 37 and 140, and
camera noise moved the noise statistic from about 5.9 to 15.4. A camera swap
moved neither far, because both views have similar statistics.

The statistics are the adapter's to compute from its image tensors; the monitor
holds each one to its declared band. Here a fake extractor stands in for the
adapter, and the shift is injected through the fake's observation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import pytest
from conftest import SuiteFactory

from fathomline.core import Observation, RunManifest, RunResult, StepRecord, run_episode
from fathomline.fakes import FakeEnv, FakePolicy
from fathomline.gates import GateConfig, evaluate
from fathomline.monitors import (
    LIBERO_FRAME_BANDS,
    LIBERO_INSTRUCTIONS,
    FrameBand,
    MonitorSuite,
    is_flagged,
)

BANDS = {"image.brightness": (92.0, 115.0), "image.noise": (5.3, 7.0)}
CLEAN = {"image.brightness": 102.2, "image.noise": 5.87}
SEEDS = tuple(range(1000, 1010))
MAX_STEPS = 10


def _record(index: int = 0) -> StepRecord:
    return StepRecord(index=index, observation={}, action=(0.0,), reward=0.0)


def _monitor(stats: Mapping[str, float]) -> FrameBand:
    return FrameBand(bands=BANDS, extract=lambda _observation: stats)


# --- The ruled declaration ----------------------------------------------------------------------


def test_the_declared_bands_cover_the_third_person_camera() -> None:
    assert set(LIBERO_FRAME_BANDS) == {"image.brightness", "image.noise"}


def test_the_declared_bands_hold_the_clean_spike_frame_and_exclude_its_shifts() -> None:
    # The spike's frame-0 values: clean 102-103 / 5.8-5.9; dim 37; bright 140; noise 15.4.
    low_b, high_b = LIBERO_FRAME_BANDS["image.brightness"]
    low_n, high_n = LIBERO_FRAME_BANDS["image.noise"]
    assert low_b < 102.0 and high_b > 103.0
    assert low_n < 5.8 and high_n > 5.9
    assert low_b > 37.0 and high_b < 140.0
    assert high_n < 15.4


def test_the_fixture_monitors_include_the_observation_shift_monitor(
    fixture_suite: SuiteFactory,
) -> None:
    suite = fixture_suite()
    assert "observation-shift" in suite.name


def test_the_fixture_suite_flags_a_dim_frame(fixture_suite: SuiteFactory) -> None:
    record = StepRecord(
        index=0, observation={"task": sorted(LIBERO_INSTRUCTIONS)}, action=(0.0,) * 7, reward=0.0
    )
    dim = {**CLEAN, "image.brightness": 37.0}
    suite = fixture_suite(frame_statistics=lambda _o: dim)
    (flag,) = suite.on_step(record)
    assert flag.monitor == "observation-shift"


# --- One step -----------------------------------------------------------------------------------


def test_a_frame_inside_every_band_is_not_flagged() -> None:
    assert list(_monitor(CLEAN).on_step(_record())) == []


@pytest.mark.parametrize("value", [92.0, 115.0])
def test_a_value_on_the_edge_of_its_band_is_inside(value: float) -> None:
    assert list(_monitor({**CLEAN, "image.brightness": value}).on_step(_record())) == []


@pytest.mark.parametrize(
    ("stat", "value"),
    [
        ("image.brightness", 37.0),  # dim
        ("image.brightness", 140.0),  # bright
        ("image.noise", 15.4),  # sensor noise
        ("image.noise", 2.2),  # blur, or the dim frame's flattened detail
        ("image.brightness", math.nan),
        ("image.noise", math.inf),
    ],
)
def test_a_value_outside_its_band_is_flagged_and_named(stat: str, value: float) -> None:
    (flag,) = _monitor({**CLEAN, stat: value}).on_step(_record(index=6))
    assert (flag.monitor, flag.step) == ("observation-shift", 6)
    assert stat in flag.reason
    (other,) = set(BANDS) - {stat}
    assert other not in flag.reason


def test_one_flag_per_step_names_every_statistic_out_of_band() -> None:
    (flag,) = _monitor({"image.brightness": 37.0, "image.noise": 2.2}).on_step(_record())
    assert "image.brightness" in flag.reason
    assert "image.noise" in flag.reason


def test_a_statistic_the_extractor_did_not_return_is_refused() -> None:
    # Skipping it would report "inside the band" for a frame nobody measured.
    with pytest.raises(ValueError, match=r"image\.noise"):
        _monitor({"image.brightness": 102.2}).on_step(_record())


def test_the_frame_is_read_from_the_step_shown() -> None:
    seen: list[Observation] = []

    def extract(observation: Observation) -> Mapping[str, float]:
        seen.append(observation)
        return CLEAN

    record = StepRecord(index=0, observation={"marker": 1}, action=(0.0,), reward=0.0)
    FrameBand(bands=BANDS, extract=extract).on_step(record)
    assert seen == [record.observation]


def test_the_monitor_name_is_configurable() -> None:
    monitor = FrameBand(bands=BANDS, extract=lambda _o: {**CLEAN, "image.noise": 15.4}, name="px")
    (flag,) = monitor.on_step(_record())
    assert flag.monitor == "px"


# --- Construction -------------------------------------------------------------------------------


def test_no_bands_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        FrameBand(bands={}, extract=lambda _o: CLEAN)


@pytest.mark.parametrize("band", [(5.0, 4.0), (math.nan, 1.0), (0.0, math.inf)])
def test_a_band_that_is_not_a_finite_low_to_high_interval_is_refused(
    band: tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match=r"image\.noise"):
        FrameBand(bands={**BANDS, "image.noise": band}, extract=lambda _o: CLEAN)


def test_the_bands_are_copied() -> None:
    bands = dict(BANDS)
    monitor = FrameBand(bands=bands, extract=lambda _o: CLEAN)
    bands["image.brightness"] = (0.0, 1.0)
    assert list(monitor.on_step(_record())) == []


# --- Through the loop, with the shift injected --------------------------------------------------


def _lights(scale: float) -> Callable[[Observation], Observation]:
    def perturb(observation: Observation) -> Observation:
        return {**observation, "brightness": 102.2 * scale}

    return perturb


def _extract(observation: Observation) -> Mapping[str, float]:
    brightness = observation["brightness"]
    assert isinstance(brightness, float)
    return {"image.brightness": brightness, "image.noise": 5.87}


def _run(scale: float) -> RunResult:
    env, policy = FakeEnv(success_rate=1.0, perturb=_lights(scale)), FakePolicy()
    manifest = RunManifest.capture(env=env, policy=policy, seeds=SEEDS, max_steps=MAX_STEPS)
    suite = MonitorSuite([FrameBand(bands=LIBERO_FRAME_BANDS, extract=_extract)])
    episodes = tuple(
        run_episode(env, policy, seed=seed, max_steps=MAX_STEPS, monitors=[suite]) for seed in SEEDS
    )
    return RunResult(manifest=manifest, episodes=episodes)


def test_a_run_under_the_declared_lighting_is_not_flagged() -> None:
    assert is_flagged(_run(1.0)) is False


@pytest.mark.parametrize("scale", [0.35, 1.4])
def test_a_run_under_shifted_lighting_is_flagged_from_its_first_frame(scale: float) -> None:
    run = _run(scale)
    for episode in run.episodes:
        assert episode.flags[0].step == 0
    verdict = evaluate(run, GateConfig(threshold=0.5, min_episodes=len(SEEDS)))
    assert all(episode.success for episode in run.episodes)
    assert verdict.passed is False
