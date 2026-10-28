"""Lane G: the provenance monitor (ruling D10, condition 1), on the fake.

D10 rules the condition as: **check what the loaded pipeline uses, not only the
on-disk digests** `pins.verify` already checks. In the envelope spike the
lost-stats fault lived in memory, after load: every file still hashed to its pin,
and the arm reached 2.3-2.7.

The monitor compares a fingerprint of the loaded pipeline, one digest per
component, against the declared one. The fingerprint is the adapter's to take;
here a fake pipeline stands in for it, and the fault is injected by changing the
fake's state after the monitor is built, the way the spike changed the
unnormalizer after load.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest

from fathomline.core import RunManifest, RunResult, StepRecord, run_episode
from fathomline.fakes import FakeEnv, FakePolicy
from fathomline.gates import GateConfig, evaluate
from fathomline.monitors import (
    LIBERO_LOADED_PIPELINE,
    MonitorSuite,
    Provenance,
    fixture_monitors,
    is_flagged,
)

DECLARED = {"policy.unnormalizer": "a" * 64, "env.cameras": "b" * 64}
SEEDS = tuple(range(1000, 1010))
MAX_STEPS = 10


def _record(index: int = 0) -> StepRecord:
    return StepRecord(index=index, observation={}, action=(0.0,), reward=0.0)


def _probe(loaded: Mapping[str, str]) -> Callable[[], Mapping[str, str]]:
    return lambda: loaded


def _monitor(loaded: Mapping[str, str]) -> Provenance:
    return Provenance(declared=DECLARED, probe=_probe(loaded))


# --- The ruled declaration ----------------------------------------------------------------------


def test_the_declaration_covers_the_action_stats_and_the_camera_wiring() -> None:
    # The two faults D10 names: the action unnormalization stats, and the cameras as wired.
    assert {
        "policy.unnormalizer",
        "policy.normalizer",
        "policy.rename_map",
        "env.cameras",
    } == set(LIBERO_LOADED_PIPELINE)


def test_every_declared_digest_is_a_sha256() -> None:
    for component, digest in LIBERO_LOADED_PIPELINE.items():
        assert len(digest) == 64, component
        int(digest, 16)


def test_the_fixture_monitors_include_the_provenance_monitor() -> None:
    suite = fixture_monitors(loaded_pipeline=_probe(LIBERO_LOADED_PIPELINE))
    assert "provenance" in suite.name


def test_the_fixture_suite_flags_a_loaded_pipeline_that_differs_from_the_declaration() -> None:
    lost = {**LIBERO_LOADED_PIPELINE, "policy.unnormalizer": "0" * 64}
    record = StepRecord(
        index=0,
        observation={"task": ["pick up the alphabet soup and place it in the basket"]},
        action=(0.0,) * 7,
        reward=0.0,
    )
    assert (
        list(fixture_monitors(loaded_pipeline=_probe(LIBERO_LOADED_PIPELINE)).on_step(record)) == []
    )
    (flag,) = fixture_monitors(loaded_pipeline=_probe(lost)).on_step(record)
    assert flag.monitor == "provenance"
    assert "policy.unnormalizer" in flag.reason


# --- One step -----------------------------------------------------------------------------------


def test_a_pipeline_matching_the_declaration_is_not_flagged() -> None:
    assert list(_monitor(dict(DECLARED)).on_step(_record())) == []


def test_a_changed_component_is_flagged_and_named() -> None:
    (flag,) = _monitor({**DECLARED, "policy.unnormalizer": "c" * 64}).on_step(_record(index=2))
    assert (flag.monitor, flag.step) == ("provenance", 2)
    assert "policy.unnormalizer" in flag.reason
    assert "env.cameras" not in flag.reason


def test_a_component_the_pipeline_no_longer_reports_is_flagged() -> None:
    (flag,) = _monitor({"policy.unnormalizer": "a" * 64}).on_step(_record())
    assert "env.cameras" in flag.reason
    assert "missing" in flag.reason


def test_a_component_the_declaration_does_not_cover_is_flagged() -> None:
    # The envelope cannot vouch for a part of the pipeline nobody declared.
    (flag,) = _monitor({**DECLARED, "policy.extra": "d" * 64}).on_step(_record())
    assert "policy.extra" in flag.reason
    assert "not declared" in flag.reason


def test_one_flag_per_step_names_every_mismatch() -> None:
    (flag,) = _monitor({"policy.unnormalizer": "c" * 64}).on_step(_record())
    assert "policy.unnormalizer" in flag.reason
    assert "env.cameras" in flag.reason


def test_the_pipeline_is_read_on_every_step() -> None:
    # A fault can enter after load, between episodes or mid-episode.
    calls = 0

    def probe() -> Mapping[str, str]:
        nonlocal calls
        calls += 1
        return DECLARED

    monitor = Provenance(declared=DECLARED, probe=probe)
    for index in range(3):
        monitor.on_step(_record(index))
    assert calls == 3


def test_the_monitor_name_is_configurable() -> None:
    monitor = Provenance(declared=DECLARED, probe=_probe({}), name="pins")
    (flag,) = monitor.on_step(_record())
    assert flag.monitor == "pins"


# --- Construction -------------------------------------------------------------------------------


def test_an_empty_declaration_is_refused() -> None:
    # Nothing declared would make every pipeline "match".
    with pytest.raises(ValueError, match="at least one"):
        Provenance(declared={}, probe=_probe({}))


def test_the_declaration_is_copied() -> None:
    declared = dict(DECLARED)
    monitor = Provenance(declared=declared, probe=_probe(dict(DECLARED)))
    declared["policy.unnormalizer"] = "c" * 64
    assert list(monitor.on_step(_record())) == []


# --- Through the loop, with the fault injected after load ---------------------------------------


class _Pipeline:
    """A stand-in for the loaded pipeline: its fingerprint is whatever its state is now."""

    def __init__(self) -> None:
        self.state = dict(DECLARED)

    def fingerprint(self) -> Mapping[str, str]:
        return dict(self.state)


def _run(pipeline: _Pipeline) -> RunResult:
    env, policy = FakeEnv(success_rate=1.0), FakePolicy()
    manifest = RunManifest.capture(env=env, policy=policy, seeds=SEEDS, max_steps=MAX_STEPS)
    suite = MonitorSuite([Provenance(declared=DECLARED, probe=pipeline.fingerprint)])
    episodes = tuple(
        run_episode(env, policy, seed=seed, max_steps=MAX_STEPS, monitors=[suite]) for seed in SEEDS
    )
    return RunResult(manifest=manifest, episodes=episodes)


def test_a_run_on_the_declared_pipeline_is_not_flagged() -> None:
    assert is_flagged(_run(_Pipeline())) is False


def test_stats_lost_after_load_flag_every_episode_from_its_first_step() -> None:
    pipeline = _Pipeline()
    pipeline.state["policy.unnormalizer"] = "c" * 64  # after the monitor was built
    run = _run(pipeline)
    for episode in run.episodes:
        assert episode.flags[0].step == 0
    verdict = evaluate(run, GateConfig(threshold=0.5, min_episodes=len(SEEDS)))
    assert all(episode.success for episode in run.episodes)
    assert verdict.passed is False
