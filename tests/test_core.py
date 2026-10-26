"""Contract tests for the core: the step loop, the fakes, and result schema v1.

Everything here runs against `FakeEnv` / `FakePolicy`, so it runs on any
platform. The real adapter is Linux-only and is tested in the container.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from fathomline import __version__
from fathomline.core import (
    SCHEMA_VERSION,
    ArtifactRef,
    Env,
    EpisodeResult,
    Flag,
    Monitor,
    Observation,
    Policy,
    RunManifest,
    RunResult,
    SchemaError,
    StepRecord,
    from_document,
    run_episode,
    to_document,
)
from fathomline.fakes import FakeEnv, FakePolicy


def _episode(seed: int, *, success_rate: float = 1.0, max_steps: int = 10) -> EpisodeResult:
    return run_episode(
        FakeEnv(success_rate=success_rate), FakePolicy(), seed=seed, max_steps=max_steps
    )


def test_fakes_satisfy_the_protocols() -> None:
    env: Env = FakeEnv(success_rate=1.0)
    policy: Policy = FakePolicy()
    assert env.ref.name == "fake-env"
    assert policy.ref.name == "fake-policy"


def test_the_fake_runs_one_episode_through_the_loop() -> None:
    result = _episode(1000)
    assert result.seed == 1000
    assert result.success is True
    assert result.steps == 5
    assert result.max_reward <= 0.0
    assert result.sum_reward <= result.max_reward
    assert len(result.action_hash) == 64
    assert result.flags == ()


def test_same_seed_gives_the_same_episode() -> None:
    assert _episode(1000) == _episode(1000)


def test_a_different_seed_gives_a_different_trajectory() -> None:
    assert _episode(1000).action_hash != _episode(2000).action_hash


@pytest.mark.parametrize(("success_rate", "expected"), [(0.0, 0), (1.0, 20)])
def test_success_rate_is_controllable(success_rate: float, expected: int) -> None:
    successes = sum(_episode(seed, success_rate=success_rate).success for seed in range(20))
    assert successes == expected


def test_an_intermediate_success_rate_mixes_outcomes() -> None:
    successes = sum(_episode(seed, success_rate=0.5).success for seed in range(200))
    assert 70 <= successes <= 130


def test_an_episode_that_hits_max_steps_is_a_failure() -> None:
    result = _episode(1000, max_steps=3)
    assert result.steps == 3
    assert result.success is False


def test_max_steps_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        _episode(1000, max_steps=0)


def test_success_rate_must_be_a_probability() -> None:
    with pytest.raises(ValueError, match="success_rate"):
        FakeEnv(success_rate=1.5)


def test_an_injected_perturbation_changes_what_the_policy_sees() -> None:
    def shift(observation: Observation) -> Observation:
        position = observation["position"]
        assert isinstance(position, float)
        return {"position": position + 10.0}

    perturbed = run_episode(
        FakeEnv(success_rate=1.0, perturb=shift), FakePolicy(), seed=1000, max_steps=10
    )
    assert perturbed.action_hash != _episode(1000).action_hash


def test_the_fake_policy_rejects_an_observation_it_cannot_read() -> None:
    policy = FakePolicy()
    policy.reset(0)
    with pytest.raises(TypeError, match="position"):
        policy.act({"position": "left"})


def test_monitors_see_every_step_and_their_flags_reach_the_result() -> None:
    class FlagStepTwo:
        name = "flag-step-two"

        def __init__(self) -> None:
            self.seen: list[int] = []

        def on_step(self, record: StepRecord) -> Sequence[Flag]:
            self.seen.append(record.index)
            if record.index == 2:
                return [Flag(monitor=self.name, step=record.index, reason="test")]
            return []

    monitor = FlagStepTwo()
    monitors: list[Monitor] = [monitor]
    result = run_episode(
        FakeEnv(success_rate=1.0), FakePolicy(), seed=1000, max_steps=10, monitors=monitors
    )
    assert monitor.seen == [0, 1, 2, 3, 4]
    assert result.flags == (Flag(monitor="flag-step-two", step=2, reason="test"),)
    # A monitor observes; it must not change the trajectory.
    assert result.action_hash == _episode(1000).action_hash


def _run_result() -> RunResult:
    env, policy = FakeEnv(success_rate=1.0), FakePolicy()
    seeds = (1000, 2000)
    manifest = RunManifest.capture(env=env, policy=policy, seeds=seeds, max_steps=10)
    episodes = tuple(run_episode(env, policy, seed=seed, max_steps=10) for seed in seeds)
    return RunResult(manifest=manifest, episodes=episodes)


def test_the_manifest_records_provenance() -> None:
    manifest = _run_result().manifest
    assert manifest.fathomline_version == __version__
    assert manifest.env == ArtifactRef(name="fake-env", revision="0")
    assert manifest.policy == ArtifactRef(name="fake-policy", revision="0")
    assert manifest.seeds == (1000, 2000)
    assert manifest.max_steps == 10
    assert manifest.python_version
    assert manifest.platform


def test_a_result_round_trips_through_json() -> None:
    result = _run_result()
    document = json.loads(json.dumps(to_document(result)))
    assert document["schema_version"] == SCHEMA_VERSION == 1
    assert from_document(document) == result


def test_flags_survive_the_round_trip() -> None:
    result = _run_result()
    flagged = EpisodeResult(
        seed=1,
        steps=1,
        success=False,
        sum_reward=0.0,
        max_reward=0.0,
        action_hash="0" * 64,
        flags=(Flag(monitor="m", step=0, reason="r"),),
    )
    with_flag = RunResult(manifest=result.manifest, episodes=(flagged,))
    assert from_document(json.loads(json.dumps(to_document(with_flag)))) == with_flag


def test_a_frozen_v1_document_still_parses() -> None:
    """Schema v1 as first written. If this breaks, bump the version; do not edit this."""
    document = {
        "schema_version": 1,
        "manifest": {
            "fathomline_version": "0.0.1",
            "policy": {"name": "fake-policy", "revision": "0"},
            "env": {"name": "fake-env", "revision": "0"},
            "seeds": [7],
            "max_steps": 10,
            "python_version": "3.12.0",
            "platform": "Linux-x86_64",
        },
        "episodes": [
            {
                "seed": 7,
                "steps": 5,
                "success": True,
                "sum_reward": -1.5,
                "max_reward": -0.1,
                "action_hash": "ab" * 32,
                "flags": [{"monitor": "m", "step": 3, "reason": "r"}],
            }
        ],
    }
    result = from_document(document)
    assert result.manifest.seeds == (7,)
    assert result.episodes[0].flags == (Flag(monitor="m", step=3, reason="r"),)


@pytest.mark.parametrize("version", [0, 2, None])
def test_an_unknown_schema_version_is_refused(version: int | None) -> None:
    document = to_document(_run_result())
    document["schema_version"] = version
    with pytest.raises(SchemaError, match="schema_version"):
        from_document(document)
