"""Lane B's done-when: run-to-run equality, and a test that fails when the seed is removed.

Equality is asserted on per-episode metrics and the action-trajectory hash, never
on video (D9), and only between runs on the same host: no cross-machine claim is
made until D9a is ruled.

Each run here is persisted and read back before it is compared, so what is shown
equal is what a later `report`, `gate` or `reproduce` would read, not an object
that never left memory.
"""

from __future__ import annotations

import secrets
from pathlib import Path

import pytest

from fathomline.core import Action, ArtifactRef, Env, Observation, Policy, RunResult, Transition
from fathomline.core.loop import run_sweep
from fathomline.fakes import FakeEnv, FakePolicy
from fathomline.persist import Host, SeedSet, compare_runs, read_run, write_run

SEED_SET = SeedSet(name="determinism", seeds=(1000, 1001, 1002, 1003, 1004), description="test")

HOST = Host(
    platform="test-platform",
    machine="x86_64",
    cpu_model="test cpu",
    cpu_flags=("avx2",),
    logical_cpus=4,
    usable_cpus=4,
    python_version="3.12.0",
    threading_env={},
    torch=None,
)


def _persisted_run(run_dir: Path, env: Env, policy: Policy) -> RunResult:
    sweep = run_sweep(env, policy, seeds=SEED_SET.seeds, max_steps=10)
    write_run(
        run_dir,
        sweep,
        seed_set=SEED_SET,
        host=HOST,
        started_at="2026-10-26T00:00:00Z",
        argv=("sweep",),
    )
    return read_run(run_dir).result


def _fresh_run(run_dir: Path) -> RunResult:
    return _persisted_run(run_dir, FakeEnv(success_rate=0.5), FakePolicy())


def test_the_same_seed_set_gives_equal_runs(tmp_path: Path) -> None:
    first = _fresh_run(tmp_path / "first")
    second = _fresh_run(tmp_path / "second")
    assert compare_runs(first, second) == ()


def test_the_equality_is_not_vacuous(tmp_path: Path) -> None:
    """If every seed gave the same episode, equal runs would prove nothing about seeding."""
    run = _fresh_run(tmp_path / "run")
    assert len({episode.action_hash for episode in run.episodes}) == len(SEED_SET.seeds)
    assert {episode.success for episode in run.episodes} == {True, False}


def test_reused_components_give_equal_runs(tmp_path: Path) -> None:
    """The seed resets all state: a second sweep on the same objects matches the first."""
    env, policy = FakeEnv(success_rate=0.5), FakePolicy()
    first = _persisted_run(tmp_path / "first", env, policy)
    second = _persisted_run(tmp_path / "second", env, policy)
    assert compare_runs(first, second) == ()


def test_an_episode_is_a_function_of_its_seed_alone() -> None:
    """Running the seed set in another order changes no episode."""
    forward = run_sweep(FakeEnv(success_rate=0.5), FakePolicy(), seeds=(1, 2, 3), max_steps=10)
    backward = run_sweep(FakeEnv(success_rate=0.5), FakePolicy(), seeds=(3, 2, 1), max_steps=10)
    assert forward.result.episodes == tuple(reversed(backward.result.episodes))


class _SeedNeverReachesPolicy:
    """The policy is never reseeded, so it runs on whatever state it was built with."""

    def __init__(self, inner: Policy) -> None:
        self._inner = inner

    @property
    def ref(self) -> ArtifactRef:
        return self._inner.ref

    def reset(self, seed: int) -> None:
        del seed

    def act(self, observation: Observation) -> Action:
        return self._inner.act(observation)


class _SeedNeverReachesEnv:
    """The env is reset from entropy, as a library does when it is handed no seed."""

    def __init__(self, inner: Env) -> None:
        self._inner = inner

    @property
    def ref(self) -> ArtifactRef:
        return self._inner.ref

    def reset(self, seed: int) -> Observation:
        del seed
        return self._inner.reset(secrets.randbits(63))

    def step(self, action: Action) -> Transition:
        return self._inner.step(action)


@pytest.mark.parametrize("removed_from", ["policy", "env", "both"])
def test_run_to_run_equality_fails_when_the_seed_is_removed(
    tmp_path: Path, removed_from: str
) -> None:
    """The check above must be able to fail. With the seed removed, it does."""

    def unseeded_run(run_dir: Path) -> RunResult:
        env: Env = FakeEnv(success_rate=0.5)
        policy: Policy = FakePolicy()
        if removed_from in ("env", "both"):
            env = _SeedNeverReachesEnv(env)
        if removed_from in ("policy", "both"):
            policy = _SeedNeverReachesPolicy(policy)
        return _persisted_run(run_dir, env, policy)

    differences = compare_runs(unseeded_run(tmp_path / "first"), unseeded_run(tmp_path / "second"))
    assert any("action_hash" in difference for difference in differences), differences
