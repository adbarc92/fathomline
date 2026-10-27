"""Lane B: the seeded sweep, `tests.lock`, host facts, the run directory and the comparison.

All against the fakes, so it runs on any platform. The run-to-run equality test
and its seed-removal counterpart are in `test_determinism.py`.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import struct
import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from fathomline.core import EpisodeResult, Flag, RunResult
from fathomline.core.loop import Sweep, action_hash, run_episode, run_sweep
from fathomline.fakes import FakeEnv, FakePolicy
from fathomline.gates import GateConfig, evaluate
from fathomline.persist import (
    DEFAULT_LOCK,
    RESULT_FILENAME,
    RUN_FILENAME,
    TRAJECTORIES_DIRNAME,
    Divergence,
    Host,
    LockError,
    RunDirError,
    SeedSet,
    capture_host,
    compare_runs,
    first_divergence,
    load_lock,
    read_run,
    seed_digest,
    seed_set,
    write_run,
)
from fathomline.persist import host as host_module
from fathomline.report import RESULT_FILENAME as REPORT_RESULT_FILENAME
from fathomline.report import load_run, report_run_dir

REPO = Path(__file__).resolve().parents[1]
WEEK2_RUN = REPO / "tests" / "fixtures" / "persist" / "week2-run"

SEEDS = (1000, 1001, 1002)
SEED_SET = SeedSet(name="three", seeds=SEEDS, description="three episodes")
HOST = Host(
    platform="test-platform",
    machine="x86_64",
    cpu_model="test cpu",
    cpu_flags=("avx", "avx2"),
    logical_cpus=8,
    usable_cpus=4,
    python_version="3.12.0",
    threading_env={"OMP_NUM_THREADS": "4"},
    torch={"version": "2.11.0+cpu", "num_threads": 4},
)


def _sweep(seeds: tuple[int, ...] = SEEDS) -> Sweep:
    return run_sweep(FakeEnv(success_rate=0.5), FakePolicy(), seeds=seeds, max_steps=10)


def _write(run_dir: Path, sweep: Sweep | None = None) -> Path:
    return write_run(
        run_dir,
        sweep or _sweep(),
        seed_set=SEED_SET,
        host=HOST,
        started_at="2026-10-26T00:00:00Z",
        argv=("sweep", "--seed-set", "three"),
    )


# --- The seeded loop -----------------------------------------------------------


def test_action_hash_recomputes_what_the_loop_records() -> None:
    actions: list[tuple[float, ...]] = []
    episode = run_episode(
        FakeEnv(success_rate=1.0), FakePolicy(), seed=7, max_steps=10, actions=actions
    )
    assert len(actions) == episode.steps
    assert action_hash(actions) == episode.action_hash


def test_the_action_hash_is_unchanged_by_lane_b() -> None:
    """CI hashes must stay comparable: this value was computed by the Lane 0 loop."""
    assert action_hash([(0.5, -0.25), (1.0,)]) == (
        "ab2c07a9c1ad36f04328d68747f784968b2db974a141e4da5189df676601d5a1"
    )


def test_a_sweep_runs_one_episode_per_seed_in_order() -> None:
    ticks = iter(range(100))
    sweep = run_sweep(
        FakeEnv(success_rate=1.0),
        FakePolicy(),
        seeds=(5, 3, 9),
        max_steps=10,
        clock=lambda: float(next(ticks)),
    )
    assert [episode.seed for episode in sweep.result.episodes] == [5, 3, 9]
    assert sweep.result.manifest.seeds == (5, 3, 9)
    assert sweep.result.manifest.max_steps == 10
    assert sweep.durations == (1.0, 1.0, 1.0)
    for episode, actions in zip(sweep.result.episodes, sweep.trajectories, strict=True):
        assert action_hash(actions) == episode.action_hash


def test_a_sweep_passes_monitors_to_every_episode() -> None:
    class FlagFirstStep:
        name = "first-step"

        def on_step(self, record: object) -> list[Flag]:
            index = getattr(record, "index")  # noqa: B009 - keeps the stub protocol-free
            return [Flag(monitor=self.name, step=0, reason="r")] if index == 0 else []

    sweep = run_sweep(
        FakeEnv(success_rate=1.0),
        FakePolicy(),
        seeds=(1, 2),
        max_steps=10,
        monitors=[FlagFirstStep()],
    )
    assert all(len(episode.flags) == 1 for episode in sweep.result.episodes)


def test_a_sweep_refuses_an_empty_seed_set() -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        run_sweep(FakeEnv(success_rate=1.0), FakePolicy(), seeds=(), max_steps=10)


def test_a_sweep_refuses_a_repeated_seed() -> None:
    """One episode counted twice would inflate a gate's n."""
    with pytest.raises(ValueError, match="repeated: 1"):
        run_sweep(FakeEnv(success_rate=1.0), FakePolicy(), seeds=(1, 2, 1), max_steps=10)


# --- tests.lock ----------------------------------------------------------------


def _lock(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "tests.lock"
    path.write_text(body, encoding="utf-8")
    return path


def _entry(name: str, seeds: list[int], *, digest: str | None = None) -> str:
    digest = seed_digest(seeds) if digest is None else digest
    return f'[seed_sets.{name}]\ndescription = "why"\nseeds = {seeds}\ndigest = "{digest}"\n'


def test_the_committed_lock_declares_the_smoke_set() -> None:
    sets = load_lock(REPO / DEFAULT_LOCK)
    assert sets["smoke"].seeds == (1000,)
    # Every declared set is runnable and distinct.
    assert all(len(set(s.seeds)) == len(s.seeds) for s in sets.values())


def test_the_committed_full_set_covers_each_libero_init_state_once() -> None:
    """The adapter maps seed to init state as seed % 50 for `libero_object`."""
    full = seed_set(REPO / DEFAULT_LOCK, "full")
    assert sorted(seed % 50 for seed in full.seeds) == list(range(50))


def test_a_seed_set_is_read_with_its_digest(tmp_path: Path) -> None:
    path = _lock(tmp_path, "version = 1\n" + _entry("a", [3, 1, 2]))
    declared = seed_set(path, "a")
    assert declared == SeedSet(name="a", seeds=(3, 1, 2), description="why")
    assert declared.digest == seed_digest((3, 1, 2))


def test_seed_order_is_part_of_the_digest() -> None:
    assert seed_digest((1, 2)) != seed_digest((2, 1))


def test_an_edited_seed_set_is_refused_with_the_digest_it_needs(tmp_path: Path) -> None:
    path = _lock(tmp_path, "version = 1\n" + _entry("a", [1, 2, 3], digest=seed_digest([1, 2])))
    with pytest.raises(LockError, match=seed_digest((1, 2, 3))):
        load_lock(path)


def test_an_unknown_seed_set_lists_the_declared_ones(tmp_path: Path) -> None:
    path = _lock(tmp_path, "version = 1\n" + _entry("b", [1]) + _entry("a", [2]))
    with pytest.raises(LockError, match="no seed set 'c'; declared: a, b"):
        seed_set(path, "c")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("version = 2\n", "unsupported version 2"),
        ("version = 1\n", "declares no"),
        ("version = 1\nseed_sets = 3\n", "declares no"),
        ("version = 1\n[seed_sets]\na = 3\n", "is not a table"),
        ('version = 1\n[seed_sets.a]\ndescription = "x"\nseeds = []\n', "non-empty list"),
        ('version = 1\n[seed_sets.a]\ndescription = "x"\nseeds = "1"\n', "non-empty list"),
        ('version = 1\n[seed_sets.a]\ndescription = "x"\nseeds = [1, true]\n', "integer"),
        ('version = 1\n[seed_sets.a]\ndescription = "x"\nseeds = [1, 1.5]\n', "integer"),
        (
            'version = 1\n[seed_sets.a]\ndescription = "x"\nseeds = [1, 2, 1]\n',
            r"repeated seeds \[1\]",
        ),
        ("version = 1\n[seed_sets.a]\nseeds = [1]\n", "needs a description"),
        ('version = 1\n[seed_sets.a]\ndescription = " "\nseeds = [1]\n', "needs a description"),
        ("version = [\n", "not valid TOML"),
    ],
)
def test_a_malformed_lock_is_refused(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(LockError, match=message):
        load_lock(_lock(tmp_path, body))


def test_a_missing_lock_is_refused(tmp_path: Path) -> None:
    with pytest.raises(LockError, match="cannot read"):
        load_lock(tmp_path / "absent.lock")


# --- Host facts ----------------------------------------------------------------

CPUINFO = """\
processor\t: 0
vendor_id\t: AuthenticAMD
model name\t: AMD EPYC 9V74 80-Core Processor
flags\t\t: fpu sse sse2 ssse3 sse4_1 avx avx2 fma avx512f avx512_bf16 lm f16c

processor\t: 1
model name\t: should not be read
flags\t\t: fpu
"""


def test_host_facts_come_from_cpuinfo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(CPUINFO, encoding="utf-8")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.delenv("ATEN_CPU_CAPABILITY", raising=False)
    host = capture_host(cpuinfo)
    assert host.cpu_model == "AMD EPYC 9V74 80-Core Processor"
    assert host.cpu_flags == (
        "avx",
        "avx2",
        "avx512_bf16",
        "avx512f",
        "f16c",
        "fma",
        "sse",
        "sse2",
        "sse4_1",
        "ssse3",
    )
    assert host.threading_env["OMP_NUM_THREADS"] == "2"
    assert "ATEN_CPU_CAPABILITY" not in host.threading_env
    assert host.python_version
    assert host.logical_cpus is None or host.logical_cpus >= 1


def test_without_cpuinfo_the_model_comes_from_platform(tmp_path: Path) -> None:
    host = capture_host(tmp_path / "absent")
    assert host.cpu_model
    assert host.cpu_flags == ()


def test_usable_cpus_is_the_affinity_mask_where_there_is_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1}, raising=False)
    assert host_module._usable_cpus() == 2
    monkeypatch.delattr(os, "sched_getaffinity")
    assert host_module._usable_cpus() is None


@pytest.fixture
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    torch = types.SimpleNamespace(
        __version__="2.11.0+cpu",
        get_num_threads=lambda: 4,
        get_num_interop_threads=lambda: 4,
        backends=types.SimpleNamespace(
            cpu=types.SimpleNamespace(get_cpu_capability=lambda: "AVX2")
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    yield


@pytest.mark.usefixtures("fake_torch")
def test_torch_settings_are_recorded_when_the_fixture_imported_torch() -> None:
    assert capture_host().torch == {
        "version": "2.11.0+cpu",
        "num_threads": 4,
        "num_interop_threads": 4,
        "cpu_capability": "AVX2",
    }


def test_torch_is_not_imported_just_to_record_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert capture_host().torch is None
    assert "torch" not in sys.modules


# --- The run directory ---------------------------------------------------------


def test_a_run_directory_reads_back_as_written(tmp_path: Path) -> None:
    sweep = _sweep()
    run_dir = _write(tmp_path / "run", sweep)
    persisted = read_run(run_dir)
    assert persisted.result == sweep.result
    assert persisted.trajectories == sweep.trajectories
    assert persisted.record.seed_set == SEED_SET
    assert persisted.record.host == HOST
    assert persisted.record.episode_durations == sweep.durations
    assert persisted.record.argv == ("sweep", "--seed-set", "three")
    assert sorted(p.name for p in run_dir.iterdir()) == [
        RESULT_FILENAME,
        RUN_FILENAME,
        TRAJECTORIES_DIRNAME,
    ]


def test_the_same_seed_set_writes_the_same_result_bytes(tmp_path: Path) -> None:
    """What varies run to run lives in run.json; result.json is the measurement."""
    first = _write(tmp_path / "first") / RESULT_FILENAME
    second = _write(tmp_path / "second") / RESULT_FILENAME
    assert first.read_bytes() == second.read_bytes()


def test_report_and_gate_read_a_run_directory_with_no_re_run(tmp_path: Path) -> None:
    run_dir = _write(tmp_path / "run")
    assert RESULT_FILENAME == REPORT_RESULT_FILENAME
    assert load_run(run_dir) == read_run(run_dir).result
    assert "## Episodes" in report_run_dir(run_dir)
    verdict = evaluate(load_run(run_dir), GateConfig(threshold=0.5, min_episodes=30))
    assert not verdict.passed


def test_an_existing_run_directory_is_never_overwritten(tmp_path: Path) -> None:
    run_dir = _write(tmp_path / "run")
    before = (run_dir / RESULT_FILENAME).read_bytes()
    with pytest.raises(RunDirError, match="already exists"):
        _write(run_dir)
    assert (run_dir / RESULT_FILENAME).read_bytes() == before


def test_a_stale_partial_directory_is_replaced(tmp_path: Path) -> None:
    stale = tmp_path / ".run.partial"
    stale.mkdir()
    (stale / "junk").write_text("from a crashed sweep", encoding="utf-8")
    run_dir = _write(tmp_path / "run")
    assert not stale.exists()
    assert not (run_dir / "junk").exists()


def test_a_sweep_is_not_persisted_under_another_seed_set(tmp_path: Path) -> None:
    with pytest.raises(RunDirError, match="not seed set 'three'"):
        _write(tmp_path / "run", _sweep((1, 2)))
    assert not (tmp_path / "run").exists()


def _tamper(run_dir: Path, name: str, edit: Callable[[dict[str, object]], None]) -> None:
    path = run_dir / name
    document = json.loads(path.read_text(encoding="utf-8"))
    edit(document)
    path.write_text(json.dumps(document), encoding="utf-8")


def _set(key: str, value: object) -> Callable[[dict[str, object]], None]:
    def edit(document: dict[str, object]) -> None:
        document[key] = value

    return edit


def _set_in(key: str, inner: str, value: object) -> Callable[[dict[str, object]], None]:
    def edit(document: dict[str, object]) -> None:
        section = document[key]
        assert isinstance(section, dict)
        section[inner] = value

    return edit


def _drop(key: str) -> Callable[[dict[str, object]], None]:
    def edit(document: dict[str, object]) -> None:
        del document[key]

    return edit


def _reseeded(document: dict[str, object]) -> None:
    """A run record for another seed set, internally consistent."""
    document["seed_set"] = {
        "name": "two",
        "description": "d",
        "seeds": [1000, 1001],
        "digest": seed_digest((1000, 1001)),
    }


def _first_action_nudged(document: dict[str, object]) -> None:
    """The last bit of the first action's first float, flipped."""
    actions = document["actions"]
    assert isinstance(actions, list)
    raw = bytearray.fromhex(actions[0])
    raw[0] ^= 1
    actions[0] = raw.hex()


def _episode_dropped(document: dict[str, object]) -> None:
    """A result whose episodes no longer match its own manifest."""
    episodes = document["episodes"]
    assert isinstance(episodes, list)
    del episodes[-1]


@pytest.mark.parametrize(
    ("name", "edit", "message"),
    [
        (RESULT_FILENAME, _set("schema_version", 9), "unsupported schema_version 9"),
        (RESULT_FILENAME, _drop("episodes"), "malformed result"),
        (RUN_FILENAME, _set("run_schema_version", 2), "unsupported run_schema_version 2"),
        (RUN_FILENAME, _drop("host"), "malformed run record"),
        (RUN_FILENAME, _set("host", {"platform": "x"}), "malformed run record"),
        (RUN_FILENAME, _reseeded, "declares seeds"),
        (RUN_FILENAME, _set_in("seed_set", "digest", "0" * 64), "but the record says"),
        (RUN_FILENAME, _set("episode_durations", [1.0]), "1 episode durations for 3 episodes"),
        (f"{TRAJECTORIES_DIRNAME}/1000.json", _first_action_nudged, "the actions hash to"),
        (f"{TRAJECTORIES_DIRNAME}/1000.json", _set("seed", 1001), "of seed 1001"),
        (f"{TRAJECTORIES_DIRNAME}/1000.json", _set("actions", ["00" * 8]), "got 1 of seed 1000"),
        (f"{TRAJECTORIES_DIRNAME}/1000.json", _set("actions", [["x"]]), "malformed trajectory"),
        (f"{TRAJECTORIES_DIRNAME}/1000.json", _set("actions", ["zz"]), "malformed trajectory"),
        (f"{TRAJECTORIES_DIRNAME}/1000.json", _set("actions", ["00"]), "malformed trajectory"),
        (RESULT_FILENAME, _episode_dropped, "has episodes for seeds"),
        (f"{TRAJECTORIES_DIRNAME}/1000.json", _drop("actions"), "malformed trajectory"),
    ],
)
def test_an_inconsistent_run_directory_is_refused(
    tmp_path: Path, name: str, edit: Callable[[dict[str, object]], None], message: str
) -> None:
    run_dir = _write(tmp_path / "run")
    _tamper(run_dir, name, edit)
    with pytest.raises(RunDirError, match=message):
        read_run(run_dir)


class _Emits:
    """A policy that emits fixed actions: here, the ones JSON cannot carry as numbers."""

    ref = FakePolicy.ref

    def __init__(self, action: tuple[float, ...]) -> None:
        self._action = action

    def reset(self, seed: int) -> None:
        del seed

    def act(self, observation: object) -> tuple[float, ...]:
        del observation
        return self._action


# x86's default NaN (sign bit set), as `0 * inf` produces it; JSON's NaN reads back as +NaN.
NEGATIVE_NAN = struct.unpack("<d", bytes.fromhex("000000000000f8ff"))[0]


@pytest.mark.parametrize("action", [(NEGATIVE_NAN, 1.0), (math.inf, -math.inf), (-0.0, 0.0)])
def test_any_action_reads_back_bit_for_bit(tmp_path: Path, action: tuple[float, ...]) -> None:
    """A run with a NaN action is exactly the run a monitor must be able to look at later."""
    sweep = run_sweep(FakeEnv(success_rate=1.0), _Emits(action), seeds=SEEDS, max_steps=10)
    persisted = read_run(_write(tmp_path / "run", sweep))
    packed = [struct.pack("<2d", *a) for t in persisted.trajectories for a in t]
    assert packed == [struct.pack("<2d", *action)] * len(packed)
    assert compare_runs(persisted.result, sweep.result) == ()


def test_a_directory_that_would_not_read_back_is_never_renamed_into_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def corrupt(action: tuple[float, ...]) -> str:
        return "00"

    monkeypatch.setattr("fathomline.persist.rundir._encode", corrupt)
    with pytest.raises(RunDirError, match="malformed trajectory"):
        _write(tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_a_missing_trajectory_is_refused(tmp_path: Path) -> None:
    run_dir = _write(tmp_path / "run")
    (run_dir / TRAJECTORIES_DIRNAME / "1001.json").unlink()
    with pytest.raises(RunDirError, match="cannot read"):
        read_run(run_dir)


@pytest.mark.parametrize(
    ("text", "message"), [("{", "not valid JSON"), ("[]", "not a JSON object")]
)
def test_an_unreadable_document_is_refused(tmp_path: Path, text: str, message: str) -> None:
    run_dir = _write(tmp_path / "run")
    (run_dir / RUN_FILENAME).write_text(text, encoding="utf-8")
    with pytest.raises(RunDirError, match=message):
        read_run(run_dir)


def test_the_week2_fixture_still_parses() -> None:
    """A run directory as Lane B first wrote it. If this breaks, bump a version; do not edit it."""
    persisted = read_run(WEEK2_RUN)
    assert persisted.result.manifest.seeds == persisted.record.seed_set.seeds
    assert persisted.record.seed_set.name == "smoke"
    assert persisted.record.host.cpu_model
    assert "## Episodes" in report_run_dir(WEEK2_RUN)


def test_the_week2_fixture_can_be_copied_and_still_parses(tmp_path: Path) -> None:
    """Nothing in a run directory depends on where it sits."""
    copy = shutil.copytree(WEEK2_RUN, tmp_path / "elsewhere")
    assert read_run(copy) == read_run(WEEK2_RUN)


# --- Comparing runs ------------------------------------------------------------


def _episode(**changes: Any) -> EpisodeResult:
    episode = EpisodeResult(
        seed=1,
        steps=5,
        success=True,
        sum_reward=-1.0,
        max_reward=-0.1,
        action_hash="a" * 64,
    )
    return dataclasses.replace(episode, **changes)


def _run(*episodes: EpisodeResult, **manifest_changes: Any) -> RunResult:
    base = _sweep((1,)).result.manifest
    seeds = tuple(episode.seed for episode in episodes)
    manifest = dataclasses.replace(base, seeds=seeds, **manifest_changes)
    return RunResult(manifest=manifest, episodes=episodes)


def test_identical_runs_have_no_differences() -> None:
    assert compare_runs(_run(_episode()), _run(_episode())) == ()


@pytest.mark.parametrize(
    "change",
    [
        {"steps": 6},
        {"success": False},
        {"sum_reward": -1.0000000001},
        {"max_reward": 0.0},
        {"action_hash": "b" * 64},
        {"flags": (Flag(monitor="m", step=0, reason="r"),)},
    ],
)
def test_every_episode_field_is_compared(change: dict[str, Any]) -> None:
    (name,) = change
    (difference,) = compare_runs(_run(_episode()), _run(_episode(**change)))
    assert difference.startswith(f"seed 1: {name} ")


def test_host_facts_are_not_compared() -> None:
    """Whether hosts may differ is D9a's call; a different platform is not a difference."""
    left = _run(_episode(), platform="Linux-x86_64", python_version="3.12.0")
    right = _run(_episode(), platform="Windows-AMD64", python_version="3.12.9")
    assert compare_runs(left, right) == ()


def test_floats_are_compared_bit_for_bit() -> None:
    """As the action hash compares them: NaN equals itself, 0.0 is not -0.0."""
    nan = _episode(sum_reward=math.nan)
    assert compare_runs(_run(nan), _run(nan)) == ()
    (difference,) = compare_runs(_run(_episode(max_reward=0.0)), _run(_episode(max_reward=-0.0)))
    assert difference.startswith("seed 1: max_reward")


def test_episodes_out_of_step_with_their_manifest_are_a_difference() -> None:
    left = _run(_episode(seed=1), _episode(seed=2))
    right = dataclasses.replace(left, episodes=tuple(reversed(left.episodes)))
    assert compare_runs(left, right) == ("episode seeds: (1, 2) != (2, 1)",)


def test_a_different_fixture_or_seed_set_is_a_difference() -> None:
    other = _run(_episode(seed=2), max_steps=99)
    differences = compare_runs(_run(_episode()), other)
    assert [d.split(":")[0] for d in differences] == ["max_steps", "seeds", "episode seeds"]


def test_first_divergence_locates_the_step() -> None:
    a = [(0.0, 1.0), (0.5, 0.5), (1.0, 1.0)]
    assert first_divergence(a, list(a)) is None
    assert first_divergence(a, [a[0], (0.5, 0.75), a[2]]) == Divergence(step=1, max_abs_diff=0.25)
    assert first_divergence(a, [a[0], (0.5,)]) == Divergence(step=1, max_abs_diff=None)
    assert first_divergence(a, a[:2]) == Divergence(step=2, max_abs_diff=None)
    assert first_divergence([(math.nan,)], [(math.nan,)]) is None
    # A NaN of another sign hashes differently, so it is a divergence here.
    divergence = first_divergence([(math.nan,)], [(NEGATIVE_NAN,)])
    assert divergence is not None
    assert divergence.step == 0
