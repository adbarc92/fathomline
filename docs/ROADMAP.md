# Fathomline — ROADMAP

_The one roadmap for this repo. Every future intention lives here. [`STATUS.md`](STATUS.md) "Next
steps" points into it, and GitHub issues exist only for entries that are scoped and about to be
worked (`— #N`)._

**Supersedes** the week-by-week table in an internal planning doc as the working plan. That
doc's settled parameters still hold unless a ruling below amends them. Consolidated 2026-09-30 from
STATUS, internal audits and the week-1 spike ([`SPIKE-week1-policy.md`](SPIKE-week1-policy.md)).

## Target

| Date | Milestone | Done when | Slips? |
|---|---|---|---|
| **2026-11-09** | **G0.1-early (hard floor)** — repo public (done 2026-10-25); one command on a clean container reproduces one published result; determinism proven | CI green on the public repo; `reproduce` exits 0 and matches the published `result.json` | **No** |
| 2026-11-09 (aim) | **G0.1 full** — all four capabilities, README with a proving command for every claim | Verified by CI | Yes. **The date moves; the monitor never gets cut** |

Cut order if the clock slips: polish → report richness (machine-readable only) → sweep breadth.
**Never the monitor.**

## Decision register

Ruled decisions are not re-litigated. An amendment is a new dated row, never an edit.

| # | Decision | Ruling | Date |
|---|---|---|---|
| D1 | Fixture | **SmolVLA (`lerobot/smolvla_libero` @ `31d453f`) on LIBERO `libero_object` task 0.** Automatic fallback to SmolVLA + Meta-World if the image does not fit the CI runner | 2026-09-29 |
| D2 | Runtime | **Python 3.12, `lerobot[smolvla,libero,evaluation]==0.6.1` pinned exactly**, CPU torch index, every Hub artifact pinned by SHA | 2026-09-29 |
| D3 | Architecture | **Fathomline owns the step loop** behind small `Env` / `Policy` protocols; LeRobot sits behind one adapter. This is the seam for retargeting to another simulator stack | 2026-09-29 |
| D4 | Nov 9 scope | **Aim for full G0.1; G0.1-early is the hard floor** | 2026-09-29 |
| D5 | Visibility | **Private until Nov 9**, public at G0.1-early. CI runs on the 2 vCPU / 8 GB private runner shape until then | 2026-09-29 |
| D5a | Visibility (amends D5) | **Public as of 2026-10-25**, as a new repository with a single initial commit. CI runs on the public runner shape (4 vCPU / 16 GB) | 2026-10-25 |
| D6 | Gate statistic | **Clopper-Pearson exact 95% lower bound ≥ threshold**, plus a declared minimum *n*. A small sample fails by construction | 2026-09-29 |
| D6a | Gate statistic (clarifies D6) | "95%" is the **lower end of the two-sided 95% exact interval**, i.e. a one-sided 97.5% lower confidence bound: the probability that the true rate is below it is at most 2.5%. The ruling is unchanged | 2026-10-25 |
| D7 | CI cadence | **Every PR:** unit + lint + fake-env tests, plus a 1-episode container smoke when the adapter, Dockerfile or lock changes. **Push to main + nightly:** the full published sweep. Image cached in GHCR | 2026-09-29 |
| D8 | Automation | **Agents build each lane in a worktree and open a PR; a separate review agent grades it (writer ≠ grader); CI must be green; Alex merges** | 2026-09-29 |
| D9 | Determinism assertion | Assert on per-episode metrics and an action-trajectory hash, **never on video** (spike finding 6). The claim is scoped to the pinned container on x86_64 CPU | default 2026-09-30 — Alex may amend |
| D9a | Determinism assertion (reopens D9) | **Open.** The action-trajectory hash differed between CI runners running identical images (`409dc548…` vs `f2c89969…`), so the cross-host scope of the claim is unresolved. No cross-machine determinism claim until this is ruled | 2026-10-25 |
| D10 | Declared envelope (U2 conditions) | **Open.** Candidates: provenance mismatch, action out of bounds, observation shift, unseen instruction/init. Ruled after the evidence spike (lane E) | — |
| D11 | Coverage for the Linux-only adapter | The unit job's 100% gate covers the core; the adapter is omitted there **with a stated reason** and covered by the container integration job | default 2026-09-30 — Alex may amend |
| D12 | Report format | Versioned JSON is canonical; Markdown is rendered from it | default 2026-09-30 |
| D13 | Distribution | The published image is `ghcr.io/adbarc92/fathomline:<tag>`; the reproduction command is `docker run … reproduce <result-id>` | default 2026-09-30 |

## Lanes

**Lane 0 goes first and alone.** It fixes the interfaces every other lane imports, so the lanes
after it touch disjoint files.

| Lane | Scope | Owns (files) | Done when | Depends on |
|---|---|---|---|---|
| **0 · Contracts** | Python 3.12 bump; `Env`/`Policy`/`Monitor` protocols; `StepRecord`, `EpisodeResult`, `RunManifest` (provenance); result schema v1; `FakeEnv`/`FakePolicy` with a controllable success rate and an injectable observation perturbation; CLI subcommand stubs (`eval`, `sweep`, `gate`, `report`, `reproduce`), each exiting 2 until its lane lands | `core/`, `fakes/`, `cli.py`, `pyproject.toml` | Protocols typed under `mypy --strict`; the fake runs one episode through the loop; CI green | D1–D3 |
| **A · LeRobot adapter + container** | Adapter over `make_env`/`make_policy` with the `rename_map`; Dockerfile per spike option C; `warm.py`; hash-check the LIBERO assets (their revision is unpinned upstream); GHCR cache; update the CI `container` job | `adapters/lerobot/`, `Dockerfile`, `.github/workflows/` (container job) | `docker run … eval --episodes 1` exits 0 **in CI**; image fits the 14 GB runner, or the D1 fallback is taken and recorded | 0 |
| **B · Determinism + sweep + persistence** | Seeded loop, seed sets, persisted runs; `tests.lock` | `core/loop.py`, `persist/`, `tests.lock` | A test asserts run-to-run equality and **fails when the seed is removed**; the schema is versioned and a week-2 fixture still parses | 0 (fake), then A (real) |
| **C · Gates** | Clopper-Pearson, threshold + minimum-*n* config | `gates/` | An under-performing fixture **fails**; an insufficient sample **fails** | 0 |
| **D · Report** | JSON → Markdown, `fathomline report <run-dir>` | `report/` | Regenerated from persisted results with no re-run | 0 |
| **E · Envelope evidence** *(spike, not build)* | Perturb LIBERO lighting/noise; measure the success-rate shift and the action-bound excursions; one-page primer for Alex | `docs/SPIKE-envelope.md` | Alex rules D10 from the data | A |
| **F · Monitor framework** | Per-step hook; flag semantics: a flagged run is **flagged, never silently reported**, the report surfaces the flag and the gate never passes a flagged run | `monitors/` (framework) | Red-first tests on the fake, with the perturbation injected | 0 |
| **G · Monitor conditions** | One monitor per condition in D10 | `monitors/` (conditions) | Every condition's test was **seen failing before it passed** | D10, F |
| **H · Publication** | `reproduce`, published result, README proving-command table, nightly sweep, go-public checklist | `README.md`, `results/`, CI (nightly) | The G0.1-early row of the target table | A, B |

## Schedule to Nov 9

Alex's hands: rulings and merges, about 2–3 hours a week, so nothing on the critical path waits on
him for more than a ruling.

| Step | Work | Alex |
|---|---|---|
| Done | Lanes 0, A, C, D, F | — |
| Next | E runs; B on the real adapter | **Rule D9; rule D10 from lane E's data** |
| Then | G; H begins; publish the first result | Merges |
| To 2026-11-09 | Integration and hardening; G0.1-early, full G0.1 if the monitors are green | Merges |

## Follow-ups

Known gaps and small fixes that are not yet scheduled.

- **CLI runs are unmonitored.** `fathomline eval` does not attach `fixture_monitors()` (lane G,
  PRs #4–#8) yet, and `sweep` (PR #1) does not either. Until `cli.py` is wired, CLI runs and any
  result published from them carry no monitor flags, so the "a flagged run never passes" guarantee
  does not reach them.
  Done when `eval` and `sweep` run under the fixture's monitors, and a flagged CLI run is
  reported as flagged and fails the gate.
- **Lane G small follow-ups** (code in PRs #4–#8).
  - The monitor check script's frame recorder (`docs/monitor-checks/check_monitors.py`)
    recomputes the frame statistics instead of calling the adapter's `frame_statistics`, so the
    two can drift apart.
  - The container check's frame-statistics tolerance (0.01, in `check_loaded_pipeline.py`) has
    never run on a CI runner, because the `container` job has not yet got past the GHCR push.
  - `FrameBand` accepts a band whose low equals its high. Such a band passes only one exact value,
    so it should be refused like an inverted one.

## Known risks

- **The ~6.6 GB image is tight on the 14 GB runner disk.** CI pins `ubuntu-24.04` and frees disk
  before the build; D1 names the fallback if it stops fitting.
- **Cross-host determinism is unverified.** The spike checked 2, 4 and 8 vCPU on one host only. CI
  against local is the first real test.
- **LIBERO is Linux-only.** On Windows, development runs against the fake; the real adapter runs
  only in the container.

## After G0.1

- Phase 1: retarget the monitors onto a PX4/Gazebo stack, unchanged.
- Lane B adoption bar G1.3b: ≥ 3 outside organizations using Fathomline in a real workflow.
- Perturbation-sweep breadth beyond the D10 set (this goes first in the cut order).
