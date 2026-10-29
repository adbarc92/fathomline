---
stage: Build
readiness: "Lanes 0, A, C, D, F merged; lane E spike done, D10 awaiting a ruling; D9a ruled (D9b); Lane B in review (PR #1); public as of 2026-10-25; no result published yet"
updated: "2026-10-28"
name: "Fathomline"
base_branch: "main"
test_cmd: "uv run pytest --cov=fathomline --cov-report=term-missing"
---

# Fathomline — STATUS

_Canonical, living status. The **State summary** is rewritten in place each session; the
**Session log** is appended newest-first._

## State summary

**As of 2026-10-28, lanes 0, A, C, D and F are merged. The lane E spike is done: the envelope
evidence is in [`SPIKE-envelope.md`](SPIKE-envelope.md), with a recommended D10 set for Alex to
rule. Lane B (sweep, seed sets, persisted runs) is in review as PR #1 and has been approved by its
review agent. Its only red check is the `container` job, which fails at the GHCR push
(`denied: permission_denied: write_package`), an owner-side package permission. G and H are not
started. D9a is ruled as D9b: bit-exact reproduction on any x86_64 host with AVX2, under a torch
thread and kernel pin that the image sets; PR #1 implements it. Open ruling: D10 (declared
envelope).** The plan and the rulings are in [`ROADMAP.md`](ROADMAP.md).

- **What exists on `main`.**
  - Lane 0: the `Env` / `Policy` / `Monitor` protocols, the owned step loop with an
    action-trajectory hash, `RunManifest` provenance and result schema v1.
  - Lane A: the LeRobot adapter and the container image, in which `eval` runs SmolVLA on LIBERO
    `libero_object` task 0 offline.
  - Lane C: the Clopper-Pearson gate, with a declared threshold and minimum *n*. Lane D: the
    JSON → Markdown report. Lane F: the monitor framework, under which a flagged run is never
    passed.
  - Lane E: the envelope evidence, 290 episodes over 9 conditions.
- **What the envelope spike found.** The fixture succeeds on 27/50 clean episodes ([0.39, 0.68]).
  Dim light, camera noise, 5 cm object shifts, a camera swap, lost action stats and a foreign
  instruction each cut that to 0–10%. Only the lost action stats show up in the actions.
- **What PR #1 adds (lane B).**
  - `fathomline sweep --seed-set NAME` runs a seed set declared in [`tests.lock`](../tests.lock),
    whose sets are checked against their digests.
  - It persists a run directory that `report` and `gate` read with no re-run:
    `result.json` (schema v1), `run.json` (run schema v1: the seed set, host facts, timings) and
    `trajectories/`. The format is documented for lane H in [`run-directory.md`](run-directory.md).
  - `tests/test_determinism.py` asserts run-to-run equality and fails when the seed is removed.
  - The D9b pin: the image sets `OMP_NUM_THREADS=2` and `ATEN_CPU_CAPABILITY=avx2`, and `sweep`
    refuses to run (exit 2) when the loaded torch is not under them.
- **What does not exist yet.** The monitor conditions (G), and `reproduce` with a published
  result (H). `reproduce` exits 2.
- **History.** Lanes 0, A, C, D and F were built and reviewed as PRs #2–#8 in an earlier private
  repository, now archived. This repository starts from that code as a single commit.
- **Determinism evidence (behind D9b).** Every hash below was measured with the checkpoint's
  shipped `n_action_steps=50`. `n_action_steps=10` is adopted (PR #7, not yet merged) and changes
  every action hash, so these are pre-#7 values, not ones a current run should reproduce.
  - In two fresh containers on one host (Ryzen 7 5800X, AVX2, 2 CPUs), sweeps of the smoke set gave
    the same action hash, `aa18c8f3…`, and the same metrics.
  - That hash is a third value. CI runners gave `409dc548…` (EPYC 9V45, AVX-512) and `f2c89969…`
    (an unrecorded CPU that ran about 3.5 times slower).
  - All three agree on the metrics: success at step 124.
  - The two CI runs were built from commits whose diff touches no numeric code.
  - So the hash tracks the host, not the code.
  - On the local host, `OMP_NUM_THREADS=1` reproduces `f2c89969…` bit for bit, and runs about 2.4
    times slower.
  - 2, 4 and 8 threads all give `aa18c8f3…`.
  - `ATEN_CPU_CAPABILITY=default` gives a fourth hash, `47e31126…`, and succeeds one step earlier,
    at step 123.
  - `MKL_ENABLE_INSTRUCTIONS=SSE4_2` changes nothing.
  - Reading: the hash depends on (a) one torch thread versus several, and (b) the instruction set
    the ATen kernels dispatch to.
    - The `f2c89969` runner was evidently running torch on one thread.
    - `409dc548` (AVX-512, 2 threads) and `aa18c8f3` (AVX2, 2 threads) differ by ISA.

**Next steps** — each points into [`ROADMAP.md`](ROADMAP.md)
1. Owner: fix the GHCR package permission, so the `container` job runs the in-image sweep check.
   Then merge PR #1.
2. Rule D10 from [`SPIKE-envelope.md`](SPIKE-envelope.md) (recommended set at its end); lane G
   follows, one monitor per ruled condition.
3. Once #7 merges, record the smoke set's action hash under the D9b pin, and show the same hash
   on a second x86_64 host before a cross-host claim is published.
4. Lane H: build `reproduce` on the run directory, and publish the first result. Any gate
   threshold must reckon with the fixture's 54% clean baseline.

**Known limitations**
- The D9b cross-host claim rests on one host so far: no second host has yet run under the pin.
- Run-to-run equality on the real fixture is shown on one local host. CI cannot run the in-image
  check until the GHCR push works.

---

## Session log

### 2026-10-28 — Lane B: main merged in; the D9b pin

- Merged `main` (lane E) into PR #1; `docs/STATUS.md` conflicted and keeps both sides.
- Alex ruled D9a as D9b. The image now sets `OMP_NUM_THREADS=2` and `ATEN_CPU_CAPABILITY=avx2`,
  and `sweep` refuses to run off the pin, test-first (the refusal tests were seen red).
- State delta: D9a is ruled; the action hashes recorded so far are marked as pre-#7 values.

### 2026-10-27 — Lane B: sweep, seed sets, persisted runs (PR #1)

- Built `sweep`, `tests.lock`, the run directory (`persist/`) and the determinism tests, test-first.
  - The equality test was seen red: `FakePolicy` was seeded with a constant, so a policy that lost
    its seed was invisible.
  - The fake now draws from OS entropy until it is reset.
  - Removing the seed from the loop turns three tests red.
- A separate review agent requested changes: NaN actions did not round-trip, episode and manifest
  seeds went unchecked, and the CI trigger missed `persist/`. All were fixed, and it then approved.
- Measured on the real fixture, locally: same-host equality holds, and a third host hash was
  recorded (see the State summary).
- State delta: Lane B is in review; D9a has new evidence.

### 2026-10-27 — Lane E: envelope evidence spike

- Ran the D1 fixture under 8 perturbations (lighting, sensor noise, camera swap, lost action
  stats, paraphrased and foreign instructions, shifted objects) and a baseline: 290 episodes,
  paired on the benchmark init states. Primer, scripts and raw JSONL:
  [`SPIKE-envelope.md`](SPIKE-envelope.md), [`spike-envelope/`](spike-envelope/).
- State delta: D10 has its evidence and a recommended set; the ruling is Alex's. A cross-check
  in LeRobot's own loop got 6/10 against the harness's 7/10 on the same 10 starts.

### 2026-10-25 — Public release

- Published as a new public repository from the merged state of lanes 0, A, C, D and F.
- State delta: the repository is public; CI runs on the public runner shape.
