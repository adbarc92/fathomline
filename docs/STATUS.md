---
stage: Build
readiness: "Lanes 0, A, C, D, F merged; Lane B in review (PR #1); public as of 2026-10-25; no result published yet"
updated: "2026-10-27"
name: "Fathomline"
base_branch: "main"
test_cmd: "uv run pytest --cov=fathomline --cov-report=term-missing"
---

# Fathomline — STATUS

_Canonical, living status. The **State summary** is rewritten in place each session; the
**Session log** is appended newest-first._

## State summary

**As of 2026-10-27, lanes 0, A, C, D and F are merged. Lane B (sweep, seed sets, persisted runs) is
in review as PR #1 and has been approved by its review agent. Its only red check is the `container`
job, which fails at the GHCR push (`denied: permission_denied: write_package`), an owner-side
package permission. E is in progress; G and H are not started. Open rulings: D9a (cross-host
determinism) and D10 (declared envelope).** The plan and the rulings are in
[`ROADMAP.md`](ROADMAP.md).

- **What exists on `main`.**
  - Lane 0: the `Env` / `Policy` / `Monitor` protocols, the owned step loop with an
    action-trajectory hash, `RunManifest` provenance and result schema v1.
  - Lane A: the LeRobot adapter and the container image, in which `eval` runs SmolVLA on LIBERO
    `libero_object` task 0 offline.
  - Lane C: the Clopper-Pearson gate. Lane D: the JSON → Markdown report. Lane F: the monitor
    framework.
- **What PR #1 adds (lane B).**
  - `fathomline sweep --seed-set NAME` runs a seed set declared in [`tests.lock`](../tests.lock),
    whose sets are checked against their digests.
  - It persists a run directory that `report` and `gate` read with no re-run:
    `result.json` (schema v1), `run.json` (run schema v1: the seed set, host facts, timings) and
    `trajectories/`. The format is documented for lane H in [`run-directory.md`](run-directory.md).
  - `tests/test_determinism.py` asserts run-to-run equality and fails when the seed is removed.
- **What does not exist yet.** The envelope evidence (E), the monitor conditions (G), and
  `reproduce` with a published result (H). `reproduce` exits 2.
- **Determinism evidence (for D9a; no ruling made).**
  - In two fresh containers on one host (Ryzen 7 5800X, AVX2, 2 CPUs), sweeps of the smoke set gave
    the same action hash, `aa18c8f3…`, and the same metrics.
  - That hash is a third value. CI runners gave `409dc548…` (EPYC 9V45, AVX-512) and `f2c89969…`
    (an unrecorded CPU that ran about 3.5 times slower).
  - All three agree on the metrics: success at step 124.
  - The two CI runs were built from commits whose diff touches no numeric code.
  - So the hash tracks the host, not the code. Variants that test thread count and kernel dispatch
    are in PR #1's body as they finish.

**Next steps** — each points into [`ROADMAP.md`](ROADMAP.md)
1. Owner: fix the GHCR package permission, so the `container` job runs the in-image sweep check.
   Then merge PR #1.
2. Rule D9a, with the evidence above and in PR #1.
3. Lane E, then rule D10 from its data; lane G follows.
4. Lane H: build `reproduce` on the run directory, and publish the first result.

**Known limitations**
- No determinism claim across machines until D9a is resolved.
- Run-to-run equality on the real fixture is shown on one local host. CI cannot run the in-image
  check until the GHCR push works.
---

## Session log

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

### 2026-10-25 — Public release

- Published as a new public repository from the merged state of lanes 0, A, C, D and F.
- State delta: the repository is public; CI runs on the public runner shape.
