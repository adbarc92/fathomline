---
stage: Build
readiness: "Lanes 0, A, C, D, F merged; lane E spike done, D10 awaiting a ruling; no result published yet"
updated: "2026-10-27"
name: "Fathomline"
base_branch: "main"
test_cmd: "uv run pytest --cov=fathomline --cov-report=term-missing"
---

# Fathomline — STATUS

_Canonical, living status. The **State summary** is rewritten in place each session; the
**Session log** is appended newest-first._

## State summary

**As of 2026-10-27 — Lanes 0, A, C, D and F are merged. The lane E spike is done: the envelope
evidence is in [`SPIKE-envelope.md`](SPIKE-envelope.md), with a recommended D10 set for Alex to
rule. B is in review; G and H are not started. Open rulings: D9a (cross-host determinism) and D10
(declared envelope).** The plan and the rulings are in [`ROADMAP.md`](ROADMAP.md).

- **What exists.** The `Env` / `Policy` / `Monitor` protocols, the owned step loop with an
  action-trajectory hash, `RunManifest` provenance and result schema v1 (Lane 0); the LeRobot
  adapter and the container image, in which `eval` runs SmolVLA on LIBERO `libero_object` task 0
  offline (Lane A); the Clopper-Pearson gate with a declared threshold and minimum *n* (Lane C);
  the JSON → Markdown report (Lane D); the monitor framework, under which a flagged run is never
  passed (Lane F); the envelope evidence, 290 episodes over 9 conditions (Lane E).
- **What the envelope spike found.** The fixture succeeds on 27/50 clean episodes ([0.39, 0.68]).
  Dim light, camera noise, 5 cm object shifts, a camera swap, lost action stats and a foreign
  instruction each cut that to 0–10%. Only the lost action stats show up in the actions.
- **What does not exist yet.** Seeded sweeps and persisted runs (B, in review), the monitor
  conditions (G), and `reproduce` with a published result (H). `sweep` and `reproduce` exit 2.
- **History.** Lanes 0, A, C, D and F were built and reviewed as PRs #2–#8 in an earlier private
  repository, now archived. This repository starts from that code as a single commit.

**Next steps** — each points into [`ROADMAP.md`](ROADMAP.md)
1. Rule D10 from [`SPIKE-envelope.md`](SPIKE-envelope.md) (recommended set at its end); lane G
   follows, one monitor per ruled condition.
2. Resolve D9a: the action hash has differed between CI runners with identical images, so the
   determinism claim is not yet scoped correctly.
3. Lane B, then lane H (`reproduce`, the first published result). Any gate threshold must
   reckon with the fixture's 54% clean baseline.

**Known limitations**
- No determinism claim across machines until D9a is resolved.
- No `tests.lock` yet; it arrives with lane B.

---

## Session log

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
