---
stage: Build
readiness: "Lanes 0, A, C, D, F merged; public as of 2026-10-25; no result published yet"
updated: "2026-10-25"
name: "Fathomline"
base_branch: "main"
test_cmd: "uv run pytest --cov=fathomline --cov-report=term-missing"
---

# Fathomline — STATUS

_Canonical, living status. The **State summary** is rewritten in place each session; the
**Session log** is appended newest-first._

## State summary

**As of 2026-10-25 — the repository is public. Lanes 0, A, C, D and F are merged; B, E, G and H are
not started. Open rulings: D9 (determinism assertion) and D10 (declared envelope).** The plan and
the rulings are in [`ROADMAP.md`](ROADMAP.md).

- **What exists.** The `Env` / `Policy` / `Monitor` protocols, the owned step loop with an
  action-trajectory hash, `RunManifest` provenance and result schema v1 (Lane 0); the LeRobot
  adapter and the container image, in which `eval` runs SmolVLA on LIBERO `libero_object` task 0
  offline (Lane A); the Clopper-Pearson gate with a declared threshold and minimum *n* (Lane C);
  the JSON → Markdown report (Lane D); the monitor framework, under which a flagged run is never
  passed (Lane F).
- **What does not.** Seeded sweeps and persisted runs (B), the envelope evidence (E), the monitor
  conditions (G), and `reproduce` with a published result (H). `sweep` and `reproduce` exit 2.
- **History.** Lanes 0, A, C, D and F were built and reviewed as PRs #2–#8 in an earlier private
  repository, now archived. This repository starts from that code as a single commit.

**Next steps** — each points into [`ROADMAP.md`](ROADMAP.md)
1. Resolve D9: the action hash has differed between CI runners with identical images, so the
   determinism claim is not yet scoped correctly.
2. Lane E, then rule D10 from its data; lane G follows.
3. Lane B, then lane H (`reproduce`, the first published result).

**Known limitations**
- No determinism claim across machines until D9 is resolved.
- No `tests.lock` yet; it arrives with lane B.

---

## Session log

### 2026-10-25 — Public release

- Published as a new public repository from the merged state of lanes 0, A, C, D and F.
- State delta: the repository is public; CI runs on the public runner shape.
