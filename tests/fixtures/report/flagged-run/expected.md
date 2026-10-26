# Fathomline run report: FLAGGED

> **FLAGGED: 1 of 4 episodes raised 1 monitor flag.** A flagged episode fell outside the declared
> envelope, so this run is not a clean measurement. Every flag is listed under "Monitor flags".

Rendered from result schema v1. The JSON is canonical; numbers here are rounded for display.

## Provenance

| Field | Value |
|---|---|
| Fathomline version | 0.0.1 |
| Policy | fixture-policy @ hand-written |
| Env | fixture-env @ hand-written |
| Seeds | 0, 1, 2, 3 |
| Max steps | 300 |
| Python | 3.12.5 |
| Platform | Linux-x86_64 |

## Summary

| Metric | Value |
|---|---|
| Episodes | 4 |
| Successes | 3 |
| Success rate | 75.0% |
| Flagged episodes | 1 |

## Gate

| Field | Value |
|---|---|
| Verdict | **FAIL** |
| Successes / n | 3 / 4 |
| Minimum n | 30 |
| Clopper-Pearson 95% lower bound | 0.1941 |
| Threshold | 0.8000 |

## Monitor flags

| Episode seed | Step | Monitor | Reason |
|---|---|---|---|
| 2 | 41 | action-bounds | joint 3 command 2.4 outside [-2.0, 2.0] |

## Episodes

| Seed | Success | Steps | Sum reward | Max reward | Flags | Action hash |
|---|---|---|---|---|---|---|
| 0 | yes | 120 | 1.0000 | 1.0000 | 0 | `aaaaaaaaaaaa` |
| 1 | no | 300 | 0.0000 | 0.0000 | 0 | `bbbbbbbbbbbb` |
| 2 | yes | 95 | 1.0000 | 1.0000 | 1 | `cccccccccccc` |
| 3 | yes | 101 | 1.0000 | 1.0000 | 0 | `dddddddddddd` |
