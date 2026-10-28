"""The declared envelope of the D1 fixture (ruling D10): SmolVLA on LIBERO `libero_object`.

A run is in the envelope only if every condition below holds. Each value is
declared here, from the ruling, and is not re-fitted from a run's data: a bound
tuned to the episodes it then judges says nothing about them.

* **Action out of bounds.** LIBERO's action is six arm deltas (three
  translations, three rotations) and then a gripper command. The arm is bounded
  at +/-1.1 on each of its six dimensions. The gripper is excluded, because it
  passes +/-1 on most clean steps. The training range is not the bound, because
  clean runs leave it. Both are measured in `docs/SPIKE-envelope.md`. The margin
  was chosen after that spike, which is why D10 fixes it rather than leaving it
  to tuning.
* **Unseen init.** Declared and enforced by construction, with no runtime
  monitor. The harness picks every episode's initial state itself, as
  `seed % n` over the task's `n` benchmark init states (`LeRobotEnv.reset`), so
  its own runs cannot start outside them. A monitor becomes necessary once the
  harness runs initial states it did not choose.
"""

from __future__ import annotations

from fathomline.monitors.action_bound import ActionBound
from fathomline.monitors.framework import MonitorSuite

#: LIBERO's arm dimensions in the action vector. Index 6, the gripper, is excluded.
LIBERO_ARM_DIMS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)

#: D10 condition 2: the bound on every arm dimension. Changing it is a ruling.
ARM_ACTION_LIMIT = 1.1


def fixture_monitors() -> MonitorSuite:
    """The monitors for a run of the D1 fixture, one per runtime condition in D10."""
    return MonitorSuite([ActionBound(limit=ARM_ACTION_LIMIT, dims=LIBERO_ARM_DIMS)])
