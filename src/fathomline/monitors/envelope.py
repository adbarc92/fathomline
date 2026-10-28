"""The declared envelope of the D1 fixture (ruling D10): SmolVLA on LIBERO `libero_object`.

A run is in the envelope only if every condition D10 rules holds; each is listed
below once its monitor, or its reason for having none, has landed. Each value is
declared here, from the ruling, and is not re-fitted from a run's data: a bound
tuned to the episodes it then judges says nothing about them.

* **Provenance mismatch.** The pipeline as loaded must be the pinned one: the
  policy's normalizer and unnormalizer (the statistics they apply, their features
  and modes), its rename map, and the env's camera wiring, each digested from
  memory by the adapter on every step. `pins.verify` covers the files on disk;
  this covers what was done with them after load. It does not see a fault in the
  pixel stream itself, such as two images exchanged after rendering.
* **Action out of bounds.** LIBERO's action is six arm deltas (three
  translations, three rotations) and then a gripper command. The arm is bounded
  at +/-1.1 on each of its six dimensions. The gripper is excluded, because it
  passes +/-1 on most clean steps. The training range is not the bound, because
  clean runs leave it. Both are measured in `docs/SPIKE-envelope.md`. The margin
  was chosen after that spike, which is why D10 fixes it rather than leaving it
  to tuning.
* **Instruction outside the declared set.** The instruction must be, exactly,
  the task's own. A paraphrase is flagged by design: it is outside what was
  tested, whether or not it is harmful.
* **Unseen init.** Declared and enforced by construction, with no runtime
  monitor. The harness picks every episode's initial state itself, as
  `seed % n` over the task's `n` benchmark init states (`LeRobotEnv.reset`), so
  its own runs cannot start outside them. That holds only while the env has the
  benchmark init states loaded, as the fixture's does; without them, `reset`
  leaves the choice to LeRobot. A monitor becomes necessary once the harness runs
  initial states it did not choose.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from fathomline.monitors.action_bound import ActionBound
from fathomline.monitors.framework import MonitorSuite
from fathomline.monitors.instruction import InstructionSet
from fathomline.monitors.provenance import Provenance

#: D10 condition 1: the loaded pipeline's fingerprint, as the adapter's `loaded_pipeline()`
#: methods take it from a clean load of the pinned fixture. Changing one is a re-pin.
LIBERO_LOADED_PIPELINE: Mapping[str, str] = {
    "env.cameras": "ff0992ecd29b354d9899254590bbd4bb88644ea40963245d1e58003c30a0b795",
    "policy.normalizer": "964135d9515808932a52118ef4145e2f63b1e60517da450665549687f627926d",
    "policy.rename_map": "eab3915127c2fe1c37779c21e9b1a1e4d5d710f907c90fee38f98bfe394f77ff",
    "policy.unnormalizer": "7ecfc571fc54861a7fd0317ad8cdc06461728b215a1659c7ae060bd738281283",
}

#: LIBERO's arm dimensions in the action vector. Index 6, the gripper, is excluded.
LIBERO_ARM_DIMS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)

#: D10 condition 2: the bound on every arm dimension. Changing it is a ruling.
ARM_ACTION_LIMIT = 1.1

#: D10 condition 4: the instructions the fixture is evaluated on. `libero_object` task 0's own,
#: as the pinned LIBERO assets state it.
LIBERO_INSTRUCTIONS = frozenset({"pick up the alphabet soup and place it in the basket"})


def fixture_monitors(*, loaded_pipeline: Callable[[], Mapping[str, str]]) -> MonitorSuite:
    """The monitors for a run of the D1 fixture, one per runtime condition in D10 built so far.

    `loaded_pipeline` returns the fingerprint of the pipeline as loaded now; the adapter
    provides it (`LeRobotEnv.loaded_pipeline` merged with `LeRobotPolicy.loaded_pipeline`).
    """
    return MonitorSuite(
        [
            Provenance(declared=LIBERO_LOADED_PIPELINE, probe=loaded_pipeline),
            ActionBound(limit=ARM_ACTION_LIMIT, dims=LIBERO_ARM_DIMS),
            InstructionSet(declared=LIBERO_INSTRUCTIONS),
        ]
    )
