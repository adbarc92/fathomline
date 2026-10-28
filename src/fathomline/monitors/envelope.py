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
* **Observation shift.** Each frame of the third-person camera must keep its
  brightness and noise statistics inside a declared band. The band is the range
  those statistics took over every frame of held-out clean episodes, with each
  end moved out by 10% of its value; the rule was fixed before the episodes it
  is scored on were run. It is coarse: it does not catch a camera swap at the
  first frame, because both views have similar statistics, nor anything else
  that leaves them alone. A swapped episode may still be flagged later, when the
  wrist view's detail drops near an object; that is incidental, not designed.
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

from fathomline.core.types import Observation
from fathomline.monitors.action_bound import ActionBound
from fathomline.monitors.framework import MonitorSuite
from fathomline.monitors.instruction import InstructionSet
from fathomline.monitors.observation_shift import FrameBand
from fathomline.monitors.provenance import Provenance

#: D10 condition 1: the loaded pipeline's fingerprint, as the adapter's `loaded_pipeline()`
#: methods take it from a clean load of the pinned fixture. Changing one is a re-pin.
LIBERO_LOADED_PIPELINE: Mapping[str, str] = {
    "env.cameras": "ff0992ecd29b354d9899254590bbd4bb88644ea40963245d1e58003c30a0b795",
    "policy.normalizer": "d69c000ead4972ec792c3ee30be6a734c2c696130633bb282311a8a82054e4e4",
    "policy.rename_map": "eab3915127c2fe1c37779c21e9b1a1e4d5d710f907c90fee38f98bfe394f77ff",
    "policy.unnormalizer": "7b99b313f36973e5f346b18c41ffd7c1eb885e315274f655b561f508e2a69b08",
}

#: LIBERO's arm dimensions in the action vector. Index 6, the gripper, is excluded.
LIBERO_ARM_DIMS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)

#: D10 condition 2: the bound on every arm dimension. Changing it is a ruling.
ARM_ACTION_LIMIT = 1.1

#: D10 condition 3: the band per third-person frame statistic, in 0-255 gray levels. Calibrated
#: on held-out clean episodes (seeds 1050-1099; `docs/MONITORS.md`), not on the spike's.
LIBERO_FRAME_BANDS: Mapping[str, tuple[float, float]] = {
    "image.brightness": (91.19, 115.82),
    "image.noise": (5.22, 7.09),
}

#: D10 condition 4: the instructions the fixture is evaluated on. `libero_object` task 0's own,
#: as the pinned LIBERO assets state it.
LIBERO_INSTRUCTIONS = frozenset({"pick up the alphabet soup and place it in the basket"})


def fixture_monitors(
    *,
    loaded_pipeline: Callable[[], Mapping[str, str]],
    frame_statistics: Callable[[Observation], Mapping[str, float]],
) -> MonitorSuite:
    """The monitors for a run of the D1 fixture, one per runtime condition in D10.

    The adapter provides both hooks. `loaded_pipeline` returns the fingerprint of the
    pipeline as loaded now (`LeRobotEnv.loaded_pipeline` merged with
    `LeRobotPolicy.loaded_pipeline`); `frame_statistics` reads one observation's frame
    (`adapter.frame_statistics`).
    """
    return MonitorSuite(
        [
            Provenance(declared=LIBERO_LOADED_PIPELINE, probe=loaded_pipeline),
            ActionBound(limit=ARM_ACTION_LIMIT, dims=LIBERO_ARM_DIMS),
            FrameBand(bands=LIBERO_FRAME_BANDS, extract=frame_statistics),
            InstructionSet(declared=LIBERO_INSTRUCTIONS),
        ]
    )
