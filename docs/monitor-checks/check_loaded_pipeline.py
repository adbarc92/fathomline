"""Lane G: the loaded pipeline's fingerprint, against the pins, in the image (D10 condition 1, D11).

The adapter's `loaded_pipeline()` methods run only in the container, so the unit job cannot
cover them. This check, run by the CI container job, loads the D1 fixture and asserts:

1. a clean load matches `LIBERO_LOADED_PIPELINE`, after load and again after a reset, an
   `act()` and an env step (the monitor probes after each of those);
2. the spike's lost-stats fault (`apply_unnorm_identity` in `run_envelope.py`) changes the
   `policy.unnormalizer` digest and nothing else;
3. exchanging the two cameras in the loaded rename map changes `policy.rename_map` and
   nothing else.

    docker run --rm --network none \
      -v "$PWD/docs/monitor-checks:/check:ro" -v "$PWD/docs/spike-envelope:/spike:ro" \
      --entrypoint python fathomline /check/check_loaded_pipeline.py

`--print` prints the clean fingerprint instead, which is how the pins were taken.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from typing import Any


def _fingerprint(env: Any, policy: Any) -> dict[str, str]:
    return {**env.loaded_pipeline(), **policy.loaded_pipeline()}


def _changed(before: dict[str, str], after: dict[str, str]) -> set[str]:
    return {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true", help="print the clean fingerprint")
    parser.add_argument("--spike", default="/spike")
    args = parser.parse_args()

    sys.path.insert(0, args.spike)
    from run_envelope import apply_unnorm_identity  # the spike's fault, unchanged

    from fathomline.adapters.lerobot.adapter import load_fixture
    from fathomline.monitors import LIBERO_LOADED_PIPELINE

    with contextlib.redirect_stdout(sys.stderr):
        env, policy, _ = load_fixture()
    loaded = _fingerprint(env, policy)
    if args.print:
        print(json.dumps(loaded, indent=2, sort_keys=True))
        return 0

    failures: list[str] = []

    def expect(label: str, changed: set[str], want: set[str]) -> None:
        verdict = "ok" if changed == want else "FAIL"
        print(f"{verdict}: {label}: changed {sorted(changed)}, expected {sorted(want)}")
        if changed != want:
            failures.append(label)

    pins = dict(LIBERO_LOADED_PIPELINE)
    expect("clean load against the pins", _changed(pins, loaded), set())
    with contextlib.redirect_stdout(sys.stderr):
        observation = env.reset(1000)
        env.step(policy.act(observation))
    expect("after reset, act and step", _changed(pins, _fingerprint(env, policy)), set())

    for step in policy._preprocessor.steps:
        if type(step).__name__ == "RenameObservationsProcessorStep":
            original = dict(step.rename_map)
            image, image2 = "observation.images.image", "observation.images.image2"
            step.rename_map = {**original, image: original[image2], image2: original[image]}
            expect(
                "cameras exchanged in the rename map",
                _changed(pins, _fingerprint(env, policy)),
                {"policy.rename_map"},
            )
            step.rename_map = original

    apply_unnorm_identity(policy)
    expect("lost action stats", _changed(pins, _fingerprint(env, policy)), {"policy.unnormalizer"})
    env.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
