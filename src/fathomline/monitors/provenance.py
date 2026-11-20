"""Provenance mismatch (ruling D10, condition 1): the pipeline as loaded, not only as stored.

`pins.verify` checks that every artifact on disk hashes to its pin. That says
nothing about what the process does with it after load: in the envelope spike,
the action unnormalization stats were replaced in memory and every file still
matched. This monitor asks the loaded pipeline for a fingerprint, one digest per
component, on every step, and compares it with the declared one.

What a component is, and how its digest is taken, is the adapter's to say; this
module only compares. A component that is missing, undeclared or different is
flagged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from fathomline.core.types import Flag, StepRecord


class Provenance:
    """Flags any step at which the loaded pipeline's fingerprint is not the declared one."""

    def __init__(
        self,
        *,
        declared: Mapping[str, str],
        probe: Callable[[], Mapping[str, str]],
        name: str = "provenance",
    ) -> None:
        if not declared:
            # Nothing declared would make every pipeline "match".
            raise ValueError("a Provenance monitor needs at least one declared component")
        self._declared = dict(declared)
        self._probe = probe
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def on_step(self, record: StepRecord) -> Sequence[Flag]:
        loaded = self._probe()
        problems: list[str] = []
        for component in sorted(self._declared.keys() | loaded.keys()):
            declared, actual = self._declared.get(component), loaded.get(component)
            if actual is None:
                problems.append(f"{component} missing from the loaded pipeline")
            elif declared is None:
                problems.append(f"{component} not declared (loaded {actual[:12]})")
            elif actual != declared:
                problems.append(f"{component} loaded {actual[:12]}, declared {declared[:12]}")
        if not problems:
            return []
        return [
            Flag(
                monitor=self._name,
                step=record.index,
                reason="loaded pipeline differs from the declaration: " + "; ".join(problems),
            )
        ]
