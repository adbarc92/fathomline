"""Instruction outside the declared set (ruling D10, condition 4): exact match.

The instruction a policy was tested on is part of its envelope. Anything else is
flagged, including a paraphrase that may well be harmless: "outside what was
tested" is the claim, not "harmful". There is no normalization of case,
whitespace or punctuation, because each one is a change the evaluation did not
cover.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from fathomline.core.types import Flag, StepRecord


class InstructionSet:
    """Flags any step whose observed instruction is not, exactly, a declared one.

    The instruction is read from `observation[key]`: one string, or a list or
    tuple of strings (one per environment in a batch). A step with no
    instruction, an empty list or a value that is not text is refused rather
    than passed, since nothing about it was checked.
    """

    def __init__(
        self, *, declared: Collection[str], key: str = "task", name: str = "instruction-set"
    ) -> None:
        if not declared:
            raise ValueError("an InstructionSet needs at least one declared instruction")
        if not all(isinstance(instruction, str) for instruction in declared):
            raise TypeError("every declared instruction must be a string")
        self._declared = frozenset(declared)
        self._key = key
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def on_step(self, record: StepRecord) -> Sequence[Flag]:
        if self._key not in record.observation:
            raise ValueError(
                f"monitor {self._name!r} reads the instruction from {self._key!r}, "
                f"which step {record.index}'s observation does not have"
            )
        instructions = _instructions(record.observation[self._key])
        if not instructions:
            raise ValueError(f"step {record.index} carries no instruction under {self._key!r}")
        outside = [text for text in instructions if text not in self._declared]
        if not outside:
            return []
        listed = ", ".join(repr(text) for text in outside)
        return [
            Flag(
                monitor=self._name,
                step=record.index,
                reason=f"instruction not in the declared set: {listed}",
            )
        ]


def _instructions(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple):
        texts = tuple(text for text in value if isinstance(text, str))
        if len(texts) == len(value):
            return texts
    raise TypeError(f"an instruction must be a string or a list of strings, got {value!r}")
